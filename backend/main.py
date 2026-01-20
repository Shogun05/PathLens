from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import os
import subprocess
import sys
import json
import logging
import pyproj

# Add project root to path for CityDataManager
sys.path.insert(0, str(Path(__file__).parent.parent))
from city_paths import CityDataManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PathLens API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "name": "PathLens API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "nodes": "/api/nodes?type={baseline|optimized}",
            "h3_aggregations": "/api/h3-aggregations?type={baseline|optimized}",
            "metrics_summary": "/api/metrics-summary?type={baseline|optimized}",
            "pois": "/api/pois",
            "suggestions": "/api/suggestions",
            "feasibility": "/api/feasibility/{amenity}",
            "placements": "/api/placements/{amenity}",
            "optimization_status": "/api/optimization/status",
            "optimize": "POST /api/optimize",
            "rescore": "POST /api/rescore"
        }
    }

@app.get("/health")
async def health():
    # Check if data directories exist
    data_exists = os.path.exists(DATA_DIR)
    baseline_exists = os.path.exists(os.path.join(DATA_DIR, "baseline_nodes_with_scores.csv"))
    
    return {
        "status": "healthy",
        "data_dir_exists": data_exists,
        "baseline_data_exists": baseline_exists,
        "data_dir": DATA_DIR
    }

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "analysis")
RUN_SCRIPT = os.path.join(BASE_DIR, "run_pathlens.py")
STATUS_PATH = os.path.join(BASE_DIR, "data", "optimization", "runs", "progress.json")

# Multi-mode optimization paths
CITIES_DIR = os.path.join(BASE_DIR, "data", "cities")
DEFAULT_CITY = "bangalore"
OPTIMIZATION_MODES = {
    "ga_only": "GA Only",
    "ga_milp": "GA + MILP", 
    "ga_milp_pnmlr": "GA + MILP + PNMLR"
}

# City name aliases (alternate names -> folder name)
CITY_ALIASES = {
    "bengaluru": "bangalore",
    "bombay": "mumbai",
    "new_mumbai": "navi_mumbai",
}

# City bounding boxes for filtering outliers (west, south, east, north)
CITY_BOUNDS = {
    "bangalore": (77.35, 12.75, 77.85, 13.20),
    "chennai": (80.00, 12.80, 80.40, 13.30),
    "kolkata": (88.20, 22.40, 88.55, 22.70),
    "chandigarh": (76.68, 30.65, 76.88, 30.82),
    "navi_mumbai": (72.95, 18.90, 73.15, 19.15),
}

def normalize_city_name(city: str) -> str:
    """Normalize city name and resolve aliases."""
    normalized = city.lower().strip().replace(" ", "_")
    return CITY_ALIASES.get(normalized, normalized)

def get_mode_dir(city: str, mode: str) -> str:
    """Get the directory path for a specific optimization mode."""
    return os.path.join(CITIES_DIR, normalize_city_name(city), "optimized", mode)


def write_optimization_status(
    status: str,
    stage: str,
    message: str,
    percent: Optional[float] = None,
    pipelines: Optional[Dict[str, str]] = None
) -> None:
    """Persist the current optimization status so the frontend can poll it."""
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    payload = {
        "status": status,
        "stage": stage,
        "message": message,
        "percent": percent if percent is not None else 0,
        "timestamp": datetime.utcnow().isoformat(),
        "pipelines": pipelines
        or {
            "data": "pending",
            "optimization": "pending",
            "landuse": "pending"
        }
    }

    try:
        with open(STATUS_PATH, 'w') as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning(f"Failed to write optimization status: {exc}")

# Models
class Node(BaseModel):
    osmid: str
    x: float
    y: float
    accessibility_score: Optional[float] = None
    walkability_score: Optional[float] = None
    equity_score: Optional[float] = None
    travel_time_min: Optional[float] = None
    betweenness_centrality: Optional[float] = None
    dist_to_school: Optional[float] = None
    dist_to_hospital: Optional[float] = None
    dist_to_park: Optional[float] = None

class Suggestion(BaseModel):
    type: str
    geometry: Dict[str, Any]
    properties: Dict[str, Any]

class OptimizeRequest(BaseModel):
    location: str
    budget: float
    max_amenities: int
    add_schools: bool
    add_hospitals: bool
    add_parks: bool

class RescoreRequest(BaseModel):
    location: str
    selected_ids: List[str]

@app.get("/api/nodes", response_model=List[Node])
async def get_nodes(
    type: str = Query(..., pattern="^(baseline|optimized)$"),
    city: Optional[str] = Query(None, description="City name for city-specific data"),
    mode: str = Query("ga_only", pattern="^(ga_only|ga_milp|ga_milp_pnmlr)$"),
    limit: Optional[int] = Query(None, ge=1, le=100000, description="Max nodes to return"),
    offset: Optional[int] = Query(0, ge=0, description="Offset for pagination"),
    bbox: Optional[str] = Query(None, description="Bounding box: west,south,east,north")
):
    # Use CityDataManager for path resolution
    target_city = city if city else DEFAULT_CITY
    cdm = CityDataManager(target_city, mode=mode)
    
    if type == "baseline":
        parquet_path = str(cdm.baseline_nodes)
        csv_path = str(cdm.baseline_nodes_csv)
    else:
        parquet_path = str(cdm.optimized_nodes(mode))
        csv_path = str(cdm.optimized_nodes_csv(mode))
    
    # Store paths as Path objects for checking existence
    parquet_path_obj = Path(parquet_path)
    csv_path_obj = Path(csv_path)
    
    # Use parquet if available, fall back to CSV
    if os.path.exists(parquet_path):
        filepath = parquet_path
        use_parquet = True
    elif os.path.exists(csv_path):
        filepath = csv_path
        use_parquet = False
    else:
        logger.warning(f"File not found: {parquet_path} or {csv_path}")
        if type == "optimized":
            return []
        raise HTTPException(status_code=404, detail=f"Data file not found for city={city}, type={type}")
    
    try:
        if use_parquet:
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded {len(df)} rows from parquet: {filepath}")
        else:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from CSV: {filepath}")
        
        # Check for missing coordinates and project from UTM if needed

        if 'x' in df.columns and 'y' in df.columns:
            # If lon/lat columns don't exist, create them
            if 'lon' not in df.columns: df['lon'] = np.nan
            if 'lat' not in df.columns: df['lat'] = np.nan
            
            # Identify missing coordinates
            mask = df['lon'].isna() | df['lat'].isna()
            missing_count = mask.sum()
            
            if missing_count > 0:
                # Dynamic UTM zone selection
                # Bangalore, Mumbai, Navi Mumbai, Chandigarh -> Zone 43N (EPSG:32643)
                # Chennai -> Zone 44N (EPSG:32644)
                # Kolkata -> Zone 45N (EPSG:32645)
                
                normalized_city = normalize_city_name(target_city)
                source_epsg = "EPSG:32643"  # Default
                
                if normalized_city == "chennai":
                    source_epsg = "EPSG:32644"
                elif normalized_city == "kolkata":
                    source_epsg = "EPSG:32645"
                
                logger.info(f"Projecting {missing_count} missing coordinates from {source_epsg} to WGS84 for {target_city}")
                try:
                    # Initialize transformer (UTM -> WGS84)
                    transformer = pyproj.Transformer.from_crs(source_epsg, "EPSG:4326", always_xy=True)
                    # Transform only missing rows
                    xs = df.loc[mask, 'x'].values
                    ys = df.loc[mask, 'y'].values
                    lons, lats = transformer.transform(xs, ys)
                    df.loc[mask, 'lon'] = lons
                    df.loc[mask, 'lat'] = lats
                except Exception as e:
                    logger.error(f"Projection failed: {e}")

        # Replace NaN with None for JSON compatibility
        df = df.replace({np.nan: None})
        
        # Filter by bounding box if provided
        if bbox:
            try:
                west, south, east, north = map(float, bbox.split(','))
                df = df[
                    (df['lon'] >= west) & (df['lon'] <= east) &
                    (df['lat'] >= south) & (df['lat'] <= north)
                ]
                logger.info(f"Filtered to {len(df)} nodes within bbox")
            except ValueError:
                logger.warning(f"Invalid bbox format: {bbox}")
        
        # Apply pagination
        total_count = len(df)
        if offset:
            df = df.iloc[offset:]
        if limit:
            df = df.iloc[:limit]
        
        
        # Optimize serialization: Vectorized conversion to dict list
        # This avoids the slow iterrows loop
        
        # Rename output columns to match Node model
        # accessibility_score logic was: optimized_... if type==optimized else ...
        if type == "optimized":
            df['accessibility_score'] = df['optimized_accessibility_score'].fillna(df['accessibility_score'])
            df['walkability_score'] = df['optimized_walkability'].fillna(df['walkability'])
            df['equity_score'] = df['optimized_equity_score'].fillna(df['equity_score'])
            df['travel_time_min'] = df['optimized_travel_time_min'].fillna(df['travel_time_min'])
        else:
            df['walkability_score'] = df['walkability']
            
        # Ensure osmid string
        df['osmid'] = df['osmid'].astype(str) if 'osmid' in df.columns else df.index.astype(str)
        
        # Select and rename columns for output
        # Node model fields: osmid, x, y, accessibility_score, walkability_score, equity_score, 
        # travel_time_min, betweenness_centrality, dist_to_school, dist_to_hospital, dist_to_park
        
        needed_cols = {
            'osmid': 'osmid',
            'lon': 'x',
            'lat': 'y',
            'accessibility_score': 'accessibility_score',
            'walkability_score': 'walkability_score',
            'equity_score': 'equity_score',
            'travel_time_min': 'travel_time_min',
            'betweenness_centrality': 'betweenness_centrality',
            'dist_to_school': 'dist_to_school',
            'dist_to_hospital': 'dist_to_hospital',
            'dist_to_park': 'dist_to_park'
        }
        
        # Only keep cols that exist in df
        final_cols = {k: v for k, v in needed_cols.items() if k in df.columns}
        
        # Filter and rename
        out_df = df[list(final_cols.keys())].rename(columns=final_cols)
        
        # Replace NaN/None with null for JSON (pandas uses NaN, passing to fastAPI might need None or handle NaNs)
        out_df = out_df.replace({np.nan: None})
        
        # Convert to list of dicts directly
        nodes = out_df.to_dict(orient='records')
        
        logger.info(f"Returning {len(nodes)}/{total_count} nodes for type {type}")
        return nodes
    except Exception as e:
        logger.error(f"Error processing nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.get("/api/h3-aggregations")
async def get_h3_aggregations(
    type: str = Query(..., pattern="^(baseline|optimized)$"),
    city: str = Query(DEFAULT_CITY),
    mode: str = Query("ga_only", pattern="^(ga_only|ga_milp|ga_milp_pnmlr)$")
):
    """Return H3 aggregations for a city's baseline or optimized state."""
    normalized_city = normalize_city_name(city)
    
    # Determine the correct path based on type
    if type == "baseline":
        filepath = os.path.join(CITIES_DIR, normalized_city, "baseline", "h3_agg.csv")
    else:
        mode_dir = os.path.join(CITIES_DIR, normalized_city, "optimized", mode)
        filepath = os.path.join(mode_dir, "h3_agg.csv")
    
    # Fallback to legacy path for backwards compatibility
    if not os.path.exists(filepath):
        filename = "baseline_h3_agg.csv" if type == "baseline" else "optimized_h3_agg.csv"
        filepath = os.path.join(DATA_DIR, filename)
        logger.info(f"Using legacy path: {filepath}")
    
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return {"hexagons": []}
    
    try:
        df = pd.read_csv(filepath)
        df = df.replace({np.nan: None})
        hexagons = df.to_dict('records')
        logger.info(f"Returning {len(hexagons)} H3 hexagons for city {city}, type {type}, mode {mode}")
        return {"hexagons": hexagons}
    except Exception as e:
        logger.error(f"Error processing H3 aggregations: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.get("/api/metrics-summary")
async def get_metrics_summary(
    type: str = Query(..., pattern="^(baseline|optimized)$"),
    city: str = Query(DEFAULT_CITY),
    mode: str = Query("ga_only", pattern="^(ga_only|ga_milp|ga_milp_pnmlr)$")
):
    """Return metrics summary for a city's baseline or optimized state.
    
    Uses CityDataManager for modular path resolution.
    Normalizes response to ensure accessibility, walkability, travel_time, and equity are always present.
    """
    cdm = CityDataManager(city, mode=mode)
    
    # Resolve path using CDM
    if type == "baseline":
        filepath = cdm.baseline_metrics
    else:
        filepath = cdm.optimized_metrics(mode)
    
    if not filepath.exists():
        logger.warning(f"Metrics file not found: {filepath}")
        # Return empty structure with default zero values
        return {
            "network": {},
            "scores": {
                "accessibility": 0,
                "walkability": 0,
                "travel_time_min": 0,
                "equity": 0,
                "citywide": {},
                "distribution": {}
            }
        }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            if not content:
                return {"network": {}, "scores": {"accessibility": 0, "walkability": 0, "equity": 0}}
            metrics = json.loads(content)
            
        # Normalize scores to ensure frontend compatibility
        scores = metrics.get("scores", {})
        
        # If stratified (baseline), flatten core metrics to top level
        if "citywide" in scores:
            cw = scores["citywide"]
            scores.setdefault("accessibility", cw.get("accessibility_mean", 0))
            scores.setdefault("walkability", cw.get("walkability_mean", 0))
            scores.setdefault("travel_time_min", cw.get("travel_time_min_mean", 0))
        
        # Ensure Equity is present (defaults to None/0 if missing in baseline)
        if "equity" not in scores:
             # Try to find equity in specific locations or default
             scores["equity"] = scores.get("equity_score", None)

        metrics["scores"] = scores
            
        logger.info(f"Returning metrics summary for {city} ({type}, {mode})")
        return metrics

    except Exception as e:
        logger.error(f"Error reading metrics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.get("/api/pois")
async def get_pois(
    city: str = Query(DEFAULT_CITY),
    limit: Optional[int] = Query(5000, ge=1, le=50000, description="Maximum POIs to return")
):
    """Return POIs for a specific city, filtered by city bounding box."""
    target_city = normalize_city_name(city) if city else DEFAULT_CITY
    cdm = CityDataManager(target_city)
    
    # Define path for processed JSON
    processed_pois_path = cdm.raw_pois.parent.parent.parent / "graph" / "processed_pois.json"
    
    filepath = str(processed_pois_path)
    
    # Check if processed file exists
    if not os.path.exists(filepath):
        logger.error(f"Processed POIs not found for {target_city} at {filepath}")
        return {"type": "FeatureCollection", "features": []}

    try:
        # Load and filter POIs by bounding box
        with open(filepath, 'r') as f:
            geojson = json.load(f)
        
        features = geojson.get("features", [])
        original_count = len(features)
        
        # Get city bounding box for filtering outliers
        bbox = CITY_BOUNDS.get(target_city)
        
        if bbox:
            west, south, east, north = bbox
            filtered_features = []
            for feature in features:
                geom = feature.get("geometry")
                if not geom or not geom.get("coordinates"):
                    continue
                coords = geom["coordinates"]
                lon, lat = coords[0], coords[1]
                # Filter by bounding box
                if west <= lon <= east and south <= lat <= north:
                    filtered_features.append(feature)
            
            features = filtered_features
            logger.info(f"Filtered {original_count} POIs to {len(features)} within bbox for {target_city}")
        
        # Apply limit for performance
        if limit and len(features) > limit:
            features = features[:limit]
            logger.info(f"Limited POIs to {limit} for {target_city}")
        
        logger.info(f"Returning {len(features)} POIs for {target_city}")
        return {"type": "FeatureCollection", "features": features}
        
    except Exception as e:
        logger.error(f"Error loading POIs for {target_city}: {e}")
        return {"type": "FeatureCollection", "features": []}


@app.get("/api/suggestions")
async def get_suggestions(
    city: str = Query(DEFAULT_CITY),
    mode: str = Query("ga_milp_pnmlr", pattern="^(ga_only|ga_milp|ga_milp_pnmlr)$")
):
    """Return optimization suggestions for a specific city and mode."""
    target_city = city if city else DEFAULT_CITY
    cdm = CityDataManager(target_city, mode=mode)
    
    best_candidate_path = cdm.best_candidate(mode)
    optimized_pois_path = cdm.optimized_pois(mode)
    
    if not best_candidate_path.exists() or not optimized_pois_path.exists():
        logger.warning(f"Optimization results not found for city {city}, mode {mode}")
        return {"type": "FeatureCollection", "features": []}
    
    try:
        # Load best candidate to get optimization details
        with open(best_candidate_path, 'r') as f:
            best_candidate = json.load(f)
        
        # Load optimized POIs (small file with only new placements)
        with open(optimized_pois_path, 'r') as f:
            pois = json.load(f)
        
        # Get features (already filtered to optimized POIs only)
        optimized_features = pois.get('features', [])
        
        logger.info(f"Returning {len(optimized_features)} optimization suggestions for {city}, mode {mode}")
        return {
            "type": "FeatureCollection",
            "features": optimized_features,
            "metadata": {
                "generation": best_candidate.get('generation'),
                "fitness": best_candidate.get('metrics', {}).get('fitness', best_candidate.get('fitness')),
                "city": city,
                "mode": mode
            }
        }
    except Exception as e:
        logger.error(f"Error reading suggestions for {city}, mode {mode}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.post("/api/optimize")
async def optimize(request: OptimizeRequest):
    logger.info(f"Optimization request received: {request}")
    
    # Construct command requested by frontend action
    cmd = [
        sys.executable, RUN_SCRIPT,
        "--skip-landuse"
    ]
    
    write_optimization_status(
        status="queued",
        stage="initializing",
        message="Optimization run queued",
        percent=2,
        pipelines={
            "data": "pending",
            "optimization": "pending",
            "landuse": "skipped"
        }
    )

    try:
        # Start optimization in background (non-blocking)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        logger.info(f"Started optimization process with PID {process.pid}")
        return {
            "status": "started",
            "message": "Optimization process started in background",
            "process_id": process.pid
        }
    except Exception as e:
        logger.error(f"Error starting optimization: {e}")
        write_optimization_status(
            status="failed",
            stage="initializing",
            message=str(e),
            percent=0
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rescore")
async def rescore(request: RescoreRequest):
    logger.info(f"Rescore request received for {len(request.selected_ids)} nodes")
    
    # Construct command to re-run scoring with custom POIs
    cmd = [
        sys.executable, RUN_SCRIPT,
        "--pipeline", "optimization",
        "--place", request.location
    ]
    
    try:
        # Start rescoring in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        logger.info(f"Started rescoring process with PID {process.pid}")
        return {
            "status": "started",
            "message": "Rescoring process started",
            "process_id": process.pid
        }
    except Exception as e:
        logger.error(f"Error starting rescore: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feasibility/{amenity}")
async def get_feasibility(amenity: str):
    filepath = os.path.join(BASE_DIR, "data", "landuse", f"feasibility_{amenity}.csv")
    
    if not os.path.exists(filepath):
        logger.warning(f"Feasibility data not found for {amenity}")
        return {"feasible_nodes": []}
    
    try:
        df = pd.read_csv(filepath)
        df = df.replace({np.nan: None})
        feasible = df.to_dict('records')
        logger.info(f"Returning {len(feasible)} feasibility records for {amenity}")
        return {"feasible_nodes": feasible}
    except Exception as e:
        logger.error(f"Error reading feasibility data: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.get("/api/placements/{amenity}")
async def get_placements(amenity: str):
    filepath = os.path.join(BASE_DIR, "data", "landuse", f"gee_placements_{amenity}.geojson")
    
    if not os.path.exists(filepath):
        logger.warning(f"Placement data not found for {amenity}")
        return {"type": "FeatureCollection", "features": []}
    
    try:
        with open(filepath, 'r') as f:
            geojson = json.load(f)
        logger.info(f"Returning {len(geojson.get('features', []))} placements for {amenity}")
        return geojson
    except Exception as e:
        logger.error(f"Error reading placements: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.get("/api/optimization/status")
async def get_optimization_status():
    progress_path = os.path.join(BASE_DIR, "data", "optimization", "runs", "progress.json")
    
    if not os.path.exists(progress_path):
        return {"status": "not_started"}
    
    try:
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        return progress
    except Exception as e:
        logger.error(f"Error reading optimization status: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/api/optimization/reset")
async def reset_optimization_status():
    """Reset the optimization status to not_started. Useful for clearing stale state."""
    progress_path = os.path.join(BASE_DIR, "data", "optimization", "runs", "progress.json")
    
    try:
        if os.path.exists(progress_path):
            os.remove(progress_path)
        logger.info("Optimization status reset")
        return {"status": "success", "message": "Optimization status reset"}
    except Exception as e:
        logger.error(f"Error resetting optimization status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/optimization/results")
async def get_optimization_results(
    city: str = Query(DEFAULT_CITY),
    mode: str = Query("ga_milp_pnmlr", pattern="^(ga_only|ga_milp|ga_milp_pnmlr)$")
):
    """Get the best optimization results (GA+MILP hybrid) for a city and mode."""
    target_city = city if city else DEFAULT_CITY
    cdm = CityDataManager(target_city, mode=mode)
    best_candidate_path = cdm.best_candidate(mode)
    
    if not best_candidate_path.exists():
        logger.warning(f"Best candidate not found for {target_city}/{mode}")
        raise HTTPException(status_code=404, detail=f"Optimization results not found for {target_city}/{mode}")
    
    try:
        with open(best_candidate_path, 'r') as f:
            best_candidate = json.load(f)
        
        return {
            "generation": best_candidate.get('generation'),
            "fitness": best_candidate.get('metrics', {}).get('fitness'),
            "metrics": best_candidate.get('metrics', {}),
            "placements": best_candidate.get('metrics', {}).get('placements', {}),
            "candidate": best_candidate.get('candidate'),
            "template": best_candidate.get('template'),
            "city": target_city,
            "mode": mode
        }
    except Exception as e:
        logger.error(f"Error reading optimization results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/optimization/pois")
async def get_optimization_pois_legacy(
    city: str = Query(DEFAULT_CITY),
    mode: str = Query("ga_milp_pnmlr", pattern="^(ga_only|ga_milp|ga_milp_pnmlr)$")
):
    """Get optimized POIs as GeoJSON for a city and mode (legacy compatibility endpoint)."""
    target_city = city if city else DEFAULT_CITY
    cdm = CityDataManager(target_city, mode=mode)
    optimized_pois_path = cdm.optimized_pois(mode)
    
    if not optimized_pois_path.exists():
        logger.warning(f"Optimized POIs not found for {target_city}/{mode}")
        raise HTTPException(status_code=404, detail=f"Optimized POIs not found for {target_city}/{mode}")
    
    try:
        with open(optimized_pois_path, 'r') as f:
            geojson = json.load(f)
        
        features = geojson.get('features', [])
        
        # Transform properties to match frontend expectations
        for feature in features:
            props = feature.get('properties', {})
            # Add amenity_type and id fields that frontend expects
            if 'amenity' in props:
                props['amenity_type'] = props['amenity']
            if 'osmid' in props:
                props['id'] = str(props['osmid'])
            # Add description
            if 'amenity' in props:
                amenity_name = props['amenity'].replace('_', ' ').title()
                props['description'] = f"Optimized {amenity_name} placement"
        
        logger.info(f"Returning {len(features)} optimized POIs for {target_city}/{mode}")
        
        return geojson
    except Exception as e:
        logger.error(f"Error reading optimized POIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/optimization/history")
async def get_optimization_history():
    """Get evolution history across all generations"""
    runs_dir = os.path.join(BASE_DIR, "data", "optimization", "runs")
    
    try:
        # Find all generation files
        generation_files = sorted([
            f for f in os.listdir(runs_dir) 
            if f.startswith('generation_') and f.endswith('.json')
        ])
        
        if not generation_files:
            return {"generations": [], "message": "No generation history found"}
        
        history = []
        for gen_file in generation_files:
            gen_path = os.path.join(runs_dir, gen_file)
            with open(gen_path, 'r') as f:
                gen_data = json.load(f)
            
            # Extract generation number from filename
            gen_num = int(gen_file.replace('generation_', '').replace('.json', ''))
            
            # Get best fitness from this generation
            best_fitness = None
            if isinstance(gen_data, list) and len(gen_data) > 0:
                best_fitness = gen_data[0].get('metrics', {}).get('fitness')
            
            history.append({
                "generation": gen_num,
                "best_fitness": best_fitness,
                "population_size": len(gen_data) if isinstance(gen_data, list) else 0
            })
        
        logger.info(f"Returning history for {len(history)} generations")
        return {"generations": history}
        
    except Exception as e:
        logger.error(f"Error reading optimization history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/optimization/summary")
async def get_optimization_summary():
    """Get optimization run summary with stats"""
    summary_path = os.path.join(BASE_DIR, "data", "optimization", "runs", "summary.json")
    
    if not os.path.exists(summary_path):
        logger.warning("Summary not found")
        return {"message": "No summary available"}
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        logger.info("Returning optimization summary")
        return summary
    except Exception as e:
        logger.error(f"Error reading summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/optimization/comparison")
async def get_baseline_vs_optimized_comparison():
    """Compare baseline vs optimized metrics"""
    baseline_metrics_path = os.path.join(DATA_DIR, "baseline_metrics_summary.json")
    optimized_metrics_path = os.path.join(DATA_DIR, "optimized_metrics_summary.json")
    
    comparison = {
        "baseline": {},
        "optimized": {},
        "improvements": {}
    }
    
    try:
        # Load baseline metrics
        if os.path.exists(baseline_metrics_path):
            with open(baseline_metrics_path, 'r') as f:
                comparison["baseline"] = json.load(f)
        
        # Load optimized metrics
        if os.path.exists(optimized_metrics_path):
            with open(optimized_metrics_path, 'r') as f:
                comparison["optimized"] = json.load(f)
        
        # Calculate improvements
        if comparison["baseline"] and comparison["optimized"]:
            for key in comparison["baseline"]:
                if key in comparison["optimized"]:
                    baseline_val = comparison["baseline"][key]
                    optimized_val = comparison["optimized"][key]
                    
                    if isinstance(baseline_val, (int, float)) and isinstance(optimized_val, (int, float)):
                        improvement = optimized_val - baseline_val
                        percent_change = (improvement / baseline_val * 100) if baseline_val != 0 else 0
                        comparison["improvements"][key] = {
                            "absolute": improvement,
                            "percent": percent_change
                        }
        
        logger.info("Returning baseline vs optimized comparison")
        return comparison
        
    except Exception as e:
        logger.error(f"Error creating comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# CITY DATA STATUS ENDPOINT
# ============================================================

@app.get("/api/city-data-status")
async def get_city_data_status(city: str = Query(...)):
    """Check if a city has baseline and/or optimized data available."""
    normalized_city = normalize_city_name(city)
    city_dir = os.path.join(CITIES_DIR, normalized_city)
    baseline_path = os.path.join(city_dir, "baseline", "metrics_summary.json")
    
    # Check for any optimized mode
    optimized_modes = {}
    for mode_id in OPTIMIZATION_MODES.keys():
        mode_metrics = os.path.join(city_dir, "optimized", mode_id, "metrics_summary.json")
        optimized_modes[mode_id] = os.path.exists(mode_metrics)
    
    return {
        "city": normalized_city,
        "has_baseline": os.path.exists(baseline_path),
        "optimized_modes": optimized_modes,
        "any_optimized": any(optimized_modes.values())
    }

# ============================================================
# MULTI-MODE OPTIMIZATION ENDPOINTS
# ============================================================

@app.get("/api/modes")
async def get_optimization_modes(city: str = Query(DEFAULT_CITY)):
    """List available optimization modes for a city."""
    modes = []
    for mode_id, mode_name in OPTIMIZATION_MODES.items():
        mode_dir = get_mode_dir(city, mode_id)
        exists = os.path.exists(mode_dir)
        has_data = exists and os.path.exists(os.path.join(mode_dir, "nodes_with_scores.parquet"))
        modes.append({
            "id": mode_id,
            "name": mode_name,
            "available": has_data,
            "path": mode_dir if exists else None
        })
    return {"city": city, "modes": modes}

@app.get("/api/modes/{mode}/nodes")
async def get_mode_nodes(
    mode: str,
    city: str = Query(DEFAULT_CITY),
    limit: Optional[int] = Query(5000, ge=1, le=100000),
    offset: Optional[int] = Query(0, ge=0)
):
    """Get nodes with scores for a specific optimization mode."""
    if mode not in OPTIMIZATION_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {list(OPTIMIZATION_MODES.keys())}")
    
    mode_dir = get_mode_dir(city, mode)
    parquet_path = os.path.join(mode_dir, "nodes_with_scores.parquet")
    
    if not os.path.exists(parquet_path):
        logger.warning(f"Mode data not found: {parquet_path}")
        return []
    
    try:
        df = pd.read_parquet(parquet_path)
        df = df.replace({np.nan: None})
        
        # Apply pagination
        total_count = len(df)
        if offset:
            df = df.iloc[offset:]
        if limit:
            df = df.iloc[:limit]
        
        nodes = []
        for idx, row in df.iterrows():
            if pd.isna(row.get('lon')) or pd.isna(row.get('lat')):
                continue
            
            osmid = row.get('osmid') if 'osmid' in row else str(idx)
            
            node = {
                "osmid": str(osmid),
                "x": float(row['lon']),
                "y": float(row['lat']),
                "accessibility_score": row.get('optimized_accessibility_score', row.get('accessibility_score')),
                "walkability_score": row.get('optimized_walkability', row.get('walkability')),
                "travel_time_min": row.get('optimized_travel_time_min', row.get('travel_time_min')),
            }
            nodes.append(node)
        
        logger.info(f"Returning {len(nodes)}/{total_count} nodes for mode {mode}")
        return nodes
    except Exception as e:
        logger.error(f"Error processing mode nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/modes/{mode}/metrics")
async def get_mode_metrics(mode: str, city: str = Query(DEFAULT_CITY)):
    """Get metrics summary for a specific optimization mode."""
    if mode not in OPTIMIZATION_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {list(OPTIMIZATION_MODES.keys())}")
    
    mode_dir = get_mode_dir(city, mode)
    # Prefer mode-prefixed file first (has correct/updated metrics)
    prefixed_path = os.path.join(mode_dir, f"{mode}_metrics_summary.json")
    generic_path = os.path.join(mode_dir, "metrics_summary.json")
    
    if os.path.exists(prefixed_path):
        metrics_path = prefixed_path
    elif os.path.exists(generic_path):
        metrics_path = generic_path
    else:
        logger.warning(f"Metrics not found: {prefixed_path} or {generic_path}")
        return {"network": {}, "scores": {}}
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logger.error(f"Error reading mode metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/modes/{mode}/pois")
async def get_mode_pois(mode: str, city: str = Query(DEFAULT_CITY)):
    """Get optimized POIs for a specific mode."""
    if mode not in OPTIMIZATION_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {list(OPTIMIZATION_MODES.keys())}")
    
    mode_dir = get_mode_dir(city, mode)
    pois_path = os.path.join(mode_dir, "optimized_pois.geojson")
    
    if not os.path.exists(pois_path):
        logger.warning(f"POIs not found: {pois_path}")
        return {"type": "FeatureCollection", "features": []}
    
    try:
        with open(pois_path, 'r') as f:
            geojson = json.load(f)
        
        # Add mode info to properties
        for feature in geojson.get('features', []):
            feature['properties']['optimization_mode'] = mode
            feature['properties']['mode_name'] = OPTIMIZATION_MODES[mode]
        
        return geojson
    except Exception as e:
        logger.error(f"Error reading mode POIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/modes/comparison")
async def get_modes_comparison(city: str = Query(DEFAULT_CITY)):
    """Compare metrics across all optimization modes."""
    # Load baseline metrics
    normalized_city = normalize_city_name(city)
    city_dir = os.path.join(CITIES_DIR, normalized_city)
    baseline_path = os.path.join(city_dir, "baseline", "baseline_metrics_summary.json")
    
    baseline_metrics = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline_metrics = json.load(f)
    else:
        # Fallback to standard name if specific one doesn't exist
        fallback_path = os.path.join(city_dir, "baseline", "metrics_summary.json")
        if os.path.exists(fallback_path):
             with open(fallback_path, 'r') as f:
                baseline_metrics = json.load(f)

    comparison = {
        "city": city,
        "baseline": baseline_metrics,
        "modes": {}
    }
    
    for mode_id in OPTIMIZATION_MODES.keys():
        mode_dir = get_mode_dir(city, mode_id)
        # Prefer mode-prefixed file first (has correct/updated metrics)
        prefixed_path = os.path.join(mode_dir, f"{mode_id}_metrics_summary.json")
        generic_path = os.path.join(mode_dir, "metrics_summary.json")
        
        if os.path.exists(prefixed_path):
            metrics_path = prefixed_path
        elif os.path.exists(generic_path):
            metrics_path = generic_path
        else:
            metrics_path = None

        mode_data = {
            "name": OPTIMIZATION_MODES[mode_id],
            "available": False,
            "metrics": {},
            "improvements": {}
        }
        
        if metrics_path and os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    mode_metrics = json.load(f)
                mode_data["available"] = True
                mode_data["metrics"] = mode_metrics
                
                # Calculate improvements vs baseline
                # Access "citywide" key where the actual mean scores are stored
                baseline_scores = baseline_metrics.get("scores", {}).get("citywide", {})
                mode_scores = mode_metrics.get("scores", {}).get("citywide", {})
                
                for key in ["accessibility_mean", "walkability_mean", "travel_time_min_mean"]:
                    # Handle both potentially missing keys and None values
                    baseline_val = baseline_scores.get(key)
                    mode_val = mode_scores.get(key)
                    
                    if baseline_val is not None and mode_val is not None:
                        if key == "travel_time_min_mean":
                            # Lower is better for travel time
                            improvement = baseline_val - mode_val
                            percent = (improvement / baseline_val * 100) if baseline_val else 0
                        else:
                            # Higher is better for accessibility/walkability
                            improvement = mode_val - baseline_val
                            percent = (improvement / baseline_val * 100) if baseline_val else 0
                        mode_data["improvements"][key] = {
                            "absolute": round(improvement, 2),
                            "percent": round(percent, 2)
                        }
            except Exception as e:
                logger.error(f"Error loading metrics for {mode_id}: {e}")
        
        comparison["modes"][mode_id] = mode_data
    
    return comparison

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

