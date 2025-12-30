from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import os
import subprocess
import json
import logging

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
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Max nodes to return"),
    offset: Optional[int] = Query(0, ge=0, description="Offset for pagination"),
    bbox: Optional[str] = Query(None, description="Bounding box: west,south,east,north")
):
    filename = "baseline_nodes_with_scores.csv" if type == "baseline" else "optimized_nodes_with_scores.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        if type == "optimized":
            return []
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    
    try:
        df = pd.read_csv(filepath)
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
        
        nodes = []
        for _, row in df.iterrows():
            # Ensure required fields are present
            if pd.isna(row.get('lon')) or pd.isna(row.get('lat')):
                continue
                
            node = Node(
                osmid=str(row['osmid']),
                x=float(row['lon']),
                y=float(row['lat']),
                accessibility_score=row.get('accessibility_score'),
                walkability_score=row.get('walkability'),
                equity_score=row.get('equity_score'),
                travel_time_min=row.get('travel_time_min'),
                betweenness_centrality=row.get('betweenness_centrality'),
                dist_to_school=row.get('dist_to_school'),
                dist_to_hospital=row.get('dist_to_hospital'),
                dist_to_park=row.get('dist_to_park')
            )
            nodes.append(node)
        
        logger.info(f"Returning {len(nodes)}/{total_count} nodes for type {type}")
        return nodes
    except Exception as e:
        logger.error(f"Error processing nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.get("/api/h3-aggregations")
async def get_h3_aggregations(type: str = Query(..., pattern="^(baseline|optimized)$")):
    filename = "baseline_h3_agg.csv" if type == "baseline" else "optimized_h3_agg.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return {"hexagons": []}
    
    try:
        df = pd.read_csv(filepath)
        df = df.replace({np.nan: None})
        hexagons = df.to_dict('records')
        logger.info(f"Returning {len(hexagons)} H3 hexagons for type {type}")
        return {"hexagons": hexagons}
    except Exception as e:
        logger.error(f"Error processing H3 aggregations: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.get("/api/metrics-summary")
async def get_metrics_summary(type: str = Query(..., pattern="^(baseline|optimized)$")):
    filename = "baseline_metrics_summary.json" if type == "baseline" else "optimized_metrics_summary.json"
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        logger.info(f"Returning metrics summary for type {type}")
        return metrics
    except Exception as e:
        logger.error(f"Error reading metrics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.get("/api/pois")
async def get_pois():
    filepath = os.path.join(BASE_DIR, "data", "optimization", "runs", "combined_pois.geojson")
    
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        # Fallback to baseline amenities
        fallback_path = os.path.join(BASE_DIR, "data", "raw", "osm", "amenities.geojson")
        if not os.path.exists(fallback_path):
            return {"type": "FeatureCollection", "features": []}
        filepath = fallback_path
    
    try:
        with open(filepath, 'r') as f:
            geojson = json.load(f)
        logger.info(f"Returning {len(geojson.get('features', []))} POIs")
        return geojson
    except Exception as e:
        logger.error(f"Error reading POIs: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.get("/api/suggestions")
async def get_suggestions():
    best_candidate_path = os.path.join(BASE_DIR, "data", "optimization", "runs", "best_candidate.json")
    combined_pois_path = os.path.join(BASE_DIR, "data", "optimization", "runs", "combined_pois.geojson")
    
    if not os.path.exists(best_candidate_path) or not os.path.exists(combined_pois_path):
        logger.warning("Optimization results not found")
        return {"type": "FeatureCollection", "features": []}
    
    try:
        # Load best candidate to get optimization details
        with open(best_candidate_path, 'r') as f:
            best_candidate = json.load(f)
        
        # Load POIs and filter for optimized ones
        with open(combined_pois_path, 'r') as f:
            pois = json.load(f)
        
        # Filter for optimized POIs only
        optimized_features = [
            feature for feature in pois.get('features', [])
            if feature.get('properties', {}).get('source') == 'optimized'
        ]
        
        logger.info(f"Returning {len(optimized_features)} optimization suggestions")
        return {
            "type": "FeatureCollection",
            "features": optimized_features,
            "metadata": {
                "generation": best_candidate.get('generation'),
                "fitness": best_candidate.get('fitness')
            }
        }
    except Exception as e:
        logger.error(f"Error reading suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.post("/api/optimize")
async def optimize(request: OptimizeRequest):
    logger.info(f"Optimization request received: {request}")
    
    # Construct command
    cmd = [
        "python3", RUN_SCRIPT,
        "--pipeline", "optimization",
        "--ga-population", str(request.max_amenities * 10),  # Scale population with amenities
        "--ga-generations", "10"
    ]
    
    # Add place if specified
    if request.location:
        cmd.extend(["--place", request.location])
    
    try:
        # Start optimization in background (non-blocking)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Started optimization process with PID {process.pid}")
        return {
            "status": "started",
            "message": "Optimization process started in background",
            "process_id": process.pid
        }
    except Exception as e:
        logger.error(f"Error starting optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rescore")
async def rescore(request: RescoreRequest):
    logger.info(f"Rescore request received for {len(request.selected_ids)} nodes")
    
    # Construct command to re-run scoring with custom POIs
    cmd = [
        "python3", RUN_SCRIPT,
        "--pipeline", "optimization",
        "--place", request.location
    ]
    
    try:
        # Start rescoring in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
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
        return {
            "status": "running",
            "current_generation": progress.get('generation'),
            "best_fitness": progress.get('best_fitness'),
            "elapsed_time": progress.get('elapsed_time'),
            "timestamp": progress.get('timestamp')
        }
    except Exception as e:
        logger.error(f"Error reading optimization status: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
