from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import pandas as pd
import json
import geopandas as gpd

# Pipelines
from pipeline.fetch_data import fetch_map_data
from pipeline.scoring import run_scoring
from optimization.hybrid_ga import run_optimization_algorithm

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_ROOT = Path("data")
current_data = {"baseline": [], "optimized": [], "suggestions": []}

class OptimizeRequest(BaseModel):
    location: str
    budget: int = 50000000
    max_amenities: int = 10
    add_schools: bool = True
    add_hospitals: bool = True
    add_parks: bool = True

class RescoreRequest(BaseModel):
    location: str
    selected_ids: list[str]

def sanitize_name(name: str):
    return "".join(c for c in name if c.isalnum()).lower()

def load_nodes_safe(path: Path):
    """Loads parquet and ensures osmid is a column, not index"""
    if not path.exists(): return []
    df = pd.read_parquet(path)
    
    # CRITICAL FIX: Move osmid from Index to Column
    if 'osmid' not in df.columns:
        df = df.reset_index()
        
    return df.fillna(0).to_dict(orient="records")

@app.post("/api/optimize")
def optimize_city(req: OptimizeRequest):
    loc_name = req.location
    folder_name = sanitize_name(loc_name)
    loc_dir = DATA_ROOT / folder_name
    
    constraints = req.dict()
    
    try:
        if loc_dir.exists(): shutil.rmtree(loc_dir)
        
        # Pipeline
        fetch_map_data(loc_name, loc_dir)
        run_scoring(loc_dir, loc_dir / "baseline_nodes.parquet", is_optimized=False)
        run_optimization_algorithm(loc_dir, loc_dir / "baseline_nodes.parquet", constraints)
        run_scoring(loc_dir, loc_dir / "optimized_nodes.parquet", is_optimized=True, new_pois_geojson=loc_dir / "suggestions.geojson")

        # Load & Cache (With Fix)
        current_data["baseline"] = load_nodes_safe(loc_dir / "baseline_nodes.parquet")
        current_data["optimized"] = load_nodes_safe(loc_dir / "optimized_nodes.parquet")
        
        if (loc_dir / "suggestions.geojson").exists():
            sug_gdf = gpd.read_file(loc_dir / "suggestions.geojson")
            current_data["suggestions"] = json.loads(sug_gdf.to_json())['features']
        else:
            current_data["suggestions"] = []
            
        return {"status": "success"}

    except Exception as e:
        print(f"Optimization Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rescore")
def rescore_scenario(req: RescoreRequest):
    print(f"🔄 Rescoring with {len(req.selected_ids)} active suggestions...")
    folder_name = sanitize_name(req.location)
    loc_dir = DATA_ROOT / folder_name
    
    if not (loc_dir / "suggestions.geojson").exists():
        return {"avg_score": 0, "node_updates": {}}
        
    all_sugs = gpd.read_file(loc_dir / "suggestions.geojson")
    
    # Robust Filtering
    active_sugs = all_sugs[all_sugs['id'].isin(req.selected_ids)]
    
    subset_path = loc_dir / "temp_subset.geojson"
    if not active_sugs.empty:
        active_sugs.to_file(subset_path, driver="GeoJSON")
    else:
        if subset_path.exists(): subset_path.unlink()
        
    # Run Scoring
    temp_out = loc_dir / "temp_nodes.parquet"
    # Note: run_scoring saves to temp_out
    nodes = run_scoring(loc_dir, temp_out, is_optimized=(not active_sugs.empty), new_pois_geojson=subset_path)
    
    avg_score = nodes['walkability'].mean()
    
    # CRITICAL FIX: Ensure OSMID mapping uses the Index
    updates = {}
    for idx, row in nodes.iterrows():
        # osmid is usually the index in OSMnx graphs
        node_id = str(idx) if 'osmid' not in row else str(row['osmid'])
        updates[node_id] = float(row['walkability'])
    
    print(f"✅ Generated updates for {len(updates)} nodes. New Avg: {avg_score:.2f}")
    return {
        "avg_score": round(avg_score, 2),
        "node_updates": updates 
    }

@app.get("/api/nodes")
def get_nodes(type: str = "baseline"):
    return current_data.get(type, [])

@app.get("/api/suggestions")
def get_suggestions():
    return current_data.get("suggestions", [])