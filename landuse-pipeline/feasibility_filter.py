#!/usr/bin/env python3
"""
Post-GEE integration for a single amenity.
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path
import argparse
import sys

# Add project root for CityDataManager
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from city_paths import CityDataManager

def load_data(cdm, amenity):
    print(f"Loading tables for {amenity}...")
    nodes = pd.read_parquet(cdm.baseline_nodes)
    
    feas_path = cdm.landuse_dir / f"landuse_feasibility_{amenity}.csv"
    feas = pd.read_csv(feas_path)
    
    place_path = cdm.landuse_dir / f"landuse_placements_{amenity}.geojson"
    placements = gpd.read_file(place_path)
    
    return nodes, feas, placements

def filter_feasible(nodes: pd.DataFrame, feas: pd.DataFrame, amenity: str) -> pd.DataFrame:
    """Return only nodes that are feasible for the target amenity."""
    feas_amen = feas[feas["amenity"] == amenity].copy()
    if feas_amen["feasible"].dtype != bool:
        feas_amen["feasible"] = feas_amen["feasible"].astype(bool)
    feasible_only = feas_amen[feas_amen["feasible"]]
    
    # osmid might be index in parquet
    if 'osmid' not in nodes.columns:
        nodes = nodes.reset_index()
        
    nodes["osmid"] = nodes["osmid"].astype("int64")
    feasible_only["node_id"] = feasible_only["node_id"].astype("int64")
    
    merged = nodes.merge(
        feasible_only,
        left_on="osmid",
        right_on="node_id",
        how="inner",
        suffixes=("", "_gee")
    )
    return feasible_only, merged

def main():
    parser = argparse.ArgumentParser(description="Filter GEE feasibility results")
    parser.add_argument("--city", default="bangalore")
    parser.add_argument("--mode", default="ga_only")
    parser.add_argument("--amenity", required=True)
    args = parser.parse_args()
    
    cdm = CityDataManager(args.city, project_root=project_root, mode=args.mode)
    nodes, feas, placements = load_data(cdm, args.amenity)
    
    feasible_only, merged = filter_feasible(nodes, feas, args.amenity)
    
    out_dir = cdm.landuse_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    feasible_only.to_csv(out_dir / f"gee_feasible_nodes_{args.amenity}.csv", index=False)
    merged.to_csv(out_dir / f"gee_feasible_nodes_{args.amenity}_merged.csv", index=False)
    placements.to_file(out_dir / f"gee_placements_{args.amenity}.geojson", driver="GeoJSON")
    
    print(f"Saved results to {out_dir}")

if __name__ == "__main__":
    main()
