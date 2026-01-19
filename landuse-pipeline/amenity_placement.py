#!/usr/bin/env python3
"""
Build GEE candidate points for a single amenity.
"""
import json
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import argparse

# Add project root for CityDataManager
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from city_paths import CityDataManager

def parse_best_candidate(path: Path) -> dict:
    """Return dict: amenity -> list[node_ids] from best_candidate.json."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cand_str = data["candidate"]
    amenity_map = {}
    for block in cand_str.split("|"):
        if not block.strip(): continue
        name, ids_str = block.split(":")
        node_ids = [int(x) for x in ids_str.split(",") if x]
        amenity_map[name.strip()] = node_ids
    return amenity_map

def make_geojson_for_amenity(cdm, amenity: str):
    # --- 1. Parse GA output ---
    best_cand_path = cdm.best_candidate(cdm.mode)
    amenity_map = parse_best_candidate(best_cand_path)
    if amenity not in amenity_map:
        raise ValueError(f"Amenity '{amenity}' not found in {best_cand_path.name}")
    candidate_ids = set(amenity_map[amenity])

    # --- 2. Load node table ---
    df = pd.read_parquet(cdm.baseline_nodes)
    if df.index.name == 'osmid':
        df = df.reset_index()

    # --- 3. Build GeoDataFrame ---
    df_sub = df[df['osmid'].astype(int).isin(candidate_ids)].copy()
    if df_sub.empty:
        raise RuntimeError(f"No rows for '{amenity}' in GA nodes.")

    df_sub["node_id"] = df_sub['osmid'].astype("int64")
    df_sub["amenity"] = amenity
    geometry = [Point(xy) for xy in zip(df_sub['lon'], df_sub['lat'])]
    gdf = gpd.GeoDataFrame(df_sub[["node_id", "amenity"]], geometry=geometry, crs="EPSG:4326")

    # --- 4. Save GeoJSON ---
    out_dir = cdm.landuse_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"node_candidates_{amenity}.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"✅ Amenity '{amenity}': {len(gdf)} nodes → {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Build GEE candidates")
    parser.add_argument("--city", default="bangalore")
    parser.add_argument("--mode", default="ga_only")
    parser.add_argument("--amenity", required=True)
    args = parser.parse_args()
    
    cdm = CityDataManager(args.city, project_root=project_root, mode=args.mode)
    make_geojson_for_amenity(cdm, args.amenity)

if __name__ == "__main__":
    main()
