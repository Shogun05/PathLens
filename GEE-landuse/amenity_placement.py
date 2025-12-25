#!/usr/bin/env python3
"""
Build GEE candidate points for a single amenity from:
- best_candidate.json  (GA result)
- optimized_nodes_with_scores.csv  (node attributes with lat/lon)

Usage:
    python build_gee_candidates.py hospital
    python build_gee_candidates.py school
"""

import json
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


BASE_DIR = Path(__file__).resolve().parent
BEST_CANDIDATE_PATH = BASE_DIR / "best_candidate.json"
NODES_CSV_PATH = BASE_DIR / "optimized_nodes_with_scores.csv"


def parse_best_candidate(path: Path) -> dict:
    """Return dict: amenity -> list[node_ids] from best_candidate.json."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cand_str = data["candidate"]
    amenity_map = {}

    # Example string:
    # bus_station:10020025539,10991934818|hospital:10035832527,...
    for block in cand_str.split("|"):
        if not block.strip():
            continue
        name, ids_str = block.split(":")
        node_ids = [int(x) for x in ids_str.split(",") if x]
        amenity_map[name.strip()] = node_ids

    return amenity_map


def make_geojson_for_amenity(amenity: str):
    # --- 1. Parse GA output ---
    amenity_map = parse_best_candidate(BEST_CANDIDATE_PATH)
    if amenity not in amenity_map:
        raise ValueError(
            f"Amenity '{amenity}' not found in best_candidate.json. "
            f"Available: {list(amenity_map.keys())}"
        )
    candidate_ids = set(amenity_map[amenity])

    # --- 2. Load node table ---
    df = pd.read_csv(NODES_CSV_PATH)

    # Ensure column names are correct for your file:
    # osmid,y,x,street_count,lon,lat,...
    id_col = "osmid"
    lat_col = "lat"
    lon_col = "lon"

    missing = [c for c in (id_col, lat_col, lon_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    # Filter only nodes used by the optimizer for this amenity
    df_sub = df[df[id_col].isin(candidate_ids)].copy()
    if df_sub.empty:
        raise RuntimeError(
            f"No rows in CSV for amenity '{amenity}' and osmid in {len(candidate_ids)} GA nodes."
        )

    # --- 3. Build GeoDataFrame ---
    df_sub["node_id"] = df_sub[id_col].astype("int64")
    df_sub["amenity"] = amenity

    geometry = [Point(xy) for xy in zip(df_sub[lon_col], df_sub[lat_col])]
    gdf = gpd.GeoDataFrame(
        df_sub[["node_id", "amenity"]],
        geometry=geometry,
        crs="EPSG:4326",
    )

    # --- 4. Save GeoJSON ---
    out_path = BASE_DIR / f"node_candidates_{amenity}.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    print(
        f"✅ Amenity '{amenity}': {len(gdf)} nodes "
        f"→ {out_path.relative_to(BASE_DIR)}"
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_gee_candidates.py <amenity>")
        print("Example: python build_gee_candidates.py hospital")
        sys.exit(1)

    amenity = sys.argv[1]
    make_geojson_for_amenity(amenity)


if __name__ == "__main__":
    main()
