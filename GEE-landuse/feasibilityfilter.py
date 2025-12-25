#!/usr/bin/env python3
"""
Post-GEE integration for a single amenity (e.g. hospital).

Inputs:
  - optimized_nodes_with_scores.csv         # PathLens node table
  - pathlens_feasibility_single_amenity.csv # GEE feasibility (per node_id)
  - pathlens_placements_single_amenity.geojson # GEE polygons (per node_id)

Outputs:
  - gee_feasible_nodes_<amenity>.csv
  - gee_feasible_nodes_<amenity>_merged.csv
  - gee_placements_<amenity>.geojson
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# ---- CONFIG: change per amenity run ----
AMENITY = "hospital"

# File locations (adjust if your filenames differ)
NODES_CSV = BASE_DIR / "optimized_nodes_with_scores.csv"
FEAS_CSV = BASE_DIR / "pathlens_feasibility_single_amenity.csv"
PLACEMENTS_GEOJSON = BASE_DIR / "pathlens_placements_single_amenity.geojson"


def load_data():
    print("Loading core tables...")

    nodes = pd.read_csv(NODES_CSV)

    feas = pd.read_csv(FEAS_CSV)
    # Ensure column names are as expected from the GEE export
    # Should contain: node_id, amenity, free_area_m2, min_area_req, feasible, has_patch, patch_count
    print("Feasibility columns:", feas.columns.tolist())

    placements = gpd.read_file(PLACEMENTS_GEOJSON)
    print("Placements columns:", placements.columns.tolist())

    return nodes, feas, placements


def filter_feasible(nodes: pd.DataFrame,
                    feas: pd.DataFrame) -> pd.DataFrame:
    """Return only nodes that are feasible for the target amenity."""
    # Filter feasibility to this amenity and feasible == True/1
    feas_amen = feas[feas["amenity"] == AMENITY].copy()

    # GEE may store booleans as 0/1; normalize to bool
    if feas_amen["feasible"].dtype != bool:
        feas_amen["feasible"] = feas_amen["feasible"].astype(bool)

    feasible_only = feas_amen[feas_amen["feasible"]]

    print(f"{AMENITY}: {len(feasible_only)} feasible nodes")

    # Join with your node table (osmid == node_id)
    nodes = nodes.copy()
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


def prepare_placements(placements: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Subset placement polygons and standardize columns."""
    gdf = placements.copy()

    # Ensure proper CRS
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)

    # Keep only the relevant columns
    keep_cols = [
        "node_id",
        "amenity",
        "free_area_m2",
        "min_area_req",
        "distance_m",
        "patch_area_m2",
        "geometry",
    ]
    gdf = gdf[keep_cols]

    # Filter for current amenity
    gdf = gdf[gdf["amenity"] == AMENITY]

    print(f"{AMENITY}: {len(gdf)} placement polygons")

    return gdf


def main():
    nodes, feas, placements = load_data()

    feasible_only, merged = filter_feasible(nodes, feas)
    placements_clean = prepare_placements(placements)

    out_feas_nodes = BASE_DIR / f"gee_feasible_nodes_{AMENITY}.csv"
    out_merged = BASE_DIR / f"gee_feasible_nodes_{AMENITY}_merged.csv"
    out_placements = BASE_DIR / f"gee_placements_{AMENITY}.geojson"

    feasible_only.to_csv(out_feas_nodes, index=False)
    merged.to_csv(out_merged, index=False)
    placements_clean.to_file(out_placements, driver="GeoJSON")

    print(f"Saved feasible node list to: {out_feas_nodes}")
    print(f"Saved merged node+gee table to: {out_merged}")
    print(f"Saved placement polygons to: {out_placements}")

    # Optional: simple sanity check – show top few candidates
    print("\nTop 5 feasible nodes (merged):")
    cols_show = [
        "osmid", "lon", "lat",
        "free_area_m2", "min_area_req", "feasible",
        "walkability", "accessibility_score"
    ]
    cols_show = [c for c in cols_show if c in merged.columns]
    print(merged[cols_show].head())


if __name__ == "__main__":
    main()
