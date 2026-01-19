#!/usr/bin/env python3
"""
Optimized scoring pipeline that keeps POIs in memory to avoid redundant disk I/O.
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import osmnx as ox
import h3
import numpy as np
import json

# Add project root for CityDataManager
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from city_paths import CityDataManager

# Add data-pipeline to path for compute_scores functions
sys.path.insert(0, str(project_root / "data-pipeline"))

from compute_scores import (
    load_config,
    load_inputs,
    ensure_poi_identifiers,
    compute_centrality_metrics,
    compute_structure_metrics,
    compute_circuity_sample,
    nearest_amenity_distances,
    compute_accessibility_score,
    compute_travel_time_metrics,
    compute_equity_metrics,
    aggregate_h3,
    clean_for_parquet,
    normalize_series,
    save_to_cache,
    load_from_cache,
    compute_stratified_metrics,
)

def merge_pois_in_memory(baseline_pois: gpd.GeoDataFrame, optimized_pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge baseline and optimized POIs in memory without disk I/O."""
    print(f"[MEMORY MERGE] Merging {len(baseline_pois)} baseline + {len(optimized_pois)} optimized POIs...")
    
    if "source" not in baseline_pois.columns:
        baseline_pois = baseline_pois.copy()
        baseline_pois["source"] = "existing"
    
    if "source" not in optimized_pois.columns:
        optimized_pois = optimized_pois.copy()
        optimized_pois["source"] = "optimized"
    
    if optimized_pois.crs != baseline_pois.crs:
        optimized_pois = optimized_pois.to_crs(baseline_pois.crs)
    
    # Align columns
    for df in [baseline_pois, optimized_pois]:
        for col in df.columns:
            if col != 'geometry' and pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype('object')
    
    merged = gpd.GeoDataFrame(
        pd.concat([baseline_pois, optimized_pois], ignore_index=True, sort=False),
        crs=baseline_pois.crs
    )
    return merged

def main():
    parser = argparse.ArgumentParser(description="Optimized scoring pipeline with in-memory merge.")
    parser.add_argument("--city", default="bangalore", help="City to process")
    parser.add_argument("--mode", default="ga_only", choices=["ga_only", "ga_milp", "ga_milp_pnmlr"], help="Optimization mode")
    parser.add_argument("--force", action="store_true", help="Force recomputation")
    args = parser.parse_args()

    cdm = CityDataManager(args.city, project_root=project_root, mode=args.mode)
    cfg = cdm.load_config()
    
    print(f"Running optimized scoring for {args.city} ({args.mode})")

    # Resolve paths from CDM
    graph_path = cdm.processed_graph
    poi_mapping_path = cdm.poi_mapping
    baseline_pois_path = cdm.baseline_pois_parquet if cdm.baseline_pois_parquet.exists() else cdm.raw_pois
    optimized_pois_path = cdm.optimized_pois(args.mode)
    out_dir = cdm.optimized_dir(args.mode)
    
    # Configuration
    h3_resolution = cfg.get("h3", {}).get("resolution", 8)
    walking_speed = cfg.get("walking_speed_kmph", 4.8)
    amenity_weights = cfg.get("amenity_weights", {})
    composite_weights = cfg.get("composite_weights", {"alpha": 0.05, "beta": 0.80, "gamma": 0.05, "delta": 0.10})
    coverage_thresholds = cfg.get("equity_thresholds", {key: 800 for key in amenity_weights.keys()})
    centrality_cfg = cfg.get("centrality", {})
    amenity_cutoff = cfg.get("amenity_distance_cutoff_m")
    circuity_sample_k = int(cfg.get("circuity_sample_k", 120))

    cache_dir = project_root / "data" / "cache" / "scoring"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    print("Loading graph and mapping...")
    G, nodes, mapping = load_inputs(graph_path, poi_mapping_path)
    
    # Load and merge POIs
    print(f"Loading loading baseline POIs...")
    # Prefer parquet source if available (raw or baseline)
    raw_pois_parquet = cdm.raw_pois.with_suffix(".parquet")
    
    if baseline_pois_path.exists() and baseline_pois_path.suffix == '.parquet':
        load_path = baseline_pois_path
    elif raw_pois_parquet.exists():
        print(f"Found cached raw POIs at {raw_pois_parquet}")
        load_path = raw_pois_parquet
    else:
        load_path = cdm.raw_pois
        
    print(f"Loading baseline POIs from {load_path}...")
    if load_path.suffix == '.parquet':
        baseline_pois = gpd.read_parquet(load_path)
    else:
        baseline_pois = gpd.read_file(load_path)
        # Auto-cache to parquet for future runs
        try:
            cache_save_path = load_path.with_suffix(".parquet")
            print(f"Caching loaded POIs to {cache_save_path} for future speed...")
            
            # Simple normalization for list columns to ensure parquet compatibility
            for col in baseline_pois.columns:
                if col != 'geometry' and baseline_pois[col].apply(lambda x: isinstance(x, (list, tuple))).any():
                    baseline_pois[col] = baseline_pois[col].astype(str)
            
            baseline_pois.to_parquet(cache_save_path)
            print("Cache saved.")
        except Exception as e:
            print(f"Warning: Failed to save POI cache: {e}")

    print(f"Loading optimized POIs from {optimized_pois_path}...")
    optimized_pois = gpd.read_file(optimized_pois_path)
    
    pois = merge_pois_in_memory(baseline_pois, optimized_pois)
    
    if pois.crs != nodes.crs:
        pois = pois.to_crs(nodes.crs)
    pois = ensure_poi_identifiers(pois)
    
    # Snapping for new POIs
    baseline_count = len(baseline_pois)
    new_pois_slice = pois.iloc[baseline_count:]
    
    if not new_pois_slice.empty:
        # Filter invalid geometries
        valid_mask = ~new_pois_slice.geometry.is_empty & new_pois_slice.geometry.notna()
        # Additionally check for finite coordinates
        if not valid_mask.empty:
            valid_mask &= (np.isfinite(new_pois_slice.geometry.x) & np.isfinite(new_pois_slice.geometry.y))
            
        valid_pois = new_pois_slice[valid_mask]
        dropped_count = len(new_pois_slice) - len(valid_pois)
        
        if dropped_count > 0:
            print(f"Warning: Dropped {dropped_count} invalid geometries from optimization candidates.")

        if not valid_pois.empty:
            print(f"Snapping {len(valid_pois)} optimization candidates to graph...")
            X = valid_pois.geometry.x.values
            Y = valid_pois.geometry.y.values
            nearest_nodes = ox.nearest_nodes(G, X, Y)
            
            new_mapping = pd.DataFrame({"nearest_node": nearest_nodes}, index=valid_pois.index)
            if "poi_id" in valid_pois.columns:
                new_mapping["poi_id"] = valid_pois["poi_id"]
            elif "id" in valid_pois.columns:
                new_mapping["poi_id"] = valid_pois["id"]
            
            mapping = pd.concat([mapping, new_mapping])
            mapping.index.name = "poi_index"
        else:
            print("Warning: No valid optimization candidates remaining after filtering.")

    # Computation metrics
    amenity_types = list(amenity_weights.keys())
    
    print("Computing structure and centrality metrics...")
    nodes = compute_structure_metrics(G, nodes)
    nodes = compute_centrality_metrics(G, nodes, centrality_cfg)
    
    print("Computing nearest amenity distances...")
    distances = nearest_amenity_distances(G, nodes, mapping, pois, amenity_types, distance_cutoff=amenity_cutoff)
    nodes = nodes.join(distances)

    print("Computing scores...")
    nodes["accessibility_score"] = compute_accessibility_score(distances, amenity_weights)
    nodes["travel_time_min"] = compute_travel_time_metrics(distances, amenity_weights, walking_speed)
    
    # Clean travel times
    travel_time_clean = nodes["travel_time_min"].replace([np.inf, -np.inf], np.nan)
    fallback = float(travel_time_clean.dropna().max()) if not travel_time_clean.dropna().empty else 1.0
    nodes["travel_time_min"] = travel_time_clean.fillna(fallback)
    nodes["travel_time_score"] = 100.0 * (1.0 - nodes["travel_time_min"].clip(upper=60) / 60.0)

    # Walkability components
    structure_components = [
        normalize_series(nodes["degree"].fillna(0)),
        normalize_series(1.0 / (nodes["avg_incident_length_m"].fillna(1.0) + 1.0)),
    ]
    if "betweenness_centrality" in nodes.columns:
        structure_components.append(normalize_series(nodes["betweenness_centrality"].fillna(0)))
    
    nodes["structure_score"] = (sum(structure_components) / max(len(structure_components), 1)) * 100.0

    nodes_latlon = nodes.to_crs(epsg=4326) if nodes.crs.to_epsg() != 4326 else nodes
    nodes["h3_index"] = [h3.geo_to_h3(pt.y, pt.x, h3_resolution) for pt in nodes_latlon.geometry]
    
    equity_variance = nodes.groupby("h3_index")["accessibility_score"].transform("var").fillna(0)
    max_variance = equity_variance.max() if equity_variance.max() > 0 else 1.0
    nodes["equity_score"] = 100.0 * (1.0 - equity_variance / max_variance)

    alpha, beta, gamma, delta = [composite_weights.get(k, 0.25) for k in ['alpha', 'beta', 'gamma', 'delta']]
    nodes["walkability"] = (alpha * nodes["structure_score"] + beta * nodes["accessibility_score"] + 
                            gamma * nodes["equity_score"] + delta * nodes["travel_time_score"])

    print("Aggregating H3 results...")
    nodes, equity_agg = compute_equity_metrics(nodes, distances, h3_resolution, coverage_thresholds)
    agg = aggregate_h3(nodes, nodes["walkability"], h3_res=h3_resolution).merge(equity_agg, on="h3", how="left")

    # Summary
    # Summary using stratified metrics to capture equity impact
    print("Computing stratified metrics (Citywide / Underserved / Gap Closure)...")
    stratified = compute_stratified_metrics(
        nodes, 
        underserved_percentile=20.0,
        gap_threshold_minutes=15.0
    )
    
    # Calculate network metrics
    circuity = compute_circuity_sample(G, nodes, sample_k=circuity_sample_k)
    link_node_ratio = G.size() / max(G.number_of_nodes(), 1)
    
    # Build summary matching baseline structure for frontend compatibility
    summary = {
        "network": {
            "circuity_sample_ratio": float(circuity) if np.isfinite(circuity) else None,
            "intersection_density_global": float(nodes["intersection_density_global"].mean()),
            "link_node_ratio_global": float(link_node_ratio),
        },
        "scores": {
            "citywide": stratified["citywide"],
            "underserved": stratified["underserved"],
            "well_served": stratified["well_served"],
            "gap_closure": stratified["gap_closure"],
            "distribution": stratified["distribution"],
            "equity": float(nodes["equity_score"].mean()),
        }
    }

    # Save outputs
    prefix = f"{args.mode}_" if args.mode else ""
    nodes_clean = clean_for_parquet(nodes)
    nodes_clean.to_parquet(out_dir / f"{prefix}nodes_with_scores.parquet")
    agg.to_parquet(out_dir / f"{prefix}h3_agg.parquet")
    with (out_dir / f"{prefix}metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Scoring complete. Results in {out_dir}")

if __name__ == "__main__":
    main()
