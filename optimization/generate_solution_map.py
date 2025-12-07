#!/usr/bin/env python3
"""Generate a combined POI GeoJSON and map from the latest GA solution and recompute post-optimization metrics.

Usage: run from project root. The script:
 - reads optimization/runs/best_candidate.json
 - extracts placements and best_distances from the GA metrics if available
 - builds an optimized POI GeoDataFrame with coordinates & distance diagnostics
 - merges with existing POIs GeoJSON (if present) and writes combined GeoJSON
 - builds an interactive folium map (optimized placements highlighted)
 - (optionally) recomputes node-level optimized accessibility/travel metrics using the saved graph & config
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd

try:
    import folium
    from folium.plugins import MarkerCluster
except ImportError as exc:  # pragma: no cover - defensive
    raise SystemExit(
        "folium is required to build the interactive map. Install it with 'pip install folium'."
    ) from exc

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization.hybrid_ga import ensure_index_on_osmid

try:  # pragma: no cover - optional dependency
    import osmnx as ox
except ImportError:  # pragma: no cover - defensive
    ox = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - defensive
    tqdm = None  # type: ignore

from pipeline.scoring import (
    compute_accessibility_score,
    compute_travel_time_metrics,
    load_config,
    nearest_amenity_distances,
)

DEFAULT_CENTER: Tuple[float, float] = (12.9716, 77.5946)  # Bengaluru fallback


def iter_with_progress(iterable, desc: str, total: Optional[int] = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False)


def parse_candidate_signature(signature: str) -> Dict[str, Tuple[str, ...]]:
    placements: Dict[str, Tuple[str, ...]] = {}
    if not signature or signature.lower() == "baseline":
        return placements
    segments = signature.split("|")
    for segment in segments:
        if ":" not in segment:
            continue
        amenity, nodes_csv = segment.split(":", 1)
        node_ids = [node.strip() for node in nodes_csv.split(",") if node.strip()]
        if node_ids:
            placements[amenity] = tuple(node_ids)
    return placements


def load_best_candidate(best_candidate_path: Path) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, object]]:
    if not best_candidate_path.exists():
        raise FileNotFoundError(f"Best candidate file not found: {best_candidate_path}")
    with best_candidate_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    # older code used key "candidate" or "candidate.signature"; support both
    signature = payload.get("candidate", "") if isinstance(payload, dict) else ""
    placements = parse_candidate_signature(str(signature))
    metrics = payload.get("metrics", {}) if isinstance(payload, Mapping) else {}
    if not placements:
        logging.warning("Best candidate signature is empty; no placements to visualise.")
    return placements, dict(metrics)


def build_optimised_geodata(
    placements: Mapping[str, Sequence[str]],
    nodes: pd.DataFrame,
    metrics: Mapping[str, object],
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame for optimized placements.

    For each placed POI we include:
      - amenity, osmid, lon/lat (from nodes), travel_time_min (if present)
      - distance_m taken from metrics['best_distances'][amenity] if available (representative)
    """
    best_distances = metrics.get("best_distances", {}) if isinstance(metrics, Mapping) else {}
    distance_lookup: Dict[str, float] = {}
    if isinstance(best_distances, Mapping):
        for amenity, value in best_distances.items():
            try:
                distance_lookup[str(amenity)] = float(value)
            except (TypeError, ValueError):
                continue

    records = []
    placement_iter = iter_with_progress(placements.items(), "Collecting optimized POIs", total=len(placements))
    for amenity, node_ids in placement_iter:
        for node_id in node_ids:
            key = str(node_id)
            if key not in nodes.index:
                logging.warning("Node %s missing from nodes dataset; skipping.", key)
                continue
            row = nodes.loc[key]
            try:
                lon = float(row.get("lon"))
                lat = float(row.get("lat"))
            except (TypeError, ValueError):
                logging.warning("Node %s lacks valid coordinates; skipping.", key)
                continue
            record = {
                "amenity": amenity,
                "osmid": key,
                "lon": lon,
                "lat": lat,
                "travel_time_min": row.get("travel_time_min"),
                # use metric-level best_distances for that amenity if present; else None
                "distance_m": distance_lookup.get(amenity),
                "source": "optimized",
            }
            records.append(record)

    if not records:
        # empty GeoDataFrame with columns matching later code
        return gpd.GeoDataFrame(columns=["amenity", "osmid", "lon", "lat", "travel_time_min", "distance_m", "source"], geometry=[], crs="EPSG:4326")

    frame = pd.DataFrame.from_records(records)
    geometry = gpd.points_from_xy(frame["lon"], frame["lat"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")
    return gdf


def load_existing_pois(pois_path: Path) -> gpd.GeoDataFrame:
    if not pois_path.exists():
        logging.warning("Existing POI GeoJSON not found: %s", pois_path)
        return gpd.GeoDataFrame(columns=["source"], geometry=[], crs="EPSG:4326")
    gdf = gpd.read_file(pois_path)
    try:
        gdf = gdf.to_crs("EPSG:4326")
    except Exception:  # pragma: no cover - defensive
        logging.debug("Existing POIs already in EPSG:4326 or CRS conversion failed; proceeding as-is.")
    if "source" not in gdf.columns:
        gdf["source"] = "existing"
    else:
        gdf["source"] = gdf["source"].fillna("existing")
    return gdf


def export_combined_geojson(
    existing_gdf: gpd.GeoDataFrame,
    optimized_gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> gpd.GeoDataFrame:
    if existing_gdf.empty and optimized_gdf.empty:
        logging.warning("No POIs available to export; skipping GeoJSON creation.")
        return gpd.GeoDataFrame(columns=["source"], geometry=[], crs="EPSG:4326")
    components = []
    if not existing_gdf.empty:
        components.append(existing_gdf)
    if not optimized_gdf.empty:
        components.append(optimized_gdf)
    combined = gpd.GeoDataFrame(pd.concat(components, ignore_index=True), crs="EPSG:4326")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_file(output_path, driver="GeoJSON")
    logging.info("Combined GeoJSON written to %s", output_path)
    return combined


def compute_map_center(*gdfs: gpd.GeoDataFrame) -> Tuple[float, float]:
    lats: List[float] = []
    lons: List[float] = []
    for gdf in gdfs:
        if gdf is None or gdf.empty:
            continue
        lats.extend(gdf.geometry.y.astype(float).tolist())
        lons.extend(gdf.geometry.x.astype(float).tolist())
    if lats and lons:
        return (float(sum(lats) / len(lats)), float(sum(lons) / len(lons)))
    return DEFAULT_CENTER


def resolve_node_identifier(node_view, node_id: str) -> object:
    """Try string id, then numeric conversion — used when matching optimized POIs to the graph."""
    if node_id in node_view:
        return node_id
    try:
        numeric_id = int(float(node_id))
        if numeric_id in node_view:
            return numeric_id
    except (TypeError, ValueError):
        pass
    return node_id


def build_map(
    existing_gdf: gpd.GeoDataFrame,
    optimized_gdf: gpd.GeoDataFrame,
    output_html: Path,
) -> None:
    center_lat, center_lon = compute_map_center(existing_gdf, optimized_gdf)
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")

    if not existing_gdf.empty:
        existing_group = folium.FeatureGroup(name="Existing POIs", show=False)
        cluster = MarkerCluster(name="Existing POIs")
        existing_iter = iter_with_progress(existing_gdf.iterrows(), "Existing POIs", total=len(existing_gdf))
        for _, row in existing_iter:
            geometry = row.geometry
            if geometry is None:
                continue
            folium.CircleMarker(
                location=[geometry.y, geometry.x],
                radius=3,
                color="#3388ff",
                opacity=0.8,
                fill=True,
                fill_opacity=0.5,
            ).add_to(cluster)
        cluster.add_to(existing_group)
        existing_group.add_to(fmap)

    if not optimized_gdf.empty:
        optimized_group = folium.FeatureGroup(name="Optimized Placements", show=True)
        optimized_iter = iter_with_progress(optimized_gdf.iterrows(), "Optimized placements", total=len(optimized_gdf))
        for _, row in optimized_iter:
            geometry = row.geometry
            if geometry is None:
                continue
            popup_lines = [f"Amenity: {row.get('amenity', '-')}", f"OSM id: {row.get('osmid', '-')}" ]
            distance_value = row.get("distance_m")
            if pd.notna(distance_value):
                popup_lines.append(f"Representative distance: {float(distance_value):.1f} m")
            travel_time = row.get("travel_time_min")
            if pd.notna(travel_time):
                popup_lines.append(f"Travel time: {float(travel_time):.1f} min")
            popup = "<br/>".join(popup_lines)
            folium.CircleMarker(
                location=[geometry.y, geometry.x],
                radius=6,
                color="#d95f02",
                weight=2,
                opacity=0.9,
                fill=True,
                fill_color="#d95f02",
                fill_opacity=0.7,
                popup=popup,
            ).add_to(optimized_group)
        optimized_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output_html)
    logging.info("Interactive map written to %s", output_html)


def compute_optimized_scores(
    placements: Mapping[str, Sequence[str]],
    optimized_gdf: gpd.GeoDataFrame,
    nodes_df: pd.DataFrame,
    graph_path: Optional[Path],
    config_path: Optional[Path],
    nodes_output: Path,
    summary_output: Path,
) -> None:
    """Recompute optimized accessibility and travel metrics using the road graph and config.

    Writes:
      - nodes_output (parquet) with optimized_* columns
      - nodes_output.csv
      - summary_output (json) with aggregated optimized metrics
    """
    if graph_path is None:
        logging.info("No graph path provided; skipping optimized score computation.")
        return
    if config_path is None:
        logging.info("No configuration file provided; skipping optimized score computation.")
        return
    if ox is None:
        logging.warning("osmnx is unavailable; cannot compute optimized scores.")
        return
    if not graph_path.exists():
        logging.warning("Graph file %s not found; skipping optimized score computation.", graph_path)
        return

    non_empty_placements = {amenity: tuple(nodes) for amenity, nodes in placements.items() if nodes}
    if not non_empty_placements:
        logging.warning("No optimized placements detected; skipping optimized score computation.")
        return

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logging.warning("Configuration file %s not found; skipping optimized score computation.", config_path)
        return
    amenity_weights = config.get("amenity_weights", {})
    if not amenity_weights:
        logging.warning("Configuration lacks amenity_weights; skipping optimized score computation.")
        return
    walking_speed = float(config.get("walking_speed_kmph", 4.8))
    composite_weights = config.get("composite_weights", {"alpha": 0.25, "beta": 0.4, "gamma": 0.2, "delta": 0.15})
    amenity_cutoff = config.get("amenity_distance_cutoff_m")

    # Keep only amenity types that both appear in placements and have configured weights
    amenity_types = sorted([amenity for amenity in non_empty_placements.keys() if amenity in amenity_weights])
    if not amenity_types:
        logging.warning("No optimized amenities align with configured weights; skipping optimized score computation.")
        return

    logging.info("Recomputing optimized accessibility metrics for %d amenity types.", len(amenity_types))
    G = ox.load_graphml(graph_path)
    nodes_gdf_result = ox.graph_to_gdfs(G, nodes=True, edges=False)
    nodes_gdf = nodes_gdf_result[0] if isinstance(nodes_gdf_result, tuple) else nodes_gdf_result

    valid_indices: List[int] = []
    mapping_records: List[Dict[str, object]] = []
    score_iter = iter_with_progress(optimized_gdf.iterrows(), "Validating placements", total=len(optimized_gdf))
    for idx, row in score_iter:
        amenity = str(row.get("amenity", "amenity"))
        node_id_raw = row.get("osmid")
        if node_id_raw is None:
            continue
        node_id_str = str(node_id_raw)
        resolved_node = resolve_node_identifier(G.nodes, node_id_str)
        if resolved_node not in G:
            logging.warning("Optimized placement node %s missing from graph; skipping.", node_id_str)
            continue
        mapping_records.append(
            {
                "poi_index": f"{amenity}_{node_id_str}",
                "nearest_node": resolved_node,
                "poi_id": f"{amenity}_{node_id_str}",
            }
        )
        valid_indices.append(idx)

    if not mapping_records:
        logging.warning("No optimized placements align with the graph; skipping optimized score computation.")
        return

    filtered_gdf = optimized_gdf.loc[valid_indices].copy()
    for col in ("shop", "leisure"):
        if col not in filtered_gdf.columns:
            filtered_gdf[col] = None

    mapping_df = pd.DataFrame(mapping_records).set_index("poi_index")

    # compute nearest distances (returns DataFrame indexed by node id)
    distances = nearest_amenity_distances(
        G,
        nodes_gdf,
        mapping_df,
        filtered_gdf,
        amenity_types,
        distance_cutoff=amenity_cutoff,
    )
    distances.index = distances.index.map(str)

    nodes_aligned = nodes_df.copy()
    nodes_aligned.index = nodes_aligned.index.map(str)
    distances = distances.reindex(nodes_aligned.index)

    optimized_accessibility = compute_accessibility_score(distances, amenity_weights)
    optimized_travel_time = compute_travel_time_metrics(distances, walking_speed)
    travel_time_clean = optimized_travel_time.replace([np.inf, -np.inf], np.nan)
    fallback = travel_time_clean.max()
    if pd.isna(fallback) or fallback <= 0:
        fallback = 1.0
    optimized_travel_score = 1.0 / (1.0 + travel_time_clean.fillna(fallback))

    structure_component = nodes_aligned.get("structure_score", pd.Series(0.0, index=nodes_aligned.index)).fillna(0.0)
    equity_component = nodes_aligned.get("equity_score", pd.Series(0.0, index=nodes_aligned.index)).fillna(0.0)

    alpha = float(composite_weights.get("alpha", 0.25))
    beta = float(composite_weights.get("beta", 0.4))
    gamma = float(composite_weights.get("gamma", 0.2))
    delta = float(composite_weights.get("delta", 0.15))

    optimized_walkability = (
        alpha * structure_component
        + beta * optimized_accessibility.fillna(0.0)
        + gamma * equity_component
        + delta * optimized_travel_score.fillna(0.0)
    )

    nodes_result = nodes_aligned.copy()
    for column in distances.columns:
        nodes_result[column] = distances[column]
    nodes_result["optimized_accessibility_score"] = optimized_accessibility
    nodes_result["optimized_travel_time_min"] = optimized_travel_time
    nodes_result["optimized_travel_time_score"] = optimized_travel_score
    nodes_result["optimized_walkability"] = optimized_walkability

    nodes_output.parent.mkdir(parents=True, exist_ok=True)
    nodes_result.to_parquet(nodes_output)
    csv_output = nodes_output.with_suffix(".csv")
    nodes_result.to_csv(csv_output, index=True)

    summary = {
        "optimized": {
            "accessibility_mean": float(optimized_accessibility.replace([np.inf, -np.inf], np.nan).mean(skipna=True)),
            "travel_time_min_mean": float(travel_time_clean.mean(skipna=True)),
            "travel_time_score_mean": float(optimized_travel_score.mean(skipna=True)),
            "walkability_mean": float(optimized_walkability.mean(skipna=True)),
        }
    }

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info(
        "Optimized scores saved to %s, %s, and %s",
        nodes_output,
        csv_output,
        summary_output,
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create GeoJSON and map for the optimised GA solution.")
    parser.add_argument(
        "--best-candidate",
        type=Path,
        default=project_root / "optimization" / "runs" / "best_candidate.json",
        help="Path to the best_candidate.json output from the GA run.",
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=project_root / "data" / "analysis" / "nodes_with_scores.parquet",
        help="Path to the nodes parquet containing coordinates.",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=project_root / "data" / "processed" / "graph.graphml",
        help="GraphML file used for recomputing accessibility scores.",
    )
    parser.add_argument(
        "--existing-pois",
        type=Path,
        default=project_root / "data" / "raw" / "pois.geojson",
        help="GeoJSON containing existing POIs to overlay.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config.yaml",
        help="Configuration file supplying amenity weights and scoring parameters.",
    )
    parser.add_argument(
        "--output-geojson",
        type=Path,
        default=project_root / "optimization" / "runs" / "poi_mapping.geojson",
        help="Destination for the combined POI GeoJSON.",
    )
    parser.add_argument(
        "--output-map",
        type=Path,
        default=project_root / "optimization" / "runs" / "optimized_map.html",
        help="Destination for the interactive HTML map.",
    )
    parser.add_argument(
        "--optimized-nodes-output",
        type=Path,
        default=project_root / "optimization" / "runs" / "optimized_nodes_with_scores.parquet",
        help="Path to write node-level optimized scores (parquet).",
    )
    parser.add_argument(
        "--optimized-summary-output",
        type=Path,
        default=project_root / "optimization" / "runs" / "optimized_metrics_summary.json",
        help="Path to write optimized score summary (JSON).",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        default=False,
        help="Skip recomputing post-optimization metrics (fast: only map + GeoJSON).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    placements, metrics = load_best_candidate(args.best_candidate)
    if not placements:
        logging.error("No placements extracted from best candidate; aborting map generation.")
        return

    nodes_df = ensure_index_on_osmid(pd.read_parquet(args.nodes))
    optimized_gdf = build_optimised_geodata(placements, nodes_df, metrics)

    existing_gdf = load_existing_pois(args.existing_pois)
    combined_gdf = export_combined_geojson(existing_gdf, optimized_gdf, args.output_geojson)

    # Use the combined GeoDataFrame for centering if export succeeded.
    if not combined_gdf.empty:
        existing_view = combined_gdf[combined_gdf["source"] == "existing"]
        optimized_view = combined_gdf[combined_gdf["source"] == "optimized"]
    else:
        existing_view = existing_gdf
        optimized_view = optimized_gdf

    build_map(existing_view, optimized_view, args.output_map)

    if not args.skip_metrics:
        compute_optimized_scores(
            placements,
            optimized_gdf,
            nodes_df,
            args.graph_path,
            args.config,
            args.optimized_nodes_output,
            args.optimized_summary_output,
        )
    else:
        logging.info("Skipping post-optimization metric recomputation (--skip-metrics).")


if __name__ == "__main__":
    main()
