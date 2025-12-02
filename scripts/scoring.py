#!/usr/bin/env python3
"""
Compute node-level structure metrics, amenity accessibility, travel-time metrics,
and aggregate to H3 hex grid for equity metrics and visualization.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import h3
import osmnx as ox
import yaml


ox.settings.use_cache = True


def load_config(config_path: Path | None) -> Dict:
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_inputs(graph_path: Path, poi_mapping_path: Path):
    G = ox.load_graphml(graph_path)
    nodes_result = ox.graph_to_gdfs(G, nodes=True, edges=False)
    # OSMnx >= 2.0 returns a GeoDataFrame when only nodes=True; older versions return (nodes, edges)
    if isinstance(nodes_result, tuple):
        nodes = nodes_result[0]
    else:
        nodes = nodes_result
    mapping = pd.read_parquet(poi_mapping_path)
    if "nearest_node" not in mapping.columns:
        raise ValueError("POI mapping parquet must include a 'nearest_node' column.")
    return G, nodes, mapping


def ensure_poi_identifiers(pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    pois_prepped = pois.copy()
    if "poi_id" in pois_prepped.columns:
        pois_prepped["poi_id"] = pois_prepped["poi_id"].astype(str)
        return pois_prepped

    if "osmid" in pois_prepped.columns:
        pois_prepped["poi_id"] = pois_prepped["osmid"].apply(
            lambda value: str(int(value)) if pd.notna(value) else None
        )
    else:
        pois_prepped["poi_id"] = None

    missing_ids = pois_prepped["poi_id"].isna()
    if missing_ids.any():
        pois_prepped.loc[missing_ids, "poi_id"] = pois_prepped.index[missing_ids].astype(str)
    return pois_prepped


def compute_structure_metrics(G, nodes_gdf):
    # Intersection density: number of intersections per km^2 (node degree >= 3)
    nodes_gdf = nodes_gdf.copy()
    degree_series = pd.Series(dict(G.degree()))
    nodes_gdf["degree"] = nodes_gdf.index.map(degree_series).fillna(0)
    nodes_gdf["is_intersection"] = nodes_gdf["degree"] >= 3
    # approximate area: bounding box area (fast), but better to use official administrative polygon
    bbox = nodes_gdf.total_bounds  # minx, miny, maxx, maxy
    area_km2 = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / 1e6  # rough if coords in meters; in degrees this is rough.
    # local structure metrics (node-level): number of incident edges, average incident length
    avg_incident_length = []
    for node in nodes_gdf.index:
        incident_edges = list(G.edges(node, data=True))
        lengths = [d.get("length", 0) for _, _, d in incident_edges]
        avg_incident_length.append(np.mean(lengths) if lengths else 0)
    nodes_gdf["avg_incident_length_m"] = avg_incident_length
    nodes_gdf["intersection_density_global"] = nodes_gdf["is_intersection"].sum() / max(area_km2, 1e-6)
    # link-node ratio: |E|/|V| (global value)
    link_node_ratio = G.size() / max(G.number_of_nodes(), 1)
    nodes_gdf["link_node_ratio_global"] = link_node_ratio
    return nodes_gdf


def compute_centrality_metrics(
    G: nx.MultiDiGraph,
    nodes_gdf: gpd.GeoDataFrame,
    centrality_cfg: Dict[str, object],
) -> gpd.GeoDataFrame:
    nodes = nodes_gdf.copy()
    if not centrality_cfg.get("enabled", True):
        nodes["betweenness_centrality"] = 0.0
        nodes["closeness_centrality"] = 0.0
        return nodes

    sample_default = 150
    sample_limit = int(centrality_cfg.get("sample_k", sample_default))
    compute_betweenness = centrality_cfg.get("compute_betweenness", True)
    compute_closeness = centrality_cfg.get("compute_closeness", False)
    rng_seed = centrality_cfg.get("seed", 42)

    if compute_betweenness:
        sample_nodes = min(len(G), max(sample_limit, 1))
        betweenness = nx.betweenness_centrality(
            G,
            k=sample_nodes if sample_nodes < len(G) else None,
            normalized=True,
            weight="length",
            seed=rng_seed,
        )
        betweenness_series = pd.Series(betweenness)
        nodes["betweenness_centrality"] = nodes.index.map(betweenness_series).fillna(0.0)
    else:
        nodes["betweenness_centrality"] = 0.0

    if compute_closeness:
        closeness = nx.closeness_centrality(G, distance="length")
        closeness_series = pd.Series(closeness)
        nodes["closeness_centrality"] = nodes.index.map(closeness_series).fillna(0.0)
    else:
        nodes["closeness_centrality"] = 0.0

    return nodes


def nearest_amenity_distances(
    G,
    nodes_gdf,
    poi_mapping: pd.DataFrame,
    pois_gdf: gpd.GeoDataFrame,
    amenity_types: List[str],
    distance_cutoff: float | None = None,
):
    """
    For each node, compute shortest-network distance to nearest amenity of each type.
    poi_mapping: mapping DataFrame linking POI indices to nearest_node
    pois_gdf: original pois GeoDataFrame
    """
    mapping_reset = poi_mapping.reset_index()
    index_to_nodes: Dict[object, set] = defaultdict(set)
    id_to_nodes: Dict[str, set] = defaultdict(set)

    has_poi_id = "poi_id" in mapping_reset.columns
    if has_poi_id:
        mapping_reset["poi_id"] = mapping_reset["poi_id"].astype(str)

    for row in mapping_reset.itertuples(index=False):
        poi_index = getattr(row, "poi_index")
        nearest_node = getattr(row, "nearest_node")
        index_to_nodes[poi_index].add(nearest_node)
        if has_poi_id:
            poi_id_val = getattr(row, "poi_id")
            if poi_id_val is not None and not (isinstance(poi_id_val, float) and np.isnan(poi_id_val)):
                id_to_nodes[str(poi_id_val)].add(nearest_node)

    # Build per-amenity node sets
    amenity_node_sets: Dict[str, set] = {}
    if pois_gdf is None or pois_gdf.empty or "amenity" not in pois_gdf.columns:
        pois_subset = None
    else:
        pois_subset = pois_gdf

    for t in amenity_types:
        if pois_subset is None:
            amenity_node_sets[t] = set()
            continue
        sel = pois_subset[pois_subset["amenity"] == t]
        if sel.empty:
            amenity_node_sets[t] = set()
            continue

        candidate_nodes: set = set()
        if "poi_id" in sel.columns:
            for poi_id_val in sel["poi_id"].astype(str):
                candidate_nodes.update(id_to_nodes.get(poi_id_val, set()))

        if not candidate_nodes:
            for idx in sel.index:
                candidate_nodes.update(index_to_nodes.get(idx, set()))

        amenity_node_sets[t] = candidate_nodes
    # For each node, compute the distance to closest node in each set using multi-source Dijkstra
    # For performance: run for each amenity type a multi-source dijkstra distances to all nodes
    distances = pd.DataFrame(index=nodes_gdf.index)
    for t, node_set in amenity_node_sets.items():
        if not node_set:
            distances[f"dist_to_{t}"] = np.nan
            continue
        length_attr = "length"
        # compute single-source-to-all by multi-source: use nx.multi_source_dijkstra_path_length on length
        dists = nx.multi_source_dijkstra_path_length(
            G,
            sources=list(node_set),
            weight=length_attr,
            cutoff=distance_cutoff,
        )
        # Fill distances for all nodes
        dist_series = pd.Series({n: dists.get(n, np.inf) for n in nodes_gdf.index})
        distances[f"dist_to_{t}"] = dist_series
    return distances


def compute_accessibility_score(distances_df: pd.DataFrame, amenity_weights: Dict[str, float]):
    acc = pd.Series(0.0, index=distances_df.index)
    for amenity, weight in amenity_weights.items():
        col = f"dist_to_{amenity}"
        if col not in distances_df.columns:
            continue
        # Accessibility formula: sum(weight / (distance + 1))
        acc += weight / (distances_df[col].fillna(np.inf) + 1.0)
    return acc


def compute_travel_time_metrics(distances_df: pd.DataFrame, walking_speed_kmph: float) -> pd.Series:
    walking_speed_mps = walking_speed_kmph * 1000 / 3600.0
    if walking_speed_mps <= 0:
        walking_speed_mps = 1.0
    times = []
    for col in distances_df.columns:
        time_minutes = distances_df[col] / walking_speed_mps / 60.0
        times.append(time_minutes)
    if not times:
        return pd.Series(0.0, index=distances_df.index)
    stacked = pd.concat(times, axis=1)
    stacked.replace([np.inf, -np.inf], np.nan, inplace=True)
    return stacked.min(axis=1)


def aggregate_h3(nodes_gdf: gpd.GeoDataFrame, value_series: pd.Series, h3_res=8):
    # ensure coordinates are in lat/lon before computing H3 indexes
    if nodes_gdf.crs is not None and nodes_gdf.crs.to_epsg() != 4326:
        nodes_latlon = nodes_gdf.to_crs(epsg=4326)
    else:
        nodes_latlon = nodes_gdf

    coords = list(zip(nodes_latlon.geometry.y, nodes_latlon.geometry.x))
    h3_indexes = [h3.geo_to_h3(lat, lng, h3_res) for lat, lng in coords]
    agg = pd.DataFrame({"h3": h3_indexes, "value": value_series.reindex(nodes_gdf.index).values})
    agg = agg.groupby("h3").agg({"value": ["mean", "std", "count"]})
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    return agg


def compute_circuity_sample(G, nodes_gdf, sample_k=200):
    # sample node pairs and compute (network_dist / euclidean_dist)
    node_list = list(nodes_gdf.index)
    rng = np.random.default_rng(0)
    pairs = rng.choice(node_list, size=(sample_k, 2), replace=True)
    ratios = []
    for a, b in pairs:
        if a == b:
            continue
        try:
            network_length = nx.shortest_path_length(G, a, b, weight="length")
            pt_a = nodes_gdf.loc[a].geometry
            pt_b = nodes_gdf.loc[b].geometry
            eucl = pt_a.distance(pt_b)
            if eucl > 0:
                ratios.append(network_length / eucl)
        except Exception:
            continue
    return np.nanmean(ratios) if ratios else np.nan


def compute_equity_metrics(nodes: gpd.GeoDataFrame, distances_df: pd.DataFrame, h3_res: int, coverage_thresholds: Dict[str, float]):
    nodes_latlon = nodes.to_crs(epsg=4326) if nodes.crs and nodes.crs.to_epsg() != 4326 else nodes
    h3_indexes = [h3.geo_to_h3(pt.y, pt.x, h3_res) for pt in nodes_latlon.geometry]
    nodes_with_h3 = nodes.copy()
    nodes_with_h3["h3_index"] = h3_indexes

    for amenity, threshold in coverage_thresholds.items():
        col_name = f"dist_to_{amenity}"
        if col_name not in distances_df.columns:
            continue
        coverage = (distances_df[col_name] <= threshold).astype(float)
        nodes_with_h3[f"coverage_{amenity}"] = coverage

    group = nodes_with_h3.groupby("h3_index")
    coverage_columns = [col for col in nodes_with_h3.columns if col.startswith("coverage_")]
    coverage_stats = group[coverage_columns].mean() if coverage_columns else pd.DataFrame(index=group.size().index)
    walkability_variance = group["walkability"].var().rename("walkability_variance")
    accessibility_mean = group["accessibility_score"].mean().rename("accessibility_mean")
    landuse_series = nodes_with_h3.get("nearest_landuse_type")
    if landuse_series is not None:
        pop_weights = np.where(landuse_series == "residential", 1.5, 1.0)
    else:
        pop_weights = np.ones(len(nodes_with_h3))
    nodes_with_h3["population_weight"] = pop_weights
    accessibility_weighted = (
        group.apply(lambda df: np.average(df["accessibility_score"], weights=df["population_weight"]) if not df.empty else np.nan)
    ).rename("population_weighted_accessibility")

    equity = pd.concat([coverage_stats, walkability_variance, accessibility_mean, accessibility_weighted], axis=1).reset_index().rename(columns={"h3_index": "h3"})
    return nodes_with_h3, equity


def normalize_series(series: pd.Series) -> pd.Series:
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index)
    min_val = valid.min()
    max_val = valid.max()
    if np.isclose(max_val, min_val):
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-path", required=True)
    p.add_argument("--poi-mapping", required=True)
    p.add_argument("--pois-path", required=False)
    p.add_argument("--out-dir", default="data/analysis")
    p.add_argument("--h3-res", type=int, default=None)
    p.add_argument("--config", default="../config.yaml")
    args = p.parse_args()

    config = load_config(Path(args.config))
    h3_resolution = args.h3_res or config.get("h3", {}).get("resolution", 8)
    walking_speed = config.get("walking_speed_kmph", 4.8)
    amenity_weights = config.get("amenity_weights", {})
    composite_weights = config.get("composite_weights", {"alpha": 0.25, "beta": 0.4, "gamma": 0.2, "delta": 0.15})
    coverage_thresholds = config.get("equity_thresholds", {key: 800 for key in amenity_weights.keys()})
    centrality_cfg = config.get("centrality", {})
    amenity_cutoff = config.get("amenity_distance_cutoff_m")
    circuity_sample_k = int(config.get("circuity_sample_k", 120))

    G, nodes, mapping = load_inputs(Path(args.graph_path), Path(args.poi_mapping))
    pois = None
    if args.pois_path:
        pois = gpd.read_file(args.pois_path)
        if pois.crs is not None and nodes.crs is not None and pois.crs != nodes.crs:
            pois = pois.to_crs(nodes.crs)
        pois = ensure_poi_identifiers(pois)

    nodes = compute_structure_metrics(G, nodes)
    nodes = compute_centrality_metrics(G, nodes, centrality_cfg)

    if not amenity_weights:
        raise ValueError("Amenity weights must be provided in the configuration file.")
    amenity_types = list(amenity_weights.keys())

    if pois is None:
        print("No POI dataset supplied; amenity distances will be NaN.")
        pois = gpd.GeoDataFrame(columns=["poi_id"], geometry=gpd.GeoSeries([], crs=nodes.crs), crs=nodes.crs)

    distances = nearest_amenity_distances(G, nodes, mapping, pois, amenity_types, distance_cutoff=amenity_cutoff)
    nodes = nodes.join(distances)

    nodes["accessibility_score"] = compute_accessibility_score(distances, amenity_weights)
    nodes["travel_time_min"] = compute_travel_time_metrics(distances, walking_speed)
    travel_time_clean = nodes["travel_time_min"].replace([np.inf, -np.inf], np.nan)
    fallback = travel_time_clean.max() if not travel_time_clean.dropna().empty else 1.0
    nodes["travel_time_score"] = 1.0 / (1.0 + travel_time_clean.fillna(fallback))

    structure_components = [
        normalize_series(nodes["degree"].fillna(0)),
        normalize_series(1.0 / (nodes["avg_incident_length_m"].fillna(1.0) + 1.0)),
    ]
    if centrality_cfg.get("compute_betweenness", True) and "betweenness_centrality" in nodes.columns:
        structure_components.append(normalize_series(nodes["betweenness_centrality"].fillna(0)))
    if centrality_cfg.get("compute_closeness", False) and "closeness_centrality" in nodes.columns:
        structure_components.append(normalize_series(nodes["closeness_centrality"].fillna(0)))
    nodes["structure_score"] = sum(structure_components) / max(len(structure_components), 1)

    nodes_latlon = nodes.to_crs(epsg=4326) if nodes.crs and nodes.crs.to_epsg() != 4326 else nodes
    nodes["h3_index"] = [h3.geo_to_h3(pt.y, pt.x, h3_resolution) for pt in nodes_latlon.geometry]
    equity_variance = nodes.groupby("h3_index")["accessibility_score"].transform("var").fillna(0)
    nodes["equity_score"] = 1.0 / (1.0 + equity_variance)

    alpha = composite_weights.get("alpha", 0.25)
    beta = composite_weights.get("beta", 0.4)
    gamma = composite_weights.get("gamma", 0.2)
    delta = composite_weights.get("delta", 0.15)

    nodes["walkability"] = (
        alpha * nodes["structure_score"]
        + beta * nodes["accessibility_score"]
        + gamma * nodes["equity_score"]
        + delta * nodes["travel_time_score"]
    )

    nodes, equity_agg = compute_equity_metrics(nodes, distances, h3_resolution, coverage_thresholds)
    agg_walkability = aggregate_h3(nodes, nodes["walkability"], h3_res=h3_resolution)
    agg = agg_walkability.merge(equity_agg, on="h3", how="left")

    circuity_value = compute_circuity_sample(G, nodes, sample_k=circuity_sample_k)
    summary = {
        "network": {
            "circuity_sample_ratio": float(circuity_value) if circuity_value == circuity_value else None,
            "intersection_density_global": float(nodes["intersection_density_global"].mean()),
            "link_node_ratio_global": float(nodes["link_node_ratio_global"].mean()),
        },
        "scores": {
            "accessibility_mean": float(nodes["accessibility_score"].mean()),
            "walkability_mean": float(nodes["walkability"].mean()),
            "equity_mean": float(nodes["equity_score"].mean()),
            "travel_time_min_mean": float(nodes["travel_time_min"].replace([np.inf, -np.inf], np.nan).mean()),
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(out_dir / "nodes_with_scores.parquet")
    agg.to_parquet(out_dir / "h3_agg.parquet")
    nodes.to_csv(out_dir / "nodes_with_scores.csv", index=True)
    agg.to_csv(out_dir / "h3_agg.csv", index=False)
    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved node-level scores and H3 aggregates to {out_dir}")


if __name__ == "__main__":
    main()