#!/usr/bin/env python3
"""
Compute node-level structure metrics, amenity accessibility, travel-time metrics,
and aggregate to H3 hex grid for equity metrics and visualization.
"""
import argparse
import json
import hashlib
import pickle
import sys
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
from tqdm import tqdm


ox.settings.use_cache = True

# Detect if running in a subprocess (no TTY) and disable tqdm
_DISABLE_TQDM = not sys.stdout.isatty()


def _progress_wrapper(iterable, desc="Processing", unit="item", total=None):
    """Wrapper for tqdm that falls back to simple logging when in subprocess."""
    if _DISABLE_TQDM:
        # Simple progress logging without tqdm
        # Try to get length without converting to list
        try:
            total = total or len(iterable)
            print(f"{desc}: {total} {unit}s")
            sys.stdout.flush()
            report_interval = max(1, total // 10)
            for i, item in enumerate(iterable, 1):
                if i % report_interval == 0 or i == total:
                    print(f"{desc}: {i}/{total} ({100*i//total}%)")
                    sys.stdout.flush()
                yield item
        except TypeError:
            # Iterable doesn't have length, just iterate
            print(f"{desc}: processing {unit}s...")
            sys.stdout.flush()
            for item in iterable:
                yield item
    else:
        yield from tqdm(iterable, desc=desc, unit=unit, total=total)


def get_cache_key(*args) -> str:
    """Generate cache key from arguments."""
    content = str(args).encode('utf-8')
    return hashlib.md5(content).hexdigest()


def load_from_cache(cache_path: Path) -> object:
    """Load cached data if exists."""
    if cache_path.exists():
        print(f"[CACHE] Loading from cache: {cache_path.name}")
        sys.stdout.flush()
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


def save_to_cache(data: object, cache_path: Path):
    """Save data to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[CACHE] Saved to cache: {cache_path.name}")
    sys.stdout.flush()


def safe_numeric(value, fallback: float = 0.0) -> float:
    """Convert value to a finite float, substituting fallback when NaN/inf/None."""
    if value is None:
        return float(fallback)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if np.isfinite(numeric):
        return numeric
    return float(fallback)


def load_config(config_path: Path | None) -> Dict:
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_inputs(graph_path: Path, poi_mapping_path: Path):
    G = ox.load_graphml(graph_path)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    sys.stdout.flush()
    nodes_result = ox.graph_to_gdfs(G, nodes=True, edges=False)
    # OSMnx >= 2.0 returns a GeoDataFrame when only nodes=True; older versions return (nodes, edges)
    if isinstance(nodes_result, tuple):
        nodes = nodes_result[0]
    else:
        nodes = nodes_result
    print(f"Extracted {len(nodes)} nodes as GeoDataFrame")
    sys.stdout.flush()
    # Add lon/lat columns for compatibility with landuse feasibility pipeline
    # Handle both WGS84 (EPSG:4326) and projected CRS (e.g., UTM)
    if 'lon' not in nodes.columns or 'lat' not in nodes.columns:
        # Check if x/y values are in projected CRS (UTM) by checking magnitude
        # Valid lat/lon values: lat in [-90, 90], lon in [-180, 180]
        # UTM values are typically in hundreds of thousands (meters)
        x_sample = nodes['x'].iloc[0] if 'x' in nodes.columns else nodes.geometry.x.iloc[0]
        y_sample = nodes['y'].iloc[0] if 'y' in nodes.columns else nodes.geometry.y.iloc[0]
        is_projected = abs(x_sample) > 180 or abs(y_sample) > 180
        
        if is_projected:
            # Coordinates are in projected CRS (e.g., UTM) - need to convert to WGS84
            # Determine the source CRS
            if nodes.crs is not None:
                source_crs = nodes.crs
            else:
                # Try to get CRS from graph attribute
                graph_crs = G.graph.get('crs')
                if graph_crs:
                    source_crs = graph_crs
                else:
                    # Default to UTM zone 43N for Bangalore area
                    source_crs = "EPSG:32643"
                    print(f"Warning: No CRS found, assuming {source_crs} for Bangalore")
                    sys.stdout.flush()
                nodes = nodes.set_crs(source_crs, allow_override=True)
            
            # Convert to WGS84
            nodes_wgs84 = nodes.to_crs(epsg=4326)
            nodes['lon'] = nodes_wgs84.geometry.x
            nodes['lat'] = nodes_wgs84.geometry.y
            print(f"Converted coordinates from {source_crs} to WGS84 (lat/lon)")
            print(f"Sample: x={x_sample:.2f}, y={y_sample:.2f} -> lon={nodes['lon'].iloc[0]:.6f}, lat={nodes['lat'].iloc[0]:.6f}")
            sys.stdout.flush()
        elif 'x' in nodes.columns and 'y' in nodes.columns:
            # Already in geographic CRS, x=lon, y=lat
            nodes['lon'] = nodes['x']
            nodes['lat'] = nodes['y']
            print(f"Using x/y as lon/lat (already in WGS84)")
            sys.stdout.flush()
        else:
            # Extract from geometry
            nodes['lon'] = nodes.geometry.x
            nodes['lat'] = nodes.geometry.y
    mapping = pd.read_parquet(poi_mapping_path)
    print(f"Loaded POI mapping: {len(mapping)} POIs")
    sys.stdout.flush()
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
    for node in _progress_wrapper(nodes_gdf.index, desc="Computing structure metrics", unit="node"):
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
        print(f"Computing betweenness centrality (sampling {sample_nodes} nodes)...")
        sys.stdout.flush()
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
        print("Computing closeness centrality...")
        sys.stdout.flush()
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

    for row in _progress_wrapper(mapping_reset.itertuples(index=False), total=len(mapping_reset), desc="Building POI-node index", unit="poi"):
        poi_index = getattr(row, "poi_index")
        nearest_node = getattr(row, "nearest_node")
        index_to_nodes[poi_index].add(nearest_node)
        if has_poi_id:
            poi_id_val = getattr(row, "poi_id")
            if poi_id_val is not None and not (isinstance(poi_id_val, float) and np.isnan(poi_id_val)):
                id_to_nodes[str(poi_id_val)].add(nearest_node)

    # Define expanded amenity type mappings (multiple OSM types -> single category)
    amenity_type_mappings = {
        "hospital": ["hospital", "clinic", "doctors"],
        "supermarket": ["supermarket", "grocery", "convenience", "mall", "marketplace"],
        "bank": ["bank", "atm"],
        # Other amenities map to themselves
        "school": ["school"],
        "pharmacy": ["pharmacy"],
        "bus_station": ["bus_station"],
        "park": ["park"],
    }
    
    # Build per-amenity node sets
    amenity_node_sets: Dict[str, set] = {}
    if pois_gdf is None or pois_gdf.empty:
        pois_subset = None
    else:
        pois_subset = pois_gdf

    for t in amenity_types:
        if pois_subset is None:
            amenity_node_sets[t] = set()
            continue
        
        # Get all OSM tag values that map to this category
        tag_values = amenity_type_mappings.get(t, [t])
        
        # Search across amenity, shop, and leisure columns for ANY matching tag value
        mask = pd.Series(False, index=pois_subset.index)
        for tag_value in tag_values:
            if "amenity" in pois_subset.columns:
                mask |= (pois_subset["amenity"] == tag_value)
            if "shop" in pois_subset.columns:
                mask |= (pois_subset["shop"] == tag_value)
            if "leisure" in pois_subset.columns:
                mask |= (pois_subset["leisure"] == tag_value)
        
        sel = pois_subset[mask]
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
    print(f"Computing network distances for {len(amenity_node_sets)} amenity types...")
    sys.stdout.flush()
    for t, node_set in _progress_wrapper(amenity_node_sets.items(), desc="Computing accessibility distances", unit="amenity_type"):
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


def compute_accessibility_score(distances_df: pd.DataFrame, amenity_weights: Dict[str, float], decay_constant: float = 2000.0):
    """
    Compute accessibility using exponential distance decay formula.
    
    Formula: accessibility = sum(weight * 100 * exp(-distance / decay_constant))
    Default decay constant of 2000m gives baseline scores in 60-70 range for typical cities.
    
    Args:
        distances_df: DataFrame with dist_to_{amenity} columns (in meters)
        amenity_weights: Dict mapping amenity types to importance weights
        decay_constant: Distance decay parameter in meters (default: 2000m)
    
    Returns:
        Series of accessibility scores (0-100 range)
    """
    acc = pd.Series(0.0, index=distances_df.index)
    total_weight = sum(amenity_weights.values())
    
    for amenity, weight in amenity_weights.items():
        col = f"dist_to_{amenity}"
        if col not in distances_df.columns:
            continue
        
        # Exponential decay: closer amenities contribute more
        distance = distances_df[col].fillna(10000.0)
        amenity_score = 100.0 * np.exp(-distance / decay_constant)
        
        # Weight by amenity importance and normalize by total weight
        acc += (weight / total_weight) * amenity_score
    
    return acc


def compute_travel_time_metrics(
    distances_df: pd.DataFrame, 
    amenity_weights: Dict[str, float],
    walking_speed_kmph: float,
    route_inefficiency_factor: float = 2.3
) -> pd.Series:
    """
    Compute weighted average travel time with route inefficiency adjustment.
    
    Applies route inefficiency factor to account for:
    - Non-straight-line routing (traffic signals, one-way streets)
    - Vertical movement (stairs, elevators, multi-level crossings)
    - Pedestrian detours and barriers
    
    Args:
        distances_df: DataFrame with dist_to_{amenity} columns (in meters)
        amenity_weights: Dict mapping amenity types to importance weights
        walking_speed_kmph: Walking speed in km/h (typically 4.8)
        route_inefficiency_factor: Multiplier for network distance (default: 2.3x)
    
    Returns:
        Series of weighted average travel time in minutes
    """
    walking_speed_mps = walking_speed_kmph * 1000 / 3600.0
    if walking_speed_mps <= 0:
        walking_speed_mps = 1.0
    
    total_weight = sum(amenity_weights.values())
    travel_time = pd.Series(0.0, index=distances_df.index)
    
    for amenity, weight in amenity_weights.items():
        col = f"dist_to_{amenity}"
        if col not in distances_df.columns:
            continue
        
        # Apply route inefficiency scaling to network distances
        adjusted_distance = distances_df[col].fillna(10000.0) * route_inefficiency_factor
        time_minutes = (adjusted_distance / 1000.0 / walking_speed_kmph) * 60.0
        
        # Weighted average across all amenity types
        travel_time += (weight / total_weight) * time_minutes
    
    return travel_time


def aggregate_h3(nodes_gdf: gpd.GeoDataFrame, value_series: pd.Series, h3_res=8):
    # ensure coordinates are in lat/lon before computing H3 indexes
    if nodes_gdf.crs is not None and nodes_gdf.crs.to_epsg() != 4326:
        nodes_latlon = nodes_gdf.to_crs(epsg=4326)
    else:
        nodes_latlon = nodes_gdf

    coords = list(zip(nodes_latlon.geometry.y, nodes_latlon.geometry.x))
    print(f"Computing H3 hexagons (resolution {h3_res})...")
    sys.stdout.flush()
    h3_indexes = [h3.geo_to_h3(lat, lng, h3_res) for lat, lng in _progress_wrapper(coords, desc="Computing H3 indexes", unit="node")]
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
    """Simple min-max normalization to 0-1 range."""
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index)
    min_val = valid.min()
    max_val = valid.max()
    if np.isclose(max_val, min_val):
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def clean_for_parquet(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean GeoDataFrame for parquet serialization by removing problematic columns.
    PyArrow cannot handle dicts with int keys, complex nested structures, etc.
    """
    df = gdf.copy()
    
    # List of OSMnx columns that commonly cause issues
    problematic_columns = ['contraction', 'ref', 'highway', 'name', 'street_count']
    
    # Drop known problematic columns if they exist
    cols_to_drop = [col for col in problematic_columns if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Check for any remaining object columns that might contain dicts/lists
    for col in df.columns:
        if col == 'geometry':  # Skip geometry column
            continue
        if df[col].dtype == 'object':
            # Check if column contains complex types
            sample = df[col].dropna().head(1)
            if not sample.empty:
                val = sample.iloc[0]
                if isinstance(val, (dict, list, set, tuple)):
                    df = df.drop(columns=[col])
    
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-path", required=True)
    p.add_argument("--poi-mapping", required=True)
    p.add_argument("--pois-path", required=False)
    p.add_argument("--out-dir", default="./data/analysis")
    p.add_argument("--h3-res", type=int, default=None)
    p.add_argument("--config", default="/config.yaml")
    p.add_argument("--force", action="store_true", help="Force recompute all cached steps")
    p.add_argument("--output-prefix", default="", help="Prefix applied to output filenames for scenario separation")
    args = p.parse_args()

    # Resolve config path relative to script location if not absolute
    project_root = Path(__file__).parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / args.config.lstrip("../")
    
    config = load_config(config_path)
    h3_resolution = args.h3_res or config.get("h3", {}).get("resolution", 8)
    walking_speed = config.get("walking_speed_kmph", 4.8)
    amenity_weights = config.get("amenity_weights", {})
    composite_weights = config.get("composite_weights", {"alpha": 0.25, "beta": 0.4, "gamma": 0.2, "delta": 0.15})
    coverage_thresholds = config.get("equity_thresholds", {key: 800 for key in amenity_weights.keys()})
    centrality_cfg = config.get("centrality", {})
    amenity_cutoff = config.get("amenity_distance_cutoff_m")
    circuity_sample_k = int(config.get("circuity_sample_k", 120))

    # Use centralized cache directory for scoring
    cache_dir = project_root / "data" / "cache" / "scoring"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve output directory relative to project root if not absolute
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / args.out_dir.lstrip("../")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading graph and POI mapping...")
    sys.stdout.flush()
    G, nodes, mapping = load_inputs(Path(args.graph_path), Path(args.poi_mapping))
    pois = None
    if args.pois_path:
        pois_path = Path(args.pois_path)
        # Prefer parquet if available (175x smaller, 20x faster: 1.5s vs 30s for 40k POIs)
        parquet_path = pois_path.parent / pois_path.stem
        parquet_path = parquet_path.with_suffix('.parquet')
        
        if parquet_path.exists():
            print(f"Loading POIs from {parquet_path} (fast!)...")
            sys.stdout.flush()
            pois = gpd.read_parquet(parquet_path)
            print(f"Loaded {len(pois)} POIs from parquet in ~1.5s")
            sys.stdout.flush()
        else:
            print(f"Loading POIs from {pois_path} (slow, ~30s)...")
            sys.stdout.flush()
            pois = gpd.read_file(pois_path)
            print(f"Loaded {len(pois)} POIs from file")
            sys.stdout.flush()
        if pois.crs is not None and nodes.crs is not None and pois.crs != nodes.crs:
            pois = pois.to_crs(nodes.crs)
        pois = ensure_poi_identifiers(pois)

    # CACHEABLE: Structure metrics (doesn't depend on weights/parameters)
    structure_cache_key = get_cache_key("structure", args.graph_path)
    structure_cache_path = cache_dir / f"structure_{structure_cache_key}.pkl"
    
    cached_nodes = load_from_cache(structure_cache_path)
    if not args.force and cached_nodes is not None:
        nodes = cached_nodes
    else:
        print("[COMPUTE] Computing structure metrics...")
        sys.stdout.flush()
        nodes = compute_structure_metrics(G, nodes)
        nodes = compute_centrality_metrics(G, nodes, centrality_cfg)
        save_to_cache(nodes, structure_cache_path)

    if not amenity_weights:
        raise ValueError("Amenity weights must be provided in the configuration file.")
    amenity_types = list(amenity_weights.keys())

    if pois is None:
        print("No POI dataset supplied; amenity distances will be NaN.")
        sys.stdout.flush()
        pois = gpd.GeoDataFrame(columns=["poi_id"], geometry=gpd.GeoSeries([], crs=nodes.crs), crs=nodes.crs)

    # CACHEABLE: Distance computations (depends on graph, POIs, amenity types, cutoff)
    distance_cache_key = get_cache_key(
        "distances",
        args.graph_path,
        args.poi_mapping,
        args.pois_path,
        tuple(sorted(amenity_types)),
        amenity_cutoff,
        args.output_prefix,
    )
    distance_cache_path = cache_dir / f"distances_{distance_cache_key}.pkl"
    
    cached_distances = load_from_cache(distance_cache_path)
    if not args.force and cached_distances is not None:
        distances = cached_distances
    else:
        print("[COMPUTE] Computing amenity distances...")
        sys.stdout.flush()
        distances = nearest_amenity_distances(G, nodes, mapping, pois, amenity_types, distance_cutoff=amenity_cutoff)
        save_to_cache(distances, distance_cache_path)
    nodes = nodes.join(distances)

    nodes["accessibility_score"] = compute_accessibility_score(distances, amenity_weights, decay_constant=2000.0)
    nodes["travel_time_min"] = compute_travel_time_metrics(distances, amenity_weights, walking_speed, route_inefficiency_factor=2.3)
    travel_time_clean = nodes["travel_time_min"].replace([np.inf, -np.inf], np.nan)
    valid_travel = travel_time_clean.dropna()
    fallback = float(valid_travel.max()) if not valid_travel.empty else 0.0
    if not np.isfinite(fallback) or fallback < 0:
        fallback = 0.0
    nodes["travel_time_min"] = travel_time_clean.fillna(fallback)
    
    # Travel time score: normalize to 0-100 scale (0 = 60+ min, 100 = 0 min)
    nodes["travel_time_score"] = 100.0 * (1.0 - nodes["travel_time_min"].clip(upper=60) / 60.0)

    structure_components = [
        normalize_series(nodes["degree"].fillna(0)),
        normalize_series(1.0 / (nodes["avg_incident_length_m"].fillna(1.0) + 1.0)),
    ]
    if centrality_cfg.get("compute_betweenness", True) and "betweenness_centrality" in nodes.columns:
        structure_components.append(normalize_series(nodes["betweenness_centrality"].fillna(0)))
    if centrality_cfg.get("compute_closeness", False) and "closeness_centrality" in nodes.columns:
        structure_components.append(normalize_series(nodes["closeness_centrality"].fillna(0)))
    
    # Structure score: normalize to 0-100 scale
    structure_normalized = sum(structure_components) / max(len(structure_components), 1)
    nodes["structure_score"] = structure_normalized * 100.0

    nodes_latlon = nodes.to_crs(epsg=4326) if nodes.crs and nodes.crs.to_epsg() != 4326 else nodes
    nodes["h3_index"] = [h3.geo_to_h3(pt.y, pt.x, h3_resolution) for pt in nodes_latlon.geometry]
    equity_variance = nodes.groupby("h3_index")["accessibility_score"].transform("var").fillna(0)
    
    # Equity score: normalize to 0-100 scale (lower variance = higher equity)
    max_variance = equity_variance.max() if equity_variance.max() > 0 else 1.0
    nodes["equity_score"] = 100.0 * (1.0 - equity_variance / max_variance)

    alpha = composite_weights.get("alpha", 0.05)
    beta = composite_weights.get("beta", 0.80)
    gamma = composite_weights.get("gamma", 0.05)
    delta = composite_weights.get("delta", 0.10)

    # Composite walkability: all components now on 0-100 scale
    nodes["walkability"] = (
        alpha * nodes["structure_score"]
        + beta * nodes["accessibility_score"]
        + gamma * nodes["equity_score"]
        + delta * nodes["travel_time_score"]
    )

    nodes, equity_agg = compute_equity_metrics(nodes, distances, h3_resolution, coverage_thresholds)
    agg_walkability = aggregate_h3(nodes, nodes["walkability"], h3_res=h3_resolution)
    agg = agg_walkability.merge(equity_agg, on="h3", how="left")

    # Normalize all scores to 1-100 scale for better interpretability
    # Scores are already on 0-100 scale (or close to it) based on formulas
    # Normalization removed to preserve absolute values for comparison
    # nodes["accessibility_score"] = normalize_series(nodes["accessibility_score"]) * 99 + 1
    # nodes["structure_score"] = normalize_series(nodes["structure_score"]) * 99 + 1
    # nodes["equity_score"] = normalize_series(nodes["equity_score"]) * 99 + 1
    # nodes["travel_time_score"] = normalize_series(nodes["travel_time_score"]) * 99 + 1
    # nodes["walkability"] = normalize_series(nodes["walkability"]) * 99 + 1
    
    circuity_value = compute_circuity_sample(G, nodes, sample_k=circuity_sample_k)
    travel_time_min_mean = safe_numeric(nodes["travel_time_min"].mean(), fallback)
    travel_time_score_mean = safe_numeric(nodes["travel_time_score"].mean(), 0.0)
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
            "travel_time_min_mean": travel_time_min_mean,
            "travel_time_score_mean": travel_time_score_mean,
        },
    }

    prefix = args.output_prefix.strip()
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"
    
    # Clean nodes GeoDataFrame before saving to parquet
    nodes_clean = clean_for_parquet(nodes)
    nodes_clean.to_parquet(out_dir / f"{prefix}nodes_with_scores.parquet")
    agg.to_parquet(out_dir / f"{prefix}h3_agg.parquet")
    nodes.to_csv(out_dir / f"{prefix}nodes_with_scores.csv", index=True)
    agg.to_csv(out_dir / f"{prefix}h3_agg.csv", index=False)
    with (out_dir / f"{prefix}metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved node-level scores and H3 aggregates to {out_dir} (prefix='{prefix}')")
    sys.stdout.flush()


if __name__ == "__main__":
    main()