#!/usr/bin/env python3
"""Generate optimized POI GeoJSON and map from the latest GA solution and recompute post-optimization metrics.

Optimization: Exports ONLY the optimized POIs (~13 POIs, instant) instead of combined poi_mapping.geojson
(40K+ POIs, 75 minutes). The massive combined GeoJSON is no longer needed for the pipeline.

Usage: run from project root. The script:
 - reads optimization/runs/best_candidate.json
 - extracts placements and best_distances from the GA metrics if available
 - builds an optimized POI GeoDataFrame with coordinates & distance diagnostics
 - exports optimized_pois.geojson (fast, ~13 POIs)
 - skips poi_mapping.geojson write (saves 75+ minutes, use --force if needed for legacy tools)
 - builds an interactive folium map (optimized placements highlighted)
 - (optionally) recomputes node-level optimized accessibility/travel metrics using the saved graph & config
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Go up from optimization-pipeline/ to root
DATA_PIPELINE_DIR = PROJECT_ROOT / "data-pipeline"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_ga import ensure_index_on_osmid

try:  # pragma: no cover - optional dependency
    import osmnx as ox
except ImportError:  # pragma: no cover - defensive
    ox = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - defensive
    tqdm = None  # type: ignore

if DATA_PIPELINE_DIR.exists() and str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))


def _load_compute_scores_fallback():
    """Load compute_scores directly via spec when sys.path imports fail."""
    compute_scores_path = DATA_PIPELINE_DIR / "compute_scores.py"
    if not compute_scores_path.exists():
        raise ModuleNotFoundError(
            f"compute_scores.py not found in {compute_scores_path.parent}"
        )
    spec = importlib.util.spec_from_file_location("compute_scores", compute_scores_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(
            f"Unable to build module spec for {compute_scores_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


try:
    from compute_scores import (
        compute_accessibility_score,
        compute_travel_time_metrics,
        load_config,
        nearest_amenity_distances,
    )
except ModuleNotFoundError:
    _compute_scores = _load_compute_scores_fallback()
    compute_accessibility_score = _compute_scores.compute_accessibility_score
    compute_travel_time_metrics = _compute_scores.compute_travel_time_metrics
    load_config = _compute_scores.load_config
    nearest_amenity_distances = _compute_scores.nearest_amenity_distances

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
    logging.info("Loading best candidate from %s", best_candidate_path)
    with best_candidate_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    # older code used key "candidate" or "candidate.signature"; support both
    signature = payload.get("candidate", "") if isinstance(payload, dict) else ""
    placements = parse_candidate_signature(str(signature))
    metrics = payload.get("metrics", {}) if isinstance(payload, Mapping) else {}
    if not placements:
        logging.warning("Best candidate signature is empty; no placements to visualise.")
    else:
        total_pois = sum(len(nodes) for nodes in placements.values())
        logging.info("Loaded %d amenity types with %d total POI placements", len(placements), total_pois)
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
    
    # Use Parquet cache for fast loading
    cache_path = pois_path.with_suffix('.parquet')
    
    # Check if cache exists and is newer than source
    if cache_path.exists():
        logging.info("Loading existing POIs from cache: %s", cache_path)
        try:
            gdf = gpd.read_parquet(cache_path)
            logging.info("Loaded %d existing POIs from cache (fast!)", len(gdf))
            return gdf
        except Exception as e:
            logging.warning("Cache load failed (%s), falling back to GeoJSON", e)
    
    # Load from GeoJSON (slow)
    logging.info("Loading existing POIs from %s (this may take a while...)", pois_path)
    gdf = gpd.read_file(pois_path)
    
    try:
        gdf = gdf.to_crs("EPSG:4326")
    except Exception:  # pragma: no cover - defensive
        logging.debug("Existing POIs already in EPSG:4326 or CRS conversion failed; proceeding as-is.")
    
    if "source" not in gdf.columns:
        gdf["source"] = "existing"
    else:
        gdf["source"] = gdf["source"].fillna("existing")
    
    logging.info("Loaded %d existing POIs", len(gdf))
    
    # Save to cache for next time
    try:
        logging.info("Saving POI cache to %s for faster future loads...", cache_path)
        gdf.to_parquet(cache_path)
        logging.info("Cache saved successfully")
    except Exception as e:
        logging.warning("Failed to save cache: %s", e)
    
    return gdf


def export_optimized_only_geojson(
    optimized_gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Export only the optimized POIs (fast, ~13 POIs vs 40K combined).
    
    This eliminates the 75-minute poi_mapping.geojson write bottleneck.
    The optimized POIs are all we need for the pipeline.
    """
    if optimized_gdf.empty:
        logging.warning("No optimized POIs to export; skipping.")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing %d optimized POIs to %s (fast!)...", len(optimized_gdf), output_path)
    optimized_gdf.to_file(output_path, driver="GeoJSON")
    logging.info("Optimized POIs written to %s", output_path)


def export_combined_geojson(
    existing_gdf: gpd.GeoDataFrame,
    optimized_gdf: gpd.GeoDataFrame,
    output_path: Path,
    skip_if_exists: bool = False,
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
    
    # Skip expensive write if file exists and we're told to skip
    if skip_if_exists and output_path.exists():
        logging.info("Combined GeoJSON already exists at %s, skipping write (use --force to overwrite)", output_path)
        return combined
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write in chunks for large files (faster and more memory efficient)
    logging.info("Writing combined GeoJSON to %s (this may take a while for large files)...", output_path)
    combined.to_file(output_path, driver="GeoJSON")
    logging.info("Combined GeoJSON written to %s", output_path)
    return combined


def compute_map_center(*gdfs: gpd.GeoDataFrame) -> Tuple[float, float]:
    lats: List[float] = []
    lons: List[float] = []
    for gdf in gdfs:
        if gdf is None or gdf.empty:
            continue
        # Use centroids to handle all geometry types (Point, LineString, Polygon, etc.)
        # Project to a local projected CRS to avoid centroid warnings
        gdf_projected = gdf.to_crs("EPSG:3857")  # Web Mercator
        centroids = gdf_projected.geometry.centroid.to_crs("EPSG:4326")
        lats.extend(centroids.y.astype(float).tolist())
        lons.extend(centroids.x.astype(float).tolist())
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
    graph_path: Optional[Path] = None,
    placements: Optional[Mapping[str, Sequence[str]]] = None,
) -> None:
    logging.info("Building interactive map...")
    center_lat, center_lon = compute_map_center(existing_gdf, optimized_gdf)
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")
    
    # Load graph for path visualization if available
    import networkx as nx
    G = None
    if graph_path and graph_path.exists() and placements and not optimized_gdf.empty:
        try:
            logging.info("Loading graph for path visualization...")
            G = nx.read_graphml(str(graph_path))
            # Fix edge weights
            for u, v, k, data in G.edges(keys=True, data=True):
                if 'length' in data and isinstance(data['length'], str):
                    data['length'] = float(data['length'])
            logging.info("Graph loaded successfully")
        except Exception as e:
            logging.warning("Failed to load graph for paths: %s", e)
            G = None

    if not existing_gdf.empty:
        existing_group = folium.FeatureGroup(name="Existing POIs", show=False)
        cluster = MarkerCluster(name="Existing POIs")
        
        # Parallel preparation of marker data
        def prepare_existing_marker(row_data):
            """Prepare marker data for an existing POI."""
            idx, row = row_data
            geometry = row.geometry
            if geometry is None:
                return None
            
            # Handle different geometry types by using centroid
            if geometry.geom_type == 'Point':
                lat, lon = geometry.y, geometry.x
            else:
                centroid = geometry.centroid
                lat, lon = centroid.y, centroid.x
            
            # Build popup with POI information
            amenity_type = row.get('amenity', row.get('fclass', 'Unknown'))
            popup_lines = [
                f"<b>Existing POI</b>",
                f"Type: {amenity_type}",
                f"OSM ID: {row.get('osmid', row.get('osm_id', '-'))}"
            ]
            
            # Add name if available
            name = row.get('name', row.get('Name', ''))
            if pd.notna(name) and str(name).strip():
                popup_lines.append(f"Name: {name}")
            
            # Add any other relevant fields
            if 'tags' in row and pd.notna(row['tags']):
                popup_lines.append(f"Tags: {row['tags']}")
            
            popup_html = "<br/>".join(popup_lines)
            
            return {
                'lat': lat,
                'lon': lon,
                'popup_html': popup_html,
                'amenity_type': amenity_type
            }
        
        # Use ThreadPoolExecutor for parallel preparation (50% of cores)
        num_workers = max(1, int(os.cpu_count() * 0.5)) if os.cpu_count() else 4
        logging.info(f"Preparing {len(existing_gdf)} existing POI markers using {num_workers} workers...")
        
        marker_data_list = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(prepare_existing_marker, row_data): idx 
                      for idx, row_data in enumerate(existing_gdf.iterrows())}
            
            for future in iter_with_progress(as_completed(futures), "Existing POIs", total=len(futures)):
                result = future.result()
                if result:
                    marker_data_list.append(result)
        
        # Add markers to cluster (must be sequential)
        logging.info(f"Adding {len(marker_data_list)} markers to map...")
        for marker_data in marker_data_list:
            folium.CircleMarker(
                location=[marker_data['lat'], marker_data['lon']],
                radius=3,
                color="#3388ff",
                opacity=0.8,
                fill=True,
                fill_opacity=0.5,
                popup=folium.Popup(marker_data['popup_html'], max_width=250),
                tooltip=f"{marker_data['amenity_type']}"
            ).add_to(cluster)
        
        cluster.add_to(existing_group)
        existing_group.add_to(fmap)

    if not optimized_gdf.empty:
        optimized_group = folium.FeatureGroup(name="Optimized Placements", show=True)
        
        # Compute paths to nearest EXISTING amenities if graph available
        paths_data = {}
        if G is not None and placements and not existing_gdf.empty:
            logging.info("Computing paths to nearest EXISTING amenities for each optimized placement...")
            
            # Precompute undirected graph and node coordinates
            G_undir = G.to_undirected()
            node_coords = {}
            for node_id in G.nodes():
                node_data = G.nodes[node_id]
                lat = float(node_data.get('lat', node_data.get('y')))
                lon = float(node_data.get('lon', node_data.get('x')))
                node_coords[node_id] = (lat, lon)
            
            # Build mapping of existing POIs by amenity type to their nearest graph nodes
            logging.info("Mapping existing POIs to graph nodes...")
            existing_by_type = {}
            for amenity_type in placements.keys():
                # Filter existing POIs by type (check multiple columns)
                matching_pois = existing_gdf[
                    (existing_gdf.get('amenity') == amenity_type) | 
                    (existing_gdf.get('fclass') == amenity_type) |
                    (existing_gdf.get('shop') == amenity_type) |
                    (existing_gdf.get('leisure') == amenity_type if amenity_type == 'park' else False)
                ]
                
                if matching_pois.empty:
                    logging.warning(f"No existing {amenity_type} POIs found")
                    continue
                
                # Find nearest graph node for each POI
                existing_nodes = set()
                for _, poi in matching_pois.iterrows():
                    geom = poi.geometry
                    if geom is None:
                        continue
                    
                    poi_lat = geom.y if geom.geom_type == 'Point' else geom.centroid.y
                    poi_lon = geom.x if geom.geom_type == 'Point' else geom.centroid.x
                    
                    # Find nearest graph node (simple approach: check nodes within 100m)
                    min_dist = float('inf')
                    nearest_node = None
                    for node_id, (lat, lon) in node_coords.items():
                        dist = abs(lat - poi_lat) + abs(lon - poi_lon)  # Manhattan distance
                        if dist < min_dist:
                            min_dist = dist
                            nearest_node = node_id
                    
                    if nearest_node and min_dist < 0.01:  # ~1km threshold
                        existing_nodes.add(nearest_node)
                
                existing_by_type[amenity_type] = list(existing_nodes)
                logging.info(f"Found {len(existing_nodes)} existing {amenity_type} nodes (from {len(matching_pois)} POIs)")
            
            def haversine_distance(lat1, lon1, lat2, lon2):
                """Calculate haversine distance in meters."""
                import math
                R = 6371000
                phi1, phi2 = math.radians(lat1), math.radians(lat2)
                dphi = math.radians(lat2 - lat1)
                dlambda = math.radians(lon2 - lon1)
                a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
                return 2 * R * math.asin(math.sqrt(a))
            
            def find_nearest_existing_paths(args_tuple):
                """Find paths to k nearest EXISTING amenities using expanding bounding box."""
                source, amenity_type, k = args_tuple
                
                existing_nodes = existing_by_type.get(amenity_type, [])
                if not existing_nodes:
                    return source, []
                
                if source not in node_coords:
                    logging.warning(f"Source node {source} not found in graph")
                    return source, []
                
                source_lat, source_lon = node_coords[source]
                
                # Sort existing amenities by straight-line distance
                candidate_distances = []
                for target in existing_nodes:
                    if target not in node_coords:
                        continue
                    target_lat, target_lon = node_coords[target]
                    straight_dist = haversine_distance(source_lat, source_lon, target_lat, target_lon)
                    candidate_distances.append((target, straight_dist))
                
                candidate_distances.sort(key=lambda x: x[1])
                
                # Expanding bounding box: start at 1200m, double until we find k candidates
                search_radius = 1200
                max_radius = 20000
                nearby_candidates = []
                
                while len(nearby_candidates) < k and search_radius <= max_radius:
                    nearby_candidates = [t for t, d in candidate_distances if d <= search_radius]
                    if len(nearby_candidates) < k:
                        search_radius *= 2
                
                if len(nearby_candidates) < k:
                    nearby_candidates = [t for t, _ in candidate_distances]
                
                # Compute network paths to nearby candidates
                results = []
                for target in nearby_candidates[:k * 2]:
                    try:
                        dist = nx.shortest_path_length(G_undir, source, target, weight='length')
                        path = nx.shortest_path(G_undir, source, target, weight='length')
                        results.append((target, dist, path))
                    except (nx.NetworkXNoPath, nx.NodeNotFound, Exception):
                        continue
                
                results.sort(key=lambda x: x[1])
                return source, results[:k]
            
            # Build task list: for each new placement, find nearest existing amenities of same type
            tasks = []
            for amenity_type, node_list in placements.items():
                for source_node in node_list:
                    tasks.append((source_node, amenity_type, 5))
            
            # Compute paths in parallel (use 50% of cores)
            num_workers = max(1, int(os.cpu_count() * 0.5)) if os.cpu_count() else 4
            logging.info(f"Computing {len(tasks)} path sets using {num_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(find_nearest_existing_paths, task): task for task in tasks}
                for future in iter_with_progress(as_completed(futures), "Path computations", total=len(futures)):
                    try:
                        source_node, paths = future.result()
                        paths_data[source_node] = paths
                        if paths:
                            logging.debug(f"Node {source_node}: found {len(paths)} existing neighbors, nearest at {paths[0][1]:.0f}m")
                    except Exception as e:
                        logging.error(f"Failed to compute paths: {e}")
        
        # Parallel preparation of optimized marker data
        def prepare_optimized_marker(row_data):
            """Prepare marker data for an optimized POI."""
            idx, row = row_data
            geometry = row.geometry
            if geometry is None:
                return None
            
            # Handle different geometry types by using centroid
            if geometry.geom_type == 'Point':
                lat, lon = geometry.y, geometry.x
            else:
                centroid = geometry.centroid
                lat, lon = centroid.y, centroid.x
            
            amenity_type = row.get('amenity', '-')
            osmid = str(row.get('osmid', '-'))
            
            popup_lines = [
                f"<b>🎯 Optimized Placement</b>",
                f"Amenity Type: <b>{amenity_type}</b>",
                f"OSM Node ID: {osmid}"
            ]
            
            distance_value = row.get("distance_m")
            if pd.notna(distance_value):
                popup_lines.append(f"Representative distance: {float(distance_value):.1f} m")
            
            travel_time = row.get("travel_time_min")
            if pd.notna(travel_time):
                popup_lines.append(f"Travel time: {float(travel_time):.1f} min")
            
            # Add nearest neighbor info if available
            if osmid in paths_data and paths_data[osmid]:
                popup_lines.append("<br/><b>📍 Nearest Existing Amenities:</b>")
                for i, (target, dist, _) in enumerate(paths_data[osmid][:5], 1):
                    popup_lines.append(f"{i}. {dist:.0f}m away")
            elif osmid in paths_data:
                popup_lines.append("<br/><i>No existing amenities found nearby</i>")
            
            popup_html = "<br/>".join(popup_lines)
            
            return {
                'lat': lat,
                'lon': lon,
                'popup_html': popup_html,
                'amenity_type': amenity_type,
                'osmid': osmid
            }
        
        # Use ThreadPoolExecutor for parallel preparation (50% of cores)
        num_workers = max(1, int(os.cpu_count() * 0.5)) if os.cpu_count() else 4
        logging.info(f"Preparing {len(optimized_gdf)} optimized markers using {num_workers} workers...")
        
        marker_data_list = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(prepare_optimized_marker, row_data): idx 
                      for idx, row_data in enumerate(optimized_gdf.iterrows())}
            
            for future in iter_with_progress(as_completed(futures), "Optimized placements", total=len(futures)):
                result = future.result()
                if result:
                    marker_data_list.append(result)
        
        # Add paths as separate feature group (hidden by default)
        if G is not None and paths_data:
            paths_group = folium.FeatureGroup(name="Paths to Nearest Existing Amenities", show=False)
            colors = ['blue', 'green', 'purple', 'orange', 'darkred']
            
            logging.info("Adding path visualizations to map...")
            for marker_data in marker_data_list:
                osmid = marker_data['osmid']
                if osmid not in paths_data:
                    continue
                
                for idx, (target, dist, path) in enumerate(paths_data[osmid]):
                    try:
                        # Get coordinates for path
                        path_coords = []
                        for node in path:
                            node_data = G.nodes[node]
                            lat = float(node_data.get('lat', node_data.get('y')))
                            lon = float(node_data.get('lon', node_data.get('x')))
                            path_coords.append([lat, lon])
                        
                        color = colors[idx % len(colors)]
                        folium.PolyLine(
                            locations=path_coords,
                            color=color,
                            weight=2,
                            opacity=0.6,
                            popup=f"Path {idx+1}: {dist:.0f}m ({len(path)-1} edges)",
                            tooltip=f"{dist:.0f}m to neighbor #{idx+1}"
                        ).add_to(paths_group)
                    except Exception as e:
                        logging.debug("Failed to create path for %s: %s", osmid, e)
            
            paths_group.add_to(fmap)
        
        # Add markers to group (must be sequential)
        logging.info(f"Adding {len(marker_data_list)} optimized markers to map...")
        for marker_data in marker_data_list:
            folium.CircleMarker(
                location=[marker_data['lat'], marker_data['lon']],
                radius=6,
                color="#d95f02",
                weight=2,
                opacity=0.9,
                fill=True,
                fill_color="#d95f02",
                fill_opacity=0.7,
                popup=folium.Popup(marker_data['popup_html'], max_width=300),
                tooltip=f"New {marker_data['amenity_type']}"
            ).add_to(optimized_group)
        
        optimized_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output_html)
    logging.info("Interactive map written to %s", output_html)


def compute_optimized_scores(
    placements: Mapping[str, Sequence[str]],
    optimized_gdf: gpd.GeoDataFrame,
    existing_pois_gdf: gpd.GeoDataFrame,
    existing_poi_mapping: pd.DataFrame,
    nodes_df: pd.DataFrame,
    graph_path: Optional[Path],
    config_path: Optional[Path],
    nodes_output: Path,
    summary_output: Path,
) -> None:
    """Recompute optimized accessibility and travel metrics using the road graph and config.
    
    Combines existing POI-to-node mapping with new optimized placements to compute distances.

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
    logging.info("Loading graph from %s", graph_path)
    G = ox.load_graphml(graph_path)
    logging.info("Converting graph to GeoDataFrame...")
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

    # Build mapping for new optimized POIs
    new_poi_mapping = pd.DataFrame(mapping_records).set_index("poi_index")
    
    # CRITICAL: Combine existing POI mapping with new optimized POI mapping
    logging.info("Combining existing POI mapping (%d entries) with new placements (%d entries)...", 
                 len(existing_poi_mapping), len(new_poi_mapping))
    logging.info("Existing POI mapping columns: %s", list(existing_poi_mapping.columns))
    logging.info("New POI mapping columns: %s", list(new_poi_mapping.columns))
    logging.info("Sample existing POI: %s", existing_poi_mapping.iloc[0].to_dict() if len(existing_poi_mapping) > 0 else "None")
    logging.info("Sample new POI: %s", new_poi_mapping.iloc[0].to_dict() if len(new_poi_mapping) > 0 else "None")
    combined_mapping = pd.concat([existing_poi_mapping, new_poi_mapping])
    logging.info("Combined mapping shape: %s", combined_mapping.shape)
    
    # Combine POI GeoDataFrames
    combined_pois = pd.concat([existing_pois_gdf, filtered_gdf], ignore_index=True)
    logging.info("Combined POI GeoDataFrame shape: %s", combined_pois.shape)
    
    # DEBUG: Check if new POIs have required columns for distance matching
    logging.info("Existing POIs columns with 'amenity': %d", existing_pois_gdf['amenity'].notna().sum() if 'amenity' in existing_pois_gdf.columns else 0)
    logging.info("New POIs columns with 'amenity': %d", filtered_gdf['amenity'].notna().sum() if 'amenity' in filtered_gdf.columns else 0)
    logging.info("Combined POIs with 'amenity': %d", combined_pois['amenity'].notna().sum() if 'amenity' in combined_pois.columns else 0)
    
    # Check if any new POIs will match amenity types
    for amenity in amenity_types[:2]:  # Check first 2
        existing_count = (existing_pois_gdf['amenity'] == amenity).sum() if 'amenity' in existing_pois_gdf.columns else 0
        new_count = (filtered_gdf['amenity'] == amenity).sum() if 'amenity' in filtered_gdf.columns else 0
        logging.info(f"  {amenity}: {existing_count} existing + {new_count} new = {existing_count + new_count} total")

    # compute nearest distances (returns DataFrame indexed by node id)
    logging.info("Computing nearest amenity distances using combined POI set (%d total POIs)...", len(combined_mapping))
    distances = nearest_amenity_distances(
        G,
        nodes_gdf,
        combined_mapping,
        combined_pois,
        amenity_types,
        distance_cutoff=amenity_cutoff,
    )
    
    # DEBUG: Check distance stats
    logging.info("Distance computation complete. Sample distances:")
    for col in distances.columns[:3]:  # Show first 3 amenity types
        vals = distances[col].dropna()
        logging.info("  %s: mean=%.1fm, median=%.1fm, min=%.1fm, max=%.1fm (n=%d)", 
                    col, vals.mean(), vals.median(), vals.min(), vals.max(), len(vals))
    
    logging.info("Distance DataFrame shape: %s, index type: %s", distances.shape, distances.index.dtype)
    logging.info("Distance DataFrame index sample: %s", distances.index[:5].tolist())
    logging.info("Distance columns: %s", distances.columns.tolist())
    distances.index = distances.index.map(str)

    nodes_aligned = nodes_df.copy()
    # Convert both to same index type for alignment
    nodes_aligned.index = nodes_aligned.index.astype(int)
    distances.index = distances.index.astype(int)
    distances = distances.reindex(nodes_aligned.index)
    
    logging.info("After reindex - distances shape: %s, nodes_aligned shape: %s", distances.shape, nodes_aligned.shape)
    logging.info("After reindex - distances.dist_to_hospital mean: %.2fm", distances['dist_to_hospital'].mean())

    logging.info("Computing accessibility scores for %d nodes...", len(nodes_aligned))
    optimized_accessibility = compute_accessibility_score(distances, amenity_weights, decay_constant=2000.0)
    logging.info("Computing travel time metrics...")
    optimized_travel_time = compute_travel_time_metrics(distances, amenity_weights, walking_speed, route_inefficiency_factor=2.3)
    travel_time_clean = optimized_travel_time.replace([np.inf, -np.inf], np.nan)
    fallback = travel_time_clean.max()
    if pd.isna(fallback) or fallback <= 0:
        fallback = 1.0
    # Travel time score: 0-100 scale (0 = 60+ min, 100 = 0 min)
    optimized_travel_score = 100.0 * (1.0 - travel_time_clean.fillna(fallback).clip(upper=60) / 60.0)

    structure_component = nodes_aligned.get("structure_score", pd.Series(0.0, index=nodes_aligned.index)).fillna(0.0)
    equity_component = nodes_aligned.get("equity_score", pd.Series(0.0, index=nodes_aligned.index)).fillna(0.0)

    # Updated composite weights (validated formula)
    alpha = float(composite_weights.get("alpha", 0.05))
    beta = float(composite_weights.get("beta", 0.80))
    gamma = float(composite_weights.get("gamma", 0.05))
    delta = float(composite_weights.get("delta", 0.10))

    logging.info("Computing composite walkability scores...")
    optimized_walkability = (
        alpha * structure_component
        + beta * optimized_accessibility.fillna(0.0)
        + gamma * equity_component
        + delta * optimized_travel_score.fillna(0.0)
    )

    nodes_result = nodes_aligned.copy()
    logging.info("Before distance copy - nodes_result dist_to_hospital mean: %.2fm", nodes_result['dist_to_hospital'].mean())
    for column in distances.columns:
        nodes_result[column] = distances[column]
    logging.info("After distance copy - nodes_result dist_to_hospital mean: %.2fm", nodes_result['dist_to_hospital'].mean())
    logging.info("Distance columns copied: %s", distances.columns.tolist())
    nodes_result["optimized_accessibility_score"] = optimized_accessibility
    nodes_result["optimized_travel_time_min"] = optimized_travel_time
    nodes_result["optimized_travel_time_score"] = optimized_travel_score
    nodes_result["optimized_walkability"] = optimized_walkability

    logging.info("Saving optimized scores to output files...")
    nodes_output.parent.mkdir(parents=True, exist_ok=True)
    nodes_result.to_parquet(nodes_output)
    csv_output = nodes_output.with_suffix(".csv")
    nodes_result.to_csv(csv_output, index=True)

    summary = {
        "network": {},
        "scores": {
            "accessibility_mean": float(optimized_accessibility.replace([np.inf, -np.inf], np.nan).mean(skipna=True)),
            "travel_time_min_mean": float(travel_time_clean.mean(skipna=True)),
            "travel_time_score_mean": float(optimized_travel_score.mean(skipna=True)),
            "walkability_mean": float(optimized_walkability.mean(skipna=True)),
            "equity_mean": float(nodes_aligned.get("equity_score", pd.Series(0.0)).mean()),
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
        default=project_root / "data" / "optimization" / "runs" / "best_candidate.json",
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
        default=project_root / "data" / "raw" / "osm"/ "pois.geojson",
        help="GeoJSON containing existing POIs to overlay.",
    )
    parser.add_argument(
        "--poi-mapping",
        type=Path,
        default=project_root / "data" / "processed" / "poi_node_mapping.parquet",
        help="Parquet file mapping existing POIs to graph nodes.",
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
        default=project_root / "data"/ "optimization" / "runs" / "poi_mapping.geojson",
        help="Destination for the combined POI GeoJSON (legacy, not needed for pipeline).",
    )
    parser.add_argument(
        "--optimized-pois-output",
        type=Path,
        default=project_root / "data" / "optimization" / "runs" / "optimized_pois.geojson",
        help="Destination for optimized-only POIs (fast, small file).",
    )
    parser.add_argument(
        "--output-map",
        type=Path,
        default=project_root / "data"/ "optimization" / "runs" / "optimized_map.html",
        help="Destination for the interactive HTML map.",
    )
    parser.add_argument(
        "--optimized-nodes-output",
        type=Path,
        default=project_root / "data" / "analysis" / "optimized_nodes_with_scores.parquet",
        help="Path to write node-level optimized scores (parquet).",
    )
    parser.add_argument(
        "--optimized-summary-output",
        type=Path,
        default=project_root / "data" / "analysis" / "optimized_metrics_summary.json",
        help="Path to write optimized score summary (JSON).",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        default=False,
        help="Skip recomputing post-optimization metrics (fast: only map + GeoJSON).",
    )
    parser.add_argument(
        "--skip-geojson",
        action="store_true",
        default=False,
        help="Skip writing the large combined GeoJSON file (much faster, map generation only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite of existing combined GeoJSON.",
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

    # Export optimized POIs only (fast, ~13 POIs)
    export_optimized_only_geojson(optimized_gdf, args.optimized_pois_output)
    
    existing_gdf = load_existing_pois(args.existing_pois)
    
    # Skip massive combined GeoJSON export by default (saves 75+ minutes)
    # Only create if explicitly requested with --force flag and not --skip-geojson
    if args.skip_geojson:
        logging.info("Skipping combined GeoJSON export (--skip-geojson)")
        # Create combined in-memory for map centering only
        components = []
        if not existing_gdf.empty:
            components.append(existing_gdf)
        if not optimized_gdf.empty:
            components.append(optimized_gdf)
        combined_gdf = gpd.GeoDataFrame(pd.concat(components, ignore_index=True), crs="EPSG:4326") if components else gpd.GeoDataFrame(columns=["source"], geometry=[], crs="EPSG:4326")
    elif args.force:
        # Only write combined GeoJSON if explicitly forced (legacy compatibility)
        logging.info("Writing combined GeoJSON (--force flag enabled, 75+ min for large datasets)")
        combined_gdf = export_combined_geojson(existing_gdf, optimized_gdf, args.output_geojson, skip_if_exists=False)
    else:
        # Default: skip combined GeoJSON write (optimized POIs already exported above)
        logging.info("Skipping combined GeoJSON export (default behavior, saves 75+ min)")
        logging.info("  Tip: Use --force to create combined poi_mapping.geojson if needed for legacy tools")
        # Create combined in-memory for map centering
        components = []
        if not existing_gdf.empty:
            components.append(existing_gdf)
        if not optimized_gdf.empty:
            components.append(optimized_gdf)
        combined_gdf = gpd.GeoDataFrame(pd.concat(components, ignore_index=True), crs="EPSG:4326") if components else gpd.GeoDataFrame(columns=["source"], geometry=[], crs="EPSG:4326")

    # Use the combined GeoDataFrame for centering if export succeeded.
    if not combined_gdf.empty:
        existing_view = combined_gdf[combined_gdf["source"] == "existing"]
        optimized_view = combined_gdf[combined_gdf["source"] == "optimized"]
    else:
        existing_view = existing_gdf
        optimized_view = optimized_gdf

    # Map generation commented out for performance
    # build_map(existing_view, optimized_view, args.output_map, args.graph_path, placements)
    # logging.info("Map generation skipped (commented out in code")

    if not args.skip_metrics:
        # Load existing POI mapping
        if args.poi_mapping.exists():
            logging.info("Loading existing POI mapping from %s", args.poi_mapping)
            existing_poi_mapping = pd.read_parquet(args.poi_mapping)
        else:
            logging.warning("POI mapping file not found: %s. Using empty mapping.", args.poi_mapping)
            existing_poi_mapping = pd.DataFrame(columns=['poi_index', 'nearest_node', 'poi_id']).set_index('poi_index')
        
        compute_optimized_scores(
            placements,
            optimized_gdf,
            existing_gdf,
            existing_poi_mapping,
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
