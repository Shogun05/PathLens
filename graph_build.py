#!/usr/bin/env python3
"""
Simplify and annotate graph, map POIs to nearest street nodes.
Inputs: raw graph.graphml and pois.geojson
Outputs: simplified graph.graphml, nodes.parquet, edges.parquet, poi_node_mapping.parquet
"""
import argparse
from pathlib import Path
import math

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from tqdm import tqdm

ox.settings.use_cache = True
ox.settings.log_console = True


def normalize_list_columns(gdf):
    """Convert columns with mixed list/non-list values to consistent string format for parquet compatibility."""
    gdf = gdf.copy()
    for col in gdf.columns:
        if col == 'geometry':
            continue
        # Check if column contains any lists
        if gdf[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            gdf[col] = gdf[col].apply(
                lambda x: ','.join(map(str, x)) if isinstance(x, (list, tuple)) else str(x) if pd.notna(x) else None
            )
    return gdf


def load_inputs(graph_path: Path, pois_path: Path):
    G = ox.load_graphml(graph_path)
    pois = gpd.read_file(pois_path)
    return G, pois


def load_optional_layer(layer_path: str | Path | None, strict: bool = False):
    if not layer_path:
        return None
    path_obj = Path(layer_path)
    if not path_obj.exists():
        if strict:
            raise FileNotFoundError(f"Optional layer not found: {path_obj}")
        return None
    layer = gpd.read_file(path_obj)
    if layer.empty:
        return None
    return layer


def _coerce_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def graph_already_simplified(G: nx.MultiDiGraph) -> bool:
    """Best-effort detection for graphs already simplified, covering common OSMnx flags."""
    # Check graph-level metadata first
    for key in ("is_simplified", "simplified"):
        coerced = _coerce_bool(G.graph.get(key))
        if coerced is not None:
            return coerced
    # Fallback: inspect a sample of edges for the simplified flag
    for _, _, data in G.edges(data=True):
        coerced = _coerce_bool(data.get("simplified"))
        if coerced is not None:
            return coerced
    return False


def add_edge_lengths_compat(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Add edge lengths using whichever OSMnx API is available, falling back to manual computation."""
    # Prefer OSMnx helpers when present to stay consistent with upstream behavior
    try:
        from osmnx import distance

        if hasattr(distance, "add_edge_lengths"):
            return distance.add_edge_lengths(G)
    except ImportError:
        pass
    except AttributeError:
        pass

    if hasattr(ox, "distance") and hasattr(ox.distance, "add_edge_lengths"):
        return ox.distance.add_edge_lengths(G)

    if hasattr(ox, "add_edge_lengths"):
        try:
            return ox.add_edge_lengths(G)
        except AttributeError:
            pass

    # Manual fallback: compute length from geometry or straight-line distance
    for u, v, k, data in G.edges(keys=True, data=True):
        if data.get("length"):
            continue
        geom = data.get("geometry")
        if geom is not None:
            data["length"] = float(geom.length)
            continue
        x1 = G.nodes[u].get("x")
        y1 = G.nodes[u].get("y")
        x2 = G.nodes[v].get("x")
        y2 = G.nodes[v].get("y")
        if None not in (x1, y1, x2, y2):
            data["length"] = math.hypot(x2 - x1, y2 - y1)
        else:
            data["length"] = 0.0
    return G


def simplify_and_annotate(G):
    # Project to UTM for metric calculations
    G_proj = ox.project_graph(G)
    # Simplify (merges nodes on straight lines) only if not already simplified
    if graph_already_simplified(G_proj):
        print("Graph is already simplified; skipping simplify_graph().")
        Gs = G_proj
    else:
        Gs = ox.simplify_graph(G_proj)
    # add edge lengths (meters)
    Gs = add_edge_lengths_compat(Gs)
    # add bearing, maybe travel_time at walking speed (4.8 km/h)
    walk_speed_mps = 4.8 * 1000 / 3600.0
    for u, v, k, data in Gs.edges(keys=True, data=True):
        length = data.get("length", 0.0)
        data["walk_time_s"] = length / walk_speed_mps
    return Gs


def nodes_edges_to_gdfs(Gs, crs_epsg=4326):
    nodes, edges = ox.graph_to_gdfs(Gs, nodes=True, edges=True)
    # reproject back to EPSG:4326 for storage/visualization
    nodes = nodes.to_crs(epsg=crs_epsg)
    edges = edges.to_crs(epsg=crs_epsg)
    # Normalize list columns for parquet compatibility
    nodes = normalize_list_columns(nodes)
    edges = normalize_list_columns(edges)
    return nodes, edges


def annotate_nodes(nodes_gdf: gpd.GeoDataFrame, G: nx.MultiDiGraph, buildings: gpd.GeoDataFrame | None, landuse: gpd.GeoDataFrame | None):
    nodes = nodes_gdf.copy()
    degree_series = pd.Series(dict(G.degree()))
    nodes["degree"] = nodes.index.map(degree_series).fillna(0)
    nodes["intersection_type"] = pd.cut(
        nodes["degree"],
        bins=[-1, 1, 2, 3, float("inf")],
        labels=["dead_end", "midblock", "three_way", "four_plus"],
    )
    if "street_count" in nodes.columns:
        nodes["street_count"].fillna(nodes["degree"], inplace=True)

    if buildings is not None and not buildings.empty:
        try:
            buildings_proj = buildings.to_crs(nodes.crs)
            if "building" in buildings_proj.columns:
                nearest_buildings = gpd.sjoin_nearest(
                    nodes,
                    buildings_proj[["geometry", "building"]],
                    how="left",
                    distance_col="dist_to_building",
                )
                nodes["nearest_building_type"] = nearest_buildings["building"]
                nodes["dist_to_building"] = nearest_buildings["dist_to_building"]
        except Exception:
            pass

    if landuse is not None and not landuse.empty:
        try:
            landuse_proj = landuse.to_crs(nodes.crs)
            if "landuse" in landuse_proj.columns:
                nearest_landuse = gpd.sjoin_nearest(
                    nodes,
                    landuse_proj[["geometry", "landuse"]],
                    how="left",
                    distance_col="dist_to_landuse",
                )
                nodes["nearest_landuse_type"] = nearest_landuse["landuse"]
                nodes["dist_to_landuse"] = nearest_landuse["dist_to_landuse"]
        except Exception:
            pass

    return nodes


def map_pois_to_nodes(pois_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame):
    # Ensure consistent CRS and filter out invalid geometries
    pois = pois_gdf.to_crs(nodes_gdf.crs).copy()
    pois = pois[pois.geometry.notnull()].copy()
    pois = pois[~pois.geometry.is_empty].copy()
    if pois.empty:
        raise ValueError("POI GeoDataFrame has no valid geometries after CRS alignment.")

    # Convert non-point geometries to centroids to enable nearest-node lookup
    non_point_mask = ~pois.geometry.geom_type.isin(["Point", "MultiPoint"])
    if non_point_mask.any():
        print("Converting non-point POI geometries to centroids for nearest-node mapping.")
        pois.loc[non_point_mask, "geometry"] = pois.loc[non_point_mask, "geometry"].centroid

    multi_point_mask = pois.geometry.geom_type == "MultiPoint"
    if multi_point_mask.any():
        pois.loc[multi_point_mask, "geometry"] = pois.loc[multi_point_mask, "geometry"].apply(
            lambda geom: next(iter(geom.geoms))
        )

    if not pois.geometry.geom_type.isin(["Point"]).all():
        raise ValueError("Unable to convert all POI geometries to Points for nearest-node mapping.")
    # nearest node mapping using OSMnx helper
    # prepare arrays of x,y
    xs = pois.geometry.x.values
    ys = pois.geometry.y.values
    # note: for large inputs, consider vectorized rtree nearest neighbor approach
    # Get node IDs from the index
    node_ids = nodes_gdf.index.tolist()
    node_coords = [(nodes_gdf.loc[nid].geometry.x, nodes_gdf.loc[nid].geometry.y) for nid in node_ids]
    
    # For each POI, find nearest node
    from scipy.spatial import cKDTree
    tree = cKDTree(node_coords)
    poi_coords = list(zip(xs, ys))
    distances, indices = tree.query(poi_coords)
    node_index = [node_ids[i] for i in indices]
    
    # Create mapping DataFrame
    primary_category = None
    for col in ("amenity", "shop", "leisure", "landuse"):
        if col in pois.columns:
            primary_category = col
            break

    if primary_category:
        category_values = pois[primary_category].astype(str)
    else:
        category_values = pd.Series([None] * len(pois), index=pois.index)

    mapping = pd.DataFrame({
        "poi_index": pois.index,
        "poi_id": pois.get("osmid").astype(str) if "osmid" in pois.columns else pois.index.astype(str),
        "nearest_node": node_index,
        "distance_m": distances,
        "category_type": primary_category,
        "category_value": category_values,
    })
    mapping = mapping.set_index("poi_index")

    # annotate node-level POI counts for downstream scoring
    poi_counts = mapping.groupby("nearest_node").size()
    nodes_gdf["poi_count"] = nodes_gdf.index.map(poi_counts).fillna(0)

    if primary_category:
        category_counts = mapping.groupby(["nearest_node", "category_value"]).size().unstack(fill_value=0)
        # retain top categories to avoid wide tables
        top_categories = category_counts.sum().sort_values(ascending=False).head(5).index
        for category in top_categories:
            safe_category = str(category).replace(" ", "_")
            nodes_gdf[f"poi_{safe_category}_count"] = nodes_gdf.index.map(
                category_counts.get(category, pd.Series(dtype=float))
            ).fillna(0)

    return mapping, pois, nodes_gdf


def save_outputs(Gs, nodes, edges, mapping, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(Gs, out_dir / "graph.graphml")
    nodes.to_parquet(out_dir / "nodes.parquet")
    edges.to_parquet(out_dir / "edges.parquet")
    mapping.to_parquet(out_dir / "poi_node_mapping.parquet")
    print(f"Saved processed graph and mapping to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-path", required=True)
    p.add_argument("--pois-path", required=True)
    p.add_argument("--buildings-path")
    p.add_argument("--landuse-path")
    p.add_argument("--out-dir", default="data/processed")
    args = p.parse_args()
    G, pois = load_inputs(Path(args.graph_path), Path(args.pois_path))
    graph_dir = Path(args.graph_path).resolve().parent
    buildings_path = args.buildings_path or graph_dir / "buildings.geojson"
    landuse_path = args.landuse_path or graph_dir / "landuse.geojson"
    buildings = load_optional_layer(buildings_path, strict=bool(args.buildings_path))
    landuse = load_optional_layer(landuse_path, strict=bool(args.landuse_path))
    Gs = simplify_and_annotate(G)
    nodes, edges = nodes_edges_to_gdfs(Gs)
    nodes = annotate_nodes(nodes, Gs, buildings, landuse)
    mapping, pois, nodes = map_pois_to_nodes(pois, nodes)
    save_outputs(Gs, nodes, edges, mapping, Path(args.out_dir))


if __name__ == "__main__":
    main()