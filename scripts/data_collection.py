#!/usr/bin/env python3
"""
Download OSM walking network and amenities for a specified place or bbox.
Saves raw outputs to out_dir.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
import yaml

ox.settings.use_cache = True
ox.settings.log_console = True


def fetch_place(place: str, network_type="walk"):
    # get the walkable network and other supporting layers
    G = ox.graph_from_place(place, network_type=network_type)
    amenity_tags = {"amenity": True, "shop": True, "leisure": True}
    pois = ox.geometries_from_place(place, amenity_tags)
    building_tags = {"building": True}
    landuse_tags = {"landuse": True}
    transit_tags = {"public_transport": True, "route": True}
    buildings = ox.geometries_from_place(place, building_tags)
    landuse = ox.geometries_from_place(place, landuse_tags)
    transit = ox.geometries_from_place(place, transit_tags)
    return G, pois, buildings, landuse, transit


def fetch_bbox(north, south, east, west, network_type="walk"):
    bbox = (north, south, east, west)
    G = ox.graph_from_bbox(north, south, east, west, network_type=network_type)
    amenity_tags = {"amenity": True, "shop": True, "leisure": True}
    pois = ox.geometries_from_bbox(north, south, east, west, amenity_tags)
    building_tags = {"building": True}
    landuse_tags = {"landuse": True}
    transit_tags = {"public_transport": True, "route": True}
    buildings = ox.geometries_from_bbox(north, south, east, west, building_tags)
    landuse = ox.geometries_from_bbox(north, south, east, west, landuse_tags)
    transit = ox.geometries_from_bbox(north, south, east, west, transit_tags)
    return G, pois, buildings, landuse, transit


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


def save_raw(G, pois, buildings, landuse, transit, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # GraphML for reproducibility
    graph_path = out_dir / "graph.graphml"
    ox.save_graphml(G, graph_path)
    
    # nodes/edges to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    
    # Normalize all list columns for parquet compatibility
    nodes = normalize_list_columns(nodes)
    edges = normalize_list_columns(edges)
    
    nodes.to_parquet(out_dir / "nodes.parquet")
    edges.to_parquet(out_dir / "edges.parquet")
    
    # POIs
    # Keep geometry + relevant tags
    pois = pois.reset_index()
    pois = normalize_list_columns(pois)
    pois.to_file(out_dir / "pois.geojson", driver="GeoJSON")

    if buildings is not None and not buildings.empty:
        buildings = normalize_list_columns(buildings.reset_index())
        buildings.to_file(out_dir / "buildings.geojson", driver="GeoJSON")

    if landuse is not None and not landuse.empty:
        landuse = normalize_list_columns(landuse.reset_index())
        landuse.to_file(out_dir / "landuse.geojson", driver="GeoJSON")

    if transit is not None and not transit.empty:
        transit = normalize_list_columns(transit.reset_index())
        transit.to_file(out_dir / "transit.geojson", driver="GeoJSON")
    
    print(f"Saved graph to {graph_path}, nodes/edges, and geospatial layers to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--place", help='Place name e.g. "Bangalore, India"')
    p.add_argument("--bbox", nargs=4, type=float, help="north south east west")
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--config", default="../config.yaml")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    place = args.place
    network_type = "walk"
    config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        if not place:
            place = config.get("place")
        network_type = config.get("network_type", network_type)

    if place:
        G, pois, buildings, landuse, transit = fetch_place(place, network_type=network_type)
    elif args.bbox:
        north, south, east, west = args.bbox
        G, pois, buildings, landuse, transit = fetch_bbox(north, south, east, west, network_type=network_type)
    else:
        raise SystemExit("Provide --place or --bbox")
    save_raw(G, pois, buildings, landuse, transit, out_dir)


if __name__ == "__main__":
    main()