#!/usr/bin/env python3
"""
Download OSM walking network and amenities for a specified place or bbox.
Saves raw outputs to out_dir.
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
import yaml

# Add project root to path for CityDataManager
sys.path.insert(0, str(Path(__file__).parent.parent))
from city_paths import CityDataManager

project_root = Path(__file__).parent.parent
ox.settings.use_cache = True
ox.settings.cache_folder = str(project_root / "data" / "cache")
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


def fetch_place(place: str, network_type="walk"):
    # get the walkable network and other supporting layers
    print(f"Fetching street network for {place}...")
    G = ox.graph_from_place(place, network_type=network_type)
    
    print(f"Fetching POIs (amenity, shop, leisure)...")
    amenity_tags = {"amenity": True, "shop": True, "leisure": True}
    try:
        pois = ox.features_from_place(place, amenity_tags)
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch POIs: {e}")
        pois = gpd.GeoDataFrame()
    
    # Optional layers - continue if they fail
    print(f"Fetching buildings (may take a while)...")
    building_tags = {"building": True}
    try:
        buildings = ox.features_from_place(place, building_tags)
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch buildings: {e}")
        buildings = gpd.GeoDataFrame()
    
    print(f"Fetching land use data...")
    landuse_tags = {"landuse": True}
    try:
        landuse = ox.features_from_place(place, landuse_tags)
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch land use: {e}")
        landuse = gpd.GeoDataFrame()
    
    print(f"Fetching transit data...")
    transit_tags = {"public_transport": True, "route": True}
    try:
        transit = ox.features_from_place(place, transit_tags)
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch transit: {e}")
        transit = gpd.GeoDataFrame()
    
    return G, pois, buildings, landuse, transit


def fetch_bbox(north, south, east, west, network_type="walk"):
    bbox = (north, south, east, west)
    print(f"Fetching street network for bbox...")
    G = ox.graph_from_bbox(north, south, east, west, network_type=network_type)
    
    print(f"Fetching POIs (amenity, shop, leisure)...")
    amenity_tags = {"amenity": True, "shop": True, "leisure": True}
    try:
        pois = ox.features_from_bbox(bbox=(north, south, east, west), tags=amenity_tags)
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch POIs: {e}")
        pois = gpd.GeoDataFrame()
    
    # Optional layers - continue if they fail
    print(f"Fetching buildings (may take a while)...")
    building_tags = {"building": True}
    try:
        buildings = ox.features_from_bbox(bbox=(north, south, east, west), tags=building_tags)
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch buildings: {e}")
        buildings = gpd.GeoDataFrame()
    
    print(f"Fetching land use data...")
    landuse_tags = {"landuse": True}
    try:
        landuse = ox.features_from_bbox(bbox=(north, south, east, west), tags=landuse_tags)
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch land use: {e}")
        landuse = gpd.GeoDataFrame()
    
    print(f"Fetching transit data...")
    transit_tags = {"public_transport": True, "route": True}
    try:
        transit = ox.features_from_bbox(bbox=(north, south, east, west), tags=transit_tags)
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch transit: {e}")
        transit = gpd.GeoDataFrame()
    
    return G, pois, buildings, landuse, transit


def save_raw(G, pois, buildings, landuse, transit, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving street network graph...")
    # GraphML for reproducibility
    graph_path = out_dir / "graph.graphml"
    ox.save_graphml(G, graph_path)
    
    print("Saving nodes and edges...")
    # nodes/edges to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    
    # Normalize all list columns for parquet compatibility
    nodes = normalize_list_columns(nodes)
    edges = normalize_list_columns(edges)
    
    nodes.to_parquet(out_dir / "nodes.parquet")
    edges.to_parquet(out_dir / "edges.parquet")
    
    # POIs
    if pois is not None and not pois.empty:
        print(f"Saving {len(pois)} POIs...")
        pois = pois.reset_index()
        pois = normalize_list_columns(pois)
        pois.to_parquet(out_dir / "pois.parquet")
    else:
        print("⚠️  No POI data to save (will use converted amenities instead)")

    # Buildings
    if buildings is not None and not buildings.empty:
        print(f"Saving {len(buildings)} buildings...")
        buildings = buildings.reset_index()
        buildings = normalize_list_columns(buildings)
        buildings.to_parquet(out_dir / "buildings.parquet")
    else:
        print("⚠️  No building data available")

    # Landuse
    if landuse is not None and not landuse.empty:
        print(f"Saving {len(landuse)} land use features...")
        landuse = landuse.reset_index()
        landuse = normalize_list_columns(landuse)
        landuse.to_parquet(out_dir / "landuse.parquet")
    else:
        print("⚠️  No land use data available")

    # Transit
    if transit is not None and not transit.empty:
        print(f"Saving {len(transit)} transit features...")
        transit = transit.reset_index()
        transit = normalize_list_columns(transit)
        transit.to_parquet(out_dir / "transit.parquet")
    else:
        print("⚠️  No transit data available")
    
    print(f"\n✅ Saved all data to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default=None, help="City name for path resolution (e.g., 'bangalore', 'mumbai')")
    p.add_argument("--place", help='Place name e.g. "Bangalore, India" (auto-resolved from city if not provided)')
    p.add_argument("--bbox", nargs=4, type=float, help="north south east west")
    p.add_argument("--out-dir", default=None, help="Output directory (auto-resolved from city if not provided)")
    p.add_argument("--config", default=None, help="Config file (auto-resolved from city if not provided)")
    args = p.parse_args()
    
    # Initialize CityDataManager if city specified
    project_root = Path(__file__).parent.parent
    config = {}
    
    if args.city:
        cdm = CityDataManager(args.city, project_root=project_root)
        print(f"Using city-specific paths for: {cdm.city}")
        
        if not args.out_dir:
            args.out_dir = str(cdm.raw_dir)
        if not args.config:
            args.config = str(cdm.config)
    else:
        # Legacy mode
        if not args.out_dir:
            args.out_dir = "../data/raw/osm"
        if not args.config:
            args.config = "../config.yaml"
    
    out_dir = Path(args.out_dir)
    place = args.place
    network_type = "walk"
    
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        if not place:
            place = config.get("place")
        network_type = config.get("network_type", network_type)

    print(f"Output directory: {out_dir}")
    print(f"Place: {place}")

    if place:
        G, pois, buildings, landuse, transit = fetch_place(place, network_type=network_type)
    elif args.bbox:
        north, south, east, west = args.bbox
        G, pois, buildings, landuse, transit = fetch_bbox(north, south, east, west, network_type=network_type)
    else:
        raise SystemExit("Provide --city, --place, or --bbox")
    save_raw(G, pois, buildings, landuse, transit, out_dir)


if __name__ == "__main__":
    main()