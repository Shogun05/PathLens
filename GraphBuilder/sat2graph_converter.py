#!/usr/bin/env python3
"""
Convert Sat2Graph pixel-coordinate graphs to OSM-compatible format.

Sat2Graph outputs: {(pixel_x, pixel_y): [(neighbor_x, neighbor_y), ...]}
OSM pipeline expects: NetworkX MultiDiGraph with lat/lon coordinates

This module handles:
1. Pixel → lat/lon coordinate transformation using tile metadata
2. Graph conversion to NetworkX format with OSM-compatible attributes
3. Export to GraphML and Parquet formats
"""

import math
import pickle
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString


# Earth constants for Web Mercator
EARTH_RADIUS_M = 6378137.0


def tile_to_latlon(
    tile_x: int,
    tile_y: int,
    zoom: int,
    pixel_x: float = 0,
    pixel_y: float = 0,
    tile_size: int = 256
) -> Tuple[float, float]:
    """
    Convert Web Mercator tile coordinates + pixel offset to WGS84 lat/lon.
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        pixel_x: Pixel offset within tile (0-255 typically)
        pixel_y: Pixel offset within tile (0-255 typically)
        tile_size: Size of each tile in pixels (default 256)
    
    Returns:
        Tuple of (latitude, longitude) in WGS84 degrees
    """
    n = 2.0 ** zoom
    
    # Convert tile + pixel to normalized coordinates [0, 1]
    x_norm = (tile_x + pixel_x / tile_size) / n
    y_norm = (tile_y + pixel_y / tile_size) / n
    
    # Convert to lon/lat
    lon = x_norm * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_norm)))
    lat = math.degrees(lat_rad)
    
    return lat, lon


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two lat/lon points.
    
    Returns:
        Distance in meters
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return EARTH_RADIUS_M * c


def load_sat2graph_output(pickle_path: Path) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Load Sat2Graph pickle output.
    
    Args:
        pickle_path: Path to *_graph.p file
    
    Returns:
        Adjacency dict: {(px_x, px_y): [(neighbor_x, neighbor_y), ...]}
    """
    with open(pickle_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    Load tile metadata JSON.
    
    Expected format:
    {
        "zoom": 17,
        "base_tile_size": 256,
        "output_tile_size": 2048,
        "tiles_per_side": 8,
        "tiles": [
            {"file": "z17_x93602_y60623_composite.png", "x_range": [93602, 93609], ...}
        ]
    }
    """
    with open(metadata_path, 'r') as f:
        return json.load(f)


def parse_tile_info_from_filename(filename: str) -> Tuple[int, int, int]:
    """
    Extract zoom, x, y from filename like 'z17_x93602_y60623_composite_graph.p'
    
    Returns:
        Tuple of (zoom, tile_x, tile_y)
    """
    import re
    match = re.search(r'z(\d+)_x(\d+)_y(\d+)', filename)
    if not match:
        raise ValueError(f"Cannot parse tile info from filename: {filename}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def convert_to_networkx(
    sat_graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
    tile_x: int,
    tile_y: int,
    zoom: int,
    image_size: int = 2048,
    tiles_per_side: int = 8,
    base_tile_size: int = 256,
    node_id_offset: int = 0,
    tile_source: str = ""
) -> Tuple[nx.MultiDiGraph, Dict[Tuple[int, int], int]]:
    """
    Convert Sat2Graph adjacency dict to NetworkX MultiDiGraph.
    
    Args:
        sat_graph: Adjacency dict from Sat2Graph
        tile_x: Starting tile X coordinate
        tile_y: Starting tile Y coordinate
        zoom: Zoom level
        image_size: Size of composite image (default 2048)
        tiles_per_side: Number of tiles per side of composite (default 8)
        base_tile_size: Size of each base tile (default 256)
        node_id_offset: Offset to add to node IDs (for merging multiple tiles)
        tile_source: Source tile identifier string
    
    Returns:
        Tuple of (NetworkX graph, pixel_to_node_id mapping)
    """
    G = nx.MultiDiGraph()
    
    # Map pixel coordinates to node IDs
    pixel_to_id: Dict[Tuple[int, int], int] = {}
    node_id = node_id_offset
    
    # Total pixel span of the composite image
    total_tile_span = tiles_per_side  # Number of base tiles
    
    # First pass: create all nodes
    all_nodes = set(sat_graph.keys())
    for neighbors in sat_graph.values():
        all_nodes.update(neighbors)
    
    for px_x, px_y in all_nodes:
        if (px_x, px_y) in pixel_to_id:
            continue
            
        # Convert pixel coordinates to lat/lon
        # The pixel coordinates are within the composite image (0 to image_size)
        # Need to map back to tile coordinates
        
        # Fraction of position within the composite
        frac_x = px_y / image_size  # Note: Sat2Graph uses (row, col) = (y, x) convention
        frac_y = px_x / image_size
        
        # Pixel position in Web Mercator tile coordinates
        pixel_in_tiles_x = frac_x * total_tile_span * base_tile_size
        pixel_in_tiles_y = frac_y * total_tile_span * base_tile_size
        
        # Get lat/lon
        lat, lon = tile_to_latlon(
            tile_x, tile_y, zoom,
            pixel_x=pixel_in_tiles_x,
            pixel_y=pixel_in_tiles_y,
            tile_size=base_tile_size
        )
        
        pixel_to_id[(px_x, px_y)] = node_id
        
        G.add_node(
            node_id,
            osmid=node_id,
            x=lon,  # OSMnx convention: x = longitude
            y=lat,  # OSMnx convention: y = latitude
            geometry=Point(lon, lat),
            tile_source=tile_source,
            pixel_x=px_x,
            pixel_y=px_y
        )
        
        node_id += 1
    
    # Second pass: create edges
    edge_id = 0
    edges_added = set()
    
    for (px_x, px_y), neighbors in sat_graph.items():
        u = pixel_to_id[(px_x, px_y)]
        
        for (n_px_x, n_px_y) in neighbors:
            v = pixel_to_id[(n_px_x, n_px_y)]
            
            # Skip self-loops
            if u == v:
                continue
            
            # Skip if edge already added (for undirected representation)
            edge_key = tuple(sorted([u, v]))
            if edge_key in edges_added:
                continue
            edges_added.add(edge_key)
            
            # Get node coordinates
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            
            # Calculate edge length using haversine
            length = haversine_distance(
                u_data['y'], u_data['x'],
                v_data['y'], v_data['x']
            )
            
            # Create edge geometry
            geometry = LineString([
                (u_data['x'], u_data['y']),
                (v_data['x'], v_data['y'])
            ])
            
            # Add edges in both directions (bidirectional road)
            for start, end in [(u, v), (v, u)]:
                G.add_edge(
                    start, end,
                    key=0,
                    osmid=edge_id,
                    length=length,
                    geometry=geometry,
                    highway="unclassified",
                    oneway=False,
                    name="",
                    source="satellite",
                    tile_source=tile_source
                )
            
            edge_id += 1
    
    return G, pixel_to_id


def graph_to_gdfs(G: nx.MultiDiGraph) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Convert NetworkX graph to GeoDataFrames for nodes and edges.
    
    Matches OSM structure: osmid is the index, not a column.
    
    Returns:
        Tuple of (nodes_gdf, edges_gdf)
    """
    # Nodes - build with node_id as a column first, then set as index
    nodes_data = []
    for node_id, data in G.nodes(data=True):
        nodes_data.append({
            'node_id': node_id,  # Temporary column, will become index
            'x': data['x'],
            'y': data['y'],
            'geometry': data['geometry'],
            'tile_source': data.get('tile_source', ''),
            'street_count': G.degree(node_id)
        })
    
    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs="EPSG:4326")
    # Set node_id as index and rename to 'osmid' to match OSM structure
    nodes_gdf = nodes_gdf.set_index('node_id')
    nodes_gdf.index.name = 'osmid'
    
    # Edges
    edges_data = []
    for u, v, key, data in G.edges(keys=True, data=True):
        edges_data.append({
            'u': u,
            'v': v,
            'key': key,
            'osmid': data.get('osmid', 0),
            'length': data.get('length', 0),
            'geometry': data.get('geometry'),
            'highway': data.get('highway', 'unclassified'),
            'oneway': data.get('oneway', False),
            'name': data.get('name', ''),
            'source': data.get('source', 'satellite'),
            'tile_source': data.get('tile_source', '')
        })
    
    edges_gdf = gpd.GeoDataFrame(edges_data, crs="EPSG:4326")
    edges_gdf = edges_gdf.set_index(['u', 'v', 'key'])
    
    return nodes_gdf, edges_gdf


def add_osmnx_metadata(G: nx.MultiDiGraph, crs: str = "EPSG:4326") -> nx.MultiDiGraph:
    """
    Add OSMnx-compatible graph metadata.
    """
    G.graph['crs'] = crs
    G.graph['simplified'] = False
    G.graph['created_by'] = 'sat2graph_converter'
    G.graph['source'] = 'satellite_imagery'
    return G


def save_outputs(
    G: nx.MultiDiGraph,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    output_dir: Path
) -> None:
    """
    Save graph and GeoDataFrames to files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save GraphML
    # Note: OSMnx uses ox.save_graphml, but we need to handle geometry serialization
    # For now, save a simplified version
    G_simple = G.copy()
    for node_id in G_simple.nodes():
        if 'geometry' in G_simple.nodes[node_id]:
            geom = G_simple.nodes[node_id]['geometry']
            G_simple.nodes[node_id]['geometry'] = geom.wkt
    
    for u, v, k in G_simple.edges(keys=True):
        if 'geometry' in G_simple.edges[u, v, k]:
            geom = G_simple.edges[u, v, k]['geometry']
            G_simple.edges[u, v, k]['geometry'] = geom.wkt
    
    nx.write_graphml(G_simple, output_dir / "graph.graphml")
    
    # Save Parquet files
    # Convert geometry to WKT for parquet storage (will be read back by geopandas)
    nodes_gdf.to_parquet(output_dir / "nodes.parquet")
    edges_gdf.to_parquet(output_dir / "edges.parquet")
    
    print(f"✅ Saved outputs to {output_dir}")
    print(f"   - graph.graphml: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   - nodes.parquet: {len(nodes_gdf)} rows")
    print(f"   - edges.parquet: {len(edges_gdf)} rows")


def convert_single_tile(
    pickle_path: Path,
    metadata: Dict[str, Any],
    node_id_offset: int = 0
) -> Tuple[nx.MultiDiGraph, Dict[Tuple[int, int], int], int]:
    """
    Convert a single tile's Sat2Graph output to NetworkX graph.
    
    Args:
        pickle_path: Path to *_graph.p file
        metadata: Tile metadata dict
        node_id_offset: Starting node ID
    
    Returns:
        Tuple of (graph, pixel_to_id mapping, next_node_id_offset)
    """
    # Load graph
    sat_graph = load_sat2graph_output(pickle_path)
    
    if not sat_graph:
        print(f"⚠️ Empty graph in {pickle_path}")
        return nx.MultiDiGraph(), {}, node_id_offset
    
    # Parse tile info from filename
    zoom, tile_x, tile_y = parse_tile_info_from_filename(pickle_path.name)
    
    # Get metadata values
    image_size = metadata.get('output_tile_size', 2048)
    tiles_per_side = metadata.get('tiles_per_side', 8)
    base_tile_size = metadata.get('base_tile_size', 256)
    
    tile_source = f"z{zoom}_x{tile_x}_y{tile_y}"
    
    print(f"Converting {tile_source}: {len(sat_graph)} nodes...")
    
    # Convert to NetworkX
    G, pixel_to_id = convert_to_networkx(
        sat_graph,
        tile_x=tile_x,
        tile_y=tile_y,
        zoom=zoom,
        image_size=image_size,
        tiles_per_side=tiles_per_side,
        base_tile_size=base_tile_size,
        node_id_offset=node_id_offset,
        tile_source=tile_source
    )
    
    next_offset = node_id_offset + len(pixel_to_id)
    
    return G, pixel_to_id, next_offset


def main():
    parser = argparse.ArgumentParser(
        description="Convert Sat2Graph output to OSM-compatible format"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to *_graph.p pickle file"
    )
    parser.add_argument(
        "--metadata", "-m",
        type=Path,
        required=True,
        help="Path to metadata.json"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("converted_graph"),
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Load metadata
    metadata = load_metadata(args.metadata)
    
    # Convert single tile
    G, pixel_to_id, _ = convert_single_tile(args.input, metadata)
    
    if G.number_of_nodes() == 0:
        print("❌ No nodes in converted graph. Check input file.")
        return
    
    # Add OSMnx metadata
    G = add_osmnx_metadata(G)
    
    # Convert to GeoDataFrames
    nodes_gdf, edges_gdf = graph_to_gdfs(G)
    
    # Save outputs
    save_outputs(G, nodes_gdf, edges_gdf, args.output)


if __name__ == "__main__":
    main()
