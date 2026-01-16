#!/usr/bin/env python3
"""
Merge multiple Sat2Graph tile graphs into a single unified graph.

Handles:
1. Loading multiple tile graphs
2. Spatial boundary node detection
3. Cross-tile node merging (snapping nearby nodes)
4. Graph topology cleanup
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree

from sat2graph_converter import (
    load_sat2graph_output,
    load_metadata,
    parse_tile_info_from_filename,
    convert_to_networkx,
    graph_to_gdfs,
    add_osmnx_metadata,
    save_outputs,
    haversine_distance
)


def find_all_tile_graphs(input_dir: Path) -> List[Path]:
    """
    Find all *_graph.p files in directory.
    """
    graphs = list(input_dir.glob("*_graph.p"))
    print(f"Found {len(graphs)} tile graph(s) in {input_dir}")
    return sorted(graphs)


def get_node_bounds(G: nx.MultiDiGraph) -> Tuple[float, float, float, float]:
    """
    Get geographic bounds of all nodes in graph.
    
    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon)
    """
    lats = [data['y'] for _, data in G.nodes(data=True)]
    lons = [data['x'] for _, data in G.nodes(data=True)]
    
    return min(lats), max(lats), min(lons), max(lons)


def find_boundary_nodes(
    G: nx.MultiDiGraph,
    threshold_fraction: float = 0.02
) -> Set[int]:
    """
    Find nodes near the boundary of the graph's geographic extent.
    
    Args:
        G: NetworkX graph
        threshold_fraction: Fraction of extent to consider as boundary (default 2%)
    
    Returns:
        Set of node IDs near boundaries
    """
    if G.number_of_nodes() == 0:
        return set()
    
    min_lat, max_lat, min_lon, max_lon = get_node_bounds(G)
    
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    
    lat_threshold = lat_range * threshold_fraction
    lon_threshold = lon_range * threshold_fraction
    
    boundary_nodes = set()
    
    for node_id, data in G.nodes(data=True):
        lat, lon = data['y'], data['x']
        
        # Check if near any boundary
        if (lat - min_lat < lat_threshold or
            max_lat - lat < lat_threshold or
            lon - min_lon < lon_threshold or
            max_lon - lon < lon_threshold):
            boundary_nodes.add(node_id)
    
    return boundary_nodes


def merge_graphs(
    graphs: List[nx.MultiDiGraph],
    snap_threshold_m: float = 10.0
) -> nx.MultiDiGraph:
    """
    Merge multiple tile graphs into a single unified graph.
    
    Nodes within snap_threshold_m meters of each other are merged.
    
    Args:
        graphs: List of NetworkX graphs to merge
        snap_threshold_m: Distance threshold in meters for merging nodes
    
    Returns:
        Merged NetworkX graph
    """
    if not graphs:
        return nx.MultiDiGraph()
    
    if len(graphs) == 1:
        return graphs[0].copy()
    
    print(f"\n🔗 Merging {len(graphs)} graphs with {snap_threshold_m}m snap threshold...")
    
    # Collect all nodes from all graphs
    all_nodes = []  # List of (graph_idx, node_id, lat, lon, data)
    
    for g_idx, G in enumerate(graphs):
        for node_id, data in G.nodes(data=True):
            all_nodes.append((g_idx, node_id, data['y'], data['x'], dict(data)))
    
    print(f"   Total nodes before merge: {len(all_nodes)}")
    
    # Build KD-tree for spatial lookup
    coords = np.array([(node[2], node[3]) for node in all_nodes])  # lat, lon
    
    # Convert threshold to approximate degrees (rough estimate)
    # 1 degree ≈ 111km at equator
    threshold_deg = snap_threshold_m / 111000.0
    
    tree = cKDTree(coords)
    
    # Find node clusters (nodes that should be merged)
    # Use Union-Find for efficient clustering
    parent = list(range(len(all_nodes)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Query pairs within threshold
    pairs = tree.query_pairs(threshold_deg)
    
    for i, j in pairs:
        # Verify with haversine distance
        lat1, lon1 = all_nodes[i][2], all_nodes[i][3]
        lat2, lon2 = all_nodes[j][2], all_nodes[j][3]
        
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        if dist <= snap_threshold_m:
            union(i, j)
    
    # Group nodes by cluster
    clusters: Dict[int, List[int]] = {}
    for i in range(len(all_nodes)):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
    
    merged_count = sum(1 for c in clusters.values() if len(c) > 1)
    print(f"   Found {merged_count} merge clusters")
    
    # Create merged graph
    merged_G = nx.MultiDiGraph()
    
    # Map (graph_idx, old_node_id) -> new_node_id
    node_mapping: Dict[Tuple[int, int], int] = {}
    new_node_id = 1
    
    # Create merged nodes
    for cluster_nodes in clusters.values():
        # Average the positions of clustered nodes
        lats = [all_nodes[i][2] for i in cluster_nodes]
        lons = [all_nodes[i][3] for i in cluster_nodes]
        
        avg_lat = sum(lats) / len(lats)
        avg_lon = sum(lons) / len(lons)
        
        # Use first node's data as template
        first_idx = cluster_nodes[0]
        template_data = all_nodes[first_idx][4].copy()
        
        # Update with averaged position
        template_data['y'] = avg_lat
        template_data['x'] = avg_lon
        template_data['geometry'] = Point(avg_lon, avg_lat)
        template_data['osmid'] = new_node_id
        
        # Track source tiles if merging across tiles
        if len(cluster_nodes) > 1:
            tiles = set()
            for i in cluster_nodes:
                ts = all_nodes[i][4].get('tile_source', '')
                if ts:
                    tiles.add(ts)
            if tiles:
                template_data['tile_source'] = ','.join(sorted(tiles))
        
        merged_G.add_node(new_node_id, **template_data)
        
        # Map all cluster nodes to this new node
        for i in cluster_nodes:
            g_idx, old_id = all_nodes[i][0], all_nodes[i][1]
            node_mapping[(g_idx, old_id)] = new_node_id
        
        new_node_id += 1
    
    print(f"   Nodes after merge: {merged_G.number_of_nodes()}")
    
    # Add edges from all graphs
    edges_added: Set[Tuple[int, int]] = set()
    edge_id = 1
    
    for g_idx, G in enumerate(graphs):
        for u, v, key, data in G.edges(keys=True, data=True):
            new_u = node_mapping[(g_idx, u)]
            new_v = node_mapping[(g_idx, v)]
            
            # Skip self-loops created by merging
            if new_u == new_v:
                continue
            
            # Skip duplicate edges
            edge_key = tuple(sorted([new_u, new_v]))
            if edge_key in edges_added:
                continue
            edges_added.add(edge_key)
            
            # Recalculate edge properties based on new node positions
            u_data = merged_G.nodes[new_u]
            v_data = merged_G.nodes[new_v]
            
            length = haversine_distance(
                u_data['y'], u_data['x'],
                v_data['y'], v_data['x']
            )
            
            from shapely.geometry import LineString
            geometry = LineString([
                (u_data['x'], u_data['y']),
                (v_data['x'], v_data['y'])
            ])
            
            edge_data = dict(data)
            edge_data['osmid'] = edge_id
            edge_data['length'] = length
            edge_data['geometry'] = geometry
            
            # Add bidirectional edges
            merged_G.add_edge(new_u, new_v, key=0, **edge_data)
            merged_G.add_edge(new_v, new_u, key=0, **edge_data)
            
            edge_id += 1
    
    print(f"   Edges after merge: {merged_G.number_of_edges()}")
    
    return merged_G


def refine_merged_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Clean up merged graph:
    - Remove isolated nodes (degree 0)
    - Remove very short edges (< 1m)
    """
    print("\n🧹 Refining merged graph...")
    
    initial_nodes = G.number_of_nodes()
    initial_edges = G.number_of_edges()
    
    # Remove isolated nodes
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)
    print(f"   Removed {len(isolated)} isolated nodes")
    
    # Remove very short edges (likely artifacts)
    short_edges = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if data.get('length', 0) < 1.0:  # Less than 1 meter
            short_edges.append((u, v, k))
    
    for u, v, k in short_edges:
        if G.has_edge(u, v, k):
            G.remove_edge(u, v, k)
    print(f"   Removed {len(short_edges)} very short edges")
    
    # Update street_count for remaining nodes
    for node_id in G.nodes():
        G.nodes[node_id]['street_count'] = G.degree(node_id)
    
    print(f"   Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Reduction: {initial_nodes - G.number_of_nodes()} nodes, "
          f"{initial_edges - G.number_of_edges()} edges")
    
    return G


def merge_tile_graphs(
    input_dir: Path,
    metadata_path: Path,
    snap_threshold_m: float = 10.0
) -> nx.MultiDiGraph:
    """
    Main function to merge all tile graphs in a directory.
    
    Args:
        input_dir: Directory containing *_graph.p files
        metadata_path: Path to metadata.json
        snap_threshold_m: Snap threshold in meters
    
    Returns:
        Merged NetworkX graph
    """
    # Find all tile graphs
    graph_files = find_all_tile_graphs(input_dir)
    
    if not graph_files:
        print("❌ No graph files found!")
        return nx.MultiDiGraph()
    
    # Load metadata
    metadata = load_metadata(metadata_path)
    
    # Convert each tile
    graphs = []
    node_id_offset = 1
    
    for graph_file in graph_files:
        from sat2graph_converter import convert_single_tile
        G, pixel_to_id, next_offset = convert_single_tile(
            graph_file, metadata, node_id_offset
        )
        
        if G.number_of_nodes() > 0:
            graphs.append(G)
            node_id_offset = next_offset
    
    if not graphs:
        print("❌ No valid graphs to merge!")
        return nx.MultiDiGraph()
    
    # Merge graphs
    if len(graphs) == 1:
        merged_G = graphs[0]
    else:
        merged_G = merge_graphs(graphs, snap_threshold_m)
    
    # Refine merged graph
    merged_G = refine_merged_graph(merged_G)
    
    # Add OSMnx metadata
    merged_G = add_osmnx_metadata(merged_G)
    
    return merged_G


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple Sat2Graph tile graphs into one"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Directory containing *_graph.p files"
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
        default=Path("merged_graph"),
        help="Output directory"
    )
    parser.add_argument(
        "--snap-threshold", "-s",
        type=float,
        default=10.0,
        help="Snap threshold in meters for merging boundary nodes (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Merge graphs
    merged_G = merge_tile_graphs(
        args.input_dir,
        args.metadata,
        args.snap_threshold
    )
    
    if merged_G.number_of_nodes() == 0:
        print("❌ Merged graph is empty!")
        return
    
    # Convert to GeoDataFrames
    nodes_gdf, edges_gdf = graph_to_gdfs(merged_G)
    
    # Save outputs
    save_outputs(merged_G, nodes_gdf, edges_gdf, args.output)


if __name__ == "__main__":
    main()
