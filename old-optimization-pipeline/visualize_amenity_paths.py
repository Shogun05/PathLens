#!/usr/bin/env python3
"""
Visualize paths from selected nodes to their 5 nearest amenities of each type.
This helps validate that amenity placements are actually meaningful and not too close.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import folium
import geopandas as gpd
import networkx as nx
import pandas as pd
from tqdm import tqdm


def load_graph(graph_path: Path) -> nx.MultiDiGraph:
    """Load graph and convert edge weights to float."""
    logging.info("Loading graph from %s", graph_path)
    G = nx.read_graphml(graph_path)
    
    # Fix edge weights
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'length' in data and isinstance(data['length'], str):
            data['length'] = float(data['length'])
    
    logging.info("Loaded graph with %d nodes, %d edges", len(G.nodes), len(G.edges))
    return G


def load_placements(best_candidate_path: Path) -> Dict[str, List[str]]:
    """Load placements from best_candidate.json."""
    with open(best_candidate_path) as f:
        data = json.load(f)
    
    candidate_str = data["candidate"]
    placements = {}
    
    for part in candidate_str.split("|"):
        amenity, nodes = part.split(":")
        placements[amenity] = nodes.split(",")
    
    return placements


def get_node_coords(G: nx.MultiDiGraph, node_id: str) -> Tuple[float, float]:
    """Get lat/lon coordinates of a node."""
    node_data = G.nodes[node_id]
    lat = float(node_data.get('lat', node_data.get('y')))
    lon = float(node_data.get('lon', node_data.get('x')))
    return lat, lon


def find_nearest_amenities(
    G: nx.MultiDiGraph,
    source_node: str,
    amenity_nodes: List[str],
    k: int = 5
) -> List[Tuple[str, float, List[str]]]:
    """
    Find k nearest amenities of a type from source node.
    
    Returns:
        List of (target_node, distance, path) tuples sorted by distance
    """
    G_undir = G.to_undirected()
    results = []
    
    for target in amenity_nodes:
        if target == source_node:
            continue
        
        try:
            path_length = nx.shortest_path_length(G_undir, source_node, target, weight='length')
            path = nx.shortest_path(G_undir, source_node, target, weight='length')
            results.append((target, path_length, path))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    
    # Sort by distance and return top k
    results.sort(key=lambda x: x[1])
    return results[:k]


def create_path_map(
    G: nx.MultiDiGraph,
    placements: Dict[str, List[str]],
    amenity_type: str,
    sample_node: Optional[str] = None,
    k: int = 5
) -> folium.Map:
    """
    Create a map showing paths from one amenity to its k nearest neighbors.
    
    Args:
        G: Street network graph
        placements: Amenity placements dict
        amenity_type: Type of amenity to visualize
        sample_node: Specific node to start from (if None, uses first placement)
        k: Number of nearest amenities to show
    """
    amenity_nodes = placements.get(amenity_type, [])
    
    if not amenity_nodes:
        raise ValueError(f"No placements found for amenity type: {amenity_type}")
    
    # Use specified node or first placement
    source = sample_node if sample_node else amenity_nodes[0]
    
    if source not in amenity_nodes:
        raise ValueError(f"Source node {source} not in {amenity_type} placements")
    
    # Get source coordinates for map center
    source_lat, source_lon = get_node_coords(G, source)
    
    # Create map centered on source
    m = folium.Map(
        location=[source_lat, source_lon],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Find nearest amenities
    logging.info("Finding %d nearest %s amenities from node %s", k, amenity_type, source)
    nearest = find_nearest_amenities(G, source, amenity_nodes, k)
    
    # Add source marker
    folium.Marker(
        location=[source_lat, source_lon],
        popup=f"<b>Source {amenity_type}</b><br>OSM ID: {source}",
        tooltip=f"Source {amenity_type}",
        icon=folium.Icon(color='red', icon='home', prefix='fa')
    ).add_to(m)
    
    # Color palette for paths
    colors = ['blue', 'green', 'purple', 'orange', 'darkred']
    
    # Add paths and markers for nearest amenities
    for idx, (target, distance, path) in enumerate(nearest):
        target_lat, target_lon = get_node_coords(G, target)
        color = colors[idx % len(colors)]
        
        # Create path coordinates
        path_coords = []
        for node in path:
            lat, lon = get_node_coords(G, node)
            path_coords.append([lat, lon])
        
        # Add path polyline
        folium.PolyLine(
            locations=path_coords,
            color=color,
            weight=3,
            opacity=0.7,
            popup=f"<b>Path {idx+1}</b><br>Distance: {distance:.0f}m<br>Edges: {len(path)-1}",
            tooltip=f"Path {idx+1}: {distance:.0f}m"
        ).add_to(m)
        
        # Add target marker
        folium.Marker(
            location=[target_lat, target_lon],
            popup=f"<b>{amenity_type.title()} #{idx+1}</b><br>OSM ID: {target}<br>Distance: {distance:.0f}m<br>Edges: {len(path)-1}",
            tooltip=f"{amenity_type.title()} #{idx+1} ({distance:.0f}m)",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    
    # Add summary
    distances_str = ", ".join([f"{d:.0f}m" for _, d, _ in nearest])
    summary_html = f"""
    <div style="position: fixed; top: 10px; right: 10px; z-index: 1000; 
                background: white; padding: 10px; border: 2px solid black; 
                border-radius: 5px; font-family: monospace;">
        <h4>{amenity_type.title()} Proximity Analysis</h4>
        <b>Source:</b> {source}<br>
        <b>Nearest {len(nearest)} amenities:</b><br>
        {distances_str}<br>
        <b>Min spacing:</b> {nearest[0][1]:.0f}m
    </div>
    """
    m.get_root().html.add_child(folium.Element(summary_html))
    
    logging.info("Created map with %d paths", len(nearest))
    return m


def main():
    parser = argparse.ArgumentParser(
        description="Visualize paths from an amenity to its nearest neighbors"
    )
    project_root = Path(__file__).resolve().parents[1]
    
    parser.add_argument(
        "--best-candidate",
        type=Path,
        default=project_root / "optimization-pipeline" / "runs" / "best_candidate.json",
        help="Path to best_candidate.json"
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=project_root / "data" / "processed" / "graph.graphml",
        help="Path to graph.graphml"
    )
    parser.add_argument(
        "--amenity",
        type=str,
        required=True,
        help="Amenity type to visualize (e.g., 'school', 'hospital')"
    )
    parser.add_argument(
        "--node",
        type=str,
        default=None,
        help="Specific node OSM ID to use as source (default: first placement)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest amenities to show (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML file path (default: <amenity>_paths.html)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Load data
    G = load_graph(args.graph)
    placements = load_placements(args.best_candidate)
    
    # Check amenity type
    if args.amenity not in placements:
        logging.error("Amenity type '%s' not found. Available: %s", 
                     args.amenity, ", ".join(placements.keys()))
        return
    
    # Create map
    m = create_path_map(G, placements, args.amenity, args.node, args.k)
    
    # Save
    output_path = args.output or project_root / "optimization-pipeline" / "runs" / f"{args.amenity}_paths.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logging.info("Path visualization saved to %s", output_path)


if __name__ == "__main__":
    main()
