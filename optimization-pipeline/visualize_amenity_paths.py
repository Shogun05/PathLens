#!/usr/bin/env python3
"""
Visualize paths from selected nodes to their nearest amenities of each type.
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
import sys

# Add project root for CityDataManager
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from city_paths import CityDataManager

def load_graph(graph_path: Path) -> nx.MultiDiGraph:
    """Load graph and convert edge weights to float."""
    logging.info("Loading graph from %s", graph_path)
    G = nx.read_graphml(graph_path)
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'length' in data and isinstance(data['length'], str):
            data['length'] = float(data['length'])
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

def find_nearest_amenities(G: nx.MultiDiGraph, source_node: str, amenity_nodes: List[str], k: int = 5) -> List[Tuple[str, float, List[str]]]:
    """Find k nearest amenities of a type from source node."""
    G_undir = G.to_undirected()
    results = []
    for target in amenity_nodes:
        if target == source_node: continue
        try:
            path_length = nx.shortest_path_length(G_undir, source_node, target, weight='length')
            path = nx.shortest_path(G_undir, source_node, target, weight='length')
            results.append((target, path_length, path))
        except (nx.NetworkXNoPath, nx.NodeNotFound): continue
    results.sort(key=lambda x: x[1])
    return results[:k]

def create_path_map(G: nx.MultiDiGraph, placements: Dict[str, List[str]], amenity_type: str, sample_node: Optional[str] = None, k: int = 5) -> folium.Map:
    """Create a map showing paths to nearest neighbors."""
    amenity_nodes = placements.get(amenity_type, [])
    if not amenity_nodes: raise ValueError(f"No placements found for: {amenity_type}")
    
    source = sample_node if sample_node else amenity_nodes[0]
    source_lat, source_lon = get_node_coords(G, source)
    
    m = folium.Map(location=[source_lat, source_lon], zoom_start=15)
    folium.Marker(location=[source_lat, source_lon], tooltip=f"Source {amenity_type}", icon=folium.Icon(color='red')).add_to(m)
    
    nearest = find_nearest_amenities(G, source, amenity_nodes, k)
    colors = ['blue', 'green', 'purple', 'orange', 'darkred']
    
    for idx, (target, distance, path) in enumerate(nearest):
        path_coords = [get_node_coords(G, n) for n in path]
        color = colors[idx % len(colors)]
        folium.PolyLine(locations=path_coords, color=color, weight=3, opacity=0.7, tooltip=f"{distance:.0f}m").add_to(m)
        folium.Marker(location=path_coords[-1], tooltip=f"{amenity_type} #{idx+1}", icon=folium.Icon(color=color)).add_to(m)
    return m

def main():
    parser = argparse.ArgumentParser(description="Visualize amenity paths")
    parser.add_argument("--city", default="bangalore")
    parser.add_argument("--mode", default="ga_only")
    parser.add_argument("--amenity", required=True)
    parser.add_argument("--node", default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    cdm = CityDataManager(args.city, project_root=project_root, mode=args.mode)
    
    G = load_graph(cdm.processed_graph)
    placements = load_placements(cdm.best_candidate(args.mode))
    
    if args.amenity not in placements:
        print(f"Error: {args.amenity} not in {list(placements.keys())}")
        return
    
    m = create_path_map(G, placements, args.amenity, args.node, args.k)
    out_path = cdm.optimized_dir(args.mode) / f"{args.amenity}_paths.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
