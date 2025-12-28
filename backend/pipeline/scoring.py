import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
from pathlib import Path
from models.pnmlr import PNMLRModel

def run_scoring(data_dir: Path, output_file: Path, is_optimized=False, new_pois_geojson=None):
    print(f"📊 Running Scoring (Optimized={is_optimized})...")
    
    # Load Graph & Nodes
    G = ox.load_graphml(data_dir / "graph.graphml")
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    
    # Load POIs
    pois = gpd.read_file(data_dir / "pois.geojson")
    
    # If Optimized, merge new suggestions
    if is_optimized and new_pois_geojson and new_pois_geojson.exists():
        new_pois = gpd.read_file(new_pois_geojson)
        pois = pd.concat([pois, new_pois], ignore_index=True)
    
    # Project to Meters for distance calc
    nodes_m = nodes.to_crs(epsg=32643)
    pois_m = pois.to_crs(epsg=32643)
    
    # Calculate Distances (Simplified Euclidian for speed, use NetworkX for accuracy)
    for amenity in ['school', 'hospital', 'park']:
        # Filter POIs
        targets = pois_m[pois_m['amenity'] == amenity]
        if targets.empty:
            nodes[f'dist_to_{amenity}'] = 5000.0
        else:
            # Find nearest POI for every node
            # (Using KDTree logic via simple geometry distance for this demo)
            # A real production app would use nx.multi_source_dijkstra
            from scipy.spatial import cKDTree
            tree = cKDTree(list(zip(targets.geometry.x, targets.geometry.y)))
            dists, _ = tree.query(list(zip(nodes_m.geometry.x, nodes_m.geometry.y)))
            nodes[f'dist_to_{amenity}'] = dists

    # PNMLR Evaluation
    pnmlr = PNMLRModel()
    nodes['walkability'] = pnmlr.predict_score(nodes)
    
    # Save Results
    # Convert Geometry to Lat/Lon strings for JSON safety
    nodes = nodes.to_crs(epsg=4326)
    nodes['lat'] = nodes.geometry.y
    nodes['lon'] = nodes.geometry.x
    nodes = nodes.drop(columns=['geometry'])
    
    nodes.to_parquet(output_file)
    return nodes