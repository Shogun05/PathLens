import random
import pandas as pd
import geopandas as gpd
from pathlib import Path
from models.pnmlr import PNMLRModel
import osmnx as ox
import numpy as np

def get_nearest_street_name(G, lat, lon):
    try:
        u, v, key = ox.nearest_edges(G, lon, lat)
        edge_data = G.get_edge_data(u, v, key)
        if 'name' in edge_data:
            name = edge_data['name']
            return name[0] if isinstance(name, list) else name
        return "Unnamed Road"
    except: return "Unknown Location"

def run_optimization_algorithm(data_dir: Path, baseline_file: Path, constraints: dict):
    print("🧬 Running Aggressive Optimization (Greedy Mode)...")
    nodes = pd.read_parquet(baseline_file)
    G = ox.load_graphml(data_dir / "graph.graphml")
    
    # SETUP TARGETS
    target_count = constraints.get('max_amenities', 10)
    allowed_types = []
    if constraints.get('add_schools'): allowed_types.append('school')
    if constraints.get('add_hospitals'): allowed_types.append('hospital')
    if constraints.get('add_parks'): allowed_types.append('park')
    if not allowed_types: allowed_types = ['school', 'park']

    # 1. Identify "Desperate" Zones (Lowest 100 scores)
    # We focus on the absolute worst areas to maximize impact
    bad_nodes = nodes.sort_values('walkability').head(100)
    
    current_suggestions = []
    
    # 2. AGGRESSIVE LOOP
    # We will loop exactly 'target_count' times. 
    # In each loop, we force an addition that improves the score.
    
    for i in range(target_count):
        best_candidate = None
        best_improvement = -1
        
        # Sample 20 random bad spots to test
        test_spots = bad_nodes.sample(min(len(bad_nodes), 20))
        
        for _, loc in test_spots.iterrows():
            for amenity_type in allowed_types:
                
                # Check Spacing (Simple Distance Check)
                # Don't put a school within 300m of another NEW school
                too_close = False
                for existing in current_suggestions:
                    if existing['type'] == amenity_type:
                        dist = np.sqrt((loc['lon']-existing['x'])**2 + (loc['lat']-existing['y'])**2) * 111000 # Rough meters
                        if dist < 300: 
                            too_close = True
                            break
                if too_close: continue

                # HEURISTIC: Estimate Impact
                # "If I put a {amenity} here, how many bad nodes get help?"
                # Simple logic: It helps anyone within 1km (approx 0.01 degrees)
                impact_radius = 0.01 
                
                # Count how many bad nodes are close to this new spot
                impact_score = ((bad_nodes['lon'] - loc['lon'])**2 + (bad_nodes['lat'] - loc['lat'])**2 < impact_radius**2).sum()
                
                # Boost score for type variety
                type_count = sum(1 for s in current_suggestions if s['type'] == amenity_type)
                impact_score = impact_score / (1 + type_count) # Diminishing returns for same type
                
                if impact_score > best_improvement:
                    best_improvement = impact_score
                    best_candidate = {
                        "amenity": amenity_type, # Standardize key to 'amenity'
                        "type": amenity_type,
                        "lat": loc['lat'],
                        "lon": loc['lon'],
                        "x": loc['lon'],
                        "y": loc['lat']
                    }
        
        # Force add the best option found this round
        if best_candidate:
            print(f"   [+] Adding {best_candidate['type']} (Impact: {best_improvement})")
            current_suggestions.append(best_candidate)
        else:
            # If we couldn't find a perfect spot, just pick a random type at a random bad node
            # This ensures we ALWAYS meet the requested number
            fallback = test_spots.iloc[0]
            t = random.choice(allowed_types)
            print(f"   [+] Force Adding {t} (Fallback)")
            current_suggestions.append({
                "amenity": t, "type": t,
                "lat": fallback['lat'], "lon": fallback['lon'],
                "x": fallback['lon'], "y": fallback['lat']
            })

    # 3. EXPORT
    final_output = []
    for item in current_suggestions:
        item["address"] = get_nearest_street_name(G, item['lat'], item['lon'])
        item["id"] = f"sug_{random.randint(10000,99999)}"
        final_output.append(item)

    gdf = gpd.GeoDataFrame(final_output, crs="EPSG:4326", geometry=gpd.points_from_xy([s['lon'] for s in final_output], [s['lat'] for s in final_output]))
    gdf.to_file(data_dir / "suggestions.geojson", driver="GeoJSON")