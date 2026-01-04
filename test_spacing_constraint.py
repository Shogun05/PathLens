
import sys
import pandas as pd
import math
from pathlib import Path
sys.path.append(str(Path("optimization-pipeline").resolve()))
from hybrid_ga import default_evaluate_candidate, Candidate, GAContext, HybridGAConfig

# Mock context setup
def create_mock_context():
    # Load a small subset of nodes or mock them
    # We need coordinates for the specific conflicting nodes mentioned by user
    # 7462101991 and 9953920284
    
    # Try to load actual nodes if possible, otherwise mock with close coordinates
    try:
        import geopandas as gpd
        nodes = gpd.read_parquet("data/analysis/baseline_nodes_with_scores.parquet")
        nodes.index = nodes.index.astype(str) # Ensure string index
        
        # Ensure x and y columns exist (projected coordinates)
        if 'x' not in nodes.columns:
            nodes['x'] = nodes.geometry.x
            nodes['y'] = nodes.geometry.y
            
        print(f"Loaded {len(nodes)} nodes.")
        
        # Check if target nodes exist
        targets = ["10062966619", "10293847418"]
        found = nodes.index.isin(targets)
        if sum(found) < 2:
            print(f"Target nodes not found in dataset. Found: {nodes.index.intersection(targets)}")
            # Mocking close coordinates (e.g. 10m apart)
            data = {
                'x': [775000.0, 775010.0],
                'y': [1440000.0, 1440000.0],
                'travel_time_min': [30.0, 30.0],
                'geometry': [None, None]
            }
            nodes = gpd.GeoDataFrame(data, index=targets)
        else:
            print("Target nodes found!")
            nodes = nodes.loc[targets].copy()
            # Verify coordinates
            print("Coordinate check:")
            print(nodes[['x', 'y']])
            
    except Exception as e:
        print(f"Could not load real data: {e}")
        # Mock data fallback
        targets = ["10062966619", "10293847418"]
        data = {
            'x': [775000.0, 775010.0], # 10 meters apart
            'y': [1440000.0, 1440000.0],
            'travel_time_min': [30.0, 30.0]
        }
        import pandas as pd
        nodes = pd.DataFrame(data, index=targets)

    amenity_weights = {"school": 1.0}
    amenity_pools = {"school": ["7462101991", "9953920284"]}
    distance_columns = {"school": "dist_to_school"}
    
    # Add dummy dist_to_school column
    nodes["dist_to_school"] = 5000.0 # Far away from existing
    
    config = HybridGAConfig()
    
    context = GAContext(
        nodes=nodes,
        high_travel_nodes=nodes,
        amenity_weights=amenity_weights,
        amenity_pools=amenity_pools,
        distance_columns=distance_columns,
        config=config
    )
    
    return context

def test_spacing():
    context = create_mock_context()
    
    # Create a candidate with both nodes selected for "school"
    # This represents the "stacking" issue
    placements = {"school": ("10062966619", "10293847418")}
    candidate = Candidate(placements=placements)
    
    effects = {"nodes": context.nodes} # Mock effects
    
    results = default_evaluate_candidate(candidate, effects, context)
    
    print("\n--- Evaluation Results ---")
    print(f"Fitness: {results['fitness']}")
    print(f"Diversity Penalty: {results.get('diversity_penalty', 0)}")
    print(f"Proximity Penalty: {results.get('proximity_penalty', 0)}")
    
    # Check distances
    coords = []
    for node_id in ["10062966619", "10293847418"]:
        row = context.nodes.loc[node_id]
        coords.append((row.x, row.y))
        print(f"Node {node_id}: ({row.x}, {row.y})")
        
    dist = math.sqrt((coords[0][0] - coords[1][0])**2 + (coords[0][1] - coords[1][1])**2)
    print(f"Distance between nodes: {dist:.2f} meters")
    
    if dist < 1200 and results.get('diversity_penalty', 0) < 1.0:
        print("\n[FAIL] Penalty is too low for conflicting nodes!")
    else:
        print("\n[PASS] Penalty applied correctly.")

if __name__ == "__main__":
    test_spacing()
