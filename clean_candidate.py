
import json
import math
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd

def clean_candidate(candidate_path, nodes_path, min_spacing=1200.0):
    print(f"Loading candidate from {candidate_path}")
    with open(candidate_path, 'r') as f:
        data = json.load(f)
        
    # Parse candidate string if it's a string
    candidate_str = None
    original_candidate_is_string = False
    
    # Determine the actual candidate dictionary or string
    candidate_data = data
    if isinstance(data, dict):
        if "best_overall" in data and "candidate" in data["best_overall"]:
            candidate_data = data["best_overall"]["candidate"]
        elif "candidate" in data:
            candidate_data = data["candidate"]

    if isinstance(candidate_data, str):
        candidate_str = candidate_data
        original_candidate_is_string = True
    elif isinstance(candidate_data, dict):
        # If it's a dict, we'll extract placements directly
        pass # No need to set candidate_str

    placements = {}
    if original_candidate_is_string:
        print(f"Parsing candidate string: {candidate_str[:50]}...")
        parts = candidate_str.split('|')
        for part in parts:
            if ':' in part:
                amenity, ids = part.split(':', 1)
                placements[amenity] = ids.split(',')
    elif isinstance(candidate_data, dict) and "placements" in candidate_data:
         placements = candidate_data["placements"]
    else:
        # Fallback for raw placements dict directly at the top level
        # This handles cases like {"school": [...], "hospital": [...]}
        if any(k in candidate_data for k in ["school", "hospital", "clinic", "pharmacy"]): # Add other amenity types if needed
            placements = candidate_data
        else:
            print(f"Unknown JSON structure. Expected 'candidate' string, 'placements' dict, or raw amenity dict. Keys: {list(data.keys())}")
            return

    print(f"Loading nodes from {nodes_path}")
    nodes = gpd.read_parquet(nodes_path)
    nodes.index = nodes.index.astype(str)
    
    # Ensure x/y
    if 'x' not in nodes.columns:
        nodes['x'] = nodes.geometry.x
        nodes['y'] = nodes.geometry.y
        
    cleaned_placements = {}
    total_removed = 0
    
    for amenity, node_ids in placements.items():
        if not node_ids:
            continue
            
        # Filter out empty strings if any
        node_ids = [n for n in node_ids if n]
        
        print(f"Processing {amenity} ({len(node_ids)} nodes)...")
        
        # Get coordinates
        valid_nodes = []
        for nid in node_ids:
            if nid in nodes.index:
                valid_nodes.append({
                    "id": nid,
                    "x": nodes.loc[nid, "x"],
                    "y": nodes.loc[nid, "y"]
                })
        
        # Greedy filtering
        accepted = []
        rejected = 0
        
        # We process in order. 
        # Ideally we should sort by some priority (e.g. travel_time or local score)
        # But looking up original score is complex here.
        # Assuming the GA output order is somewhat meaningful or random.
        
        for node in valid_nodes:
            conflict = False
            for acc in accepted:
                dist = math.sqrt((node['x'] - acc['x'])**2 + (node['y'] - acc['y'])**2)
                if dist < min_spacing:
                    conflict = True
                    # print(f"  Constraint violation: {node['id']} is {dist:.1f}m from {acc['id']}")
                    break
            
            if not conflict:
                accepted.append(node)
            else:
                rejected += 1
                
        cleaned_placements[amenity] = [n['id'] for n in accepted]
        total_removed += rejected
        print(f"  {amenity}: Kept {len(accepted)}, Removed {rejected}")

    # Reconstruct string if original was a string
    if original_candidate_is_string:
        ordered = sorted(cleaned_placements.items())
        new_candidate_str = "|".join(f"{amenity}:{','.join(sorted(ids))}" for amenity, ids in ordered)
    
    # Save back
    if original_candidate_is_string:
        # Update the candidate string in the appropriate location within 'data'
        if "best_overall" in data and "candidate" in data["best_overall"]:
            data["best_overall"]["candidate"] = new_candidate_str
        elif "candidate" in data:
            data["candidate"] = new_candidate_str
        
        # Also clean metrics placements counts if they exist
        if "metrics" in data and "placements" in data["metrics"]:
             for amenity, ids in cleaned_placements.items():
                 data["metrics"]["placements"][amenity] = len(ids)
    else:
        # If it was a direct dict, update its placements
        if "best_overall" in data and "candidate" in data["best_overall"] and isinstance(data["best_overall"]["candidate"], dict):
            data["best_overall"]["candidate"]["placements"] = cleaned_placements
        elif "placements" in data:
            data["placements"] = cleaned_placements
        else: # Raw amenity dict at top level
            data.update(cleaned_placements)

    # Backup original
    backup_path = str(candidate_path) + ".bak"
    with open(backup_path, 'w') as f:
        json.dump(data, f, indent=2) 
    print(f"Backed up original to {backup_path}")
        
    with open(candidate_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved cleaned candidate to {candidate_path}")
    print(f"Total nodes removed: {total_removed}")

if __name__ == "__main__":
    base_dir = Path("data/optimization/runs")
    candidate_file = base_dir / "best_candidate.json"
    
    # If best_candidate doesn't exist, try checkpoint
    if not candidate_file.exists():
        candidate_file = base_dir / "checkpoint.json"
        
    nodes_file = Path("data/analysis/baseline_nodes_with_scores.parquet")
    
    clean_candidate(candidate_file, nodes_file)
