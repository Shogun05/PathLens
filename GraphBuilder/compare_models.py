#!/usr/bin/env python3.14
"""
Compare all 7 Sat2Graph models on the same location.
Generates outputs and visualizations for side-by-side comparison.
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path

# Model configurations
MODELS = {
    0: {"name": "Sat2Graph-V1 80-City US (1m GSD, DLA 2x)", "type": "graph"},
    1: {"name": "Sat2Graph-V1 20-City US Paper (1m GSD)", "type": "graph"},
    2: {"name": "Sat2Graph-V2 20-City US (50cm GSD)", "type": "graph"},
    3: {"name": "Sat2Graph-V2 80-City Global (50cm GSD)", "type": "graph"},
    4: {"name": "UNet Segmentation", "type": "segmentation"},
    5: {"name": "DeepRoadMapper Segmentation", "type": "segmentation"},
    6: {"name": "JointOrientation Segmentation", "type": "segmentation"},
}

def run_model_comparison(lat, lon, tile_size, output_dir):
    """Run all 7 models on the same location"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print(f"\n{'='*70}")
    print(f"RUNNING MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"Location: ({lat}, {lon})")
    print(f"Tile size: {tile_size}m")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    scripts_dir = Path(__file__).parent / "docker" / "scripts"
    os.chdir(scripts_dir)
    
    for model_id, model_info in MODELS.items():
        print(f"\n[Model {model_id}] {model_info['name']}")
        print(f"  Type: {model_info['type']}")
        
        output_json = output_dir / f"model_{model_id}_result.json"
        
        cmd = [
            "python3.14",
            "infer_mapbox_input.py",
            "-lat", str(lat),
            "-lon", str(lon),
            "-tile_size", str(tile_size),
            "-model_id", str(model_id),
            "-output", str(output_json)
        ]
        
        print(f"  Running inference...")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"  ✓ Success ({elapsed:.1f}s)")
                
                # Parse result to get task ID
                if "Task ID:" in result.stdout:
                    task_id = result.stdout.split("Task ID:")[-1].strip().split()[0]
                    results[model_id] = {
                        "status": "success",
                        "time": elapsed,
                        "output": str(output_json),
                        "task_id": task_id,
                        "url": f"http://localhost:8010/t{task_id}/"
                    }
                else:
                    results[model_id] = {
                        "status": "success",
                        "time": elapsed,
                        "output": str(output_json)
                    }
                
                # Generate visualization if output exists
                if output_json.exists():
                    vis_output = output_dir / f"model_{model_id}_visualization.png"
                    vis_cmd = [
                        "python3.14",
                        "vis.py",
                        str(tile_size),
                        str(output_json),
                        str(vis_output)
                    ]
                    
                    vis_result = subprocess.run(vis_cmd, capture_output=True, text=True)
                    if vis_result.returncode == 0:
                        print(f"  ✓ Visualization saved: {vis_output.name}")
                        results[model_id]["visualization"] = str(vis_output)
                    else:
                        print(f"  ⚠ Visualization failed: {vis_result.stderr}")
                
            else:
                print(f"  ✗ Failed ({elapsed:.1f}s)")
                print(f"  Error: {result.stderr[:200]}")
                results[model_id] = {
                    "status": "failed",
                    "time": elapsed,
                    "error": result.stderr[:500]
                }
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout (>120s)")
            results[model_id] = {
                "status": "timeout",
                "time": 120
            }
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results[model_id] = {
                "status": "error",
                "error": str(e)
            }
    
    # Save comparison summary
    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "location": {"lat": lat, "lon": lon, "tile_size": tile_size},
            "models": MODELS,
            "results": results
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults:")
    for model_id, result in results.items():
        status_icon = "✓" if result["status"] == "success" else "✗"
        print(f"  {status_icon} Model {model_id} ({MODELS[model_id]['name'][:40]}...): {result['status']}")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"Visualizations in: {output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare all 7 Sat2Graph models")
    parser.add_argument("-lat", type=float, required=True, help="Latitude")
    parser.add_argument("-lon", type=float, required=True, help="Longitude")
    parser.add_argument("-tile_size", type=int, default=500, help="Tile size in meters")
    parser.add_argument("-output", type=str, default="comparison_results", help="Output directory")
    
    args = parser.parse_args()
    
    run_model_comparison(args.lat, args.lon, args.tile_size, args.output)
