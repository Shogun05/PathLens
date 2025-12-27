#!/usr/bin/env python3.14
"""
Evaluate Sat2Graph model predictions against 20cities ground truth.
Runs TOPO and APLS metrics on test set regions.
"""

import sys
import os
import json
import pickle
import subprocess
from pathlib import Path
import time

def get_test_regions(data_dir):
    """Get list of test region IDs (hash % 6 >= 5)"""
    data_dir = Path(data_dir)
    test_regions = []
    
    for file in sorted(data_dir.glob("region_*_gt_graph.json")):
        region_id = int(file.stem.split("_")[1])
        if hash(region_id) % 6 >= 5:
            test_regions.append(region_id)
    
    return test_regions[:5]  # Start with first 5 for testing


def run_topo_metric(gt_path, pred_path, output_file):
    """Run TOPO metric comparison"""
    
    metrics_dir = Path(__file__).parent / "metrics" / "topo"
    
    cmd = [
        "python3.14",
        str(metrics_dir / "main.py"),
        "-graph_gt", str(gt_path),
        "-graph_prop", str(pred_path),
        "-output", str(output_file)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # Parse output
            if output_file.exists():
                with open(output_file, 'r') as f:
                    content = f.read()
                    # Extract metrics from output
                    return {"status": "success", "output": content}
            return {"status": "success", "stdout": result.stdout}
        else:
            return {"status": "failed", "error": result.stderr}
            
    except Exception as e:
        return {"status": "error", "error": str(e)}


def run_apls_metric(gt_json, pred_json, output_file):
    """Run APLS metric comparison"""
    
    metrics_dir = Path(__file__).parent / "metrics" / "apls"
    
    # First compile the Go program if needed
    binary = metrics_dir / "apls_metric"
    if not binary.exists():
        print("  Compiling APLS metric...")
        compile_cmd = ["go", "build", "-o", str(binary), str(metrics_dir / "main.go")]
        compile_result = subprocess.run(compile_cmd, cwd=metrics_dir, capture_output=True)
        if compile_result.returncode != 0:
            return {"status": "compile_failed", "error": compile_result.stderr.decode()}
    
    cmd = [
        str(binary),
        str(gt_json),
        str(pred_json),
        str(output_file)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            if output_file.exists():
                with open(output_file, 'r') as f:
                    content = f.read()
                    return {"status": "success", "output": content}
            return {"status": "success", "stdout": result.stdout}
        else:
            return {"status": "failed", "error": result.stderr}
            
    except Exception as e:
        return {"status": "error", "error": str(e)}


def evaluate_dataset(model_id, data_dir, predictions_dir, output_dir):
    """Evaluate model predictions against dataset ground truth"""
    
    data_dir = Path(data_dir)
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL {model_id} ON 20CITIES DATASET")
    print(f"{'='*70}\n")
    
    test_regions = get_test_regions(data_dir)
    print(f"Test regions: {test_regions}\n")
    
    results = {}
    
    for region_id in test_regions:
        print(f"[Region {region_id}]")
        
        # Ground truth file
        gt_pickle = data_dir / f"region_{region_id}_refine_gt_graph.p"
        
        if not gt_pickle.exists():
            print(f"  ✗ Ground truth not found: {gt_pickle}")
            continue
        
        # Prediction file (would come from running model on this region)
        pred_pickle = predictions_dir / f"region_{region_id}_prediction.p"
        
        if not pred_pickle.exists():
            print(f"  ⚠ Prediction not found: {pred_pickle}")
            print(f"  (Run model inference on this region first)")
            continue
        
        region_results = {}
        
        # Run TOPO metric
        print(f"  Running TOPO metric...")
        topo_output = output_dir / f"region_{region_id}_topo.txt"
        topo_result = run_topo_metric(gt_pickle, pred_pickle, topo_output)
        region_results["topo"] = topo_result
        
        if topo_result["status"] == "success":
            print(f"  ✓ TOPO complete")
        else:
            print(f"  ✗ TOPO failed: {topo_result.get('error', 'Unknown')[:100]}")
        
        # Run APLS metric (requires JSON conversion)
        print(f"  Running APLS metric...")
        
        # Convert pickles to JSON for APLS
        scripts_dir = Path(__file__).parent / "docker" / "scripts"
        gt_json = output_dir / f"region_{region_id}_gt.json"
        pred_json = output_dir / f"region_{region_id}_pred.json"
        
        # Use convert.py to create JSON files
        for pkl, jsn in [(gt_pickle, gt_json), (pred_pickle, pred_json)]:
            conv_cmd = ["python3.14", str(scripts_dir / "convert.py"), str(pkl), str(jsn)]
            subprocess.run(conv_cmd, capture_output=True)
        
        if gt_json.exists() and pred_json.exists():
            apls_output = output_dir / f"region_{region_id}_apls.txt"
            apls_result = run_apls_metric(gt_json, pred_json, apls_output)
            region_results["apls"] = apls_result
            
            if apls_result["status"] == "success":
                print(f"  ✓ APLS complete")
            else:
                print(f"  ✗ APLS failed: {apls_result.get('error', 'Unknown')[:100]}")
        else:
            print(f"  ⚠ JSON conversion failed, skipping APLS")
        
        results[region_id] = region_results
        print()
    
    # Save evaluation summary
    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "model_id": model_id,
            "test_regions": test_regions,
            "results": results
        }, f, indent=2)
    
    print(f"{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Summary saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model on 20cities dataset")
    parser.add_argument("-model_id", type=int, required=True, help="Model ID (0-6)")
    parser.add_argument("-data_dir", type=str, default="data/20cities", help="Dataset directory")
    parser.add_argument("-predictions", type=str, required=True, help="Predictions directory")
    parser.add_argument("-output", type=str, default="evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    evaluate_dataset(args.model_id, args.data_dir, args.predictions, args.output)
