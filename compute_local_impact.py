
import pandas as pd
import json
import argparse
from pathlib import Path

def compute_local_impact():
    # Load baseline and optimized nodes
    baseline_path = Path("data/analysis/baseline_nodes_with_scores.parquet")
    optimized_path = Path("data/analysis/optimized_nodes_with_scores.parquet")
    summary_path = Path("data/analysis/optimized_metrics_summary.json")

    if not baseline_path.exists() or not optimized_path.exists():
        print("Error: Files not found.")
        return

    print("Loading datasets...")
    baseline = pd.read_parquet(baseline_path)
    optimized = pd.read_parquet(optimized_path)

    # Ensure index alignment (cast to string)
    baseline.index = baseline.index.astype(str)
    optimized.index = optimized.index.astype(str)

    common_indices = baseline.index.intersection(optimized.index)
    baseline = baseline.loc[common_indices]
    optimized = optimized.loc[common_indices]

    # Calculate travel time improvement (reduction in minutes)
    # Using 'travel_time_min' column which we confirmed exists and is ~27.5 on average
    
    # Delta: Positive means improvement (reduction in time)
    delta_minutes = baseline["travel_time_min"] - optimized["travel_time_min"]
    
    # Filter for "Affected Areas": Nodes where improvement > 1 minute
    affected_mask = delta_minutes > 1.0
    affected_nodes = delta_minutes[affected_mask]
    
    count_affected = len(affected_nodes)
    avg_reduction = affected_nodes.mean()
    
    # Calculate pre and post averages for affected nodes
    affected_baseline_avg = baseline.loc[affected_mask, "travel_time_min"].mean()
    affected_optimized_avg = optimized.loc[affected_mask, "travel_time_min"].mean()

    print(f"Affected Nodes ( > 1 min improvement): {count_affected} / {len(baseline)}")
    print(f"Average Reduction in Affected Areas: {avg_reduction:.2f} min")
    print(f"Baseline Time (Affected): {affected_baseline_avg:.2f} min")
    print(f"Optimized Time (Affected): {affected_optimized_avg:.2f} min")

    # Update summary JSON
    if summary_path.exists():
        with open(summary_path, "r") as f:
            data = json.load(f)
        
        # Add new metrics under a 'local_impact' key
        data["local_impact"] = {
            "affected_node_count": int(count_affected),
            "avg_reduction_min": float(avg_reduction) if count_affected > 0 else 0.0,
            "baseline_avg_min": float(affected_baseline_avg) if count_affected > 0 else 0.0,
            "optimized_avg_min": float(affected_optimized_avg) if count_affected > 0 else 0.0
        }
        
        with open(summary_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Updated {summary_path} with local impact metrics.")

if __name__ == "__main__":
    compute_local_impact()
