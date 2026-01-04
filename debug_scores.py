
import pandas as pd
import numpy as np

def analyze(path, name):
    print(f"\n--- {name} ---")
    try:
        df = pd.read_parquet(path)
        print(f"Node count: {len(df)}")
        for col in ["accessibility_score", "travel_time_score", "equity_score", "structure_score", "walkability"]:
            if col in df.columns:
                print(f"{col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
            else:
                print(f"{col}: [MISSING]")
        
        if "degree" in df.columns:
             print(f"Avg Degree: {df['degree'].mean():.4f}")
        if "betweenness_centrality" in df.columns:
             print(f"Avg Betweenness: {df['betweenness_centrality'].mean():.8f}")

    except Exception as e:
        print(f"Error loading {path}: {e}")

analyze("data/analysis/baseline_nodes_with_scores.parquet", "BASELINE")
analyze("data/analysis/optimized_nodes_with_scores.parquet", "OPTIMIZED")
