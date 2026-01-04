
import pandas as pd
from pathlib import Path

baseline_path = Path("data/analysis/baseline_nodes_with_scores.parquet")
optimized_path = Path("data/optimization/runs/optimized_nodes_with_scores.parquet")

base = pd.read_parquet(baseline_path)
opt = pd.read_parquet(optimized_path)

print(f"Baseline shape: {base.shape}")
print(f"Optimized shape: {opt.shape}")

print(f"Baseline index type: {base.index.dtype}")
print(f"Optimized index type: {opt.index.dtype}")

print(f"Baseline index sample: {base.index[:5]}")
print(f"Optimized index sample: {opt.index[:5]}")

# Check overlap
overlap = base.index.intersection(opt.index)
print(f"Overlap size: {len(overlap)}")
