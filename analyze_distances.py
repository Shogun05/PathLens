
import pandas as pd
import geopandas as gpd

path = "data/optimization/runs/optimized_nodes_with_scores.parquet"
df = pd.read_parquet(path)

dist_cols = [c for c in df.columns if c.startswith("dist_to_")]
print(f"Distance columns: {dist_cols}")

for col in dist_cols:
    print(f"\n--- {col} ---")
    print(df[col].describe())
    zeros = (df[col] == 0).sum()
    nans = df[col].isna().sum()
    print(f"Zeros: {zeros}")
    print(f"NaNs: {nans}")
    
print("\nTravel Time Min Stats:")
if "travel_time_min" in df.columns:
    print(df["travel_time_min"].describe())
else:
    print("travel_time_min column missing")
