
import pandas as pd

path = "data/analysis/baseline_nodes_with_scores.parquet"
try:
    df = pd.read_parquet(path)
    print(f"Columns: {list(df.columns)}")
    if 'x' in df.columns and 'y' in df.columns:
        print("X and Y columns PRESNET.")
    else:
        print("X and Y columns MISSING.")
        if 'geometry' in df.columns:
            print("Geometry column PRESENT.")
except Exception as e:
    print(f"Error: {e}")
