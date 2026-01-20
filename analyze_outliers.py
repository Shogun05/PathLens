import pandas as pd
import numpy as np

# Load Kolkata baseline nodes
path = "data/cities/kolkata/baseline/nodes_with_scores.parquet"
try:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} nodes from {path}")
    
    # Check bounds
    print(f"Longitude range: {df['x'].min()} to {df['x'].max()}")
    print(f"Latitude range: {df['y'].min()} to {df['y'].max()}")
    
    # Filter anomalous nodes (Longitude < 85)
    # Kolkata is roughly 88.3 E
    outliers = df[df['x'] < 85]
    print(f"\nFound {len(outliers)} outliers with Longitude < 85")
    
    if len(outliers) > 0:
        print("\nSample outliers:")
        print(outliers[['osmid', 'x', 'y']].head())
        
        # Check if they cluster around the reported coordinates (76.36)
        near_indore = outliers[
            (outliers['x'] > 76) & (outliers['x'] < 77) &
            (outliers['y'] > 22) & (outliers['y'] < 23)
        ]
        print(f"\nNodes near reported anomaly (76.36, 22.52): {len(near_indore)}")
        if len(near_indore) > 0:
            print(near_indore[['osmid', 'x', 'y']].head())

except Exception as e:
    print(f"Error: {e}")
