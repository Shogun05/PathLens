#!/usr/bin/env python3
"""Analyze actual distance distributions to understand travel time discrepancy."""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# Load data
nodes = pd.read_parquet("data/analysis/baseline_nodes_with_scores.parquet")

# Get distance columns
distance_cols = [c for c in nodes.columns if c.startswith("dist_to_")]

print("=" * 80)
print("DISTANCE ANALYSIS")
print("=" * 80)

for col in distance_cols:
    amenity = col.replace("dist_to_", "")
    dists = nodes[col].replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"\n{amenity.upper()}:")
    print(f"  Mean: {dists.mean():.0f}m")
    print(f"  Median: {dists.median():.0f}m")
    print(f"  25th: {dists.quantile(0.25):.0f}m")
    print(f"  75th: {dists.quantile(0.75):.0f}m")

# Compute current "minimum" travel time (what's in the data)
print("\n" + "=" * 80)
print("CURRENT TRAVEL TIME (MIN ACROSS ALL AMENITIES)")
print("=" * 80)

current_travel = nodes["travel_time_min"].replace([np.inf, -np.inf], np.nan).dropna()
print(f"Mean: {current_travel.mean():.2f} min")
print(f"Median: {current_travel.median():.2f} min")

# Compute NEW weighted average travel time
print("\n" + "=" * 80)
print("NEW WEIGHTED AVERAGE TRAVEL TIME")
print("=" * 80)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

amenity_weights = config["amenity_weights"]
walking_speed_kmph = config["walking_speed_kmph"]

total_weight = sum(amenity_weights.values())
weighted_time = pd.Series(0.0, index=nodes.index)

for amenity, weight in amenity_weights.items():
    col = f"dist_to_{amenity}"
    if col not in nodes.columns:
        continue
    
    distance_m = nodes[col].fillna(10000.0)
    time_hours = distance_m / 1000.0 / walking_speed_kmph
    time_minutes = time_hours * 60.0
    weighted_time += (weight / total_weight) * time_minutes

print(f"Mean: {weighted_time.mean():.2f} min")
print(f"Median: {weighted_time.median():.2f} min")

# Compute ratio
ratio = weighted_time.mean() / current_travel.mean()
print(f"\nRatio (weighted/minimum): {ratio:.2f}x")

print("\n" + "=" * 80)
print("SOLUTION")
print("=" * 80)
print("\nThe travel_time_min in the data is the MIN distance to ANY amenity.")
print("Our new formula computes WEIGHTED AVERAGE across ALL amenities.")
print(f"\nTo match your target of 25-35 minutes baseline:")
print(f"  Current weighted average: {weighted_time.mean():.1f} min")
print(f"  Need to scale distances by: {30 / weighted_time.mean():.2f}x")
print(f"  OR adjust walking speed to: {walking_speed_kmph * (weighted_time.mean() / 30):.2f} km/h")

# Test scaling factor
scale_factor = 30 / weighted_time.mean()
print(f"\n--- Testing scale factor {scale_factor:.2f}x ---")

scaled_time = weighted_time * scale_factor
print(f"Scaled mean: {scaled_time.mean():.2f} min (target: 25-35)")

# For accessibility, test if we need to scale distances too
print("\n" + "=" * 80)
print("TESTING ACCESSIBILITY WITH SCALED DISTANCES")
print("=" * 80)

for scale in [1.0, 1.5, 1.8, 2.0, 2.2]:
    DECAY = 2000
    accessibility = pd.Series(0.0, index=nodes.index)
    
    for amenity, weight in amenity_weights.items():
        col = f"dist_to_{amenity}"
        if col not in nodes.columns:
            continue
        
        distance = (nodes[col].fillna(10000.0)) * scale
        amenity_score = 100 * np.exp(-distance / DECAY)
        accessibility += (weight / total_weight) * amenity_score
    
    print(f"Scale={scale:.1f}x: Accessibility mean = {accessibility.mean():.2f}")

print("\nConclusion: The distances are ALREADY correct.")
print("The issue is that travel_time_min measures something different (minimum, not average).")
print("We should just DISPLAY our weighted average as the correct metric.")
