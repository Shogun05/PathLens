import json
import os
from pathlib import Path

cities = ["chennai", "kolkata", "navi_mumbai"]
modes = ["baseline", "ga_only", "ga_milp", "ga_milp_pnmlr"]

# Define metrics structure for extraction
# (Category, Metric Label, JSON accessor function)
metrics_def = [
    ("Citywide", "Accessibility", lambda d: d["scores"]["citywide"]["accessibility_mean"]),
    ("Citywide", "Travel Time", lambda d: d["scores"]["citywide"]["travel_time_min_mean"]),
    ("Citywide", "Walkability", lambda d: d["scores"]["citywide"]["walkability_mean"]),
    ("Underserved (bottom 20%)", "Accessibility", lambda d: d["scores"]["underserved"]["accessibility_mean"]),
    ("Underserved (bottom 20%)", "Travel Time", lambda d: d["scores"]["underserved"]["travel_time_min_mean"]),
    ("Distribution", "Travel P90", lambda d: d["scores"]["distribution"]["travel_time_p90"]),
    ("Distribution", "Travel P95", lambda d: d["scores"]["distribution"]["travel_time_p95"]),
    ("Distribution", "Travel Max", lambda d: d["scores"]["distribution"]["travel_time_max"]),
    ("Equity Score", "", lambda d: d["scores"]["equity"]),
]

project_root = "/home/shogun/Documents/proj/PathLens"

def get_metrics(city, mode):
    if mode == "baseline":
        path = Path(project_root) / "data" / "cities" / city / "baseline" / "metrics_summary.json"
    else:
        # Try standard name first
        path = Path(project_root) / "data" / "cities" / city / "optimized" / mode / "metrics_summary.json"
        
        # Fallback to prefixed name if standard doesn't exist
        if not path.exists():
             prefixed_path = Path(project_root) / "data" / "cities" / city / "optimized" / mode / f"{mode}_metrics_summary.json"
             if prefixed_path.exists():
                 path = prefixed_path
    
    if not path.exists():
        return None
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

def format_value(val):
    if val is None:
        return "-"
    if isinstance(val, (int, float)):
        return f"{val:.2f}"
    return str(val)

for city in cities:
    print(f"\n# Metrics for {city.upper()}\n")
    
    # Header
    headers = ["Metric", "Baseline", "ga_only", "ga_milp", "ga_milp_pnmlr"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    # Pre-fetch data for this city to avoid re-reading
    data_map = {}
    for mode in modes:
        data_map[mode] = get_metrics(city, mode)

    last_category = None
    
    for category, label, accessor in metrics_def:
        row = []
        
        # Format Metric Column 
        metric_name = ""
        if category != "Equity Score":
             if category != last_category:
                 # Print a sub-header row or just include category in the name?
                 # The user prompt has "Citywide" then indented items. 
                 # I'll just put the category as a separate row or bolded if needed, 
                 # but for a pure MD table, flat is easier.
                 # Let's try to stick to the requested visual format closely.
                 # The requested format implies:
                 # Metric
                 # Citywide
                 #   Accessibility
                 metric_name = f"**{category}**"
                 
                 # Logic to handle the grouping visually in a single table is tricky.
                 # I will prefix: "Category: Metric"
                 
        if category == "Equity Score":
            metric_col = "Equity Score"
        else:
            metric_col = f"{category}: {label}"

        row.append(metric_col)

        for mode in modes:
            data = data_map[mode]
            val = None
            if data:
                try:
                    val = accessor(data)
                except KeyError:
                    val = None
            row.append(format_value(val))
        
        print("| " + " | ".join(row) + " |")
