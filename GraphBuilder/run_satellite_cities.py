#!/usr/bin/env python3
"""
Run Satellite Pipeline for Multiple Cities.
Downloads -> Inference -> Build -> Organize
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

# Paths
GRAPH_BUILDER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GRAPH_BUILDER_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CITIES_DIR = DATA_DIR / "cities"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
# Paths
GRAPH_BUILDER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GRAPH_BUILDER_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CITIES_DIR = DATA_DIR / "cities"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"

# Determine Inference Python (venv37 or similar)
possible_venvs = [
    GRAPH_BUILDER_DIR / "venv37" / "bin" / "python",
    GRAPH_BUILDER_DIR / "venvg" / "bin" / "python",
    GRAPH_BUILDER_DIR / "venv" / "bin" / "python",
]
INFERENCE_PYTHON = None
for p in possible_venvs:
    if p.exists():
        INFERENCE_PYTHON = p
        break

# Determine Downloader Python (System python usually has cv2)
import shutil
SYSTEM_PYTHON = shutil.which("python3")

CITIES = [
    "Bengaluru, India",
    "Chandigarh, India",
    "Mumbai, India",
    "Navi Mumbai, India"
]

def log(msg):
    print(f"[SatelliteJob] {msg}", flush=True)

def slugify(city_name: str) -> str:
    city = city_name.split(",")[0].strip().lower()
    if city == "bengaluru":
        return "bangalore"
    return city.replace(" ", "_").replace("-", "_")

def get_bbox(city_name):
    """Get bbox using OSMnx (imports inside function)."""
    # Use VENV_PYTHON to run a snippet since this process might presumably be system python or other
    # We need a robust way to get bbox without conflicting imports
    snippet = f"""
import osmnx as ox
try:
    name = "{city_name}"
    if name == "Mumbai, India": name = "Greater Mumbai, India"
    gdf = ox.geocode_to_gdf(name)
    bbox = gdf.total_bounds
    print(f"{{bbox[3]}},{{bbox[1]}},{{bbox[2]}},{{bbox[0]}}") # N, S, E, W
except Exception as e:
    print("ERROR")
"""
    try:
        res = subprocess.run(
            [str(VENV_PYTHON), "-c", snippet],
            capture_output=True, text=True
        )
        out = res.stdout.strip()
        if "ERROR" in out or not out:
            return None
        return [float(x) for x in out.split(",")]
    except Exception:
        return None

def main():
    log("Starting multi-city satellite pipeline...")
    
    if not INFERENCE_PYTHON:
        log("WARNING: No suitable venv found for inference (venv37/venvg). Inference will be skipped.")
    else:
        log(f"Using inference python: {INFERENCE_PYTHON}")

    # 1. Download Cities
    for city in CITIES:
        slug = slugify(city)
        log(f"\nProcessing {city} ({slug})...")
        
        city_raw_dir = CITIES_DIR / slug / "raw"
        sat_img_dir = city_raw_dir / "satellite_images"
        sat_graph_dir = CITIES_DIR / slug / "satellite_graph" # For visualizations
        osm_out_dir = city_raw_dir / "osm"
        
        sat_img_dir.mkdir(parents=True, exist_ok=True)
        sat_graph_dir.mkdir(parents=True, exist_ok=True)
        osm_out_dir.mkdir(parents=True, exist_ok=True)
        
        # A. GET BBOX
        bbox = get_bbox(city)
        if not bbox:
            log(f"Skipping {city} due to geocoding failure")
            continue
            
        # B. DOWNLOAD
        log(f"Downloading images for {city}...")
        download_script = GRAPH_BUILDER_DIR / "downloader" / "download_batch.py"
        # Use SYSTEM_PYTHON for download (needs cv2)
        downloader_bin = SYSTEM_PYTHON if SYSTEM_PYTHON else sys.executable
        
        cmd = [
            str(downloader_bin),
            str(download_script),
            "--bbox", *map(str, bbox),
            "--output-dir", str(sat_img_dir),
            "--zoom", "17" 
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            log(f"Download failed for {city}")
            continue
            
        # C. INFERENCE
        if INFERENCE_PYTHON:
            log(f"Running inference for {city}...")
            inf_script = GRAPH_BUILDER_DIR / "run_custom_inference.py"
            
            # We need a temp output dir for inference results (pickles)
            temp_out = GRAPH_BUILDER_DIR / "temp_outputs" / slug
            temp_out.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                str(INFERENCE_PYTHON),
                str(inf_script),
                "--input-dir", str(sat_img_dir),
                "--output-dir", str(temp_out)
            ]
            res = subprocess.run(cmd)
            
            if res.returncode == 0:
                # D. BUILD GRAPH
                log(f"Building graph for {city}...")
                build_script = GRAPH_BUILDER_DIR / "build_satellite_graph.py"
                meta_file = sat_img_dir / "metadata.json"
                
                cmd = [
                    str(VENV_PYTHON),
                    str(build_script),
                    "--input-dir", str(temp_out),
                    "--metadata", str(meta_file),
                    "--output-dir", str(osm_out_dir)
                ]
                subprocess.run(cmd)
                
                # E. CLEANUP / MOVE OUTPUTS
                log(f"Moving visualizations to {sat_graph_dir}")
                for png in temp_out.glob("*.png"):
                    try:
                        shutil.copy2(str(png), str(sat_graph_dir / png.name))
                    except Exception as e:
                        log(f"Failed to copy {png.name}: {e}")
            else:
                log(f"Inference failed for {city}")
        else:
            log(f"Skipping inference for {city} (missing environment)")
        
    log("All cities processed.")

if __name__ == "__main__":
    main()
