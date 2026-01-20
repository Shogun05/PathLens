import argparse
import subprocess
import sys
import os
from pathlib import Path

# Add project root to sys.path to import city_paths
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from city_paths import CityDataManager


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print("\n" + "=" * 60)
    print(f"[RUNNING] {description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='', flush=True)
    
    returncode = process.wait()
    
    if returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {returncode}")
        sys.exit(1)
    
    print(f"\n[SUCCESS] {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Run PathLens data pipeline")
    parser.add_argument("--city", default="bangalore", help="City to process")
    parser.add_argument("--force", action="store_true", help="Force recompute all steps")
    args = parser.parse_args()
    
    cdm = CityDataManager(args.city, project_root=project_root)
    cfg = cdm.load_config()
    
    # Get settings from config
    data_cfg = cfg.get('data', {})
    
    # Generate place name from city (capitalize properly for OSM geocoding)
    # Config can override with project.place if needed for special cases
    city_to_place = {
        'bangalore': 'Bangalore, Karnataka, India',
        'mumbai': 'Mumbai, Maharashtra, India', 
        'navi_mumbai': 'Navi Mumbai, Maharashtra, India',
        'chandigarh': 'Chandigarh, India',
        'chennai': 'Chennai, Tamil Nadu, India',
        'delhi': 'Delhi, India',
        'hyderabad': 'Hyderabad, Telangana, India',
        'kolkata': 'Kolkata, West Bengal, India',
        'pune': 'Pune, Maharashtra, India',
    }
    default_place = city_to_place.get(cdm.city, f"{cdm.city.replace('_', ' ').title()}, India")
    place = cfg.get('project', {}).get('place', default_place)
    
    skip_convert = data_cfg.get('skip_convert', False)
    skip_graph = data_cfg.get('skip_graph', False)
    skip_scoring = data_cfg.get('skip_scoring', False)
    skip_viz = data_cfg.get('skip_viz', False)
    
    # Execution dir
    pipeline_dir = Path(__file__).parent
    python_exe = sys.executable
    
    # Step 1: Convert amenities (Legacy Bangalore support)
    legacy_source_dir = project_root / "data" / "raw" / "bengaluru" / "bengaluru_amenities"
    should_run_convert = (
        not skip_convert 
        and args.city == 'bangalore' 
        and legacy_source_dir.exists() 
        and any(legacy_source_dir.glob("*.json"))
    )
    
    if should_run_convert:
        if cdm.raw_pois.with_suffix('.parquet').exists() and not args.force:
            print(f"\n[OK] POIs already converted: {cdm.raw_pois.with_suffix('.parquet')}")
        else:
            run_command(
                [python_exe, str(pipeline_dir / "convert_amenities.py"), "--city", args.city],
                "Converting amenities to GeoJSON (Legacy source)"
            )
    elif not skip_convert:
        print("\n[INFO] Skipping legacy amenity conversion (source files not found or not Bangalore). Relying on OSM download.")
    
    # Step 2: Download network data
    if (not cdm.raw_graph.exists() or args.force) and not skip_graph:
        run_command(
            [
                python_exe,
                str(pipeline_dir / "collect_osm_data.py"),
                "--place", place,
                "--out-dir", str(cdm.raw_dir)
            ],
            "Downloading OSM network data"
        )
    
    # Step 3: Build graph
    if not skip_graph:
        if cdm.processed_graph.exists() and cdm.poi_mapping.exists() and not args.force:
            print(f"\n[OK] Using cached processed graph: {cdm.processed_graph}")
        else:
            cmd = [
                python_exe,
                str(pipeline_dir / "build_graph.py"),
                "--graph-path", str(cdm.raw_graph),
                "--pois-path", str(cdm.raw_pois.with_suffix('.parquet')),
                "--out-dir", str(cdm.processed_dir)
            ]
            
            if cdm.raw_buildings.exists():
                cmd.extend(["--buildings-path", str(cdm.raw_buildings)])
            if cdm.raw_landuse.exists():
                cmd.extend(["--landuse-path", str(cdm.raw_landuse)])
            
            run_command(cmd, "Building graph and mapping POIs")
    
    # Step 4: Compute scores
    if not skip_scoring:
        if cdm.baseline_nodes.exists() and cdm.baseline_metrics.exists() and not args.force:
            print(f"\n[OK] Using cached walkability scores: {cdm.baseline_nodes}")
        else:
            run_command(
                [
                    python_exe,
                    str(pipeline_dir / "compute_scores.py"),
                    "--city", args.city,
                    "--graph-path", str(cdm.processed_graph),
                    "--poi-mapping", str(cdm.poi_mapping),
                    "--pois-path", str(cdm.raw_pois.with_suffix('.parquet')),
                    "--out-dir", str(cdm.baseline_dir),
                    "--config", str(cdm.config)
                ],
                "Computing walkability scores"
            )
    
    print("\n" + "=" * 60)
    print(f"[COMPLETE] PathLens pipeline for {args.city} completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
