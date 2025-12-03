#!/usr/bin/env python3
"""
Complete PathLens pipeline for Bengaluru amenity data:
1. Convert JSON amenities to GeoJSON
2. Build street network graph
3. Compute walkability scores
4. Generate interactive visualization
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print("\n" + "=" * 60)
    print(f"▶ {description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✅ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Run PathLens pipeline for Bengaluru")
    parser.add_argument("--skip-convert", action="store_true", help="Skip amenity conversion step")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph building step")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip scoring step")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization step")
    parser.add_argument("--place", default="Bangalore, India", help="Place name for OSM data")
    parser.add_argument("--force", action="store_true", help="Force recompute all steps (ignore cache)")
    args = parser.parse_args()
    
    # If force flag is set, inform user
    if args.force:
        print("⚠️  --force flag detected: All cached data will be ignored and recomputed")
        print("   This may take a long time!\n")
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    analysis_dir = data_dir / "analysis"
    
    # Ensure directories exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    python_exe = sys.executable
    
    # Step 1: Convert amenities to GeoJSON
    pois_path = raw_dir / "pois.geojson"
    if not args.skip_convert:
        if pois_path.exists() and not args.force:
            print("\n✓ POIs already converted:", pois_path)
            print("  (Use --force to re-convert)")
        else:
            if args.force and pois_path.exists():
                print("\n🔄 Force mode: Re-converting POIs...")
            run_command(
                [python_exe, str(script_dir / "convert_amenities_to_geojson.py")],
                "Converting amenities to GeoJSON"
            )
    
    # Check if we need to download network data
    graph_path = raw_dir / "graph.graphml"
    buildings_path = raw_dir / "buildings.geojson"
    
    # Check what data already exists
    has_graph = graph_path.exists()
    has_buildings = buildings_path.exists()
    
    if (not has_graph or args.force) and not args.skip_graph:
        if args.force and has_graph:
            print("\n🔄 Force mode: Re-downloading street network...")
        print("\n" + "=" * 60)
        print("📡 Downloading street network data from OpenStreetMap...")
        print("⏱️  This may take 5-10 minutes for large cities...")
        print("=" * 60)
        run_command(
            [
                python_exe,
                str(script_dir / "data_collection.py"),
                "--place", args.place,
                "--out-dir", str(raw_dir)
            ],
            "Downloading OSM network data"
        )
    elif has_graph and not args.force:
        print("\n✓ Using cached street network data from", graph_path)
        if has_buildings:
            print("✓ Using cached buildings data from", buildings_path)
    
    # Step 2: Build graph and map POIs to nodes
    processed_graph = processed_dir / "graph.graphml"
    poi_mapping = processed_dir / "poi_node_mapping.parquet"
    
    if not args.skip_graph:
        if processed_graph.exists() and poi_mapping.exists() and not args.force:
            print("\n✓ Using cached processed graph:", processed_graph)
            print("✓ Using cached POI mapping:", poi_mapping)
            print("  (Use --force to re-process)")
        else:
            if args.force and processed_graph.exists():
                print("\n🔄 Force mode: Re-processing graph...")
            # Build command with only files that exist
            cmd = [
                python_exe,
                str(script_dir / "graph_build.py"),
                "--graph-path", str(raw_dir / "graph.graphml"),
                "--pois-path", str(raw_dir / "pois.geojson"),
                "--out-dir", str(processed_dir)
            ]
            
            # Add optional layers only if they exist
            buildings_file = raw_dir / "buildings.geojson"
            landuse_file = raw_dir / "landuse.geojson"
            
            if buildings_file.exists():
                cmd.extend(["--buildings-path", str(buildings_file)])
                print(f"  Using buildings data: {buildings_file}")
            
            if landuse_file.exists():
                cmd.extend(["--landuse-path", str(landuse_file)])
                print(f"  Using landuse data: {landuse_file}")
            
            run_command(cmd, "Building graph and mapping POIs")
    
    # Step 3: Compute walkability scores
    nodes_scores = analysis_dir / "nodes_with_scores.parquet"
    h3_agg = analysis_dir / "h3_agg.parquet"
    metrics_summary = analysis_dir / "metrics_summary.json"
    
    if not args.skip_scoring:
        if nodes_scores.exists() and h3_agg.exists() and metrics_summary.exists() and not args.force:
            print("\n✓ Using cached walkability scores:", nodes_scores)
            print("✓ Using cached H3 aggregates:", h3_agg)
            print("  (Use --force to re-compute)")
        else:
            if args.force and nodes_scores.exists():
                print("\n🔄 Force mode: Re-computing scores...")
            run_command(
                [
                    python_exe,
                    str(script_dir / "scoring.py"),
                    "--graph-path", str(processed_dir / "graph.graphml"),
                    "--poi-mapping", str(processed_dir / "poi_node_mapping.parquet"),
                    "--pois-path", str(raw_dir / "pois.geojson"),
                    "--out-dir", str(analysis_dir),
                    "--config", str(project_dir / "config.yaml")
                ],
                "Computing walkability scores"
            )
    
    # Step 4: Generate visualization
    map_output = script_dir / "interactive_map.html"
    
    if not args.skip_viz:
        if map_output.exists() and not args.force:
            print("\n✓ Map already exists:", map_output)
            print("  (Use --force to re-generate)")
        else:
            if args.force and map_output.exists():
                print("\n🔄 Force mode: Re-generating map...")
            run_command(
                [
                    python_exe,
                    str(script_dir / "visualize.py"),
                    "--graph-path", str(processed_dir / "graph.graphml"),
                    "--pois-path", str(raw_dir / "pois.geojson"),
                    "--nodes-path", str(analysis_dir / "nodes_with_scores.parquet"),
                    "--mapping-path", str(processed_dir / "poi_node_mapping.parquet"),
                    "--out", str(map_output)
                ],
                "Generating interactive map"
            )
    
    print("\n" + "=" * 60)
    print("🎉 PathLens pipeline completed successfully!")
    print("=" * 60)
    print(f"📊 Results:")
    print(f"  - Processed graph: {processed_dir / 'graph.graphml'}")
    print(f"  - Node scores: {analysis_dir / 'nodes_with_scores.csv'}")
    print(f"  - H3 aggregates: {analysis_dir / 'h3_agg.csv'}")
    print(f"  - Interactive map: {script_dir / 'interactive_map.html'}")
    print(f"  - Metrics summary: {analysis_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
