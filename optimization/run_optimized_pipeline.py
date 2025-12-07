#!/usr/bin/env python3
"""Orchestrate baseline and optimized PathLens pipeline runs.

Steps:
 1. Refresh optimized POI artifacts from the latest GA solution (optional)
 2. Rebuild baseline and optimized processed graphs with clear prefixes
 3. Recompute scoring outputs for both scenarios with NaN-safe summaries

Usage examples:
    python optimization/run_optimized_pipeline.py --refresh-map
    python optimization/run_optimized_pipeline.py --skip-map --force-graph
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_PREFIX = "baseline"
DEFAULT_OPTIMIZED_PREFIX = "optimized"


def run_step(command: Sequence[str], description: str) -> None:
    """Execute a subprocess command with logging and error handling."""
    print("\n" + "=" * 80)
    print(f"▶ {description}")
    print("=" * 80)
    print("Command:", " ".join(str(piece) for piece in command))
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed ({description}) with exit code {result.returncode}.")


def ensure_optimized_layer(combined_path: Path, optimized_path: Path) -> int:
    """Derive the optimized-only GeoJSON layer from the combined export."""
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined POI layer not found at {combined_path}")

    data = json.loads(combined_path.read_text(encoding="utf-8"))
    optimized_features = [
        feature
        for feature in data.get("features", [])
        if feature.get("properties", {}).get("source") == "optimized"
    ]

    optimized_collection = {
        "type": "FeatureCollection",
        "crs": data.get("crs"),
        "features": optimized_features,
    }
    optimized_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_path.write_text(json.dumps(optimized_collection, indent=2), encoding="utf-8")
    return len(optimized_features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline & optimized pipeline passes.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python interpreter to use")
    parser.add_argument("--refresh-map", action="store_true", help="Rebuild combined/optimized POI layers with the GA solution")
    parser.add_argument("--skip-map", action="store_true", help="Skip GA/map refresh and reuse existing GeoJSON outputs")
    parser.add_argument("--force-graph", action="store_true", help="Rebuild graph artifacts even if prefixed outputs already exist")
    parser.add_argument("--force-scoring", action="store_true", help="Recompute scoring outputs even if prefixed files already exist")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph rebuilding steps")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip scoring recomputation steps")
    parser.add_argument("--baseline-prefix", default=DEFAULT_BASELINE_PREFIX, help="Filename prefix for baseline artifacts")
    parser.add_argument("--optimized-prefix", default=DEFAULT_OPTIMIZED_PREFIX, help="Filename prefix for optimized artifacts")
    parser.add_argument("--skip-map-metrics", action="store_true", help="Pass --skip-metrics to generate_solution_map.py")
    args = parser.parse_args()

    python_exe = str(args.python)

    runs_dir = PROJECT_ROOT / "optimization" / "runs"
    combined_geojson = runs_dir / "poi_mapping.geojson"
    optimized_geojson = runs_dir / "optimized_pois.geojson"
    merged_pois_output = runs_dir / "merged_pois.geojson"

    if not args.skip_map:
        map_command = [
            python_exe,
            str(PROJECT_ROOT / "optimization" / "generate_solution_map.py"),
        ]
        if args.skip_map_metrics:
            map_command.append("--skip-metrics")
        run_step(map_command, "Generate optimized POIs & map from GA best candidate")
        args.refresh_map = True

    if args.refresh_map:
        created = ensure_optimized_layer(combined_geojson, optimized_geojson)
        print(f"Prepared optimized-only POI layer with {created} features -> {optimized_geojson}")
    elif not optimized_geojson.exists():
        created = ensure_optimized_layer(combined_geojson, optimized_geojson)
        print(f"Derived optimized-only POI layer (missing previously) -> {optimized_geojson} ({created} features)")

    baseline_prefix = args.baseline_prefix.strip()
    optimized_prefix = args.optimized_prefix.strip()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    analysis_dir = PROJECT_ROOT / "data" / "analysis"

    baseline_graph = processed_dir / f"{baseline_prefix}_graph.graphml"
    baseline_mapping = processed_dir / f"{baseline_prefix}_poi_node_mapping.parquet"
    optimized_graph = processed_dir / f"{optimized_prefix}_graph.graphml"
    optimized_mapping = processed_dir / f"{optimized_prefix}_poi_node_mapping.parquet"

    if not args.skip_graph:
        if args.force_graph or not (baseline_graph.exists() and baseline_mapping.exists()):
            baseline_command = [
                python_exe,
                str(PROJECT_ROOT / "pipeline" / "graph_build.py"),
                "--graph-path",
                str(PROJECT_ROOT / "data" / "raw" / "graph.graphml"),
                "--pois-path",
                str(PROJECT_ROOT / "data" / "raw" / "pois.geojson"),
                "--out-dir",
                str(processed_dir),
                "--output-prefix",
                baseline_prefix,
            ]
            run_step(baseline_command, "Build baseline processed graph")
        else:
            print(f"Skipping baseline graph build (reuse {baseline_graph.name})")

        if args.force_graph or not (optimized_graph.exists() and optimized_mapping.exists()):
            optimized_command = [
                python_exe,
                str(PROJECT_ROOT / "pipeline" / "graph_build.py"),
                "--graph-path",
                str(PROJECT_ROOT / "data" / "raw" / "graph.graphml"),
                "--pois-path",
                str(PROJECT_ROOT / "data" / "raw" / "pois.geojson"),
                "--optimized-pois-path",
                str(optimized_geojson),
                "--merged-pois-out",
                str(merged_pois_output),
                "--out-dir",
                str(processed_dir),
                "--output-prefix",
                optimized_prefix,
            ]
            run_step(optimized_command, "Build optimized processed graph with merged POIs")
        else:
            print(f"Skipping optimized graph build (reuse {optimized_graph.name})")

    baseline_nodes = analysis_dir / f"{baseline_prefix}_nodes_with_scores.parquet"
    optimized_nodes = analysis_dir / f"{optimized_prefix}_nodes_with_scores.parquet"

    if not args.skip_scoring:
        if args.force_scoring or not baseline_nodes.exists():
            baseline_scoring_command = [
                python_exe,
                str(PROJECT_ROOT / "pipeline" / "scoring.py"),
                "--graph-path",
                str(baseline_graph),
                "--poi-mapping",
                str(baseline_mapping),
                "--pois-path",
                str(PROJECT_ROOT / "data" / "raw" / "pois.geojson"),
                "--out-dir",
                str(analysis_dir),
                "--config",
                str(PROJECT_ROOT / "config.yaml"),
                "--output-prefix",
                baseline_prefix,
            ]
            run_step(baseline_scoring_command, "Compute baseline scoring outputs")
        else:
            print(f"Skipping baseline scoring (reuse {baseline_nodes.name})")

        if args.force_scoring or not optimized_nodes.exists():
            optimized_scoring_command = [
                python_exe,
                str(PROJECT_ROOT / "pipeline" / "scoring.py"),
                "--graph-path",
                str(optimized_graph),
                "--poi-mapping",
                str(optimized_mapping),
                "--pois-path",
                str(merged_pois_output if merged_pois_output.exists() else combined_geojson),
                "--out-dir",
                str(analysis_dir),
                "--config",
                str(PROJECT_ROOT / "config.yaml"),
                "--output-prefix",
                optimized_prefix,
            ]
            run_step(optimized_scoring_command, "Compute optimized scoring outputs")
        else:
            print(f"Skipping optimized scoring (reuse {optimized_nodes.name})")

    print("\n" + "=" * 80)
    print("Pipeline orchestration complete. Key artifacts:")
    print(f"  Baseline graph:    {baseline_graph}")
    print(f"  Optimized graph:   {optimized_graph}")
    print(f"  Baseline scores:   {baseline_nodes}")
    print(f"  Optimized scores:  {optimized_nodes}")
    print(f"  Metrics summaries: {analysis_dir / (baseline_prefix + '_metrics_summary.json')} | "
          f"{analysis_dir / (optimized_prefix + '_metrics_summary.json')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
