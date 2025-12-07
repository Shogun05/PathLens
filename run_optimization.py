#!/usr/bin/env python3
"""End-to-end runner for the PathLens optimization workflow.

It executes the following sequence from the repository root:
 1. ``optimization/list_optimizable_nodes.py`` to refresh high-travel nodes
 2. ``optimization/hybrid_ga.py`` to search for improved amenity placements
 3. ``optimization/generate_solution_map.py`` to export GeoJSON + HTML map
 4. ``optimization/run_optimized_pipeline.py`` to rebuild prefixed graphs/scores

Each step can be skipped or customised via CLI flags; see ``--help``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_step(description: str, command: List[str]) -> None:
    """Execute a subprocess while echoing context to stdout."""
    print("\n" + "=" * 80)
    print(f"▶ {description}")
    print("=" * 80)
    print("Command:", " ".join(command))
    print()

    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Step failed ({description}) with exit code {result.returncode}.")

    print(f"\n✔ {description} completed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PathLens optimization stack from the repository root.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python interpreter used for all subprocesses")

    parser.add_argument("--skip-candidates", action="store_true", help="Skip regenerating high-travel nodes (list_optimizable_nodes.py)")
    parser.add_argument("--skip-ga", action="store_true", help="Skip the hybrid GA search (hybrid_ga.py)")
    parser.add_argument("--skip-map-refresh", action="store_true", help="Skip refreshing the combined POI map (generate_solution_map.py)")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip the prefixed pipeline rebuild (run_optimized_pipeline.py)")

    parser.add_argument("--candidate-threshold", type=float, default=None, help="Override travel time threshold in minutes for optimisation candidates")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Limit number of candidate rows written")

    parser.add_argument("--ga-population", type=int, default=None, help="Population size override for hybrid GA")
    parser.add_argument("--ga-generations", type=int, default=None, help="Generation count override for hybrid GA")
    parser.add_argument("--ga-workers", type=int, default=None, help="Worker thread count override for hybrid GA")
    parser.add_argument("--ga-random-seed", type=int, default=None, help="Seed override for hybrid GA")
    parser.add_argument("--ga-config", type=Path, default=None, help="Alternate YAML config for hybrid GA")

    parser.add_argument("--map-skip-metrics", action="store_true", help="Pass --skip-metrics to generate_solution_map.py")

    parser.add_argument("--pipeline-refresh-map", action="store_true", help="Allow run_optimized_pipeline.py to refresh the GA map step")
    parser.add_argument("--pipeline-skip-map-metrics", action="store_true", help="Pass --skip-map-metrics to run_optimized_pipeline.py")
    parser.add_argument("--pipeline-force-graph", action="store_true", help="Force graph rebuild in run_optimized_pipeline.py")
    parser.add_argument("--pipeline-force-scoring", action="store_true", help="Force scoring recomputation in run_optimized_pipeline.py")
    parser.add_argument("--pipeline-skip-graph", action="store_true", help="Skip graph rebuild in run_optimized_pipeline.py")
    parser.add_argument("--pipeline-skip-scoring", action="store_true", help="Skip scoring recomputation in run_optimized_pipeline.py")
    parser.add_argument("--baseline-prefix", default="baseline", help="Filename prefix for baseline artifacts")
    parser.add_argument("--optimized-prefix", default="optimized", help="Filename prefix for optimized artifacts")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    optimization_dir = project_root / "optimization"
    runs_dir = optimization_dir / "runs"
    analysis_dir = project_root / "data" / "analysis"

    python_exe = str(args.python)
    nodes_scores = analysis_dir / "nodes_with_scores.parquet"
    high_travel_csv = optimization_dir / "high_travel_time_nodes.csv"

    if not args.skip_candidates:
        candidate_command: List[str] = [
            python_exe,
            str(optimization_dir / "list_optimizable_nodes.py"),
            "--input",
            str(nodes_scores),
            "--output",
            str(high_travel_csv),
        ]
        if args.candidate_threshold is not None:
            candidate_command.extend(["--threshold", str(args.candidate_threshold)])
        if args.candidate_limit is not None:
            candidate_command.extend(["--limit", str(args.candidate_limit)])
        run_step("Extract high travel time nodes", candidate_command)
    else:
        print("\nSkipping candidate extraction (--skip-candidates)")

    if not args.skip_ga:
        ga_command: List[str] = [
            python_exe,
            str(optimization_dir / "hybrid_ga.py"),
            "--nodes-scores",
            str(nodes_scores),
            "--high-travel",
            str(high_travel_csv),
            "--analysis-dir",
            str(runs_dir),
        ]
        if args.ga_config is not None:
            ga_command.extend(["--config", str(args.ga_config)])
        if args.ga_population is not None:
            ga_command.extend(["--population", str(args.ga_population)])
        if args.ga_generations is not None:
            ga_command.extend(["--generations", str(args.ga_generations)])
        if args.ga_workers is not None:
            ga_command.extend(["--workers", str(args.ga_workers)])
        if args.ga_random_seed is not None:
            ga_command.extend(["--random-seed", str(args.ga_random_seed)])
        run_step("Run hybrid genetic algorithm", ga_command)
    else:
        print("\nSkipping hybrid GA (--skip-ga)")

    combined_geojson = runs_dir / "poi_mapping.geojson"
    if not args.skip_map_refresh:
        map_command: List[str] = [python_exe, str(optimization_dir / "generate_solution_map.py")]
        if args.map_skip_metrics:
            map_command.append("--skip-metrics")
        run_step("Generate combined POI GeoJSON and map", map_command)
    else:
        print("\nSkipping map refresh (--skip-map-refresh)")

    orchestrator = optimization_dir / "run_optimized_pipeline.py"
    if not orchestrator.exists():
        raise SystemExit(f"Expected orchestrator not found at {orchestrator}.")

    if not args.skip_pipeline:
        pipeline_command: List[str] = [
            python_exe,
            str(orchestrator),
            "--baseline-prefix",
            args.baseline_prefix,
            "--optimized-prefix",
            args.optimized_prefix,
        ]
        if args.pipeline_refresh_map:
            pipeline_command.append("--refresh-map")
        else:
            pipeline_command.append("--skip-map")
        if args.pipeline_skip_map_metrics:
            pipeline_command.append("--skip-map-metrics")
        if args.pipeline_force_graph:
            pipeline_command.append("--force-graph")
        if args.pipeline_force_scoring:
            pipeline_command.append("--force-scoring")
        if args.pipeline_skip_graph:
            pipeline_command.append("--skip-graph")
        if args.pipeline_skip_scoring:
            pipeline_command.append("--skip-scoring")
        run_step("Run optimization pipeline orchestrator", pipeline_command)
    else:
        print("\nSkipping prefixed pipeline rebuild (--skip-pipeline)")

    print("\n" + "=" * 80)
    print("Optimization workflow complete. Review outputs under optimization/runs and data/analysis.")
    print("=" * 80)


if __name__ == "__main__":
    main()
