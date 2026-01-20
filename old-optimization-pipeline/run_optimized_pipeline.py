#!/usr/bin/env python3
"""Orchestrate baseline and optimized PathLens pipeline runs.

Steps:
 1. Refresh optimized POI artifacts from the latest GA solution (optional)
 2. Rebuild baseline and optimized processed graphs with clear prefixes
 3. Recompute scoring outputs for both scenarios with in-memory POI merging

Optimizations:
 - Eliminates merged_pois.geojson write (saves 79+ minutes)
 - In-memory POI merging during scoring (parquet load ~1.5s vs GeoJSON ~30s)
 - Total optimization savings: 79+ minutes per optimized pipeline run

Usage examples:
    python optimization/run_optimized_pipeline.py --refresh-map
    python optimization/run_optimized_pipeline.py --skip-map --force-graph
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Go up from optimization-pipeline/ to root
DEFAULT_BASELINE_PREFIX = "baseline"
DEFAULT_OPTIMIZED_PREFIX = "optimized"

LOGGER_NAME = "PathLensOptimizedPipeline"
logger = logging.getLogger(LOGGER_NAME)

# Global counters for progress tracking
_current_step = 0
_total_steps = 0


def _stream_command(command: Sequence[str], prefix: str = "") -> int:
    """Run a command and stream stdout/stderr to the console with optional prefix while returning the exit code."""
    with subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,  # Line buffered
    ) as process:
        assert process.stdout is not None  # for type-checkers
        for line in process.stdout:
            stripped = line.rstrip()
            if stripped:  # Only log non-empty lines
                prefixed_line = f"[{prefix}] {stripped}" if prefix else stripped
                print(prefixed_line)
                logger.info(prefixed_line)
            elif line:  # Preserve empty lines for formatting
                sys.stdout.write(line)
        process.wait()
        return process.returncode if process.returncode is not None else -1


def run_step(command: Sequence[str], description: str) -> None:
    """Execute a subprocess command with logging, timing, and better failure diagnostics."""
    global _current_step
    _current_step += 1

    progress_prefix = f"Step {_current_step}/{_total_steps}" if _total_steps > 0 else f"Step {_current_step}"
    header = f"[{progress_prefix}] {description}"
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)
    
    # Add -u flag for unbuffered Python output if command is a Python script
    modified_command = list(command)
    if len(modified_command) >= 2 and 'python' in str(modified_command[0]).lower():
        if modified_command[1] != '-u':
            modified_command.insert(1, '-u')
    
    command_str = " ".join(str(piece) for piece in modified_command)
    print("Command:", command_str)
    logger.info("Starting step: %s", header)
    logger.info("Executing command: %s", command_str)
    start_time = time.time()

    try:
        return_code = _stream_command(modified_command, prefix=progress_prefix)
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        logger.exception("Executable not found while running step '%s'", description)
        raise SystemExit(f"Unable to launch command for '{description}': {exc}") from exc

    elapsed = time.time() - start_time
    if return_code != 0:
        logger.error(
            "Step '%s' failed with exit code %s. Command: %s",
            description,
            return_code,
            command_str,
        )
        raise SystemExit(f"Step failed ({description}) with exit code {return_code}.")

    logger.info("Step completed: %s (%.1fs)", description, elapsed)
    print(f"Completed in {elapsed:.1f}s")


def ensure_optimized_layer(combined_path: Path, optimized_path: Path) -> int:
    """Derive the optimized-only GeoJSON layer from the combined export."""
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined POI layer not found at {combined_path}")

    logger.info("Extracting optimized features from %s", combined_path)
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
    count = len(optimized_features)
    logger.info("Extracted %d optimized POIs into %s", count, optimized_path)
    return count


def main() -> None:
    global _total_steps, _current_step
    
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity for this orchestrator",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger.debug("Logging configured at %s level", args.log_level.upper())

    python_exe = str(args.python)

    runs_dir = PROJECT_ROOT / "data" / "optimization" / "runs"
    combined_geojson = runs_dir / "poi_mapping.geojson"
    optimized_geojson = runs_dir / "optimized_pois.geojson"
    merged_pois_output = runs_dir / "merged_pois.geojson"

    # Define file paths early so we can check their existence for progress tracking
    baseline_prefix = args.baseline_prefix.strip()
    optimized_prefix = args.optimized_prefix.strip()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    analysis_dir = PROJECT_ROOT / "data" / "analysis"

    baseline_graph = processed_dir / f"{baseline_prefix}_graph.graphml"
    baseline_mapping = processed_dir / f"{baseline_prefix}_poi_node_mapping.parquet"
    optimized_graph = processed_dir / f"{optimized_prefix}_graph.graphml"
    optimized_mapping = processed_dir / f"{optimized_prefix}_poi_node_mapping.parquet"
    baseline_nodes = analysis_dir / f"{baseline_prefix}_nodes_with_scores.parquet"
    optimized_nodes = analysis_dir / f"{optimized_prefix}_nodes_with_scores.parquet"

    # Calculate total steps for progress tracking
    _current_step = 0
    _total_steps = 0
    if not args.skip_map:
        _total_steps += 1
    if not args.skip_graph:
        if args.force_graph or not (baseline_graph.exists() and baseline_mapping.exists()):
            _total_steps += 1
        if args.force_graph or not (optimized_graph.exists() and optimized_mapping.exists()):
            _total_steps += 1
    if not args.skip_scoring:
        if args.force_scoring or not baseline_nodes.exists():
            _total_steps += 1
        if args.force_scoring or not optimized_nodes.exists():
            _total_steps += 1
    
    print("\n" + "=" * 80)
    print("PathLens Optimization Pipeline")
    print(f"Total steps to execute: {_total_steps}")
    print("=" * 80)
    logger.info("Beginning pipeline run with %d steps", _total_steps)

    if not args.skip_map:
        map_command = [
            python_exe,
            "-u",  # Unbuffered output for real-time logging
            str(PROJECT_ROOT / "optimization" / "generate_solution_map.py"),
        ]
        if args.skip_map_metrics:
            map_command.append("--skip-metrics")
        map_script = PROJECT_ROOT / "optimization" / "generate_solution_map.py"
        logger.debug("Map script resolved to %s", map_script)
        run_step(map_command, "Generate optimized POIs & map from GA best candidate")
        args.refresh_map = True

    if args.refresh_map:
        created = ensure_optimized_layer(combined_geojson, optimized_geojson)
        logger.info("Prepared optimized-only POI layer with %d features at %s", created, optimized_geojson)
        print(f"Prepared optimized-only POI layer with {created} features -> {optimized_geojson}")
    elif not optimized_geojson.exists():
        created = ensure_optimized_layer(combined_geojson, optimized_geojson)
        logger.info("Derived optimized-only layer with %d features (previously missing)", created)
        print(f"Derived optimized-only POI layer (missing previously) -> {optimized_geojson} ({created} features)")

    if not args.skip_graph:
        if args.force_graph or not (baseline_graph.exists() and baseline_mapping.exists()):
            baseline_command = [
                python_exe,
                "-u",  # Unbuffered output
                str(PROJECT_ROOT / "data-pipeline" / "build_graph.py"),
                "--graph-path",
                str(PROJECT_ROOT / "data" / "raw" / "osm" / "graph.graphml"),
                "--pois-path",
                str(PROJECT_ROOT / "data" / "raw" / "osm" / "pois.parquet"),
                "--out-dir",
                str(processed_dir),
                "--output-prefix",
                baseline_prefix,
            ]
            run_step(baseline_command, "Build baseline processed graph")
        else:
            logger.info("Skipping baseline graph build; reusing %s", baseline_graph)
            print(f"\n[Skip] Baseline graph build (reusing {baseline_graph.name})")

        if args.force_graph or not (optimized_graph.exists() and optimized_mapping.exists()):
            optimized_command = [
                python_exe,
                "-u",  # Unbuffered output
                str(PROJECT_ROOT / "data-pipeline" / "build_graph.py"),
                "--graph-path",
                str(PROJECT_ROOT / "data" / "raw" / "osm" / "graph.graphml"),
                "--pois-path",
                str(PROJECT_ROOT / "data" / "raw" / "osm" / "pois.parquet"),
                "--optimized-pois-path",
                str(optimized_geojson),
                # Note: --merged-pois-out removed - merging now done in-memory during scoring (saves 79+ min)
                "--out-dir",
                str(processed_dir),
                "--output-prefix",
                optimized_prefix,
            ]
            run_step(optimized_command, "Build optimized processed graph (POI merge in scoring)")
        else:
            logger.info("Skipping optimized graph build; reusing %s", optimized_graph)
            print(f"\n[Skip] Optimized graph build (reusing {optimized_graph.name})")

    if not args.skip_scoring:
        if args.force_scoring or not baseline_nodes.exists():
            baseline_scoring_command = [
                python_exe,
                "-u",  # Unbuffered output
                str(PROJECT_ROOT / "data-pipeline" / "compute_scores.py"),
                "--graph-path",
                str(baseline_graph),
                "--poi-mapping",
                str(baseline_mapping),
                "--pois-path",
                str(PROJECT_ROOT / "data" / "raw" / "osm" / "pois.parquet"),
                "--out-dir",
                str(analysis_dir),
                "--config",
                str(PROJECT_ROOT / "config.yaml"),
                "--output-prefix",
                baseline_prefix,
            ]
            run_step(baseline_scoring_command, "Compute baseline scoring outputs")
        else:
            logger.info("Skipping baseline scoring; reusing %s", baseline_nodes)
            print(f"\n[Skip] Baseline scoring (reusing {baseline_nodes.name})")

        # ALWAYS recompute optimized scores to reflect new POI placements from GA optimization
        # Use optimized in-memory scoring (70+ seconds faster, no intermediate disk I/O)
        optimized_scoring_command = [
            python_exe,
            "-u",  # Unbuffered output
            str(PROJECT_ROOT / "optimization-pipeline" / "run_optimized_scoring.py"),
            "--graph-path",
            str(optimized_graph),
            "--poi-mapping",
            str(optimized_mapping),
            "--baseline-pois-path",
            str(PROJECT_ROOT / "data" / "raw" / "osm" / "pois.parquet"),
            "--optimized-pois-path",
            str(optimized_geojson),
            "--out-dir",
            str(analysis_dir),
            "--config",
            str(PROJECT_ROOT / "config.yaml"),
            "--output-prefix",
            optimized_prefix,
        ]
        run_step(optimized_scoring_command, "[OPTIMIZED] Compute optimized scoring outputs (in-memory merge)")
        logger.info("Optimized scoring completed with new POI placements")

    print("\n" + "=" * 80)
    print("Pipeline orchestration complete!")
    print("=" * 80)
    logger.info("Pipeline orchestration complete")
    print(f"\nKey artifacts:")
    print(f"  Baseline graph:    {baseline_graph}")
    print(f"  Optimized graph:   {optimized_graph}")
    print(f"  Baseline scores:   {baseline_nodes}")
    print(f"  Optimized scores:  {optimized_nodes}")
    print(f"  Metrics summaries: {analysis_dir / (baseline_prefix + '_metrics_summary.json')} | "
          f"{analysis_dir / (optimized_prefix + '_metrics_summary.json')}")
    if optimized_geojson.exists():
        print(f"  Optimized POIs:    {optimized_geojson}")
    # Note: merged_pois.geojson no longer created (in-memory merge saves 79 min)
    print("=" * 80)


if __name__ == "__main__":
    main()
