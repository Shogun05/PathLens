#!/usr/bin/env python3
"""Orchestrate baseline and optimized PathLens pipeline runs."""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

# Add project root for CityDataManager
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from city_paths import CityDataManager

LOGGER_NAME = "PathLensOptimizedPipeline"
logger = logging.getLogger(LOGGER_NAME)

def _stream_command(command: Sequence[str], prefix: str = "") -> int:
    """Run a command and stream output."""
    with subprocess.Popen(
        command,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip()
            if stripped:
                msg = f"[{prefix}] {stripped}" if prefix else stripped
                print(msg)
                logger.info(msg)
        process.wait()
        return process.returncode if process.returncode is not None else -1

def run_step(command: Sequence[str], description: str) -> None:
    """Execute a subprocess command."""
    print("\n" + "=" * 80)
    print(f"Executing: {description}")
    print("=" * 80)
    
    start_time = time.time()
    return_code = _stream_command(command)
    elapsed = time.time() - start_time
    
    if return_code != 0:
        logger.error(f"Step '{description}' failed code {return_code}")
        raise SystemExit(f"Step failed ({description})")
    
    print(f"Completed in {elapsed:.1f}s")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline & optimized pipeline passes.")
    parser.add_argument("--city", default="bangalore", help="City to process")
    parser.add_argument("--mode", default="ga_only", choices=["ga_only", "ga_milp", "ga_milp_pnmlr"], help="Optimization mode")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all steps")
    parser.add_argument("--force-graph", action="store_true", help="Force rebuild graph")
    parser.add_argument("--force-scoring", action="store_true", help="Force recompute scoring")
    parser.add_argument("--skip-map", action="store_true", help="Skip map refresh")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph building")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip scoring")
    parser.add_argument("--skip-baseline", action="store_true", 
                        help="Skip baseline steps (already run by run_baseline_prep.py)")
    
    args = parser.parse_args()
    if args.force:
        args.force_graph = True
        args.force_scoring = True
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cdm = CityDataManager(args.city, project_root=project_root, mode=args.mode)
    cfg = cdm.load_config()
    
    python_exe = sys.executable

    # 1. Generate map and optimized POIs
    if not args.skip_map:
        map_cmd = [
            python_exe,
            str(project_root / "optimization-pipeline" / "generate_solution_map.py"),
            "--city", args.city,
            "--mode", args.mode
        ]
        run_step(map_cmd, "Generate optimized POIs & map")

    # 2. Build graphs (baseline and optimized)
    if not args.skip_graph:
        # Baseline (skip if --skip-baseline is set)
        if not args.skip_baseline:
            baseline_graph_cmd = [
                python_exe,
                str(project_root / "data-pipeline" / "build_graph.py"),
                "--graph-path", str(cdm.raw_graph),
                "--pois-path", str(cdm.raw_pois),
                "--out-dir", str(cdm.processed_dir),
                "--output-prefix", "baseline"
            ]
            run_step(baseline_graph_cmd, "Build baseline graph")
        else:
            print("Skipping baseline graph (--skip-baseline)")

        # Optimized
        optimized_graph_cmd = [
            python_exe,
            str(project_root / "data-pipeline" / "build_graph.py"),
            "--graph-path", str(cdm.raw_graph),
            "--pois-path", str(cdm.raw_pois),
            "--optimized-pois-path", str(cdm.optimized_pois(args.mode)),
            "--out-dir", str(cdm.processed_dir),
            "--output-prefix", "optimized"
        ]
        run_step(optimized_graph_cmd, "Build optimized graph")

    # 3. Compute scores
    if not args.skip_scoring:
        # Baseline (skip if --skip-baseline is set)
        if not args.skip_baseline:
            baseline_scoring_cmd = [
                python_exe,
                str(project_root / "data-pipeline" / "compute_scores.py"),
                "--city", args.city,
                # compute_scores.py now resolves output-prefix from args if possible, 
                # but we might need to be explicit if it doesn't handle 'baseline' vs 'optimized' internally.
                # Actually, compute_scores.py uses --output-prefix.
                "--output-prefix", "baseline"
            ]
            run_step(baseline_scoring_cmd, "Compute baseline scores")
        else:
            print("Skipping baseline scoring (--skip-baseline)")

        # Optimized (using the specialized in-memory merge script)
        optimized_scoring_cmd = [
            python_exe,
            str(project_root / "optimization-pipeline" / "run_optimized_scoring.py"),
            "--city", args.city,
            "--mode", args.mode
        ]
        run_step(optimized_scoring_cmd, "Compute optimized scores (in-memory)")

    print("\nOrchestration complete!")

if __name__ == "__main__":
    main()
