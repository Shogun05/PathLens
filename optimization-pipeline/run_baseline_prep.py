#!/usr/bin/env python3
"""Run baseline graph build and scoring (mode-agnostic, runs once before mode loop)."""
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

LOGGER_NAME = "BaselinePrep"
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
    return_code = _stream_command(command, description.split()[0])
    elapsed = time.time() - start_time
    
    if return_code != 0:
        logger.error(f"Step '{description}' failed with code {return_code}")
        raise SystemExit(f"Step failed ({description})")
    
    print(f"Completed in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline graph build and scoring (mode-agnostic, runs once)."
    )
    parser.add_argument("--city", default="bangalore", help="City to process")
    parser.add_argument("--force", action="store_true", help="Force recomputation")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph building")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip scoring")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    cdm = CityDataManager(args.city, project_root=project_root)
    python_exe = sys.executable
    
    logger.info(f"=== Baseline Preparation for {args.city} ===")
    
    # 1. Build baseline graph
    if not args.skip_graph:
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
        logger.info("Skipping baseline graph build (--skip-graph)")
    
    # 2. Compute baseline scores
    if not args.skip_scoring:
        baseline_scoring_cmd = [
            python_exe,
            str(project_root / "data-pipeline" / "compute_scores.py"),
            "--city", args.city,
            "--output-prefix", "baseline"
        ]
        run_step(baseline_scoring_cmd, "Compute baseline scores")
    else:
        logger.info("Skipping baseline scoring (--skip-scoring)")
    
    logger.info("=== Baseline preparation complete ===")
    logger.info(f"  Graph: {cdm.processed_dir}/baseline_*.parquet")
    logger.info(f"  Scores: {cdm.baseline_dir}/metrics_summary.json")


if __name__ == "__main__":
    main()
