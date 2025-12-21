#!/usr/bin/env python3
"""End-to-end runner for the PathLens optimization workflow.

It executes the following sequence from the repository root:
 1. ``optimization/list_optimizable_nodes.py`` to refresh high-travel nodes
 2. ``optimization/hybrid_ga.py`` to search for improved amenity placements
    (with optional MILP refinement if enabled in config)
 3. ``optimization/generate_solution_map.py`` to export GeoJSON + HTML map
 4. ``optimization/run_optimized_pipeline.py`` to rebuild prefixed graphs/scores

Each step can be skipped or customised via CLI flags; see ``--help``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import json
import yaml
from pathlib import Path
from typing import List
from datetime import datetime


def setup_logging(log_dir: Path) -> None:
    """Setup logging for the optimization run."""
    import logging
    
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"optimization_run_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: {log_file}")
    return logger


def save_run_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    """Save metadata about the optimization run."""
    # Convert arguments to JSON-serializable format
    args_dict = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
        else:
            args_dict[key] = value
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'arguments': args_dict,
        'python_executable': str(args.python),
        'steps_executed': {
            'candidates': not args.skip_candidates,
            'ga': not args.skip_ga,
            'map_refresh': not args.skip_map_refresh,
            'pipeline': not args.skip_pipeline
        }
    }
    
    metadata_file = output_dir / "run_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
        json.dump(metadata, f, indent=2)


def check_hybrid_milp_enabled(config_path: Path) -> bool:
    """Check if hybrid MILP is enabled in config."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('hybrid_milp', {}).get('enabled', False)
    except Exception:
        return False


def run_step(description: str, command: List[str], logger) -> None:
    """Execute a subprocess while echoing context to stdout."""
    logger.info("=" * 80)
    logger.info(f"▶ {description}")
    logger.info("=" * 80)
    logger.info(f"Command: {' '.join(command)}")
    
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        logger.error(f"Step failed ({description}) with exit code {result.returncode}")
        raise SystemExit(f"Step failed ({description}) with exit code {result.returncode}.")
    
    logger.info(f"✔ {description} completed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the PathLens optimization stack from the repository root.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full optimization with hybrid MILP (if enabled in config)
  python run_optimization.py
  
  # Skip candidate generation (use existing high_travel_time_nodes.csv)
  python run_optimization.py --skip-candidates
  
  # Custom GA parameters with hybrid MILP
  python run_optimization.py --ga-population 100 --ga-generations 75
  
  # Resume after GA crash (skip completed steps)
  python run_optimization.py --skip-candidates --skip-ga
        """
    )
    
    parser.add_argument("--python", type=Path, default=Path(sys.executable), 
                       help="Python interpreter used for all subprocesses")

    # Step control
    parser.add_argument("--skip-candidates", action="store_true", 
                       help="Skip regenerating high-travel nodes (list_optimizable_nodes.py)")
    parser.add_argument("--skip-ga", action="store_true", 
                       help="Skip the hybrid GA search (hybrid_ga.py)")
    parser.add_argument("--skip-map-refresh", action="store_true", 
                       help="Skip refreshing the combined POI map (generate_solution_map.py)")
    parser.add_argument("--skip-pipeline", action="store_true", 
                       help="Skip the prefixed pipeline rebuild (run_optimized_pipeline.py)")

    # Candidate extraction parameters
    parser.add_argument("--candidate-threshold", type=float, default=None, 
                       help="Override travel time threshold in minutes for optimisation candidates")
    parser.add_argument("--candidate-limit", type=int, default=None, 
                       help="Limit number of candidate rows written")

    # GA parameters
    parser.add_argument("--ga-population", type=int, default=None, 
                       help="Population size override for hybrid GA")
    parser.add_argument("--ga-generations", type=int, default=None, 
                       help="Generation count override for hybrid GA")
    parser.add_argument("--ga-workers", type=int, default=None, 
                       help="Worker thread count override for hybrid GA")
    parser.add_argument("--ga-random-seed", type=int, default=None, 
                       help="Seed override for hybrid GA")
    parser.add_argument("--ga-config", type=Path, default=None, 
                       help="Alternate YAML config for hybrid GA")

    # Map generation parameters
    parser.add_argument("--map-skip-metrics", action="store_true", 
                       help="Pass --skip-metrics to generate_solution_map.py")

    # Pipeline parameters
    parser.add_argument("--pipeline-refresh-map", action="store_true", 
                       help="Allow run_optimized_pipeline.py to refresh the GA map step")
    parser.add_argument("--pipeline-skip-map-metrics", action="store_true", 
                       help="Pass --skip-map-metrics to run_optimized_pipeline.py")
    parser.add_argument("--pipeline-force-graph", action="store_true", 
                       help="Force graph rebuild in run_optimized_pipeline.py")
    parser.add_argument("--pipeline-force-scoring", action="store_true", 
                       help="Force scoring recomputation in run_optimized_pipeline.py")
    parser.add_argument("--pipeline-skip-graph", action="store_true", 
                       help="Skip graph rebuild in run_optimized_pipeline.py")
    parser.add_argument("--pipeline-skip-scoring", action="store_true", 
                       help="Skip scoring recomputation in run_optimized_pipeline.py")
    parser.add_argument("--baseline-prefix", default="baseline", 
                       help="Filename prefix for baseline artifacts")
    parser.add_argument("--optimized-prefix", default="optimized", 
                       help="Filename prefix for optimized artifacts")
    
    # Logging
    parser.add_argument("--log-dir", type=Path, default=None,
                       help="Directory for run logs (default: optimization/runs/logs)")
    
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    optimization_dir = project_root / "optimization"
    runs_dir = optimization_dir / "runs"
    analysis_dir = project_root / "data" / "analysis"
    
    # Setup logging
    log_dir = args.log_dir or (runs_dir / "logs")
    logger = setup_logging(log_dir)
    
    logger.info("PathLens Optimization Workflow")
    logger.info(f"Project root: {project_root}")
    
    # Save run metadata
    save_run_metadata(runs_dir, args)
    
    # Check hybrid MILP status
    config_path = args.ga_config or (project_root / "config.yaml")
    hybrid_milp_enabled = check_hybrid_milp_enabled(config_path)
    
    if hybrid_milp_enabled:
        logger.info("Hybrid GA-MILP enabled in config.yaml")
    else:
        logger.info("Hybrid GA-MILP disabled (pure GA mode)")

    python_exe = str(args.python)
    nodes_scores = analysis_dir / "nodes_with_scores.parquet"
    high_travel_csv = optimization_dir / "high_travel_time_nodes.csv"

    # Step 1: Extract optimization candidates
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
        
        run_step("Extract high travel time nodes", candidate_command, logger)
    else:
        logger.info("Skipping candidate extraction (--skip-candidates)")
        if not high_travel_csv.exists():
            logger.warning(f"Candidate file not found: {high_travel_csv}")

    # Step 2: Run hybrid GA (with optional MILP refinement)
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
        
        step_desc = "Run hybrid genetic algorithm"
        if hybrid_milp_enabled:
            step_desc += " with MILP refinement"
        
        run_step(step_desc, ga_command, logger)
    else:
        logger.info("Skipping hybrid GA (--skip-ga)")
        best_candidate = runs_dir / "best_candidate.json"
        if not best_candidate.exists():
            logger.warning(f"Best candidate file not found: {best_candidate}")

    # Step 3: Generate solution visualization
    if not args.skip_map_refresh:
        map_command: List[str] = [
            python_exe, 
            str(optimization_dir / "generate_solution_map.py")
        ]
        if args.map_skip_metrics:
            map_command.append("--skip-metrics")
        
        run_step("Generate combined POI GeoJSON and map", map_command, logger)
    else:
        logger.info("Skipping map refresh (--skip-map-refresh)")

    # Step 4: Run comparative pipeline
    orchestrator = optimization_dir / "run_optimized_pipeline.py"
    if not orchestrator.exists():
        logger.error(f"Expected orchestrator not found at {orchestrator}")
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
        
        run_step("Run optimization pipeline orchestrator", pipeline_command, logger)
    else:
        logger.info("Skipping prefixed pipeline rebuild (--skip-pipeline)")

    logger.info("=" * 80)
    logger.info("✓ Optimization workflow complete")
    logger.info(f"  Results: {runs_dir}")
    logger.info(f"  Analysis: {analysis_dir}")
    logger.info(f"  Logs: {log_dir}")
    
    if hybrid_milp_enabled:
        milp_stats = runs_dir / "milp_refinement_stats.json"
        if milp_stats.exists():
            logger.info(f"  MILP stats: {milp_stats}")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
