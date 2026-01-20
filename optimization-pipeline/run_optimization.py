import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List

# Import CityDataManager for multi-city support
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from city_paths import CityDataManager


def setup_logging(log_dir: Path):
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


def save_run_metadata(output_dir: Path, args: argparse.Namespace, cfg: dict) -> None:
    """Save metadata about the optimization run."""
    metadata = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'city': args.city,
        'mode': args.mode,
        'effective_config': cfg,
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


def run_step(description: str, command: List[str], logger) -> None:
    """Execute a subprocess while echoing context to stdout."""
    logger.info("=" * 80)
    logger.info(f"> {description}")
    logger.info("=" * 80)
    logger.info(f"Command: {' '.join(command)}")
    
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        logger.error(f"Step failed ({description}) with exit code {result.returncode}")
        raise SystemExit(f"Step failed ({description}) with exit code {result.returncode}.")
    
    logger.info(f"{description} completed successfully")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the PathLens optimization stack using centralized configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Core execution flags
    parser.add_argument("--city", default="bangalore", help="City to process")
    parser.add_argument("--mode", default="ga_only", choices=["ga_only", "ga_milp", "ga_milp_pnmlr"], help="Optimization mode")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all steps")
    
    # Step control (overrides config)
    parser.add_argument("--skip-candidates", action="store_true", help="Skip candidate generation")
    parser.add_argument("--skip-ga", action="store_true", help="Skip optimization search")
    parser.add_argument("--skip-map-refresh", action="store_true", help="Skip map generation")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip final pipeline rebuild")
    
    args = parser.parse_args()
    
    # Initialize CDM and load config
    cdm = CityDataManager(args.city, project_root=project_root, mode=args.mode)
    cfg = cdm.load_config()
    
    # Resolve directories
    runs_dir = cdm.optimized_dir(args.mode)
    analysis_dir = cdm.baseline_dir
    optimization_dir = Path(__file__).parent
    
    # Setup logging
    logger = setup_logging(runs_dir / "logs")
    logger.info(f"Starting PathLens Optimization: {args.city} ({args.mode})")
    
    # Save metadata
    save_run_metadata(runs_dir, args, cfg)
    
    python_exe = sys.executable
    nodes_scores = cdm.baseline_nodes
    high_travel_csv = cdm.high_travel_nodes(args.mode)
    
    # Get settings from config
    opt_cfg = cfg.get('optimization', {})
    
    # Step 0: Train PNMLR model (if required and mode is ga_milp_pnmlr)
    if args.mode == 'ga_milp_pnmlr':
        pnmlr_cfg = cfg.get('pnmlr', {})
        models_dir = Path(pnmlr_cfg.get('models_dir', 'models'))
        if not models_dir.is_absolute():
            models_dir = project_root / models_dir
        model_path = models_dir / args.city / "pnmlr_model.pkl"
        
        should_train = args.force or not model_path.exists()
        if should_train:
            train_command = [
                python_exe,
                str(optimization_dir / "train_pnmlr.py"),
                "--city", args.city
            ]
            run_step("Train PNMLR model", train_command, logger)
        else:
            logger.info("PNMLR model exists and force not requested. Skipping training.")

    # Step 1: Extract optimization candidates
    skip_candidates = args.skip_candidates or opt_cfg.get('skip_candidates', False)
    if not skip_candidates:
        candidate_command = [
            python_exe,
            str(optimization_dir / "list_optimizable_nodes.py"),
            "--city", args.city,
            "--mode", args.mode
        ]
        if args.force: candidate_command.append("--force")
        run_step("Extract high travel time nodes", candidate_command, logger)
    
    # Step 2: Run hybrid GA
    skip_ga = args.skip_ga or opt_cfg.get('skip_ga', False)
    if not skip_ga:
        ga_command = [
            python_exe,
            str(optimization_dir / "hybrid_ga.py"),
            "--city", args.city,
            "--mode", args.mode
        ]
        if args.force: ga_command.append("--force") 
        # hybrid_ga.py does not accept --force (config driven)
        if "--force" in ga_command:
            ga_command.remove("--force")
        run_step("Run hybrid genetic algorithm", ga_command, logger)
    
    # Step 3: Generate solution visualization
    skip_map = args.skip_map_refresh or opt_cfg.get('skip_map_refresh', False)
    if not skip_map:
        map_command = [
            python_exe, 
            str(optimization_dir / "generate_solution_map.py"),
            "--city", args.city,
            "--mode", args.mode,
            "--skip-metrics"  # Skip scoring here - run_optimized_scoring.py handles it correctly
        ]
        run_step("Generate solution map", map_command, logger)
    
    # Step 4: Run comparative pipeline (prefixed graphs/scores)
    skip_pipeline = args.skip_pipeline or opt_cfg.get('skip_pipeline', False)
    if not skip_pipeline:
        pipeline_command = [
            python_exe,
            str(optimization_dir / "run_optimized_pipeline.py"),
            "--city", args.city,
            "--mode", args.mode,
            "--skip-baseline"  # Baseline already run by run_baseline_prep.py
        ]
        if args.force: pipeline_command.append("--force")
        run_step("Run full optimized pipeline rebuild", pipeline_command, logger)

    logger.info("=" * 80)
    logger.info(f"Optimization workflow complete for {args.city}")
    logger.info(f"  Results: {runs_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
