import argparse
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
from city_paths import CityDataManager


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging for master orchestrator"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pathlens_master_{timestamp}.log"

    logger = logging.getLogger('PathLensMaster')
    logger.setLevel(logging.INFO)
    logger.propagate = False 

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(f"Logging initialized: {log_file}")
    return logger


def run_pipeline(pipeline_name: str, script_path: Path, args: List[str], logger: logging.Logger) -> bool:
    """Run a pipeline script with arguments and live-stream its logs."""
    logger.info("=" * 80)
    logger.info(f"PIPELINE: {pipeline_name.upper()}")
    logger.info("=" * 80)

    command = [sys.executable, '-u', str(script_path)] + args
    logger.info(f"Command: {' '.join(command)}")

    try:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                stripped = line.rstrip()
                if stripped:
                    prefixed = f"[{pipeline_name}] {stripped}"
                    print(prefixed)
                    logger.info(prefixed)
                elif line:
                    print(line, end='')

            returncode = process.wait()

        if returncode == 0:
            logger.info(f"{pipeline_name} completed successfully")
            return True

        logger.error(f"{pipeline_name} failed with exit code {returncode}")
        return False
    except Exception as e:
        logger.error(f"{pipeline_name} encountered error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="PathLens Master Orchestrator - Coordinate all pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run by config name (loads configs/mumbai.yaml and merges with base.yaml)
  python run_pathlens.py --config mumbai
  
  # Run with explicit flags (overrides config settings)
  python run_pathlens.py --city bangalore --mode ga_milp
        """
    )
    
    # Config path/name
    parser.add_argument('--config', type=str, default=None,
                       help='Config name in configs/ (e.g., "mumbai") or direct YAML path')

    # Pipeline selection
    parser.add_argument('--pipeline', 
                       choices=['all', 'data', 'optimization'],
                       default=None,
                       help='Which pipeline(s) to run')
    parser.add_argument('--skip-data', action='store_true', default=None,
                       help='Skip data collection pipeline')
    parser.add_argument('--skip-optimization', action='store_true', default=None,
                       help='Skip optimization pipeline')
    
    # Shared arguments
    parser.add_argument('--force', action='store_true', default=None,
                       help='Force recomputation of all steps')
    parser.add_argument('--log-dir', type=Path, default=None,
                       help='Directory for logs (default: data/logs)')
    parser.add_argument('--city', default=None,
                       help='City to process (overrides config)')
    parser.add_argument('--mode', default='all',
                        choices=['all', 'ga_only', 'ga_milp', 'ga_milp_pnmlr'],
                        help='Optimization mode (overrides config)')
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent
    
    # Initialize CityDataManager
    # If --config is used as a name, we use it as the city if --city is not provided
    city_name = args.city or args.config or 'bangalore'
    # Use 'ga_only' as safe default when mode is 'all' or unspecified
    initial_mode = args.mode if args.mode and args.mode != 'all' else 'ga_only'
    cdm = CityDataManager(city_name, project_root=project_root, mode=initial_mode)
    
    # Load and merge config
    cfg = cdm.load_config()
    
    # Resolve actual values from merged config
    project_cfg = cfg.get('project', {})
    pipeline_cfg = cfg.get('pipeline', {})
    
    city = args.city or project_cfg.get('city', city_name)
    mode = args.mode or project_cfg.get('mode', 'ga_only')
    
    requested_pipeline = args.pipeline or pipeline_cfg.get('pipeline', 'all')
    skip_data = args.skip_data if args.skip_data is not None else pipeline_cfg.get('skip_data', False)
    skip_optimization = args.skip_optimization if args.skip_optimization is not None else pipeline_cfg.get('skip_optimization', False)
    force = args.force if args.force is not None else pipeline_cfg.get('force', False)
    
    # Setup logging
    log_dir = args.log_dir or (project_root / 'data' / 'logs')
    logger = setup_logging(log_dir)
    logger.info("PathLens Master Orchestrator Starting")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Active City: {city}, Mode: {mode}")
    
    # Determine which pipelines to run
    run_data = requested_pipeline in ['all', 'data'] and not skip_data
    run_optimization = requested_pipeline in ['all', 'optimization'] and not skip_optimization

    progress_path = project_root / 'data' / 'optimization' / 'runs' / 'progress.json'
    stage_percents = {
        'initializing': 5,
        'data': 30,
        'optimization': 90,
        'finalizing': 100
    }
    pipeline_states: Dict[str, str] = {
        'data': 'pending' if run_data else 'skipped',
        'optimization': 'pending' if run_optimization else 'skipped'
    }

    def write_progress(
        stage: str,
        overall_status: str,
        message: str,
        percent: Optional[int] = None,
        extra: Optional[Dict[str, object]] = None
    ) -> None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {
            'status': overall_status,
            'stage': stage,
            'message': message,
            'percent': percent if percent is not None else stage_percents.get(stage, 0),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'pipelines': pipeline_states.copy()
        }
        if extra:
            payload['details'] = extra
        with open(progress_path, 'w') as progress_file:
            json.dump(payload, progress_file, indent=2)

    write_progress(
        stage='initializing',
        overall_status='running',
        message='Starting PathLens orchestrator',
        percent=stage_percents['initializing'],
        extra={'requested_pipeline': requested_pipeline}
    )
    
    # Track results
    results = {}
    
    # Pipeline 1: Data Collection
    if run_data:
        pipeline_states['data'] = 'running'
        write_progress(
            stage='data',
            overall_status='running',
            message='Running data collection pipeline',
            percent=stage_percents['data']
        )

        data_script = project_root / 'data-pipeline' / 'run_pipeline.py'
        data_args = ['--city', city]
        if force:
            data_args.append('--force')
        
        results['data'] = run_pipeline('data-collection', data_script, data_args, logger)

        pipeline_states['data'] = 'completed' if results['data'] else 'failed'
        write_progress(
            stage='data',
            overall_status='running' if results['data'] else 'failed',
            message='Data pipeline completed' if results['data'] else 'Data pipeline failed',
            percent=stage_percents['data']
        )
        
        if not results['data']:
            logger.error("Data pipeline failed. Stopping execution.")
            write_progress(
                stage='finalizing',
                overall_status='failed',
                message='Data pipeline failed. Stopping execution.',
                percent=stage_percents['finalizing'],
                extra={'failed_stage': 'data'}
            )
            sys.exit(1)
    else:
        logger.info("Skipping data collection pipeline")
        write_progress(
            stage='data',
            overall_status='running',
            message='Data pipeline skipped (not requested)',
            percent=stage_percents['data']
        )
    
    # Pipeline 2: Optimization
    if run_optimization:
        pipeline_states['optimization'] = 'running'
        write_progress(
            stage='optimization',
            overall_status='running',
            message='Running optimization pipeline',
            percent=stage_percents['optimization']
        )

        modes_to_run = ['ga_only', 'ga_milp', 'ga_milp_pnmlr'] if mode == 'all' else [mode]
        optimization_success = True
        
        # Step 2a: Run baseline preparation ONCE (before mode loop)
        if len(modes_to_run) > 1:
            logger.info("Running baseline preparation (one-time, shared by all modes)")
            baseline_prep_script = project_root / 'optimization-pipeline' / 'run_baseline_prep.py'
            baseline_args = ['--city', city]
            if force:
                baseline_args.append('--force')
            
            baseline_success = run_pipeline('baseline-preparation', baseline_prep_script, baseline_args, logger)
            if not baseline_success:
                logger.error("Baseline preparation failed. Stopping execution.")
                optimization_success = False
        
        # Step 2b: Run mode-specific optimization for each mode
        if optimization_success:
            for current_mode in modes_to_run:
                logger.info(f"Starting optimization run for mode: {current_mode}")
                opt_script = project_root / 'optimization-pipeline' / 'run_optimization.py'
                opt_args = ['--city', city, '--mode', current_mode]
                if force:
                    opt_args.append('--force')
                
                success = run_pipeline(f'optimization-{current_mode}', opt_script, opt_args, logger)
                if not success:
                    optimization_success = False
                    logger.error(f"Optimization failed for mode {current_mode}")
                    # We stop on failure to allow debugging, rather than continuing to next mode
                    break
        
        results['optimization'] = optimization_success

        pipeline_states['optimization'] = 'completed' if results['optimization'] else 'failed'
        write_progress(
            stage='optimization',
            overall_status='running',
            message='Optimization pipeline completed' if results['optimization'] else 'Optimization pipeline failed',
            percent=stage_percents['optimization']
        )
        
        if not results['optimization']:
            logger.warning("Optimization pipeline failed.")
    else:
        logger.info("Skipping optimization pipeline")
        write_progress(
            stage='optimization',
            overall_status='running',
            message='Optimization pipeline skipped (not requested)',
            percent=stage_percents['optimization']
        )
    
    # Save run summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'pipelines_executed': list(results.keys()),
        'results': results,
        'effective_config': cfg
    }
    
    summary_file = log_dir / f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Final report
    logger.info("=" * 80)
    logger.info("PATHLENS MASTER ORCHESTRATOR - SUMMARY")
    logger.info("=" * 80)
    for pipeline, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {pipeline:20} | {status}")
    logger.info("=" * 80)
    logger.info(f"Summary saved: {summary_file}")
    
    # Exit with appropriate code
    overall_success = all(results.values()) if results else True
    final_status = 'completed' if overall_success else 'failed'
    final_message = "All requested pipelines completed successfully" if overall_success else "One or more pipelines failed"
    write_progress(
        stage='finalizing',
        overall_status=final_status,
        message=final_message,
        percent=stage_percents['finalizing'],
        extra={'results': results, 'summary_file': str(summary_file)}
    )

    if overall_success:
        logger.info("All pipelines completed successfully")
        sys.exit(0)
    else:
        logger.error("Some pipelines failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
