#!/usr/bin/env python3
"""
PathLens Master Orchestrator

Coordinates all three pipelines:
1. data-pipeline: OSM data collection and graph processing
2. optimization-pipeline: GA+MILP amenity placement optimization
3. landuse-pipeline: GEE feasibility analysis

Usage:
    python run_pathlens.py --pipeline all                    # Run full workflow
    python run_pathlens.py --pipeline data                   # Data collection only
    python run_pathlens.py --pipeline optimization           # Optimization only
    python run_pathlens.py --pipeline landuse hospital       # GEE analysis for hospital
    python run_pathlens.py --skip-data --skip-optimization   # Run landuse only
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import json


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging for master orchestrator"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pathlens_master_{timestamp}.log"

    logger = logging.getLogger('PathLensMaster')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate uvicorn/stdout logs

    # Clear existing handlers so repeated runs still attach the latest file
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

    # Use -u flag for unbuffered output so print statements flow through immediately
    command = [sys.executable, '-u', str(script_path)] + args
    logger.info(f"Command: {' '.join(command)}")

    try:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        ) as process:
            assert process.stdout is not None  # for type checkers
            for line in process.stdout:
                stripped = line.rstrip()
                if stripped:  # Only log non-empty lines
                    prefixed = f"[{pipeline_name}] {stripped}"
                    print(prefixed)  # Print to console
                    logger.info(prefixed)  # Log to file
                elif line:  # Preserve empty lines for formatting
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
  # Full workflow (all pipelines)
  python run_pathlens.py --pipeline all
  
  # Individual pipelines
  python run_pathlens.py --pipeline data
  python run_pathlens.py --pipeline optimization
  python run_pathlens.py --pipeline landuse hospital
  
  # Skip specific pipelines
  python run_pathlens.py --skip-data
  python run_pathlens.py --skip-optimization --skip-landuse
  
  # Pass custom config
  python run_pathlens.py --config custom_config.yaml
        """
    )
    
    # Pipeline selection
    parser.add_argument('--pipeline', 
                       choices=['all', 'data', 'optimization', 'landuse'],
                       default='all',
                       help='Which pipeline(s) to run')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data collection pipeline')
    parser.add_argument('--skip-optimization', action='store_true',
                       help='Skip optimization pipeline')
    parser.add_argument('--skip-landuse', action='store_true',
                       help='Skip landuse feasibility pipeline')
    
    # Shared arguments
    parser.add_argument('--config', type=Path, default=None,
                       help='Custom config file (default: config.yaml)')
    parser.add_argument('--force', action='store_true',
                       help='Force recomputation of all steps')
    parser.add_argument('--log-dir', type=Path, default=None,
                       help='Directory for logs (default: data/logs)')
    
    # Data pipeline arguments
    parser.add_argument('--place', default='Bangalore, India',
                       help='Place name for OSM data collection')
    
    # Optimization pipeline arguments
    parser.add_argument('--ga-population', type=int, default=None,
                       help='GA population size')
    parser.add_argument('--ga-generations', type=int, default=None,
                       help='GA generation count')
    
    # Landuse pipeline arguments
    parser.add_argument('amenity', nargs='?', default=None,
                       help='Amenity type for landuse analysis (hospital, school, etc.)')
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent
    log_dir = args.log_dir or (project_root / 'data' / 'logs')
    config_path = args.config or (project_root / 'config.yaml')
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("PathLens Master Orchestrator Starting")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config: {config_path}")
    
    # Determine which pipelines to run
    run_data = args.pipeline in ['all', 'data'] and not args.skip_data
    run_optimization = args.pipeline in ['all', 'optimization'] and not args.skip_optimization
    run_landuse = args.pipeline in ['all', 'landuse'] and not args.skip_landuse

    progress_path = project_root / 'data' / 'optimization' / 'runs' / 'progress.json'
    stage_percents = {
        'initializing': 5,
        'data': 30,
        'optimization': 80,
        'landuse': 95,
        'finalizing': 100
    }
    pipeline_states: Dict[str, str] = {
        'data': 'pending' if run_data else 'skipped',
        'optimization': 'pending' if run_optimization else 'skipped',
        'landuse': 'pending' if run_landuse else 'skipped'
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
        extra={'requested_pipeline': args.pipeline}
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
        data_args = ['--place', args.place]
        if args.force:
            data_args.append('--force')
        # Note: run_pipeline.py uses hardcoded config.yaml path, doesn't accept --config arg
        
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

        opt_script = project_root / 'optimization-pipeline' / 'run_optimization.py'
        opt_args = []
        if args.ga_population:
            opt_args.extend(['--ga-population', str(args.ga_population)])
        if args.ga_generations:
            opt_args.extend(['--ga-generations', str(args.ga_generations)])
        if config_path:
            opt_args.extend(['--ga-config', str(config_path)])
        
        results['optimization'] = run_pipeline('optimization', opt_script, opt_args, logger)

        pipeline_states['optimization'] = 'completed' if results['optimization'] else 'failed'
        write_progress(
            stage='optimization',
            overall_status='running',
            message='Optimization pipeline completed' if results['optimization'] else 'Optimization pipeline failed',
            percent=stage_percents['optimization']
        )
        
        if not results['optimization']:
            logger.warning("Optimization pipeline failed. Continuing to landuse if requested.")
    else:
        logger.info("Skipping optimization pipeline")
        write_progress(
            stage='optimization',
            overall_status='running',
            message='Optimization pipeline skipped (not requested)',
            percent=stage_percents['optimization']
        )
    
    # Pipeline 3: Landuse Feasibility
    if run_landuse:
        if args.pipeline == 'landuse' and not args.amenity:
            logger.error("Landuse pipeline requires --amenity argument")
            sys.exit(1)
        pipeline_states['landuse'] = 'running'
        write_progress(
            stage='landuse',
            overall_status='running',
            message='Running landuse feasibility pipeline',
            percent=stage_percents['landuse']
        )
        
        landuse_script = project_root / 'landuse-pipeline' / 'run_feasibility.py'
        landuse_args = []
        if args.amenity:
            landuse_args.append(args.amenity)
        elif args.pipeline == 'all':
            landuse_args.append('--all')
        
        results['landuse'] = run_pipeline('landuse-feasibility', landuse_script, landuse_args, logger)
        pipeline_states['landuse'] = 'completed' if results['landuse'] else 'failed'
        write_progress(
            stage='landuse',
            overall_status='running',
            message='Landuse pipeline completed' if results.get('landuse') else 'Landuse pipeline failed',
            percent=stage_percents['landuse']
        )
    else:
        logger.info("Skipping landuse feasibility pipeline")
        write_progress(
            stage='landuse',
            overall_status='running',
            message='Landuse pipeline skipped (not requested)',
            percent=stage_percents['landuse']
        )
    
    # Save run summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'pipelines_executed': list(results.keys()),
        'results': results,
        'config': str(config_path),
        'arguments': vars(args)
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
