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
from typing import List
import json


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging for master orchestrator"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pathlens_master_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('PathLensMaster')
    logger.info(f"Logging initialized: {log_file}")
    return logger


def run_pipeline(pipeline_name: str, script_path: Path, args: List[str], logger: logging.Logger) -> bool:
    """Run a pipeline script with arguments"""
    logger.info("=" * 80)
    logger.info(f"PIPELINE: {pipeline_name.upper()}")
    logger.info("=" * 80)
    
    command = [sys.executable, str(script_path)] + args
    logger.info(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, text=True)
        logger.info(f"✓ {pipeline_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {pipeline_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"✗ {pipeline_name} encountered error: {e}")
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
    
    # Track results
    results = {}
    
    # Pipeline 1: Data Collection
    if run_data:
        data_script = project_root / 'data-pipeline' / 'run_pipeline.py'
        data_args = ['--place', args.place]
        if args.force:
            data_args.append('--force')
        # Note: run_pipeline.py uses hardcoded config.yaml path, doesn't accept --config arg
        
        results['data'] = run_pipeline('data-collection', data_script, data_args, logger)
        
        if not results['data']:
            logger.error("Data pipeline failed. Stopping execution.")
            sys.exit(1)
    else:
        logger.info("Skipping data collection pipeline")
    
    # Pipeline 2: Optimization
    if run_optimization:
        opt_script = project_root / 'optimization-pipeline' / 'run_optimization.py'
        opt_args = []
        if args.ga_population:
            opt_args.extend(['--ga-population', str(args.ga_population)])
        if args.ga_generations:
            opt_args.extend(['--ga-generations', str(args.ga_generations)])
        if config_path:
            opt_args.extend(['--ga-config', str(config_path)])
        
        results['optimization'] = run_pipeline('optimization', opt_script, opt_args, logger)
        
        if not results['optimization']:
            logger.warning("Optimization pipeline failed. Continuing to landuse if requested.")
    else:
        logger.info("Skipping optimization pipeline")
    
    # Pipeline 3: Landuse Feasibility
    if run_landuse:
        if args.pipeline == 'landuse' and not args.amenity:
            logger.error("Landuse pipeline requires --amenity argument")
            sys.exit(1)
        
        landuse_script = project_root / 'landuse-pipeline' / 'run_feasibility.py'
        landuse_args = []
        if args.amenity:
            landuse_args.append(args.amenity)
        elif args.pipeline == 'all':
            landuse_args.append('--all')
        
        results['landuse'] = run_pipeline('landuse-feasibility', landuse_script, landuse_args, logger)
    else:
        logger.info("Skipping landuse feasibility pipeline")
    
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
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {pipeline:20} | {status}")
    logger.info("=" * 80)
    logger.info(f"Summary saved: {summary_file}")
    
    # Exit with appropriate code
    if all(results.values()):
        logger.info("✓ All pipelines completed successfully")
        sys.exit(0)
    else:
        logger.error("✗ Some pipelines failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
