#!/usr/bin/env python3
"""
Satellite Graph Generation Pipeline

Automates the complete pipeline for generating road network graphs from satellite imagery:
1. Download satellite images using downloader/main.py
2. Run Sat2Graph inference using run_custom_inference.py (in venvg)
3. Build OSM-compatible graph using build_satellite_graph.py (in main venv)

Usage:
    python run_satellite_pipeline.py
    python run_satellite_pipeline.py --skip-download
    python run_satellite_pipeline.py --skip-inference
    python run_satellite_pipeline.py --output-dir custom/path
"""

import argparse
import subprocess
import sys
from pathlib import Path
import platform


def print_step(message: str):
    """Print a step header"""
    print(f"\n{'='*60}")
    print(f"{message}")
    print(f"{'='*60}\n")


def print_success(message: str):
    """Print a success message"""
    print(f"[SUCCESS] {message}")


def print_error(message: str):
    """Print an error message"""
    print(f"[ERROR] {message}")


def print_info(message: str):
    """Print an info message"""
    print(f"[INFO] {message}")


def get_python_executable(venv_path: Path) -> Path:
    """Get the Python executable path for a virtual environment"""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def run_command(cmd: list, cwd: Path = None, description: str = ""):
    """Run a command and handle errors"""
    if description:
        print_info(f"Running: {description}")
    
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            text=True,
            capture_output=False  # Show output in real-time
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed with exit code {e.returncode}")
        raise


def verify_file_exists(file_path: Path, description: str):
    """Verify that a file exists"""
    if not file_path.exists():
        print_error(f"{description} not found: {file_path}")
        sys.exit(1)
    print_success(f"Found: {file_path}")


def verify_directory_exists(dir_path: Path, description: str):
    """Verify that a directory exists"""
    if not dir_path.exists():
        print_error(f"{description} not found: {dir_path}")
        sys.exit(1)
    print_success(f"Found: {dir_path}")


def step_download_images(graphbuilder_dir: Path):
    """Step 1: Download satellite images"""
    print_step("STEP 1: Downloading Satellite Images")
    
    downloader_script = graphbuilder_dir / "downloader" / "main.py"
    verify_file_exists(downloader_script, "Downloader script")
    
    print_info("Running downloader/main.py...")
    print_info("You will be prompted to enter bounding box coordinates")
    
    # Run downloader (uses system Python, no venv needed)
    run_command(
        [sys.executable, str(downloader_script)],
        cwd=graphbuilder_dir,
        description="Image downloader"
    )
    
    print_success("Images downloaded successfully")
    
    # Verify metadata was created
    metadata_file = graphbuilder_dir / "downloader" / "metadata.json"
    verify_file_exists(metadata_file, "Metadata file")


def step_run_inference(graphbuilder_dir: Path):
    """Step 2: Run Sat2Graph inference"""
    print_step("STEP 2: Running Sat2Graph Inference")
    
    # Check venvg exists
    venvg_path = graphbuilder_dir / "venvg"
    verify_directory_exists(venvg_path, "venvg virtual environment")
    
    # Get Python executable from venvg
    python_exe = get_python_executable(venvg_path)
    verify_file_exists(python_exe, "venvg Python executable")
    
    print_success(f"Using Python: {python_exe}")
    
    # Run inference
    inference_script = graphbuilder_dir / "run_custom_inference.py"
    verify_file_exists(inference_script, "Inference script")
    
    print_info("Running run_custom_inference.py...")
    print_info("This may take a while depending on the number of tiles...")
    
    run_command(
        [str(python_exe), str(inference_script)],
        cwd=graphbuilder_dir,
        description="Sat2Graph inference"
    )
    
    print_success("Inference completed successfully")
    
    # Verify output files
    output_dir = graphbuilder_dir / "custom_outputs"
    if not output_dir.exists():
        print_error(f"Output directory not created: {output_dir}")
        sys.exit(1)
    
    graph_files = list(output_dir.glob("*_graph.p"))
    if not graph_files:
        print_error(f"No graph pickle files found in {output_dir}")
        sys.exit(1)
    
    print_success(f"Found {len(graph_files)} graph pickle files")


def step_build_graph(graphbuilder_dir: Path, project_root: Path, output_dir: str):
    """Step 3: Build OSM-compatible graph"""
    print_step("STEP 3: Building OSM-Compatible Graph")
    
    # Check main venv exists (in project root)
    venv_path = project_root / "venv"
    verify_directory_exists(venv_path, "Main virtual environment")
    
    # Get Python executable from main venv
    python_exe = get_python_executable(venv_path)
    verify_file_exists(python_exe, "Main venv Python executable")
    
    print_success(f"Using Python: {python_exe}")
    
    # Run graph builder
    builder_script = graphbuilder_dir / "build_satellite_graph.py"
    verify_file_exists(builder_script, "Graph builder script")
    
    input_dir = graphbuilder_dir / "custom_outputs"
    metadata_file = graphbuilder_dir / "downloader" / "metadata.json"
    
    print_info(f"Input: {input_dir}")
    print_info(f"Metadata: {metadata_file}")
    print_info(f"Output: {output_dir}")
    
    run_command(
        [
            str(python_exe),
            str(builder_script),
            "--input-dir", str(input_dir),
            "--metadata", str(metadata_file),
            "--output-dir", output_dir
        ],
        cwd=graphbuilder_dir,
        description="Graph builder"
    )
    
    print_success("Graph built successfully")


def print_summary(output_dir: str):
    """Print pipeline completion summary"""
    print_step("PIPELINE COMPLETE")
    
    print_success("Satellite graph generation completed successfully!")
    
    print_info("\nOutput files:")
    print(f"  - {output_dir}/graph.graphml")
    print(f"  - {output_dir}/nodes.parquet")
    print(f"  - {output_dir}/edges.parquet")
    
    print_info("\nNext steps:")
    print("  1. Review the generated graph files")
    print("  2. Ensure POI data is available (copy from OSM or provide separately)")
    print("  3. Run the main PathLens pipeline:")
    print(f"     cd ..")
    print(f"     python run_pathlens.py")
    
    print("\nAll done!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Satellite Graph Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_satellite_pipeline.py
  python run_satellite_pipeline.py --skip-download
  python run_satellite_pipeline.py --skip-inference
  python run_satellite_pipeline.py --output-dir ../data/raw/osm
        """
    )
    
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip image download step (use existing images)"
    )
    
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference step (use existing graph pickles)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/raw/osm",
        help="Output directory for the final graph (default: ../data/raw/osm)"
    )
    
    args = parser.parse_args()
    
    # Determine paths - script is now in GraphBuilder/
    graphbuilder_dir = Path(__file__).resolve().parent
    project_root = graphbuilder_dir.parent
    
    print_info(f"GraphBuilder: {graphbuilder_dir}")
    print_info(f"Project root: {project_root}")
    
    try:
        # Step 1: Download images
        if not args.skip_download:
            step_download_images(graphbuilder_dir)
        else:
            print_info("Skipping image download (using existing images)")
        
        # Step 2: Run inference
        if not args.skip_inference:
            step_run_inference(graphbuilder_dir)
        else:
            print_info("Skipping inference (using existing graph pickles)")
        
        # Step 3: Build graph
        step_build_graph(graphbuilder_dir, project_root, args.output_dir)
        
        # Print summary
        print_summary(args.output_dir)
        
    except subprocess.CalledProcessError as e:
        print_error(f"Pipeline failed at step with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print_error("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
