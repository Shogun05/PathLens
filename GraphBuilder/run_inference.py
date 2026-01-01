#!/usr/bin/env python3
"""
Run Sat2Graph inference on the 20cities dataset using the Docker container.
This script communicates with the inference server running on localhost:8010
"""

import requests
import json
import os
import sys
from pathlib import Path

def run_inference_on_region(region_id, model_id=1, output_dir="output"):
    """Run inference on a specific region from the 20cities dataset"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # The data is mounted at /data in the container
    # For this demo, we'll use a sample from the dataset
    # In a real scenario, you'd need to extract satellite imagery
    
    print(f"Running inference on region {region_id} with model {model_id}")
    print("Note: This is a demo script. To run actual inference, you would need:")
    print("1. Satellite imagery for the region")
    print("2. Use the infer_custom_input.py script from docker/scripts/")
    print("3. Or use infer_mapbox_input.py with lat/lon coordinates")
    
    # Example using the API directly (requires actual image data)
    # url = "http://localhost:8010/inference"
    # For now, just show how to check if server is running
    
    try:
        response = requests.get("http://localhost:8010/", timeout=2)
        print(f"\n✓ Inference server is running!")
        print(f"Server response status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Could not connect to inference server: {e}")
        print("Make sure the container is running: podman ps")
        return False

if __name__ == "__main__":
    # Test server connectivity
    if run_inference_on_region(0):
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("\n1. Use the scripts in docker/scripts/ for actual inference:")
        print("   cd docker/scripts")
        print("   python infer_mapbox_input.py -lat 42.3601 -lon -71.0589 -tile_size 500 -model_id 1 -output boston.json")
        print("\n2. Or run inference on custom images:")
        print("   python infer_custom_input.py -input sample.png -gsd 1.0 -model_id 1 -output result.json")
        print("\n3. Models available:")
        print("   0: Sat2Graph-V1, 80-City US, 1 meter GSD")
        print("   1: Sat2Graph-V1, 20-City US (paper model)")
        print("   2: Sat2Graph-V2, 20-City US, 50cm GSD")
        print("   3: Sat2Graph-V2, 80-City Global, 50cm GSD")
        print("\n4. Access results at: http://localhost:8010/t{task_id}/")
