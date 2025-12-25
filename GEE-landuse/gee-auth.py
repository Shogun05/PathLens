#!/usr/bin/env python3
"""
Earth Engine Service Account Authentication Test
Run this FIRST to verify your JSON key + permissions work.
"""

import ee
from google.oauth2 import service_account
from pathlib import Path
import json

# CONFIG - UPDATE THESE PATHS
SERVICE_ACCOUNT_JSON = Path("pathlens-service-account.json")  # Your downloaded JSON
PROJECT_ID = "tidecon-9913b"  # Your GEE project ID

def test_auth():
    print("🔑 Testing Earth Engine service account authentication...")
    
    # Check JSON file exists
    if not SERVICE_ACCOUNT_JSON.exists():
        print(f"❌ JSON file not found: {SERVICE_ACCOUNT_JSON}")
        print("Download it from IAM → Service Accounts → KEYS → ADD KEY → JSON")
        return False
    
    try:
        # Load credentials
        credentials = service_account.Credentials.from_service_account_file(
            str(SERVICE_ACCOUNT_JSON),
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        
        # Initialize
        ee.Initialize(credentials=credentials, project=PROJECT_ID)
        print("✅ Authentication successful!")
        
        # Test 1: Basic dataset access
        worldcover = ee.ImageCollection('ESA/WorldCover/v200').first()
        info = worldcover.getInfo()
        print(f"✅ WorldCover loaded: {info['id']}")
        
        # Test 2: Simple computation
        nodes = ee.FeatureCollection('projects/tidecon-9913b/assets/hospital_nodes')
        count = nodes.size().getInfo()
        print(f"✅ Asset access: {count} hospital nodes found")
        
        # Test 3: Export permission
        test_export = ee.batch.Export.table.toDrive(
            collection=nodes.limit(1),
            description='test_export',
            fileFormat='CSV'
        )
        test_export.start()
        print("✅ Export permissions OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("\n🔧 Fix steps:")
        print("1. Check PROJECT_ID matches your GEE project")
        print("2. Verify service account has these IAM roles:")
        print("   - Earth Engine Resource Viewer")
        print("   - Service Usage Consumer")
        print("   - Earth Engine Resource Writer")
        return False

if __name__ == "__main__":
    success = test_auth()
    if success:
        print("\n🎉 Service account READY for production pipeline!")
        print("Run your full gee_pipeline.py now.")
    else:
        print("\n⚠️  Fix auth issues above, then re-run.")
