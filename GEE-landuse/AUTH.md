# PathLens Google Earth Engine Service Account Setup

**Purpose:** Automate GEE land-use feasibility analysis for PathLens. Replaces manual Code Editor workflow.

**🕒 Time required:** 10 minutes  
**💰 Cost:** Free (non-commercial tier)

## Prerequisites

- [ ] Google Cloud project with Earth Engine enabled
- [ ] `GEE-landscape/` directory with pipeline files
- [ ] Python 3.8+ with `earthengine-api`, `geopandas`

## 🚀 Quick Setup (Copy-Paste)

### 1. Create Service Account
```
Google Cloud Console → IAM & Admin → Service Accounts → CREATE SERVICE ACCOUNT
Name: `pathlens-gee` → CREATE AND CONTINUE → DONE
```

### 2. Download JSON Key
```
Service Accounts → pathlens-gee → KEYS → ADD KEY → JSON → CREATE
✅ `pathlens-gee@your-project.iam.gserviceaccount.com.json` downloads
```

### 3. Assign IAM Roles
```
IAM & Admin → IAM → + ADD → `pathlens-gee@your-project.iam.gserviceaccount.com`
Roles:
✅ Earth Engine Resource Viewer
✅ Service Usage Consumer
✅ Earth Engine Resource Writer
SAVE
```

### 4. Test (Copy to `gee_auth.py`)
```
import ee
from google.oauth2 import service_account
from pathlib import Path

SERVICE_ACCOUNT_JSON = Path("pathlens-service-account.json")  # Rename your JSON here
PROJECT_ID = "your-project-id"  # UPDATE

credentials = service_account.Credentials.from_service_account_file(
    str(SERVICE_ACCOUNT_JSON),
    scopes=['https://www.googleapis.com/auth/earthengine']
)
ee.Initialize(credentials=credentials, project=PROJECT_ID)
print("✅ Auth OK:", ee.Image('ESA/WorldCover/v200').getInfo()['id'])
```

```
python test_gee_auth.py
```

## 📁 Directory Structure

```
GEE-landuse/                    
├── pathlens-service-account.json  # ← ADD YOUR JSON HERE (.gitignore'd)
├── gee_auth.py              # ← Test auth
├── amenity_placement.py          # Generate node candidates
├── pathlens_pipeline.py               # ← Full automation (coming soon)
├── feasibilityfilter.py          # Post-process results
```

## 🔒 Security (.gitignore)

```
# GEE Service Account
pathlens-service-account.json
*.iam.gserviceaccount.com.json
secrets/
.env
```
## ❌ Common Errors & Fixes

| Error | ✅ Fix |
|-------|--------|
| `JSON not found` | `mv *.json pathlens-service-account.json` |
| `Earth Engine access denied` | Add "Earth Engine Resource Viewer" role |
| `serviceusage.services.use` | Add "Service Usage Consumer" role |
| `project mismatch` | Update `PROJECT_ID` |