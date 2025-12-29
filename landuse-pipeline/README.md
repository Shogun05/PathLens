# PathLens Landuse Pipeline

Google Earth Engine land-use feasibility analysis for amenity placements.

## Overview

This pipeline validates proposed amenity placements using satellite imagery and land-use classification:
1. Load optimized amenity placements from GA/MILP
2. Upload candidate locations to Google Earth Engine
3. Analyze land-use availability around each location
4. Filter placements based on free land requirements
5. Download feasible placements locally

## Structure

```
landuse-pipeline/
├── run_feasibility.py          # Main GEE pipeline
├── amenity_placement.py        # Candidate builder
├── feasibility_filter.py       # Results filtering
└── utils.py                    # Helper functions
```

## Usage

### Analyze single amenity
```bash
python landuse-pipeline/run_feasibility.py hospital
```

### Analyze all amenities
```bash
python landuse-pipeline/run_feasibility.py --all
```

### From master orchestrator
```bash
python run_pathlens.py --pipeline landuse hospital
```

## Prerequisites

### Google Earth Engine Setup

1. **Create GEE account**: https://earthengine.google.com/
2. **Create Google Cloud project**: https://console.cloud.google.com/
3. **Enable Earth Engine API**
4. **Create service account** and download credentials JSON
5. **Place credentials**: `landuse-pipeline/service-account-key.json`

### First-time authentication
```python
import ee
ee.Authenticate()
ee.Initialize()
```

## Inputs

### From other pipelines
- `../data/optimization/runs/best_candidate.json` - GA/MILP results
- `../data/analysis/optimized_nodes_with_scores.csv` - Node attributes

### Configuration
Minimum free land area per amenity (m²):
```python
MIN_AREA = {
    'hospital': 1000,
    'school': 800,
    'park': 500,
    'pharmacy': 100,
    'supermarket': 600,
    'bus_station': 400
}
```

Buffer radius: 200 meters (configurable)

## Outputs

All outputs go to `../data/landuse/`:

### Feasibility Reports
- `pathlens_feasibility_{amenity}.csv` - Raw GEE analysis results
  - Columns: node_id, amenity, free_area_m2, min_area_req, feasible, has_patch, patch_count

### Filtered Results
- `gee_feasible_nodes_{amenity}.csv` - Nodes meeting land requirements
- `gee_feasible_nodes_{amenity}_merged.csv` - Merged with node attributes

### Placement Polygons
- `pathlens_placements_{amenity}.geojson` - Best placement polygons per node
  - Properties: node_id, amenity, distance_m, patch_area_m2

## How It Works

### Land-Use Analysis Workflow

For each candidate node:

1. **Buffer creation**: 200m radius around node
2. **Land-use classification**: ESA WorldCover 10m resolution
   - Class 30: Herbaceous vegetation
   - Class 40: Cropland
   - Class 60: Bare/sparse vegetation
   
3. **Free land detection**: Pixels in allowed classes
4. **Area calculation**: Sum of free pixel areas
5. **Feasibility check**: Compare to minimum requirement
6. **Patch extraction**: Find contiguous free land regions
7. **Best patch selection**: Closest to node with sufficient area

### ESA WorldCover Classes

```
10: Tree cover
20: Shrubland
30: Grassland          ✓ Usable
40: Cropland           ✓ Usable
50: Built-up           ✗ Not usable
60: Bare/sparse veg    ✓ Usable
70: Snow and ice
80: Permanent water
90: Herbaceous wetland
95: Mangroves
100: Moss and lichen
```

## Example Output

```csv
node_id,amenity,free_area_m2,min_area_req,feasible,has_patch,patch_count
10020025539,hospital,1250.5,1000,True,True,3
10991934818,hospital,450.2,1000,False,False,0
10035832527,hospital,2150.8,1000,True,True,5
```

## Configuration

### Adjust buffer radius
```python
# In run_feasibility.py
BUFFER_METERS = 200  # Change to desired radius
```

### Adjust minimum areas
```python
MIN_AREA = {
    'hospital': 1000,   # Increase/decrease as needed
    'school': 800,
    # ...
}
```

### Change land-use classes
```python
# In run_feasibility.py, modify freeMask
freeMask = (worldCover.eq(30)   # Grassland
           .Or(worldCover.eq(40))  # Cropland
           .Or(worldCover.eq(60))  # Bare vegetation
           .Or(worldCover.eq(20))) # Add shrubland if desired
```

## Google Earth Engine Details

### Asset Management
- Temporary assets created: `projects/{PROJECT_ID}/assets/{amenity}_nodes`
- Automatically cleaned up after analysis
- Uses batch export tasks

### Computational Limits
- Scale: 10m resolution (ESA WorldCover)
- Max pixels per reduction: 1e8
- Processing time: ~30 seconds per amenity

### Data Sources
- **ESA WorldCover v200**: Global land cover at 10m
  - Dataset: `ESA/WorldCover/v200`
  - Year: 2021
  - Coverage: Global

## Troubleshooting

### "User memory limit exceeded"
Reduce buffer radius or process fewer candidates at once.

### "Computation timeout"
Increase GEE computation timeout or process in smaller batches.

### "No feasible placements found"
- Check if minimum area requirements are too strict
- Verify land-use classes include appropriate categories for your region
- Increase buffer radius to search larger area

## Dependencies

Key libraries:
- `ee` (earthengine-api) - Google Earth Engine Python API
- `geemap` - GEE visualization and export utilities
- `geopandas` - Geospatial operations
- `pandas` - Data processing

See `../requirements.txt` for complete list.

## Performance

Typical processing times:
- **Upload to GEE**: 30-60 seconds per amenity
- **GEE Analysis**: 20-40 seconds per amenity
- **Download results**: 10-20 seconds per amenity
- **Total per amenity**: ~2-3 minutes

Batch processing all amenities: 15-20 minutes
