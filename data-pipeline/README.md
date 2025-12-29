# PathLens Data Pipeline

OSM data collection, graph processing, and walkability scoring.

## Overview

This pipeline handles the entire data collection and processing workflow:
1. Fetch amenity data from Overpass API
2. Download street network from OpenStreetMap
3. Build spatial graph with amenities mapped to nodes
4. Compute walkability scores for all nodes
5. Generate interactive visualizations

## Structure

```
data-pipeline/
├── run.py                  # Main orchestrator
├── fetch_amenities.py      # Overpass API queries
├── collect_osm_data.py     # OSM network download
├── convert_amenities.py    # JSON to GeoJSON conversion
├── build_graph.py          # Graph construction
├── compute_scores.py       # Walkability scoring
├── visualize.py            # Map generation
└── cache_manager.py        # Cache utilities
```

## Usage

### Run full pipeline
```bash
python data-pipeline/run.py --place "Bangalore, India"
```

### Individual steps
```bash
# 1. Fetch amenities
python data-pipeline/fetch_amenities.py

# 2. Download OSM data
python data-pipeline/collect_osm_data.py --place "Bangalore, India" --out-dir ../data/raw/osm

# 3. Build graph
python data-pipeline/build_graph.py \
    --graph-path ../data/raw/osm/graph.graphml \
    --pois-path ../data/raw/osm/pois.geojson \
    --out-dir ../data/processed

# 4. Compute scores
python data-pipeline/compute_scores.py \
    --graph-path ../data/processed/graph.graphml \
    --poi-mapping ../data/processed/poi_node_mapping.parquet \
    --pois-path ../data/raw/osm/pois.geojson \
    --out-dir ../data/analysis \
    --config ../config.yaml

# 5. Visualize
python data-pipeline/visualize.py \
    --graph-path ../data/processed/graph.graphml \
    --pois-path ../data/raw/osm/pois.geojson \
    --nodes-path ../data/analysis/nodes_with_scores.parquet \
    --mapping-path ../data/processed/poi_node_mapping.parquet \
    --out ../interactive_map.html
```

## Inputs

- **Config**: `../config.yaml` - Amenity weights, equity thresholds
- **External APIs**: 
  - Overpass API (amenity data)
  - OpenStreetMap (street network via osmnx)

## Outputs

All outputs go to `../data/`:

### `data/raw/osm/`
- `graph.graphml` - Raw street network
- `pois.geojson` - Amenity locations
- `buildings.geojson` - Building footprints (optional)
- `landuse.geojson` - Land use polygons (optional)

### `data/processed/`
- `graph.graphml` - Processed graph with POI nodes
- `poi_node_mapping.parquet` - POI to node assignments

### `data/analysis/`
- `nodes_with_scores.parquet` - All nodes with walkability scores
- `nodes_with_scores.csv` - Same as CSV
- `h3_agg.parquet` - H3 hexagon aggregates
- `h3_agg.csv` - Same as CSV
- `metrics_summary.json` - Summary statistics

### `data/cache/scoring/`
- Cached scoring computations for faster re-runs

### `data/cache/nominatim/`
- Geocoding cache for area lookups

## Configuration

Edit `../config.yaml` to customize:

```yaml
amenities:
  hospital:
    weight: 5.0
    max_distance_meters: 2000
  school:
    weight: 4.0
    max_distance_meters: 1000
  # ... more amenities

equity:
  gini_weight: 0.3
  coefficient_variation_weight: 0.7

h3:
  resolution: 9
```

## Dependencies

Key libraries:
- `osmnx` - OSM data download
- `networkx` - Graph algorithms
- `geopandas` - Geospatial operations
- `h3` - Hexagonal grids
- `folium` - Interactive maps
- `pandas` - Data processing

See `../requirements.txt` for complete list.
