# PathLens Pipeline Output Mapping

This document maps all output files created by the PathLens pipeline system, organized by pipeline stage.

---

## Pipeline Architecture

PathLens consists of 3 main pipelines orchestrated by `run_pathlens.py`:

1. **Data Pipeline** (`data-pipeline/run_pipeline.py`) - OSM data collection and baseline scoring
2. **Optimization Pipeline** (`optimization-pipeline/run_optimization.py`) - GA+MILP amenity placement optimization
3. **Landuse Pipeline** (`landuse-pipeline/run_feasibility.py`) - GEE satellite-based feasibility analysis

---

## Data Pipeline Outputs

**Script:** `data-pipeline/run_pipeline.py`

### Raw Data Collection

#### Amenities
- **File:** `data/raw/osm/amenities.geojson`
  - **Format:** GeoJSON
  - **Contains:** All amenities (shops, leisure, services) with geometry, OSM IDs, names, and attributes
  - **Purpose:** Input POI layer for graph building and scoring
  - **Created by:** `pipeline/convert_amenities_to_geojson.py`

#### Street Network
- **File:** `data/raw/osm/street_network.graphml`
  - **Format:** GraphML
  - **Contains:** Raw street network graph from OpenStreetMap with nodes and edges
  - **Purpose:** Base network for all subsequent analysis
  - **Created by:** `pipeline/data_collection.py` via OSMnx

- **File:** `data/raw/osm/nodes.parquet`
  - **Format:** Parquet
  - **Contains:** Street network nodes with coordinates and OSM attributes
  - **Purpose:** Internal reference for raw network nodes
  - **Created by:** `pipeline/data_collection.py`

- **File:** `data/raw/osm/edges.parquet`
  - **Format:** Parquet
  - **Contains:** Street network edges with lengths, geometries, and OSM attributes
  - **Purpose:** Internal reference for raw network edges
  - **Created by:** `pipeline/data_collection.py`

#### Context Layers
- **File:** `data/raw/osm/buildings.geojson`
  - **Format:** GeoJSON
  - **Contains:** Building footprints from OSM
  - **Purpose:** Node annotation with building proximity
  - **Created by:** `pipeline/data_collection.py`

- **File:** `data/raw/osm/landuse.geojson`
  - **Format:** GeoJSON
  - **Contains:** Land use polygons (residential, commercial, etc.)
  - **Purpose:** Node annotation with land use type
  - **Created by:** `pipeline/data_collection.py`

- **File:** `data/raw/osm/public_transport.geojson`
  - **Format:** GeoJSON
  - **Contains:** Public transport features from OSM
  - **Purpose:** Optional analysis layer (not currently used)
  - **Created by:** `pipeline/data_collection.py`

### Processed Network

- **File:** `data/processed/graph.graphml`
  - **Format:** GraphML
  - **Contains:** Simplified street network with building/landuse annotations
  - **Purpose:** Main graph for walkability scoring and optimization
  - **Created by:** `pipeline/graph_build.py`

- **File:** `data/processed/nodes.parquet`
  - **Format:** Parquet
  - **Contains:** Processed nodes with building/landuse proximity, POI counts per category
  - **Purpose:** Enriched node dataset for scoring
  - **Created by:** `pipeline/graph_build.py`

- **File:** `data/processed/edges.parquet`
  - **Format:** Parquet
  - **Contains:** Processed edges for simplified graph
  - **Purpose:** Network topology reference
  - **Created by:** `pipeline/graph_build.py`

- **File:** `data/processed/poi_mapping.parquet`
  - **Format:** Parquet
  - **Contains:** Mapping of each POI to nearest street node with distance, category, source
  - **Purpose:** Link POIs to network for accessibility computation
  - **Created by:** `pipeline/graph_build.py`

### Analysis Outputs (Baseline Scenario)

#### 🎯 FRONTEND CORE DATA

- **File:** `data/analysis/baseline_nodes_with_scores.parquet`
  - **Format:** Parquet
  - **Contains:** Every node with computed metrics: walkability, accessibility_score, travel_time_min, structure_score, equity_score, betweenness_centrality, distances to each amenity type, H3 index
  - **Purpose:** **Core dataset for frontend visualization and optimization**
  - **Created by:** `pipeline/scoring.py`

- **File:** `data/analysis/baseline_nodes_with_scores.csv`
  - **Format:** CSV
  - **Contains:** Same as parquet (CSV export for portability)
  - **Purpose:** **Human-readable export, backend API serving**
  - **Created by:** `pipeline/scoring.py`

- **File:** `data/analysis/baseline_h3_agg.parquet`
  - **Format:** Parquet
  - **Contains:** H3 hexagon aggregations with mean/std walkability, amenity coverage percentages, population-weighted accessibility
  - **Purpose:** **Hexagon-level visualization in frontend**
  - **Created by:** `pipeline/scoring.py`

- **File:** `data/analysis/baseline_h3_agg.csv`
  - **Format:** CSV
  - **Contains:** Same as parquet (CSV export)
  - **Purpose:** **Backend API serving, external analysis**
  - **Created by:** `pipeline/scoring.py`

- **File:** `data/analysis/baseline_metrics_summary.json`
  - **Format:** JSON
  - **Contains:** Global summary statistics - mean walkability, accessibility, circuity ratio, intersection density, travel time metrics
  - **Purpose:** **Dashboard summary cards in frontend**
  - **Created by:** `pipeline/scoring.py`

#### Visualization

- **File:** `data/analysis/map_visualization.html`
  - **Format:** HTML (Folium/Leaflet map)
  - **Contains:** Interactive map with POI layers by category, node walkability heatmap, POI-to-node links
  - **Purpose:** Standalone visualization for data exploration
  - **Created by:** `pipeline/visualize.py`

---

## Optimization Pipeline Outputs

**Script:** `optimization-pipeline/run_optimization.py`

### Candidate Analysis

- **File:** `data/optimization/runs/high_travel_time_nodes.csv`
  - **Format:** CSV
  - **Contains:** Nodes filtered by travel_time_min > threshold (default 15 min), sorted by travel time, with walkability and accessibility scores
  - **Purpose:** Candidate nodes where new amenities could have high impact
  - **Created by:** `optimization/list_optimizable_nodes.py`

### Genetic Algorithm Outputs

#### 🎯 FRONTEND OPTIMIZATION RESULTS

- **File:** `data/optimization/runs/best_candidate.json`
  - **Format:** JSON
  - **Contains:** Best amenity placement solution found by GA - includes generation number, candidate signature (amenity→node mapping), template ID, fitness metrics (best_distances, accessibility improvements)
  - **Purpose:** **Primary optimization result used by frontend and downstream pipelines**
  - **Created by:** `optimization/hybrid_ga.py`

- **File:** `data/optimization/runs/run_summary.json`
  - **Format:** JSON
  - **Contains:** Full optimization run summary - best candidate, best metrics, generation-by-generation history, operator credits
  - **Purpose:** Detailed optimization analytics and diagnostics
  - **Created by:** `optimization/hybrid_ga.py`

#### Progress Tracking

- **File:** `data/optimization/runs/generation_NNNN.json` (one per generation)
  - **Format:** JSON
  - **Contains:** Top candidates from each generation with their metrics
  - **Purpose:** Generation-by-generation tracking for debugging and analysis
  - **Created by:** `optimization/hybrid_ga.py`

- **File:** `data/optimization/runs/progress.json`
  - **Format:** JSON
  - **Contains:** Live progress indicator - current generation, best metrics, elapsed time, timestamp
  - **Purpose:** Monitor long-running optimizations
  - **Created by:** `optimization/hybrid_ga.py`

#### Runtime State

- **File:** `data/optimization/runs/checkpoint.json`
  - **Format:** JSON
  - **Contains:** Resume checkpoint with current generation, population state, history, best candidate
  - **Purpose:** Resume interrupted optimization runs
  - **Created by:** `optimization/hybrid_ga.py` (deleted after successful completion)

- **File:** `data/optimization/runs/metadata.json`
  - **Format:** JSON
  - **Contains:** Run configuration and execution metadata
  - **Purpose:** Track optimization run parameters
  - **Created by:** `optimization/hybrid_ga.py`

### Solution Visualization

#### 🎯 FRONTEND POI LAYER

- **File:** `data/optimization/runs/combined_pois.geojson`
  - **Format:** GeoJSON
  - **Contains:** **Combined POI layer** - all existing POIs + all optimized placements, each with `source` field ("existing" or "optimized")
  - **Purpose:** **Complete POI dataset for frontend map visualization**
  - **Created by:** `optimization/generate_solution_map.py`

- **File:** `data/optimization/runs/solution_map.html`
  - **Format:** HTML (Folium/Leaflet map)
  - **Contains:** Interactive map showing existing vs optimized POIs with different markers, optional path visualization
  - **Purpose:** Visual validation of optimization results
  - **Created by:** `optimization/generate_solution_map.py`

### Re-scored Network (Optimized Scenario)

The optimization pipeline runs the full scoring pipeline twice via `optimization/run_optimized_pipeline.py`:
- Once with **baseline** POIs (prefix: `baseline_*`)
- Once with **optimized** POIs (prefix: `optimized_*`)

#### Baseline Prefixed Outputs

- **File:** `data/processed/baseline_graph.graphml`
  - **Format:** GraphML
  - **Contains:** Baseline network (existing POIs only)
  - **Purpose:** Reference graph for comparison
  - **Created by:** Re-run of `pipeline/graph_build.py` with `amenities.geojson`

- **File:** `data/processed/baseline_nodes.parquet`
  - **Format:** Parquet
  - **Contains:** Baseline node data
  - **Purpose:** Reference nodes
  - **Created by:** Re-run of `pipeline/graph_build.py`

- **File:** `data/processed/baseline_edges.parquet`
  - **Format:** Parquet
  - **Contains:** Baseline edges
  - **Purpose:** Reference edges
  - **Created by:** Re-run of `pipeline/graph_build.py`

- **File:** `data/processed/baseline_poi_mapping.parquet`
  - **Format:** Parquet
  - **Contains:** Baseline POI mapping
  - **Purpose:** Reference mapping
  - **Created by:** Re-run of `pipeline/graph_build.py`

#### 🎯 FRONTEND OPTIMIZED DATA

- **File:** `data/analysis/baseline_nodes_with_scores.parquet`
  - **Format:** Parquet
  - **Contains:** Baseline walkability scores
  - **Purpose:** **Baseline metrics for frontend comparison**
  - **Created by:** Re-run of `pipeline/scoring.py` with baseline graph

- **File:** `data/analysis/baseline_nodes_with_scores.csv`
  - **Format:** CSV
  - **Contains:** Same as parquet
  - **Purpose:** **Backend API serving**
  - **Created by:** Re-run of `pipeline/scoring.py`

- **File:** `data/analysis/baseline_h3_agg.parquet`
  - **Format:** Parquet
  - **Contains:** Baseline H3 aggregations
  - **Purpose:** **Baseline hexagon metrics for frontend**
  - **Created by:** Re-run of `pipeline/scoring.py`

- **File:** `data/analysis/baseline_h3_agg.csv`
  - **Format:** CSV
  - **Contains:** Same as parquet
  - **Purpose:** **Backend API serving**
  - **Created by:** Re-run of `pipeline/scoring.py`

- **File:** `data/analysis/baseline_metrics_summary.json`
  - **Format:** JSON
  - **Contains:** Baseline global summary
  - **Purpose:** **Baseline dashboard metrics**
  - **Created by:** Re-run of `pipeline/scoring.py`

#### Optimized Prefixed Outputs

- **File:** `data/processed/optimized_graph.graphml`
  - **Format:** GraphML
  - **Contains:** Network with optimized POI placements merged
  - **Purpose:** Optimized scenario graph
  - **Created by:** Re-run of `pipeline/graph_build.py` with `combined_pois.geojson`

- **File:** `data/processed/optimized_nodes.parquet`
  - **Format:** Parquet
  - **Contains:** Optimized node data
  - **Purpose:** Optimized nodes
  - **Created by:** Re-run of `pipeline/graph_build.py`

- **File:** `data/processed/optimized_edges.parquet`
  - **Format:** Parquet
  - **Contains:** Optimized edges
  - **Purpose:** Optimized edges
  - **Created by:** Re-run of `pipeline/graph_build.py`

- **File:** `data/optimization/runs/optimized_pois_only.geojson`
  - **Format:** GeoJSON
  - **Contains:** **Optimized POIs only** (extracted from combined layer with source="optimized")
  - **Purpose:** Input layer for optimized graph building
  - **Created by:** `optimization/run_optimized_pipeline.py`

- **File:** `data/processed/optimized_poi_mapping.parquet`
  - **Format:** Parquet
  - **Contains:** Optimized POI mapping (includes new amenities)
  - **Purpose:** Reference mapping for optimized scenario
  - **Created by:** Re-run of `pipeline/graph_build.py`

#### 🎯 FRONTEND OPTIMIZED METRICS

- **File:** `data/analysis/optimized_nodes_with_scores.parquet`
  - **Format:** Parquet
  - **Contains:** Optimized walkability scores
  - **Purpose:** **Optimized metrics for frontend comparison**
  - **Created by:** Re-run of `pipeline/scoring.py` with optimized graph

- **File:** `data/analysis/optimized_nodes_with_scores.csv`
  - **Format:** CSV
  - **Contains:** Same as parquet
  - **Purpose:** **Backend API serving**
  - **Created by:** Re-run of `pipeline/scoring.py`

- **File:** `data/analysis/optimized_h3_agg.parquet`
  - **Format:** Parquet
  - **Contains:** Optimized H3 aggregations
  - **Purpose:** **Optimized hexagon metrics for frontend**
  - **Created by:** Re-run of `pipeline/scoring.py`

- **File:** `data/analysis/optimized_h3_agg.csv`
  - **Format:** CSV
  - **Contains:** Same as parquet
  - **Purpose:** **Backend API serving**
  - **Created by:** Re-run of `pipeline/scoring.py`

- **File:** `data/analysis/optimized_metrics_summary.json`
  - **Format:** JSON
  - **Contains:** Optimized global summary
  - **Purpose:** **Optimized dashboard metrics**
  - **Created by:** Re-run of `pipeline/scoring.py`

---

## Landuse Pipeline Outputs

**Script:** `landuse-pipeline/run_feasibility.py`

**Note:** This pipeline performs Google Earth Engine-based feasibility analysis. Currently no output files exist in the workspace, but the pipeline would create the following structure:

### Per-Amenity Outputs

For each amenity type (hospital, school, park, pharmacy, supermarket, bus_station):

#### Input Data
- **File:** `data/landuse/candidates_{amenity}.geojson`
  - **Format:** GeoJSON
  - **Contains:** GA-optimized candidate node locations for specific amenity type
  - **Purpose:** Input for GEE upload
  - **Created by:** `landuse-pipeline/extract_candidates.py`

#### Feasibility Analysis

- **File:** `data/landuse/feasibility_{amenity}.csv`
  - **Format:** CSV
  - **Contains:** Feasibility analysis results - node_id, amenity, free_area_m2, min_area_req, feasible (boolean), has_patch, patch_count
  - **Purpose:** Identify which candidate nodes have sufficient land availability
  - **Created by:** `landuse-pipeline/gee_analysis.py`

#### 🎯 FRONTEND PLACEMENT GEOMETRIES

- **File:** `data/landuse/gee_placements_{amenity}.geojson`
  - **Format:** GeoJSON
  - **Contains:** Placement polygons showing best available land patches for each feasible node
  - **Purpose:** **Visualize actual buildable areas for frontend**
  - **Created by:** `landuse-pipeline/gee_analysis.py`

#### Filtered Results

- **File:** `data/landuse/feasible_nodes_{amenity}.csv`
  - **Format:** CSV
  - **Contains:** Filtered list of feasible nodes only
  - **Purpose:** Quick reference for feasible placements
  - **Created by:** `landuse-pipeline/filter_feasible.py`

- **File:** `data/landuse/feasible_nodes_full_{amenity}.csv`
  - **Format:** CSV
  - **Contains:** Feasible nodes merged with full node attributes (walkability, accessibility, etc.)
  - **Purpose:** **Complete feasibility dataset with walkability context**
  - **Created by:** `landuse-pipeline/merge_attributes.py`

- **File:** `data/landuse/cleaned_placements_{amenity}.geojson`
  - **Format:** GeoJSON
  - **Contains:** Cleaned placement polygons filtered by amenity type
  - **Purpose:** Final placement geometries for frontend
  - **Created by:** `landuse-pipeline/clean_geometries.py`

---

## Logging & Metadata Outputs

### Master Orchestrator Logs

- **File:** `data/logs/pathlens_master_YYYYMMDD_HHMMSS.log`
  - **Format:** Text log
  - **Contains:** Master orchestrator execution logs
  - **Purpose:** Debugging and audit trail
  - **Created by:** `run_pathlens.py`

- **File:** `data/logs/run_summary_YYYYMMDD_HHMMSS.json`
  - **Format:** JSON
  - **Contains:** Summary of pipeline execution results - pipelines run, success/failure status, arguments
  - **Purpose:** Run tracking and status
  - **Created by:** `run_pathlens.py`

### Pipeline-Specific Logs

- **Directory:** `data/optimization/runs/logs/`
  - **Format:** Directory with optimization run logs
  - **Contains:** Detailed optimization execution logs
  - **Purpose:** Debugging optimization runs
  - **Created by:** `optimization/hybrid_ga.py`

### Cache Files

- **Directory:** `data/analysis/.cache/`
  - **Format:** Pickle files (.pkl)
  - **Contains:** Cached structure metrics and distance computations
  - **Purpose:** Speed up repeated scoring runs
  - **Created by:** `pipeline/scoring.py`

- **Directories:** `cache/`, `optimization/cache/`, `landuse/cache/`
  - **Format:** Various cache formats (JSON, pickle)
  - **Contains:** Cached MILP solutions, geocoding results, Overpass API responses
  - **Purpose:** Avoid repeated expensive operations
  - **Created by:** Various pipeline components

---

## Summary Tables

### Files Required by Frontend

| File | API Endpoint | Purpose |
|------|-------------|---------|
| `data/analysis/baseline_nodes_with_scores.csv` | `GET /api/nodes?type=baseline` | Baseline node metrics for map |
| `data/analysis/optimized_nodes_with_scores.csv` | `GET /api/nodes?type=optimized` | Optimized node metrics for map |
| `data/analysis/baseline_h3_agg.csv` | Not yet implemented | Baseline hexagon heatmap |
| `data/analysis/optimized_h3_agg.csv` | Not yet implemented | Optimized hexagon heatmap |
| `data/analysis/baseline_metrics_summary.json` | Not yet implemented | Baseline dashboard cards |
| `data/analysis/optimized_metrics_summary.json` | Not yet implemented | Optimized dashboard cards |
| `data/optimization/runs/combined_pois.geojson` | `GET /api/suggestions` (partial) | POI markers on map |
| `data/optimization/runs/best_candidate.json` | `GET /api/suggestions` (partial) | Optimization solution details |
| `data/landuse/gee_placements_{amenity}.geojson` | Not yet implemented | Feasible buildable areas |

### File Size Categories

**Large Files (>50MB, not suitable for direct frontend loading):**
- `*.graphml` - Graph structure files
- `*_nodes_with_scores.csv` - Can be 50-200MB+ for large cities
- `*_nodes_with_scores.parquet` - Binary format, smaller but still large

**Medium Files (1-50MB):**
- `combined_pois.geojson` - All POIs
- `amenities.geojson` - All existing amenities
- `h3_agg.csv` - Hexagon aggregations

**Small Files (<1MB):**
- `*_metrics_summary.json` - Summary statistics
- `best_candidate.json` - Optimization solution
- `run_summary.json` - Run metadata
- `feasibility_*.csv` - Per-amenity feasibility results

### Processing Flow

```
run_pathlens.py
│
├─── Data Pipeline
│    ├─ Collect OSM data → data/raw/osm/
│    ├─ Build graph → data/processed/graph.graphml
│    └─ Compute scores → data/analysis/baseline_*
│
├─── Optimization Pipeline
│    ├─ Find candidates → high_travel_time_nodes.csv
│    ├─ Run GA → best_candidate.json
│    ├─ Generate solution → combined_pois.geojson
│    └─ Re-score both scenarios → data/analysis/{baseline,optimized}_*
│
└─── Landuse Pipeline (Optional)
     ├─ Extract candidates → candidates_{amenity}.geojson
     ├─ GEE analysis → feasibility_{amenity}.csv
     └─ Generate placements → gee_placements_{amenity}.geojson
```

---

## Notes

1. **Parquet vs CSV:** Most analysis outputs are saved in both formats. Parquet is faster and smaller; CSV is for portability and backend serving.

2. **Baseline vs Optimized:** The optimization pipeline creates **prefixed versions** of all baseline outputs to enable side-by-side comparison.

3. **Cache Strategy:** All pipelines use aggressive caching to avoid re-downloading OSM data, re-computing expensive metrics, or re-running MILP solvers.

4. **GEE Dependency:** The landuse pipeline requires Google Earth Engine credentials and may take hours to process satellite imagery.

5. **Frontend Data Loading:** Large CSV files should be served via backend API with optional filtering/pagination, not loaded directly in browser.
