# PathLens Step 1–3 Pipeline

This repository provides command-line scripts to collect OpenStreetMap data, build an annotated walking network, score walkability metrics, and generate an interactive map preview.

## Requirements

- Python 3.9 or newer
- Recommended: virtual environment (venv, Conda, etc.)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

- Edit `config.yaml` to set the target place/bbox, amenity weights, scoring weights, H3 resolution, and performance knobs (centrality sampling, distance cutoffs, etc.).
- Script flags override config values when supplied.

## Running the pipeline

### 1. Download raw data

```powershell
python scripts/data_collection.py --place "Bangalore, India" --out-dir data/raw
```

Options:

- `--bbox north south east west` for bounding box extraction
- Reads `network_type` and default place from `config.yaml` when omitted

Outputs written to `data/raw/`:

- `graph.graphml`, `nodes.parquet`, `edges.parquet`
- `pois.geojson`, and optional `buildings.geojson`, `landuse.geojson`, `transit.geojson`

### 2. Build and annotate the graph

```powershell
python scripts/graph_build.py ^
   --graph-path data/raw/graph.graphml ^
   --pois-path data/raw/pois.geojson ^
   --out-dir data/processed ^
   --buildings-path data/raw/buildings.geojson ^
   --landuse-path data/raw/landuse.geojson
```

Key results in `data/processed/`:

- Simplified graph (`graph.graphml`), enriched `nodes.parquet` / `edges.parquet`
- `poi_node_mapping.parquet` linking POIs to nearest nodes with metadata

### 3. Compute walkability scores

```powershell
python scripts/scoring.py ^
   --graph-path data/processed/graph.graphml ^
   --poi-mapping data/processed/poi_node_mapping.parquet ^
   --pois-path data/raw/pois.geojson ^
   --out-dir data/analysis
```

Important arguments:

- `--config` alternate config file
- `--h3-res` override H3 resolution

Outputs in `data/analysis/`:

- `nodes_with_scores.(parquet|csv)` with structure/accessibility/equity/travel-time metrics
- `h3_agg.(parquet|csv)` aggregated metrics per hex
- `metrics_summary.json` with headline indicators

### 4. Create the interactive map (optional)

```powershell
python scripts/visualize.py --out interactive_map.html
```

Flags:

- `--no-links` disables POI-to-node link lines
- `--graph-path`, `--pois-path`, `--nodes-path`, `--mapping-path` to customize sources

Open the generated `interactive_map.html` in a browser to review category-colored POIs, link segments, and (when available) node walkability overlays.

## Tips

- For large geographies, tune the `centrality.sample_k`, `amenity_distance_cutoff_m`, and `circuity_sample_k` in `config.yaml` to manage runtime.
- Ensure CRS consistency: data collection emits projected nodes/edges; later steps expect geometries in meters for length computations.
- Clean the `data/` folders between runs if switching study areas to avoid mixing outputs.
