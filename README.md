# PathLens: Urban Walkability Analysis & Optimization Framework

PathLens is a comprehensive geospatial analysis framework that evaluates and optimizes urban walkability through network analysis, amenity accessibility scoring, and genetic algorithm-based optimization. The system identifies optimal locations for new amenities to maximize citywide walkability and accessibility.

## Overview

PathLens combines graph theory, spatial analysis, and evolutionary optimization to:
1. **Analyze** pedestrian network structure and amenity accessibility
2. **Score** walkability using multi-factor composite metrics
3. **Optimize** amenity placement to improve accessibility for underserved areas
4. **Visualize** results through interactive maps and spatial aggregations

## Methodology & Techniques

## Methodology & Techniques

### 1. Network Analysis & Structure Metrics

**Graph-Based Pedestrian Network Modeling**
- Extracts walking networks from OpenStreetMap using OSMnx
- Models streets as weighted directed graphs where edge weights represent walking distances
- Computes topological importance using network centrality measures:
  - **Betweenness Centrality**: Identifies nodes that serve as critical connection points in the pedestrian network
  - **Closeness Centrality**: Measures how accessible a node is from all other nodes (optional, computationally expensive)
  - Uses k-sampling for betweenness (default k=150) to balance accuracy and performance for large networks

**Structural Connectivity**
- Degree centrality quantifies local connectivity (number of intersections)
- Edge length distribution analysis identifies walkable vs. auto-oriented street patterns
- Circuity index measures network directness (network distance / euclidean distance)

### 2. Amenity Accessibility Scoring

**Multi-Source Shortest Path Analysis**
- For each amenity type (schools, hospitals, parks, supermarkets, etc.), computes shortest network path from every node to the nearest amenity
- Uses Dijkstra's multi-source algorithm for efficient distance computation across 180k+ nodes
- Supports distance cutoffs to limit search radius and improve performance

**Inverse Distance Weighting**
- Accessibility score formula: `Σ(weight_i / (distance_i + 1))` for each amenity type
- Configurable weights per amenity type reflect relative importance (e.g., hospitals=2.0, parks=0.8)
- Closer amenities contribute more to accessibility score (inverse relationship)

**Coverage Analysis**
- Binary coverage metrics based on distance thresholds (e.g., school within 800m)
- Aggregated to H3 hexagonal grids for equity analysis

### 3. Composite Walkability Scoring

**Four-Component Weighted Model**
```
Walkability = α·Structure + β·Accessibility + γ·Equity + δ·Travel-Time
```

Where:
- **α (Structure, 0.25)**: Network topology metrics (centrality, connectivity)
- **β (Accessibility, 0.40)**: Inverse-distance weighted amenity access (highest weight)
- **γ (Equity, 0.20)**: Spatial variance in accessibility across H3 hexagons
- **δ (Travel-Time, 0.15)**: Reciprocal of minimum walking time to any amenity

This composite approach balances infrastructure quality, service access, spatial fairness, and time efficiency.

### 4. Equity & Spatial Aggregation

**H3 Hexagonal Binning**
- Aggregates node-level metrics to H3 hexagons (default resolution 8 ≈ 0.46 km² area)
- Computes hexagon-level statistics: mean, variance, coverage percentages
- Enables neighborhood-scale equity analysis

**Population-Weighted Accessibility**
- Residential land-use areas receive 1.5x weighting in accessibility calculations
- Identifies disparities between residential and non-residential areas
- Highlights underserved neighborhoods for targeted intervention

### 5. Optimization via Hybrid Genetic Algorithm

**Problem Formulation**
- **Objective**: Maximize average accessibility reduction across nodes with longest travel times
- **Decision Variables**: Locations (network nodes) for placing new amenities of each type
- **Constraints**: Limited candidate pool (nodes with travel time > threshold, default 15 min)

**Hybrid GA Architecture**

*Initialization*
- Greedy template generation: Creates initial solutions by placing amenities at nodes with worst accessibility for each type
- Random seeding: Adds diversity with random placements from candidate pool
- Population size: 80 candidates (configurable)

*Genetic Operators*
- **Crossover (75% rate)**: Uniform crossover combines amenity placements from two parents, split at random pivot points
- **Mutation (20% rate)**: Randomly adds or removes single amenity placements
- **Elitism**: Top 4 candidates preserved unchanged each generation

*Memetic Local Search*
- Budget: 20 iterations per selected candidate
- Greedy hill-climbing: Tests swapping placements to nearby high-travel-time nodes
- Applied to top-k (default 5) candidates per generation
- Operator credit tracking to analyze which operators drive improvement

*Fitness Evaluation*
- For each candidate placement configuration:
  1. Recompute accessibility distances using multi-source Dijkstra
  2. Calculate mean distance reduction for each amenity type
  3. Aggregate fitness as weighted average of distance improvements
- Parallel evaluation with configurable worker threads

*Adaptive Features*
- Checkpointing: Saves population, history, and random state each generation
- Heartbeat monitoring: Writes progress JSON for external monitoring
- Convergence detection: Tracks best fitness and diversity metrics

**Output Analysis**
- Best candidate exported with detailed metrics (mean improvements per amenity)
- Generation history tracking shows optimization trajectory
- Post-optimization pipeline recomputes full walkability scores with new amenities

### 6. MILP-Based Optimization (Alternative/Complement)

**Mixed Integer Linear Programming Approach**

PathLens includes a MILP-based optimizer that provides exact (or near-exact) solutions to the amenity placement problem using facility location formulation.

*Mathematical Model*
- **Decision Variables**:
  - `x[i,a]`: Binary variable indicating amenity type `a` placed at candidate node `i`
  - `y[n,a]`: Binary variable indicating demand node `n` covered by amenity `a`
- **Objective**: Maximize population-weighted coverage

# Full run with hybrid MILP
python run_optimization.py --enable-hybrid-milp

# Resume after GA crash
python run_optimization.py --skip-candidates --skip-ga

# Custom MILP strategy
python run_optimization.py --milp-strategy least_contributing

### 7. Comparative Scenario Analysis

**Baseline vs. Optimized Pipeline**
- Builds separate graph artifacts with distinct prefixes (`baseline_`, `optimized_`)
- Merges optimized amenity placements with existing POIs for fair comparison
- Recomputes all four walkability components under both scenarios
- Generates comparative metrics summaries showing improvements

**Key Metrics Tracked**
- Mean accessibility score change
- Travel time reduction (minutes)
- Equity variance reduction across hexagons
- Coverage percentage improvements for each amenity type

### 8. Interactive Visualization & Map Generation

**Multiple Interactive Maps Generated**

PathLens produces several interactive HTML maps throughout the workflow:

*1. Basic Network Visualization* (Optional, Step 4 of basic pipeline)
```bash
python pipeline/visualize.py --out interactive_map.html
```
- Shows pedestrian network with existing POIs
- POI-to-node connection lines
- Node walkability scores (if available)
- Colored by amenity category

*2. GA Optimization Solution Map* (Step 3 of optimization workflow)
```bash
python optimization/generate_solution_map.py
```
**Outputs:**
- `optimization/runs/poi_mapping.geojson`: Combined existing + new amenities
- `optimization/runs/optimized_map.html`: Interactive map highlighting:
  - Existing POIs (original amenities)
  - GA-optimized new placements (color-coded)
  - High-travel-time nodes that triggered optimization
  - Coverage circles showing amenity service areas

*3. MILP Solution Map* (Standalone MILP mode)
```bash
python optimization/milp_placement.py
```
**Outputs:**
- `optimization/milp_results/milp_placements.geojson`
- Visualization showing MILP-selected optimal locations

*4. Comparative Scenario Maps* (Step 4 of optimization workflow)
```bash
python optimization/run_optimized_pipeline.py
```
**Outputs:**
- `data/analysis/baseline_map.html`: Before optimization
- `data/analysis/optimized_map.html`: After optimization
- Side-by-side comparison of walkability scores
- Hexagon-level accessibility improvements
- Travel time reduction heatmaps

**Map Features**
- **Interactive layers**: Toggle POI categories, hexagon overlays, new placements
- **Pop-up information**: Click nodes/POIs for detailed metrics
- **Color coding**: 
  - POIs by amenity type
  - Nodes by walkability score (gradient)
  - New placements highlighted in distinct color
- **Zoom/pan**: Full Leaflet.js map controls
- **Export ready**: Self-contained HTML files, shareable

**Final Visualization Outputs**

After running the full optimization workflow:
```bash
python run_optimization.py
```

You'll find interactive maps at:
- `optimization/runs/optimized_map.html` ← **Main GA solution map**
- `data/analysis/baseline_map.html` ← Baseline scenario
- `data/analysis/optimized_map.html` ← Optimized scenario with improvements
- `optimization/milp_results/milp_placements.geojson` (if MILP used separately)

**Opening the Maps**
```bash
# Linux/Mac
xdg-open optimization/runs/optimized_map.html

# Or just open in browser
firefox optimization/runs/optimized_map.html
google-chrome optimization/runs/optimized_map.html
```

The maps automatically open in your default browser and work offline (no API keys needed).

## Requirements

PathLens requires Python 3.8+ and the following packages:

- `numpy`
- `scipy`
- `pandas`
- `geopandas`
- `shapely`
- `networkx`
- `matplotlib`
- `osmox`
- `h3-py`
- `flask`
- `gunicorn`
- `pytest`

















































**Final Deliverables:**- Outputs prefixed files: `baseline_*`, `optimized_*`  - Equity variance reduction  - Coverage increase per amenity type  - Travel time reduction  - Mean accessibility gain- Computes improvement metrics:  - `data/analysis/optimized_map.html`: After optimization with improvements shown  - `data/analysis/baseline_map.html`: Before optimization- Generates **two comparison maps**:- Rebuilds graphs and scores for baseline and optimized scenarios**Step 4: Comparative Analysis with Maps**- Outputs: `optimization/runs/optimized_map.html` ← **Main result visualization**- **Opens automatically** in your default browser  - **Interactive controls** (layer toggles, zoom, click for details)  - **Candidate nodes** (high travel time nodes considered)  - **New placements** (GA/MILP-optimized locations, highlighted)  - **Existing amenities** (current POIs from OSM)- Builds `optimized_map.html` with:- Creates `poi_mapping.geojson` merging optimized + existing POIs**Step 3: Generate Interactive Solution Map** ← **Key Visualization Step**  - `milp_refinement_stats.json`: MILP statistics (if hybrid mode)  - `summary.json`: Optimization statistics  - `generation_NNNN.json`: Population snapshots  - `best_candidate.json`: Best solution found- Outputs to `optimization/runs/`:- Default: 80 population, 50 generations, 4 workers- Optional: MILP refinement for elite candidates (if `hybrid_milp.enabled: true`)- Searches for optimal amenity placements using GA + local search**Step 2: Run Hybrid Genetic Algorithm**- Outputs: `optimization/high_travel_time_nodes.csv`- Scans `nodes_with_scores.parquet` for nodes with high travel times (>15 min default)**Step 1: Identify Optimization Candidates**This orchestrates the complete optimization workflow:```python run_optimization.py```bash#### Full Optimization Pipeline### Advanced: Optimization WorkflowInstall all dependencies using `pip`:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Run the full analysis and optimization pipeline** (including data download):
```bash
python run_optimization.py
```
2. **View the main GA solution map**:
```bash
xdg-open optimization/runs/optimized_map.html
```
3. **Explore other outputs** in the `data/analysis` and `optimization/runs` folders.

### Configuration

Edit `config.yaml` to customize parameters:
- Data sources and download options
- Network analysis settings (e.g., distance thresholds)
- Amenity scoring weights and cutoffs
- GA optimization parameters (population size, mutation rate)
- MILP solver options (if used)

### Advanced Usage

**Run specific pipeline stages** with options:
```bash
# Only download and preprocess data
python run_optimization.py --step 1

# Run network analysis only
python run_optimization.py --step 2

# Skip to optimization
python run_optimization.py --skip-analysis

# Enable hybrid GA + MILP optimization
python run_optimization.py --enable-hybrid-milp
```

**Test with smaller data/sample**
```bash
# Use built-in small test network
python run_optimization.py --test-mode

# Limit to 10 iterations for quick GA testing
python run_optimization.py --ga-iterations 10
```

## Troubleshooting

Common issues and solutions:

- **MemoryError**: Reduce data size or number of nodes (use `--max-nodes` option)
- **Timeouts**: Increase timeout settings in `config.yaml`
- **Installation issues**: Ensure all system dependencies are met (e.g., GEOS, H3)

See `docs/troubleshooting.md` for more details.

## Limitations & Future Work

Current limitations:
- OSM data quality and completeness
- Computational intensity for large networks
- GA optimization may converge to local minima

Planned improvements:
- Support for additional amenity types and custom datasets
- Enhanced optimization algorithms (e.g., simulated annealing)
- Integration of real-time data (e.g., foot traffic, events)

## License

PathLens is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

PathLens was developed by [Your Name/Organization] as part of the Urban Analytics initiative.

Special thanks to the contributors of the OpenStreetMap project and the developers of the used Python libraries.

## Contact

For questions, suggestions, or contributions, please contact:

- **Email**: [your.email@example.com](mailto:your.email@example.com)
- **GitHub**: [YourGitHubProfile](https://github.com/YourGitHubProfile)
