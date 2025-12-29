# PathLens Optimization Pipeline

Genetic algorithm + MILP optimization for strategic amenity placement.

## Overview

This pipeline optimizes amenity placement to minimize average travel time:
1. Extract candidate nodes with high travel times
2. Run hybrid GA to search for better placements
3. (Optional) Refine with MILP for guaranteed optimality
4. Generate comparison visualizations
5. Rebuild graph/scores with optimized placements

## Structure

```
optimization-pipeline/
├── run.py                          # Main orchestrator
├── list_optimizable_nodes.py       # Extract high travel time candidates
├── hybrid_ga.py                    # Genetic algorithm with optional MILP
├── hybrid_milp_refiner.py          # MILP refinement module
├── generate_solution_map.py        # Visualization generator
├── run_optimized_pipeline.py       # Comparative analysis
├── visualize_amenity_paths.py      # Path visualization
└── milp_placement.py               # Pure MILP solver
```

## Usage

### Run full optimization
```bash
python optimization-pipeline/run.py
```

### With hybrid MILP enabled
First, edit `../config.yaml`:
```yaml
hybrid_milp:
  enabled: true
  trigger_condition: "stagnation"  # or "periodic" or "final"
  stagnation_generations: 10
  max_refinements: 3
```

Then run:
```bash
python optimization-pipeline/run.py --ga-generations 100
```

### Individual steps
```bash
# 1. Extract candidates
python optimization-pipeline/list_optimizable_nodes.py \
    --input ../data/analysis/nodes_with_scores.parquet \
    --output ../data/optimization/high_travel_time_nodes.csv \
    --threshold 15.0

# 2. Run GA
python optimization-pipeline/hybrid_ga.py \
    --nodes-scores ../data/analysis/nodes_with_scores.parquet \
    --high-travel ../data/optimization/high_travel_time_nodes.csv \
    --analysis-dir ../data/optimization/runs \
    --population 50 \
    --generations 100

# 3. Generate map
python optimization-pipeline/generate_solution_map.py

# 4. Comparative analysis
python optimization-pipeline/run_optimized_pipeline.py \
    --baseline-prefix baseline \
    --optimized-prefix optimized
```

## Inputs

### Required
- `../data/analysis/nodes_with_scores.parquet` - Node scores from data pipeline
- `../config.yaml` - Configuration including MILP settings

### Generated
- `../data/optimization/high_travel_time_nodes.csv` - Candidate nodes for optimization

## Outputs

All outputs go to `../data/optimization/`:

### `data/optimization/runs/`
- `best_candidate.json` - Best solution found by GA
- `generation_NNNN.json` - Generation-by-generation history
- `run_metadata.json` - Run parameters and statistics
- `milp_refinement_stats.json` - MILP refinement details (if enabled)

### `data/optimization/outputs/`
- `optimized_pois.geojson` - Optimized amenity locations
- `solution_map.html` - Interactive comparison map

### `data/analysis/` (prefixed outputs)
- `baseline_graph.graphml` - Original graph
- `baseline_nodes_with_scores.csv` - Original scores
- `optimized_graph.graphml` - Optimized graph
- `optimized_nodes_with_scores.csv` - Optimized scores

### `data/cache/milp/`
- Cached MILP solutions for faster re-runs

## Configuration

### Genetic Algorithm
```yaml
# In config.yaml
ga:
  population_size: 50
  generations: 100
  mutation_rate: 0.1
  crossover_rate: 0.8
```

### Hybrid MILP
```yaml
hybrid_milp:
  enabled: true
  trigger_condition: "stagnation"  # When to invoke MILP
  stagnation_generations: 10       # Generations without improvement
  max_refinements: 3               # Max MILP calls per run
  time_limit_seconds: 300          # MILP solver timeout
```

## How It Works

### Hybrid GA-MILP Strategy

1. **GA Phase**: Explores solution space broadly
   - Generates diverse placement combinations
   - Evaluates fitness (average travel time reduction)
   - Evolves population over generations

2. **MILP Refinement** (optional, triggered by):
   - **Stagnation**: No improvement for N generations
   - **Periodic**: Every N generations
   - **Final**: At the end of GA run
   
   When triggered:
   - Fixes some amenity placements from GA's best solution
   - Optimizes remaining placements using MILP
   - Guarantees local optimality for refined region

3. **Result**: Best of both worlds
   - GA's global exploration
   - MILP's local optimality

### Evaluation Metrics

Solutions are compared on:
- **Average travel time** (minutes)
- **Median travel time** (minutes)
- **90th percentile travel time** (minutes)
- **Equity metrics** (Gini coefficient, CV)

## Dependencies

Key libraries:
- `pulp` - MILP solver interface
- `networkx` - Graph algorithms
- `numpy` - Numerical operations
- `pandas` - Data processing

See `../requirements.txt` for complete list.

## Performance

Typical runtimes on modern hardware:
- **GA only** (50 pop, 100 gen): 5-15 minutes
- **With MILP** (3 refinements): 15-30 minutes
- **Pure MILP**: 10-60 minutes (problem-dependent)

Multi-threading automatically uses 75% of available CPU cores.
