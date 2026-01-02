# PathLens Data Flow Documentation

## Overview

This document maps all data flowing from backend files to frontend components.

---

## 🎯 **OVERALL ACCESSIBILITY SCORE - Complete Data Flow**

### Where It's Displayed

**Frontend:** `frontend/app/(analysis)/baseline/page.tsx` - Line 149

```tsx
<span className="text-4xl font-bold text-white">{baselineScore.toFixed(1)}</span>
<p className="text-sm text-gray-400">Overall Accessibility Score</p>
```

### Complete Data Journey

#### 1. **Calculation in Python** (`data-pipeline/compute_scores.py`)

**Step 1: Individual Node Accessibility (Line 546)**

```python
nodes["accessibility_score"] = compute_accessibility_score(distances, amenity_weights)
```

**Formula (Lines 298-308):**

```python
def compute_accessibility_score(distances_df: pd.DataFrame, amenity_weights: Dict[str, float]):
    acc = pd.Series(0.0, index=distances_df.index)
    for amenity, weight in amenity_weights.items():
        col = f"dist_to_{amenity}"
        if col not in distances_df.columns:
            continue
        # Accessibility formula: sum(weight / (distance + 1))
        acc += weight / (distances_df[col].fillna(np.inf) + 1.0)
    return acc
```

**Amenity Weights (from config.yaml):**

- school: 2.0
- hospital: 1.5
- pharmacy: 1.0
- supermarket: 1.0
- bus_station: 0.8
- park: 1.2

**Example:** For a node with:

- dist_to_school = 500m
- dist_to_hospital = 800m
- dist_to_pharmacy = 1200m
- dist_to_park = 300m

Raw accessibility = (2.0/501) + (1.5/801) + (1.0/1201) + (1.2/301)
= 0.00399 + 0.00187 + 0.00083 + 0.00399
= **0.01068** (raw score)

**Step 2: Normalization to 1-100 Scale (Line 590)**

```python
nodes["accessibility_score"] = normalize_series(nodes["accessibility_score"]) * 99 + 1
```

**Normalization Logic (Lines 395-403):**

```python
def normalize_series(series: pd.Series) -> pd.Series:
    """Simple min-max normalization to 0-1 range."""
    min_val = valid.min()
    max_val = valid.max()
    return (series - min_val) / (max_val - min_val)
```

Then: `normalized_value * 99 + 1` → **1 to 100 scale**

Example: If raw score 0.01068 is normalized to 0.15, final score = 0.15 × 99 + 1 = **15.85**

#### 2. **Saved to CSV** (`data/analysis/baseline_nodes_with_scores.csv`)

Column 23: `accessibility_score`

- 182,191 rows (one per node)
- Values range: 1.0 to 100.0
- Example value: `1.4685004947463627`

#### 3. **Backend API** (`backend/main.py`)

**GET `/api/nodes?type=baseline&limit=2000`** (Lines 131-193)

```python
node = Node(
    osmid=str(row['osmid']),
    x=float(row['lon']),
    y=float(row['lat']),
    accessibility_score=row.get('accessibility_score'),  # ← From CSV column 23
    walkability_score=row.get('walkability'),
    equity_score=row.get('equity_score'),
    ...
)
```

**Returns:** JSON array of up to 2,000 nodes with accessibility_score field

#### 4. **Frontend API Call** (`frontend/lib/api.ts`)

```typescript
getNodes: async (params: NodesQueryParams): Promise<Node[]> => {
  const response = await api.get(`/api/nodes?${queryParams.toString()}`);
  return response.data;
};
```

#### 5. **Frontend Calculation** (`frontend/app/(analysis)/baseline/page.tsx`)

**Line 46-57: Load Data**

```typescript
const nodes = await pathLensAPI.getNodes({ type: "baseline", limit: 2000 });
setBaselineNodes(nodes);
```

**Line 56-57: Calculate Average**

```typescript
// Average accessibility across loaded nodes (max 2000)
const avgScore =
  nodes.reduce((sum, n) => sum + (n.accessibility_score || 0), 0) /
  nodes.length;

setBaselineScore(avgScore);
```

**Line 149-152: Display**

```tsx
<span className="text-4xl font-bold text-white">
  {baselineScore.toFixed(1)}
</span>
<p className="text-sm text-gray-400">Overall Accessibility Score</p>
```

### Important Notes

1. **Sampling Effect**: Frontend only loads 2,000 nodes (out of 182,191), so the displayed score is the average of a **sample**, not all nodes.

2. **True City-Wide Average**: Available in `data/analysis/baseline_metrics_summary.json`:

   ```json
   {
     "scores": {
       "accessibility_mean": 6.12 // Average of ALL 182,191 nodes
     }
   }
   ```

   This is the more accurate city-wide accessibility score.

3. **Scale Interpretation**:

   - **1-20**: Very poor accessibility (far from amenities)
   - **20-40**: Low accessibility (suburban areas)
   - **40-60**: Moderate accessibility
   - **60-80**: Good accessibility (urban cores)
   - **80-100**: Excellent accessibility (downtown/dense areas)

4. **Why Frontend Calculates Average**:
   - Could fetch from `/api/metrics-summary?type=baseline` → `scores.accessibility_mean` (already computed)
   - Instead, frontend calculates average client-side from loaded nodes
   - This creates **discrepancy** between displayed score and true city average

### Data Flow Diagram

```
compute_scores.py
  ↓ Computes: sum(weight/(distance+1)) for each node
  ↓ Normalizes: min-max → 1-100 scale
  ↓
baseline_nodes_with_scores.csv (182,191 nodes)
  ↓ Column 23: accessibility_score
  ↓
backend/main.py → GET /api/nodes
  ↓ Returns: JSON with accessibility_score field
  ↓
frontend/lib/api.ts → pathLensAPI.getNodes()
  ↓ Fetches: max 2,000 nodes
  ↓
frontend/page.tsx
  ↓ Calculates: average of loaded nodes
  ↓ Displays: {baselineScore.toFixed(1)}
  ↓
USER SEES: "6.1" (example)
```

---

## 🔄 API Endpoints & Data Sources

### 1. **GET `/api/nodes`**

**Called by:** `pathLensAPI.getNodes()` in frontend

**Source Files:**

- `data/analysis/baseline_nodes_with_scores.csv` (76 MB, 182,191 rows)
- `data/analysis/optimized_nodes_with_scores.csv` (78 MB, 182,191 rows)

**Data Structure Sent:**

```typescript
interface Node {
  osmid: string; // OSM node ID
  x: float; // Longitude
  y: float; // Latitude
  accessibility_score: float; // 1-100 scale (normalized)
  walkability_score: float; // 1-100 scale (normalized)
  equity_score: float; // 1-100 scale (normalized)
  travel_time_min: float; // Average travel time in minutes
  betweenness_centrality: float;
  dist_to_school: float; // Distance in meters
  dist_to_hospital: float; // Distance in meters
  dist_to_park: float; // Distance in meters
}
```

**Complete CSV Fields (35 columns):**

- `osmid, y, x, street_count, lon, lat, highway, ref, geometry`
- `degree, is_intersection, avg_incident_length_m`
- `intersection_density_global, link_node_ratio_global`
- `betweenness_centrality, closeness_centrality`
- `dist_to_school, dist_to_hospital, dist_to_pharmacy, dist_to_supermarket, dist_to_bus_station, dist_to_park`
- `accessibility_score, travel_time_min, travel_time_score, structure_score`
- `h3_index, equity_score, walkability`
- `coverage_school, coverage_hospital, coverage_pharmacy, coverage_supermarket, coverage_bus_station, coverage_park`
- `population_weight`

**Query Parameters:**

- `type`: "baseline" | "optimized" (required)
- `limit`: Max 10,000 nodes (optional)
- `offset`: Pagination offset (optional)
- `bbox`: "west,south,east,north" for spatial filtering (optional)

**Used in:**

- `frontend/app/(analysis)/baseline/page.tsx` - Loads baseline nodes
- `frontend/app/(analysis)/optimized/page.tsx` - Loads optimized nodes
- Frontend map components for visualization

---

### 2. **GET `/api/metrics-summary`**

**Called by:** `pathLensAPI.getMetricsSummary(type)`

**Source Files:**

- `data/analysis/baseline_metrics_summary.json` (0.42 KB)
- `data/analysis/optimized_metrics_summary.json` (0.42 KB)

**Data Structure Sent:**

```json
{
  "network": {
    "circuity_sample_ratio": 1.211, // Road network straightness (1.0 = perfect)
    "intersection_density_global": 119.16, // Intersections per km²
    "link_node_ratio_global": 2.676 // Edges/Nodes ratio
  },
  "scores": {
    "accessibility_mean": 6.12, // 1-100 scale
    "walkability_mean": 28.82, // 1-100 scale
    "equity_mean": 94.32, // 1-100 scale
    "travel_time_min_mean": 4.41, // Minutes
    "travel_time_score_mean": 29.06 // 1-100 scale
  }
}
```

**Used in:**

- `frontend/components/MetricsCard.tsx` - Displays network metrics & gauge charts
- `frontend/app/(analysis)/baseline/page.tsx` - Shows overall accessibility score

---

### 3. **GET `/api/h3-aggregations`**

**Called by:** `pathLensAPI.getH3Aggregations(type)` (if implemented)

**Source Files:**

- `data/analysis/baseline_h3_agg.csv` (188 KB, ~2,000 hexagons)
- `data/analysis/optimized_h3_agg.csv` (168 KB, ~1,800 hexagons)

**Data Structure Sent:**

```typescript
{
  hexagons: Array<{
    h3: string; // H3 hexagon ID
    value_mean: float; // Mean walkability
    value_std: float; // Standard deviation
    value_count: int; // Number of nodes
    coverage_school: float; // 0-1 coverage ratio
    coverage_hospital: float;
    coverage_pharmacy: float;
    coverage_supermarket: float;
    coverage_bus_station: float;
    coverage_park: float;
    walkability_variance: float;
    accessibility_mean: float;
    population_weighted_accessibility: float;
  }>;
}
```

**Used in:**

- Future hexagonal heatmap visualizations
- Spatial aggregation analysis

---

### 4. **GET `/api/optimization/results`**

**Called by:** `pathLensAPI.getOptimizationResults()`

**Source File:**

- `data/optimization/runs/best_candidate.json` (2.3 KB)

**Data Structure Sent:**

```json
{
  "generation": 1,
  "candidate": "bus_station:10206640491,10206649677,...|hospital:...|park:...|pharmacy:...|school:...|supermarket:...",
  "template": null,
  "metrics": {
    "fitness": 0.0,
    "distance_gain": 2.01,
    "travel_penalty": 22.62,
    "diversity_penalty": 2319.32,
    "proximity_penalty": 0.0,
    "placements": {
      "school": 20,
      "hospital": 20,
      "pharmacy": 20,
      "supermarket": 20,
      "bus_station": 20,
      "park": 20
    },
    "best_distances": {
      "school": 2872.085, // meters
      "hospital": 4746.043,
      "pharmacy": 4056.718,
      "supermarket": 5483.496,
      "bus_station": 8174.472,
      "park": 2330.732
    },
    "amenity_scores": {
      "school": 0.2795, // 0-1 normalized
      "hospital": 0.3355,
      "pharmacy": 0.3767,
      "supermarket": 0.3735,
      "bus_station": 0.3617,
      "park": 0.2836
    }
  }
}
```

**Used in:**

- Optimization result displays
- Amenity placement analysis

---

### 5. **GET `/api/optimization/summary`**

**Called by:** `pathLensAPI.getOptimizationSummary()`

**Source File:**

- `data/optimization/runs/summary.json` (2.56 KB)

**Data Structure Sent:**

```json
{
  "best_candidate": "...",
  "best_metrics": {
    /* same as optimization/results */
  },
  "config": {
    "population_size": 30,
    "generations": 50,
    "mutation_rate": 0.15,
    "crossover_rate": 0.7,
    "tournament_size": 5,
    "elite_count": 3,
    "max_amenities_per_type": 20,
    "distance_cutoff_km": 10
  },
  "runtime": {
    "total_seconds": 4535.23,
    "start_time": "2026-01-01T03:15:10Z",
    "end_time": "2026-01-01T04:30:45Z"
  },
  "evolution_stats": {
    "initial_fitness": 1234.56,
    "final_fitness": 0.0,
    "improvement_percent": 100.0,
    "generations_with_improvement": 15
  }
}
```

---

### 6. **GET `/api/optimization/pois`**

**Called by:** `pathLensAPI.getOptimizationPois()`

**Source File:**

- `data/optimization/runs/optimized_pois.geojson` (149 KB)

**Data Structure Sent:**

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [77.5987, 12.9105]
      },
      "properties": {
        "osmid": "10206640491",
        "amenity": "bus_station",
        "name": "Bus Stop Name",
        "optimized": true
      }
    }
  ]
}
```

**Contains:** 120 optimized POI placements (20 per amenity type)

---

### 7. **GET `/api/optimization/status`**

**Called by:** `pathLensAPI.getOptimizationStatus()`

**Source File:**

- `data/optimization/runs/progress.json` (0.49 KB)

**Data Structure Sent:**

```json
{
  "status": "completed", // "queued" | "running" | "completed" | "failed"
  "stage": "finalizing", // Current pipeline stage
  "message": "All pipelines completed successfully",
  "percent": 100, // 0-100 progress
  "timestamp": "2026-01-01T04:03:45Z",
  "pipelines": {
    "data": "completed", // "pending" | "running" | "completed" | "failed"
    "optimization": "completed",
    "landuse": "skipped"
  },
  "details": {
    "results": {
      "data": true,
      "optimization": true
    },
    "summary_file": "path/to/summary.json"
  }
}
```

**Used in:**

- Real-time progress tracking during optimization runs
- Frontend polling to show progress bars

---

### 8. **GET `/api/optimization/comparison`**

**Called by:** `pathLensAPI.getOptimizationComparison()`

**Source Files:**

- `data/analysis/baseline_metrics_summary.json`
- `data/analysis/optimized_metrics_summary.json`

**Data Structure Sent:**

```json
{
  "baseline": {
    "network": {
      /* baseline network metrics */
    },
    "scores": {
      /* baseline scores */
    }
  },
  "optimized": {
    "network": {
      /* optimized network metrics */
    },
    "scores": {
      /* optimized scores */
    }
  },
  "improvements": {
    "accessibility_mean": 15.3, // % improvement
    "walkability_mean": 8.7,
    "equity_mean": 2.1,
    "travel_time_min_mean": -12.4 // Negative = reduction
  }
}
```

**Used in:**

- Baseline vs Optimized comparison views
- Impact analysis dashboards

---

### 9. **GET `/api/optimization/history`**

**Called by:** `pathLensAPI.getOptimizationHistory()`

**Source Files:**

- `data/optimization/runs/generation_0001.json` through `generation_0050.json` (10-23 KB each)

**Data Structure Sent:**

```json
{
  "generations": [
    {
      "generation": 1,
      "best_fitness": 1234.56,
      "avg_fitness": 2345.67,
      "candidates": [
        {
          "candidate": "school:...|hospital:...",
          "fitness": 1234.56,
          "placements": {
            /* counts */
          },
          "metrics": {
            /* detailed metrics */
          }
        }
      ]
    }
  ]
}
```

**Contains:** Evolution history across 50 generations (30 candidates per generation)

---

### 10. **GET `/api/suggestions`**

**Called by:** `pathLensAPI.getSuggestions()`

**Source Files:**

- `data/optimization/runs/best_candidate.json`
- `data/optimization/runs/combined_pois.geojson` (725 MB - large!)

**Data Structure Sent:**

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Point", "coordinates": [lon, lat] },
      "properties": {
        "osmid": "string",
        "amenity": "school" | "hospital" | "park" | etc,
        "name": "string",
        "score": 0.85,
        "optimized": true
      }
    }
  ]
}
```

---

## 📊 File Size Summary

### Analysis Directory (`data/analysis/`)

| File                              | Size    | Records | Usage                |
| --------------------------------- | ------- | ------- | -------------------- |
| `baseline_nodes_with_scores.csv`  | 76 MB   | 182,191 | Node-level metrics   |
| `optimized_nodes_with_scores.csv` | 78 MB   | 182,191 | Node-level metrics   |
| `baseline_h3_agg.csv`             | 188 KB  | ~2,000  | Hexagon aggregations |
| `optimized_h3_agg.csv`            | 168 KB  | ~1,800  | Hexagon aggregations |
| `baseline_metrics_summary.json`   | 0.42 KB | 1       | Summary statistics   |
| `optimized_metrics_summary.json`  | 0.42 KB | 1       | Summary statistics   |

### Optimization Directory (`data/optimization/runs/`)

| File                           | Size     | Records | Usage                           |
| ------------------------------ | -------- | ------- | ------------------------------- |
| `best_candidate.json`          | 2.3 KB   | 1       | Best optimization result        |
| `summary.json`                 | 2.56 KB  | 1       | Run statistics                  |
| `optimized_pois.geojson`       | 149 KB   | 120     | Optimized POIs only             |
| `combined_pois.geojson`        | 725 MB   | 40,556  | All POIs (baseline + optimized) |
| `generation_*.json` (50 files) | 10-23 KB | 30 each | Evolution history               |
| `progress.json`                | 0.49 KB  | 1       | Real-time status                |

---

## 🎯 Performance Considerations

### Large Files (Load Time Impact)

1. **`baseline_nodes_with_scores.csv` (76 MB)**:
   - Backend loads in ~3-5 seconds
   - Frontend pagination recommended (limit=2000)
2. **`combined_pois.geojson` (725 MB)**:
   - Very slow to load (~30-60 seconds)
   - Consider using optimized_pois.geojson (149 KB) instead

### Optimization Opportunities

1. **Use Parquet files**: Already available, 3-4x faster loading

   - `baseline_nodes_with_scores.parquet` (29.7 MB) vs CSV (76 MB)
   - Backend could load parquet for better performance

2. **Implement bbox filtering**: Reduce data transfer for map views

3. **Cache frequently accessed data**: metrics_summary.json, best_candidate.json

---

## 🔌 Frontend Usage Map

### `page.tsx` (Baseline Analysis)

- **Calls**: `getNodes({ type: 'baseline', limit: 2000 })`
- **Displays**: Node distribution, critical nodes, overall score

### `MetricsCard.tsx`

- **Calls**: `getMetricsSummary(type)`
- **Displays**: Network metrics (circuity, density, link-node ratio)
- **Displays**: Score gauges (walkability, equity, travel time, access)

### Map Components (future)

- **Calls**: `getNodes({ type, bbox: "..." })`
- **Calls**: `getOptimizationPois()`
- **Displays**: Interactive map with nodes and POI markers

---

## 🚀 Quick Reference: What Shows Where

### Baseline Page Sidebar

- **Overall Score**: From `baselineScore` state (calculated from nodes)
- **Network Metrics**: From `/api/metrics-summary?type=baseline`
- **Node Distribution**: From `/api/nodes?type=baseline&limit=2000`
- **Critical Nodes**: Filtered from loaded nodes (high travel time)

### Optimization Results

- **Best Placements**: From `/api/optimization/results`
- **Evolution History**: From `/api/optimization/history`
- **Comparison Charts**: From `/api/optimization/comparison`

### Real-time Progress

- **Status Bar**: Polls `/api/optimization/status` every 2 seconds
- **Pipeline Progress**: From `progress.json` (pipelines object)
