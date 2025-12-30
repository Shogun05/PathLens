# PathLens Backend API

FastAPI-based backend service for the PathLens urban planning optimization system.

## Overview

This backend provides RESTful API endpoints to:
- Serve baseline and optimized walkability/accessibility node data
- Provide H3 hexagon aggregations for heatmap visualization
- Return optimization suggestions and POI data
- Trigger optimization and rescoring pipelines
- Monitor optimization progress
- Serve landuse feasibility analysis results

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Option 1: Using the startup script
./start_server.sh

# Option 2: Directly with Python
python3 main.py

# Option 3: With uvicorn (for production)
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

The API will be available at:
- **API Base**: http://localhost:8001
- **Interactive Docs**: http://localhost:8001/docs
- **OpenAPI Schema**: http://localhost:8001/openapi.json

## API Endpoints

### Core Data Endpoints

#### `GET /`
Root endpoint showing API information and available endpoints.

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "data_dir_exists": true,
  "baseline_data_exists": true,
  "data_dir": "/path/to/data/analysis"
}
```

#### `GET /api/nodes?type={baseline|optimized}`
Retrieve node-level walkability and accessibility scores.

**Parameters:**
- `type` (required): Either `baseline` or `optimized`

**Response:** Array of nodes with scores
```json
[
  {
    "osmid": "123456",
    "x": 77.5946,
    "y": 12.9716,
    "accessibility_score": 0.85,
    "walkability_score": 0.72,
    "equity_score": 0.68,
    "travel_time_min": 12.5,
    "betweenness_centrality": 0.003,
    "dist_to_school": 450,
    "dist_to_hospital": 1200,
    "dist_to_park": 300
  }
]
```

#### `GET /api/h3-aggregations?type={baseline|optimized}`
Retrieve H3 hexagon aggregations for heatmap visualization.

**Parameters:**
- `type` (required): Either `baseline` or `optimized`

**Response:**
```json
{
  "hexagons": [
    {
      "h3_index": "8c2a100890c3fff",
      "mean_walkability": 0.75,
      "mean_accessibility": 0.82,
      "node_count": 145,
      "coverage_school": 0.85,
      "coverage_hospital": 0.65
    }
  ]
}
```

#### `GET /api/metrics-summary?type={baseline|optimized}`
Get global summary statistics for dashboard cards.

**Parameters:**
- `type` (required): Either `baseline` or `optimized`

**Response:**
```json
{
  "mean_walkability": 0.72,
  "mean_accessibility": 0.68,
  "mean_travel_time": 14.5,
  "circuity_ratio": 1.15,
  "intersection_density": 120.5,
  "total_nodes": 15420
}
```

### POI & Optimization Endpoints

#### `GET /api/pois`
Retrieve all POIs (existing + optimized).

**Response:** GeoJSON FeatureCollection
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [77.5946, 12.9716]
      },
      "properties": {
        "amenity": "hospital",
        "name": "City Hospital",
        "source": "existing"
      }
    }
  ]
}
```

#### `GET /api/suggestions`
Get optimization suggestions (newly placed amenities only).

**Response:** GeoJSON FeatureCollection with metadata
```json
{
  "type": "FeatureCollection",
  "features": [...],
  "metadata": {
    "generation": 45,
    "fitness": 0.92
  }
}
```

#### `GET /api/optimization/status`
Monitor optimization progress.

**Response:**
```json
{
  "status": "running",
  "current_generation": 25,
  "best_fitness": 0.85,
  "elapsed_time": 120.5,
  "timestamp": "2025-12-30T10:15:00"
}
```

### Pipeline Execution Endpoints

#### `POST /api/optimize`
Trigger optimization pipeline.

**Request Body:**
```json
{
  "location": "Bangalore, India",
  "budget": 1000000,
  "max_amenities": 10,
  "add_schools": true,
  "add_hospitals": true,
  "add_parks": true
}
```

**Response:**
```json
{
  "status": "started",
  "message": "Optimization process started in background",
  "process_id": 12345
}
```

#### `POST /api/rescore`
Trigger rescoring with custom POI placements.

**Request Body:**
```json
{
  "location": "Bangalore, India",
  "selected_ids": ["node_123", "node_456"]
}
```

**Response:**
```json
{
  "status": "started",
  "message": "Rescoring process started",
  "process_id": 12346
}
```

### Landuse Analysis Endpoints

#### `GET /api/feasibility/{amenity}`
Get feasibility analysis for specific amenity type.

**Parameters:**
- `amenity` (path): Amenity type (e.g., `hospital`, `school`, `park`)

**Response:**
```json
{
  "feasible_nodes": [
    {
      "node_id": "123456",
      "amenity": "hospital",
      "free_area_m2": 5000,
      "min_area_req": 3000,
      "feasible": true,
      "has_patch": true,
      "patch_count": 2
    }
  ]
}
```

#### `GET /api/placements/{amenity}`
Get buildable area geometries for amenity type.

**Parameters:**
- `amenity` (path): Amenity type

**Response:** GeoJSON FeatureCollection with placement polygons

## Data File Mapping

The backend serves data from the following files (see `files.md` for details):

| Endpoint | File Path |
|----------|-----------|
| `/api/nodes?type=baseline` | `data/analysis/baseline_nodes_with_scores.csv` |
| `/api/nodes?type=optimized` | `data/analysis/optimized_nodes_with_scores.csv` |
| `/api/h3-aggregations?type=baseline` | `data/analysis/baseline_h3_agg.csv` |
| `/api/h3-aggregations?type=optimized` | `data/analysis/optimized_h3_agg.csv` |
| `/api/metrics-summary?type=baseline` | `data/analysis/baseline_metrics_summary.json` |
| `/api/metrics-summary?type=optimized` | `data/analysis/optimized_metrics_summary.json` |
| `/api/pois` | `data/optimization/runs/combined_pois.geojson` |
| `/api/suggestions` | `data/optimization/runs/best_candidate.json` + `combined_pois.geojson` |
| `/api/feasibility/{amenity}` | `data/landuse/feasibility_{amenity}.csv` |
| `/api/placements/{amenity}` | `data/landuse/gee_placements_{amenity}.geojson` |
| `/api/optimization/status` | `data/optimization/runs/progress.json` |

## Architecture

### Technology Stack
- **Framework**: FastAPI
- **Server**: Uvicorn
- **Data Processing**: Pandas
- **Validation**: Pydantic

### Design Patterns
- RESTful API with clear resource naming
- Graceful fallbacks when optional data is missing
- Background process execution for long-running pipelines
- Health checks for service monitoring
- CORS enabled for frontend integration

### Error Handling
- 404: Data file not found
- 500: Server-side processing errors
- Detailed error messages in logs

## Development

### Project Structure
```
backend/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── start_server.sh      # Startup script
├── files.md            # Data file documentation
└── README.md           # This file
```

### Adding New Endpoints

1. Define Pydantic model if needed:
```python
class MyModel(BaseModel):
    field: str
```

2. Add endpoint handler:
```python
@app.get("/api/my-endpoint")
async def my_endpoint():
    # Implementation
    return {"data": "value"}
```

3. Update root endpoint documentation in `root()` function

### Testing

```bash
# Test root endpoint
curl http://localhost:8001/

# Test health check
curl http://localhost:8001/health

# Test data endpoint
curl http://localhost:8001/api/nodes?type=baseline

# Interactive API documentation
# Navigate to http://localhost:8001/docs
```

## Integration with Frontend

The frontend should connect to this backend at:
```javascript
const API_BASE_URL = 'http://localhost:8001';
```

All endpoints return JSON (except GeoJSON endpoints which return GeoJSON).

## Notes

- The backend runs on **port 8001** (not 8000) to avoid conflicts
- Large CSV files are loaded into memory - for production, consider pagination
- Pipeline execution endpoints return immediately and run processes in background
- Monitor optimization progress via `/api/optimization/status`
- The backend serves as a data gateway between the pipeline outputs and the frontend
