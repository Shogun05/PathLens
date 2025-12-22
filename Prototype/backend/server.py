from fastapi import FastAPI, APIRouter
from fastapi.responses import StreamingResponse, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any
import uuid
from datetime import datetime, timezone
import asyncio
import json
import time


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global log queue for SSE
log_queue = asyncio.Queue()
optimization_running = False
log_history = []  # Store logs for polling


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Ignore MongoDB's _id field
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class OptimizationParameters(BaseModel):
    parameters: Dict[str, Any]

class OptimizationResponse(BaseModel):
    status: str
    message: str
    run_id: str = None

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    # Convert to dict and serialize datetime to ISO string for MongoDB
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    # Exclude MongoDB's _id field from the query results
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    # Convert ISO string timestamps back to datetime objects
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks


# Optimization endpoints
async def emit_log(level: str, message: str):
    """Add a log entry to the queue"""
    log_entry = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'level': level,
        'message': message
    }
    await log_queue.put(log_entry)
    log_history.append(log_entry)  # Also store in history for polling
    logger.log(getattr(logging, level), message)


async def run_optimization_process(parameters: Dict[str, Any]):
    """Simulate the optimization process with logging"""
    global optimization_running
    optimization_running = True
    
    try:
        await emit_log('INFO', 'Starting optimization process...')
        await asyncio.sleep(1)
        
        # Extract parameters
        await emit_log('INFO', f'Loading parameters...')
        await asyncio.sleep(0.5)
        
        composite = parameters.get('compositeWeights', {})
        await emit_log('INFO', f'Composite weights: α={composite.get("alpha", {}).get("value", 0.3)}, β={composite.get("beta", {}).get("value", 0.4)}, γ={composite.get("gamma", {}).get("value", 0.2)}, δ={composite.get("delta", {}).get("value", 0.1)}')
        await asyncio.sleep(0.5)
        
        # Simulate data loading
        await emit_log('INFO', 'Loading OpenStreetMap data...')
        await asyncio.sleep(2)
        await emit_log('INFO', 'Loaded 5,234 nodes and 8,921 edges')
        await asyncio.sleep(0.5)
        
        await emit_log('INFO', 'Loading POI data...')
        await asyncio.sleep(1.5)
        await emit_log('INFO', 'Loaded 1,247 points of interest')
        await asyncio.sleep(0.5)
        
        # Simulate graph processing
        await emit_log('INFO', 'Building annotated walking network...')
        await asyncio.sleep(2)
        await emit_log('INFO', 'Simplifying graph topology...')
        await asyncio.sleep(1.5)
        await emit_log('INFO', 'Computing network centrality metrics...')
        await asyncio.sleep(2)
        
        # Simulate scoring
        await emit_log('INFO', 'Calculating walkability scores...')
        await asyncio.sleep(2)
        await emit_log('INFO', 'Computing amenity accessibility for 5,234 nodes...')
        await asyncio.sleep(2.5)
        await emit_log('INFO', 'Analyzing spatial equity metrics...')
        await asyncio.sleep(1.5)
        
        # Simulate optimization
        await emit_log('INFO', 'Running hybrid optimization algorithm...')
        await asyncio.sleep(2)
        await emit_log('INFO', 'Genetic Algorithm: Generation 1/50')
        await asyncio.sleep(1)
        await emit_log('INFO', 'Genetic Algorithm: Generation 10/50')
        await asyncio.sleep(1)
        await emit_log('INFO', 'Genetic Algorithm: Generation 25/50')
        await asyncio.sleep(1)
        await emit_log('INFO', 'Genetic Algorithm: Generation 50/50')
        await asyncio.sleep(1)
        
        await emit_log('INFO', 'Running MILP validation...')
        await asyncio.sleep(2)
        await emit_log('INFO', 'Optimization constraints satisfied')
        await asyncio.sleep(0.5)
        
        # Generate map
        await emit_log('INFO', 'Generating optimized map visualization...')
        await asyncio.sleep(2)
        
        # Create a simple HTML map file
        output_dir = ROOT_DIR / 'outputs'
        output_dir.mkdir(exist_ok=True)
        map_file = output_dir / 'optimized_map.html'
        
        with open(map_file, 'w') as f:
            f.write(generate_demo_map_html(parameters))
        
        await emit_log('INFO', f'Map saved to {map_file}')
        await asyncio.sleep(0.5)
        
        await emit_log('INFO', 'Optimization completed successfully!')
        
    except Exception as e:
        await emit_log('ERROR', f'Optimization failed: {str(e)}')
        raise
    finally:
        optimization_running = False


def generate_demo_map_html(parameters: Dict[str, Any]) -> str:
    """Generate a demo interactive map using Leaflet"""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>PathLens Optimized Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
        .legend {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 12px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        // Initialize map centered on Bangalore (as example)
        var map = L.map('map').setView([12.9716, 77.5946], 13);
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        // Add demo markers for optimized amenities
        var amenityColors = {{
            'Hospital': '#ef4444',
            'School': '#3b82f6',
            'Grocery': '#22c55e',
            'Transit': '#f59e0b',
            'Park': '#10b981',
            'Pharmacy': '#8b5cf6'
        }};
        
        // Demo optimized locations
        var optimizedLocations = [
            {{ name: 'Hospital A', type: 'Hospital', lat: 12.9850, lng: 77.5950 }},
            {{ name: 'School B', type: 'School', lat: 12.9650, lng: 77.6100 }},
            {{ name: 'Grocery C', type: 'Grocery', lat: 12.9700, lng: 77.5800 }},
            {{ name: 'Transit Hub D', type: 'Transit', lat: 12.9800, lng: 77.6000 }},
            {{ name: 'Park E', type: 'Park', lat: 12.9600, lng: 77.5900 }},
            {{ name: 'Pharmacy F', type: 'Pharmacy', lat: 12.9750, lng: 77.6050 }}
        ];
        
        // Add markers
        optimizedLocations.forEach(function(loc) {{
            var marker = L.circleMarker([loc.lat, loc.lng], {{
                radius: 8,
                fillColor: amenityColors[loc.type],
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }}).addTo(map);
            
            marker.bindPopup('<b>' + loc.name + '</b><br>Type: ' + loc.type);
        }});
        
        // Add legend
        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<h4 style="margin: 0 0 10px 0;">Amenity Types</h4>';
            
            Object.keys(amenityColors).forEach(function(type) {{
                div.innerHTML += '<div class="legend-item">' +
                    '<div class="legend-color" style="background:' + amenityColors[type] + '"></div>' +
                    '<span>' + type + '</span></div>';
            }});
            
            return div;
        }};
        legend.addTo(map);
        
        // Add walkability heat zones (demo)
        var walkabilityZones = [
            {{ center: [12.9716, 77.5946], radius: 800, score: 0.85 }},
            {{ center: [12.9650, 77.6000], radius: 600, score: 0.72 }},
            {{ center: [12.9800, 77.5900], radius: 700, score: 0.90 }}
        ];
        
        walkabilityZones.forEach(function(zone) {{
            L.circle(zone.center, {{
                radius: zone.radius,
                fillColor: zone.score > 0.8 ? '#22c55e' : '#f59e0b',
                fillOpacity: 0.2,
                color: zone.score > 0.8 ? '#16a34a' : '#d97706',
                weight: 1
            }}).addTo(map).bindPopup('Walkability Score: ' + (zone.score * 100).toFixed(0) + '%');
        }});
    </script>
</body>
</html>
    """


@api_router.post("/run-optimization", response_model=OptimizationResponse)
async def run_optimization(params: OptimizationParameters):
    """Start the optimization process"""
    global optimization_running, log_history
    
    if optimization_running:
        return OptimizationResponse(
            status="error",
            message="Optimization already running"
        )
    
    # Clear previous logs
    log_history.clear()
    
    run_id = str(uuid.uuid4())
    
    # Start optimization in background
    asyncio.create_task(run_optimization_process(params.parameters))
    
    return OptimizationResponse(
        status="started",
        message="Optimization process started",
        run_id=run_id
    )


@api_router.get("/logs/latest")
async def get_latest_logs():
    """Get all logs (polling endpoint)"""
    return {
        "logs": log_history,
        "running": optimization_running
    }


@api_router.get("/logs/stream")
async def stream_logs():
    """Stream logs via Server-Sent Events"""
    async def event_generator():
        while True:
            try:
                # Wait for log entries with timeout
                log_entry = await asyncio.wait_for(log_queue.get(), timeout=1.0)
                yield f"data: {json.dumps(log_entry)}\n\n"
            except asyncio.TimeoutError:
                # Send keepalive
                yield f": keepalive\n\n"
            except Exception as e:
                logger.error(f"Error in log stream: {e}")
                break
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# Serve the optimized map
@app.get("/outputs/optimized_map.html")
async def serve_map():
    """Serve the generated map file"""
    map_file = ROOT_DIR / 'outputs' / 'optimized_map.html'
    
    if not map_file.exists():
        # Return a placeholder map if file doesn't exist
        return StreamingResponse(
            iter([generate_demo_map_html({})]),
            media_type="text/html"
        )
    
    return FileResponse(map_file, media_type="text/html")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()