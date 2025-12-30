## Plan: Frontend Integration with Backend API

Complete frontend implementation to utilize all backend routes and provide a fully functional PathLens interface.

---

## Current State Analysis

### Backend Routes Available
✅ **Data Retrieval:**
- `GET /api/nodes?type=baseline|optimized` - Node-level metrics
- `GET /api/h3-aggregations?type=baseline|optimized` - Hexagon aggregations
- `GET /api/metrics-summary?type=baseline|optimized` - Summary statistics
- `GET /api/pois` - All POI markers
- `GET /api/suggestions` - Optimized placement suggestions

✅ **Analysis & Optimization:**
- `POST /api/optimize` - Trigger optimization pipeline
- `GET /api/optimization/status` - Poll optimization progress
- `POST /api/rescore` - Re-run scoring with selections

✅ **Landuse (Optional):**
- `GET /api/feasibility/{amenity}` - Feasibility analysis results
- `GET /api/placements/{amenity}` - Buildable land polygons

### Frontend Current State
⚠️ **Issues:**
- TypeScript errors in `optimized/page.tsx` - accessing `suggestion.id` instead of `suggestion.properties.id`
- Missing properties in `Suggestion` interface (`description`, `impact_score`, `cost_estimate`)
- Not using most backend endpoints (only `getNodes` and `getSuggestions`)
- Demo mode is the default experience

✅ **Working:**
- Map component with smooth transitions
- Demo mode toggle
- Basic layout structure
- Baseline and optimized pages

---

## Implementation Steps

### Phase 1: Fix Critical TypeScript Errors 🔧

**1.1. Update Suggestion Interface** (`lib/store.ts`)
```typescript
export interface Suggestion {
  type: string;
  geometry: {
    type: string;
    coordinates: [number, number];
  };
  properties: {
    id: string;
    amenity_type: string;
    expected_impact?: number;
    affected_nodes?: number;
    land_availability?: 'feasible' | 'limited' | 'unavailable';
    available_area_sqm?: number;
    satellite_confidence?: number;
    description?: string;         // ADD
    impact_score?: number;        // ADD
    cost_estimate?: string | number; // ADD
  };
}
```

**1.2. Fix Suggestion Access Pattern** (`app/(analysis)/optimized/page.tsx`)

Replace all instances:
- `suggestion.id` → `suggestion.properties.id`
- Lines: 179, 183, 189, 202

**1.3. Update API Base URL** (`lib/api.ts`)
```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
```

---

### Phase 2: Integrate Metrics Dashboard 📊

**2.1. Create MetricsCard Component** (`components/MetricsCard.tsx`)

Display summary statistics from `/api/metrics-summary`:

```typescript
interface MetricsCardProps {
  type: 'baseline' | 'optimized';
}

export function MetricsCard({ type }: MetricsCardProps) {
  const [metrics, setMetrics] = useState(null);
  
  useEffect(() => {
    pathLensAPI.getMetricsSummary(type).then(setMetrics);
  }, [type]);
  
  return (
    <Card>
      <h3>Network Metrics</h3>
      <div>Walkability: {metrics?.scores.walkability_mean}</div>
      <div>Accessibility: {metrics?.scores.accessibility_mean}</div>
      <div>Equity: {metrics?.scores.equity_mean}</div>
      <div>Avg Travel Time: {metrics?.scores.travel_time_min_mean} min</div>
    </Card>
  );
}
```

**2.2. Add API Method** (`lib/api.ts`)
```typescript
getMetricsSummary: async (type: 'baseline' | 'optimized') => {
  const response = await api.get(`/api/metrics-summary?type=${type}`);
  return response.data;
},
```

**2.3. Integrate into Pages**

Add `<MetricsCard type="baseline" />` to `baseline/page.tsx`
Add `<MetricsCard type="optimized" />` to `optimized/page.tsx`

---

### Phase 3: Add H3 Hexagon Heatmap Visualization 🗺️

**3.1. Create H3Layer Component** (`components/H3Layer.tsx`)

Renders hexagon aggregations on map:

```typescript
import L from 'leaflet';

interface H3LayerProps {
  type: 'baseline' | 'optimized';
  mapInstance: L.Map | null;
  visible: boolean;
}

export function H3Layer({ type, mapInstance, visible }: H3LayerProps) {
  const [hexagons, setHexagons] = useState([]);
  
  useEffect(() => {
    if (!visible) return;
    
    pathLensAPI.getH3Aggregations(type).then(data => {
      setHexagons(data.hexagons);
    });
  }, [type, visible]);
  
  useEffect(() => {
    if (!mapInstance || !visible) return;
    
    const layer = L.layerGroup();
    hexagons.forEach(hex => {
      // Convert H3 to polygon and add to map
      const polygon = L.polygon(hexToBoundary(hex.h3_index), {
        fillColor: getColorByWalkability(hex.walkability_mean),
        fillOpacity: 0.5,
        weight: 1
      });
      polygon.addTo(layer);
    });
    
    layer.addTo(mapInstance);
    return () => layer.remove();
  }, [mapInstance, hexagons, visible]);
}
```

**3.2. Add Toggle for Heatmap**

Add checkbox in sidebar to show/hide hexagon layer:
```typescript
<Switch
  checked={showHexagons}
  onCheckedChange={setShowHexagons}
  label="Show Hexagon Heatmap"
/>
```

**3.3. Add API Method** (`lib/api.ts`)
```typescript
getH3Aggregations: async (type: 'baseline' | 'optimized') => {
  const response = await api.get(`/api/h3-aggregations?type=${type}`);
  return response.data;
},
```

**Note:** Need to handle H3 to polygon conversion. Options:
- Use `h3-js` library client-side
- Or have backend return full GeoJSON with polygons

---

### Phase 4: Enhance POI Visualization 📍

**4.1. Create POILayer Component** (`components/POILayer.tsx`)

Show existing vs optimized POIs with different markers:

```typescript
interface POILayerProps {
  mapInstance: L.Map | null;
  source?: 'all' | 'existing' | 'optimized';
}

export function POILayer({ mapInstance, source = 'all' }: POILayerProps) {
  const [pois, setPOIs] = useState(null);
  
  useEffect(() => {
    pathLensAPI.getPOIs(source).then(setPOIs);
  }, [source]);
  
  useEffect(() => {
    if (!mapInstance || !pois) return;
    
    const markers = pois.features.map(feature => {
      const icon = feature.properties.source === 'optimized' 
        ? greenMarkerIcon 
        : blueMarkerIcon;
      
      return L.marker(feature.geometry.coordinates.reverse(), { icon })
        .bindPopup(`
          <strong>${feature.properties.category}</strong><br/>
          ${feature.properties.name || 'Unnamed'}<br/>
          Source: ${feature.properties.source}
        `);
    });
    
    const layer = L.layerGroup(markers);
    layer.addTo(mapInstance);
    return () => layer.remove();
  }, [mapInstance, pois]);
}
```

**4.2. Add Filter Controls**

In sidebar, add POI filtering:
```typescript
<Select value={poiFilter} onValueChange={setPOIFilter}>
  <option value="all">All POIs</option>
  <option value="existing">Existing Only</option>
  <option value="optimized">Optimized Only</option>
</Select>

<Select value={categoryFilter} onValueChange={setCategoryFilter}>
  <option value="">All Categories</option>
  <option value="school">Schools</option>
  <option value="hospital">Hospitals</option>
  <option value="park">Parks</option>
</Select>
```

**4.3. Add API Method** (`lib/api.ts`)
```typescript
getPOIs: async (source?: string) => {
  const params = source ? `?source=${source}` : '';
  const response = await api.get(`/api/pois${params}`);
  return response.data;
},
```

---

### Phase 5: Implement Optimization Progress Tracking ⏳

**5.1. Create OptimizationStatus Component** (`components/OptimizationStatus.tsx`)

Poll optimization status and show progress:

```typescript
export function OptimizationStatus() {
  const [status, setStatus] = useState(null);
  const [polling, setPolling] = useState(false);
  
  useEffect(() => {
    if (!polling) return;
    
    const interval = setInterval(async () => {
      const data = await pathLensAPI.getOptimizationStatus();
      setStatus(data);
      
      if (data.status === 'completed' || data.status === 'failed') {
        setPolling(false);
      }
    }, 2000); // Poll every 2 seconds
    
    return () => clearInterval(interval);
  }, [polling]);
  
  if (!status || status.status === 'not_started') return null;
  
  return (
    <Card>
      <h3>Optimization Progress</h3>
      <Progress 
        value={(status.current_generation / 100) * 100} 
      />
      <p>Generation {status.current_generation} / 100</p>
      <p>Best Fitness: {status.best_fitness?.toFixed(3)}</p>
      <p>Elapsed: {status.elapsed_time?.toFixed(1)}s</p>
    </Card>
  );
}
```

**5.2. Add API Method** (`lib/api.ts`)
```typescript
getOptimizationStatus: async () => {
  const response = await api.get('/api/optimization/status');
  return response.data;
},
```

**5.3. Integrate into Optimize Flow**

When user clicks "Run Optimization":
1. Call `POST /api/optimize`
2. Start polling with `<OptimizationStatus />`
3. When complete, reload nodes and suggestions

---

### Phase 6: Add Landuse Feasibility Visualization (Optional) 🌍

**6.1. Create FeasibilityLayer Component** (`components/FeasibilityLayer.tsx`)

Show buildable land polygons for each amenity:

```typescript
interface FeasibilityLayerProps {
  amenity: string;
  mapInstance: L.Map | null;
}

export function FeasibilityLayer({ amenity, mapInstance }: FeasibilityLayerProps) {
  const [placements, setPlacements] = useState(null);
  
  useEffect(() => {
    pathLensAPI.getPlacements(amenity).then(setPlacements);
  }, [amenity]);
  
  useEffect(() => {
    if (!mapInstance || !placements) return;
    
    const layer = L.geoJSON(placements, {
      style: {
        fillColor: '#10b981',
        fillOpacity: 0.3,
        color: '#059669',
        weight: 2
      },
      onEachFeature: (feature, layer) => {
        layer.bindPopup(`
          <strong>Feasible Site</strong><br/>
          Area: ${feature.properties.free_area_m2} m²<br/>
          Confidence: ${feature.properties.satellite_confidence}%
        `);
      }
    });
    
    layer.addTo(mapInstance);
    return () => layer.remove();
  }, [mapInstance, placements]);
}
```

**6.2. Add API Methods** (`lib/api.ts`)
```typescript
getFeasibility: async (amenity: string) => {
  const response = await api.get(`/api/feasibility/${amenity}`);
  return response.data;
},

getPlacements: async (amenity: string) => {
  const response = await api.get(`/api/placements/${amenity}`);
  return response.data;
},
```

**6.3. Add UI Toggle**

In optimized page, for each suggestion, show feasibility status:
```typescript
{suggestion.properties.land_availability === 'feasible' && (
  <Badge variant="success">Feasible Land Available</Badge>
)}
```

---

### Phase 7: Enhance Map Component 🗺️

**7.1. Add Layer Management**

Update `MapComponent` to support multiple layers:

```typescript
interface MapComponentProps {
  nodes?: MapNode[];
  suggestions?: MapSuggestion[];
  showHexagons?: boolean;
  showPOIs?: boolean;
  poiSource?: 'all' | 'existing' | 'optimized';
  feasibilityAmenity?: string;
  // ... existing props
}
```

**7.2. Add Legend Component** (`components/MapLegend.tsx`)

Show what each marker/color represents:

```typescript
export function MapLegend() {
  return (
    <div className="absolute bottom-4 right-4 bg-white/90 p-4 rounded-lg shadow-lg z-10">
      <h4>Legend</h4>
      <div className="flex items-center gap-2">
        <div className="w-4 h-4 bg-blue-500 rounded-full" />
        <span>Existing POIs</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-4 h-4 bg-green-500 rounded-full" />
        <span>Optimized Placements</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-4 h-4" style={{ background: 'linear-gradient(to right, red, yellow, green)' }} />
        <span>Walkability Score</span>
      </div>
    </div>
  );
}
```

**7.3. Add Map Controls**

Add zoom controls, layer switcher, and map style toggle:

```typescript
<div className="absolute top-4 left-4 z-10 space-y-2">
  <Button onClick={() => map.zoomIn()}>+</Button>
  <Button onClick={() => map.zoomOut()}>-</Button>
  <Button onClick={() => map.fitBounds(bounds)}>Fit Bounds</Button>
</div>
```

---

### Phase 8: Create Comparison View 📊

**8.1. Add Comparison Page** (`app/(analysis)/compare/page.tsx`)

Side-by-side or split-screen comparison:

```typescript
export default function ComparePage() {
  return (
    <div className="grid grid-cols-2 gap-4 h-full">
      <div>
        <h2>Baseline</h2>
        <MetricsCard type="baseline" />
        <MapView type="baseline" />
      </div>
      <div>
        <h2>Optimized</h2>
        <MetricsCard type="optimized" />
        <MapView type="optimized" />
      </div>
    </div>
  );
}
```

**8.2. Add Difference Metrics**

Calculate and display improvements:

```typescript
const improvement = {
  walkability: (optimized.walkability - baseline.walkability) / baseline.walkability * 100,
  accessibility: (optimized.accessibility - baseline.accessibility) / baseline.accessibility * 100,
  travelTime: (baseline.travelTime - optimized.travelTime) / baseline.travelTime * 100,
};

<Card>
  <h3>Improvements</h3>
  <div>Walkability: +{improvement.walkability.toFixed(1)}%</div>
  <div>Accessibility: +{improvement.accessibility.toFixed(1)}%</div>
  <div>Travel Time: -{improvement.travelTime.toFixed(1)}%</div>
</Card>
```

---

### Phase 9: Add Configuration/Settings Page ⚙️

**9.1. Create Settings Page** (`app/settings/page.tsx`)

Allow users to configure optimization parameters:

```typescript
export default function SettingsPage() {
  const { 
    budget, 
    maxAmenities, 
    addSchools, 
    addHospitals, 
    addParks,
    setBudget,
    setMaxAmenities,
    setAddSchools,
    // ... other setters
  } = usePathLensStore();
  
  return (
    <div className="p-6">
      <h1>Settings</h1>
      
      <Card>
        <h2>Optimization Parameters</h2>
        <Label>Budget ($)</Label>
        <Input type="number" value={budget} onChange={e => setBudget(+e.target.value)} />
        
        <Label>Max Amenities</Label>
        <Input type="number" value={maxAmenities} onChange={e => setMaxAmenities(+e.target.value)} />
        
        <Label>Amenity Types to Add</Label>
        <Switch checked={addSchools} onCheckedChange={setAddSchools} label="Schools" />
        <Switch checked={addHospitals} onCheckedChange={setAddHospitals} label="Hospitals" />
        <Switch checked={addParks} onCheckedChange={setAddParks} label="Parks" />
      </Card>
      
      <Card>
        <h2>API Configuration</h2>
        <Label>Backend URL</Label>
        <Input value={apiUrl} onChange={e => setApiUrl(e.target.value)} />
      </Card>
    </div>
  );
}
```

---

### Phase 10: Error Handling & Loading States 🔄

**10.1. Create Loading Component** (`components/LoadingState.tsx`)

```typescript
export function LoadingState({ message = 'Loading...' }) {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <Spinner className="mx-auto mb-4" />
        <p>{message}</p>
      </div>
    </div>
  );
}
```

**10.2. Create Error Component** (`components/ErrorState.tsx`)

```typescript
export function ErrorState({ error, retry }: { error: string; retry?: () => void }) {
  return (
    <div className="flex items-center justify-center h-full">
      <Card className="p-6 text-center">
        <AlertCircle className="mx-auto mb-4 text-red-500" size={48} />
        <h3>Error Loading Data</h3>
        <p className="text-gray-500">{error}</p>
        {retry && (
          <Button onClick={retry} className="mt-4">
            Try Again
          </Button>
        )}
      </Card>
    </div>
  );
}
```

**10.3. Add to All Data Fetching**

Wrap all API calls with error handling:

```typescript
const [error, setError] = useState(null);
const [loading, setLoading] = useState(true);

useEffect(() => {
  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await pathLensAPI.getNodes('baseline');
      setNodes(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  load();
}, []);

if (loading) return <LoadingState />;
if (error) return <ErrorState error={error} retry={() => window.location.reload()} />;
```

---

## Implementation Priority

### Sprint 1: Critical Fixes (Immediate)
1. ✅ Fix TypeScript errors in `Suggestion` interface
2. ✅ Fix suggestion access pattern (`suggestion.properties.id`)
3. ✅ Update API base URL to port 8001
4. ✅ Test basic data loading from backend

### Sprint 2: Core Features (Week 1)
5. 📊 Add MetricsCard component and integrate summary stats
6. 🗺️ Add H3 hexagon heatmap visualization
7. 📍 Enhance POI layer with filtering
8. ⏳ Add optimization status polling

### Sprint 3: Enhanced UX (Week 2)
9. 🗺️ Add map legend and controls
10. 🔄 Add loading states and error handling
11. 📊 Create comparison view
12. ⚙️ Add settings page

### Sprint 4: Advanced Features (Week 3)
13. 🌍 Add landuse feasibility visualization
14. 📱 Add responsive design improvements
15. 🎨 Polish animations and transitions
16. 📝 Add tooltips and help text

---

## File Structure

```
frontend/
├── app/
│   ├── (analysis)/
│   │   ├── layout.tsx          # Shared layout with map
│   │   ├── baseline/
│   │   │   └── page.tsx        # ✅ Existing - needs metrics integration
│   │   ├── optimized/
│   │   │   └── page.tsx        # ⚠️ Existing - needs TS fixes
│   │   └── compare/
│   │       └── page.tsx        # 🆕 NEW - side-by-side comparison
│   ├── settings/
│   │   └── page.tsx            # 🆕 NEW - configuration
│   └── page.tsx
├── components/
│   ├── MapComponent.tsx        # ✅ Existing - working well
│   ├── AnalysisHeader.tsx      # ✅ Existing - demo toggle
│   ├── MetricsCard.tsx         # 🆕 NEW
│   ├── H3Layer.tsx             # 🆕 NEW
│   ├── POILayer.tsx            # 🆕 NEW
│   ├── FeasibilityLayer.tsx   # 🆕 NEW
│   ├── OptimizationStatus.tsx # 🆕 NEW
│   ├── MapLegend.tsx           # 🆕 NEW
│   ├── LoadingState.tsx        # 🆕 NEW
│   └── ErrorState.tsx          # 🆕 NEW
└── lib/
    ├── api.ts                  # ⚠️ Needs new methods
    ├── store.ts                # ⚠️ Needs Suggestion fix
    ├── demo-data.ts            # ✅ Existing
    └── utils.ts                # ✅ Existing
```

---

## Testing Checklist

### Data Loading
- [ ] Baseline nodes load correctly
- [ ] Optimized nodes load correctly
- [ ] Suggestions load correctly
- [ ] Metrics load correctly
- [ ] H3 aggregations load correctly
- [ ] POIs load correctly

### Map Interactions
- [ ] Nodes render on map
- [ ] Markers have correct colors
- [ ] Smooth transitions work
- [ ] Hexagon layer displays
- [ ] POI markers show
- [ ] Legend is readable

### Optimization Flow
- [ ] Optimize button triggers backend
- [ ] Status polling works
- [ ] Progress updates in real-time
- [ ] Results load after completion
- [ ] Suggestions appear on map

### Error Handling
- [ ] Missing data shows appropriate message
- [ ] Network errors handled gracefully
- [ ] Loading states display correctly
- [ ] Retry functionality works

### Demo Mode
- [ ] Toggle switches between demo/real data
- [ ] Demo data generates correctly
- [ ] No API calls in demo mode
- [ ] Real data loads when toggled off

---

## Further Considerations

### Performance
- Large datasets (50K+ nodes) may be slow to render
- Consider pagination or clustering for node markers
- Use virtualization for long lists
- Cache API responses to avoid repeated requests

### Accessibility
- Add ARIA labels to interactive elements
- Ensure keyboard navigation works
- Add alt text to icons
- Test with screen readers

### Mobile Responsiveness
- Map component should resize properly
- Sidebar should collapse on mobile
- Touch events for map interactions
- Responsive grid for comparison view

### Documentation
- Add JSDoc comments to components
- Create user guide for features
- Document API integration points
- Add inline help tooltips
