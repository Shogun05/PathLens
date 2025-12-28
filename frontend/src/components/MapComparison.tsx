import { useEffect, useState, useMemo } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// --- Fix for Leaflet Icons ---
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

// --- Types ---
interface NodeData {
  osmid: string;
  lat: number;
  lon: number;
  walkability: number;
  dist_to_school: number;
  dist_to_hospital: number;
  dist_to_park: number;
  population_density?: number;
}

export default function MapComparison() {
  const [baselineData, setBaselineData] = useState<NodeData[]>([]);
  const [optimizedData, setOptimizedData] = useState<NodeData[]>([]);
  const [viewMode, setViewMode] = useState<'baseline' | 'optimized'>('baseline');
  const [loading, setLoading] = useState(true);

  // Fetch Data
  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log("Fetching data...");
        const [baseRes, optRes] = await Promise.all([
          axios.get('http://127.0.0.1:8000/api/nodes?type=baseline'),
          axios.get('http://127.0.0.1:8000/api/nodes?type=optimized')
        ]);
        setBaselineData(baseRes.data);
        setOptimizedData(optRes.data);
        setLoading(false);
      } catch (err) {
        console.error("API Error:", err);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // Determine active dataset
  const activeData = viewMode === 'baseline' ? baselineData : optimizedData;

  // Calculate Map Center (Average of all points to auto-center)
  const mapCenter = useMemo(() => {
    if (baselineData.length === 0) return [12.9716, 77.5946]; // Default: Bangalore
    const latSum = baselineData.reduce((sum, node) => sum + node.lat, 0);
    const lonSum = baselineData.reduce((sum, node) => sum + node.lon, 0);
    return [latSum / baselineData.length, lonSum / baselineData.length];
  }, [baselineData]);

  const getColor = (score: number) => {
    if (score >= 80) return '#00E676'; // Bright Green
    if (score >= 60) return '#C6FF00'; // Lime
    if (score >= 40) return '#FFEA00'; // Yellow
    if (score >= 20) return '#FF9100'; // Orange
    return '#FF3D00';                  // Red
  };

  if (loading) return <div className="loading-screen"><h2>Loading PathLens AI Model...</h2></div>;

  return (
    <div style={{ position: 'relative', height: '100vh', width: '100vw' }}>
      
      {/* --- CUSTOM CONTROL PANEL --- */}
      <div className="map-controls">
        <h3 style={{ margin: '0 0 10px 0', color: '#fff' }}>PathLens Control</h3>
        
        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            className={`toggle-btn ${viewMode === 'baseline' ? 'active' : ''}`}
            onClick={() => setViewMode('baseline')}
          >
            Baseline
          </button>
          <button 
            className={`toggle-btn ${viewMode === 'optimized' ? 'active' : ''}`}
            onClick={() => setViewMode('optimized')}
          >
            Optimized (AI)
          </button>
        </div>

        <div style={{ marginTop: '15px', fontSize: '0.9em', color: '#ccc' }}>
          <strong>Avg Score:</strong>{' '}
          {(activeData.reduce((s, n) => s + n.walkability, 0) / (activeData.length || 1)).toFixed(1)} / 100
        </div>
      </div>

      {/* --- MAP --- */}
      <MapContainer 
        center={mapCenter as [number, number]} 
        zoom={13} 
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CartoDB</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />
        
        <DataMarkers data={activeData} getColor={getColor} />
        
        {/* Auto-recenter map when data loads */}
        <RecenterMap center={mapCenter as [number, number]} />
      </MapContainer>
    </div>
  );
}

// Sub-component to handle map fly-to animations
const RecenterMap = ({ center }: { center: [number, number] }) => {
  const map = useMap();
  useEffect(() => {
    map.setView(center);
  }, [center, map]);
  return null;
};

// Sub-component for rendering markers efficiently
const DataMarkers = ({ data, getColor }: { data: NodeData[], getColor: (s: number) => string }) => {
  return (
    <>
      {data.map((node) => (
        <CircleMarker
          key={node.osmid}
          center={[node.lat, node.lon]} // Ensure this is [Lat, Lon]
          radius={6}
          pathOptions={{
            color: getColor(node.walkability),
            fillColor: getColor(node.walkability),
            fillOpacity: 0.8,
            weight: 1,
            stroke: true,
          }}
        >
          <Popup className="custom-popup">
            <div style={{ minWidth: '200px' }}>
              <h4 style={{ margin: '0 0 8px 0', borderBottom: '1px solid #ccc', paddingBottom: '4px' }}>
                Node Analysis
              </h4>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span>Walkability:</span>
                <strong style={{ color: getColor(node.walkability) }}>
                  {node.walkability.toFixed(1)}
                </strong>
              </div>
              
              <div style={{ background: '#f5f5f5', padding: '8px', borderRadius: '4px', fontSize: '0.9em', color: '#333' }}>
                <div style={{marginBottom: '4px'}}>🏫 School: <b>{node.dist_to_school.toFixed(0)}m</b></div>
                <div style={{marginBottom: '4px'}}>🏥 Hospital: <b>{node.dist_to_hospital.toFixed(0)}m</b></div>
                <div>🌳 Park: <b>{node.dist_to_park.toFixed(0)}m</b></div>
              </div>
              <small style={{ color: '#666', marginTop: '5px', display: 'block' }}>
                ID: {node.osmid}
              </small>
            </div>
          </Popup>
        </CircleMarker>
      ))}
    </>
  );
};