import { useEffect, useState, useMemo } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, LayersControl, LayerGroup } from 'react-leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix Icons
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({ iconUrl: markerIcon, shadowUrl: markerShadow });

interface NodeData {
  osmid: string;
  lat: number;
  lon: number;
  walkability: number;
  dist_to_school: number;
  dist_to_hospital: number;
}

export default function MapViewer({ suggestions }: { suggestions: any[] }) {
  const [baselineData, setBaselineData] = useState<NodeData[]>([]);
  const [optimizedData, setOptimizedData] = useState<NodeData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [baseRes, optRes] = await Promise.all([
          axios.get('http://127.0.0.1:8000/api/nodes?type=baseline'),
          axios.get('http://127.0.0.1:8000/api/nodes?type=optimized')
        ]);
        setBaselineData(baseRes.data);
        setOptimizedData(optRes.data);
        setLoading(false);
      } catch (e) { console.error(e); setLoading(false); }
    };
    fetchData();
  }, []);

  const center = useMemo(() => {
    if (baselineData.length) return [baselineData[0].lat, baselineData[0].lon];
    return [12.9716, 77.5946];
  }, [baselineData]);

  const getColor = (s: number) => s > 80 ? '#00E676' : s > 50 ? '#FFEB3B' : '#F44336';

  if (loading) return <div>Loading Map Data...</div>;

  return (
    <MapContainer center={center as [number, number]} zoom={14} style={{ height: "100%", width: "100%" }} scrollWheelZoom={false}>
      <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution="CartoDB" />
      
      <LayersControl position="topright">
        
        <LayersControl.BaseLayer checked name="Baseline (Current)">
          <LayerGroup>
            {baselineData.map(n => (
              <CircleMarker key={n.osmid} center={[n.lat, n.lon]} radius={4} 
                pathOptions={{ color: getColor(n.walkability), fillColor: getColor(n.walkability), fillOpacity: 0.6, weight: 0 }}>
                <Popup>Baseline Score: {n.walkability.toFixed(1)}</Popup>
              </CircleMarker>
            ))}
          </LayerGroup>
        </LayersControl.BaseLayer>

        <LayersControl.BaseLayer name="Optimized (With Suggestions)">
          <LayerGroup>
            {optimizedData.map(n => (
              <CircleMarker key={n.osmid} center={[n.lat, n.lon]} radius={4} 
                pathOptions={{ color: getColor(n.walkability), fillColor: getColor(n.walkability), fillOpacity: 0.8, weight: 0 }}>
                <Popup>
                   <strong>Optimized Score: {n.walkability.toFixed(1)}</strong><br/>
                   School Dist: {n.dist_to_school.toFixed(0)}m
                </Popup>
              </CircleMarker>
            ))}
            
            {/* RENDER SUGGESTIONS AS HIGHLIGHTED MARKERS */}
            {suggestions.map((s, i) => (
              <CircleMarker key={`sug-${i}`} center={[s.lat, s.lon]} radius={10} 
                pathOptions={{ color: "white", fillColor: "blue", fillOpacity: 0.5, weight: 2, dashArray: "5, 5" }}>
                <Popup>✨ <strong>Suggestion: Build {s.type} here!</strong></Popup>
              </CircleMarker>
            ))}
          </LayerGroup>
        </LayersControl.BaseLayer>

      </LayersControl>
    </MapContainer>
  );
}