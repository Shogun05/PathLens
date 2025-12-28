import { useState, useMemo } from 'react';
import axios from 'axios';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './App.css';
import L from 'leaflet';

// Icon Fix
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({ iconRetinaUrl: markerIcon2x, iconUrl: markerIcon, shadowUrl: markerShadow });

export default function App() {
  const [location, setLocation] = useState("");
  const [budget, setBudget] = useState(50000000);
  const [maxAmenities, setMaxAmenities] = useState(10);
  
  const [loading, setLoading] = useState(false);
  const [mapNodes, setMapNodes] = useState<any[]>([]); 
  const [suggestions, setSuggestions] = useState<any[]>([]);
  
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [currentScore, setCurrentScore] = useState(0);
  const [baselineScore, setBaselineScore] = useState(0);

  // 10-LEVEL COLOR SCALE
  const colorScale = [
    '#ff0000', '#ff4d00', '#ff9900', '#ffcc00', '#ffff00', 
    '#ccff00', '#99ff00', '#66ff00', '#33ff00', '#00ff00'
  ];
  
  const getColor = (s: number) => {
    const idx = Math.floor(Math.max(0, Math.min(99, s)) / 10);
    return colorScale[idx];
  };

  const handleRun = async () => {
    if (!location) return;
    setLoading(true);
    try {
      await axios.post('http://localhost:8000/api/optimize', {
        location, budget, max_amenities: maxAmenities,
        add_schools: true, add_hospitals: true, add_parks: true 
      });

      const baseRes = await axios.get('http://localhost:8000/api/nodes?type=baseline');
      const sugRes = await axios.get('http://localhost:8000/api/suggestions');
      const optRes = await axios.get('http://localhost:8000/api/nodes?type=optimized');

      setMapNodes(optRes.data);
      setSuggestions(sugRes.data);
      
      const baseAvg = baseRes.data.reduce((s:any, n:any) => s + n.walkability, 0) / (baseRes.data.length || 1);
      const optAvg = optRes.data.reduce((s:any, n:any) => s + n.walkability, 0) / (optRes.data.length || 1);
      
      setBaselineScore(baseAvg);
      setCurrentScore(optAvg);
      
      const allIds = new Set(sugRes.data.map((s:any) => s.properties.id));
      setSelectedIds(allIds);

    } catch (e) {
      alert("Error: " + e);
    } finally {
      setLoading(false);
    }
  };

  const updateMapState = async (ids: string[]) => {
    try {
      // 1. Get New Scores from Backend
      const res = await axios.post('http://localhost:8000/api/rescore', {
        location, selected_ids: ids
      });
      
      setCurrentScore(res.data.avg_score);
      const updates = res.data.node_updates;
      
      // 2. Update Local State (Efficiently)
      setMapNodes(prev => prev.map(node => {
        // MATCHING FIX: Check both String and Number formats to be safe
        const idKey = node.osmid !== undefined ? String(node.osmid) : String(node.id);
        const newScore = updates[idKey];
        
        if (newScore !== undefined) {
          return { ...node, walkability: newScore };
        }
        return node;
      }));
      
    } catch (e) { console.error(e); }
  };

  const toggleSuggestion = (id: string) => {
    const newSet = new Set(selectedIds);
    if (newSet.has(id)) newSet.delete(id);
    else newSet.add(id);
    setSelectedIds(newSet);
    updateMapState(Array.from(newSet));
  };

  // Performance Sampling for Large Cities
  const displayNodes = useMemo(() => {
    if (mapNodes.length <= 5000) return mapNodes;
    const step = Math.ceil(mapNodes.length / 5000);
    return mapNodes.filter((_, i) => i % step === 0);
  }, [mapNodes]);

  const center = useMemo(() => {
    if (mapNodes.length) return [mapNodes[0].lat, mapNodes[0].lon];
    return [12.9716, 77.5946];
  }, [mapNodes]);

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="header">
          <h1>PathLens AI</h1>
          <p style={{color:'#94a3b8', margin:0}}>City-Scale Optimization</p>
        </div>
        <div className="controls">
          <div className="input-group">
            <label>City / Neighborhood</label>
            <input className="fancy-input" value={location} onChange={e=>setLocation(e.target.value)} placeholder="e.g. Indiranagar, Bengaluru"/>
          </div>
          <div className="input-group" style={{display:'flex', gap:10}}>
             <div style={{flex:1}}> <label>Budget</label><input type="number" className="fancy-input" value={budget} onChange={e=>setBudget(Number(e.target.value))}/> </div>
             <div style={{width:80}}> <label>Limit</label><input type="number" className="fancy-input" value={maxAmenities} onChange={e=>setMaxAmenities(Number(e.target.value))}/> </div>
          </div>
          <button className="btn-primary" onClick={handleRun} disabled={loading}>{loading ? "Running AI Models..." : "Run Optimization"}</button>
          
          {suggestions.length > 0 && (
             <div style={{marginTop:20}}>
               <h3 style={{fontSize:'1rem'}}>AI Suggestions ({suggestions.length})</h3>
               <div className="suggestion-list">
                 {suggestions.map((s:any) => (
                   <div key={s.properties.id} className="suggestion-card">
                     <input type="checkbox" checked={selectedIds.has(s.properties.id)} onChange={()=>toggleSuggestion(s.properties.id)}/>
                     <div className="card-info">
                       <h4 style={{color: s.properties.amenity==='school'?'#60a5fa':s.properties.amenity==='park'?'#34d399':'#f87171'}}>+ {s.properties.amenity}</h4>
                       <p>{s.properties.address}</p>
                     </div>
                   </div>
                 ))}
               </div>
             </div>
          )}
        </div>
        <div className="score-board">
          <div className="score-row"><span>Baseline Score</span><span style={{color:getColor(baselineScore)}}>{baselineScore.toFixed(1)}</span></div>
          <div className="score-row"><span>Projected Score</span><span className="big-score" style={{color:getColor(currentScore)}}>{currentScore.toFixed(1)}</span></div>
        </div>
      </div>

      <div className="map-area">
        <MapContainer center={center as [number, number]} zoom={13} scrollWheelZoom={true} preferCanvas={true}>
          <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution="CartoDB"/>
          
          {displayNodes.map((n) => (
            <CircleMarker 
              // ID + SCORE KEY: Forces React to re-render the color when score changes
              key={`${n.osmid}_${n.walkability}`} 
              center={[n.lat, n.lon]} 
              radius={5} 
              pathOptions={{ 
                color: getColor(n.walkability), 
                fillColor: getColor(n.walkability), 
                fillOpacity: 0.6, 
                weight: 0 
              }} 
            >
              <Popup>Score: {n.walkability.toFixed(1)}</Popup>
            </CircleMarker>
          ))}

          {suggestions.map((s:any) => {
            if (!selectedIds.has(s.properties.id)) return null;
            return (
              <CircleMarker key={s.properties.id} center={[s.geometry.coordinates[1], s.geometry.coordinates[0]]} radius={10} 
                pathOptions={{ color: '#fff', fillColor: 'transparent', weight: 2, dashArray: '4 4' }}>
                <Popup>New {s.properties.amenity}</Popup>
              </CircleMarker>
            )
          })}
        </MapContainer>
        
        <div style={{position:'absolute', bottom:20, right:20, background:'rgba(0,0,0,0.8)', padding:10, borderRadius:8, zIndex:999}}>
          <div style={{color:'white', marginBottom:5, fontSize:12}}>Walkability Scale</div>
          <div style={{display:'flex', height:10, width:150, background:'linear-gradient(to right, #ff0000, #ffff00, #00ff00)'}}></div>
        </div>
      </div>
    </div>
  );
}