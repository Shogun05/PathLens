'use client';

import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default marker icons in Next.js
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

interface MapNode {
  osmid: string;
  x: number; // longitude
  y: number; // latitude
  score?: number;
  type?: string;
}

interface MapSuggestion {
  geometry: {
    coordinates: [number, number];
  };
  properties: {
    id: string;
    amenity_type: string;
  };
}

interface MapComponentProps {
  nodes?: MapNode[];
  suggestions?: MapSuggestion[];
  center?: [number, number];
  zoom?: number;
  className?: string;
  onMapReady?: (map: L.Map) => void;
  drawingMode?: boolean;
  onBoundsDrawn?: (bounds: L.LatLngBounds) => void;
}

export default function MapComponent({
  nodes = [],
  suggestions = [],
  center = [12.9716, 77.5946], // Bangalore default
  zoom = 12,
  className,
  onMapReady,
  drawingMode = false,
  onBoundsDrawn,
}: MapComponentProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const rectangleRef = useRef<L.Rectangle | null>(null);
  const markersRef = useRef<Map<string, L.CircleMarker>>(new Map());
  const [mapInstance, setMapInstance] = useState<L.Map | null>(null);

  useEffect(() => {
    if (!mapContainerRef.current || mapInstance) return;

    // Initialize map
    const map = L.map(mapContainerRef.current, {
      center,
      zoom,
      zoomControl: true,
    });

    // Add OpenStreetMap tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
    }).addTo(map);

    setMapInstance(map);
    
    if (onMapReady) {
      onMapReady(map);
    }

    return () => {
      map.remove();
      setMapInstance(null);
    };
  }, []);

  // Update markers when nodes or suggestions change
  useEffect(() => {
    if (!mapInstance) return;

    const map = mapInstance;
    const currentMarkers = markersRef.current;
    const newMarkerIds = new Set<string>();

    // Process Nodes
    console.log(`[MapComponent] Processing ${nodes.length} nodes. Map ready: ${!!mapInstance}`);
    
    nodes.forEach((node) => {
      if (!node.y || !node.x) {
        // console.warn('Invalid node coordinates:', node);
        return;
      }
      const id = `node-${node.osmid}`;
      newMarkerIds.add(id);

      const color = getNodeColor(node.score);
      const popupContent = `
        <div class="text-xs">
          <strong>Node ID:</strong> ${node.osmid}<br/>
          ${node.score !== undefined ? `<strong>Score:</strong> ${node.score.toFixed(1)}/100<br/>` : ''}
          ${node.type ? `<strong>Type:</strong> ${node.type}<br/>` : ''}
          <strong>Lat:</strong> ${node.y.toFixed(6)}<br/>
          <strong>Lng:</strong> ${node.x.toFixed(6)}
        </div>
      `;

      if (currentMarkers.has(id)) {
        // Update existing marker
        const marker = currentMarkers.get(id)!;
        
        // Cancel any pending removal
        if ((marker as any)._removeTimeout) {
          clearTimeout((marker as any)._removeTimeout);
          (marker as any)._removeTimeout = null;
        }

        // Update style and content
        marker.setStyle({ 
          fillColor: color,
          opacity: 0.8,
          fillOpacity: 0.6
        });
        marker.setPopupContent(popupContent);
      } else {
        // Add new marker (start visible)
        const marker = L.circleMarker([node.y, node.x], {
          radius: 5,
          fillColor: color,
          color: '#fff',
          weight: 1,
          opacity: 0.8,
          fillOpacity: 0.6,
          className: 'leaflet-marker-transition'
        });

        marker.bindPopup(popupContent);
        marker.addTo(map);
        currentMarkers.set(id, marker);
        
        // No delayed fade-in for now to ensure visibility
        // setTimeout(() => {
        //   marker.setStyle({ opacity: 0.8, fillOpacity: 0.6 });
        // }, 50);
      }
    });

    // Process Suggestions
    suggestions.forEach((suggestion) => {
      const [lng, lat] = suggestion.geometry.coordinates;
      if (!lat || !lng) return;
      const id = `sugg-${suggestion.properties.id}`;
      newMarkerIds.add(id);

      const popupContent = `
        <div class="text-xs">
          <strong>New Amenity</strong><br/>
          <strong>Type:</strong> ${suggestion.properties.amenity_type}<br/>
          <strong>ID:</strong> ${suggestion.properties.id}
        </div>
      `;

      if (currentMarkers.has(id)) {
        // Update existing suggestion
        const marker = currentMarkers.get(id)!;
        
        if ((marker as any)._removeTimeout) {
          clearTimeout((marker as any)._removeTimeout);
          (marker as any)._removeTimeout = null;
        }

        marker.setStyle({ 
          opacity: 1,
          fillOpacity: 0.8
        });
        marker.setPopupContent(popupContent);
      } else {
        // Add new suggestion
        const marker = L.circleMarker([lat, lng], {
          radius: 8,
          fillColor: '#8fd6ff', // Cyan for suggestions
          color: '#fff',
          weight: 2,
          opacity: 1,
          fillOpacity: 0.8,
          className: 'leaflet-marker-transition'
        });

        marker.bindPopup(popupContent);
        marker.addTo(map);
        currentMarkers.set(id, marker);

        // No delayed fade-in for now
        // setTimeout(() => {
        //   marker.setStyle({ opacity: 1, fillOpacity: 0.8 });
        // }, 50);
      }
    });

    // Remove markers that are no longer present
    currentMarkers.forEach((marker, id) => {
      if (!newMarkerIds.has(id)) {
        // If already scheduled for removal, skip
        if ((marker as any)._removeTimeout) return;

        // Fade out
        marker.setStyle({ opacity: 0, fillOpacity: 0 });
        
        // Remove after transition
        (marker as any)._removeTimeout = setTimeout(() => {
          map.removeLayer(marker);
          currentMarkers.delete(id);
        }, 500); // Match CSS transition duration
      }
    });

    // Initial bounds fit
    if (currentMarkers.size === 0 && newMarkerIds.size > 0) {
       const points: L.LatLngTuple[] = [
        ...nodes.map(n => [n.y, n.x] as L.LatLngTuple),
        ...suggestions.map(s => [s.geometry.coordinates[1], s.geometry.coordinates[0]] as L.LatLngTuple)
      ];

      if (points.length > 0) {
        const bounds = L.latLngBounds(points);
        map.fitBounds(bounds, { padding: [50, 50] });
      }
    }

  }, [mapInstance, nodes, suggestions]);

  const startLatLngRef = useRef<L.LatLng | null>(null);

  const handleMouseDown = (e: React.MouseEvent) => {
    const map = mapInstance;
    if (!map) return;
    
    const point = L.point(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    const latlng = map.containerPointToLatLng(point);
    
    startLatLngRef.current = latlng;
    
    if (rectangleRef.current) {
      map.removeLayer(rectangleRef.current);
      rectangleRef.current = null;
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!startLatLngRef.current || !mapInstance) return;
    
    const map = mapInstance;
    const point = L.point(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    const latlng = map.containerPointToLatLng(point);
    
    if (rectangleRef.current) {
      map.removeLayer(rectangleRef.current);
    }
    
    const bounds = L.latLngBounds(startLatLngRef.current, latlng);
    rectangleRef.current = L.rectangle(bounds, {
      color: '#23CE6B',
      weight: 2,
      fillColor: '#23CE6B',
      fillOpacity: 0.2,
      interactive: false,
    }).addTo(map);
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    if (!startLatLngRef.current || !mapInstance) return;
    
    const map = mapInstance;
    const point = L.point(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    const latlng = map.containerPointToLatLng(point);
    
    if (rectangleRef.current) {
      map.removeLayer(rectangleRef.current);
    }
    
    const bounds = L.latLngBounds(startLatLngRef.current, latlng);
    rectangleRef.current = L.rectangle(bounds, {
      color: '#23CE6B',
      weight: 2,
      fillColor: '#23CE6B',
      fillOpacity: 0.2,
      interactive: false,
    }).addTo(map);

    if (onBoundsDrawn) {
      onBoundsDrawn(bounds);
    }

    startLatLngRef.current = null;
  };

  return (
    <div className="relative w-full h-full">
      <style jsx global>{`
        .leaflet-marker-transition {
          transition: stroke-opacity 0.5s ease-in-out, fill-opacity 0.5s ease-in-out, fill 0.5s ease-in-out;
        }
      `}</style>
      <div ref={mapContainerRef} className={className} />
      {drawingMode && (
        <div 
          className="absolute inset-0 z-[1000] cursor-crosshair"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        />
      )}
    </div>
  );
}

function getNodeColor(score?: number): string {
  if (score === undefined) return '#8fd6ff';
  
  // 10-level color gradient from red to green
  const colorScale = [
    '#ff0000', // 0-10: Critical - Red
    '#ff4d00', // 10-20: Severe - Red-Orange
    '#ff9900', // 20-30: Poor - Orange
    '#ffcc00', // 30-40: Below Average - Yellow-Orange
    '#ffff00', // 40-50: Average - Yellow
    '#ccff00', // 50-60: Above Average - Yellow-Green
    '#99ff00', // 60-70: Good - Light Green
    '#66ff00', // 70-80: Very Good - Green
    '#33ff00', // 80-90: Excellent - Bright Green
    '#00ff00', // 90-100: Exceptional - Pure Green
  ];
  
  const index = Math.min(9, Math.floor(score / 10));
  return colorScale[index];
}
