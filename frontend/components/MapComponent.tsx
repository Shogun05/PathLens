'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.markercluster';
import 'leaflet.markercluster/dist/MarkerCluster.css';
import 'leaflet.markercluster/dist/MarkerCluster.Default.css';

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
  } | null;
  properties: {
    // Frontend naming
    id?: string;
    amenity_type?: string;
    description?: string;
    // Backend naming (from optimization POIs)
    osmid?: string;
    amenity?: string;
    travel_time_min?: number;
    optimization_mode?: string;
    mode_name?: string;
  };
}

interface MapComponentProps {
  nodes?: MapNode[];
  suggestions?: MapSuggestion[];
  selectedSuggestionIds?: Set<string>;
  onSuggestionClick?: (id: string) => void;
  center?: [number, number];
  zoom?: number;
  className?: string;
  onMapReady?: (map: L.Map) => void;
  onContainerReady?: (container: HTMLDivElement) => void;
  drawingMode?: boolean;
  onBoundsDrawn?: (bounds: L.LatLngBounds) => void;
  enableClustering?: boolean;
  maxMarkersBeforeClustering?: number;
}

export default function MapComponent({
  nodes = [],
  suggestions = [],
  selectedSuggestionIds = new Set(),
  onSuggestionClick,
  center = [12.9716, 77.5946], // Bangalore default
  zoom = 12,
  className,
  onMapReady,
  onContainerReady,
  drawingMode = false,
  onBoundsDrawn,
  enableClustering = true,
  maxMarkersBeforeClustering = 500,
}: MapComponentProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const rectangleRef = useRef<L.Rectangle | null>(null);
  const markersRef = useRef<Map<string, L.CircleMarker>>(new Map());
  const markerClusterGroupRef = useRef<L.MarkerClusterGroup | null>(null);
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

  // Update markers when nodes or suggestions change - with clustering support
  useEffect(() => {
    if (!mapInstance) return;

    const map = mapInstance;
    const shouldCluster = enableClustering && nodes.length > maxMarkersBeforeClustering;

    console.log(`[MapComponent] Processing ${nodes.length} nodes. Clustering: ${shouldCluster}`);

    // Clear previous markers/clusters
    const currentMarkers = markersRef.current;
    currentMarkers.forEach(marker => map.removeLayer(marker));
    currentMarkers.clear();

    if (markerClusterGroupRef.current) {
      markerClusterGroupRef.current.clearLayers();
      map.removeLayer(markerClusterGroupRef.current);
      markerClusterGroupRef.current = null;
    }

    // Initialize cluster group if needed
    if (shouldCluster) {
      markerClusterGroupRef.current = L.markerClusterGroup({
        maxClusterRadius: 50,
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: false,
        zoomToBoundsOnClick: true,
        iconCreateFunction: (cluster) => {
          const count = cluster.getChildCount();
          let size = 'small';
          if (count > 100) size = 'large';
          else if (count > 10) size = 'medium';

          return L.divIcon({
            html: `<div><span>${count}</span></div>`,
            className: `marker-cluster marker-cluster-${size}`,
            iconSize: L.point(40, 40)
          });
        },
        chunkedLoading: true, // Enable chunked loading for performance
        chunkInterval: 200, // Process for 200ms
        chunkDelay: 50, // Wait 50ms between chunks
      });
      map.addLayer(markerClusterGroupRef.current);
    }

    // Process Nodes
    const markers: L.CircleMarker[] = [];
    nodes.forEach((node) => {
      if (!node.y || !node.x) return;

      const id = `node-${node.osmid}`;
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

      const marker = L.circleMarker([node.y, node.x], {
        radius: shouldCluster ? 4 : 5,
        fillColor: color,
        color: '#fff',
        weight: 1,
        opacity: 0.8,
        fillOpacity: 0.6,
      });

      marker.bindPopup(popupContent);
      markers.push(marker);
      currentMarkers.set(id, marker);
    });

    // Add markers to cluster group or map
    if (shouldCluster && markerClusterGroupRef.current) {
      markerClusterGroupRef.current.addLayers(markers);
    } else {
      markers.forEach(marker => marker.addTo(map));
    }

    // Process Suggestions (always unclustered)
    console.log(`[MapComponent] Rendering ${suggestions.length} suggestions`);
    suggestions.forEach((suggestion) => {
      if (!suggestion.geometry || !suggestion.geometry.coordinates) return;
      const [lng, lat] = suggestion.geometry.coordinates;
      if (!lat || !lng) return;
      // Support both naming conventions: id/amenity_type (frontend) and osmid/amenity (backend)
      const poiId = suggestion.properties.id || suggestion.properties.osmid || 'unknown';
      const poiType = suggestion.properties.amenity_type || suggestion.properties.amenity || 'unknown';
      const id = `sugg-${poiId}`;
      const isSelected = selectedSuggestionIds.has(poiId);

      const tooltipContent = `
        <div class="text-xs">
          <strong>New Amenity</strong><br/>
          <strong>Type:</strong> ${poiType}<br/>
          <strong>ID:</strong> ${poiId}<br/>
          <strong>Status:</strong> ${isSelected ? 'Selected' : 'Not Selected'}
          ${suggestion.properties.description ? `<br/><em>${suggestion.properties.description}</em>` : ''}
        </div>
      `;

      const marker = L.circleMarker([lat, lng], {
        radius: isSelected ? 10 : 8,
        fillColor: isSelected ? '#10b981' : '#8fd6ff',
        color: '#fff',
        weight: isSelected ? 3 : 2,
        opacity: 1,
        fillOpacity: isSelected ? 0.9 : 0.8,
      });

      marker.bindTooltip(tooltipContent, {
        permanent: false,
        direction: 'top',
        offset: [0, -10],
        className: 'custom-tooltip'
      });

      if (onSuggestionClick) {
        marker.on('click', () => {
          onSuggestionClick(poiId);
        });
      }

      marker.addTo(map);
      currentMarkers.set(id, marker);
    });

    // Fit bounds on initial load
    if (nodes.length > 0 || suggestions.length > 0) {
      const points: L.LatLngTuple[] = [
        ...nodes.slice(0, 100).map(n => [n.y, n.x] as L.LatLngTuple), // Sample for performance
        ...suggestions.filter(s => s.geometry).map(s => [s.geometry!.coordinates[1], s.geometry!.coordinates[0]] as L.LatLngTuple)
      ];

      if (points.length > 0) {
        const bounds = L.latLngBounds(points);
        map.fitBounds(bounds, { padding: [50, 50], maxZoom: 14 });
      }
    }

  }, [mapInstance, nodes, suggestions, enableClustering, maxMarkersBeforeClustering, selectedSuggestionIds, onSuggestionClick]);

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

  // Ref for the entire wrapper (for screenshots)
  const wrapperRef = useRef<HTMLDivElement>(null);
  const onContainerReadyRef = useRef(onContainerReady);
  onContainerReadyRef.current = onContainerReady;

  // Expose wrapper for screenshot capture when map is ready (runs once when map initializes)
  useEffect(() => {
    if (mapInstance && wrapperRef.current && onContainerReadyRef.current) {
      onContainerReadyRef.current(wrapperRef.current);
    }
  }, [mapInstance]); // Only depend on mapInstance

  return (
    <div ref={wrapperRef} className="relative w-full h-full">
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
