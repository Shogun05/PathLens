'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Layers, MapPin, Search, Plus, Minus } from 'lucide-react';

interface MapControlsProps {
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  onLayerToggle?: (layer: string, enabled: boolean) => void;
  enabledLayers?: Set<string>;
  className?: string;
  showBaselineToggle?: boolean;
  showBaselineAmenities?: boolean;
  onBaselineToggle?: (enabled: boolean) => void;
  showNodesToggle?: boolean;
  showNodes?: boolean;
  onNodesToggle?: (enabled: boolean) => void;
}

export function MapControls({ 
  onZoomIn, 
  onZoomOut, 
  onLayerToggle,
  enabledLayers = new Set(['nodeScores']),
  className = '',
  showBaselineToggle = false,
  showBaselineAmenities = false,
  onBaselineToggle,
  showNodesToggle = false,
  showNodes = true,
  onNodesToggle,
}: MapControlsProps) {
  const [showLayerMenu, setShowLayerMenu] = useState(false);

  const layers = [
    { id: 'nodeScores', label: 'Node Scores', icon: '📍', color: 'text-red-500' },
    { id: 'heatmap', label: 'Heatmap', icon: '🔥', color: 'text-orange-500' },
    { id: 'pois', label: 'POIs', icon: '📌', color: 'text-blue-500' },
    { id: 'suggestions', label: 'Suggestions', icon: '💡', color: 'text-green-500' },
  ];

  const handleLayerToggle = (layerId: string) => {
    if (onLayerToggle) {
      onLayerToggle(layerId, !enabledLayers.has(layerId));
    }
  };

  return (
    <div className={`pointer-events-none ${className}`}>
      <div className="absolute inset-0">
        {/* Top Left: Search Bar */}
        <div className="absolute top-4 left-4 pointer-events-auto">
          <Card className="w-64 bg-[#1b2328]/90 backdrop-blur-xl border-white/10 shadow-lg">
            <div className="flex items-center px-3 h-10">
              <Search className="h-4 w-4 text-gray-400" />
              <input
                className="bg-transparent border-none text-white text-sm focus:ring-0 w-full placeholder:text-gray-400 ml-2"
                placeholder="Search nodes..."
              />
            </div>
          </Card>
        </div>

        {/* Top Right: Layer Controls & Zoom */}
        <div className="absolute top-4 right-4 flex flex-col gap-2 items-end pointer-events-auto">
          {/* Zoom Controls */}
          <Card className="bg-[#1b2328]/90 backdrop-blur-xl border-white/10 p-1 flex flex-col gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-white hover:bg-white/10"
              onClick={onZoomIn}
              title="Zoom In"
            >
              <Plus className="h-5 w-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-white hover:bg-white/10"
              onClick={onZoomOut}
              title="Zoom Out"
            >
              <Minus className="h-5 w-5" />
            </Button>
          </Card>
        </div>

        {/* Bottom Left: Layer Chips */}
        <div className="absolute bottom-4 left-4 right-4 flex justify-between items-end pointer-events-none">
          <div className="pointer-events-auto flex flex-wrap gap-2 max-w-[70%]" style={{ marginLeft: showNodesToggle ? '420px' : '0' }}>
            {/* All Layers Toggle */}
            <button
              className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#1b2328]/90 border border-white/10 backdrop-blur-sm cursor-pointer hover:bg-[#1b2328] transition-colors"
              onClick={() => setShowLayerMenu(!showLayerMenu)}
            >
              <Layers className="h-4 w-4 text-[#8fd6ff]" />
              <span className="text-xs font-medium text-white">Layers</span>
            </button>

            {/* Baseline Amenities Toggle (only on /optimized) */}
            {showBaselineToggle && onBaselineToggle && (
              <button
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full border backdrop-blur-sm cursor-pointer transition-colors ${
                  showBaselineAmenities
                    ? 'bg-[#8fd6ff]/20 border-[#8fd6ff]/40 hover:bg-[#8fd6ff]/30'
                    : 'bg-[#1b2328]/90 border-white/10 hover:bg-[#1b2328]'
                }`}
                onClick={() => onBaselineToggle(!showBaselineAmenities)}
              >
                <MapPin className={`h-4 w-4 ${
                  showBaselineAmenities ? 'text-[#8fd6ff]' : 'text-gray-400'
                }`} />
                <span className={`text-xs font-medium ${
                  showBaselineAmenities ? 'text-[#8fd6ff]' : 'text-gray-400'
                }`}>Existing Amenities</span>
              </button>
            )}

            {/* Nodes Toggle (only on /optimized) */}
            {showNodesToggle && onNodesToggle && (
              <button
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full border backdrop-blur-sm cursor-pointer transition-colors ${
                  showNodes
                    ? 'bg-[#8fd6ff]/20 border-[#8fd6ff]/40 hover:bg-[#8fd6ff]/30'
                    : 'bg-[#1b2328]/90 border-white/10 hover:bg-[#1b2328]'
                }`}
                onClick={() => onNodesToggle(!showNodes)}
              >
                <Layers className={`h-4 w-4 ${
                  showNodes ? 'text-[#8fd6ff]' : 'text-gray-400'
                }`} />
                <span className={`text-xs font-medium ${
                  showNodes ? 'text-[#8fd6ff]' : 'text-gray-400'
                }`}>Network Nodes</span>
              </button>
            )}

            {/* Active Layer Chips */}
            {layers
              .filter(layer => enabledLayers.has(layer.id))
              .map(layer => (
                <button
                  key={layer.id}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#8fd6ff]/20 border border-[#8fd6ff]/40 backdrop-blur-sm cursor-pointer hover:bg-[#8fd6ff]/30 transition-colors"
                  onClick={() => handleLayerToggle(layer.id)}
                >
                  <span className={`size-2 rounded-full ${layer.id === 'nodeScores' ? 'bg-red-500 animate-pulse' : 'bg-[#8fd6ff]'}`}></span>
                  <span className="text-xs font-medium text-[#8fd6ff]">{layer.label}</span>
                </button>
              ))}
          </div>

          {/* Legend */}
          <div className="pointer-events-auto bg-[#1b2328]/90 backdrop-blur-xl px-3 py-2 rounded-lg border border-white/10">
            <div className="text-[10px] text-gray-400 mb-1 uppercase tracking-wider font-semibold">
              Node Efficiency
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-white font-mono">0</span>
              <div className="w-24 h-2 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"></div>
              <span className="text-xs text-white font-mono">100</span>
            </div>
          </div>
        </div>

        {/* Layer Menu Popup */}
        {showLayerMenu && (
          <div className="absolute bottom-20 left-4 pointer-events-auto">
            <Card className="bg-[#1b2328]/95 backdrop-blur-xl border-white/10 p-4 w-64">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-bold text-white">Map Layers</h3>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 text-xs text-gray-400 hover:text-white"
                  onClick={() => setShowLayerMenu(false)}
                >
                  Close
                </Button>
              </div>
              <div className="space-y-2">
                {layers.map(layer => (
                  <button
                    key={layer.id}
                    className="w-full flex items-center justify-between p-2 rounded hover:bg-white/5 transition-colors"
                    onClick={() => handleLayerToggle(layer.id)}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{layer.icon}</span>
                      <span className="text-sm text-white">{layer.label}</span>
                    </div>
                    <div className={`w-10 h-5 rounded-full transition-colors ${
                      enabledLayers.has(layer.id) 
                        ? 'bg-[#8fd6ff]' 
                        : 'bg-gray-600'
                    }`}>
                      <div className={`w-4 h-4 rounded-full bg-white transition-transform mt-0.5 ${
                        enabledLayers.has(layer.id) 
                          ? 'translate-x-5' 
                          : 'translate-x-0.5'
                      }`}></div>
                    </div>
                  </button>
                ))}
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
