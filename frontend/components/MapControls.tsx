'use client';

import { Layers, MapPin, Plus, Minus } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface MapControlsProps {
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  className?: string;
  showBaselineToggle?: boolean;
  showBaselineAmenities?: boolean;
  onBaselineToggle?: (enabled: boolean) => void;
}

export function MapControls({ 
  onZoomIn, 
  onZoomOut, 
  className = '',
  showBaselineToggle = false,
  showBaselineAmenities = false,
  onBaselineToggle,
}: MapControlsProps) {

  return (
    <div className={`pointer-events-none ${className}`}>
      <div className="absolute inset-0">
        {/* Search bar removed - was not functional */}

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

        {/* Bottom Left: Toggle Chips */}
        <div className="absolute bottom-4 left-4 right-4 flex justify-between items-end pointer-events-none">
          <div className="pointer-events-auto flex flex-wrap gap-2 max-w-[70%]" style={{ marginLeft: showBaselineToggle ? '420px' : '0' }}>
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

            {/* Nodes Toggle removed - nodes are now always visible */}
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

        {/* Layer menu removed - was not functional */}
      </div>
    </div>
  );
}
