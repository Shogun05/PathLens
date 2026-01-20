'use client';

import { usePathLensStore } from '@/lib/store';
import { CircleDot, MapPin } from 'lucide-react';

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface AnalysisHeaderProps {
  showNodes?: boolean;
  showPois?: boolean;
  onNodesToggle?: (enabled: boolean) => void;
  onPoisToggle?: (enabled: boolean) => void;
}

export default function AnalysisHeader({
  showNodes = true,
  showPois = true,
  onNodesToggle,
  onPoisToggle,
}: AnalysisHeaderProps) {
  const { selectedCity, setSelectedCity, reset } = usePathLensStore();

  const handleCityChange = (value: string) => {
    if (value !== selectedCity) {
      reset(); // Clear existing data when switching cities
      setSelectedCity(value);
      // Optional: Force reload to ensure clean state if needed, though react state should handle it
      window.location.reload();
    }
  };

  return (
    <header className="flex items-center justify-between border-b border-white/10 bg-[#101518] px-6 py-3 pointer-events-auto z-20 relative">
      <div className="flex items-center gap-4">
        <div className="size-6 text-[#8fd6ff]">
          <span className="material-symbols-outlined">api</span>
        </div>
        <h2 className="text-lg font-bold text-white">PathLens</h2>
      </div>
      <div className="flex items-center gap-6">
        {/* City Selection Dropdown */}
        <div className="w-[180px]">
          <Select value={selectedCity} onValueChange={handleCityChange}>
            <SelectTrigger className="w-full bg-white/5 border-white/10 text-white h-9">
              <SelectValue placeholder="Select City" />
            </SelectTrigger>
            <SelectContent className="bg-[#1b2328] border-white/10 text-white">
              <SelectItem value="bangalore">Bengaluru</SelectItem>
              <SelectItem value="chandigarh">Chandigarh</SelectItem>
              <SelectItem value="navi_mumbai">Navi Mumbai</SelectItem>
              <SelectItem value="kolkata">Kolkata</SelectItem>
              <SelectItem value="chennai">Chennai</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Layer Toggle Buttons - Horizontal */}
        {(onNodesToggle || onPoisToggle) && (
          <div className="flex items-center gap-2 bg-white/5 rounded-lg px-2 py-1 border border-white/10">
            <span className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold mr-1">Layers</span>
            {onNodesToggle && (
              <button
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                  showNodes
                    ? 'bg-[#8fd6ff]/20 text-[#8fd6ff]'
                    : 'text-gray-400 hover:bg-white/5'
                }`}
                onClick={() => onNodesToggle(!showNodes)}
                title="Toggle Nodes visibility"
              >
                <CircleDot className="h-3 w-3" />
                Nodes
              </button>
            )}
            {onPoisToggle && (
              <button
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                  showPois
                    ? 'bg-blue-500/20 text-blue-400'
                    : 'text-gray-400 hover:bg-white/5'
                }`}
                onClick={() => onPoisToggle(!showPois)}
                title="Toggle POIs visibility"
              >
                <MapPin className="h-3 w-3" />
                POIs
              </button>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
