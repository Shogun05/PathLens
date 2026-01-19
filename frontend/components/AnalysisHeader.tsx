'use client';

import { usePathLensStore } from '@/lib/store';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

export default function AnalysisHeader() {
  const { demoMode, setDemoMode, selectedCity, setSelectedCity, reset } = usePathLensStore();

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
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-3 bg-white/5 px-3 py-1.5 rounded-full border border-white/10">
          <span className="text-sm font-medium text-gray-300">Demo Mode</span>
          <Switch 
            checked={demoMode}
            onCheckedChange={setDemoMode}
            className="data-[state=checked]:bg-[#8fd6ff]"
          />
        </div>
      </div>
    </header>
  );
}
