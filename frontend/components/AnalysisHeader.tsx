'use client';

import { usePathLensStore } from '@/lib/store';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';

export default function AnalysisHeader() {
  const { demoMode, setDemoMode } = usePathLensStore();

  return (
    <header className="flex items-center justify-between border-b border-white/10 bg-[#101518] px-6 py-3 pointer-events-auto z-20 relative">
      <div className="flex items-center gap-4">
        <div className="size-6 text-[#8fd6ff]">
          <span className="material-symbols-outlined">api</span>
        </div>
        <h2 className="text-lg font-bold text-white">PathLens</h2>
      </div>
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3 bg-white/5 px-3 py-1.5 rounded-full border border-white/10">
          <span className="text-sm font-medium text-gray-300">Demo Mode</span>
          <Switch 
            checked={demoMode}
            onCheckedChange={setDemoMode}
            className="data-[state=checked]:bg-[#8fd6ff]"
          />
        </div>

        <nav className="hidden md:flex items-center gap-6">
          <a className="text-white text-sm font-medium border-b-2 border-[#8fd6ff] pb-1">Dashboard</a>
          <a className="text-gray-400 hover:text-white text-sm font-medium transition-colors">Projects</a>
          <a className="text-gray-400 hover:text-white text-sm font-medium transition-colors">Analysis</a>
          <a className="text-gray-400 hover:text-white text-sm font-medium transition-colors">Settings</a>
        </nav>
        <div className="flex items-center gap-4 pl-6 border-l border-white/10">
          <button className="relative">
            <span className="material-symbols-outlined text-gray-400 hover:text-white">notifications</span>
            <span className="absolute top-0 right-0 size-2 bg-red-500 rounded-full"></span>
          </button>
          <div className="size-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-600"></div>
        </div>
      </div>
    </header>
  );
}
