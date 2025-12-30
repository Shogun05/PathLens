'use client';

import { useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';
import dynamic from 'next/dynamic';
import { usePathLensStore } from '@/lib/store';
import { Loader2 } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import AnalysisHeader from '@/components/AnalysisHeader';
import { MapControls } from '@/components/MapControls';

const MapComponent = dynamic(() => import('@/components/MapComponent'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-neutral-900">
      <Loader2 className="h-8 w-8 animate-spin text-[#8fd6ff]" />
    </div>
  ),
});

export default function AnalysisLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const { 
    baselineNodes, 
    optimizedNodes, 
    suggestions,
  } = usePathLensStore();
  
  const [isMounted, setIsMounted] = useState(false);
  const [mapInstance, setMapInstance] = useState<any>(null);
  const [enabledLayers, setEnabledLayers] = useState(new Set(['nodeScores']));

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Determine which data to show based on route
  const isOptimized = pathname?.includes('/optimized');
  const displayNodes = isOptimized ? optimizedNodes : baselineNodes;
  const displaySuggestions = isOptimized ? suggestions : [];

  const handleLayerToggle = (layer: string, enabled: boolean) => {
    setEnabledLayers(prev => {
      const next = new Set(prev);
      if (enabled) {
        next.add(layer);
      } else {
        next.delete(layer);
      }
      return next;
    });
  };

  if (!isMounted) return null;

  return (
    <div className="flex h-screen flex-col bg-[#0f1c23]">
      <AnalysisHeader />
      
      <div className="flex-1 relative overflow-hidden">
        {/* Persistent Map Layer */}
        <div className="absolute inset-0 z-0">
          <MapComponent 
            nodes={displayNodes}
            suggestions={displaySuggestions}
            center={[12.9716, 77.5946]}
            zoom={13}
            className="w-full h-full"
            onMapReady={(map) => setMapInstance(map)}
          />
        </div>

        {/* Map Controls Overlay */}
        <MapControls 
          className="absolute inset-0 z-5"
          onZoomIn={() => mapInstance?.zoomIn()}
          onZoomOut={() => mapInstance?.zoomOut()}
          onLayerToggle={handleLayerToggle}
          enabledLayers={enabledLayers}
        />

        {/* Page Content Overlay */}
        <div className="absolute inset-0 z-10 pointer-events-none">
          {/* We use a div here to ensure children (the pages) take full height */}
          {/* pointer-events-none allows clicking through to map where there is no UI */}
          {/* The pages themselves must re-enable pointer-events on their sidebars */}
          <AnimatePresence mode="wait">
            <motion.div key={pathname} className="h-full w-full">
              {children}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
