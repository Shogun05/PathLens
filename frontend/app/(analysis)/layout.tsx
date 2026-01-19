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

// City center coordinates
const CITY_CENTERS: Record<string, [number, number]> = {
  bangalore: [12.9716, 77.5946],
  chandigarh: [30.7333, 76.7794],
  mumbai: [19.076, 72.8777],
  navi_mumbai: [19.033, 73.0297],
  greater_mumbai: [19.076, 72.8777],
  mumbai_city: [18.9387, 72.8353],
};

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
    selectedSuggestionIds,
    selectedCity,
  } = usePathLensStore();

  const [isMounted, setIsMounted] = useState(false);
  const [mapInstance, setMapInstance] = useState<any>(null);
  const [showNodes, setShowNodes] = useState(true);
  const [mapCenter, setMapCenter] = useState<[number, number]>([12.9716, 77.5946]);

  useEffect(() => {
    setIsMounted(true);
    setMapCenter(CITY_CENTERS[selectedCity] || CITY_CENTERS.bangalore);
  }, [selectedCity]);

  // Determine which data to show based on route
  const isOptimized = pathname?.includes('/optimized');
  // Force showNodes to true for optimized view (nodes always visible)
  const displayNodes = isOptimized
    ? optimizedNodes
    : baselineNodes;
  
  // Filter suggestions to only show those that are "selected" (visible)
  // Strictly only show on optimized page
  const displaySuggestions = (isOptimized && !pathname?.includes('/baseline'))
    ? suggestions.filter(s => {
        const id = s.properties.id || s.properties.osmid;
        return id && selectedSuggestionIds.has(id);
      })
    : [];

  // handleLayerToggle removed - layer control functionality removed from UI

  if (!isMounted) return null;

  return (
    <div className="flex h-screen flex-col bg-[#0f1c23]">
      <AnalysisHeader />

      <div className="flex-1 relative overflow-hidden">
        {/* Persistent Map Layer - Hidden on /comparison since it has its own map */}
        {!pathname?.includes('/comparison') && (
          <div className="absolute inset-0 z-0">
            <MapComponent
              nodes={displayNodes}
              suggestions={displaySuggestions}
              selectedSuggestionIds={selectedSuggestionIds}
              onSuggestionClick={(id) => {
                const toggleSuggestion = usePathLensStore.getState().toggleSuggestion;
                toggleSuggestion(id);
              }}
              center={mapCenter}
              zoom={13}
              className="w-full h-full"
              onMapReady={(map) => setMapInstance(map)}
              onContainerReady={(container) => {
                usePathLensStore.getState().setMapContainer(container);
              }}
            />
          </div>
        )}

        {/* Map Controls Overlay */}
        {/* Map Controls Overlay */}
        <MapControls
          className="absolute inset-0 z-5"
          onZoomIn={() => mapInstance?.zoomIn()}
          onZoomOut={() => mapInstance?.zoomOut()}
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
