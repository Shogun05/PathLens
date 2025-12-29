'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { usePathLensStore } from '@/lib/store';
import { pathLensAPI } from '@/lib/api';
import { School, Hospital, Bus, Trees, Check, Search, Square, Sparkles, CheckCircle, Loader2, Circle } from 'lucide-react';

// Dynamic import to avoid SSR issues with Leaflet
const MapComponent = dynamic(() => import('@/components/MapComponent'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-neutral-900">
      <Loader2 className="h-8 w-8 animate-spin text-[#8fd6ff]" />
    </div>
  ),
});

export default function HomePage() {
  const router = useRouter();
  const {
    location,
    budget,
    maxAmenities,
    addSchools,
    addHospitals,
    addParks,
    setLocation,
    setBudget,
    setMaxAmenities,
    setAddSchools,
    setAddHospitals,
    setAddParks,
    setIsOptimizing,
    setOptimizationProgress,
  } = usePathLensStore();

  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [drawingMode, setDrawingMode] = useState(false);
  const [customBounds, setCustomBounds] = useState<any>(null);

  const steps = [
    { label: 'Satellite Imagery', detail: 'Downloaded 2.4GB tiles' },
    { label: 'Road Network Graph', detail: 'Extracted 14,500 nodes' },
    { label: 'Genetic Algorithm', detail: 'Generation 45/100' },
    { label: 'Land Validation', detail: 'Verifying availability' },
  ];

  const handleOptimize = async () => {
    if (!location && !customBounds) {
      alert('Please enter a location or draw a custom bounding box');
      return;
    }

    setIsRunning(true);
    setIsOptimizing(true);
    setProgress(0);

    try {
      const progressSteps = [
        { step: 0, label: 'Downloading satellite imagery...', progress: 20 },
        { step: 1, label: 'Extracting road network...', progress: 40 },
        { step: 2, label: 'Running genetic algorithm...', progress: 70 },
        { step: 3, label: 'Validating land availability...', progress: 90 },
      ];

      for (const { label, progress: p } of progressSteps) {
        setCurrentStep(label);
        setProgress(p);
        setOptimizationProgress(label);
        await new Promise((resolve) => setTimeout(resolve, 1500));
      }

      await pathLensAPI.optimize({
        location,
        budget,
        max_amenities: maxAmenities,
        add_schools: addSchools,
        add_hospitals: addHospitals,
        add_parks: addParks,
      });

      setProgress(100);
      setCurrentStep('Complete!');
      
      setTimeout(() => {
        router.push('/baseline');
      }, 1000);
    } catch (error) {
      console.error('Optimization failed:', error);
      alert('Optimization failed. Please try again.');
      setIsRunning(false);
      setIsOptimizing(false);
    }
  };

  const handleDrawBoundingBox = () => {
    setDrawingMode(!drawingMode);
  };

  const handleBoundsDrawn = (bounds: any) => {
    setCustomBounds(bounds);
    setDrawingMode(false);
    const center = bounds.getCenter();
    const area = (
      (bounds.getNorth() - bounds.getSouth()) *
      (bounds.getEast() - bounds.getWest()) *
      111 * 111 * Math.cos((center.lat * Math.PI) / 180)
    ).toFixed(1);
    setLocation(`Custom Area (${area} km²)`);
  };

  return (
    <div className="relative flex h-screen w-full flex-col overflow-hidden bg-[#0f1c23]">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-white/10 bg-[#101518] px-6 py-3 z-30">
        <div className="flex items-center gap-4">
          <div className="size-8 flex items-center justify-center rounded bg-[#8fd6ff]/20 text-[#8fd6ff]">
            <span className="material-symbols-outlined">share_location</span>
          </div>
          <h2 className="text-lg font-bold">PathLens</h2>
        </div>
        <div className="flex gap-3 items-center">
          <Button variant="ghost" className="text-gray-400 hover:text-white">
            Documentation
          </Button>
          <div className="h-9 w-[1px] bg-white/10 mx-1"></div>
          <div className="flex items-center gap-3">
            <div className="size-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-600"></div>
            <span className="text-sm font-medium hidden sm:block">City Planner</span>
          </div>
        </div>
      </header>

      <div className="flex flex-1 relative overflow-hidden">
        {/* Left  disabled={drawingMode}
                />
              </div>
              <Button 
                variant="outline" 
                className={`w-full border-dashed text-gray-300 ${drawingMode ? 'bg-[#8fd6ff]/20 border-[#8fd6ff]' : ''}`}
                onClick={handleDrawBoundingBox}
                type="button"
              >
                <Square className="mr-2 h-4 w-4" />
                {drawingMode ? 'Click and drag on map...' : 'Draw Custom Bounding Box'}
              </Button>
              {customBounds && (
                <div className="text-xs text-[#8fd6ff] bg-[#8fd6ff]/10 px-3 py-2 rounded-lg">
                  ✓ Custom area selected
                </div>
              )}sName="flex items-center justify-between">
                <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-400">Target Area</h3>
                <Badge variant="outline" className="bg-[#8fd6ff]/10 text-[#8fd6ff] border-[#8fd6ff]/20">Step 1/3</Badge>
              </div>
              <div className="relative">
                <Search className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
                <Input
                  className="pl-10 bg-[#1b2328] border-0 text-white placeholder:text-gray-500"
                  placeholder="Search city (e.g. Bangalore, India)"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                />
              </div>
              <Button variant="outline" className="w-full border-dashed text-gray-300">
                <Square className="mr-2 h-4 w-4" />
                Draw Custom Bounding Box
              </Button>
            </section>

            <div className="border-t border-white/10" />

            {/* Budget */}
            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-400">Constraints</h3>
                <Badge variant="outline" className="bg-[#8fd6ff]/10 text-[#8fd6ff] border-[#8fd6ff]/20">Step 2/3</Badge>
              </div>
              <div className="bg-[#1b2328] p-4 rounded-xl border border-white/5">
                <div className="flex justify-between items-end mb-4">
                  <span className="text-sm font-medium text-gray-300">Total Budget</span>
                  <span className="text-xl font-bold text-[#8fd6ff] font-mono">
                    ₹{(budget / 10000000).toFixed(1)} Cr
                  </span>
                </div>
                <Slider
                  value={[budget / 1000000]}
                  onValueChange={(val) => setBudget(val[0] * 1000000)}
                  min={10}
                  max={500}
                  step={10}
                  className="mb-2"
                />
                <div className="flex justify-between text-xs text-gray-500 font-mono">
                  <span>₹1 Cr</span>
                  <span>₹50 Cr</span>
                </div>
              </div>
            </section>

            <div className="border-t border-white/10" />

            {/* Amenities */}
            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-400">Priority Amenities</h3>
                <Badge variant="outline" className="bg-[#8fd6ff]/10 text-[#8fd6ff] border-[#8fd6ff]/20">Step 3/3</Badge>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <AmenityCard icon={School} label="Schools" checked={addSchools} onCheckedChange={setAddSchools} priority={3} />
                <AmenityCard icon={Hospital} label="Hospitals" checked={addHospitals} onCheckedChange={setAddHospitals} priority={2} />
                <AmenityCard icon={Bus} label="Bus Stn" checked={false} onCheckedChange={() => {}} priority={1} />
                <AmenityCard icon={Trees} label="Parks" checked={addParks} onCheckedChange={setAddParks} priority={0} />
              </div>
            </section>

            <div className="border-t border-white/10" />

            {/* Advanced */}
            <Accordion type="single" collapsible className="rounded-lg border border-white/5 bg-[#1b2328]">
              <AccordionItem value="advanced" className="border-none">
                <AccordionTrigger className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <span className="material-symbols-outlined text-[#8fd6ff]">tune</span>
                    <span>Advanced Config</span>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="px-4 pb-4 space-y-4 border-t border-white/5">
                  <div>
                    <div className="flex justify-between mb-2 text-sm">
                      <span className="text-gray-400">Max New Amenities</span>
                      <span className="font-mono">{maxAmenities}</span>
                    </div>
                    <Slider value={[maxAmenities]} onValueChange={(val) => setMaxAmenities(val[0])} min={5} max={50} step={5} />
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>

          {/* CTA */}
          <div className="p-6 border-t border-white/10 bg-[#101518]/90">
            <Button
              onClick={handleOptimize}
              disabled={isRunning || (!location && !customBounds)}
              className="w-full h-12 bg-[#8fd6ff] hover:bg-[#b0e2ff] text-[#101518] font-bold shadow-[0_0_20px_rgba(143,214,255,0.3)] disabled:opacity-50 disabled:cursor-not-allowed"
              type="button"
            >
              {isRunning ? <><Loader2 className="mr-2 h-5 w-5 animate-spin" />Running...</> : <><Sparkles className="mr-2 h-5 w-5" />Run City Optimization</>}
            </Button>
          </div>
        </aside>

        {/* Map */}
        <main className="flex-1 relative bg-neutral-900">
          <MapComponent 
            nodes={[]}
            center={[12.9716, 77.5946]}
            zoom={12}
            className="w-full h-full"
            drawingMode={drawingMode}
            onBoundsDrawn={handleBoundsDrawn}
          />
          
          <div className={`absolute inset-0 bg-gradient-to-r from-[#0f1c23] via-[#0f1c23]/60 to-transparent pointer-events-none ${drawingMode ? 'opacity-50' : ''}`}></div>

          {/* Progress */}
          {isRunning && (
            <div className="absolute top-6 right-6 w-80 bg-[#1b2328]/95 backdrop-blur-md rounded-xl border border-white/10 z-20">
              <div className="px-4 py-3 border-b border-white/10 flex justify-between items-center">
                <h4 className="text-sm font-semibold">System Status</h4>
                <div className="h-2 w-2 rounded-full bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.5)]"></div>
              </div>
              <div className="p-4 space-y-4">
                {steps.map((step, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <div className="mt-0.5">
                      {progress >= (idx + 1) * 25 ? <CheckCircle className="h-5 w-5 text-green-400" /> :
                       progress > idx * 25 ? <Loader2 className="h-5 w-5 text-[#8fd6ff] animate-spin" /> :
                       <Circle className="h-5 w-5 text-gray-500" />}
                    </div>
                    <div className="flex-1">
                      <p className="text-xs font-medium">{step.label}</p>
                      {progress > idx * 25 && progress < (idx + 1) * 25 && (
                        <>
                          <Progress value={((progress - idx * 25) / 25) * 100} className="mt-2 h-1" />
                          <p className="text-[10px] text-[#8fd6ff] mt-1">{step.detail}</p>
                        </>
                      )}
                    </div>
                
        checked={checked} 
        onCheckedChange={onCheckedChange} 
        className="absolute top-3 right-3"
        id={`amenity-${label}`}
     
                ))}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

function AmenityCard({ icon: Icon, label, checked, onCheckedChange, priority }: any) {
  return (
    <label className="cursor-pointer relative flex flex-col gap-3 rounded-xl border border-white/5 bg-[#1b2328] p-3 hover:border-[#8fd6ff]/50 transition-all">
      <Checkbox checked={checked} onCheckedChange={onCheckedChange} className="absolute top-3 right-3" />
      <div className={`flex items-center justify-center size-8 rounded-lg ${checked ? 'bg-[#8fd6ff] text-black' : 'bg-[#27333a] text-gray-400'}`}>
        <Icon className="h-4 w-4" />
      </div>
      <div>
        <p className="text-sm font-medium">{label}</p>
        <div className="flex gap-1 mt-1.5">
          {[...Array(3)].map((_, i) => (
            <div key={i} className={`h-1 w-3 rounded-full ${i < priority ? 'bg-[#8fd6ff]' : 'bg-gray-600'}`}></div>
          ))}
        </div>
      </div>
    </label>
  );
}
