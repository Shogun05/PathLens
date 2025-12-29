'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { ScrollArea } from '@/components/ui/scroll-area';
import { usePathLensStore, Suggestion } from '@/lib/store';
import { pathLensAPI } from '@/lib/api';
import { generateDemoNodes } from '@/lib/demo-data';
import { ArrowLeft, RefreshCw, Check, Plus, School, Hospital, Bus, Trees, Map as MapIcon } from 'lucide-react';
import { motion } from 'framer-motion';

export default function OptimizedPage() {
  const router = useRouter();
  const {
    location,
    optimizedNodes,
    suggestions,
    selectedSuggestionIds,
    setOptimizedNodes,
    setSuggestions,
    toggleSuggestion,
    isRescoring,
    setIsRescoring,
    setOptimizedScore,
    optimizedScore,
    baselineScore,
    demoMode,
  } = usePathLensStore();

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        
        if (demoMode) {
          // Use demo data
          const demoNodes = generateDemoNodes([12.9716, 77.5946], 50).map(n => ({
            ...n,
            score: Math.min(100, (n.score || 0) + 15)
          }));
          setOptimizedNodes(demoNodes);
          
          const avgScore = demoNodes.reduce((sum, n) => sum + (n.accessibility_score || 0), 0) / demoNodes.length;
          setOptimizedScore(avgScore);
          
          // Add some demo suggestions if needed, or clear them
          // For now, let's keep suggestions empty or mock them if requested
          setSuggestions([]); 
        } else {
          // Use API data
          const [nodes, suggsCollection] = await Promise.all([
            pathLensAPI.getNodes('optimized'),
            pathLensAPI.getSuggestions(),
          ]);
          
          if (nodes && nodes.length > 0) setOptimizedNodes(nodes);
          else setOptimizedNodes([]);
          
          // Handle both array (direct list) and FeatureCollection formats
          const suggestionsList = Array.isArray(suggsCollection) 
            ? suggsCollection 
            : (suggsCollection as any).features || [];
          
          if (suggestionsList && suggestionsList.length > 0) setSuggestions(suggestionsList);
          else setSuggestions([]);
          
          // Calculate initial optimized score if not set
          if (nodes && nodes.length > 0) {
            const avgScore = nodes.reduce((sum, n) => sum + (n.accessibility_score || 0), 0) / nodes.length;
            setOptimizedScore(avgScore);
          } else {
            setOptimizedScore(0);
          }
        }
      } catch (error) {
        console.error('Failed to load optimized data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [demoMode]); // Re-run when demoMode changes

  const handleRescore = async () => {
    setIsRescoring(true);
    try {
      const result = await pathLensAPI.rescore({
        location,
        selected_ids: Array.from(selectedSuggestionIds)
      });
      setOptimizedScore(result.new_score);
    } catch (error) {
      console.error('Rescoring failed:', error);
    } finally {
      setIsRescoring(false);
    }
  };

  const getAmenityIcon = (type: string) => {
    switch (type?.toLowerCase()) {
      case 'school': return School;
      case 'hospital': return Hospital;
      case 'park': return Trees;
      case 'bus': return Bus;
      default: return MapIcon;
    }
  };

  const scoreImprovement = optimizedScore - baselineScore;

  return (
    <div className="flex h-full flex-col pointer-events-none">
      {/* Header removed - moved to layout */}

      {/* Breadcrumbs - Pointer events enabled */}
      <div className="border-b border-white/10 bg-[#101518] px-6 py-3 pointer-events-auto z-20">
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <a className="hover:text-[#8fd6ff]">Projects</a>
          <span>/</span>
          <a className="hover:text-[#8fd6ff]">{location || 'City Analysis'}</a>
          <span>/</span>
          <span className="text-white font-medium">Optimization</span>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden relative">
        {/* Sidebar - Moved to LEFT side for contrast with baseline */}
        <motion.aside 
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -50 }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
          className="absolute left-0 top-0 bottom-0 w-[400px] bg-[#0f1c23]/90 backdrop-blur-xl border-r border-white/10 flex flex-col z-20 pointer-events-auto"
        >
          <div className="p-6 border-b border-white/10">
            <div className="flex items-center gap-2 mb-4">
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8 -ml-2 text-gray-400 hover:text-white"
                onClick={() => router.push('/baseline')}
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <h1 className="text-2xl font-bold text-white">Optimization</h1>
            </div>
            
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-end gap-2">
                <span className="text-4xl font-bold text-[#8fd6ff]">{optimizedScore.toFixed(1)}</span>
                <span className="text-sm text-gray-400 mb-1">/ 100</span>
              </div>
              <Badge className="bg-green-500/20 text-green-400 border-green-500/50">
                +{scoreImprovement.toFixed(1)} Improvement
              </Badge>
            </div>
            <p className="text-sm text-gray-400">Projected Accessibility Score</p>
          </div>

          <div className="flex-1 overflow-hidden flex flex-col">
            <div className="p-4 border-b border-white/10 bg-[#1b2328]">
              <h3 className="font-semibold text-sm text-gray-300 mb-2">Proposed Interventions</h3>
              <p className="text-xs text-gray-500">
                Select interventions to include in the rescoring simulation.
              </p>
            </div>

            <ScrollArea className="flex-1 p-4">
              <div className="space-y-3">
                {suggestions.map((suggestion) => {
                  const Icon = getAmenityIcon(suggestion.properties?.amenity_type);
                  const isSelected = selectedSuggestionIds.has(suggestion.id);
                  
                  return (
                    <Card 
                      key={suggestion.id}
                      className={`p-3 border transition-all cursor-pointer ${
                        isSelected 
                          ? 'bg-[#8fd6ff]/10 border-[#8fd6ff]/50' 
                          : 'bg-[#1b2328] border-white/5 hover:border-white/20'
                      }`}
                      onClick={() => toggleSuggestion(suggestion.id)}
                    >
                      <div className="flex items-start gap-3">
                        <div className={`p-2 rounded-lg ${isSelected ? 'bg-[#8fd6ff]/20 text-[#8fd6ff]' : 'bg-white/5 text-gray-400'}`}>
                          <Icon className="h-4 w-4" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex justify-between items-start">
                            <h4 className="font-medium text-sm text-white truncate">
                              New {suggestion.properties?.amenity_type || 'Amenity'}
                            </h4>
                            <Switch 
                              checked={isSelected}
                              onCheckedChange={() => toggleSuggestion(suggestion.id)}
                              className="scale-75 data-[state=checked]:bg-[#8fd6ff]"
                            />
                          </div>
                          <p className="text-xs text-gray-400 mt-1 truncate">
                            {suggestion.properties?.description || 'Proposed location based on gap analysis'}
                          </p>
                          <div className="flex items-center gap-2 mt-2">
                            <Badge variant="secondary" className="text-[10px] h-5 bg-white/5 text-gray-400">
                              +{suggestion.properties?.impact_score || 5} pts
                            </Badge>
                            <span className="text-[10px] text-gray-500">
                              Cost: ${suggestion.properties?.cost_estimate || '50k'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </Card>
                  );
                })}
              </div>
            </ScrollArea>
          </div>

          <div className="p-4 border-t border-white/10 bg-[#0f1c23]">
            <Button 
              className="w-full bg-[#8fd6ff] hover:bg-[#b0e2ff] text-[#101518] font-bold"
              onClick={handleRescore}
              disabled={isRescoring}
            >
              {isRescoring ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Recalculating...
                </>
              ) : (
                <>
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Update Score
                </>
              )}
            </Button>
          </div>
        </motion.aside>
      </div>
    </div>
  );
}
