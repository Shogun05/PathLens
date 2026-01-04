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
import { MetricsCard } from '@/components/MetricsCard';
import { NodeDistribution } from '@/components/NodeDistribution';
import { CriticalNodes } from '@/components/CriticalNodes';
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
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);

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
          // Load metrics, nodes, and suggestions in parallel
          console.log('Loading optimized data (full city)...');
          const [metrics, nodes, suggsCollection, optimizationResults] = await Promise.all([
            pathLensAPI.getMetricsSummary('optimized').catch(err => {
              console.warn('Failed to load metrics:', err);
              return { scores: { accessibility_mean: 0 } };
            }),
            pathLensAPI.getNodes({ type: 'optimized' }),
            pathLensAPI.getOptimizationPois().catch(() => ({ features: [] })), // Use optimized POIs endpoint
            pathLensAPI.getOptimizationResults().catch(() => null),
          ]);

          console.log(`Loaded ${nodes?.length || 0} optimized nodes`);
          if (nodes && nodes.length > 0) setOptimizedNodes(nodes);
          else setOptimizedNodes([]);

          // Handle GeoJSON FeatureCollection from optimization POIs
          const suggestionsList = suggsCollection?.features || [];

          if (suggestionsList && suggestionsList.length > 0) {
            console.log(`Loaded ${suggestionsList.length} optimization suggestions`);
            setSuggestions(suggestionsList);
            // Select all suggestions by default
            const selectAllSuggestions = usePathLensStore.getState().selectAllSuggestions;
            selectAllSuggestions();
          } else {
            setSuggestions([]);
          }

          // Use optimization results for score if available
          if (optimizationResults?.metrics) {
            console.log('Optimization metrics:', optimizationResults.metrics);
            // Score is derived from nodes, but we can show optimization metadata
          }

          // Use pre-computed city-wide average from metrics API (all 182k nodes)
          const metricsScore = metrics?.scores?.accessibility_mean;
          if (metricsScore && metricsScore > 0) {
            setOptimizedScore(metricsScore);
            console.log(`City-wide optimized accessibility: ${metricsScore.toFixed(2)}`);
          } else {
            // Fallback: calculate from loaded nodes if metrics unavailable
            if (nodes && nodes.length > 0) {
              const avgScore = nodes.reduce((sum, n) => sum + (n.accessibility_score || 0), 0) / nodes.length;
              setOptimizedScore(avgScore);
              console.log(`Calculated accessibility from ${nodes.length} sampled nodes: ${avgScore.toFixed(2)}`);
            } else {
              setOptimizedScore(0);
            }
          }
        }
      } catch (error) {
        console.error('Failed to load optimized data:', error);
        setError('Failed to load optimized data. Please refresh the page.');
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

      {/* Breadcrumbs & Controls - Pointer events enabled */}
      <div className="border-b border-white/10 bg-[#101518] px-6 py-3 pointer-events-auto z-20">
        <div className="flex items-center justify-between gap-4">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <a className="hover:text-[#8fd6ff]">Projects</a>
              <span>/</span>
              <a className="hover:text-[#8fd6ff]">{location || 'City Analysis'}</a>
              <span>/</span>
              <span className="text-white font-medium">Optimization</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center bg-[#1b2328] rounded-lg p-1 border border-white/10">
              <button className="px-3 py-1.5 rounded text-xs font-medium bg-[#8fd6ff] text-[#101518]">
                Single View
              </button>
              <button
                className="px-3 py-1.5 rounded text-xs font-medium text-gray-400 hover:text-white"
                onClick={() => router.push('/comparison')}
              >
                Split View
              </button>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="flex items-center gap-2 bg-[#8fd6ff]/10 hover:bg-[#8fd6ff]/20 text-[#8fd6ff] border-[#8fd6ff]/20"
              onClick={() => {
                // TODO: Implement export functionality
                console.log('Export report');
              }}
            >
              <span className="material-symbols-outlined text-[18px]">download</span>
              <span className="text-sm font-bold">Export Report</span>
            </Button>
          </div>
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

            {/* Error message display */}
            {error && (
              <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                <p className="text-xs text-red-400">{error}</p>
              </div>
            )}
          </div>

          {/* Suggestions Section - Fixed height, scrollable if needed */}
          <div className="border-b border-white/10 bg-[#1b2328]">
            <div className="p-4">
              <h3 className="font-semibold text-sm text-gray-300 mb-2">Proposed Interventions</h3>
              <p className="text-xs text-gray-500">
                Select interventions to include in the rescoring simulation.
              </p>
            </div>

            <div className="max-h-48 overflow-y-auto px-4 pb-4">
              <div className="space-y-3">
                {suggestions.length === 0 ? (
                  <div className="text-center py-8 text-gray-400 text-sm">
                    No interventions available
                  </div>
                ) : (
                  suggestions.map((suggestion, index) => {
                    const Icon = getAmenityIcon(suggestion.properties?.amenity_type);
                    // Create unique key combining node ID and amenity type to handle multiple POIs at same node
                    const suggestionId = suggestion.properties?.id
                      ? `${suggestion.properties.id}-${suggestion.properties?.amenity_type || index}`
                      : `suggestion-${index}`;
                    const isSelected = selectedSuggestionIds.has(suggestionId);

                    return (
                      <Card
                        key={suggestionId}
                        className={`p-3 border transition-all cursor-pointer ${isSelected
                          ? 'bg-[#8fd6ff]/10 border-[#8fd6ff]/50'
                          : 'bg-[#1b2328] border-white/5 hover:border-white/20'
                          }`}
                        onClick={() => toggleSuggestion(suggestionId)}
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
                                onCheckedChange={() => toggleSuggestion(suggestionId)}
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
                              {/* Node ID for manual verification */}
                              <span className="text-[10px] text-gray-600 font-mono">
                                ID: {suggestion.properties?.id || 'N/A'}
                              </span>
                            </div>
                          </div>
                        </div>
                      </Card>
                    );
                  }))}
              </div>
            </div>
          </div>

          {/* Network & Accessibility Metrics - Main scrollable section */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {!demoMode && <MetricsCard type="optimized" />}

            {/* Distribution & Critical Nodes */}
            {!demoMode && (
              <>
                <NodeDistribution nodes={optimizedNodes} />
                <CriticalNodes
                  nodes={optimizedNodes}
                  onLocateNode={(node) => {
                    console.log('Locate node:', node);
                    // TODO: Pan map to node location
                  }}
                />
              </>
            )}
          </div>

          {/* Update Score Button - Fixed at bottom */}
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
