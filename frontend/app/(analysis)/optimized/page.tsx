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
import { ArrowLeft, RefreshCw, Check, Plus, School, Hospital, Bus, Trees, Map as MapIcon, Download, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { generateOptimizationReport, transformSuggestionsToAmenities } from '@/lib/report-generator';

export default function OptimizedPage() {
  const router = useRouter();
  const {
    location,
    optimizedNodes,
    suggestions,
    selectedSuggestionIds,
    setOptimizedNodes,
    setSuggestions,
    setOptimizedScore,
    optimizedScore,
    baselineScore, // Ensure baselineScore is available from store
    setBaselineScore,
    demoMode,
    mapContainer,
    selectedCity,
  } = usePathLensStore();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);

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
          // Load metrics, nodes, suggestions, and baseline metrics in parallel
          console.log(`Loading optimized data for ${selectedCity}...`);
          const [metrics, nodes, suggsCollection, optimizationResults, baselineResults] = await Promise.all([
            pathLensAPI.getMetricsSummary('optimized', selectedCity).catch(err => {
              console.warn('Failed to load metrics:', err);
              return { scores: { accessibility_mean: 0 } };
            }),
            pathLensAPI.getNodes({ type: 'optimized', city: selectedCity }),
            pathLensAPI.getOptimizationPois(selectedCity).catch(() => ({ features: [] })), // Use optimized POIs endpoint
            pathLensAPI.getOptimizationResults(selectedCity).catch(() => null),
            pathLensAPI.getMetricsSummary('baseline', selectedCity).catch(() => null), // Fetch baseline metrics too
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

          // Use pre-computed city-wide average from metrics API (all nodes)
          const metricsScore = metrics?.scores?.citywide?.accessibility_mean
            || metrics?.scores?.accessibility
            || metrics?.scores?.accessibility_mean;
          
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

          // Also set baseline score from fetched baseline metrics
          const baselineMetrics = baselineResults?.scores;
          const baselineVal = baselineMetrics?.citywide?.accessibility_mean
            || baselineMetrics?.accessibility
            || baselineMetrics?.accessibility_mean;
            
          if (baselineVal && baselineVal > 0) {
            setBaselineScore(baselineVal);
            console.log(`Baseline accessibility: ${baselineVal.toFixed(2)}`);
          } else {
             // If baseline metrics failed, try to use a fallback or keep previous 0
             console.warn('Baseline metrics unavailable for PDF report');
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
    loadData();
  }, [demoMode, selectedCity]); // Re-run when demoMode or selectedCity changes

  // Removed rescoring simulation as requested - using static city-wide scores instead

  const getAmenityIcon = (type: string) => {
    switch (type?.toLowerCase()) {
      case 'school': return School;
      case 'hospital': return Hospital;
      case 'park': return Trees;
      case 'bus': return Bus;
      default: return MapIcon;
    }
  };

  const handleExportReport = async () => {
    setIsExporting(true);
    try {
      const amenities = transformSuggestionsToAmenities(suggestions);
      
      // Debug: Check if mapContainer is available
      console.log('Map container for PDF:', mapContainer);
      console.log('Map container dimensions:', mapContainer?.offsetWidth, 'x', mapContainer?.offsetHeight);

      const cityNameFormatted = selectedCity.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
      await generateOptimizationReport(mapContainer, {
        location: location || `${cityNameFormatted}, India`,
        baselineScore,
        optimizedScore,
        amenities,
        generatedAt: new Date(),
      });

      console.log('Report generated successfully');
    } catch (error) {
      console.error('Failed to generate report:', error);
      setError('Failed to generate report. Please try again.');
    } finally {
      setIsExporting(false);
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
              className="flex items-center gap-2 bg-[#8fd6ff]/10 hover:bg-[#8fd6ff]/20 text-[#8fd6ff] border-[#8fd6ff]/20 disabled:opacity-50"
              onClick={handleExportReport}
              disabled={isExporting || suggestions.length === 0}
            >
              {isExporting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="text-sm font-bold">Generating...</span>
                </>
              ) : (
                <>
                  <Download className="h-4 w-4" />
                  <span className="text-sm font-bold">Export Report</span>
                </>
              )}
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

          {/* Proposed Interventions section removed as requested */}

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

          {/* Update Score button removed as requested */}
        </motion.aside>
      </div>
    </div>
  );
}
