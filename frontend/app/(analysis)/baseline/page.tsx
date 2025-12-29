'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { usePathLensStore } from '@/lib/store';
import { pathLensAPI } from '@/lib/api';
import { generateDemoNodes } from '@/lib/demo-data';
import { ArrowRight, TrendingUp, Users, MapPin, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

export default function BaselinePage() {
  const router = useRouter();
  const {
    location,
    baselineNodes,
    setBaselineNodes,
    setBaselineScore,
    baselineScore,
    demoMode,
  } = usePathLensStore();

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadBaselineData = async () => {
      try {
        setLoading(true);
        
        if (demoMode) {
          // Use demo data
          const demoNodes = generateDemoNodes([12.9716, 77.5946], 50);
          setBaselineNodes(demoNodes);
          const avgScore = demoNodes.reduce((sum, n) => sum + (n.accessibility_score || 0), 0) / demoNodes.length;
          setBaselineScore(avgScore);
        } else {
          // Use API data
          const nodes = await pathLensAPI.getNodes('baseline');
          if (nodes && nodes.length > 0) {
            setBaselineNodes(nodes);
            const avgScore = nodes.reduce((sum, n) => sum + (n.accessibility_score || 0), 0) / nodes.length;
            setBaselineScore(avgScore);
          } else {
            // Clear nodes if no data returned
            setBaselineNodes([]);
            setBaselineScore(0);
          }
        }
      } catch (error) {
        console.error('Failed to load baseline data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadBaselineData();
  }, [demoMode]); // Re-run when demoMode changes

  const handleProceedToOptimization = () => {
    router.push('/optimized');
  };

  // Calculate stats
  const totalNodes = baselineNodes.length || 1;
  const avgAccessibility = baselineNodes.reduce((sum, n) => sum + (n.accessibility_score || 0), 0) / totalNodes || 0;
  const avgWalkability = baselineNodes.reduce((sum, n) => sum + (n.walkability_score || 0), 0) / totalNodes || 0;
  const avgEquity = baselineNodes.reduce((sum, n) => sum + (n.equity_score || 0), 0) / totalNodes || 0;
  const avgTravelTime = baselineNodes.reduce((sum, n) => sum + (n.travel_time_min || 0), 0) / totalNodes || 0;

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
          <span className="text-white font-medium">Baseline Analysis</span>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden relative">
        {/* Sidebar - Moved to RIGHT side as per request "fade away the pane on the right" */}
        
        <motion.aside 
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 50 }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
          className="absolute right-0 top-0 bottom-0 w-[400px] bg-[#0f1c23]/90 backdrop-blur-xl border-l border-white/10 flex flex-col z-20 pointer-events-auto"
        >
          <div className="p-6 border-b border-white/10">
            <div className="flex items-center justify-between mb-4">
              <h1 className="text-2xl font-bold text-white">Baseline Analysis</h1>
              <Badge variant="outline" className="border-red-500/50 text-red-400 bg-red-500/10">
                Current State
              </Badge>
            </div>
            
            <div className="flex items-end gap-2 mb-2">
              <span className="text-4xl font-bold text-white">{baselineScore.toFixed(1)}</span>
              <span className="text-sm text-gray-400 mb-1">/ 100</span>
            </div>
            <p className="text-sm text-gray-400">Overall Accessibility Score</p>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* Metrics */}
            <div className="space-y-4">
              <MetricItem 
                icon={<TrendingUp className="h-4 w-4" />}
                label="Accessibility Index"
                value={avgAccessibility.toFixed(1)}
                max={100}
                color="text-[#8fd6ff]"
              />
              <MetricItem 
                icon={<Activity className="h-4 w-4" />}
                label="Walkability Score"
                value={avgWalkability.toFixed(1)}
                max={100}
                color="text-green-400"
              />
              <MetricItem 
                icon={<Users className="h-4 w-4" />}
                label="Social Equity"
                value={avgEquity.toFixed(1)}
                max={100}
                color="text-purple-400"
              />
              <MetricItem 
                icon={<MapPin className="h-4 w-4" />}
                label="Avg Travel Time"
                value={avgTravelTime.toFixed(0)}
                max={60}
                unit="min"
                color="text-orange-400"
              />
            </div>

            <Separator className="bg-white/10" />

            {/* Key Findings */}
            <Card className="bg-[#1b2328] border-white/5 p-4">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <span className="material-symbols-outlined text-yellow-500 text-sm">lightbulb</span>
                Key Findings
              </h3>
              <div className="space-y-3 text-sm text-gray-300">
                <div className="flex justify-between">
                  <span className="text-gray-400">Underserved Areas</span>
                  <span className="text-white font-mono">12 Zones</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Critical Gaps</span>
                  <span className="text-white font-mono">Schools, Parks</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Parks</span>
                  <span className="text-white font-mono">
                    {(baselineNodes.reduce((sum, n) => sum + (n.dist_to_park || 0), 0) / totalNodes).toFixed(0)}m avg
                  </span>
                </div>
              </div>
            </Card>

            {/* CTA */}
            <Button
              onClick={handleProceedToOptimization}
              className="w-full h-12 bg-[#8fd6ff] hover:bg-[#b0e2ff] text-[#101518] font-bold shadow-[0_0_20px_rgba(143,214,255,0.2)]"
            >
              View Optimized Scenario
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </div>
        </motion.aside>
      </div>
    </div>
  );
}

function MetricItem({ icon, label, value, max, unit = '', color }: any) {
  const percentage = (parseFloat(value) / max) * 100;
  
  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center gap-2">
          <span className={color}>{icon}</span>
          <span className="text-sm text-gray-400">{label}</span>
        </div>
        <span className={`text-lg font-bold ${color}`}>
          {value}{unit ? ` ${unit}` : '/100'}
        </span>
      </div>
      <div className="w-full h-2 bg-[#27333a] rounded-full overflow-hidden">
        <div
          className={`h-full ${color.replace('text-', 'bg-')} transition-all`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
