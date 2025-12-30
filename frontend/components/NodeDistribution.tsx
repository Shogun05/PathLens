'use client';

import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Loader2 } from 'lucide-react';

interface NodeDistributionProps {
  nodes: Array<{ accessibility_score?: number }>;
  title?: string;
}

interface BinData {
  min: number;
  max: number;
  count: number;
  percentage: number;
  color: string;
}

export function NodeDistribution({ nodes, title = 'Node Score Distribution' }: NodeDistributionProps) {
  const [bins, setBins] = useState<BinData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!nodes || nodes.length === 0) {
      setLoading(false);
      return;
    }

    // Create 7 bins for score distribution (0-100 range)
    const binRanges = [
      { min: 0, max: 15, color: 'bg-red-500/20 hover:bg-red-500/40' },
      { min: 15, max: 30, color: 'bg-red-500/40 hover:bg-red-500/60' },
      { min: 30, max: 45, color: 'bg-orange-500/30 hover:bg-orange-500/50' },
      { min: 45, max: 60, color: 'bg-yellow-500/30 hover:bg-yellow-500/50' },
      { min: 60, max: 75, color: 'bg-yellow-500/40 hover:bg-yellow-500/60' },
      { min: 75, max: 90, color: 'bg-green-500/30 hover:bg-green-500/50' },
      { min: 90, max: 100, color: 'bg-green-500/20 hover:bg-green-500/40' },
    ];

    // Count nodes in each bin
    const binCounts = binRanges.map(range => {
      const count = nodes.filter(node => {
        const score = node.accessibility_score || 0;
        return score >= range.min && score < range.max;
      }).length;
      return {
        ...range,
        count,
        percentage: (count / nodes.length) * 100,
      };
    });

    setBins(binCounts);
    setLoading(false);
  }, [nodes]);

  if (loading) {
    return (
      <Card className="bg-[#1b2328] border-white/10 p-4">
        <div className="flex items-center justify-center h-48">
          <Loader2 className="h-6 w-6 animate-spin text-[#8fd6ff]" />
        </div>
      </Card>
    );
  }

  if (nodes.length === 0) {
    return (
      <Card className="bg-[#1b2328] border-white/10 p-4">
        <h4 className="text-sm font-bold text-white mb-3">{title}</h4>
        <div className="flex items-center justify-center h-32 text-gray-400 text-sm">
          No data available
        </div>
      </Card>
    );
  }

  const maxCount = Math.max(...bins.map(b => b.count));

  return (
    <Card className="bg-[#1b2328] border-white/10 p-4">
      <h4 className="text-sm font-bold text-white mb-3">{title}</h4>
      
      {/* Chart */}
      <div className="flex items-end gap-1 h-32 w-full">
        {bins.map((bin, index) => {
          const height = maxCount > 0 ? (bin.count / maxCount) * 100 : 0;
          return (
            <div
              key={index}
              className={`w-full rounded-t-sm transition-colors relative group cursor-pointer ${bin.color}`}
              style={{ height: `${height}%` }}
            >
              {/* Tooltip on hover */}
              <div className="hidden group-hover:block absolute -top-14 left-1/2 -translate-x-1/2 bg-black/90 text-white text-[10px] px-2 py-1 rounded whitespace-nowrap z-10">
                <div className="font-semibold">{bin.min}-{bin.max}</div>
                <div>Count: {bin.count}</div>
                <div>{bin.percentage.toFixed(1)}%</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* X-axis labels */}
      <div className="flex justify-between text-[10px] text-gray-400 mt-1 px-1">
        <span>0</span>
        <span>50</span>
        <span>100</span>
      </div>

      {/* Legend */}
      <div className="mt-3 pt-3 border-t border-white/10">
        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-red-500/40"></div>
            <span className="text-gray-400">Critical (0-30)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-orange-500/40"></div>
            <span className="text-gray-400">Poor (30-45)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-yellow-500/40"></div>
            <span className="text-gray-400">Average (45-75)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-green-500/40"></div>
            <span className="text-gray-400">Good (75-100)</span>
          </div>
        </div>
      </div>
    </Card>
  );
}
