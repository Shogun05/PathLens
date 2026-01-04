"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { pathLensAPI } from "@/lib/api";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface MetricsCardProps {
  type: "baseline" | "optimized";
}

interface MetricsSummary {
  network: {
    circuity_sample_ratio: number;
    intersection_density_global: number;
    link_node_ratio_global: number;
  };
  scores: {
    accessibility_mean: number;
    walkability_mean: number;
    equity_mean: number;
    travel_time_min_mean: number;
    travel_time_score_mean: number;
  };
}

export function MetricsCard({ type }: MetricsCardProps) {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadMetrics = async () => {
      try {
        setLoading(true);
        const data = await pathLensAPI.getMetricsSummary(type);
        setMetrics(data);
      } catch (error) {
        console.error("Failed to load metrics:", error);
      } finally {
        setLoading(false);
      }
    };

    loadMetrics();
  }, [type]);

  if (loading || !metrics) {
    return (
      <Card className="bg-[#1b2328] border-white/10 p-4">
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-white/10 rounded w-1/2"></div>
          <div className="h-8 bg-white/10 rounded w-3/4"></div>
        </div>
      </Card>
    );
  }

  const getTrendIcon = (
    value: number,
    threshold: number,
    higher: boolean = true
  ) => {
    if (higher) {
      return value > threshold ? (
        <TrendingUp className="h-3 w-3 text-green-400" />
      ) : (
        <TrendingDown className="h-3 w-3 text-red-400" />
      );
    } else {
      return value < threshold ? (
        <TrendingDown className="h-3 w-3 text-green-400" />
      ) : (
        <TrendingUp className="h-3 w-3 text-red-400" />
      );
    }
  };

  const getStatusBadge = (value: number, low: number, high: number) => {
    if (value < low)
      return <span className="text-xs text-red-400 font-medium">Low</span>;
    if (value > high)
      return <span className="text-xs text-green-400 font-medium">High</span>;
    return <span className="text-xs text-yellow-400 font-medium">Avg</span>;
  };

  return (
    <div className="space-y-3">
      {/* Network Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Circuity Ratio */}
        <Card className="bg-[#1b2328] border-white/5 p-4 flex items-center justify-between hover:border-white/20 transition-colors">
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wider font-semibold">
              Circuity Ratio
            </p>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-2xl font-bold text-white">
                {metrics.network.circuity_sample_ratio?.toFixed(2) ?? 'N/A'}
              </span>
              {metrics.network.circuity_sample_ratio && getStatusBadge(metrics.network.circuity_sample_ratio, 1.1, 1.5)}
            </div>
          </div>
          <div className="size-10 rounded-lg bg-red-500/10 flex items-center justify-center text-red-400">
            <svg
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
              />
            </svg>
          </div>
        </Card>

        {/* Intersection Density */}
        <Card className="bg-[#1b2328] border-white/5 p-4 flex items-center justify-between hover:border-white/20 transition-colors">
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wider font-semibold">
              Intersection Density
            </p>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-2xl font-bold text-white">
                {metrics.network.intersection_density_global ? Math.round(metrics.network.intersection_density_global) : 'N/A'}
              </span>
              <span className="text-sm text-gray-400 font-normal">/km²</span>
            </div>
          </div>
          <div className="size-10 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-400">
            <svg
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zM14 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"
              />
            </svg>
          </div>
        </Card>

        {/* Link-Node Ratio */}
        <Card className="bg-[#1b2328] border-white/5 p-4 flex items-center justify-between hover:border-white/20 transition-colors">
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wider font-semibold">
              Link-Node Ratio
            </p>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-2xl font-bold text-white">
                {metrics.network.link_node_ratio_global?.toFixed(2) ?? 'N/A'}
              </span>
              {metrics.network.link_node_ratio_global && getStatusBadge(metrics.network.link_node_ratio_global, 2.0, 3.0)}
            </div>
          </div>
          <div className="size-10 rounded-lg bg-yellow-500/10 flex items-center justify-center text-yellow-400">
            <svg
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </div>
        </Card>
      </div>

      {/* Accessibility Scores */}
      <Card className="bg-[#1b2328] border-white/5 p-5">
        <div className="flex justify-between items-center mb-4">
          <h4 className="text-sm font-bold text-white">Accessibility Scores</h4>
          <button className="text-[#8fd6ff] text-xs hover:underline">
            View details
          </button>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Walkability Gauge */}
          <div className="flex flex-col items-center gap-2">
            <div className="relative size-20">
              <svg className="size-full -rotate-90" viewBox="0 0 36 36">
                <path
                  className="text-[#27333a]"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                />
                <path
                  className="text-orange-500"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="currentColor"
                  strokeDasharray={`${metrics.scores.walkability_mean}, 100`}
                  strokeWidth="3"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-sm font-bold text-white">
                  {metrics.scores.walkability_mean.toFixed(1)}
                </span>
              </div>
            </div>
            <span className="text-xs text-gray-400">Walkability</span>
          </div>

          {/* Equity Gauge */}
          <div className="flex flex-col items-center gap-2">
            <div className="relative size-20">
              <svg className="size-full -rotate-90" viewBox="0 0 36 36">
                <path
                  className="text-[#27333a]"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                />
                <path
                  className="text-red-500"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="currentColor"
                  strokeDasharray={`${metrics.scores.equity_mean}, 100`}
                  strokeWidth="3"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-sm font-bold text-white">
                  {metrics.scores.equity_mean.toFixed(1)}
                </span>
              </div>
            </div>
            <span className="text-xs text-gray-400">Equity</span>
          </div>

          {/* Travel Time Gauge */}
          <div className="flex flex-col items-center gap-2">
            <div className="relative size-20">
              <svg className="size-full -rotate-90" viewBox="0 0 36 36">
                <path
                  className="text-[#27333a]"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                />
                <path
                  className="text-yellow-500"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="currentColor"
                  strokeDasharray={`${metrics.scores.travel_time_score_mean}, 100`}
                  strokeWidth="3"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-sm font-bold text-white">
                  {metrics.scores.travel_time_score_mean.toFixed(1)}
                </span>
              </div>
            </div>
            <span className="text-xs text-gray-400">Travel Time</span>
          </div>

          {/* Accessibility Gauge */}
          <div className="flex flex-col items-center gap-2">
            <div className="relative size-20">
              <svg className="size-full -rotate-90" viewBox="0 0 36 36">
                <path
                  className="text-[#27333a]"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                />
                <path
                  className="text-orange-400"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="currentColor"
                  strokeDasharray={`${metrics.scores.accessibility_mean}, 100`}
                  strokeWidth="3"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-sm font-bold text-white">
                  {metrics.scores.accessibility_mean.toFixed(2)}
                </span>
              </div>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-xs text-gray-400">Access</span>
              <span className="text-[10px] text-gray-500">/100</span>
            </div>
          </div>
        </div>

        {/* Score Interpretation Guide */}
        <div className="col-span-2 lg:col-span-4 mt-2">
          <div className="text-xs text-gray-500 p-2 bg-[#1b2328] rounded border border-white/5">
            <span className="font-semibold text-gray-400">Score Guide:</span>{' '}
            <span className="text-red-400">1-20 (Critical)</span> |{' '}
            <span className="text-orange-400">20-40 (Poor)</span> |{' '}
            <span className="text-yellow-400">40-60 (Fair)</span> |{' '}
            <span className="text-green-400">60-80 (Good)</span> |{' '}
            <span className="text-emerald-400">80-100 (Excellent)</span>
          </div>
        </div>
      </Card>
    </div>
  );
}
