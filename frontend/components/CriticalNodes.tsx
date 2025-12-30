'use client';

import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { AlertTriangle, Navigation } from 'lucide-react';

interface Node {
  osmid: string;
  x: number;
  y: number;
  accessibility_score?: number;
  [key: string]: any;
}

interface CriticalNodesProps {
  nodes: Node[];
  onLocateNode?: (node: Node) => void;
  limit?: number;
}

export function CriticalNodes({ nodes, onLocateNode, limit = 3 }: CriticalNodesProps) {
  const [criticalNodes, setCriticalNodes] = useState<Node[]>([]);

  useEffect(() => {
    if (!nodes || nodes.length === 0) {
      setCriticalNodes([]);
      return;
    }

    // Sort nodes by accessibility score (ascending) and take the worst ones
    const sorted = [...nodes]
      .filter(node => node.accessibility_score !== undefined)
      .sort((a, b) => (a.accessibility_score || 0) - (b.accessibility_score || 0))
      .slice(0, limit);

    setCriticalNodes(sorted);
  }, [nodes, limit]);

  if (criticalNodes.length === 0) {
    return (
      <Card className="bg-[#1b2328] border-white/10 p-4">
        <h4 className="text-sm font-bold text-red-400 flex items-center gap-2 mb-2">
          <AlertTriangle className="h-4 w-4" />
          Critical Nodes
        </h4>
        <div className="text-sm text-gray-400">
          No critical nodes identified
        </div>
      </Card>
    );
  }

  const getNodeLabel = (node: Node, index: number) => {
    // Use osmid from the Node type
    if (node.osmid) {
      return `Node #${node.osmid}`;
    }
    return `Node #${index + 1}`;
  };

  const getNodeSubLabel = (node: Node) => {
    // Use x (lon) and y (lat) from the Node type
    const lat = node.y?.toFixed(4);
    const lon = node.x?.toFixed(4);
    if (lat && lon) {
      return `${lat}, ${lon}`;
    }
    return 'Location unknown';
  };

  const getScoreColor = (score: number) => {
    if (score < 20) return 'text-red-500';
    if (score < 40) return 'text-orange-500';
    return 'text-yellow-500';
  };

  return (
    <Card className="bg-[#1b2328] border-white/10 p-4">
      <h4 className="text-sm font-bold text-red-400 flex items-center gap-2 mb-3">
        <AlertTriangle className="h-4 w-4" />
        Critical Nodes
      </h4>
      
      <div className="space-y-2">
        {criticalNodes.map((node, index) => {
          const score = node.accessibility_score || 0;
          return (
            <div
              key={node.osmid || index}
              className="flex justify-between items-center bg-[#101518] p-2 rounded border border-white/10 hover:border-[#8fd6ff]/30 transition-colors"
            >
              <div className="flex flex-col flex-1 min-w-0">
                <span className="text-xs text-white font-medium truncate">
                  {getNodeLabel(node, index)}
                </span>
                <span className="text-[10px] text-gray-400 truncate">
                  {getNodeSubLabel(node)}
                </span>
              </div>
              
              <div className="flex items-center gap-3 ml-2">
                <span className={`text-xs font-bold ${getScoreColor(score)}`}>
                  {score.toFixed(0)}/100
                </span>
                {onLocateNode && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-gray-400 hover:text-[#8fd6ff]"
                    onClick={() => onLocateNode(node)}
                  >
                    <Navigation className="h-3 w-3" />
                  </Button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {criticalNodes.length > 0 && (
        <div className="mt-3 pt-3 border-t border-white/10 text-[10px] text-gray-400">
          Showing {criticalNodes.length} worst-performing node{criticalNodes.length !== 1 ? 's' : ''}
        </div>
      )}
    </Card>
  );
}
