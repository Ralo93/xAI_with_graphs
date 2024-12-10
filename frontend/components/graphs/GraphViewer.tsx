import React, { useEffect, useRef } from 'react';
import { Network } from 'vis-network';
import { DataSet } from 'vis-data';
import { LAYOUT_PRESETS } from '@/lib/constants/layoutPresets';

interface Node {
  id: number;
  label: string;
  color?: string;
  size?: number;
  title?: string;
}

interface Edge {
  id: string;
  from: number;
  to: number;
  width?: number;
  title?: string;
  color?: string;
}

interface GraphViewerProps {
  nodes: Node[];
  edges: Edge[];
  height?: string;
  width?: string;
  backgroundColor?: string;
  layout?: string;
  physics?: boolean;
  attentionWeights?: number[][];
  attentionLayers?: number[][][];
  selectedLayer?: number;
  onNodeSelect?: (nodeId: number) => void;
}

export default function GraphViewer({
  nodes,
  edges,
  height = "800px",
  width = "100%",
  backgroundColor = "#222222",
  layout = "forceAtlas2Based",
  physics = true,
  attentionWeights,
  onNodeSelect
}: GraphViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<Network | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const edgesWithAttention = edges.map((edge, idx) => ({
      ...edge,
      width: attentionWeights ? 1 + (attentionWeights[idx] || []).reduce((a, b) => a + b, 0) * 5 : edge.width,
      color: attentionWeights ? {
        color: '#2B7CE9',
        opacity: Math.min((attentionWeights[idx] || []).reduce((a, b) => a + b, 0), 1)
      } : undefined
    }));

    const nodesDataSet = new DataSet(nodes);
    const edgesDataSet = new DataSet(edgesWithAttention);

    const layoutConfig = LAYOUT_PRESETS[layout as keyof typeof LAYOUT_PRESETS];
    const options = {
      height,
      width,
      backgroundColor,
      nodes: {
        font: { color: 'white' }
      },
      edges: {
        color: {
          inherit: false
        },
        smooth: {
          enabled: true,
          type: 'continuous',
          roundness: 0.5
        }
      },
      physics: physics ? {
        enabled: true,
        ...layoutConfig.physics
      } : { enabled: false },
      ...layoutConfig.layout
    }

    networkRef.current = new Network(
      containerRef.current,
      { nodes: nodesDataSet, edges: edgesDataSet },
      options
    );

    if (onNodeSelect) {
      networkRef.current.on('selectNode', function(params) {
        onNodeSelect(params.nodes[0]);
      });
    }

    return () => {
      networkRef.current?.destroy();
    };
  }, [nodes, edges, height, width, backgroundColor, layout, physics, attentionWeights, onNodeSelect]);

  return <div ref={containerRef} />;
}