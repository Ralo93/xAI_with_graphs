'use client';

import { useState } from 'react';
import { usePrediction } from '@/lib/hooks/usePrediction';
import GraphViewer from './GraphViewer';
import GraphControls from './GraphControls';
import ClassProbabilities from './ClassProbabilities';
import LayerSelector from './LayerSelector';

export default function PredictionView() {
  const [layout, setLayout] = useState('forceAtlas2');
  const [physics, setPhysics] = useState(true);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const { predict, data, loading, error } = usePrediction();

  const handlePredict = async () => {
    const features = Array(3).fill(null).map(() => 
      Array(1433).fill(0).map((_, i) => i % 2)
    );
    const inputData = {
      node_features: features,
      edge_index: [[0,1], [1,2], [2,0]]
    };
    await predict(inputData);
  };

  const nodes = data ? data.model_output.map((output, id) => {
    const maxProbIndex = data.class_probabilities[id].indexOf(Math.max(...data.class_probabilities[id]));
    return {
      id,
      label: `Node ${id} (Class ${maxProbIndex})`,
      color: `hsl(${maxProbIndex * 360/7}, 70%, 50%)`,
      size: selectedNode === id ? 25 : 15,
      title: `Class probabilities:\n${data.class_probabilities[id]
        .map((p, i) => `Class ${i}: ${(p * 100).toFixed(1)}%`)
        .join('\n')}`
    };
  }) : [];

  const edges = data ? data.edge_index[0].map((source, idx) => ({
    id: `${idx}`,
    from: source,
    to: data.edge_index[1][idx],
    width: 2,
    title: `Attention: ${(data.attention_weights[selectedLayer][idx][0] * 100).toFixed(1)}%`
  })) : [];

  const layerCount = data?.attention_weights?.length ?? 0;

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div className="flex gap-4">
          <GraphControls 
            onLayoutChange={setLayout}
            onPhysicsToggle={setPhysics}
          />
          {layerCount > 0 && (
            <LayerSelector
              layerCount={layerCount}
              selectedLayer={selectedLayer}
              onLayerChange={setSelectedLayer}
            />
          )}
        </div>
        <button
          onClick={handlePredict}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-300"
        >
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </div>
      
      {error && (
        <div className="p-4 bg-red-100 text-red-700 rounded-lg">
          {error}
        </div>
      )}

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 h-[800px] border border-gray-200 rounded-lg">
          {nodes.length > 0 && edges.length > 0 && (
            <GraphViewer 
              nodes={nodes}
              edges={edges}
              layout={layout}
              physics={physics}
              attentionWeights={data?.attention_weights[selectedLayer]}
              onNodeSelect={setSelectedNode}
            />
          )}
        </div>
        <div className="space-y-4">
          <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">Class Probabilities</h3>
            {selectedNode !== null && data ? (
              <ClassProbabilities 
                probabilities={data.class_probabilities[selectedNode]}
              />
            ) : (
              <p className="text-gray-500">Select a node to view probabilities</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
