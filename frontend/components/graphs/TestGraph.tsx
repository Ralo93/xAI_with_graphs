// components/graphs/TestGraph.tsx
'use client';

import { useState } from 'react';
import GraphViewer from './GraphViewer';
import GraphControls from './GraphControls';

export default function TestGraph() {
  const [layout, setLayout] = useState('forceAtlas2Based');
  const [physics, setPhysics] = useState(true);

  const nodes = [
    { id: 1, label: 'Node 1', color: 'blue', size: 20 },
    { id: 2, label: 'Node 2', color: 'green', size: 15 },
    { id: 3, label: 'Node 3', color: 'red', size: 15 }
  ];

  const edges = [
    { id: '1-2', from: 1, to: 2, width: 2, title: 'Edge 1-2' },
    { id: '2-3', from: 2, to: 3, width: 2, title: 'Edge 2-3' }
  ];

  return (
    <div>
      <GraphControls 
        onLayoutChange={setLayout}
        onPhysicsToggle={setPhysics}
      />
      <div className="w-full h-[800px] border border-gray-200 rounded-lg">
        <GraphViewer 
          nodes={nodes} 
          edges={edges}
          layout={layout}
          physics={physics}
        />
      </div>
    </div>
  );
}