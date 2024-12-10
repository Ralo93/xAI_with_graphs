'use client';

interface LayerSelectorProps {
  layerCount: number;
  selectedLayer: number;
  onLayerChange: (layer: number) => void;
}

export default function LayerSelector({ 
  layerCount, 
  selectedLayer, 
  onLayerChange 
}: LayerSelectorProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm font-medium">Attention Layer:</span>
      <select
        value={selectedLayer}
        onChange={(e) => onLayerChange(parseInt(e.target.value))}
        className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg p-2 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
      >
        {Array.from({ length: layerCount }, (_, i) => (
          <option key={i} value={i}>Layer {i + 1}</option>
        ))}
      </select>
    </div>
  );
}