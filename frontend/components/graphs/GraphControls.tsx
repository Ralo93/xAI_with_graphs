'use client';

import { LAYOUT_PRESETS } from '@/lib/constants/layoutPresets';

interface GraphControlsProps {
  onLayoutChange: (preset: string) => void;
  onPhysicsToggle: (enabled: boolean) => void;
  selectedLayout?: string;
  physicsEnabled?: boolean;
}

export default function GraphControls({ 
  onLayoutChange, 
  onPhysicsToggle,
  selectedLayout = 'forceAtlas2',
  physicsEnabled = true
}: GraphControlsProps) {
  return (
    <div className="flex gap-4 items-center">
      <select 
        value={selectedLayout}
        onChange={(e) => onLayoutChange(e.target.value)}
        className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
      >
        {Object.entries(LAYOUT_PRESETS).map(([key, preset]) => (
          <option key={key} value={key}>
            {preset.name}
          </option>
        ))}
      </select>

      <label className="relative inline-flex items-center cursor-pointer">
        <input
          type="checkbox"
          checked={physicsEnabled}
          className="sr-only peer"
          onChange={(e) => onPhysicsToggle(e.target.checked)}
        />
        <div className="w-11 h-6 bg-gray-200 rounded-full peer peer-checked:after:translate-x-full peer-checked:bg-blue-600 after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all" />
        <span className="ml-3 text-sm font-medium text-gray-900 dark:text-gray-300">
          Physics
        </span>
      </label>
    </div>
  );
}