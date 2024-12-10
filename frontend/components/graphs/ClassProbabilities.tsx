'use client';

interface ClassProbabilitiesProps {
  probabilities: number[];
  classNames?: string[];
}

export default function ClassProbabilities({ 
  probabilities, 
  classNames = [] 
}: ClassProbabilitiesProps) {
  return (
    <div className="space-y-2">
      {probabilities.map((prob, idx) => (
        <div key={idx} className="flex items-center gap-2">
          <span className="w-20 text-sm">
            {classNames[idx] || `Class ${idx}`}
          </span>
          <div className="flex-1 bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-blue-600 h-2.5 rounded-full"
              style={{ width: `${prob * 100}%` }}
            />
          </div>
          <span className="w-16 text-sm text-right">
            {(prob * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}