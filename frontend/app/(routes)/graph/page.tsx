import PredictionView from '@/components/graphs/PredictionView';

export default function GraphPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h2 className="text-2xl font-bold mb-4">Graph Analysis</h2>
      <PredictionView />
    </div>
  );
}