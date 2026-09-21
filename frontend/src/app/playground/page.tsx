import { getModels } from "@/lib/api";
import PredictionForm from "@/components/PredictionForm";

export default async function PlaygroundPage({
  searchParams,
}: {
  searchParams: Promise<{ model_id?: string }>;
}) {
  const { model_id } = await searchParams;
  const models = await getModels();

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900 p-8">
      <h1 className="text-2xl font-bold mb-6">Prediction Playground</h1>
      <PredictionForm models={models} initialModelId={model_id} />
    </main>
  );
}