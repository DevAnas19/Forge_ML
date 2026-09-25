import { getDatasets } from "@/lib/api";
import AssistantPanel from "@/components/AssistantPanel";

export default async function AssistantPage() {
  const datasets = await getDatasets();

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900 p-8">
      <h1 className="text-2xl font-bold mb-6">AI Assistant</h1>
      <AssistantPanel datasets={datasets} />
    </main>
  );
}