import { getDatasets, getExperiments, getModels } from "@/lib/api";

export default async function DashboardPage() {
  const [datasets, experiments, models] = await Promise.all([
    getDatasets(),
    getExperiments(),
    getModels(),
  ]);

  const deployedCount = models.filter((m) => m.status === "deployed").length;

  const recentExperiments = experiments.slice(0, 5);

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900 p-8">
      <h1 className="text-3xl font-bold mb-8">ForgeML</h1>

      <div className="grid grid-cols-4 gap-6 mb-10">
        <StatCard label="Datasets" value={datasets.length} />
        <StatCard label="Experiments" value={experiments.length} />
        <StatCard label="Models" value={models.length} />
        <StatCard label="Deployments" value={deployedCount} />
      </div>

      <h2 className="text-xl font-semibold mb-4">Recent Experiments</h2>
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="w-full text-left">
          <thead className="bg-gray-100 text-sm text-gray-600">
            <tr>
              <th className="p-3">Model</th>
              <th className="p-3">F1</th>
              <th className="p-3">ROC-AUC</th>
              <th className="p-3">Status</th>
              <th className="p-3">Trained</th>
            </tr>
          </thead>
          <tbody>
            {recentExperiments.map((exp) => (
              <tr key={exp.experiment_id} className="border-t">
                <td className="p-3">{exp.model_name}</td>
                <td className="p-3">{exp.metrics?.f1?.toFixed(4) ?? "-"}</td>
                <td className="p-3">{exp.metrics?.roc_auc?.toFixed(4) ?? "-"}</td>
                <td className="p-3 capitalize">{exp.status}</td>
                <td className="p-3 text-sm text-gray-500">
                  {new Date(exp.timestamp).toLocaleString()}
                </td>
              </tr>
            ))}
            {recentExperiments.length === 0 && (
              <tr>
                <td colSpan={5} className="p-6 text-center text-gray-400">
                  No experiments yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <p className="text-sm text-gray-500">{label}</p>
      <p className="text-3xl font-bold mt-1">{value}</p>
    </div>
  );
}