import { getExperiments } from "@/lib/api";
import Link from "next/link";
import RegisterModelButton from "@/components/RegisterModelButton";

export default async function ExperimentsPage() {
  const experiments = await getExperiments();

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900 p-8">
      <h1 className="text-2xl font-bold mb-6">Experiments</h1>

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="w-full text-left text-sm">
          <thead className="bg-gray-100 text-gray-600">
            <tr>
              <th className="p-3">Model</th>
              <th className="p-3">F1</th>
              <th className="p-3">ROC-AUC</th>
              <th className="p-3">Time</th>
              <th className="p-3">Status</th>
              <th className="p-3">Trained</th>
              <th className="p-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((exp) => (
              <tr key={exp.experiment_id} className="border-t hover:bg-gray-50">
                <td className="p-3">
                  <Link href={`/experiments/${exp.experiment_id}`} className="text-blue-600 hover:underline">
                    {exp.model_name}
                  </Link>
                </td>
                <td className="p-3">
                    <RegisterModelButton experimentId={exp.experiment_id} />
                </td>
                <td className="p-3">{exp.metrics?.f1?.toFixed(4) ?? "-"}</td>
                <td className="p-3">{exp.metrics?.roc_auc?.toFixed(4) ?? "-"}</td>
                <td className="p-3">{exp.training_time}s</td>
                <td className="p-3 capitalize">{exp.status}</td>
                <td className="p-3 text-sm text-gray-500">
                  {new Date(exp.timestamp).toLocaleString()}
                </td>
              </tr>
            ))}
            {experiments.length === 0 && (
              <tr>
                <td colSpan={6} className="p-6 text-center text-gray-400">
                  No experiments yet. Create one from a dataset's page.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}