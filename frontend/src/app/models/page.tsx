import { getModels } from "@/lib/api";
import Link from "next/link";

export default async function ModelsPage() {
  const models = await getModels();

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900 p-8">
      <h1 className="text-2xl font-bold mb-6">Model Registry</h1>

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="w-full text-left text-sm">
          <thead className="bg-gray-100 text-gray-600">
            <tr>
              <th className="p-3">Name</th>
              <th className="p-3">Version</th>
              <th className="p-3">Status</th>
              <th className="p-3">Registered</th>
              <th className="p-3"></th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.model_id} className="border-t hover:bg-gray-50">
                <td className="p-3 capitalize">{m.name}</td>
                <td className="p-3">{m.version}</td>
                <td className="p-3 capitalize">{m.status}</td>
                <td className="p-3 text-sm text-gray-500">
                  {new Date(m.created_at).toLocaleString()}
                </td>
                <td className="p-3">
                  <Link href={`/playground?model_id=${m.model_id}`} className="text-blue-600 hover:underline text-sm">
                    Try prediction →
                  </Link>
                </td>
              </tr>
            ))}
            {models.length === 0 && (
              <tr>
                <td colSpan={5} className="p-6 text-center text-gray-400">
                  No models registered yet. Register one from the Experiments page.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}