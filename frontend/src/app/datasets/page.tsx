import { getDatasets } from "@/lib/api";
import Link from "next/link";
import DatasetUploadForm from "@/components/DatasetUploadForm";

export default async function DatasetsPage() {
  const datasets = await getDatasets();

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900 p-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Datasets</h1>
      </div>
      <DatasetUploadForm />

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="w-full text-left">
          <thead className="bg-gray-100 text-sm text-gray-600">
            <tr>
              <th className="p-3">Name</th>
              <th className="p-3">Rows</th>
              <th className="p-3">Columns</th>
              <th className="p-3">Target</th>
              <th className="p-3">Task Type</th>
              <th className="p-3">Uploaded</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((ds) => (
              <tr key={ds.dataset_id} className="border-t hover:bg-gray-50">
                <td className="p-3">
                  <Link href={`/datasets/${ds.dataset_id}`} className="text-blue-600 hover:underline">
                    {ds.name}
                  </Link>
                </td>
                <td className="p-3">{ds.rows}</td>
                <td className="p-3">{ds.columns}</td>
                <td className="p-3">{ds.target_column ?? "-"}</td>
                <td className="p-3 capitalize">{ds.task_type ?? "-"}</td>
                <td className="p-3 text-sm text-gray-500">
                  {new Date(ds.created_at).toLocaleString()}
                </td>
              </tr>
            ))}
            {datasets.length === 0 && (
              <tr>
                <td colSpan={6} className="p-6 text-center text-gray-400">
                  No datasets uploaded yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}