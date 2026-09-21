import { getDataset } from "@/lib/api";
import CreateExperimentForm from "@/components/CreateExperimentForm";

export default async function DatasetDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const { dataset, profile } = await getDataset(id);

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900 p-8">
      <h1 className="text-2xl font-bold mb-1">{dataset.name}</h1>
      <p className="text-gray-500 mb-6">
        {dataset.rows} rows · {dataset.columns} columns
        {dataset.target_column && ` · target: ${dataset.target_column}`}
      </p>

      <CreateExperimentForm
        datasetId={dataset.dataset_id}
        numericalColumns={profile.column_types.numerical_columns}
        categoricalColumns={profile.column_types.categorical_columns}
    />

      <h2 className="font-semibold mb-3">Numerical Columns</h2>
      <div className="bg-white rounded-lg shadow overflow-hidden mb-8">
        <table className="w-full text-left text-sm">
          <thead className="bg-gray-100 text-gray-600">
            <tr>
              <th className="p-3">Column</th>
              <th className="p-3">Mean</th>
              <th className="p-3">Median</th>
              <th className="p-3">Min</th>
              <th className="p-3">Max</th>
              <th className="p-3">Std</th>
              <th className="p-3">Missing</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(profile.numerical_stats).map(([col, stats]: [string, any]) => (
              <tr key={col} className="border-t">
                <td className="p-3 font-medium">{col}</td>
                <td className="p-3">{stats.mean}</td>
                <td className="p-3">{stats.median}</td>
                <td className="p-3">{stats.min}</td>
                <td className="p-3">{stats.max}</td>
                <td className="p-3">{stats.std}</td>
                <td className="p-3">{stats.missing}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2 className="font-semibold mb-3">Categorical Columns</h2>
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="w-full text-left text-sm">
          <thead className="bg-gray-100 text-gray-600">
            <tr>
              <th className="p-3">Column</th>
              <th className="p-3">Unique Values</th>
              <th className="p-3">Missing</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(profile.categorical_stats).map(([col, stats]: [string, any]) => (
              <tr key={col} className="border-t">
                <td className="p-3 font-medium">{col}</td>
                <td className="p-3">{stats.unique_values}</td>
                <td className="p-3">{stats.missing}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
}