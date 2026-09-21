"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createExperiments } from "@/lib/api";

const AVAILABLE_MODELS = [
  { value: "logistic_regression", label: "Logistic Regression" },
  { value: "random_forest", label: "Random Forest" },
  { value: "xgboost", label: "XGBoost" },
];

export default function CreateExperimentForm({
  datasetId,
  numericalColumns,
  categoricalColumns,
}: {
  datasetId: string;
  numericalColumns: string[];
  categoricalColumns: string[];
}) {
  const [targetColumn, setTargetColumn] = useState("");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const allColumns = [...numericalColumns, ...categoricalColumns];

  async function handleRun() {
    if (!targetColumn || selectedModels.length === 0) return;

    setRunning(true);
    setError(null);

    // Exclude the target column from whichever list it actually belongs to
    // -- it's a real column type from the profiler, we just don't want it
    // treated as an input feature anymore now that it's the prediction target.
    const finalNumerical = numericalColumns.filter((c) => c !== targetColumn);
    const finalCategorical = categoricalColumns.filter((c) => c !== targetColumn);

    try {
      await createExperiments({
        dataset_id: datasetId,
        target_column: targetColumn,
        numerical_columns: finalNumerical,
        categorical_columns: finalCategorical,
        model_names: selectedModels,
      });

      router.push("/experiments");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setRunning(false);
    }
  }

  function toggleModel(value: string) {
    setSelectedModels((prev) =>
      prev.includes(value) ? prev.filter((m) => m !== value) : [...prev, value]
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-8">
      <h2 className="font-semibold mb-4">Create Experiment</h2>

      <label className="block text-sm text-gray-600 mb-1">Target Column</label>
      <select
        value={targetColumn}
        onChange={(e) => setTargetColumn(e.target.value)}
        className="border rounded px-3 py-2 text-sm mb-4 w-full max-w-xs"
      >
        <option value="">Select a column</option>
        {allColumns.map((col) => (
          <option key={col} value={col}>{col}</option>
        ))}
      </select>

      <label className="block text-sm text-gray-600 mb-1">Models</label>
      <div className="flex gap-4 mb-4">
        {AVAILABLE_MODELS.map((m) => (
          <label key={m.value} className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={selectedModels.includes(m.value)}
              onChange={() => toggleModel(m.value)}
            />
            {m.label}
          </label>
        ))}
      </div>

      <button
        onClick={handleRun}
        disabled={!targetColumn || selectedModels.length === 0 || running}
        className="bg-blue-600 text-white text-sm px-4 py-2 rounded disabled:bg-gray-300"
      >
        {running ? "Training..." : "Run Experiment"}
      </button>

      {error && <p className="text-red-600 text-sm mt-2">{error}</p>}
    </div>
  );
}