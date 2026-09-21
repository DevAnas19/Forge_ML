"use client";

import { useState, useEffect } from "react";
import { ModelSummary } from "@/types";
import { getExperiment, getDataset, predictModel } from "@/lib/api";

export default function PredictionForm({
  models,
  initialModelId,
}: {
  models: ModelSummary[];
  initialModelId?: string;
}) {
  const [selectedModelId, setSelectedModelId] = useState(initialModelId ?? "");
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [loadingSchema, setLoadingSchema] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<{ prediction: string; probability: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedModelId) {
      setFeatureColumns([]);
      return;
    }

    async function loadSchema() {
      setLoadingSchema(true);
      setResult(null);
      setError(null);
      try {
        const model = models.find((m) => m.model_id === selectedModelId);
        if (!model) return;

        // We don't have an endpoint that directly says "here are the
        // features this model expects" -- so we walk the chain: model ->
        // its experiment -> that experiment's dataset -> the dataset's
        // profiled columns, minus whichever one was the target.
        const experiment = await getExperiment(model.experiment_id);
        const { dataset, profile } = await getDataset(experiment.dataset_id);

        const allColumns = [
          ...profile.column_types.numerical_columns,
          ...profile.column_types.categorical_columns,
        ];
        const inputColumns = allColumns.filter((c: string) => c !== dataset.target_column);

        setFeatureColumns(inputColumns);
        setFormValues(Object.fromEntries(inputColumns.map((c: string) => [c, ""])));
      } catch {
        setError("Failed to load model's expected inputs");
      } finally {
        setLoadingSchema(false);
      }
    }

    loadSchema();
  }, [selectedModelId, models]);

  function handleChange(col: string, value: string) {
    setFormValues((prev) => ({ ...prev, [col]: value }));
  }

  async function handlePredict() {
    setPredicting(true);
    setError(null);
    setResult(null);

    // Convert numeric-looking strings to actual numbers -- the backend's
    // pipeline expects real numeric types for numerical columns; sending
    // "5000" as a string instead of 5000 would break the scaler step.
    const payload: Record<string, string | number> = {};
    for (const [key, value] of Object.entries(formValues)) {
      const num = Number(value);
      payload[key] = value !== "" && !isNaN(num) ? num : value;
    }

    try {
      const res = await predictModel(selectedModelId, payload);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setPredicting(false);
    }
  }

  return (
    <div className="bg-white rounded-lg shadow p-6 max-w-xl">
      <label className="block text-sm text-gray-600 mb-1">Model</label>
      <select
        value={selectedModelId}
        onChange={(e) => setSelectedModelId(e.target.value)}
        className="border rounded px-3 py-2 text-sm mb-4 w-full"
      >
        <option value="">Select a registered model</option>
        {models.map((m) => (
          <option key={m.model_id} value={m.model_id}>
            {m.name} ({m.version})
          </option>
        ))}
      </select>

      {loadingSchema && <p className="text-sm text-gray-400">Loading input fields...</p>}

      {featureColumns.length > 0 && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          {featureColumns.map((col) => (
            <div key={col}>
              <label className="block text-xs text-gray-500 mb-1">{col}</label>
              <input
                type="text"
                value={formValues[col] ?? ""}
                onChange={(e) => handleChange(col, e.target.value)}
                className="border rounded px-2 py-1 text-sm w-full"
              />
            </div>
          ))}
        </div>
      )}

      {featureColumns.length > 0 && (
        <button
          onClick={handlePredict}
          disabled={predicting}
          className="bg-blue-600 text-white text-sm px-4 py-2 rounded disabled:bg-gray-300"
        >
          {predicting ? "Predicting..." : "Predict"}
        </button>
      )}

      {error && <p className="text-red-600 text-sm mt-3">{error}</p>}

      {result && (
        <div className="mt-4 p-4 bg-gray-50 rounded border">
          <p className="text-sm text-gray-500">Prediction</p>
          <p className="text-xl font-bold">{result.prediction}</p>
          <p className="text-sm text-gray-500 mt-2">
            Confidence: {(result.probability * 100).toFixed(1)}%
          </p>
        </div>
      )}
    </div>
  );
}