"use client";

import { useState } from "react";
import { DatasetSummary } from "@/types";
import { analyzeDataset, createPlan, analyzeExperiments } from "@/lib/api";

type Mode = "analyze" | "plan" | "results";

export default function AssistantPanel({ datasets }: { datasets: DatasetSummary[] }) {
  const [datasetId, setDatasetId] = useState("");
  const [mode, setMode] = useState<Mode>("analyze");
  const [goal, setGoal] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  async function handleRun() {
    if (!datasetId) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      if (mode === "analyze") {
        setResult(await analyzeDataset(datasetId));
      } else if (mode === "plan") {
        if (!goal.trim()) {
          setError("Enter a goal first");
          setLoading(false);
          return;
        }
        setResult(await createPlan(datasetId, goal));
      } else {
        setResult(await analyzeExperiments(datasetId));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl">
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <label className="block text-sm text-gray-600 mb-1">Dataset</label>
        <select
          value={datasetId}
          onChange={(e) => { setDatasetId(e.target.value); setResult(null); }}
          className="border rounded px-3 py-2 text-sm mb-4 w-full"
        >
          <option value="">Select a dataset</option>
          {datasets.map((ds) => (
            <option key={ds.dataset_id} value={ds.dataset_id}>{ds.name}</option>
          ))}
        </select>

        <div className="flex gap-2 mb-4">
          <ModeButton active={mode === "analyze"} onClick={() => { setMode("analyze"); setResult(null); }}>
            Analyze Dataset
          </ModeButton>
          <ModeButton active={mode === "plan"} onClick={() => { setMode("plan"); setResult(null); }}>
            Suggest Plan
          </ModeButton>
          <ModeButton active={mode === "results"} onClick={() => { setMode("results"); setResult(null); }}>
            Explain Results
          </ModeButton>
        </div>

        {mode === "plan" && (
          <input
            type="text"
            placeholder="What are you trying to predict?"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            className="border rounded px-3 py-2 text-sm mb-4 w-full"
          />
        )}

        <button
          onClick={handleRun}
          disabled={!datasetId || loading}
          className="bg-blue-600 text-white text-sm px-4 py-2 rounded disabled:bg-gray-300"
        >
          {loading ? "Thinking..." : "Run"}
        </button>

        {error && <p className="text-red-600 text-sm mt-3">{error}</p>}
      </div>

      {result && <ResultDisplay mode={mode} result={result} />}
    </div>
  );
}

function ModeButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`text-xs px-3 py-1.5 rounded ${active ? "bg-gray-800 text-white" : "bg-gray-100 text-gray-600"}`}
    >
      {children}
    </button>
  );
}

function ResultDisplay({ mode, result }: { mode: Mode; result: any }) {
  if (mode === "results") {
    return (
      <div className="bg-white rounded-lg shadow p-6 whitespace-pre-wrap text-sm leading-relaxed">
        {result.analysis}
      </div>
    );
  }

  if (mode === "analyze") {
    return (
      <div className="bg-white rounded-lg shadow p-6 text-sm">
        <h3 className="font-semibold mb-2">Recommended Preprocessing</h3>
        <p className="text-xs text-gray-500 mb-1">Numerical</p>
        <ul className="list-disc pl-5 mb-3">
          {result.recommended_preprocessing.numerical.map((r: string, i: number) => <li key={i}>{r}</li>)}
        </ul>
        <p className="text-xs text-gray-500 mb-1">Categorical</p>
        <ul className="list-disc pl-5 mb-3">
          {result.recommended_preprocessing.categorical.map((r: string, i: number) => <li key={i}>{r}</li>)}
        </ul>
        <h3 className="font-semibold mb-2 mt-4">Potential Risks</h3>
        <ul className="list-disc pl-5">
          {result.potential_risks.map((r: string, i: number) => <li key={i}>{r}</li>)}
        </ul>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6 text-sm">
      <p><span className="font-semibold">Target:</span> {result.target}</p>
      <p><span className="font-semibold">Task:</span> {result.task}</p>
      <p className="mb-3"><span className="font-semibold">Suggested models:</span> {result.models.join(", ")}</p>
      <h3 className="font-semibold mb-1">Preprocessing</h3>
      <ul className="list-disc pl-5">
        {result.preprocessing.numerical.map((r: string, i: number) => <li key={i}>{r}</li>)}
        {result.preprocessing.categorical.map((r: string, i: number) => <li key={i}>{r}</li>)}
      </ul>
    </div>
  );
}