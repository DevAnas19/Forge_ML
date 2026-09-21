import { DatasetSummary, ExperimentSummary, ModelSummary } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export async function getDatasets(): Promise<DatasetSummary[]> {
  const res = await fetch(`${API_URL}/api/datasets`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch datasets");
  return res.json();
}

export async function getExperiments(): Promise<ExperimentSummary[]> {
  const res = await fetch(`${API_URL}/api/experiments`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch experiments");
  return res.json();
}

export async function createExperiments(payload: {
  dataset_id: string;
  target_column: string;
  numerical_columns: string[];
  categorical_columns: string[];
  model_names: string[];
  test_size?: number;
  random_seed?: number;
}): Promise<{ experiments: ExperimentSummary[] }> {
  const res = await fetch(`${API_URL}/api/experiments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to create experiments");
  return res.json();
}

export async function registerModel(experimentId: string): Promise<ModelSummary> {
  const res = await fetch(`${API_URL}/api/models/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ experiment_id: experimentId }),
  });
  if (!res.ok) throw new Error("Failed to register model");
  return res.json();
}

export async function getExperiment(id: string): Promise<ExperimentSummary> {
  const res = await fetch(`${API_URL}/api/experiments/${id}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch experiment");
  return res.json();
}

export async function predictModel(
  modelId: string,
  payload: Record<string, string | number>
): Promise<{ prediction: string; probability: number }> {
  const res = await fetch(`${API_URL}/api/models/${modelId}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Prediction failed");
  return res.json();
}

export async function getDataset(id: string): Promise<{ dataset: DatasetSummary; profile: any }> {
  const res = await fetch(`${API_URL}/api/datasets/${id}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch dataset");
  return res.json();
}

export async function getModels(): Promise<ModelSummary[]> {
  const res = await fetch(`${API_URL}/api/models`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch models");
  return res.json();
}
