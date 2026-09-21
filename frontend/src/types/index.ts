export interface DatasetSummary {
  dataset_id: string;
  name: string;
  rows: number;
  columns: number;
  target_column: string | null;
  task_type: string | null;
  created_at: string;
}

export interface ExperimentSummary {
  experiment_id: string;
  dataset_id: string;
  model_name: string;
  hyperparameters: Record<string, string>;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
    confusion_matrix: number[][];
  };
  label_classes: string[];
  status: string;
  training_time: number;
  timestamp: string;
}

export interface ModelSummary {
  model_id: string;
  experiment_id: string;
  name: string;
  version: string;
  artifact_path: string;
  status: string;
  created_at: string;
}