/** Types mirroring the FastAPI contract (see docs/API.md). Kept in one place so the
 *  client is fully typed and no `any` leaks into components. */

export interface Prediction {
  rank: number;
  class_id: string;
  common_name: string;
  scientific_name: string | null;
  confidence: number; // 0..1
}

export interface ModelInfo {
  name: string;
  backend: string;
  num_classes: number;
  input_size: number;
}

export interface PredictResponse {
  predictions: Prediction[];
  top_prediction: Prediction;
  low_confidence: boolean;
  inference_ms: number;
  model: ModelInfo;
  gradcam_png_base64: string | null;
}

export interface HealthResponse {
  status: string;
  ready: boolean;
  model: ModelInfo | null;
}

export interface ApiErrorBody {
  error: { code: string; message: string };
}
