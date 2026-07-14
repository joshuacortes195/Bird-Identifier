/** Typed client for the inference API. Base URL comes from an env var so the same
 *  build points at localhost in dev and the deployed API in production. */

import type { HealthResponse, PredictResponse } from "./types";

const API_BASE: string = (
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000"
).replace(/\/$/, "");

/** Error carrying the API's structured `code` so the UI can react (e.g. rate-limited). */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function parseError(res: Response): Promise<ApiError> {
  let code = "http_error";
  let message = `Request failed (${res.status})`;
  try {
    const body = (await res.json()) as { error?: { code?: string; message?: string } };
    if (body?.error) {
      code = body.error.code ?? code;
      message = body.error.message ?? message;
    }
  } catch {
    // non-JSON error body; keep defaults
  }
  return new ApiError(res.status, code, message);
}

export interface PredictOptions {
  topK?: number;
  includeGradcam?: boolean;
  signal?: AbortSignal;
}

export async function predict(file: Blob, opts: PredictOptions = {}): Promise<PredictResponse> {
  const form = new FormData();
  // Preserve a filename so the server sees the right content type for HEIC etc.
  const name = file instanceof File ? file.name : "upload.jpg";
  form.append("file", file, name);
  if (opts.topK != null) form.append("top_k", String(opts.topK));
  if (opts.includeGradcam != null) form.append("include_gradcam", String(opts.includeGradcam));

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: form,
      signal: opts.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(0, "network_error", "Could not reach the server. Check your connection.");
  }

  if (!res.ok) throw await parseError(res);
  return (await res.json()) as PredictResponse;
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, { signal });
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as HealthResponse;
}

export { API_BASE };
