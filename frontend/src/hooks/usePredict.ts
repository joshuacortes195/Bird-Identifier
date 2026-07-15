import { useCallback, useRef, useState } from "react";
import { ApiError, predict } from "../api/client";
import type { PredictResponse } from "../api/types";
import { prepareImage, type PreparedImage } from "../lib/image";

export type Status = "idle" | "preparing" | "loading" | "success" | "error";

export interface PredictError {
  code: string;
  message: string;
}

interface UsePredictOptions {
  topK?: number;
  onIdentified?: (result: PredictResponse, prepared: PreparedImage) => void;
}

export function usePredict({ topK = 5, onIdentified }: UsePredictOptions = {}) {
  const [status, setStatus] = useState<Status>("idle");
  const [prepared, setPrepared] = useState<PreparedImage | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<PredictError | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const preparedRef = useRef<PreparedImage | null>(null);

  const swapPrepared = useCallback((next: PreparedImage | null) => {
    if (preparedRef.current) URL.revokeObjectURL(preparedRef.current.previewUrl);
    preparedRef.current = next;
    setPrepared(next);
  }, []);

  const selectFile = useCallback(
    async (file: File) => {
      abortRef.current?.abort();
      setError(null);
      setResult(null);
      setStatus("preparing");

      let prep: PreparedImage;
      try {
        prep = await prepareImage(file);
      } catch {
        setError({ code: "prepare_failed", message: "Could not read that image. Try another." });
        setStatus("error");
        return;
      }
      swapPrepared(prep);

      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setStatus("loading");
      try {
        const res = await predict(prep.blob, { topK, signal: ctrl.signal });
        setResult(res);
        setStatus("success");
        onIdentified?.(res, prep);
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return;
        const { code, message } =
          e instanceof ApiError
            ? { code: e.code, message: e.message }
            : { code: "unknown", message: "Something went wrong. Please try again." };
        setError({ code, message });
        setStatus("error");
      }
    },
    [topK, onIdentified, swapPrepared],
  );

  const reset = useCallback(() => {
    abortRef.current?.abort();
    swapPrepared(null);
    setResult(null);
    setError(null);
    setStatus("idle");
  }, [swapPrepared]);

  return {
    status,
    prepared,
    result,
    error,
    selectFile,
    reset,
  };
}
