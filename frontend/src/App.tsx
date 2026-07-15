import { useCallback, useState } from "react";
import type { PredictResponse } from "./api/types";
import { Dropzone } from "./components/Dropzone";
import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { ImagePreview } from "./components/ImagePreview";
import { OtherPredictions } from "./components/OtherPredictions";
import { RecentUploads, type RecentItem } from "./components/RecentUploads";
import { ResultsPanel } from "./components/ResultsPanel";
import { ErrorState, LoadingState } from "./components/States";
import { usePredict } from "./hooks/usePredict";
import { formatBytes, type PreparedImage } from "./lib/image";

const HISTORY_LIMIT = 8;

export default function App() {
  const [recents, setRecents] = useState<RecentItem[]>([]);

  const handleIdentified = useCallback((result: PredictResponse, prepared: PreparedImage) => {
    const item: RecentItem = {
      id: crypto.randomUUID(),
      thumbUrl: URL.createObjectURL(prepared.blob),
      commonName: result.top_prediction.common_name,
      confidence: result.top_prediction.confidence,
    };
    setRecents((prev) => {
      const combined = [item, ...prev];
      combined.slice(HISTORY_LIMIT).forEach((d) => URL.revokeObjectURL(d.thumbUrl));
      return combined.slice(0, HISTORY_LIMIT);
    });
  }, []);

  const p = usePredict({ topK: 5, onIdentified: handleIdentified });
  const hasImage = p.prepared !== null;

  return (
    <div className="flex min-h-dvh flex-col">
      <Header />

      <main className="mx-auto w-full max-w-3xl flex-1 px-5 py-4">
        {!hasImage ? (
          <div className="pt-4">
            <Dropzone onSelect={p.selectFile} disabled={p.status === "preparing"} />
            <RecentUploads items={recents} />
          </div>
        ) : (
          <div className="grid gap-5 lg:grid-cols-2 lg:gap-6">
            <div className="space-y-5 lg:self-start">
              {p.prepared && (
                <ImagePreview
                  previewUrl={p.prepared.previewUrl}
                  filename={p.prepared.filename}
                  sizeLabel={formatBytes(p.prepared.blob.size)}
                  downscaled={p.prepared.downscaled}
                  onReset={p.reset}
                />
              )}
              {p.status === "success" && p.result && (
                <OtherPredictions predictions={p.result.predictions.slice(1)} />
              )}
            </div>

            <div>
              {p.status === "loading" && <LoadingState />}
              {p.status === "error" && p.error && (
                <ErrorState code={p.error.code} message={p.error.message} onRetry={p.reset} />
              )}
              {p.status === "success" && p.result && <ResultsPanel result={p.result} />}
            </div>
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}
