import type { PredictResponse } from "../api/types";
import { AlertIcon, CheckIcon } from "../icons";
import { ConfidenceBar, formatPct } from "./ConfidenceBar";
import { SpeciesInfo } from "./SpeciesInfo";

export function ResultsPanel({ result }: { result: PredictResponse }) {
  const top = result.predictions[0];

  return (
    <div className="space-y-4">
      {result.low_confidence && (
        <div
          role="status"
          className="flex items-start gap-3 rounded-xl border border-border bg-surface-2 p-3 text-sm"
        >
          <AlertIcon width={18} height={18} className="mt-0.5 shrink-0 text-warn" />
          <p className="text-muted">
            <span className="font-medium text-fg">Not very confident.</span> This bird may be
            outside the model's 555 species, or the photo is hard (distance, blur, pose). Treat the
            guesses below as a shortlist.
          </p>
        </div>
      )}

      {/* Top prediction — the headline */}
      <div className="rounded-2xl border border-border bg-surface p-5 shadow-[var(--shadow)]">
        <div className="mb-3 flex items-center justify-between">
          <span className="inline-flex items-center gap-1.5 text-xs font-medium tracking-wide text-primary uppercase">
            <CheckIcon width={14} height={14} />
            Best match
          </span>
          <span className="font-mono text-sm tabular-nums text-muted">{formatPct(top.confidence)}</span>
        </div>
        <h3 className="font-serif text-2xl leading-tight font-semibold tracking-tight">
          {top.common_name}
        </h3>
        {top.scientific_name && (
          <p className="mt-0.5 text-sm italic text-muted">{top.scientific_name}</p>
        )}
        <div className="mt-3">
          <ConfidenceBar value={top.confidence} emphasis />
        </div>
      </div>

      {/* Reference info for the top match — summary, conservation status, source link */}
      <SpeciesInfo commonName={top.common_name} />

      <p className="px-1 text-center text-xs text-muted">
        {result.model.name} · {result.model.backend} · {result.model.num_classes} species ·{" "}
        {result.inference_ms.toFixed(0)} ms
      </p>
    </div>
  );
}
