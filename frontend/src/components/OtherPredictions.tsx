import type { Prediction } from "../api/types";
import { ConfidenceBar, formatPct } from "./ConfidenceBar";

/** The runner-up predictions (everything below the best match). Rendered under the image so
 *  the two columns stay balanced. */
export function OtherPredictions({ predictions }: { predictions: Prediction[] }) {
  if (predictions.length === 0) return null;

  return (
    <div className="rounded-2xl border border-border bg-surface p-2 shadow-[var(--shadow)]">
      <h4 className="px-3 pt-2 pb-1 text-xs font-medium tracking-wide text-muted uppercase">
        Other possibilities
      </h4>
      <ul className="divide-y divide-border">
        {predictions.map((p) => (
          <li key={p.rank} className="flex items-center gap-4 px-3 py-3">
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-3">
                <span className="truncate font-medium">{p.common_name}</span>
                <span className="shrink-0 font-mono text-sm tabular-nums text-muted">
                  {formatPct(p.confidence)}
                </span>
              </div>
              {p.scientific_name && (
                <p className="truncate text-xs italic text-muted">{p.scientific_name}</p>
              )}
              <div className="mt-2">
                <ConfidenceBar value={p.confidence} />
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
