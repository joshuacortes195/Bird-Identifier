import { useEffect, useState } from "react";
import { AlertIcon, SpinnerIcon } from "../icons";

const EXPECTED_SECONDS = 30;

/** Waiting screen shown while the model runs. The API is hosted on a free CPU tier, so a
 *  request takes ~30 s (and longer on a cold start), so we set expectations clearly and keep
 *  a live timer + progress bar so it never feels frozen. */
export function LoadingState() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);

  // Ease toward ~92% over the expected window; never hit 100% until the result actually lands.
  const pct = Math.min(92, (elapsed / EXPECTED_SECONDS) * 92);
  const overtime = elapsed >= EXPECTED_SECONDS;

  return (
    <div
      role="status"
      aria-live="polite"
      className="rounded-2xl border border-border bg-surface p-6 text-center shadow-[var(--shadow)]"
    >
      <span className="mx-auto mb-4 inline-flex h-12 w-12 items-center justify-center rounded-full bg-surface-2 text-primary">
        <SpinnerIcon width={24} height={24} />
      </span>
      <h3 className="font-serif text-lg font-semibold tracking-tight">Identifying your bird…</h3>
      <p className="mx-auto mt-2 max-w-sm text-sm text-muted">
        {overtime
          ? "Almost there — the free server may be waking up from sleep. Hang tight, this only happens on the first request."
          : "This is a large, powerful AI model running on a free hosting plan, so it can take about 30 seconds. Please stand by."}
      </p>

      <div className="mx-auto mt-5 max-w-xs">
        <div className="h-2 overflow-hidden rounded-full bg-surface-2">
          <div
            className="h-full rounded-full bg-primary transition-all duration-1000 ease-linear"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="mt-2 font-mono text-xs tabular-nums text-muted">{elapsed}s</p>
      </div>
    </div>
  );
}

interface ErrorStateProps {
  code: string;
  message: string;
  onRetry?: () => void;
}

export function ErrorState({ code, message, onRetry }: ErrorStateProps) {
  return (
    <div
      role="alert"
      className="rounded-2xl border border-border bg-surface p-5 text-center shadow-[var(--shadow)]"
    >
      <span className="mx-auto mb-3 inline-flex h-12 w-12 items-center justify-center rounded-full bg-surface-2 text-danger">
        <AlertIcon width={22} height={22} />
      </span>
      <h3 className="font-medium">Couldn't identify that</h3>
      <p className="mx-auto mt-1 max-w-sm text-sm text-muted">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 inline-flex h-11 items-center justify-center rounded-xl bg-primary px-5 font-medium text-primary-fg transition-colors hover:bg-primary-hover"
        >
          Try another photo
        </button>
      )}
      <p className="mt-3 font-mono text-[11px] text-muted">{code}</p>
    </div>
  );
}
