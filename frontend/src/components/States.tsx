import { AlertIcon, SpinnerIcon } from "../icons";

/** Skeleton shown while the model runs — reserves layout so there's no shift on arrival. */
export function LoadingState() {
  return (
    <div
      className="space-y-4"
      role="status"
      aria-live="polite"
      aria-label="Identifying the bird"
    >
      <div className="flex items-center gap-2 px-1 text-sm text-muted">
        <SpinnerIcon width={18} height={18} />
        Identifying…
      </div>
      <div className="rounded-2xl border border-border bg-surface p-5 shadow-[var(--shadow)]">
        <div className="mb-3 h-3 w-24 animate-pulse rounded bg-surface-2" />
        <div className="h-7 w-2/3 animate-pulse rounded bg-surface-2" />
        <div className="mt-2 h-4 w-1/3 animate-pulse rounded bg-surface-2" />
        <div className="mt-4 h-2 w-full animate-pulse rounded-full bg-surface-2" />
      </div>
      <div className="rounded-2xl border border-border bg-surface p-4 shadow-[var(--shadow)]">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="py-3">
            <div className="h-4 w-1/2 animate-pulse rounded bg-surface-2" />
            <div className="mt-2 h-2 w-full animate-pulse rounded-full bg-surface-2" />
          </div>
        ))}
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
