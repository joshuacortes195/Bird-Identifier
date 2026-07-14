interface ConfidenceBarProps {
  value: number; // 0..1
  emphasis?: boolean;
}

/** A slim meter. The numeric label carries the value for screen readers; the bar is
 *  decorative reinforcement (never the only signal). */
export function ConfidenceBar({ value, emphasis }: ConfidenceBarProps) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2" aria-hidden="true">
      <div
        className="h-full origin-left rounded-full"
        style={{
          width: `${pct}%`,
          backgroundColor: emphasis ? "var(--primary)" : "var(--muted)",
          animation: "grow-in 350ms ease-out",
        }}
      />
    </div>
  );
}

export function formatPct(value: number): string {
  const pct = value * 100;
  if (pct < 1) return "<1%";
  return `${Math.round(pct)}%`;
}
