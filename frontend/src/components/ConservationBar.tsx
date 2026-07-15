import { SPECTRUM, type Conservation } from "../lib/species";

// Semantic status colors (green → red → near-black). Fixed hex so the meaning reads the same
// in light and dark themes; the inactive segments are dimmed via opacity.
const SEGMENT_COLOR = [
  "#2E9E5B", // LC
  "#8DB600", // NT
  "#E4B400", // VU
  "#F08C00", // EN
  "#E03131", // CR
  "#9C1F1F", // EW
  "#3A3A3A", // EX
];

export function ConservationBar({ conservation }: { conservation: Conservation }) {
  const active = conservation.spectrumIndex; // null for DD/NE

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <span className="text-xs font-medium tracking-wide text-muted uppercase">
          Conservation status
        </span>
        <span className="text-sm font-medium text-fg">
          {conservation.label} ({conservation.code})
        </span>
      </div>

      <div
        className="flex gap-1"
        role="img"
        aria-label={`IUCN Red List status: ${conservation.label}`}
      >
        {SPECTRUM.map((s, i) => {
          const on = active === i;
          return (
            <div key={s.code} className="min-w-0 flex-1">
              <div
                className="h-2.5 rounded-full transition-opacity"
                style={{
                  backgroundColor: SEGMENT_COLOR[i],
                  opacity: active === null ? 0.45 : on ? 1 : 0.24,
                }}
              />
              <div
                className={`mt-1 text-center text-[10px] tabular-nums ${
                  on ? "font-semibold text-fg" : "text-muted"
                }`}
              >
                {s.code}
              </div>
            </div>
          );
        })}
      </div>

      {active === null && (
        <p className="mt-1 text-xs text-muted">Not placed on the threat spectrum.</p>
      )}
    </div>
  );
}
