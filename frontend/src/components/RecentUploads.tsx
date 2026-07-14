import { formatPct } from "./ConfidenceBar";

export interface RecentItem {
  id: string;
  thumbUrl: string;
  commonName: string;
  confidence: number;
}

/** In-memory history of this session's identifications. Not persisted; informational. */
export function RecentUploads({ items }: { items: RecentItem[] }) {
  if (items.length === 0) return null;
  return (
    <section aria-label="Recent identifications" className="mt-8">
      <h4 className="mb-3 px-1 text-xs font-medium tracking-wide text-muted uppercase">
        This session
      </h4>
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {items.map((item) => (
          <li
            key={item.id}
            className="overflow-hidden rounded-xl border border-border bg-surface"
          >
            <img
              src={item.thumbUrl}
              alt={`Earlier upload identified as ${item.commonName}`}
              className="aspect-square w-full object-cover"
              loading="lazy"
            />
            <div className="p-2">
              <p className="truncate text-xs font-medium">{item.commonName}</p>
              <p className="font-mono text-[11px] tabular-nums text-muted">
                {formatPct(item.confidence)}
              </p>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
