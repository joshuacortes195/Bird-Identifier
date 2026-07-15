import { formatPct } from "./ConfidenceBar";

export interface RecentItem {
  id: string;
  thumbUrl: string;
  commonName: string;
  confidence: number;
  /** The prepared image, kept so the thumbnail can re-run identification on click. */
  file: File;
}

/** In-memory history of this session's identifications. Click a thumbnail to re-identify it. */
export function RecentUploads({
  items,
  onSelect,
}: {
  items: RecentItem[];
  onSelect: (file: File) => void;
}) {
  if (items.length === 0) return null;
  return (
    <section aria-label="Recent identifications" className="mt-8">
      <h4 className="mb-3 px-1 text-xs font-medium tracking-wide text-muted uppercase">
        This session
      </h4>
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {items.map((item) => (
          <li key={item.id}>
            <button
              type="button"
              onClick={() => onSelect(item.file)}
              title={`Re-identify ${item.commonName}`}
              className="block w-full overflow-hidden rounded-xl border border-border bg-surface text-left transition-colors hover:border-primary focus-visible:border-primary focus-visible:outline-none"
            >
              <img
                src={item.thumbUrl}
                alt={`Earlier upload identified as ${item.commonName}. Click to re-identify.`}
                className="aspect-square w-full object-cover"
                loading="lazy"
              />
              <div className="p-2">
                <p className="truncate text-xs font-medium">{item.commonName}</p>
                <p className="font-mono text-[11px] tabular-nums text-muted">
                  {formatPct(item.confidence)}
                </p>
              </div>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
