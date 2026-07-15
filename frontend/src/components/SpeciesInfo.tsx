import { useSpeciesInfo } from "../hooks/useSpeciesInfo";
import { ExternalLinkIcon, SpinnerIcon } from "../icons";
import { ConservationBar } from "./ConservationBar";

/** Reference card for the top prediction: Wikipedia summary, IUCN conservation bar, and a
 *  "Learn more" link to the cited source. All data is fetched live — nothing is invented. */
export function SpeciesInfo({ commonName }: { commonName: string }) {
  const state = useSpeciesInfo(commonName);

  if (state.status === "loading") {
    return (
      <div className="flex items-center gap-2 rounded-2xl border border-border bg-surface p-5 text-sm text-muted shadow-[var(--shadow)]">
        <SpinnerIcon width={16} height={16} />
        Looking up species info…
      </div>
    );
  }

  if (state.status !== "success") {
    return (
      <div className="rounded-2xl border border-border bg-surface p-5 text-sm text-muted shadow-[var(--shadow)]">
        No verified reference info found for this species.
      </div>
    );
  }

  const { info } = state;

  return (
    <div className="space-y-4 rounded-2xl border border-border bg-surface p-5 shadow-[var(--shadow)]">
      <p className="text-sm leading-relaxed text-muted">{info.extract}</p>

      {info.conservation && <ConservationBar conservation={info.conservation} />}

      <div className="flex flex-wrap items-center justify-between gap-2">
        <a
          href={info.url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
        >
          Learn more on Wikipedia
          <ExternalLinkIcon width={15} height={15} />
        </a>
        <span className="text-[11px] text-muted">Source: Wikipedia / Wikidata</span>
      </div>
    </div>
  );
}
