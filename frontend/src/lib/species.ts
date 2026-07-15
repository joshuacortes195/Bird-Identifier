/** Live species enrichment from Wikipedia + Wikidata — real, cited data, never fabricated.
 *
 *  The classifier only outputs a species *name*. Habitat/range summary and IUCN conservation
 *  status are fetched at prediction time so the "Learn more" link points at the actual source.
 *  Both endpoints are public and CORS-enabled, so this works from a static (serverless) deploy.
 */

export type IucnCode = "LC" | "NT" | "VU" | "EN" | "CR" | "EW" | "EX" | "DD" | "NE";

export interface Conservation {
  code: IucnCode;
  label: string;
  /** Position on the LC→EX threat spectrum (0..6), or null for off-spectrum (DD/NE). */
  spectrumIndex: number | null;
}

export interface SpeciesInfo {
  title: string;
  extract: string;
  /** Wikipedia article — the cited source shown behind "Learn more". */
  url: string;
  thumbnail: string | null;
  conservation: Conservation | null;
}

// Wikidata Q-ids for IUCN status (property P141), verified against live Wikidata.
const IUCN_BY_QID: Record<string, Conservation> = {
  Q211005: { code: "LC", label: "Least Concern", spectrumIndex: 0 },
  Q719675: { code: "NT", label: "Near Threatened", spectrumIndex: 1 },
  Q278113: { code: "VU", label: "Vulnerable", spectrumIndex: 2 },
  Q11394: { code: "EN", label: "Endangered", spectrumIndex: 3 },
  Q219127: { code: "CR", label: "Critically Endangered", spectrumIndex: 4 },
  Q239509: { code: "EW", label: "Extinct in the Wild", spectrumIndex: 5 },
  Q237350: { code: "EX", label: "Extinct", spectrumIndex: 6 },
  Q3245245: { code: "DD", label: "Data Deficient", spectrumIndex: null },
  Q3350324: { code: "NE", label: "Not Evaluated", spectrumIndex: null },
};

/** The threat spectrum, least → most threatened. Drives the conservation bar. */
export const SPECTRUM: { code: IucnCode; label: string }[] = [
  { code: "LC", label: "Least Concern" },
  { code: "NT", label: "Near Threatened" },
  { code: "VU", label: "Vulnerable" },
  { code: "EN", label: "Endangered" },
  { code: "CR", label: "Critically Endangered" },
  { code: "EW", label: "Extinct in the Wild" },
  { code: "EX", label: "Extinct" },
];

/** NABirds labels carry plumage/age qualifiers, e.g. "Bald Eagle (Adult, subadult)".
 *  Strip parentheticals + stray whitespace to get a Wikipedia-searchable species name. */
export function baseSpeciesName(name: string): string {
  return name
    .replace(/\([^)]*\)/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

interface WikiSummary {
  type?: string;
  title: string;
  extract?: string;
  thumbnail?: { source?: string };
  content_urls?: { desktop?: { page?: string } };
  wikibase_item?: string;
}

async function fetchSummary(title: string, signal?: AbortSignal): Promise<WikiSummary | null> {
  const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}?redirect=true`;
  const res = await fetch(url, { signal, headers: { accept: "application/json" } });
  if (!res.ok) return null;
  return (await res.json()) as WikiSummary;
}

async function fetchConservation(qid: string, signal?: AbortSignal): Promise<Conservation | null> {
  const url = `https://www.wikidata.org/w/api.php?action=wbgetclaims&entity=${qid}&property=P141&format=json&origin=*`;
  const res = await fetch(url, { signal });
  if (!res.ok) return null;
  const data = (await res.json()) as {
    claims?: { P141?: { mainsnak?: { datavalue?: { value?: { id?: string } } } }[] };
  };
  const valueId = data.claims?.P141?.[0]?.mainsnak?.datavalue?.value?.id;
  return valueId ? (IUCN_BY_QID[valueId] ?? null) : null;
}

/** Resolve a predicted common name to a Wikipedia summary + IUCN status. Returns null when
 *  nothing reliable is found — the UI then shows "unavailable" rather than anything invented. */
export async function fetchSpeciesInfo(
  commonName: string,
  signal?: AbortSignal,
): Promise<SpeciesInfo | null> {
  const base = baseSpeciesName(commonName);
  if (!base) return null;

  let summary = await fetchSummary(base, signal);
  // Disambiguation page or a miss → retry biased toward the bird sense.
  if (!summary || summary.type === "disambiguation" || !summary.extract) {
    summary = (await fetchSummary(`${base} (bird)`, signal)) ?? summary;
  }
  if (!summary || !summary.extract) return null;

  const url =
    summary.content_urls?.desktop?.page ??
    `https://en.wikipedia.org/wiki/${encodeURIComponent(summary.title)}`;

  let conservation: Conservation | null = null;
  if (summary.wikibase_item) {
    try {
      conservation = await fetchConservation(summary.wikibase_item, signal);
    } catch {
      conservation = null; // status is a nice-to-have; never block the summary on it.
    }
  }

  return {
    title: summary.title,
    extract: summary.extract,
    url,
    thumbnail: summary.thumbnail?.source ?? null,
    conservation,
  };
}
