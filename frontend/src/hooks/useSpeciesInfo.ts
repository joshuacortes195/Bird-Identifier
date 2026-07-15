import { useEffect, useState } from "react";
import { fetchSpeciesInfo, type SpeciesInfo } from "../lib/species";

type State =
  | { status: "idle" | "loading" | "error" }
  | { status: "success"; info: SpeciesInfo };

/** Fetch reference info for a predicted species. Re-runs (and aborts the prior request)
 *  whenever the name changes, so switching photos never shows stale info. */
export function useSpeciesInfo(commonName: string | null): State {
  const [state, setState] = useState<State>({ status: "idle" });

  useEffect(() => {
    if (!commonName) {
      setState({ status: "idle" });
      return;
    }
    const ctrl = new AbortController();
    setState({ status: "loading" });
    fetchSpeciesInfo(commonName, ctrl.signal)
      .then((info) => {
        if (ctrl.signal.aborted) return;
        setState(info ? { status: "success", info } : { status: "error" });
      })
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setState({ status: "error" });
      });
    return () => ctrl.abort();
  }, [commonName]);

  return state;
}
