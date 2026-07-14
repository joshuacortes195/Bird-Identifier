import { useState } from "react";
import { EyeIcon, SpinnerIcon, XIcon } from "../icons";

interface ImagePreviewProps {
  previewUrl: string;
  filename: string;
  sizeLabel: string;
  downscaled: boolean;
  /** Grad-CAM overlay PNG (base64, no data: prefix) or null if the backend didn't return one. */
  gradcamPng: string | null;
  gradcamRequested: boolean;
  gradcamLoading: boolean;
  gradcamSupported: boolean;
  onRequestGradcam: () => void;
  onReset: () => void;
}

export function ImagePreview({
  previewUrl,
  filename,
  sizeLabel,
  downscaled,
  gradcamPng,
  gradcamRequested,
  gradcamLoading,
  gradcamSupported,
  onRequestGradcam,
  onReset,
}: ImagePreviewProps) {
  const [overlayOn, setOverlayOn] = useState(true);
  const [opacity, setOpacity] = useState(0.6);

  const hasOverlay = Boolean(gradcamPng);
  const showUnavailable = gradcamRequested && !gradcamLoading && !gradcamPng;

  function toggle() {
    if (!hasOverlay) {
      onRequestGradcam();
      setOverlayOn(true);
    } else {
      setOverlayOn((v) => !v);
    }
  }

  return (
    <div className="rounded-2xl border border-border bg-surface p-3 shadow-[var(--shadow)]">
      <div className="relative overflow-hidden rounded-xl bg-black">
        <img
          src={previewUrl}
          alt="Your uploaded bird photo"
          className="mx-auto max-h-[42vh] w-full object-contain"
        />
        {hasOverlay && overlayOn && (
          <img
            src={`data:image/png;base64,${gradcamPng}`}
            alt="Model attention heatmap overlay"
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 mx-auto max-h-[42vh] w-full object-contain transition-opacity"
            style={{ opacity }}
          />
        )}
        <button
          type="button"
          onClick={onReset}
          aria-label="Remove photo and start over"
          className="absolute top-2 right-2 inline-flex h-9 w-9 items-center justify-center rounded-lg bg-black/55 text-white backdrop-blur transition-colors hover:bg-black/75"
        >
          <XIcon width={18} height={18} />
        </button>
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 px-1">
        <p className="min-w-0 truncate text-xs text-muted">
          <span className="text-fg">{filename}</span> · {sizeLabel}
          {downscaled && " · downscaled for upload"}
        </p>

        {gradcamSupported && (
          <button
            type="button"
            onClick={toggle}
            aria-pressed={hasOverlay && overlayOn}
            disabled={gradcamLoading}
            className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-border bg-surface px-3 text-sm font-medium transition-colors hover:bg-surface-2 disabled:opacity-60"
          >
            {gradcamLoading ? (
              <SpinnerIcon width={16} height={16} />
            ) : (
              <EyeIcon width={16} height={16} />
            )}
            {hasOverlay ? (overlayOn ? "Hide attention" : "Show attention") : "What did it see?"}
          </button>
        )}
      </div>

      {hasOverlay && overlayOn && (
        <div className="mt-2 flex items-center gap-3 px-1">
          <label htmlFor="overlay-opacity" className="text-xs text-muted">
            Overlay
          </label>
          <input
            id="overlay-opacity"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="h-1 flex-1 cursor-pointer accent-[var(--primary)]"
          />
        </div>
      )}

      {showUnavailable && (
        <p className="mt-2 px-1 text-xs text-muted">
          Attention overlay isn't available from the current model backend.
        </p>
      )}
    </div>
  );
}
