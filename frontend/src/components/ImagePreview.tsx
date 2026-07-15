import { UploadIcon } from "../icons";

interface ImagePreviewProps {
  previewUrl: string;
  filename: string;
  sizeLabel: string;
  downscaled: boolean;
  onReset: () => void;
}

export function ImagePreview({
  previewUrl,
  filename,
  sizeLabel,
  downscaled,
  onReset,
}: ImagePreviewProps) {
  return (
    <div className="rounded-2xl border border-border bg-surface p-3 shadow-[var(--shadow)]">
      <div className="overflow-hidden rounded-xl bg-black">
        <img
          src={previewUrl}
          alt="Your uploaded bird photo"
          className="mx-auto max-h-[42vh] w-full object-contain"
        />
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 px-1">
        <p className="min-w-0 truncate text-xs text-muted">
          <span className="text-fg">{filename}</span> · {sizeLabel}
          {downscaled && " · downscaled for upload"}
        </p>

        <button
          type="button"
          onClick={onReset}
          className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-primary px-3 text-sm font-medium text-primary-fg transition-colors hover:bg-primary-hover"
        >
          <UploadIcon width={16} height={16} />
          New photo
        </button>
      </div>
    </div>
  );
}
