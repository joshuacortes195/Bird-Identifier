import { useId, useRef, useState } from "react";
import { CameraIcon, UploadIcon } from "../icons";

interface DropzoneProps {
  onSelect: (file: File) => void;
  disabled?: boolean;
}

const ACCEPTED = "image/*,.heic,.heif";

export function Dropzone({ onSelect, disabled }: DropzoneProps) {
  const fileInput = useRef<HTMLInputElement>(null);
  const cameraInput = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [hint, setHint] = useState<string | null>(null);
  const descId = useId();

  function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/") && !/\.(heic|heif)$/i.test(file.name)) {
      setHint("That doesn't look like an image. Try a JPEG, PNG, WEBP, or HEIC.");
      return;
    }
    setHint(null);
    onSelect(file);
  }

  return (
    <section
      aria-label="Upload a bird photo"
      aria-describedby={descId}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      className={[
        "rounded-2xl border-2 border-dashed bg-surface p-8 text-center transition-colors sm:p-12",
        dragging ? "border-primary bg-surface-2" : "border-border",
        disabled ? "opacity-60" : "",
      ].join(" ")}
    >
      <span className="mx-auto mb-5 inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-surface-2 text-primary">
        <UploadIcon width={26} height={26} />
      </span>
      <h2 className="font-serif text-2xl font-semibold tracking-tight">Identify a bird</h2>
      <p id={descId} className="mx-auto mt-2 max-w-sm text-sm text-muted">
        Drop a photo here, choose one from your device, or use your camera. Your image is sent
        only to the classifier and isn't stored.
      </p>

      <div className="mt-6 flex flex-col justify-center gap-3 sm:flex-row">
        <button
          type="button"
          disabled={disabled}
          onClick={() => fileInput.current?.click()}
          className="inline-flex h-12 min-w-44 items-center justify-center gap-2 rounded-xl bg-primary px-5 font-medium text-primary-fg transition-colors hover:bg-primary-hover disabled:cursor-not-allowed"
        >
          <UploadIcon width={20} height={20} />
          Choose photo
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => cameraInput.current?.click()}
          className="inline-flex h-12 min-w-44 items-center justify-center gap-2 rounded-xl border border-border bg-surface px-5 font-medium text-fg transition-colors hover:bg-surface-2 disabled:cursor-not-allowed sm:hidden"
        >
          <CameraIcon width={20} height={20} />
          Take photo
        </button>
      </div>

      {hint && (
        <p role="alert" className="mt-4 text-sm text-danger">
          {hint}
        </p>
      )}

      <input
        ref={fileInput}
        type="file"
        accept={ACCEPTED}
        className="sr-only"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <input
        ref={cameraInput}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </section>
  );
}
