/** Client-side image prep: downscale + JPEG-compress large photos before upload so
 *  mobile uploads are fast and cheap on cellular. HEIC (and anything the browser can't
 *  decode) is passed through untouched — the API decodes those server-side. */

export interface PreparedImage {
  /** The blob to upload (compressed JPEG, or the original if we couldn't/needn't touch it). */
  blob: Blob;
  /** Object URL for a preview thumbnail. Caller must revokeObjectURL when done. */
  previewUrl: string;
  /** Filename to send. */
  filename: string;
  downscaled: boolean;
}

const MAX_DIMENSION = 1600; // longest side after downscale
const JPEG_QUALITY = 0.85;
const HEIC_RE = /\.(heic|heif)$/i;

function isProbablyHeic(file: File): boolean {
  return HEIC_RE.test(file.name) || file.type === "image/heic" || file.type === "image/heif";
}

async function loadBitmap(file: File): Promise<ImageBitmap | null> {
  try {
    return await createImageBitmap(file);
  } catch {
    return null; // e.g. HEIC on a non-Safari browser
  }
}

export async function prepareImage(file: File): Promise<PreparedImage> {
  // Don't try to canvas-encode HEIC; the browser usually can't decode it. Ship as-is.
  if (isProbablyHeic(file)) {
    return {
      blob: file,
      previewUrl: URL.createObjectURL(file),
      filename: file.name,
      downscaled: false,
    };
  }

  const bitmap = await loadBitmap(file);
  if (!bitmap) {
    return {
      blob: file,
      previewUrl: URL.createObjectURL(file),
      filename: file.name,
      downscaled: false,
    };
  }

  const { width, height } = bitmap;
  const longest = Math.max(width, height);
  const scale = longest > MAX_DIMENSION ? MAX_DIMENSION / longest : 1;
  const targetW = Math.max(1, Math.round(width * scale));
  const targetH = Math.max(1, Math.round(height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    bitmap.close();
    return {
      blob: file,
      previewUrl: URL.createObjectURL(file),
      filename: file.name,
      downscaled: false,
    };
  }
  ctx.drawImage(bitmap, 0, 0, targetW, targetH);
  bitmap.close();

  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY),
  );

  // If compression somehow produced a larger blob than the original, keep the original.
  if (!blob || blob.size >= file.size) {
    return {
      blob: file,
      previewUrl: URL.createObjectURL(file),
      filename: file.name,
      downscaled: false,
    };
  }

  const base = file.name.replace(/\.[^.]+$/, "");
  return {
    blob,
    previewUrl: URL.createObjectURL(blob),
    filename: `${base}.jpg`,
    downscaled: scale < 1,
  };
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
