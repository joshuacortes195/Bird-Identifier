"""Image loading that Just Works for both dataset images and phone uploads.

Applies EXIF orientation and registers the HEIF/HEIC opener (iPhone photos) so the
exact same loader serves training data and the Phase 7 OOD / Phase 9 API uploads.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

# Decompression-bomb guard: a tiny crafted file (e.g. a PNG) can declare enormous pixel
# dimensions and blow up memory on decode — dangerous on a 512 MB free instance, and it
# happens *before* any server-side downscale. Cap total pixels so Pillow raises
# DecompressionBombError well before that. 24 MP comfortably covers real phone photos.
Image.MAX_IMAGE_PIXELS = 24_000_000

_HEIF_REGISTERED = False


def _ensure_heif() -> None:
    global _HEIF_REGISTERED
    if not _HEIF_REGISTERED:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            pass  # HEIC just won't be supported; JPEG/PNG still work.
        _HEIF_REGISTERED = True


def load_image(path: str | Path) -> Image.Image:
    """Open an image as RGB with EXIF orientation applied."""
    _ensure_heif()
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Same as :func:`load_image` but from an in-memory buffer (API uploads)."""
    import io

    _ensure_heif()
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def crop_to_bbox(img: Image.Image, bbox: tuple[int, int, int, int], pad_frac: float = 0.1) -> Image.Image:
    """Crop to (x, y, w, h) with symmetric padding, clamped to image bounds.

    Cropping to the subject is the single biggest lever for fine-grained accuracy;
    the padding keeps context (and tolerates loose boxes).
    """
    x, y, w, h = bbox
    W, H = img.size
    px, py = int(w * pad_frac), int(h * pad_frac)
    left = max(0, x - px)
    top = max(0, y - py)
    right = min(W, x + w + px)
    bottom = min(H, y + h + py)
    if right <= left or bottom <= top:
        return img  # degenerate box -> no crop
    return img.crop((left, top, right, bottom))
