"""Safe image loading for PFP-AI."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}


def load_image_bgr(path, max_side: int = 1280):
    """Load an image as a BGR uint8 array (OpenCV format).

    - Applies EXIF rotation so phone photos are upright.
    - Downsizes so the longest side is at most `max_side` (saves RAM).
    Returns (bgr_array, (original_width, original_height)).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Pillow handles Windows/unicode paths and EXIF; cv2.imread does not always.
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        original_size = im.size
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        rgb = np.asarray(im)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), original_size
try:  # optional: lets Pillow open iPhone-style HEIC files
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass