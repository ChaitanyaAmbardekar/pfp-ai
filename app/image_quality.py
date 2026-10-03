"""Objective image-quality metrics (CPU-only, no ML).

Every metric is computed on a grayscale (luminance) version of a region.
Labels use provisional thresholds in THRESH. Calibrate them on your own
photos, because they are starting points, not laws.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

THRESH = {
    "under_mean": 70,        # mean luminance below this -> underexposed
    "over_mean": 190,        # mean luminance above this -> overexposed
    "dark_pixel": 20,        # a pixel below this counts as crushed shadow
    "bright_pixel": 235,     # a pixel above this counts as blown highlight
    "dark_fraction": 0.40,   # too many crushed pixels -> underexposed
    "bright_fraction": 0.25, # too many blown pixels -> overexposed
    "sharp_very_soft": 20,   # Laplacian variance at the fixed size
    "sharp_soft": 60,
    "noise_low": 3.0,        # estimated noise sigma (0..255 scale)
    "noise_high": 8.0,
}

FACE_SIZE = 160    # face crop is resized to this before measuring sharpness
IMAGE_SIDE = 512   # full image long side before measuring sharpness
MIN_REGION = 16    # skip regions smaller than this (in pixels)


@dataclass
class RegionQuality:
    brightness: float        # mean luminance, 0..255
    contrast: float          # std of luminance, 0..~127
    local_contrast: float    # mean local std (7x7), detail/texture contrast
    dynamic_range: float     # 95th minus 5th percentile
    sharpness: float         # variance of Laplacian at a fixed size
    noise: float             # Immerkaer noise sigma estimate
    dark_fraction: float     # share of crushed-shadow pixels
    bright_fraction: float   # share of blown-highlight pixels
    exposure: str            # underexposed / overexposed / balanced
    sharpness_label: str     # very soft / soft / sharp
    noise_label: str         # low / moderate / high


@dataclass
class QualityReport:
    whole: RegionQuality
    face: Optional[RegionQuality]          # None if no face was detected
    face_minus_image_brightness: Optional[float]  # >0: face brighter than scene


def _to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _resize_to(gray, size=None, long_side=None):
    h, w = gray.shape
    if size is not None:
        interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC
        return cv2.resize(gray, (size, size), interpolation=interp)
    scale = long_side / max(h, w)
    if scale >= 1:
        return gray
    return cv2.resize(gray, (round(w * scale), round(h * scale)),
                      interpolation=cv2.INTER_AREA)


def laplacian_variance(gray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_noise(gray) -> float:
    """Immerkaer (1996) fast noise-sigma estimate from one image."""
    g = gray.astype(np.float32)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(g, -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) / (6.0 * (w - 2) * (h - 2)) * np.abs(conv).sum())


def _local_contrast(gray) -> float:
    g = gray.astype(np.float32)
    mu = cv2.blur(g, (7, 7))
    mu2 = cv2.blur(g * g, (7, 7))
    return float(np.sqrt(np.maximum(mu2 - mu * mu, 0)).mean())


def _exposure_label(mean, dark_frac, bright_frac) -> str:
    if mean < THRESH["under_mean"] or dark_frac > THRESH["dark_fraction"]:
        return "underexposed"
    if mean > THRESH["over_mean"] or bright_frac > THRESH["bright_fraction"]:
        return "overexposed"
    return "balanced"


def analyze_region(gray, is_face: bool = False) -> RegionQuality:
    """Compute all metrics for one grayscale (uint8) region."""
    mean = float(gray.mean())
    std = float(gray.std())
    p5, p95 = np.percentile(gray, [5, 95])
    dark = float((gray < THRESH["dark_pixel"]).mean())
    bright = float((gray > THRESH["bright_pixel"]).mean())

    fixed = _resize_to(gray, size=FACE_SIZE) if is_face else \
        _resize_to(gray, long_side=IMAGE_SIDE)
    sharp = laplacian_variance(fixed)
    noise = estimate_noise(gray)

    if sharp < THRESH["sharp_very_soft"]:
        s_label = "very soft"
    elif sharp < THRESH["sharp_soft"]:
        s_label = "soft"
    else:
        s_label = "sharp"

    if noise < THRESH["noise_low"]:
        n_label = "low"
    elif noise < THRESH["noise_high"]:
        n_label = "moderate"
    else:
        n_label = "high"

    return RegionQuality(
        brightness=mean, contrast=std,
        local_contrast=_local_contrast(gray),
        dynamic_range=float(p95 - p5),
        sharpness=sharp, noise=noise,
        dark_fraction=dark, bright_fraction=bright,
        exposure=_exposure_label(mean, dark, bright),
        sharpness_label=s_label, noise_label=n_label)


def face_crop(bgr, face, pad: float = 0.10):
    """Crop the face box with a little padding, clipped to the image."""
    h, w = bgr.shape[:2]
    px, py = int(face.w * pad), int(face.h * pad)
    x0, y0 = max(0, face.x - px), max(0, face.y - py)
    x1, y1 = min(w, face.x + face.w + px), min(h, face.y + face.h + py)
    return bgr[y0:y1, x0:x1]


def analyze_quality(bgr, detection) -> QualityReport:
    """Whole-image and primary-face quality metrics."""
    whole = analyze_region(_to_gray(bgr), is_face=False)

    face_q = None
    diff = None
    f = detection.primary
    if f is not None:
        crop = face_crop(bgr, f)
        if min(crop.shape[:2]) >= MIN_REGION:
            face_q = analyze_region(_to_gray(crop), is_face=True)
            diff = face_q.brightness - whole.brightness
    return QualityReport(whole, face_q, diff)
