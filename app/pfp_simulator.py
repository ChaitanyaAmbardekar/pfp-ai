"""PFP simulation: square crop, circular crop, small-size previews and a
PFP Robustness Score. CPU-only (OpenCV + NumPy).

All thresholds in TH / WEIGHTS / SIZE_WEIGHTS are provisional starting
points. Calibrate them on real photos; they are not laws.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

SIZES = (64, 96, 128, 256)
CROP_FACTOR = 2.4     # square side = CROP_FACTOR x face-box height
CENTER_SHIFT = 0.10   # move crop centre down by this share of the side (shoulder room)
# framing -> (crop side as a multiple of face height, downward shift of the crop centre)
FRAMINGS = {
    "face":           (2.4, 0.10),
    "head_shoulders": (3.2, 0.14),
    "upper_body":     (4.5, 0.20),
}
CIRCLE_FULL = 512     # resolution of the full-size circle.png
DETECT_SIDE = 256     # small versions are viewed at this size for re-detection

# ramp(value, zero, full): 0 at `zero`, 1 at `full`, linear between.
TH = {
    "face_px": (10, 20),        # face width in px at the display size
    "eye_px": (4, 8),           # distance between eyes in px
    "circle": (1.10, 0.85),     # farthest face-box corner / circle radius
    "shadows": (0.50, 0.15),    # share of near-black pixels on the face
    "resolution": (0.35, 1.0),  # source crop px / display px
}
WEIGHTS = {"size": 0.25, "eyes": 0.15, "detect": 0.20,
           "circle": 0.15, "shadows": 0.10, "resolution": 0.15}
SIZE_WEIGHTS = {64: 0.4, 96: 0.2, 128: 0.2, 256: 0.2}  # 64 px is the stress test


@dataclass
class SquareCrop:
    x: int
    y: int
    side: int
    mode: str        # "face" or "center"
    limited: bool    # True if the image was too small for the ideal crop


@dataclass
class SizeCheck:
    size: int
    face_px: float
    eye_px: float
    detected: Optional[bool]   # None when no detector was supplied
    det_conf: float
    shadows: float
    score: float
    parts: dict = field(default_factory=dict)


@dataclass
class PfpReport:
    crop: SquareCrop
    face_fraction: float       # face width / crop side
    circle_overflow: float     # <1 inside the circle, >1 outside
    checks: dict               # size -> SizeCheck
    robustness: float          # 0..100
    flags: list = field(default_factory=list)


def _ramp(v, zero, full):
    return float(np.clip((v - zero) / (full - zero), 0.0, 1.0))


def _resize(img, size):
    interp = cv2.INTER_AREA if img.shape[0] >= size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def plan_square_crop(img_w, img_h, face, mode="face") -> SquareCrop:
    """Decide where the square crop goes. Falls back to center if no face."""
    if mode == "center" or face is None:
        side = min(img_w, img_h)
        return SquareCrop((img_w - side) // 2, (img_h - side) // 2, side, "center", False)
    factor, shift = FRAMINGS[mode]
    desired = int(round(factor * face.h))
    side = max(8, min(desired, img_w, img_h))
    limited = desired > min(img_w, img_h)   # photo has no room for this framing
    cx, cy = face.center
    cy += shift * side
    x = max(0, min(int(round(cx - side / 2)), img_w - side))
    y = max(0, min(int(round(cy - side / 2)), img_h - side))
    return SquareCrop(x, y, side, mode, limited)


def crop_square(bgr, plan: SquareCrop):
    return bgr[plan.y:plan.y + plan.side, plan.x:plan.x + plan.side]


_mask_cache = {}


def circle_mask(size):
    """Anti-aliased circular alpha mask (drawn at 4x, then shrunk)."""
    if size not in _mask_cache:
        big = size * 4
        m = np.zeros((big, big), np.uint8)
        cv2.circle(m, (big // 2, big // 2), big // 2, 255, -1, cv2.LINE_AA)
        _mask_cache[size] = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
    return _mask_cache[size]


def to_circle(square_bgr, size):
    """Square BGR -> circular BGRA of the given size (transparent corners)."""
    out = cv2.cvtColor(_resize(square_bgr, size), cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = circle_mask(size)
    return out


def contact_sheet(circles, bg=235, pad=24):
    """Place the circular previews side by side at their real pixel sizes."""
    sizes = sorted(circles)
    H = max(sizes) + 2 * pad
    W = sum(sizes) + pad * (len(sizes) + 1)
    sheet = np.full((H, W, 3), bg, np.uint8)
    x = pad
    for s in sizes:
        c = circles[s]
        y = (H - s) // 2
        a = c[:, :, 3:4].astype(np.float32) / 255.0
        roi = sheet[y:y + s, x:x + s].astype(np.float32)
        sheet[y:y + s, x:x + s] = (c[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)
        cv2.putText(sheet, f"{s}px", (x, y + s + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (90, 90, 90), 1, cv2.LINE_AA)
        x += s + pad
    return sheet


def evaluate(square, plan: SquareCrop, face, detector=None) -> PfpReport:
    """Robustness checks for one crop. `detector` is optional (a FaceDetector)."""
    if face is None:
        return PfpReport(plan, 0.0, 9.99, {}, 0.0, ["no_face_detected"])

    S = plan.side
    fx, fy, fw, fh = face.x - plan.x, face.y - plan.y, face.w, face.h
    face_fraction = fw / S

    # Is the head inside the circle? Use the farthest corner of the face box.
    mid = S / 2
    corners = [(fx, fy), (fx + fw, fy), (fx, fy + fh), (fx + fw, fy + fh)]
    overflow = max(math.hypot(px - mid, py - mid) for px, py in corners) / mid

    (rx, ry), (lx, ly) = face.landmarks[0], face.landmarks[1]
    eye_src = math.hypot(lx - rx, ly - ry)

    checks = {}
    for size in SIZES:
        k = size / S
        face_px, eye_px = fw * k, eye_src * k
        small = _resize(square, size)

        # Share of crushed (near-black) pixels inside the face box at this size.
        x0, y0 = max(0, int(fx * k)), max(0, int(fy * k))
        x1, y1 = min(size, int((fx + fw) * k)), min(size, int((fy + fh) * k))
        shadows = 0.0
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            g = cv2.cvtColor(small[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            shadows = float((g < 20).mean())

        # Re-detect the face the way a viewer sees it: shrink, then enlarge.
        detected, conf = None, 0.0
        if detector is not None:
            view = small if size == DETECT_SIDE else cv2.resize(
                small, (DETECT_SIDE, DETECT_SIDE), interpolation=cv2.INTER_CUBIC)
            kv = DETECT_SIDE / S
            ex0, ey0 = (fx - 0.2 * fw) * kv, (fy - 0.2 * fh) * kv
            ex1, ey1 = (fx + 1.2 * fw) * kv, (fy + 1.2 * fh) * kv
            detected = False
            for f in detector.detect(view).faces:
                cx, cy = f.center
                if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
                    detected, conf = True, max(conf, f.score)

        parts = {
            "size": _ramp(face_px, *TH["face_px"]),
            "eyes": _ramp(eye_px, *TH["eye_px"]),
            "circle": _ramp(overflow, *TH["circle"]),
            "shadows": _ramp(shadows, *TH["shadows"]),
            "resolution": _ramp(S / size, *TH["resolution"]),
        }
        if detected is not None:
            parts["detect"] = conf if detected else 0.0
        score = 100 * sum(WEIGHTS[n] * v for n, v in parts.items()) \
            / sum(WEIGHTS[n] for n in parts)
        checks[size] = SizeCheck(size, face_px, eye_px, detected, conf, shadows, score, parts)

    robustness = sum(SIZE_WEIGHTS[s] * checks[s].score for s in SIZES)

    flags, c64 = [], checks[64]
    if c64.face_px < 14: flags.append("face_small_at_64px")
    if c64.eye_px < 6: flags.append("eyes_unclear_at_64px")
    if c64.detected is False: flags.append("face_not_redetected_at_64px")
    if overflow > 1.0: flags.append("head_outside_circle")
    elif overflow > 0.85: flags.append("head_near_circle_edge")
    if face_fraction > 0.60: flags.append("face_too_large_in_crop")
    if plan.limited: flags.append("image_too_small_for_ideal_crop")
    if S < 200: flags.append("crop_low_resolution")
    if c64.shadows > 0.30: flags.append("face_shadows_crushed")
    return PfpReport(plan, face_fraction, overflow, checks, robustness, flags)
