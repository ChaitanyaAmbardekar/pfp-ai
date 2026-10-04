"""Composition analysis for a photo or a candidate crop (pure Python, CPU-only).

Everything is measured relative to a region (x, y, w, h). Thresholds are
provisional; calibrate them on real photos.
"""
from dataclasses import dataclass, field

import numpy as np

HAIR = 0.25                  # YuNet's box starts near the brows; hair sits above it
TARGET_HEADROOM = 0.35       # face heights above the head that we aim for
TARGET_FACE_H = 0.32         # face height / crop height that we aim for
TH = {
    "headroom": (0.15, 0.90),        # below = too little, above = excessive
    "face_h": (0.12, 0.45, 0.70),    # too_small / large / excessively_cropped
    "edge": 0.02,                    # face this close to a region edge = touching it
    "center": (0.40, 0.60),
    "vertical": (0.30, 0.60),
    "source_face_px": (30, 120),     # face width in ORIGINAL pixels: 0 -> 1 detail
}


def _ramp(v, zero, full):
    return float(np.clip((v - zero) / (full - zero), 0.0, 1.0))


def source_detail(face, source_scale=1.0) -> float:
    """0..1: how many real pixels the face has in the ORIGINAL photo."""
    return _ramp(face.w * source_scale, *TH["source_face_px"])


@dataclass
class Adjustment:
    kind: str       # move_crop_up/down, shift_crop_left/right, zoom_in/out
    amount: float   # fraction (shifts) or factor (zoom)
    text: str


@dataclass
class CompositionReport:
    face_h_ratio: float
    face_area_ratio: float
    size_label: str
    rel_x: float
    rel_y: float
    h_pos: str
    v_pos: str
    headroom: float
    headroom_label: str
    visible: str                     # unknown/head_only/head_shoulders/upper_body/full_body
    edge_cut: list = field(default_factory=list)
    adjustments: list = field(default_factory=list)


def _visible_level(body, ry, rh):
    """How much of the (real) body falls inside the region."""
    if body is None or body.level == "unknown":
        return "unknown"
    bottom, lm = ry + rh, body.landmarks

    def inside(*names):
        return all(n in lm and lm[n][1] <= bottom for n in names)

    if body.ankles and inside("l_ankle", "r_ankle"):
        return "full_body"
    if body.hips and inside("l_hip", "r_hip"):
        return "upper_body"
    if body.shoulders and inside("l_shoulder", "r_shoulder"):
        return "head_shoulders"
    return "head_only"


def analyze_composition(face, region, body=None) -> CompositionReport:
    rx, ry, rw, rh = region
    cx, cy = face.center
    rel_x, rel_y = (cx - rx) / rw, (cy - ry) / rh
    face_h_ratio = face.h / rh
    area = (face.w * face.h) / (rw * rh)

    headroom = (face.y - HAIR * face.h - ry) / face.h
    lo, hi = TH["headroom"]
    if headroom < 0:
        hr_label = "cut_off"
    elif headroom < lo:
        hr_label = "too_little"
    elif headroom > hi:
        hr_label = "excessive"
    else:
        hr_label = "balanced"

    ex, ey = TH["edge"] * rw, TH["edge"] * rh
    edge_cut = []
    if face.x <= rx + ex: edge_cut.append("left")
    if face.x + face.w >= rx + rw - ex: edge_cut.append("right")
    if face.y <= ry + ey: edge_cut.append("top")
    if face.y + face.h >= ry + rh - ey: edge_cut.append("bottom")

    small, large, excessive = TH["face_h"]
    if face_h_ratio < small:
        size_label = "too_small"
    elif face_h_ratio > excessive or edge_cut:
        size_label = "excessively_cropped"
    elif face_h_ratio > large:
        size_label = "large"
    else:
        size_label = "acceptable"

    cl, cr = TH["center"]
    h_pos = "left" if rel_x < cl else "right" if rel_x > cr else "center"
    vt, vb = TH["vertical"]
    v_pos = "top" if rel_y < vt else "bottom" if rel_y > vb else "middle"

    adj = []
    if hr_label != "balanced":
        delta = (TARGET_HEADROOM - headroom) * face.h / rh   # >0: need more room above
        up = delta > 0
        adj.append(Adjustment(
            "move_crop_up" if up else "move_crop_down", abs(delta),
            f"Move the crop {'up' if up else 'down'} about {abs(delta) * 100:.0f}% "
            f"of its height ({'more' if up else 'less'} headroom)"))
    if size_label == "too_small":
        z = TARGET_FACE_H / face_h_ratio
        adj.append(Adjustment("zoom_in", z, f"Crop tighter: zoom in about {z:.1f}x"))
    elif size_label in ("large", "excessively_cropped"):
        z = face_h_ratio / TARGET_FACE_H
        adj.append(Adjustment("zoom_out", z, f"Zoom out about {z:.1f}x to give the face room"))
    if h_pos != "center":
        dx = rel_x - 0.5
        side = "right" if dx > 0 else "left"
        adj.append(Adjustment(f"shift_crop_{side}", abs(dx),
                              f"Shift the crop {side} about {abs(dx) * 100:.0f}% of its width"))

    return CompositionReport(face_h_ratio, area, size_label, rel_x, rel_y, h_pos, v_pos,
                             headroom, hr_label, _visible_level(body, ry, rh),
                             edge_cut, adj)
