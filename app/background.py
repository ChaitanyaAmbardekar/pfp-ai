"""Background analysis for a candidate PFP crop (OpenCV + NumPy, CPU-only).

Only pixels inside the circle and outside the person rectangle count.
All thresholds are provisional; calibrate them on real photos.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from app.pfp_simulator import circle_mask

TH = {
    "colorful": (15, 60),      # Hasler-Suesstrunk colourfulness
    "structure": (0.01, 0.05), # edge density: below = blank, above = has structure
    "tonal": (15, 50),         # luminance std of the background
    "separation": (8, 35),     # Lab distance person vs background
    "clutter_edges": (0.12, 0.25),
    "clutter_bright": (0.15, 0.40),
    "weight": (0.15, 0.50),    # background share of the circle
}


@dataclass
class BackgroundInfo:
    bg_fraction: float
    colorfulness: float
    structure: float
    tonal: float
    interest: float
    separation: float
    clutter: float
    scenery: float     # interest * separation * (1 - clutter)
    weight: float      # how much the background matters for this crop


def _ramp(v, zero, full):
    return float(np.clip((v - zero) / (full - zero), 0.0, 1.0))


def analyze_background(crop_bgr, plan, face, body=None, size=128) -> BackgroundInfo:
    k = size / plan.side
    small = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_AREA)

    # Person rectangle in small-crop coordinates.
    cx = (face.center[0] - plan.x) * k
    if body is not None and body.shoulders and body.landmarks:
        span = abs(body.landmarks["l_shoulder"][0] - body.landmarks["r_shoulder"][0])
        half = max(0.65 * span, 0.8 * face.w) * k
    else:
        half = 1.4 * face.w * k
    x0, x1 = int(max(0, cx - half)), int(min(size, cx + half))
    y0 = int(max(0, (face.y - plan.y - 0.35 * face.h) * k))   # include hair
    person = np.zeros((size, size), bool)
    person[y0:size, x0:x1] = True
    visible = circle_mask(size) > 127
    pers = visible & person
    frac = float((visible & ~person).sum() / max(1, visible.sum()))
    # Keep a margin around the person so their outline and resize blending
    # are not mistaken for background texture or colour.
    margin = cv2.dilate(person.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    bg = visible & ~margin
    if bg.sum() < 100 or pers.sum() < 50:
        return BackgroundInfo(frac, 0, 0, 0, 0, 0, 0, 0, 0.0)

    b, g, r = [small[..., i].astype(np.float32)[bg] for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorf = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                   + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 150)
    edge_density = float((edges[bg] > 0).mean())
    tonal = float(gray[bg].std())
    bright = float((gray[bg] > 235).mean())

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    dist = float(np.linalg.norm(lab[bg].mean(0) - lab[pers].mean(0)))

    color_s = _ramp(colorf, *TH["colorful"])
    structure = _ramp(edge_density, *TH["structure"])
    tonal_s = _ramp(tonal, *TH["tonal"])
    interest = 0.4 * color_s + 0.3 * structure + 0.3 * tonal_s
    separation = _ramp(dist, *TH["separation"])
    clutter = max(_ramp(edge_density, *TH["clutter_edges"]),
                  _ramp(bright, *TH["clutter_bright"]))
    scenery = interest * separation * (1 - clutter)
    return BackgroundInfo(frac, color_s, structure, tonal_s, interest, separation,
                          clutter, scenery, _ramp(frac, *TH["weight"]))