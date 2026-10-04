"""Combine all measurements into sub-scores and one overall PFP score.

All thresholds and weights are provisional. Weights can be overridden in
config.json. This scores the PHOTOGRAPH as a profile picture, not the person.
"""
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.composition import analyze_composition, source_detail
from app.framing_selector import _extra_faces, select_framing
from app.image_io import load_image_bgr
from app.image_quality import analyze_quality

DEFAULT_WEIGHTS = {
    "face_visibility": 0.20,
    "robustness": 0.20,
    "framing": 0.15,
    "lighting": 0.15,
    "sharpness": 0.15,
    "background": 0.15,
}
LABELS = {
    "face_visibility": "Face visibility",
    "robustness": "Small-size robustness",
    "framing": "Framing",
    "lighting": "Lighting",
    "sharpness": "Sharpness",
    "background": "Background",
}
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
OTHER_FACE_POINTS, OTHER_FACE_CAP = 10.0, 20.0


def ramp(v, zero, full):
    return float(np.clip((v - zero) / (full - zero), 0.0, 1.0))


def load_weights(path=CONFIG_PATH):
    """Defaults, overridden by valid entries in config.json (if present)."""
    w = dict(DEFAULT_WEIGHTS)
    path = Path(path)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in data.get("weights", {}).items():
                if k in w and isinstance(v, (int, float)) and v >= 0:
                    w[k] = float(v)
        except (OSError, ValueError) as e:
            warnings.warn(f"config.json ignored ({e}); using default weights")
    return w


@dataclass
class Penalty:
    code: str
    points: float
    text: str


@dataclass
class PhotoAnalysis:
    name: str
    path: str
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    problem: str = ""
    overall: float = 0.0
    sub: dict = field(default_factory=dict)        # name -> 0..100
    penalties: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)      # raw measurements
    best: object = None                            # chosen Candidate
    comp: object = None                            # composition of the chosen crop
    candidates: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.problem and bool(self.sub)


# ---- sub-scores: each returns 0..1 ------------------------------------------

def sharpness_score(face_sharpness, detail):
    """Face sharpness is measured at a fixed 160 px, so for a face with few real
    pixels it mostly reflects our upscaling. Low `detail` pulls toward neutral."""
    raw = ramp(face_sharpness, 20, 200)
    return raw * detail + 0.5 * (1 - detail)


def lighting_score(face_q, face_minus_scene):
    """Relative signals only. No absolute face brightness, no face contrast."""
    crushed = 1 - ramp(face_q.dark_fraction, 0.10, 0.50)
    blown = 1 - ramp(face_q.bright_fraction, 0.05, 0.30)
    backlit = 1.0 if face_minus_scene is None else 1 - ramp(-face_minus_scene, 40, 100)
    return 0.40 * crushed + 0.30 * blown + 0.30 * backlit


def face_visibility_score(conf, detail, detected64, eye_px64):
    c = ramp(conf, 0.60, 0.95)
    d = 0.5 if detected64 is None else (1.0 if detected64 else 0.0)
    e = ramp(eye_px64, 4, 8)
    return 0.30 * c + 0.25 * detail + 0.25 * d + 0.20 * e


def framing_score(comp, context):
    c = 1.0
    if comp.headroom_label != "balanced":
        c -= 0.25
    if comp.size_label != "acceptable":
        c -= 0.25
    if comp.edge_cut:
        c -= 0.25
    return 0.75 * max(0.0, c) + 0.25 * context


def background_score(b):
    """Neutral when the background barely shows in the crop."""
    if b is None or b.weight < 0.15:
        return 0.7
    return 0.4 * (1 - b.clutter) + 0.3 * b.separation + 0.3 * b.interest


def combine(sub, weights, penalties):
    """sub is 0..100 per name. Weights are renormalised over the names present."""
    wsum = sum(weights.get(k, 0.0) for k in sub)
    if wsum <= 0:
        return 0.0
    base = sum(weights.get(k, 0.0) * v for k, v in sub.items()) / wsum
    return max(0.0, base - sum(p.points for p in penalties))


# ---- one photo --------------------------------------------------------------

def analyze_photo(path, det, det_small, pose, weights=None) -> PhotoAnalysis:
    weights = weights or load_weights()
    path = Path(path)
    a = PhotoAnalysis(name=path.name, path=str(path), weights=weights)
    try:
        bgr, orig = load_image_bgr(path)
    except Exception as e:              # unreadable or unsupported file
        a.problem = f"could not open image ({e})"
        return a

    h, w = bgr.shape[:2]
    scale = max(orig) / max(h, w)
    d = det.detect(bgr)
    if d.primary is None:
        a.problem = "no face detected"
        return a
    if d.ambiguous:
        a.notes.append(d.note)
    face = d.primary

    body = pose.detect(bgr, face)
    cands = select_framing(bgr, d, body, det_small, source_scale=scale)
    best = next((c for c in cands if c.available), None)
    if best is None:
        a.problem = "no usable framing"
        return a
    q = analyze_quality(bgr, d)
    if q.face is None:
        a.problem = "face too small to measure quality"
        return a

    p = best.plan
    comp = analyze_composition(face, (p.x, p.y, p.side, p.side), body)
    detail = source_detail(face, scale)
    c64 = best.report.checks.get(64) if best.report else None
    fq, bg = q.face, best.background

    sub = {
        "face_visibility": face_visibility_score(
            face.score, detail, c64.detected if c64 else None, c64.eye_px if c64 else 0.0),
        "robustness": best.readability,
        "framing": framing_score(comp, best.context),
        "lighting": lighting_score(fq, q.face_minus_image_brightness),
        "sharpness": sharpness_score(fq.sharpness, detail),
        "background": background_score(bg),
    }
    a.sub = {k: round(100 * v, 1) for k, v in sub.items()}

    extra = _extra_faces(p, d)
    if extra:
        a.penalties.append(Penalty("other_people", min(OTHER_FACE_CAP, OTHER_FACE_POINTS * extra),
                                   f"{extra} other face(s) inside the crop"))
    a.overall = round(combine(a.sub, weights, a.penalties), 1)

    a.facts = {
        "orig_size": orig, "face_px_original": face.w * scale, "detail": detail,
        "framing": best.name, "body_level": body.level, "extra_faces": extra,
        "detected64": c64.detected if c64 else None, "eye_px64": c64.eye_px if c64 else 0.0,
        "face_sharpness": fq.sharpness, "sharp_label": fq.sharpness_label,
        "face_dark_fraction": fq.dark_fraction, "face_bright_fraction": fq.bright_fraction,
        "face_diff": q.face_minus_image_brightness, "whole_dynamic_range": q.whole.dynamic_range,
        "bg_weight": bg.weight if bg else 0.0, "bg_clutter": bg.clutter if bg else 0.0,
        "bg_separation": bg.separation if bg else 0.0, "bg_interest": bg.interest if bg else 0.0,
    }
    a.best, a.comp, a.candidates = best, comp, cands
    return a
