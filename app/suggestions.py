"""Suggested edits (each can be toggled) and presets.

Rules use relative measurements and the photo overall, never absolute skin
brightness. Presets describe how a photo reads; they make no outcome claims.
"""
from dataclasses import replace

from app.analyzer import ramp
from app.editor import Edit


def build_suggestions(a):
    f, edits = a.facts, []
    if a.best is not None:
        p = a.best.plan
        edits.append(Edit("crop", "crop", f"Use the {a.best.name.replace('_', ' ')} crop",
                          {"x": p.x, "y": p.y, "side": p.side},
                          "Face-aware square crop that keeps the face readable in a circle"))

    need = max(ramp(f.get("face_dark_fraction", 0.0), 0.10, 0.45),
               ramp(-(f.get("face_diff") or 0.0), 40, 100))
    if need > 0.15:
        edits.append(Edit("lift_shadows", "light", "Lift shadows",
                          {"shadows": round(0.2 + 0.6 * need, 2)},
                          "Face is darker than the scene or has crushed shadows"))

    bright = f.get("face_bright_fraction", 0.0)
    if bright > 0.08:
        edits.append(Edit("tame_highlights", "light", "Reduce highlights",
                          {"highlights": round(0.2 + 0.6 * ramp(bright, 0.08, 0.30), 2)},
                          "Part of the face is close to pure white"))

    if f.get("whole_dynamic_range", 180.0) < 100:
        edits.append(Edit("add_contrast", "light", "Add contrast", {"contrast": 0.12},
                          "Tones across the photo are compressed (flat look)"))

    if f.get("sharp_label") in ("soft", "very soft") and f.get("detail", 1.0) >= 0.5:
        edits.append(Edit("sharpen", "detail", "Mild sharpen", {"sharpen": 0.4},
                          "Face looks soft. This helps only slightly: real blur cannot be fixed"))
    return edits


PRESETS = {
    "polished_confident": ("Polished / confident",
                           "Clean crop, balanced light, gentle contrast and crispness. "
                           "Describes how the photo reads, with no claim about outcomes."),
    "warm_film": ("Warm film", "Trend look: warm tone, softer colour. Trends change, so this is taste."),
    "moody": ("Moody contrast", "Trend look: darker, higher contrast, muted colour. This is taste."),
    "soft_bright": ("Soft and bright", "Trend look: airy and light. This is taste."),
}
_TREND = {
    "warm_film": [("warmth", 0.12), ("saturation", -0.08), ("contrast", 0.06)],
    "moody": [("contrast", 0.18), ("brightness", -0.05), ("saturation", -0.10)],
    "soft_bright": [("brightness", 0.08), ("contrast", -0.08), ("warmth", 0.05)],
}
_KIND = {"warmth": "color", "saturation": "color", "brightness": "light", "contrast": "light"}


def apply_preset(name, base_edits, a):
    """Base suggestions plus the preset's edits layered on top (copies; originals untouched)."""
    if name not in PRESETS:
        raise KeyError(name)
    out = [replace(e, params=dict(e.params)) for e in base_edits]

    def has(key):
        return any(key in e.params for e in out)

    if name == "polished_confident":
        if not has("contrast"):
            out.append(Edit("polish_contrast", "light", "Gentle contrast", {"contrast": 0.06},
                            "Slightly cleaner tonal separation"))
        if a.facts.get("detail", 1.0) >= 0.5 and not has("sharpen"):
            out.append(Edit("polish_sharpen", "detail", "Light crispness", {"sharpen": 0.25},
                            "Subtle clarity at small sizes"))
    else:
        for key, val in _TREND[name]:
            out.append(Edit(f"{name}_{key}", _KIND[key], f"{PRESETS[name][0]}: {key}",
                            {key: val}, "Trend style (subjective)"))
    return out
