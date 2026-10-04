"""Photo editing. Only crop, light, colour and sharpness. Never faces or bodies.

`Editor` is the interface. LocalEditor works offline with OpenCV. A future
ApiEditor (cloud AI) will implement the same interface with uploads_photo=True
so the UI can ask for consent first.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Edit:
    id: str
    kind: str            # crop | light | color | detail
    label: str
    params: dict
    reason: str = ""
    enabled: bool = True  # the user's Apply / Skip toggle


LIMITS = {   # every value is clipped to these ranges, so edits stay natural
    "brightness": (-0.25, 0.25), "contrast": (-0.40, 0.40),
    "shadows": (-0.50, 1.00), "highlights": (0.0, 1.0),
    "saturation": (-0.40, 0.40), "warmth": (-0.20, 0.20), "sharpen": (0.0, 1.0),
}
KIND_ORDER = {"crop": 0, "light": 1, "color": 2, "detail": 3}


def _brightness(x, v): return np.power(x, 1.0 - v)            # v>0 brightens
def _contrast(x, v): return (x - 0.5) * (1 + v) + 0.5
def _shadows(x, v): return x + v * 0.25 * (1 - x) ** 3        # lifts dark tones
def _highlights(x, v): return x - v * 0.25 * x ** 3           # pulls bright tones down


def _saturation(x, v):
    hsv = cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + v), 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _warmth(x, v):
    y = x.copy()
    y[..., 2] *= 1 + 0.4 * v      # red up when warm
    y[..., 0] *= 1 - 0.4 * v      # blue down when warm
    return y


def _sharpen(x, v):
    return x + v * 0.8 * (x - cv2.GaussianBlur(x, (0, 0), 1.2))


OPS = {"brightness": _brightness, "contrast": _contrast, "shadows": _shadows,
       "highlights": _highlights, "saturation": _saturation, "warmth": _warmth,
       "sharpen": _sharpen}


def _crop(bgr, p):
    h, w = bgr.shape[:2]
    if p:
        x, y, s = int(p["x"]), int(p["y"]), int(p["side"])
        if s >= 8 and x >= 0 and y >= 0 and x + s <= w and y + s <= h:
            return bgr[y:y + s, x:x + s]
    s = min(h, w)                     # fallback: centre square
    y0, x0 = (h - s) // 2, (w - s) // 2
    return bgr[y0:y0 + s, x0:x0 + s]


class Editor:
    name = "base"
    uploads_photo = False

    def apply(self, bgr, edits):
        raise NotImplementedError


class LocalEditor(Editor):
    name = "local"
    uploads_photo = False

    def apply(self, bgr, edits):
        """Apply only the enabled edits: crop first, then light, colour, detail."""
        enabled = sorted((e for e in edits if e.enabled),
                         key=lambda e: KIND_ORDER.get(e.kind, 9))
        crop_edit = next((e for e in enabled if e.kind == "crop"), None)
        img = _crop(bgr, crop_edit.params if crop_edit else None)
        ops = [(k, v) for e in enabled if e.kind != "crop"
               for k, v in e.params.items() if k in OPS]
        if not ops:
            return img.copy()
        x = img.astype(np.float32) / 255.0
        for k, v in ops:
            lo, hi = LIMITS[k]
            x = np.clip(OPS[k](np.clip(x, 0, 1), float(np.clip(v, lo, hi))), 0, 1)
            x = x.astype(np.float32)
        return (x * 255.0 + 0.5).astype(np.uint8)