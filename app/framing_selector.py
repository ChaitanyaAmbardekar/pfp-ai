"""Choose the best PFP framing for each photo from several candidate crops.

score = 100 * (W_READ*readability + W_CONTEXT*context + W_CLEAN*cleanliness)
        + PREF[framing] - low-resolution penalty
Weights are provisional. PREF is a placeholder for learned taste (Version 2).
"""
from dataclasses import dataclass, field
from typing import Optional

from app.pfp_simulator import SquareCrop, crop_square, evaluate, plan_square_crop

NAMES = ("face", "head_shoulders", "upper_body", "full_body")
W_READ, W_CONTEXT, W_CLEAN = 0.50, 0.25, 0.25
PREF = {n: 0.0 for n in NAMES}        # to be learned from your pairwise choices
EXTRA_FACE_PENALTY = 0.8              # cleanliness lost per other face in the crop
JOINT_CUT_PENALTY = 0.3               # crop edge cutting at hips / knees
LOW_RES_PENALTY = 8.0
JOINT_TOL = 0.4                       # in face heights


@dataclass
class Candidate:
    name: str
    available: bool
    plan: Optional[SquareCrop] = None
    report: object = None
    readability: float = 0.0
    context: float = 0.0
    cleanliness: float = 0.0
    score: float = 0.0
    notes: list = field(default_factory=list)


def _full_body_plan(w, h, face, body):
    if body.level != "full_body":
        return None
    ankle_y = max(body.landmarks["l_ankle"][1], body.landmarks["r_ankle"][1])
    top = max(0.0, face.y - 0.5 * face.h)
    bottom = min(float(h), ankle_y + 0.5 * face.h)
    desired = int(round((bottom - top) * 1.05))
    side = max(8, min(desired, w, h))
    cx = face.center[0]
    x = max(0, min(int(round(cx - side / 2)), w - side))
    y = max(0, min(int(round(top)), h - side))
    return SquareCrop(x, y, side, "full_body", desired > min(w, h))


def _context(name, body):
    if name == "face":
        return 0.0
    if body.level == "unknown":                 # pose failed: neutral guess
        return {"head_shoulders": 0.5, "upper_body": 0.3}.get(name, 0.0)
    if name == "head_shoulders":
        return 0.7 if body.shoulders else 0.0
    if name == "upper_body":
        return 1.0 if body.hips else (0.6 if body.shoulders else 0.0)
    return 1.0                                  # full_body (only built when real)


def _extra_faces(plan, detection):
    n = 0
    for i, f in enumerate(detection.faces):
        if i == detection.primary_index:
            continue
        cx, cy = f.center
        if plan.x <= cx <= plan.x + plan.side and plan.y <= cy <= plan.y + plan.side:
            n += 1
    return n


def _cuts_at_joint(plan, body, face, img_h):
    bottom = plan.y + plan.side
    if bottom >= img_h - 1:                     # the image edge itself is a natural cut
        return False
    joints = []
    if body.hips:
        joints += ["l_hip", "r_hip"]
    if body.knees:
        joints += ["l_knee", "r_knee"]
    return any(abs(bottom - body.landmarks[j][1]) < JOINT_TOL * face.h for j in joints)


def select_framing(bgr, detection, body, det_small=None):
    """Return candidates sorted best-first (unavailable ones last)."""
    face = detection.primary
    if face is None:
        return []
    h, w = bgr.shape[:2]
    out = []
    for name in NAMES:
        plan = _full_body_plan(w, h, face, body) if name == "full_body" \
            else plan_square_crop(w, h, face, name)
        if plan is None:
            out.append(Candidate(name, False, notes=["full body is not really in this photo"]))
            continue
        if plan.limited and name != "face":
            out.append(Candidate(name, False, plan, notes=["photo has no room for this framing"]))
            continue

        report = evaluate(crop_square(bgr, plan), plan, face, det_small)
        readability = report.robustness / 100
        notes = []
        if "face_small_at_64px" in report.flags:
            readability *= 0.6
            notes.append("face too small at 64 px")
        if "eyes_unclear_at_64px" in report.flags:
            readability *= 0.85
            notes.append("eyes unclear at 64 px")
        if "head_outside_circle" in report.flags:
            notes.append("head pokes outside the circle")

        context = _context(name, body)
        extra = _extra_faces(plan, detection)
        clean = max(0.0, 1.0 - EXTRA_FACE_PENALTY * extra)
        if extra:
            notes.append(f"{extra} other face(s) inside the crop")
        if name != "face" and _cuts_at_joint(plan, body, face, h):
            clean = max(0.0, clean - JOINT_CUT_PENALTY)
            notes.append("crop edge cuts at a joint (hips/knees)")

        score = 100 * (W_READ * readability + W_CONTEXT * context + W_CLEAN * clean) + PREF[name]
        if "crop_low_resolution" in report.flags:
            score -= LOW_RES_PENALTY
            notes.append("few source pixels (looks soft)")
        out.append(Candidate(name, True, plan, report, readability, context, clean, score, notes))

    out.sort(key=lambda c: (not c.available, -c.score))
    return out