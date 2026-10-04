"""Retake / photo-choice advice. These are tips about the photograph, not edits."""


def build_advice(a):
    f, tips = a.facts, []
    n = f.get("extra_faces", 0)
    if n:
        tips.append(
            f"{n} other face(s) appear in the crop. This could be a much stronger PFP "
            "if only you were in the frame. Try a similar shot taken alone, or use a "
            "photo where the others can be cropped out without making the face too small.")
    if f.get("detail", 1.0) < 0.5:
        tips.append(
            f"Your face is only {f.get('face_px_original', 0):.0f}px wide in the original. "
            "Retake closer, or use a higher-resolution original, so the PFP stays crisp.")
    diff = f.get("face_diff")
    if (diff is not None and diff < -40) or f.get("face_dark_fraction", 0) > 0.30:
        tips.append("Your face is in shadow compared with the scene. Turn toward the light "
                    "(a window or sunlight on the face works well) and retake.")
    if f.get("bg_weight", 0) >= 0.15 and f.get("bg_clutter", 0) > 0.5:
        tips.append("The background is busy. Move a few steps, or open up some distance "
                    "between you and what is behind you.")
    return tips