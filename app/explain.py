"""Plain-language explanations. Markers are ASCII so Windows consoles never
break; a UI can map them to icons."""
from app.analyzer import LABELS

GOOD, WARN = "[+]", "[!]"


def assessment(a):
    """Rule-based judgments as (good, warn) lists of sentences."""
    f, s, c = a.facts, a.sub, a.comp
    good, warn = [], []

    if f.get("detected64") is False:
        warn.append("Face is not re-detected after shrinking to 64x64")
    elif s.get("face_visibility", 0) >= 80:
        good.append("Face is clearly detected and stays readable at 64x64")

    rob = s.get("robustness", 0)
    if rob >= 80:
        good.append(f"Survives the small circular crop (robustness {rob:.0f}/100)")
    elif rob < 60:
        warn.append("Face loses clarity once shrunk into a small circle")

    if c is not None:
        name = f.get("framing", "").replace("_", " ")
        if c.headroom_label == "balanced" and c.size_label == "acceptable" and not c.edge_cut:
            good.append(f"Balanced {name} crop with good headroom")
        else:
            warn.append(f"Crop composition: headroom {c.headroom_label.replace('_', ' ')}, "
                        f"face size {c.size_label.replace('_', ' ')}")

    diff = f.get("face_diff")
    if diff is not None and diff < -40:
        warn.append("Face is much darker than the scene (backlit or in shadow)")
    if f.get("face_dark_fraction", 0) > 0.30:
        warn.append("Deep shadows cover a large part of the face")
    if f.get("face_bright_fraction", 0) > 0.15:
        warn.append("Highlights are close to pure white on the face")
    if s.get("lighting", 0) >= 80:
        good.append("Face lighting is even compared with the scene")

    if f.get("detail", 1.0) < 0.5:
        warn.append(f"Face is only {f.get('face_px_original', 0):.0f}px wide in the original, "
                    "so sharpness and fine detail cannot be judged reliably")
    elif s.get("sharpness", 0) >= 80:
        good.append("Face is sharp")
    elif s.get("sharpness", 0) < 50:
        warn.append("Face looks soft")

    if f.get("bg_weight", 0) >= 0.15:
        if f.get("bg_clutter", 0) > 0.5:
            warn.append("Background is busy and competes with the face")
        elif f.get("bg_separation", 1) < 0.2:
            warn.append("Subject blends into the background (low colour separation)")
        elif s.get("background", 0) >= 70:
            good.append("Background supports the subject without competing")

    for p in a.penalties:
        warn.append(f"{p.text} ({-p.points:+.0f} points)")
    warn.extend(a.notes)
    return good, warn


def explain(a):
    if not a.ok:
        return f"{a.name}: could not be scored ({a.problem})"
    f = a.facts
    good, warn = assessment(a)
    L = ["=" * 58,
         f"{a.name}    PFP suitability: {a.overall:.0f} / 100",
         f"Recommended framing: {f['framing'].replace('_', ' ')}",
         "-" * 58, "Sub-scores (0-100, rule-based):"]
    L += [f"  {LABELS[k]:<24}{v:5.0f}" for k, v in a.sub.items()]
    L += ["", "Why this works:"] + [f"  {GOOD} {t}" for t in good] if good else ["", "Why this works:  (nothing stands out)"]
    L += ["", "Potential issues:"] + ([f"  {WARN} {t}" for t in warn] if warn else ["  none found"])
    d64 = {True: "yes", False: "no", None: "n/a"}[f.get("detected64")]
    diff = f.get("face_diff")
    L += ["", "Measured facts (objective numbers):",
          f"  face width in original photo : {f['face_px_original']:.0f} px",
          f"  face sharpness               : {f['face_sharpness']:.0f} ({f['sharp_label']})",
          f"  face vs scene brightness     : {'n/a' if diff is None else f'{diff:+.0f}'}",
          f"  crushed shadows / highlights : {f['face_dark_fraction'] * 100:.0f}% / {f['face_bright_fraction'] * 100:.0f}%",
          f"  re-detected at 64 px         : {d64}  (eyes {f['eye_px64']:.1f} px apart)",
          f"  body visible in photo        : {f['body_level']}",
          "",
          "Not scored yet: expression/presence and style match (Versions 2 and 3).",
          "This rates the photograph as a profile picture, not the person."]
    return "\n".join(L)


def explain_comparison(a, b):
    """Why the higher-scoring photo beat the other one."""
    if a.overall < b.overall:
        a, b = b, a
    keys = [k for k in a.sub if k in b.sub]
    wsum = sum(a.weights.get(k, 0) for k in keys) or 1.0
    d = {k: a.weights.get(k, 0) * (a.sub[k] - b.sub[k]) / wsum for k in keys}
    helps = sorted((k for k in keys if d[k] > 0.3), key=lambda k: -d[k])[:3]
    hurts = sorted((k for k in keys if d[k] < -0.3), key=lambda k: d[k])[:2]
    pen = sum(p.points for p in b.penalties) - sum(p.points for p in a.penalties)

    L = [f"{a.name} leads {b.name} by {a.overall - b.overall:.1f} points."]
    if helps:
        L.append("  Ahead on: " + "; ".join(
            f"{LABELS.get(k, k)} {a.sub[k]:.0f} vs {b.sub[k]:.0f} ({d[k]:+.1f} pts)" for k in helps))
    if abs(pen) >= 1:
        L.append(f"  Penalties differ by {pen:+.0f} points (e.g. other people in the crop).")
    if hurts:
        L.append("  Behind on: " + "; ".join(
            f"{LABELS.get(k, k)} {a.sub[k]:.0f} vs {b.sub[k]:.0f}" for k in hurts))
    return "\n".join(L)