"""Analyse every photo in samples\\, rank them, explain the winner, suggest edits."""
import argparse
import csv
from pathlib import Path

import cv2

from app.analyzer import LABELS, PhotoAnalysis, analyze_photo, load_weights
from app.body_pose import PoseDetector
from app.editor import LocalEditor
from app.explain import explain, explain_comparison
from app.face_detector import FaceDetector
from app.image_io import SUPPORTED, load_image_bgr
from app.pfp_simulator import to_circle
from app.ranking import close_call, rank, top_n
from app.suggestions import PRESETS, apply_preset, build_suggestions

OUT = Path("outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--preset", default="polished_confident", choices=sorted(PRESETS))
    args = ap.parse_args()
    paths = [Path(p) for p in args.paths] or sorted(
        p for p in Path("samples").iterdir() if p.suffix.lower() in SUPPORTED)

    weights = load_weights()
    det, det_small = FaceDetector(), FaceDetector(score_threshold=0.5)
    pose = PoseDetector()
    results = []
    try:
        for p in paths:
            try:
                results.append(analyze_photo(p, det, det_small, pose, weights))
            except Exception as e:
                results.append(PhotoAnalysis(name=p.name, path=str(p), weights=weights,
                                             problem=f"unexpected error: {e}"))
    finally:
        pose.close()

    ranked = rank(results)
    print("\nRANKING")
    print(f"{'#':>2}  {'photo':<12}{'score':>6}  " + " ".join(f"{k[:6]:>6}" for k in LABELS) + "  framing")
    for i, a in enumerate(ranked, 1):
        if a.ok:
            print(f"{i:>2}  {a.name:<12}{a.overall:6.1f}  "
                  + " ".join(f"{a.sub[k]:6.0f}" for k in LABELS) + f"  {a.facts['framing']}")
        else:
            print(f"{i:>2}  {a.name:<12}  skipped: {a.problem}")

    top = top_n(ranked, 3)
    if not top:
        print("\nNo photo could be scored.")
        return
    print("\nTOP 3: " + ", ".join(a.name for a in top))
    if close_call(ranked):
        print("Close call: the top two are within 3 points, so treat the order as a toss-up.")
    best = top[0]
    print("\n" + explain(best))
    if len(top) > 1:
        print("\n" + explain_comparison(top[0], top[1]))

    base = build_suggestions(best)
    edits = apply_preset(args.preset, base, best)
    print(f"\nSUGGESTED EDITS for {best.name} (preset: {PRESETS[args.preset][0]})")
    for e in edits:
        print(f"  [{'x' if e.enabled else ' '}] {e.label:<26} {e.params}  - {e.reason}")

    bgr, _ = load_image_bgr(best.path)
    editor = LocalEditor()
    before = editor.apply(bgr, [e for e in base if e.kind == "crop"])
    after = editor.apply(bgr, edits)
    folder = OUT / Path(best.path).stem
    folder.mkdir(parents=True, exist_ok=True)
    size = min(512, before.shape[0])
    cv2.imwrite(str(folder / "pfp_before.png"), to_circle(before, size))
    cv2.imwrite(str(folder / "pfp_after.png"), to_circle(after, size))
    print(f"\nSaved {folder}\\pfp_before.png and pfp_after.png")

    try:
        with open(OUT / "ranking.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "photo", "overall", *LABELS, "framing", "penalties"])
            for i, a in enumerate(ranked, 1):
                if a.ok:
                    w.writerow([i, a.name, a.overall, *[a.sub[k] for k in LABELS], a.facts["framing"],
                                "; ".join(p.text for p in a.penalties)])
        print(f"Saved {OUT}\\ranking.csv")
    except PermissionError:
        print("Could not write ranking.csv (is it open in Excel?)")


if __name__ == "__main__":
    main()
    