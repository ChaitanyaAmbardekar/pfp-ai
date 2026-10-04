"""Make square/circle crops and small-size previews for each sample photo."""
import sys
from pathlib import Path

import cv2

from app.face_detector import FaceDetector
from app.image_io import SUPPORTED, load_image_bgr
from app.pfp_simulator import (CIRCLE_FULL, SIZES, contact_sheet, crop_square,
                               evaluate, plan_square_crop, to_circle)

OUT = Path("outputs")


def write(path, img):
    if not cv2.imwrite(str(path), img):
        raise IOError(f"Could not write {path}")


def run(path, det_main, det_small):
    bgr, _ = load_image_bgr(path)
    h, w = bgr.shape[:2]
    det = det_main.detect(bgr)
    face = det.primary
    print(f"\n{path.name}  ({len(det.faces)} face(s))")
    if det.ambiguous:
        print(f"  WARNING: {det.note}")
    if face is None:
        print("  no face found: plain center crop used, robustness not scored")

    plan = plan_square_crop(w, h, face, "face")
    square = crop_square(bgr, plan)
    report = evaluate(square, plan, face, det_small)

    folder = OUT / path.stem
    folder.mkdir(parents=True, exist_ok=True)
    circles = {s: to_circle(square, s) for s in SIZES}
    write(folder / "original.jpg", bgr)            # the analysed (<=1280 px) version
    write(folder / "square.jpg", square)
    write(folder / "circle.png", to_circle(square, min(plan.side, CIRCLE_FULL)))
    for s in SIZES:
        write(folder / f"pfp_{s}.png", circles[s])
    write(folder / "preview.png", contact_sheet(circles))

    print(f"  crop ({plan.mode}): x={plan.x} y={plan.y} side={plan.side}px")
    if face:
        print(f"  face fills {report.face_fraction * 100:.0f}% of crop width; "
              f"head-in-circle margin {report.circle_overflow:.2f} (1.00 = touching the edge)")
        print("  size  face_px  eye_px  re-detected  shadows  score")
        for s in SIZES:
            c = report.checks[s]
            d = "n/a" if c.detected is None else (f"yes {c.det_conf:.2f}" if c.detected else "NO")
            print(f"  {s:>4}  {c.face_px:7.1f}  {c.eye_px:6.1f}  {d:<11}  {c.shadows * 100:5.1f}%  {c.score:5.1f}")

        plan_c = plan_square_crop(w, h, face, "center")
        rep_c = evaluate(crop_square(bgr, plan_c), plan_c, face, det_small)
        print(f"  PFP Robustness: {report.robustness:.0f}/100   "
              f"(plain center crop for comparison: {rep_c.robustness:.0f}/100)")
        print(f"  flags: {', '.join(report.flags) or 'none'}")
        print("  framing comparison (robustness | image can support it?):")
        for m in ("face", "head_shoulders", "upper_body"):
            p = plan_square_crop(w, h, face, m)
            sq = crop_square(bgr, p)
            r = evaluate(sq, p, face, det_small)
            print(f"    {m:<15} {r.robustness:5.1f}   {'no (image too small)' if p.limited else 'yes'}")
            write(folder / f"circle_{m}.png", to_circle(sq, 256))
    print(f"  saved: {folder}\\  (open preview.png)")


def main():
    det_main = FaceDetector()                       # finds the primary face
    det_small = FaceDetector(score_threshold=0.5)   # more forgiving for tiny previews
    if len(sys.argv) > 1:
        paths = [Path(sys.argv[1])]
    else:
        paths = sorted(p for p in Path("samples").iterdir() if p.suffix.lower() in SUPPORTED)
    for p in paths:
        try:
            run(p, det_main, det_small)
        except Exception as e:
            print(f"\n{p.name}: ERROR -> {e}")


if __name__ == "__main__":
    main()
