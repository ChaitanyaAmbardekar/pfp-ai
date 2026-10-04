"""Composition of the source photo and of the recommended crop, for each sample."""
import sys
from pathlib import Path

import cv2

from app.body_pose import PoseDetector
from app.composition import analyze_composition, source_detail
from app.face_detector import FaceDetector
from app.framing_selector import select_framing
from app.image_io import SUPPORTED, load_image_bgr
from app.pfp_simulator import crop_square, to_circle

OUT = Path("outputs")


def show(label, c, adjustments=False):
    print(f"  [{label}]")
    print(f"    face size    : {c.face_h_ratio * 100:.0f}% of height, "
          f"{c.face_area_ratio * 100:.1f}% of area -> {c.size_label}")
    print(f"    position     : {c.h_pos} / {c.v_pos}  (x={c.rel_x:.2f}, y={c.rel_y:.2f})")
    print(f"    headroom     : {c.headroom:.2f} face heights -> {c.headroom_label}")
    print(f"    body visible : {c.visible}")
    if c.edge_cut:
        print(f"    face touches : {', '.join(c.edge_cut)} edge")
    if adjustments:
        for a in c.adjustments:
            print(f"    suggest      : {a.text}")
        if not c.adjustments:
            print("    suggest      : none needed")


def run(path, det, det_small, pose):
    bgr, orig = load_image_bgr(path)
    h, w = bgr.shape[:2]
    scale = max(orig) / max(h, w)
    d = det.detect(bgr)
    print(f"\n{path.name}  (original {orig[0]}x{orig[1]}, analysed {w}x{h})")
    if d.primary is None:
        print("  no face found, skipped")
        return
    if d.ambiguous:
        print(f"  WARNING: {d.note}")
    face = d.primary
    body = pose.detect(bgr, face)
    print(f"  face is {face.w * scale:.0f}px wide in the original -> "
          f"source detail {source_detail(face, scale):.2f}")
    show("SOURCE PHOTO", analyze_composition(face, (0, 0, w, h), body))

    cands = select_framing(bgr, d, body, det_small, source_scale=scale)
    best = next((c for c in cands if c.available), None)
    if best is None:
        print("  no usable framing")
        return
    p = best.plan
    print(f"  >> {best.name}  score {best.score:.1f}  readability {best.readability:.2f}")
    show("RECOMMENDED CROP", analyze_composition(face, (p.x, p.y, p.side, p.side), body), True)
    folder = OUT / path.stem
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "best_pfp.png"), to_circle(crop_square(bgr, p), 256))


def main():
    det, det_small = FaceDetector(), FaceDetector(score_threshold=0.5)
    pose = PoseDetector()
    paths = [Path(sys.argv[1])] if len(sys.argv) > 1 else sorted(
        p for p in Path("samples").iterdir() if p.suffix.lower() in SUPPORTED)
    try:
        for p in paths:
            try:
                run(p, det, det_small, pose)
            except Exception as e:
                print(f"\n{p.name}: ERROR -> {e}")
    finally:
        pose.close()


if __name__ == "__main__":
    main()
    