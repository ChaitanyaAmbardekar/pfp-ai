"""Pick the best framing for each photo in samples\\ (or one given photo)."""
import sys
from pathlib import Path

import cv2

from app.body_pose import PoseDetector
from app.face_detector import FaceDetector
from app.framing_selector import select_framing
from app.image_io import SUPPORTED, load_image_bgr
from app.pfp_simulator import crop_square, to_circle

OUT = Path("outputs")


def run(path, det, det_small, pose):
    bgr, _ = load_image_bgr(path)
    d = det.detect(bgr)
    print(f"\n{path.name}")
    if d.primary is None:
        print("  no face found, skipped")
        return
    if d.ambiguous:
        print(f"  WARNING: {d.note}")
    body = pose.detect(bgr, d.primary)
    print(f"  body visible: {body.level}  (shoulders={body.shoulders} hips={body.hips} "
          f"knees={body.knees} ankles={body.ankles})")
    for n in body.notes:
        print(f"  note: {n}")

    cands = select_framing(bgr, d, body, det_small)
    print("  framing          score  read  ctx  clean  notes")
    folder = OUT / path.stem
    folder.mkdir(parents=True, exist_ok=True)
    for c in cands:
        if not c.available:
            print(f"  {c.name:<15}    --   unavailable: {'; '.join(c.notes)}")
            continue
        print(f"  {c.name:<15} {c.score:6.1f}  {c.readability:4.2f} {c.context:4.1f}  "
              f"{c.cleanliness:4.1f}  {'; '.join(c.notes)}")
        cv2.imwrite(str(folder / f"cand_{c.name}.png"),
                    to_circle(crop_square(bgr, c.plan), 256))
    best = next((c for c in cands if c.available), None)
    if best:
        cv2.imwrite(str(folder / "best_pfp.png"), to_circle(crop_square(bgr, best.plan), 256))
        print(f"  >> RECOMMENDED FRAMING: {best.name}   (saved best_pfp.png)")


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
    