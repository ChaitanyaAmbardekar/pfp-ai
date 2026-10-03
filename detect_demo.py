"""Run face detection on one image, or on every image in samples\\."""
import sys
from pathlib import Path

import cv2

from app.face_detector import FaceDetector
from app.image_io import SUPPORTED, load_image_bgr

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def run(path, detector):
    bgr, orig = load_image_bgr(path)
    res = detector.detect(bgr)
    print(f"\n{path.name}  (original {orig[0]}x{orig[1]}, analysed {res.image_w}x{res.image_h})")
    print(f"  faces found : {len(res.faces)}")
    print(f"  note        : {res.note}")
    if res.primary:
        f = res.primary
        rx, ry = res.relative_center()
        print(f"  primary box : x={f.x} y={f.y} w={f.w} h={f.h}  conf={f.score:.2f}")
        print(f"  face/image  : {res.face_area_ratio() * 100:.1f}% of the image area")
        print(f"  position    : {rx:.2f} across, {ry:.2f} down (0.5 = centered)")

    vis = bgr.copy()
    for i, f in enumerate(res.faces):
        color = (0, 200, 0) if i == res.primary_index else (0, 165, 255)
        cv2.rectangle(vis, (f.x, f.y), (f.x + f.w, f.y + f.h), color, 3)
        for px, py in f.landmarks:
            cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1)
    out = OUT / f"{path.stem}_faces.jpg"
    cv2.imwrite(str(out), vis)
    print(f"  saved       : {out}")


def main():
    detector = FaceDetector()
    if len(sys.argv) > 1:
        paths = [Path(sys.argv[1])]
    else:
        paths = sorted(p for p in Path("samples").iterdir()
                       if p.suffix.lower() in SUPPORTED)
    if not paths:
        print("No images found. Put some photos in D:\\pfp-ai\\samples\\")
        return
    for p in paths:
        try:
            run(p, detector)
        except Exception as e:  # keep going if one file is bad
            print(f"\n{p.name}: ERROR -> {e}")


if __name__ == "__main__":
    main()
    