"""Print quality metrics for one image or every image in samples\\."""
import sys
from pathlib import Path

from app.face_detector import FaceDetector
from app.image_io import SUPPORTED, load_image_bgr
from app.image_quality import analyze_quality


def show(label, q):
    print(f"  [{label}]")
    print(f"    brightness     : {q.brightness:6.1f} / 255   exposure: {q.exposure}")
    print(f"    contrast       : {q.contrast:6.1f}   local: {q.local_contrast:.1f}   range: {q.dynamic_range:.0f}")
    print(f"    sharpness      : {q.sharpness:6.1f}   ({q.sharpness_label})")
    print(f"    noise sigma    : {q.noise:6.2f}   ({q.noise_label})")
    print(f"    crushed/blown  : {q.dark_fraction * 100:.1f}% dark, {q.bright_fraction * 100:.1f}% bright")


def main():
    detector = FaceDetector()
    if len(sys.argv) > 1:
        paths = [Path(sys.argv[1])]
    else:
        paths = sorted(p for p in Path("samples").iterdir()
                       if p.suffix.lower() in SUPPORTED)
    for p in paths:
        try:
            bgr, _ = load_image_bgr(p)
            det = detector.detect(bgr)
            rep = analyze_quality(bgr, det)
            print(f"\n{p.name}  ({len(det.faces)} face(s))")
            show("WHOLE IMAGE", rep.whole)
            if rep.face:
                show("FACE", rep.face)
                print(f"  face vs scene brightness: {rep.face_minus_image_brightness:+.1f}")
            else:
                print("  [FACE] none detected, so no face metrics")
        except Exception as e:
            print(f"\n{p.name}: ERROR -> {e}")


if __name__ == "__main__":
    main()