"""Environment sanity check for PFP-AI. Run this after installing dependencies."""
import sys

def main() -> int:
    print(f"Python : {sys.version.split()[0]}")
    ok = True
    try:
        import numpy as np
        print(f"NumPy  : {np.__version__}")
    except ImportError as e:
        print(f"NumPy  : MISSING ({e})"); ok = False
    try:
        import cv2
        print(f"OpenCV : {cv2.__version__}")
        # YuNet face detector lives in this class (needs OpenCV >= 4.5.4)
        if hasattr(cv2, "FaceDetectorYN"):
            print("YuNet  : FaceDetectorYN available")
        else:
            print("YuNet  : NOT available, upgrade opencv-python"); ok = False
    except ImportError as e:
        print(f"OpenCV : MISSING ({e})"); ok = False
    try:
        import PIL
        print(f"Pillow : {PIL.__version__}")
    except ImportError as e:
        print(f"Pillow : MISSING ({e})"); ok = False

    print("\nRESULT:", "ALL GOOD" if ok else "PROBLEM FOUND")
    return 0 if ok else 1
if __name__ == "__main__":
    raise SystemExit(main())