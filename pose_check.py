"""Check that MediaPipe Pose (Tasks API) runs here and show how much body it sees."""
import sys

import cv2
import mediapipe as mp

from app.image_io import load_image_bgr

MODEL = r"models\pretrained\pose_landmarker_lite.task"

print("mediapipe:", mp.__version__)
try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except Exception as e:
    raise SystemExit(f"Tasks API import failed: {e}\nSend me this message.")

path = sys.argv[1] if len(sys.argv) > 1 else r"samples\s3.jpeg"
bgr, _ = load_image_bgr(path)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
)
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(image)

if not result.pose_landmarks:
    raise SystemExit(f"{path}: no body found")

lms = result.pose_landmarks[0]
names = {0: "nose", 11: "l_shoulder", 12: "r_shoulder", 23: "l_hip", 24: "r_hip",
         25: "l_knee", 26: "r_knee", 27: "l_ankle", 28: "r_ankle"}
print(f"\n{path}")
for i, n in names.items():
    lm = lms[i]
    print(f"  {n:<11} y={lm.y:5.2f}  visibility={lm.visibility:4.2f}")
    