import sys
from app.face_detector import FaceDetector
from app.image_io import load_image_bgr
from app.body_pose import PoseDetector

bgr, _ = load_image_bgr(sys.argv[1])
f = FaceDetector().detect(bgr).primary
print("image h,w:", bgr.shape[:2], " face x,y,w,h:", f.x, f.y, f.w, f.h)
p = PoseDetector()
b = p.detect(bgr, f)
print("level:", b.level, "| notes:", b.notes)
for n, (x, y, v) in b.landmarks.items():
    print(f"  {n:<11} x={x:6.0f} y={y:6.0f} vis={v:.2f}")
if b.landmarks:
    sh = (b.landmarks["l_shoulder"][1] + b.landmarks["r_shoulder"][1]) / 2
    hp = (b.landmarks["l_hip"][1] + b.landmarks["r_hip"][1]) / 2
    print("torso in face heights:", round((hp - sh) / f.h, 2))
p.close()
