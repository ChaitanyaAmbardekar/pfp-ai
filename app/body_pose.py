"""How much body is really visible? (MediaPipe Pose, Tasks API, CPU-only)

MediaPipe returns a full skeleton even when the body is cut off, and its
visibility score can be confident about points that are not in the photo.
So every landmark must be (a) inside the frame, (b) high visibility and
(c) at a plausible distance from the shoulders measured in FACE HEIGHTS.
All thresholds are provisional; calibrate them on your photos.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

POSE_MODEL = (Path(__file__).resolve().parent.parent
              / "models" / "pretrained" / "pose_landmarker_lite.task")

TH = {
    "vis": 0.8,         # minimum visibility for shoulders / hips / knees
    "ankle_vis": 0.6,   # ankles are often partly occluded
    "edge": 0.98,       # landmark must be above this fraction of image height
    "min_torso": 2.6,   # shoulder->hip distance, in face heights
    "min_knee": 4.0,    # shoulder->knee
    "min_ankle": 5.5,   # shoulder->ankle
}

LANDMARK_IDS = {"nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_hip": 23,
                "r_hip": 24, "l_knee": 25, "r_knee": 26,
                "l_ankle": 27, "r_ankle": 28}


@dataclass
class BodyExtent:
    level: str            # unknown / head_only / head_shoulders / upper_body / full_body
    shoulders: bool
    hips: bool
    knees: bool
    ankles: bool
    landmarks: dict = field(default_factory=dict)   # name -> (x_px, y_px, visibility)
    notes: list = field(default_factory=list)


def classify_landmarks(lm: dict, img_h: float, face_h: float) -> BodyExtent:
    """lm: name -> (x_px, y_px, visibility). Pure function, easy to test."""
    notes = []

    def inside(name, vmin):
        _, y, v = lm[name]
        return v >= vmin and 0 <= y <= TH["edge"] * img_h

    def both(a, b, vmin):
        return inside(a, vmin) and inside(b, vmin)

    sh_y = (lm["l_shoulder"][1] + lm["r_shoulder"][1]) / 2

    def ydist(a, b):  # mean distance below the shoulders, in face heights
        return ((lm[a][1] + lm[b][1]) / 2 - sh_y) / face_h

    shoulders = both("l_shoulder", "r_shoulder", TH["vis"])
    hips = shoulders and both("l_hip", "r_hip", TH["vis"])
    if hips and ydist("l_hip", "r_hip") < TH["min_torso"]:
        hips = False
        notes.append("hip landmarks look estimated, not real (torso too short for the face size)")
    knees = hips and both("l_knee", "r_knee", TH["vis"]) \
        and ydist("l_knee", "r_knee") >= TH["min_knee"]
    ankles = knees and both("l_ankle", "r_ankle", TH["ankle_vis"]) \
        and ydist("l_ankle", "r_ankle") >= TH["min_ankle"]

    if ankles:
        level = "full_body"
    elif hips:
        level = "upper_body"
    elif shoulders:
        level = "head_shoulders"
    else:
        level = "head_only"
    return BodyExtent(level, shoulders, hips, knees, ankles, dict(lm), notes)


class PoseDetector:
    def __init__(self, model_path=POSE_MODEL):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Pose model missing: {model_path}")
        import mediapipe as mp                      # lazy: keeps tests light
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self._mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE, num_poses=3)
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, bgr, face) -> BodyExtent:
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(image)
        if not res.pose_landmarks:
            return BodyExtent("unknown", False, False, False, False, {},
                              ["no body found by the pose model"])

        # Several people may be found: use the one whose nose is closest to the primary face.
        fcx, fcy = face.center
        best, best_d = None, 1e18
        for pose in res.pose_landmarks:
            nose = pose[0]
            d = ((nose.x * w - fcx) ** 2 + (nose.y * h - fcy) ** 2) ** 0.5
            if d < best_d:
                best, best_d = pose, d
        if best_d > face.h:
            return BodyExtent("unknown", False, False, False, False, {},
                              ["pose found, but not on the primary face"])

        lm = {n: (best[i].x * w, best[i].y * h, float(getattr(best[i], "visibility", 0.0) or 0.0))
              for n, i in LANDMARK_IDS.items()}
        return classify_landmarks(lm, h, face.h)

    def close(self):
        self._landmarker.close()
