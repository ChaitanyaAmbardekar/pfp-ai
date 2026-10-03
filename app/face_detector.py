"""Face detection with OpenCV's YuNet (CPU-only, ~340 KB model)."""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = (Path(__file__).resolve().parent.parent
              / "models" / "pretrained" / "face_detection_yunet_2023mar.onnx")


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float
    landmarks: list  # 5 points: right eye, left eye, nose, right mouth, left mouth

    @property
    def center(self):
        return (self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class DetectionResult:
    image_w: int
    image_h: int
    faces: list = field(default_factory=list)
    primary_index: int = -1          # -1 means no face found
    ambiguous: bool = False
    note: str = ""

    @property
    def primary(self):
        return self.faces[self.primary_index] if self.primary_index >= 0 else None

    def relative_center(self):
        """Primary face center as fractions of the image (0..1)."""
        f = self.primary
        if f is None:
            return None
        cx, cy = f.center
        return (cx / self.image_w, cy / self.image_h)

    def face_area_ratio(self):
        """face_area / image_area for the primary face."""
        f = self.primary
        if f is None:
            return 0.0
        return (f.w * f.h) / (self.image_w * self.image_h)


def _rank(face, img_w, img_h):
    """Higher = more likely the main subject: big, confident, near center."""
    area = (face.w * face.h) / (img_w * img_h)
    cx, cy = face.center
    dx, dy = (cx / img_w - 0.5), (cy / img_h - 0.5)
    centrality = 1.0 - min(1.0, (dx * dx + dy * dy) ** 0.5 * 1.4)
    return area * face.score * (0.5 + 0.5 * centrality)


def choose_primary(faces, img_w, img_h, ambiguity_ratio: float = 0.6):
    """Pick the primary face. Returns (index, ambiguous, note)."""
    if not faces:
        return -1, False, "No face detected."
    if len(faces) == 1:
        return 0, False, "Single face detected."
    ranks = [_rank(f, img_w, img_h) for f in faces]
    order = sorted(range(len(faces)), key=lambda i: ranks[i], reverse=True)
    best, second = order[0], order[1]
    ambiguous = ranks[second] >= ambiguity_ratio * ranks[best]
    note = (f"{len(faces)} faces detected; the second candidate is nearly as "
            "prominent as the first, so please confirm the main subject."
            if ambiguous else
            f"{len(faces)} faces detected; the largest, most central one was chosen.")
    return best, ambiguous, note


class FaceDetector:
    def __init__(self, model_path=MODEL_PATH, score_threshold: float = 0.7):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model missing: {model_path}\n"
                "Download it with the Invoke-WebRequest command from Step 2.")
        # Input size is a placeholder; we set the real size per image.
        self._det = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 50)

    def detect(self, bgr) -> DetectionResult:
        h, w = bgr.shape[:2]
        self._det.setInputSize((w, h))
        _, raw = self._det.detect(bgr)  # raw is None when nothing is found

        faces = []
        if raw is not None:
            for r in raw:
                x, y, fw, fh = [int(round(v)) for v in r[:4]]
                # Clip boxes that stick out of the image.
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(w, x + fw), min(h, y + fh)
                if x1 <= x0 or y1 <= y0:
                    continue
                lm = [(float(r[4 + 2 * i]), float(r[5 + 2 * i])) for i in range(5)]
                faces.append(Face(x0, y0, x1 - x0, y1 - y0, float(r[14]), lm))

        idx, amb, note = choose_primary(faces, w, h)
        return DetectionResult(w, h, faces, idx, amb, note)