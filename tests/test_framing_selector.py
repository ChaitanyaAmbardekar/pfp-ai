import numpy as np

from app.body_pose import BodyExtent, classify_landmarks
from app.face_detector import DetectionResult, Face
from app.framing_selector import select_framing


def face_at(x, y, w, h):
    lm = [(x + .30 * w, y + .38 * h), (x + .70 * w, y + .38 * h),
          (x + .50 * w, y + .60 * h), (x + .35 * w, y + .80 * h), (x + .65 * w, y + .80 * h)]
    return Face(x, y, w, h, 0.95, lm)


def lms(sh, hip, knee, ankle, vis=1.0):
    d = {}
    for side, x in (("l", 400), ("r", 600)):
        d[f"{side}_shoulder"] = (x, sh, vis)
        d[f"{side}_hip"] = (x, hip, vis)
        d[f"{side}_knee"] = (x, knee, vis)
        d[f"{side}_ankle"] = (x, ankle, vis)
    return d


def test_full_body_photo_is_recognised():          # numbers like s3
    b = classify_landmarks(lms(768, 934, 1050, 1152), 1280, 57)
    assert b.level == "full_body"


def test_seated_selfie_hips_rejected():            # numbers like s2: confident but fake hips
    b = classify_landmarks(lms(413, 710, 912, 826), 960, 390)
    assert b.level == "head_shoulders" and not b.hips and b.notes


def test_hips_below_frame_rejected():              # numbers like s8
    b = classify_landmarks(lms(640, 1500, 2000, 2600, vis=0.2), 1280, 300)
    assert b.level in ("head_only", "head_shoulders") and not b.hips


def _setup(extra=False):
    img = np.full((1000, 1000, 3), 128, np.uint8)
    faces = [face_at(450, 300, 100, 120)]
    if extra:
        faces.append(face_at(270, 300, 100, 120))
    det = DetectionResult(1000, 1000, faces, 0)
    body = BodyExtent("upper_body", True, True, False, False, {
        "l_hip": (450, 700, 1.0), "r_hip": (550, 700, 1.0)}, [])
    return img, det, body


def test_head_shoulders_wins_when_face_stays_readable():
    img, det, body = _setup()
    cands = select_framing(img, det, body)
    assert cands[0].name == "head_shoulders"


def test_other_faces_in_crop_push_choice_to_tight_crop():
    img, det, body = _setup(extra=True)
    assert select_framing(img, det, body)[0].name == "face"


def test_full_body_unavailable_when_body_not_visible():
    img, det, body = _setup()
    full = [c for c in select_framing(img, det, body) if c.name == "full_body"][0]
    assert not full.available
    
def test_s2_real_numbers_fake_hips_rejected():     # measured from the real s2 photo
    b = classify_landmarks(lms(536, 925, 1170, 1075), 1254, 185)
    assert not b.hips and b.level == "head_shoulders"