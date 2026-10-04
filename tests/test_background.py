import cv2
import numpy as np

from app.background import analyze_background
from app.face_detector import Face
from app.pfp_simulator import SquareCrop

FACE = Face(150, 120, 100, 120, 0.95, [(0.0, 0.0)] * 5)
PLAN = SquareCrop(0, 0, 400, "face", False)
PERSON = (40, 40, 200)


def with_person(bg):
    img = bg.copy()
    img[78:, 60:340] = PERSON          # the rectangle the analyser treats as "person"
    return img


def plain():
    return with_person(np.full((400, 400, 3), 128, np.uint8))


def rich():
    img = np.zeros((400, 400, 3), np.uint8)
    xs = np.linspace(0, 1, 400)
    img[:, :, 0] = (255 * (1 - xs))[None, :]
    img[:, :, 1] = (255 * np.sin(np.pi * xs))[None, :]
    img[:, :, 2] = (255 * xs)[None, :]
    for i in range(5):
        cv2.rectangle(img, (20 + 70 * i, 20), (60 + 70 * i, 90 + 20 * i), (30, 30, 30), -1)
    return with_person(img)


def checker():
    yy, xx = np.indices((400, 400))
    chk = (((yy // 16) + (xx // 16)) % 2 * 255).astype(np.uint8)
    return with_person(np.dstack([chk] * 3))


def test_blank_background_has_no_scenery():
    assert analyze_background(plain(), PLAN, FACE).scenery < 0.05


def test_colourful_background_scores_more_than_blank():
    r, p = analyze_background(rich(), PLAN, FACE), analyze_background(plain(), PLAN, FACE)
    assert r.interest > 0.3 and r.scenery > p.scenery


def test_busy_checkerboard_is_clutter():
    info = analyze_background(checker(), PLAN, FACE)
    assert info.clutter > 0.8 and info.scenery < 0.1


def test_person_blending_into_background_has_low_separation():
    flat = np.full((400, 400, 3), 128, np.uint8)       # person and background identical
    assert analyze_background(flat, PLAN, FACE).separation < 0.05