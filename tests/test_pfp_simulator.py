import numpy as np

from app.face_detector import Face
from app.pfp_simulator import (circle_mask, contact_sheet, crop_square, evaluate,
                               plan_square_crop, to_circle)


def face_at(x, y, w, h):
    lm = [(x + .30 * w, y + .38 * h), (x + .70 * w, y + .38 * h),
          (x + .50 * w, y + .60 * h), (x + .35 * w, y + .80 * h), (x + .65 * w, y + .80 * h)]
    return Face(x, y, w, h, 0.95, lm)


def test_crop_stays_inside_image():
    for x, y in [(0, 0), (880, 680), (450, 0), (0, 400)]:
        p = plan_square_crop(1000, 800, face_at(x, y, 100, 120))
        assert p.x >= 0 and p.y >= 0 and p.x + p.side <= 1000 and p.y + p.side <= 800


def test_face_aware_crop_contains_face():
    f = face_at(400, 300, 100, 120)
    p = plan_square_crop(1000, 1000, f)
    assert p.x <= f.x and p.y <= f.y
    assert p.x + p.side >= f.x + f.w and p.y + p.side >= f.y + f.h


def test_circle_mask_and_alpha():
    m = circle_mask(64)
    assert m[32, 32] == 255 and m[0, 0] == 0 and m[63, 63] == 0
    c = to_circle(np.full((200, 200, 3), 100, np.uint8), 64)
    assert c.shape == (64, 64, 4) and c[0, 0, 3] == 0 and c[32, 32, 3] == 255


def test_face_aware_beats_center_for_small_off_center_face():
    img = np.full((1000, 1000, 3), 128, np.uint8)
    f = face_at(60, 70, 120, 150)
    pf, pc = plan_square_crop(1000, 1000, f, "face"), plan_square_crop(1000, 1000, f, "center")
    rf = evaluate(crop_square(img, pf), pf, f)
    rc = evaluate(crop_square(img, pc), pc, f)
    assert rf.robustness > rc.robustness + 20


def test_tiny_source_face_is_penalised_and_flagged():
    img = np.full((1000, 1000, 3), 128, np.uint8)
    big, tiny = face_at(350, 350, 300, 360), face_at(480, 480, 30, 36)
    pb, pt = plan_square_crop(1000, 1000, big), plan_square_crop(1000, 1000, tiny)
    rb = evaluate(crop_square(img, pb), pb, big)
    rt = evaluate(crop_square(img, pt), pt, tiny)
    assert rb.robustness > rt.robustness
    assert "crop_low_resolution" in rt.flags and "crop_low_resolution" not in rb.flags


def test_no_face_gives_zero_and_flag():
    img = np.full((500, 500, 3), 128, np.uint8)
    p = plan_square_crop(500, 500, None)
    r = evaluate(crop_square(img, p), p, None)
    assert p.mode == "center" and r.robustness == 0 and "no_face_detected" in r.flags


def test_contact_sheet_is_bgr_image():
    sheet = contact_sheet({s: to_circle(np.full((300, 300, 3), 90, np.uint8), s)
                           for s in (64, 128)})
    assert sheet.ndim == 3 and sheet.shape[2] == 3

def test_wider_framing_gives_bigger_crop():
    f = face_at(450, 300, 100, 120)
    sides = [plan_square_crop(2000, 2000, f, m).side
             for m in ("face", "head_shoulders", "upper_body")]
    assert sides[0] < sides[1] < sides[2]