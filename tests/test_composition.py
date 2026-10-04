from app.body_pose import BodyExtent
from app.composition import analyze_composition, source_detail
from app.face_detector import Face


def face_at(x, y, w, h):
    return Face(x, y, w, h, 0.95, [(0.0, 0.0)] * 5)


REGION = (0, 0, 1000, 1000)


def test_balanced_centered_crop():
    c = analyze_composition(face_at(350, 300, 300, 360), REGION)
    assert c.size_label == "acceptable" and c.headroom_label == "balanced"
    assert c.h_pos == "center" and c.v_pos == "middle" and c.adjustments == []


def test_too_little_headroom_suggests_moving_up():
    c = analyze_composition(face_at(400, 60, 200, 200), REGION)
    assert c.headroom_label == "too_little"
    assert any(a.kind == "move_crop_up" for a in c.adjustments)


def test_excessive_headroom_suggests_moving_down():
    c = analyze_composition(face_at(450, 500, 100, 100), REGION)
    assert c.headroom_label == "excessive"
    assert any(a.kind == "move_crop_down" for a in c.adjustments)


def test_small_face_suggests_zoom_in():
    c = analyze_composition(face_at(475, 450, 40, 50), REGION)
    assert c.size_label == "too_small"
    assert any(a.kind == "zoom_in" and a.amount > 2 for a in c.adjustments)


def test_off_center_face_suggests_shift():
    c = analyze_composition(face_at(700, 300, 200, 240), REGION)
    assert c.h_pos == "right"
    assert any(a.kind == "shift_crop_right" for a in c.adjustments)


def test_face_touching_edge_is_excessively_cropped():
    c = analyze_composition(face_at(0, 300, 300, 360), REGION)
    assert "left" in c.edge_cut and c.size_label == "excessively_cropped"


def test_source_detail_uses_original_pixels():
    f = face_at(0, 0, 40, 50)
    assert source_detail(f, 1.0) < 0.2
    assert source_detail(f, 3.0) == 1.0


def test_visible_level_depends_on_region_bottom():
    lm = {"l_shoulder": (400, 400, 1), "r_shoulder": (600, 400, 1),
          "l_hip": (420, 700, 1), "r_hip": (580, 700, 1)}
    body = BodyExtent("upper_body", True, True, False, False, lm, [])
    f = face_at(450, 250, 100, 120)
    assert analyze_composition(f, (0, 0, 1000, 800), body).visible == "upper_body"
    assert analyze_composition(f, (0, 0, 1000, 600), body).visible == "head_shoulders"