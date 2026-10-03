from app.face_detector import Face, choose_primary


def mk(x, y, w, h, score=0.95):
    return Face(x, y, w, h, score, [(0.0, 0.0)] * 5)


def test_no_faces():
    assert choose_primary([], 1000, 1000)[0] == -1


def test_single_face():
    idx, amb, _ = choose_primary([mk(400, 300, 200, 200)], 1000, 1000)
    assert idx == 0 and not amb


def test_clear_primary_is_not_ambiguous():
    faces = [mk(100, 100, 60, 60), mk(350, 250, 300, 300)]
    idx, amb, _ = choose_primary(faces, 1000, 1000)
    assert idx == 1 and not amb


def test_two_similar_faces_are_ambiguous():
    faces = [mk(150, 300, 220, 220), mk(600, 300, 220, 220)]
    _, amb, _ = choose_primary(faces, 1000, 1000)
    assert amb
    