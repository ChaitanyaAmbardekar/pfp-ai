from app.advice import build_advice
from app.analyzer import DEFAULT_WEIGHTS, PhotoAnalysis


def mk(**kw):
    base = dict(extra_faces=0, detail=1.0, face_px_original=300, face_diff=0.0,
                face_dark_fraction=0.0, bg_weight=0.0, bg_clutter=0.0)
    base.update(kw)
    return PhotoAnalysis(name="a", path="a", weights=dict(DEFAULT_WEIGHTS), facts=base)


def test_other_people_gets_alone_advice():
    assert any("only you" in t for t in build_advice(mk(extra_faces=2)))


def test_clean_photo_has_no_advice():
    assert build_advice(mk()) == []
    