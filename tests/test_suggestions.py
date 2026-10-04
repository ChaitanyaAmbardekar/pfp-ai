from app.analyzer import DEFAULT_WEIGHTS, PhotoAnalysis
from app.suggestions import apply_preset, build_suggestions


def mk(**kw):
    base = dict(face_dark_fraction=0.0, face_bright_fraction=0.0, face_diff=0.0,
                whole_dynamic_range=180.0, sharp_label="sharp", detail=1.0)
    base.update(kw)
    return PhotoAnalysis(name="a", path="a", weights=dict(DEFAULT_WEIGHTS),
                         sub={"lighting": 50.0}, facts=base)


def test_backlit_face_gets_shadow_suggestion():
    ids = [e.id for e in build_suggestions(mk(face_diff=-70.0))]
    assert "lift_shadows" in ids


def test_clean_photo_gets_no_edit_suggestions():
    assert build_suggestions(mk()) == []      # no crop edit either: best is None


def test_polished_preset_adds_gentle_contrast():
    a = mk()
    edits = apply_preset("polished_confident", build_suggestions(a), a)
    assert any("contrast" in e.params for e in edits)