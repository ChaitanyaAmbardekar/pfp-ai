from types import SimpleNamespace

from app.analyzer import (DEFAULT_WEIGHTS, Penalty, PhotoAnalysis, background_score, combine,
                          face_visibility_score, lighting_score, load_weights, sharpness_score)
from app.explain import explain_comparison
from app.ranking import rank


def face_q(dark=0.0, bright=0.0):
    return SimpleNamespace(dark_fraction=dark, bright_fraction=bright)


def mk(name, overall, sub=None, problem=""):
    return PhotoAnalysis(name=name, path=name, weights=dict(DEFAULT_WEIGHTS),
                         overall=overall, sub=sub or {"lighting": 50.0}, problem=problem)


def test_sharpness_trusts_only_reliable_faces():
    assert sharpness_score(5, 1.0) == 0.0
    assert sharpness_score(300, 1.0) == 1.0
    assert sharpness_score(5, 0.0) == 0.5          # no real pixels: neutral, not "soft"


def test_backlit_face_scores_lower():
    assert lighting_score(face_q(), 10) > lighting_score(face_q(), -100) + 0.2


def test_brighter_face_is_not_penalised():
    assert lighting_score(face_q(), 50) == lighting_score(face_q(), 0)


def test_combine_renormalises_and_applies_penalties():
    w = {"a": 0.5, "b": 0.5}
    assert combine({"a": 100, "b": 0}, w, []) == 50
    assert combine({"a": 100, "b": 100}, w, [Penalty("x", 20, "t")]) == 80
    assert combine({"a": 10, "b": 10}, w, [Penalty("x", 50, "t")]) == 0


def test_load_weights_ignores_bad_entries(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"weights": {"lighting": 0.5, "bogus": 3, "framing": -1}}')
    w = load_weights(p)
    assert w["lighting"] == 0.5 and "bogus" not in w
    assert w["framing"] == DEFAULT_WEIGHTS["framing"]


def test_face_visibility_drops_when_not_redetected():
    ok = face_visibility_score(0.95, 1.0, True, 10)
    bad = face_visibility_score(0.95, 1.0, False, 10)
    assert ok > 0.95 and bad < ok - 0.2


def test_background_neutral_when_not_visible():
    assert background_score(None) == 0.7
    assert background_score(SimpleNamespace(weight=0.0, clutter=0, separation=0, interest=0)) == 0.7


def test_rank_orders_by_score_and_puts_failures_last():
    r = rank([mk("a", 60), mk("bad", 0, problem="no face"), mk("b", 80)])
    assert [x.name for x in r] == ["b", "a", "bad"]


def test_comparison_names_the_biggest_driver():
    a = mk("a", 70, {"lighting": 90.0, "sharpness": 60.0})
    b = mk("b", 50, {"lighting": 30.0, "sharpness": 55.0})
    assert "lighting" in explain_comparison(a, b).lower()
    