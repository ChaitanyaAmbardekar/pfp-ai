import numpy as np

from app.editor import LIMITS, Edit, LocalEditor


def img(v, size=200):
    return np.full((size, size, 3), v, np.uint8)


def edit(**params):
    return Edit("e", "light", "e", params, "")


def test_no_edits_returns_centre_square_unchanged():
    src = np.random.default_rng(0).integers(0, 255, (300, 200, 3), dtype=np.uint8)
    out = LocalEditor().apply(src, [])
    assert out.shape == (200, 200, 3) and np.array_equal(out, src[50:250, 0:200])


def test_lift_shadows_brightens_dark_pixels():
    assert LocalEditor().apply(img(30), [edit(shadows=0.8)]).mean() > 35


def test_highlights_reduction_darkens_bright_pixels():
    assert LocalEditor().apply(img(240), [edit(highlights=0.8)]).mean() < 235


def test_contrast_increases_spread():
    g = np.tile(np.linspace(60, 200, 200).astype(np.uint8), (200, 1))
    src = np.dstack([g] * 3)
    assert LocalEditor().apply(src, [edit(contrast=0.3)]).std() > src.std()


def test_extreme_values_are_clipped_to_limits():
    a = LocalEditor().apply(img(100), [edit(brightness=5.0)])
    b = LocalEditor().apply(img(100), [edit(brightness=LIMITS["brightness"][1])])
    assert np.array_equal(a, b)


def test_crop_edit_can_be_switched_off():
    src = np.zeros((300, 300, 3), np.uint8)
    src[0:100, 0:100] = 255
    crop = Edit("crop", "crop", "Crop", {"x": 0, "y": 0, "side": 100}, "")
    assert LocalEditor().apply(src, [crop]).mean() == 255
    crop.enabled = False
    assert LocalEditor().apply(src, [crop]).mean() < 255   # falls back to the centre square
    