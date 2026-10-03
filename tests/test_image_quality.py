import cv2
import numpy as np

from app.image_quality import analyze_region, estimate_noise


def textured(seed=0, size=256):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (size, size), dtype=np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def test_blur_lowers_sharpness():
    sharp = textured()
    blurry = cv2.GaussianBlur(sharp, (0, 0), 4)
    assert analyze_region(sharp).sharpness > 5 * analyze_region(blurry).sharpness


def test_exposure_labels():
    assert analyze_region(np.full((200, 200), 10, np.uint8)).exposure == "underexposed"
    assert analyze_region(np.full((200, 200), 250, np.uint8)).exposure == "overexposed"
    assert analyze_region(np.full((200, 200), 128, np.uint8)).exposure == "balanced"


def test_noise_estimate_is_close():
    rng = np.random.default_rng(1)
    flat = np.full((300, 300), 128, np.float32)
    noisy = np.clip(flat + rng.normal(0, 10, flat.shape), 0, 255).astype(np.uint8)
    assert 7 < estimate_noise(noisy) < 13
    assert estimate_noise(np.full((300, 300), 128, np.uint8)) < 0.5


def test_contrast_ordering():
    flat = np.full((200, 200), 128, np.uint8)
    ramp = np.tile(np.linspace(0, 255, 200).astype(np.uint8), (200, 1))
    assert analyze_region(ramp).contrast > analyze_region(flat).contrast