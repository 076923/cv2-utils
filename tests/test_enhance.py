import numpy as np
import pytest

import cv2u as cv


@pytest.fixture
def low_contrast():
    rng = np.random.default_rng(5)
    return rng.integers(90, 140, (64, 64, 3), dtype=np.uint8)


# --- equalizeHistEx ----------------------------------------------------------------


def test_equalizeHistEx_gray_matches_equalizeHist(low_contrast):
    gray = cv.cvtColor(low_contrast, cv.COLOR_BGR2GRAY)
    np.testing.assert_array_equal(cv.equalizeHistEx(gray), cv.equalizeHist(gray))


def test_equalizeHistEx_color_stretches_lightness(low_contrast):
    out = cv.equalizeHistEx(low_contrast)
    assert out.shape == low_contrast.shape
    lightness = cv.cvtColor(out, cv.COLOR_BGR2LAB)[..., 0]
    assert int(lightness.max()) - int(lightness.min()) > 200


def test_equalizeHistEx_keeps_hue():
    # A dim, low-contrast red gradient must stay red (equalizing B, G and R separately would not guarantee that).
    img = np.zeros((10, 50, 3), np.uint8)
    img[..., 2] = np.linspace(60, 90, 50, dtype=np.uint8)
    out = cv.equalizeHistEx(img)
    assert (out[..., 2].astype(int) >= out[..., 1]).all()
    assert (out[..., 2].astype(int) >= out[..., 0]).all()


def test_equalizeHistEx_bgra_keeps_alpha(low_contrast):
    bgra = np.dstack([low_contrast, np.full(low_contrast.shape[:2], 123, np.uint8)])
    out = cv.equalizeHistEx(bgra, clahe=True)
    assert out.shape == bgra.shape
    assert (out[..., 3] == 123).all()


def test_equalizeHistEx_clahe_matches_opencv(low_contrast):
    gray = cv.cvtColor(low_contrast, cv.COLOR_BGR2GRAY)
    expected = cv.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    np.testing.assert_array_equal(cv.equalizeHistEx(gray, clahe=True, clipLimit=3.0, tileGridSize=(4, 4)), expected)


@pytest.mark.parametrize("src", [np.zeros((4, 4), np.float32), np.zeros((4, 4, 2), np.uint8)])
def test_equalizeHistEx_validation(src):
    with pytest.raises(ValueError):
        cv.equalizeHistEx(src)


# --- unsharpMask -------------------------------------------------------------------


@pytest.fixture
def step_edge():
    img = np.full((20, 40), 80, np.uint8)
    img[:, 20:] = 160
    return img


def test_unsharpMask_increases_edge_contrast(step_edge):
    out = cv.unsharpMask(step_edge, sigma=2, amount=1)
    assert out[:, 19].mean() < 80  # undershoot on the dark side
    assert out[:, 20].mean() > 160  # overshoot on the bright side
    assert out[0, 0] == 80  # flat areas stay the same


def test_unsharpMask_amount_zero_is_identity(step_edge):
    np.testing.assert_array_equal(cv.unsharpMask(step_edge, amount=0), step_edge)


def test_unsharpMask_threshold_protects_flat_noise():
    rng = np.random.default_rng(0)
    noisy = (128 + rng.integers(-2, 3, (30, 30))).astype(np.uint8)
    np.testing.assert_array_equal(cv.unsharpMask(noisy, threshold=10), noisy)


def test_unsharpMask_color(low_contrast):
    assert cv.unsharpMask(low_contrast).shape == low_contrast.shape


@pytest.mark.parametrize("kwargs", [{"sigma": 0}, {"amount": -1}, {"threshold": -1}])
def test_unsharpMask_validation(step_edge, kwargs):
    with pytest.raises(ValueError):
        cv.unsharpMask(step_edge, **kwargs)
