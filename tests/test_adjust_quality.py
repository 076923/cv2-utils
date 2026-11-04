import cv2
import numpy as np
import pytest

import cv2u as cv

# --- hexToBGR / bgrToHex ------------------------------------------------------------------


@pytest.mark.parametrize("text", ["#ff8800", "ff8800", "#F80", " #ff8800 "])
def test_hexToBGR(text):
    assert cv.hexToBGR(text) == (0, 136, 255)


def test_hex_alpha_and_roundtrip():
    assert cv.hexToBGR("#ff880080") == (0, 136, 255, 128)
    assert cv.hexToBGR("#f808") == (0, 136, 255, 136)
    assert cv.bgrToHex((0, 136, 255)) == "#ff8800"
    assert cv.bgrToHex((0, 136, 255, 128)) == "#ff880080"
    for color in ((12, 34, 56), (255, 0, 128, 7)):
        assert cv.hexToBGR(cv.bgrToHex(color)) == color


@pytest.mark.parametrize("bad", ["#ff888", "#gg0000", "", "#1234567"])  # "#ff88" would be a valid #rgba
def test_hexToBGR_validation(bad):
    with pytest.raises(ValueError):
        cv.hexToBGR(bad)


@pytest.mark.parametrize("bad", [(1, 2), (0, 0, 256), (1, 2, 3, 4, 5)])
def test_bgrToHex_validation(bad):
    with pytest.raises(ValueError):
        cv.bgrToHex(bad)


# --- adjustBrightnessContrast / adjustHSV --------------------------------------------------


@pytest.fixture
def photo():
    rng = np.random.default_rng(0)
    return rng.integers(40, 220, (30, 40, 3), dtype=np.uint8)


def test_adjustBrightnessContrast(photo):
    np.testing.assert_array_equal(cv.adjustBrightnessContrast(photo), photo)
    brighter = cv.adjustBrightnessContrast(photo, brightness=20)
    np.testing.assert_array_equal(brighter, np.clip(photo.astype(int) + 20, 0, 255))
    assert (cv.adjustBrightnessContrast(photo, contrast=0) == 128).all()
    stronger = cv.adjustBrightnessContrast(photo, contrast=2)
    assert stronger.std() > photo.std()
    assert stronger[photo == 128].tolist() == photo[photo == 128].tolist()  # pivots around mid-gray


def test_adjustBrightnessContrast_keeps_alpha_and_validates(photo):
    bgra = np.dstack([photo, np.full(photo.shape[:2], 9, np.uint8)])
    assert (cv.adjustBrightnessContrast(bgra, brightness=50)[..., 3] == 9).all()
    with pytest.raises(ValueError):
        cv.adjustBrightnessContrast(photo, contrast=-1)
    with pytest.raises(ValueError):
        cv.adjustBrightnessContrast(photo.astype(np.float32))


def test_adjustHSV_hue_saturation_value():
    red = np.full((2, 2, 3), (0, 0, 255), np.uint8)
    np.testing.assert_array_equal(cv.adjustHSV(red), red)
    assert tuple(cv.adjustHSV(red, hue=120)[0, 0]) == (0, 255, 0)  # red -> green
    assert tuple(cv.adjustHSV(red, hue=-120)[0, 0]) == (255, 0, 0)  # wraps around: red -> blue
    gray = cv.adjustHSV(red, saturation=0)[0, 0]
    assert gray[0] == gray[1] == gray[2]
    assert tuple(cv.adjustHSV(red, value=0.5)[0, 0]) == (0, 0, 128)


def test_adjustHSV_alpha_and_validation(photo):
    bgra = np.dstack([photo, np.full(photo.shape[:2], 3, np.uint8)])
    assert (cv.adjustHSV(bgra, hue=30)[..., 3] == 3).all()
    with pytest.raises(ValueError):
        cv.adjustHSV(photo[..., 0])
    with pytest.raises(ValueError):
        cv.adjustHSV(photo, saturation=-1)


# --- autoContrast / whiteBalance ------------------------------------------------------------


def test_autoContrast_stretches_despite_outliers():
    rng = np.random.default_rng(1)
    dull = rng.integers(100, 151, (50, 50), dtype=np.uint8)
    dull[0, 0], dull[0, 1] = 0, 255  # two outliers: cv.normalize cannot stretch at all
    np.testing.assert_array_equal(cv.normalize(dull, None, 0, 255, cv.NORM_MINMAX), dull)
    out = cv.autoContrast(dull)
    assert np.percentile(out, 2) < 10
    assert np.percentile(out, 98) > 245


def test_autoContrast_perChannel_removes_a_cast():
    rng = np.random.default_rng(2)
    base = rng.integers(60, 200, (40, 40), dtype=np.uint8)
    cast = np.dstack([base // 2 + 20, base, base])  # weak blue channel
    shared = cv.autoContrast(cast)
    separate = cv.autoContrast(cast, perChannel=True)
    assert abs(float(separate[..., 0].mean()) - float(separate[..., 1].mean())) < 10
    assert abs(float(shared[..., 0].mean()) - float(shared[..., 1].mean())) > 30  # colors kept as they were


def test_autoContrast_flat_and_validation():
    flat = np.full((5, 5), 77, np.uint8)
    np.testing.assert_array_equal(cv.autoContrast(flat), flat)
    with pytest.raises(ValueError):
        cv.autoContrast(flat, clip=0.5)


@pytest.fixture
def green_object():
    """A green object with a white part on a black background, like a product photo."""
    img = np.zeros((60, 80, 3), np.uint8)
    img[10:50, 10:60] = (40, 160, 50)
    img[20:30, 15:55] = (235, 235, 235)  # white laces
    return img


def test_whiteBalance_restores_white(green_object):
    cast = np.clip(green_object.astype(float) * (1.25, 1.0, 0.75), 0, 255).astype(np.uint8)
    white = cv.whiteBalance(cast)[25, 30].astype(int)  # default: whitePatch
    assert white.max() - white.min() <= 2  # the white part is neutral again


def test_whiteBalance_whitePatch_leaves_a_neutral_photo_alone(green_object):
    # gray-world assumes the average color is gray, so it wrongly "corrects" a mostly green image (e.g. to pink).
    assert np.abs(cv.whiteBalance(green_object).astype(int) - green_object).max() <= 2
    shifted = cv.whiteBalance(green_object, method="grayWorld")
    assert np.abs(shifted.astype(int) - green_object).max() > 30


def test_whiteBalance_grayWorld_equalizes_the_means(photo):
    cast = np.clip(photo.astype(float) * (1.3, 1.0, 0.7), 0, 255).astype(np.uint8)
    means = cv.whiteBalance(cast, method="grayWorld").reshape(-1, 3).mean(axis=0)
    assert means.max() - means.min() < 3


def test_whiteBalance_validation(photo):
    with pytest.raises(ValueError):
        cv.whiteBalance(photo[..., 0])
    with pytest.raises(ValueError, match="method"):
        cv.whiteBalance(photo, method="auto")


# --- sharpnessScore / ssim / diffMask --------------------------------------------------------


def test_sharpnessScore_detects_blur(photo):
    blurred = cv2.GaussianBlur(photo, (0, 0), 3)
    assert cv.sharpnessScore(photo) > 10 * cv.sharpnessScore(blurred)
    assert cv.sharpnessScore(np.full((10, 10), 50, np.uint8)) == 0


def test_ssim_identical_and_ordering(photo):
    assert cv.ssim(photo, photo) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    light = np.clip(photo + rng.normal(0, 5, photo.shape), 0, 255).astype(np.uint8)
    heavy = np.clip(photo + rng.normal(0, 30, photo.shape), 0, 255).astype(np.uint8)
    assert 1.0 > cv.ssim(photo, light) > cv.ssim(photo, heavy)


def test_ssim_matches_the_formula_on_flat_images():
    a, b = np.full((20, 20), 100, np.uint8), np.full((20, 20), 110, np.uint8)
    c1 = (0.01 * 255) ** 2
    expected = (2 * 100 * 110 + c1) / (100**2 + 110**2 + c1)  # zero variance: the structure term is 1
    assert cv.ssim(a, b) == pytest.approx(expected)
    assert cv.ssim(a.astype(np.float32) / 255, b.astype(np.float32) / 255) == pytest.approx(expected, rel=1e-6)


def test_ssim_validation(photo):
    with pytest.raises(ValueError):
        cv.ssim(photo, photo[:10])


def test_diffMask(photo):
    changed = photo.copy()
    changed[5:15, 10:20] = 0
    changed[25, 35] = 255  # a single noisy pixel
    mask = cv.diffMask(photo, changed)
    assert mask.shape == photo.shape[:2]
    assert mask[10, 15] == 255
    assert mask[0, 0] == 0
    cleaned = cv.diffMask(photo, changed, minArea=5)
    assert cleaned[10, 15] == 255
    assert cleaned[25, 35] == 0
    assert not cv.diffMask(photo, photo).any()
    with pytest.raises(ValueError):
        cv.diffMask(photo, photo[:5])
