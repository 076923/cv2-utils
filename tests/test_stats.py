import cv2
import numpy as np
import pytest

from cv2u.imgproc import stats
from cv2u.imgproc.stats import (
    ImageStats,
    brightnessScore,
    colorfulness,
    contrastScore,
    dominantHue,
    entropy,
    estimateNoise,
    exposureScore,
    histogramCompare,
    histogramEx,
    imageStats,
    isBlank,
    isBlurry,
    isGrayscale,
    mae,
    meanColor,
    mse,
)


@pytest.fixture
def checkerboard():
    tile = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    return np.tile(tile, (32, 32))


@pytest.fixture
def photo():
    rng = np.random.default_rng(0)
    img = np.zeros((96, 128, 3), dtype=np.uint8)
    cv2.circle(img, (64, 48), 30, (40, 160, 230), -1)
    cv2.rectangle(img, (5, 5), (40, 90), (200, 50, 10), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def test_public_names_not_in_cv2():
    for name in stats.__all__:
        assert not hasattr(cv2, name), name


# --- isGrayscale / isBlank / isBlurry -------------------------------------------------------


def test_isGrayscale():
    gray = np.full((8, 8), 7, dtype=np.uint8)
    assert isGrayscale(gray)
    assert isGrayscale(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    assert isGrayscale(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA))
    color = np.zeros((8, 8, 3), dtype=np.uint8)
    color[..., 2] = 3
    assert not isGrayscale(color)
    assert isGrayscale(color, tolerance=3)
    bgra = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
    bgra[..., 3] = 0
    assert isGrayscale(bgra)  # alpha ignored


def test_isBlank(photo):
    assert isBlank(np.full((10, 10), 128, dtype=np.uint8))
    assert isBlank(np.full((10, 10, 3), (255, 0, 0), dtype=np.uint8))
    assert isBlank(np.zeros((4, 4), dtype=np.float32))
    assert not isBlank(photo)
    almost = np.full((10, 10), 128, dtype=np.uint8)
    almost[0, 0] = 130
    assert isBlank(almost)
    assert not isBlank(almost, tolerance=0)


def test_isBlurry(checkerboard):
    assert not isBlurry(checkerboard)
    assert isBlurry(cv2.GaussianBlur(checkerboard, (15, 15), 5))
    assert isBlurry(np.zeros((16, 16, 3), dtype=np.uint8))


@pytest.mark.parametrize("fn", [isGrayscale, isBlank])
def test_tolerance_validation(fn):
    with pytest.raises(ValueError, match="tolerance"):
        fn(np.zeros((4, 4), np.uint8), tolerance=-1)


def test_isBlurry_validation():
    with pytest.raises(ValueError, match="threshold"):
        isBlurry(np.zeros((4, 4), np.uint8), threshold=float("nan"))


# --- brightness / contrast / colorfulness / exposure / entropy ------------------------------


def test_brightnessScore():
    assert brightnessScore(np.zeros((4, 4), np.uint8)) == 0.0
    assert brightnessScore(np.full((4, 4, 3), 255, np.uint8)) == pytest.approx(1.0)
    assert brightnessScore(np.full((4, 4), 0.5, np.float32)) == pytest.approx(0.5)
    assert brightnessScore(np.full((4, 4), 65535, np.uint16)) == pytest.approx(1.0)


def test_contrastScore(checkerboard):
    assert contrastScore(np.full((4, 4), 9, np.uint8)) == 0.0
    assert contrastScore(checkerboard) == pytest.approx(0.5)
    assert contrastScore(cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2BGR)) == pytest.approx(0.5)


def test_colorfulness(photo):
    assert colorfulness(np.full((4, 4), 100, np.uint8)) == 0.0
    assert colorfulness(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)) == 0.0
    gray3 = cv2.cvtColor(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert colorfulness(gray3) == pytest.approx(0.0)
    vivid = np.zeros((8, 8, 3), np.uint8)
    vivid[:, :4] = (0, 0, 255)
    vivid[:, 4:] = (255, 0, 0)
    assert colorfulness(vivid) > colorfulness(photo) > 0
    # the same picture as float gives the same score
    assert colorfulness(photo.astype(np.float32) / 255) == pytest.approx(colorfulness(photo), rel=1e-3)
    assert colorfulness(cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)) == pytest.approx(colorfulness(photo))


def test_exposureScore():
    img = np.zeros((10, 10), np.uint8)
    img[:3] = 255
    img[3:5] = 128
    under, over = exposureScore(img)
    assert under == pytest.approx(0.5)
    assert over == pytest.approx(0.3)
    assert exposureScore(np.full((4, 4, 3), 128, np.uint8)) == (0.0, 0.0)
    assert exposureScore(img, lowThreshold=0.6, highThreshold=0.6) == (pytest.approx(0.7), pytest.approx(0.3))


@pytest.mark.parametrize(
    "kwargs",
    [{"lowThreshold": -0.1}, {"highThreshold": 1.5}, {"lowThreshold": 0.9, "highThreshold": 0.1}],
)
def test_exposureScore_validation(kwargs):
    with pytest.raises(ValueError, match="Threshold"):
        exposureScore(np.zeros((4, 4), np.uint8), **kwargs)


def test_entropy(checkerboard, photo):
    assert entropy(np.full((8, 8), 42, np.uint8)) == 0.0
    assert entropy(checkerboard) == pytest.approx(1.0)
    ramp = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    assert entropy(ramp) == pytest.approx(8.0)
    assert entropy(ramp, bins=16) == pytest.approx(4.0)
    assert 0 < entropy(photo) < 8
    assert entropy(ramp.astype(np.float32) / 255) == pytest.approx(8.0)


def test_entropy_validation():
    with pytest.raises(ValueError, match="bins"):
        entropy(np.zeros((4, 4), np.uint8), bins=0)


# --- estimateNoise ------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [5.0, 15.0])
def test_estimateNoise(sigma):
    rng = np.random.default_rng(1)
    base = np.full((200, 200), 128.0)
    noisy = base + rng.normal(0, sigma, base.shape)
    assert estimateNoise(noisy) == pytest.approx(sigma, rel=0.1)
    noisy8 = np.clip(noisy, 0, 255).astype(np.uint8)
    assert estimateNoise(noisy8) == pytest.approx(sigma, rel=0.1)
    assert estimateNoise(cv2.cvtColor(noisy8, cv2.COLOR_GRAY2BGR)) == pytest.approx(sigma, rel=0.1)


def test_estimateNoise_clean_is_low():
    clean = np.zeros((64, 64), np.uint8)
    cv2.circle(clean, (32, 32), 20, 200, -1)
    assert estimateNoise(cv2.GaussianBlur(clean, (7, 7), 2)) < 2
    assert estimateNoise(np.full((5, 5), 10, np.uint8)) == 0.0


def test_estimateNoise_validation():
    with pytest.raises(ValueError, match="3x3"):
        estimateNoise(np.zeros((2, 5), np.uint8))


# --- mse / mae ----------------------------------------------------------------------------


def test_mse_mae():
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.full((4, 4, 3), 2, np.uint8)
    assert mse(a, a) == 0.0
    assert mse(a, b) == 4.0
    assert mae(a, b) == 2.0
    assert mse(b, a) == 4.0  # no uint8 wrap-around
    assert mae(np.zeros(3), np.array([1.0, -2.0, 3.0])) == pytest.approx(2.0)


@pytest.mark.parametrize("fn", [mse, mae])
def test_pair_validation(fn):
    with pytest.raises(ValueError, match="same shape"):
        fn(np.zeros((4, 4)), np.zeros((4, 5)))
    with pytest.raises(ValueError, match="empty"):
        fn(np.zeros((0, 4)), np.zeros((0, 4)))


# --- histogramEx / histogramCompare -------------------------------------------------------


def test_histogramEx(photo):
    hist = histogramEx(photo)
    assert hist.shape == (3, 256)
    assert hist.dtype == np.float64
    assert hist.sum(axis=1).tolist() == [photo.shape[0] * photo.shape[1]] * 3
    for c in range(3):
        expected = cv2.calcHist([photo], [c], None, [256], [0, 256]).ravel()
        assert np.array_equal(hist[c], expected)
    gray = histogramEx(np.zeros((4, 4), np.uint8), bins=4)
    assert gray.shape == (1, 4)
    assert gray[0].tolist() == [16, 0, 0, 0]
    normalized = histogramEx(photo, bins=8, normalize=True, channels=[2, 0])
    assert normalized.shape == (2, 8)
    assert normalized.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert np.array_equal(normalized[0] * photo.shape[0] * photo.shape[1], histogramEx(photo, bins=8)[2])
    mask = np.zeros(photo.shape[:2], np.uint8)
    mask[:10, :10] = 255
    assert histogramEx(photo, mask=mask).sum() == 300
    assert histogramEx(np.full((2, 2), 1.0, np.float32), bins=2)[0].tolist() == [0, 4]
    assert histogramEx(np.full((2, 2), 65535, np.uint16), bins=2)[0].tolist() == [0, 4]


def test_histogramEx_validation(photo):
    with pytest.raises(ValueError, match="bins"):
        histogramEx(photo, bins=0)
    with pytest.raises(ValueError, match="channels"):
        histogramEx(photo, channels=[3])
    with pytest.raises(ValueError, match="mask"):
        histogramEx(photo, mask=np.zeros((3, 3), np.uint8))
    with pytest.raises(ValueError, match="empty"):
        histogramEx(np.zeros((0, 4), np.uint8))
    with pytest.raises(ValueError, match="2-D or 3-D"):
        histogramEx(np.zeros((2, 2, 2, 2), np.uint8))


def test_histogramCompare(photo):
    assert histogramCompare(photo, photo) == pytest.approx(1.0)
    assert histogramCompare(photo, photo, colorSpace="bgr") == pytest.approx(1.0)
    assert histogramCompare(photo, photo, colorSpace="gray") == pytest.approx(1.0)
    assert histogramCompare(photo, photo, method=cv2.HISTCMP_BHATTACHARYYA) == pytest.approx(0.0, abs=1e-6)
    # different sizes and channel counts are fine
    small = cv2.resize(photo, (32, 24))
    assert histogramCompare(photo, small) > 0.8
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    assert histogramCompare(photo, gray, colorSpace="gray") == pytest.approx(1.0)
    assert histogramCompare(photo, cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)) == pytest.approx(1.0)
    # a different color distribution scores lower
    other = photo[..., ::-1].copy()
    assert histogramCompare(photo, other) < 0.5
    assert histogramCompare(photo.astype(np.float32) / 255, photo) == pytest.approx(1.0)


def test_histogramCompare_validation(photo):
    with pytest.raises(ValueError, match="bins"):
        histogramCompare(photo, photo, bins=0)
    with pytest.raises(ValueError, match="colorSpace"):
        histogramCompare(photo, photo, colorSpace="lab")


# --- dominantHue / meanColor ----------------------------------------------------------------


@pytest.mark.parametrize(("color", "degrees"), [((0, 0, 255), 0), ((0, 255, 0), 120), ((255, 0, 0), 240)])
def test_dominantHue(color, degrees):
    img = np.full((8, 8, 3), color, np.uint8)
    assert dominantHue(img) == pytest.approx(degrees, abs=2)


def test_dominantHue_ignores_gray_and_mixed():
    img = np.full((10, 10, 3), 128, np.uint8)  # gray majority
    img[:3] = (0, 255, 255)  # yellow minority but saturated
    assert dominantHue(img) == pytest.approx(60, abs=2)
    assert dominantHue(np.full((4, 4), 9, np.uint8)) == 0.0
    assert dominantHue(np.full((4, 4, 3), 200, np.uint8)) == 0.0
    red = np.full((8, 8, 3), (0, 0, 1.0), np.float32)
    assert dominantHue(red) == pytest.approx(0, abs=2)
    with pytest.raises(ValueError, match="bins"):
        dominantHue(img, bins=0)


def test_meanColor():
    img = np.zeros((4, 4, 3), np.uint8)
    img[:, :2] = (10, 20, 30)
    assert meanColor(img) == (5.0, 10.0, 15.0)
    mask = np.zeros((4, 4), np.uint8)
    mask[:, :2] = 1
    assert meanColor(img, mask) == (10.0, 20.0, 30.0)
    assert meanColor(img, np.zeros((4, 4), np.uint8)) == (0.0, 0.0, 0.0)
    assert meanColor(np.full((3, 3), 7, np.uint8)) == (7.0,)
    assert len(meanColor(np.zeros((2, 2, 4), np.uint8))) == 4
    with pytest.raises(ValueError, match="mask"):
        meanColor(img, np.zeros((2, 2), np.uint8))


# --- imageStats -----------------------------------------------------------------------------


def test_imageStats(photo, checkerboard):
    s = imageStats(photo)
    assert isinstance(s, ImageStats)
    assert (s.width, s.height, s.channels, s.dtype) == (128, 96, 3, "uint8")
    assert len(s.mean) == 3 and len(s.std) == 3
    assert s.mean == pytest.approx(meanColor(photo))
    assert 0 <= s.minVal <= s.maxVal <= 255
    assert s.brightness == pytest.approx(brightnessScore(photo))
    assert s.contrast == pytest.approx(contrastScore(photo))
    assert s.sharpness > 0 and s.entropy > 0
    assert not s.isGray

    g = imageStats(checkerboard)
    assert (g.channels, g.minVal, g.maxVal, g.isGray) == (1, 0.0, 255.0, True)
    assert g.mean == (pytest.approx(127.5),)
    assert g.entropy == pytest.approx(1.0)


@pytest.mark.parametrize(
    "fn",
    [
        imageStats,
        isGrayscale,
        isBlank,
        isBlurry,
        brightnessScore,
        contrastScore,
        colorfulness,
        exposureScore,
        entropy,
        estimateNoise,
        dominantHue,
        meanColor,
    ],
)
def test_bad_images(fn):
    with pytest.raises(ValueError):
        fn(np.zeros((0, 4), np.uint8))
    with pytest.raises(ValueError):
        fn(np.zeros((4, 4, 2), np.uint8))
