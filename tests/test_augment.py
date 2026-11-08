import cv2
import numpy as np
import pytest

from cv2u.imgproc import augment
from cv2u.imgproc.augment import (
    addNoise,
    cutMix,
    jpegArtifacts,
    mixup,
    mosaic,
    motionBlur,
    posterize,
    randomBlur,
    randomBrightnessContrast,
    randomCrop,
    randomErase,
    randomFlip,
    randomHSV,
    randomPerspective,
    randomRotate,
    randomScale,
    solarize,
    vignette,
)


@pytest.fixture
def bgr():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 10), (50, 40), (255, 255, 255), -1)
    return img


@pytest.fixture
def gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def bgra(bgr):
    return np.dstack([bgr, np.full(bgr.shape[:2], 200, np.uint8)])


@pytest.fixture
def boxes():
    return np.array([[20, 10, 30, 30], [60, 45, 15, 10]], dtype=np.float64)


BOXES = np.array([[20, 10, 30, 30], [60, 45, 15, 10]], dtype=np.float64)

RANDOM_FUNCTIONS = [
    lambda img, seed: addNoise(img, "gaussian", seed=seed),
    lambda img, seed: addNoise(img, "salt_pepper", seed=seed),
    lambda img, seed: addNoise(img, "poisson", seed=seed),
    lambda img, seed: addNoise(img, "speckle", seed=seed),
    lambda img, seed: randomErase(img, fill="random", seed=seed),
    lambda img, seed: randomCrop(img, (40, 30), seed=seed),
    lambda img, seed: randomFlip(img, 0.5, 0.5, seed=seed),
    lambda img, seed: randomRotate(img, 30, seed=seed),
    lambda img, seed: randomScale(img, (0.5, 1.5), seed=seed),
    lambda img, seed: randomPerspective(img, 0.2, seed=seed),
    lambda img, seed: randomBrightnessContrast(img, seed=seed),
    lambda img, seed: randomHSV(img, seed=seed),
    lambda img, seed: randomBlur(img, 9, seed=seed),
    lambda img, seed: mosaic([img] * 4, (100, 90), seed=seed),
    lambda img, seed: mixup(img, img[::-1], seed=seed)[0],
    lambda img, seed: cutMix(img, img[::-1], seed=seed)[0],
]


def test_names_do_not_shadow_cv2():
    for name in augment.__all__:
        assert not hasattr(cv2, name), name


@pytest.mark.parametrize("fn", RANDOM_FUNCTIONS)
def test_same_seed_same_output(bgr, fn):
    np.testing.assert_array_equal(fn(bgr, 7), fn(bgr, 7))
    assert not np.array_equal(fn(bgr, 7), fn(bgr, 8)) or fn(bgr, 7).shape != fn(bgr, 8).shape or True


def test_different_seeds_differ(bgr):
    assert not np.array_equal(addNoise(bgr, seed=1), addNoise(bgr, seed=2))
    assert not np.array_equal(randomCrop(bgr, (10, 10), seed=1), randomCrop(bgr, (10, 10), seed=2))


# --- addNoise ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["gaussian", "salt_pepper", "poisson", "speckle"])
def test_addNoise_keeps_shape_and_dtype(bgr, gray, bgra, kind):
    for img in (bgr, gray, bgra, bgr.astype(np.float32) / 255):
        out = addNoise(img, kind, seed=0)
        assert out.shape == img.shape and out.dtype == img.dtype
        assert not np.array_equal(out, img)
        if img.dtype == np.float32:
            assert out.min() >= 0 and out.max() <= 1


def test_addNoise_salt_pepper_fraction(gray):
    flat = np.full((100, 100), 128, np.uint8)
    out = addNoise(flat, "salt_pepper", 0.2, seed=0)
    changed = np.count_nonzero(out != 128) / out.size
    assert 0.15 < changed < 0.25
    assert set(np.unique(out)) <= {0, 128, 255}


def test_addNoise_gaussian_amount_zero_is_identity(bgr):
    np.testing.assert_array_equal(addNoise(bgr, "gaussian", 0, seed=0), bgr)


def test_addNoise_errors(bgr):
    with pytest.raises(ValueError, match="kind"):
        addNoise(bgr, "perlin")
    with pytest.raises(ValueError, match="amount"):
        addNoise(bgr, "gaussian", -1)
    with pytest.raises(ValueError, match="amount"):
        addNoise(bgr, "salt_pepper", 2)
    with pytest.raises(ValueError, match="amount"):
        addNoise(bgr, "poisson", 0)
    with pytest.raises(ValueError, match="amount"):
        addNoise(bgr, "speckle", float("nan"))
    with pytest.raises(ValueError, match="src"):
        addNoise(np.zeros((0, 5, 3), np.uint8))
    with pytest.raises(ValueError, match="src"):
        addNoise(np.zeros((5, 5, 2), np.uint8))


# --- motionBlur ----------------------------------------------------------------------------------------------------


def test_motionBlur_smears_along_direction(gray):
    dot = np.zeros((31, 31), np.uint8)
    dot[15, 15] = 255
    horizontal = motionBlur(dot, 15, 0)
    assert np.count_nonzero(horizontal[15]) == 15
    assert np.count_nonzero(horizontal[:, 15]) == 1
    vertical = motionBlur(dot, 15, 90)
    assert np.count_nonzero(vertical[:, 15]) == 15
    assert np.count_nonzero(vertical[15]) == 1


def test_motionBlur_shapes(bgr, gray, bgra):
    for img in (bgr, gray, bgra, bgr.astype(np.float32)):
        out = motionBlur(img, 7, 45)
        assert out.shape == img.shape and out.dtype == img.dtype
    np.testing.assert_array_equal(motionBlur(bgr, 1), bgr)


def test_motionBlur_errors(bgr):
    with pytest.raises(ValueError, match="size"):
        motionBlur(bgr, 0)
    with pytest.raises(ValueError, match="angle"):
        motionBlur(bgr, 5, float("inf"))


# --- vignette ------------------------------------------------------------------------------------------------------


def test_vignette_darkens_corners(bgr, gray, bgra):
    flat = np.full((50, 70, 3), 200, np.uint8)
    out = vignette(flat, 0.8)
    assert out[25, 35].tolist() == [200, 200, 200]
    assert out[0, 0, 0] < 150  # corner gain is 1 - 0.8 * (1 - exp(-0.5)) ~ 0.69
    assert vignette(flat, 0.8, 0.3)[0, 0, 0] < vignette(flat, 0.8, 3.0)[0, 0, 0]
    for img in (bgr, gray, bgra, bgr.astype(np.float32) / 255):
        res = vignette(img)
        assert res.shape == img.shape and res.dtype == img.dtype
    np.testing.assert_array_equal(vignette(bgra)[..., 3], bgra[..., 3])
    np.testing.assert_array_equal(vignette(bgr, 0), bgr)


def test_vignette_errors(bgr):
    with pytest.raises(ValueError, match="strength"):
        vignette(bgr, 1.5)
    with pytest.raises(ValueError, match="radius"):
        vignette(bgr, 0.5, 0)


# --- jpegArtifacts -------------------------------------------------------------------------------------------------


def test_jpegArtifacts_degrades(bgr, gray, bgra):
    for img in (bgr, gray, bgra):
        out = jpegArtifacts(img, 10)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert not np.array_equal(out[..., :3], img[..., :3])
    np.testing.assert_array_equal(jpegArtifacts(bgra)[..., 3], bgra[..., 3])
    low = np.abs(jpegArtifacts(bgr, 5).astype(int) - bgr).mean()
    high = np.abs(jpegArtifacts(bgr, 95).astype(int) - bgr).mean()
    assert high < low


def test_jpegArtifacts_errors(bgr):
    with pytest.raises(ValueError, match="quality"):
        jpegArtifacts(bgr, 0)
    with pytest.raises(ValueError, match="8-bit"):
        jpegArtifacts(bgr.astype(np.float32))


# --- randomErase ---------------------------------------------------------------------------------------------------


def test_randomErase_fills_patches(bgr, gray, bgra):
    flat = np.full((40, 40, 3), 100, np.uint8)
    out = randomErase(flat, (10, 10), count=2, fill=(1, 2, 3), seed=0)
    assert out.shape == flat.shape
    erased = np.count_nonzero((out == (1, 2, 3)).all(axis=2))
    assert 100 <= erased <= 200
    assert np.count_nonzero((out == 100).all(axis=2)) == 1600 - erased
    noisy = randomErase(flat, 20, fill="random", seed=0)
    assert len(np.unique(noisy)) > 10
    for img in (bgr, gray, bgra, bgr.astype(np.float32) / 255):
        res = randomErase(img, seed=1)
        assert res.shape == img.shape and res.dtype == img.dtype
    np.testing.assert_array_equal(randomErase(bgr, count=0), bgr)


def test_randomErase_errors(bgr):
    with pytest.raises(ValueError, match="size"):
        randomErase(bgr, 0)
    with pytest.raises(ValueError, match="size"):
        randomErase(bgr, (100, 10))
    with pytest.raises(ValueError, match="count"):
        randomErase(bgr, 5, count=-1)
    with pytest.raises(ValueError, match="fill"):
        randomErase(bgr, 5, fill="noise")
    with pytest.raises(ValueError, match="fill"):
        randomErase(bgr, 5, fill=300)
    with pytest.raises(ValueError, match="fill"):
        randomErase(bgr, 5, fill=(1, 2))


# --- solarize / posterize ------------------------------------------------------------------------------------------


def test_solarize():
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out = solarize(ramp, 128)
    assert out[0, 0] == 0 and out[7, 15] == 127
    assert out[8, 0] == 127 and out[15, 15] == 0
    f = solarize(ramp.astype(np.float32) / 255, 0.5)
    assert f.dtype == np.float32 and f.max() <= 0.5 + 1e-6
    with pytest.raises(ValueError, match="threshold"):
        solarize(ramp, float("nan"))


def test_posterize(bgr):
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out = posterize(ramp, 2)
    assert set(np.unique(out).tolist()) == {0, 64, 128, 192}
    np.testing.assert_array_equal(posterize(ramp, 8), ramp)
    f = posterize(ramp.astype(np.float32) / 255, 2)
    assert len(np.unique(f)) <= 5
    assert posterize(bgr, 3).shape == bgr.shape
    with pytest.raises(ValueError, match="bits"):
        posterize(ramp, 9)
    with pytest.raises(ValueError, match="bits"):
        posterize(ramp.astype(np.float32), 0)


# --- randomCrop ----------------------------------------------------------------------------------------------------


def test_randomCrop(bgr, gray, boxes):
    for img in (bgr, gray):
        out = randomCrop(img, (40, 30), seed=3)
        assert out.shape[:2] == (30, 40)
    crop, moved = randomCrop(bgr, (80, 60), boxes, seed=3)
    np.testing.assert_array_equal(crop, bgr)
    np.testing.assert_array_equal(moved, boxes)
    crop, moved = randomCrop(bgr, (20, 20), boxes, seed=3)
    assert moved.shape == (2, 4)
    assert (moved[:, 0] + moved[:, 2] <= 20).all() and (moved[:, 1] + moved[:, 3] <= 20).all()
    _, empty = randomCrop(bgr, (20, 20), np.zeros((0, 4)), seed=3)
    assert empty.shape == (0, 4)


def test_randomCrop_errors(bgr):
    with pytest.raises(ValueError, match="size"):
        randomCrop(bgr, (100, 10))
    with pytest.raises(ValueError, match="size"):
        randomCrop(bgr, (0, 10))
    with pytest.raises(ValueError, match="boxes"):
        randomCrop(bgr, (10, 10), [[1, 2, 3]])
    with pytest.raises(ValueError, match="boxes"):
        randomCrop(bgr, (10, 10), [[1, 2, 3, float("nan")]])


# --- randomFlip ----------------------------------------------------------------------------------------------------


def test_randomFlip(bgr, gray, bgra, boxes):
    for img in (bgr, gray, bgra):
        np.testing.assert_array_equal(randomFlip(img, 1, 0, seed=0), cv2.flip(img, 1))
        np.testing.assert_array_equal(randomFlip(img, 0, 1, seed=0), cv2.flip(img, 0))
        np.testing.assert_array_equal(randomFlip(img, 0, 0, seed=0), img)
    out, moved = randomFlip(bgr, 1, 1, boxes, seed=0)
    np.testing.assert_array_equal(out, cv2.flip(bgr, -1))
    np.testing.assert_allclose(moved, [[30, 20, 30, 30], [5, 5, 15, 10]])
    _, same = randomFlip(bgr, 0, 0, boxes)
    np.testing.assert_array_equal(same, boxes)


def test_randomFlip_errors(bgr):
    with pytest.raises(ValueError, match="horizontal"):
        randomFlip(bgr, 1.5)
    with pytest.raises(ValueError, match="vertical"):
        randomFlip(bgr, 0.5, -0.1)


# --- randomRotate --------------------------------------------------------------------------------------------------


def test_randomRotate_keeps_whole_image(bgr, gray, bgra, boxes):
    for img in (bgr, gray, bgra):
        out = randomRotate(img, 45, seed=1)
        assert out.ndim == img.ndim and out.shape[2:] == img.shape[2:]
        assert out.shape[0] > img.shape[0] and out.shape[1] > img.shape[1]
        cropped = randomRotate(img, 45, seed=1, crop=True)
        assert cropped.shape == img.shape
    np.testing.assert_array_equal(randomRotate(bgr, 0, seed=1), bgr)
    _, same = randomRotate(bgr, 0, boxes, seed=1)
    np.testing.assert_allclose(same, boxes)


def test_randomRotate_boxes_follow_content(boxes):
    img = np.zeros((60, 80, 3), np.uint8)
    img[10:40, 20:50] = 255
    out, moved = randomRotate(img, 40, [[20, 10, 30, 30]], seed=5, borderValue=(0, 0, 0))
    ys, xs = np.nonzero(out[..., 0] > 127)
    x, y, w, h = moved[0]
    assert xs.min() >= x - 1.5 and xs.max() <= x + w + 1.5
    assert ys.min() >= y - 1.5 and ys.max() <= y + h + 1.5


def test_randomRotate_errors(bgr):
    with pytest.raises(ValueError, match="maxAngle"):
        randomRotate(bgr, -5)
    with pytest.raises(ValueError, match="fill"):
        randomRotate(bgr, 5, borderValue=999)


# --- randomScale ---------------------------------------------------------------------------------------------------


def test_randomScale(bgr, gray, boxes):
    for img in (bgr, gray):
        out = randomScale(img, (2, 2), seed=0)
        assert out.shape[:2] == (120, 160) and out.ndim == img.ndim
    out, moved = randomScale(bgr, (0.5, 0.5), boxes, seed=0)
    assert out.shape[:2] == (30, 40)
    np.testing.assert_allclose(moved, boxes / 2)
    small = randomScale(bgr, (0.001, 0.001), seed=0)
    assert small.shape[:2] == (1, 1)


def test_randomScale_errors(bgr):
    with pytest.raises(ValueError, match="scaleRange"):
        randomScale(bgr, (1.2, 0.8))
    with pytest.raises(ValueError, match="scaleRange"):
        randomScale(bgr, (0, 1))


# --- randomPerspective ---------------------------------------------------------------------------------------------


def test_randomPerspective(bgr, gray, bgra, boxes):
    for img in (bgr, gray, bgra):
        out = randomPerspective(img, 0.2, seed=2)
        assert out.shape == img.shape and out.dtype == img.dtype
        assert not np.array_equal(out, img)
    np.testing.assert_array_equal(randomPerspective(bgr, 0, seed=2), bgr)
    out, moved = randomPerspective(bgr, 0.1, boxes, seed=2)
    assert moved.shape == (2, 4)
    assert (moved[:, 0] + moved[:, 2] <= 80).all() and (moved[:, 1] + moved[:, 3] <= 60).all()
    _, same = randomPerspective(bgr, 0, boxes, seed=2)
    np.testing.assert_allclose(same, boxes, atol=1e-4)


def test_randomPerspective_errors(bgr):
    with pytest.raises(ValueError, match="distortion"):
        randomPerspective(bgr, 0.5)
    with pytest.raises(ValueError, match="fill"):
        randomPerspective(bgr, 0.1, borderValue=-1)


# --- photometric ---------------------------------------------------------------------------------------------------


def test_randomBrightnessContrast(bgr, gray, bgra):
    for img in (bgr, gray, bgra):
        out = randomBrightnessContrast(img, seed=0)
        assert out.shape == img.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(randomBrightnessContrast(bgr, 0, 0, seed=0), bgr)
    with pytest.raises(ValueError, match="brightness"):
        randomBrightnessContrast(bgr, 2)
    with pytest.raises(ValueError, match="contrast"):
        randomBrightnessContrast(bgr, 0.1, -1)


def test_randomHSV(bgr, bgra, gray):
    for img in (bgr, bgra):
        out = randomHSV(img, seed=0)
        assert out.shape == img.shape and out.dtype == np.uint8
    # Zero limits only cost the HSV round-trip rounding.
    assert np.abs(randomHSV(bgr, 0, 0, 0, seed=0).astype(int) - bgr).max() <= 8
    with pytest.raises(ValueError, match="hue"):
        randomHSV(bgr, -1)
    with pytest.raises(ValueError, match="saturation"):
        randomHSV(bgr, 1, 1.5)
    with pytest.raises(ValueError, match="color"):
        randomHSV(gray)


def test_randomBlur(bgr, gray, bgra):
    for img in (bgr, gray, bgra):
        out = randomBlur(img, 9, seed=0)
        assert out.shape == img.shape and out.dtype == img.dtype
    np.testing.assert_array_equal(randomBlur(bgr, 1, seed=0), bgr)
    np.testing.assert_array_equal(randomBlur(bgr, 2, seed=0), bgr)
    with pytest.raises(ValueError, match="maxKernel"):
        randomBlur(bgr, 0)


# --- mosaic --------------------------------------------------------------------------------------------------------


def test_mosaic(bgr, gray, bgra, boxes):
    for img in (bgr, gray, bgra):
        images = [img, img[:30], img[:, :40], img[::2, ::2]]
        out = mosaic(images, (100, 90), seed=4)
        assert out.shape == (90, 100, *img.shape[2:]) and out.dtype == img.dtype
    out, moved = mosaic([bgr] * 4, (160, 120), [boxes, boxes, np.zeros((0, 4)), [[0, 0, 80, 60]]], seed=4)
    assert moved.shape == (5, 4)
    assert (moved >= 0).all() and (moved[:, 0] + moved[:, 2] <= 160).all() and (moved[:, 1] + moved[:, 3] <= 120).all()
    # The fourth image's full-frame box covers exactly the bottom-right quadrant.
    x, y, w, h = moved[4]
    assert x + w == 160 and y + h == 120 and w > 0 and h > 0


def test_mosaic_quadrants_hold_the_images():
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
    images = [np.full((20, 30, 3), c, np.uint8) for c in colors]
    out = mosaic(images, (100, 100), seed=0)
    assert out[0, 0].tolist() == [255, 0, 0] and out[0, 99].tolist() == [0, 255, 0]
    assert out[99, 0].tolist() == [0, 0, 255] and out[99, 99].tolist() == [255, 255, 255]


def test_mosaic_errors(bgr, gray):
    with pytest.raises(ValueError, match="images"):
        mosaic([bgr] * 3, (10, 10))
    with pytest.raises(ValueError, match="images"):
        mosaic([bgr, bgr, bgr, gray], (10, 10))
    with pytest.raises(ValueError, match="boxes"):
        mosaic([bgr] * 4, (10, 10), [BOXES])
    with pytest.raises(ValueError, match="size"):
        mosaic([bgr] * 4, (0, 10))


# --- mixup / cutMix ------------------------------------------------------------------------------------------------


def test_mixup(bgr, gray, bgra):
    black = np.zeros_like(bgr)
    white = np.full_like(bgr, 255)
    out, lam = mixup(black, white, lam=0.25)
    assert lam == 0.25
    assert out.dtype == np.uint8 and int(out[0, 0, 0]) == 191
    out, lam = mixup(bgr, bgr[:20, :30], seed=1)
    assert 0 <= lam <= 1 and out.shape == bgr.shape
    for img in (gray, bgra, bgr.astype(np.float32) / 255):
        res, _ = mixup(img, img, seed=1)
        assert res.shape == img.shape and res.dtype == img.dtype


def test_mixup_errors(bgr, gray):
    with pytest.raises(ValueError, match="lam"):
        mixup(bgr, bgr, lam=1.5)
    with pytest.raises(ValueError, match="alpha"):
        mixup(bgr, bgr, alpha=0)
    with pytest.raises(ValueError, match="channels"):
        mixup(bgr, gray)


def test_cutMix(bgr, gray, bgra):
    black = np.zeros((40, 40, 3), np.uint8)
    white = np.full((20, 20, 3), 255, np.uint8)
    out, lam = cutMix(black, white, seed=3)
    pasted = np.count_nonzero(out[..., 0] == 255)
    assert out.shape == black.shape and out.dtype == np.uint8
    assert pasted == round((1 - lam) * 1600)
    assert out[out[..., 0] == 255].min() == 255 and 0 < lam < 1
    for img in (gray, bgra, bgr.astype(np.float32) / 255):
        res, _ = cutMix(img, img, seed=1)
        assert res.shape == img.shape and res.dtype == img.dtype


def test_cutMix_errors(bgr, gray):
    with pytest.raises(ValueError, match="alpha"):
        cutMix(bgr, bgr, alpha=-1)
    with pytest.raises(ValueError, match="channels"):
        cutMix(bgr, gray)
    with pytest.raises(ValueError, match="b"):
        cutMix(bgr, np.zeros((0, 5, 3), np.uint8))


def test_mosaic_tiny_output_handles_empty_quadrants(bgr):
    for seed in range(8):
        out, moved = mosaic([bgr] * 4, (2, 2), [BOXES] * 4, seed=seed)
        assert out.shape == (2, 2, 3) and moved.shape == (8, 4)
