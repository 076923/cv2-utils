import sys
import time

import numpy as np
import pytest

import cv2u as cv

RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)


def _three_colors(seed=0, noise="gaussian"):
    img = np.zeros((20, 50, 3), np.uint8)
    img[:, :30], img[:, 30:45], img[:, 45:] = RED, GREEN, BLUE
    rng = np.random.default_rng(seed)
    jitter = rng.integers(-6, 7, img.shape) if noise == "uniform" else rng.normal(0, 4, img.shape)
    return np.clip(img.astype(float) + jitter, 0, 255).astype(np.uint8)


def _close(a, b, tol=20):
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("noise", ["uniform", "gaussian"])
def test_peakColors_finds_the_number_of_colors(seed, noise):
    img = _three_colors(seed, noise)
    maps, centers = cv.peakColors(img)
    assert len(centers) == 3
    assert maps.shape == img.shape[:2]
    for color in (RED, GREEN, BLUE):
        assert any(_close(c, color) for c in centers), color


def test_peakColors_labels_are_sorted_by_size():
    maps, centers = cv.peakColors(_three_colors())
    sizes = np.bincount(maps.ravel())
    assert list(sizes) == sorted(sizes, reverse=True)
    assert _close(centers[0], RED)  # 60% of the pixels
    assert _close(centers[2], BLUE)  # 10%


def test_peakColors_is_deterministic():
    img = _three_colors()
    a, b = cv.peakColors(img), cv.peakColors(img)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_peakColors_flat_image():
    maps, centers = cv.peakColors(np.full((30, 30, 3), 77, np.uint8))
    assert len(centers) == 1
    assert (maps == 0).all()
    assert tuple(centers[0]) == (77, 77, 77)


def test_peakColors_minShare_merges_small_colors():
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :50] = RED
    img[:, 50:] = GREEN
    img[0, 0] = BLUE  # a single blue pixel: 0.01% of the image
    assert len(cv.peakColors(img, minShare=0.0)[1]) == 3
    assert len(cv.peakColors(img)[1]) == 2


def test_peakColors_minRegion_removes_speckles():
    img = np.zeros((60, 60, 3), np.uint8)
    img[:, :30] = RED
    img[:, 30:] = GREEN
    img[10:12, 10:12] = GREEN  # a 4-pixel green speck inside the red half
    maps, _ = cv.peakColors(img, minShare=0.0)
    assert maps[10, 10] == maps[0, 59]  # without cleanup the speck is green
    cleaned, centers = cv.peakColors(img, minShare=0.0, minRegion=10)
    assert cleaned[10, 10] == cleaned[0, 0]  # absorbed by the surrounding red region
    assert len(centers) == 2


def test_peakColors_works_with_drawDBSCAN():
    img = _three_colors()
    maps, _ = cv.peakColors(img)
    assert len(np.unique(cv.drawDBSCAN(img, maps, maps.ravel()).reshape(-1, 3), axis=0)) == 3


def test_peakColors_needs_no_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    assert len(cv.peakColors(_three_colors())[1]) == 3


def test_peakColors_uses_every_pixel_quickly():
    img = np.zeros((1500, 2000, 3), np.uint8)  # 3 megapixels, no sampling
    img[:, 1000:] = (30, 160, 220)
    start = time.perf_counter()
    maps, centers = cv.peakColors(img)
    assert time.perf_counter() - start < 10
    assert len(centers) == 2
    assert maps[0, 0] != maps[0, 1999]


@pytest.mark.parametrize(
    ("src", "kwargs"),
    [
        (np.zeros((4, 4), np.uint8), {}),
        (np.zeros((4, 4, 3), np.float32), {}),
        (np.zeros((4, 4, 3), np.uint8), {"bins": 1}),
        (np.zeros((4, 4, 3), np.uint8), {"bins": 65}),
        (np.zeros((4, 4, 3), np.uint8), {"sigma": -1}),
        (np.zeros((4, 4, 3), np.uint8), {"minShare": 1.0}),
        (np.zeros((4, 4, 3), np.uint8), {"minRegion": -1}),
    ],
)
def test_peakColors_validation(src, kwargs):
    with pytest.raises(ValueError):
        cv.peakColors(src, **kwargs)
