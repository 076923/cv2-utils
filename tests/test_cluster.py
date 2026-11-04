import sys

import numpy as np
import pytest

import cv2u as cv

RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)


@pytest.fixture
def three_colors():
    """60% red, 30% green, 10% blue, with a little noise."""
    img = np.zeros((20, 50, 3), np.uint8)
    img[:, :30] = RED
    img[:, 30:45] = GREEN
    img[:, 45:] = BLUE
    noise = np.random.default_rng(0).integers(-6, 7, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def _close(a, b, tol=20):
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b, strict=True))


# --- kmeansColors / dominantColors / colorQuantize (OpenCV only) ---------------------------


def test_kmeansColors_finds_the_colors(three_colors):
    maps, centers = cv.kmeansColors(three_colors, 3)
    assert maps.shape == three_colors.shape[:2]
    assert centers.shape == (3, 3)
    assert centers.dtype == np.uint8
    assert len(np.unique(maps)) == 3
    for color in (RED, GREEN, BLUE):
        assert any(_close(c, color) for c in centers), color
    # Every column of the same color gets the same label.
    assert len(np.unique(maps[:, :30])) == 1


def test_kmeansColors_is_deterministic_and_leaves_opencv_rng_alone(three_colors):
    cv.setRNGSeed(123)
    expected = cv.randu(np.zeros(3, np.float32), 0, 1).copy()
    cv.setRNGSeed(123)
    first = cv.kmeansColors(three_colors, 3, seed=7)
    after = cv.randu(np.zeros(3, np.float32), 0, 1)
    np.testing.assert_array_equal(after, expected)  # OpenCV's global RNG was not consumed
    second = cv.kmeansColors(three_colors, 3, seed=7)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_kmeansColors_large_image_is_sampled():
    img = np.zeros((600, 800, 3), np.uint8)
    img[:, 400:] = (200, 100, 50)
    maps, _ = cv.kmeansColors(img, 2, sampleSize=1000)
    assert maps.shape == (600, 800)
    assert len(np.unique(maps)) == 2
    assert maps[0, 0] != maps[0, 799]


def test_kmeansColors_gray():
    gray = np.zeros((10, 10), np.uint8)
    gray[:, 5:] = 200
    _, centers = cv.kmeansColors(gray, 2)
    assert centers.shape == (2, 1)
    assert sorted(int(c[0]) for c in centers) == [0, 200]


@pytest.mark.parametrize(
    ("src", "kwargs"),
    [
        (np.zeros((4, 4, 3), np.float32), {"k": 2}),
        (np.zeros((4, 4, 4), np.uint8), {"k": 2}),
        (np.zeros((4, 4, 3), np.uint8), {"k": 0}),
        (np.zeros((4, 4, 3), np.uint8), {"k": 17}),
        (np.zeros((4, 4, 3), np.uint8), {"k": 3, "sampleSize": 2}),
    ],
)
def test_kmeansColors_validation(src, kwargs):
    with pytest.raises(ValueError):
        cv.kmeansColors(src, **kwargs)


def test_dominantColors_sorted_by_share(three_colors):
    palette = cv.dominantColors(three_colors, k=3)
    colors = [c for c, _ in palette]
    ratios = [r for _, r in palette]
    assert ratios == sorted(ratios, reverse=True)
    assert sum(ratios) == pytest.approx(1.0)
    assert ratios == pytest.approx([0.6, 0.3, 0.1], abs=0.02)
    assert _close(colors[0], RED)
    assert _close(colors[1], GREEN)
    assert _close(colors[2], BLUE)
    assert all(isinstance(v, int) for v in colors[0])


def test_dominantColors_omits_unused_colors():
    flat = np.full((5, 5, 3), 77, np.uint8)
    palette = cv.dominantColors(flat, k=3)
    assert palette[0][1] == pytest.approx(1.0)
    assert all(ratio > 0 for _, ratio in palette)


def test_colorQuantize(three_colors):
    out = cv.colorQuantize(three_colors, 3)
    assert out.shape == three_colors.shape
    assert out.dtype == np.uint8
    assert len(np.unique(out.reshape(-1, 3), axis=0)) <= 3
    assert len(np.unique(three_colors.reshape(-1, 3), axis=0)) > 3


def test_kmeans_maps_work_with_drawDBSCAN(three_colors):
    maps, _ = cv.kmeansColors(three_colors, 3)
    painted = cv.drawDBSCAN(three_colors, maps, maps.ravel())
    assert len(np.unique(painted.reshape(-1, 3), axis=0)) == 3


def test_kmeans_functions_need_no_extra(monkeypatch, three_colors):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    assert len(cv.dominantColors(three_colors, 3)) == 3


# --- meanShiftColors / clusterPoints (cluster extra) -----------------------------------------


@pytest.mark.parametrize("noise", ["uniform", "gaussian"])
def test_meanShiftColors_finds_the_number_of_colors(noise):
    pytest.importorskip("sklearn")
    img = np.zeros((20, 50, 3), np.uint8)
    img[:, :30], img[:, 30:45], img[:, 45:] = RED, GREEN, BLUE
    rng = np.random.default_rng(1)
    jitter = rng.integers(-6, 7, img.shape) if noise == "uniform" else rng.normal(0, 4, img.shape)
    img = np.clip(img.astype(float) + jitter, 0, 255).astype(np.uint8)
    maps, centers = cv.meanShiftColors(img)
    assert len(centers) == 3  # k was not given
    assert maps.shape == img.shape[:2]
    for color in (RED, GREEN, BLUE):
        assert any(_close(c, color) for c in centers), color
    assert cv.drawDBSCAN(img, maps, maps.ravel()).shape == img.shape


def test_meanShiftColors_bandwidth_controls_merging(three_colors):
    pytest.importorskip("sklearn")
    _, few = cv.meanShiftColors(three_colors, bandwidth=500)
    assert len(few) == 1  # a huge kernel merges everything


def test_meanShiftColors_flat_image_and_determinism(three_colors):
    pytest.importorskip("sklearn")
    _, centers = cv.meanShiftColors(np.full((10, 10, 3), 40, np.uint8))
    assert len(centers) == 1
    a = cv.meanShiftColors(three_colors, seed=3)
    b = cv.meanShiftColors(three_colors, seed=3)
    np.testing.assert_array_equal(a[1], b[1])


def test_meanShiftColors_validation(three_colors):
    with pytest.raises(ValueError):
        cv.meanShiftColors(three_colors[..., 0])
    with pytest.raises(ValueError):
        cv.meanShiftColors(three_colors.astype(np.float32))
    with pytest.raises(ValueError):
        cv.meanShiftColors(three_colors, quantile=0)


def test_clusterPoints_groups_and_noise():
    pytest.importorskip("sklearn")
    points = [[0, 0], [1, 0], [0, 1], [50, 50], [51, 50], [200, 0]]
    labels = cv.clusterPoints(points, eps=3)
    assert labels.shape == (6,)
    assert labels[0] == labels[1] == labels[2] != 0
    assert labels[3] == labels[4] != 0
    assert labels[3] != labels[0]
    assert labels[5] == 0  # isolated point is noise


def test_clusterPoints_contour_input_and_minSamples():
    pytest.importorskip("sklearn")
    contour = np.array([[[0, 0]], [[1, 1]], [[100, 100]]], np.int32)
    assert cv.clusterPoints(contour, eps=2, minSamples=1).tolist() == [1, 1, 2]


def test_clusterPoints_validation():
    with pytest.raises(ValueError):
        cv.clusterPoints(np.zeros((0, 2)), eps=1)


@pytest.mark.parametrize(
    ("func", "args"),
    [
        ("DBSCAN", (np.zeros((2, 2, 3), np.uint8), 1.0, 2)),
        ("meanShiftColors", (np.zeros((2, 2, 3), np.uint8),)),
        ("clusterPoints", ([[0, 0]], 1.0)),
    ],
)
def test_sklearn_functions_explain_the_extra(monkeypatch, func, args):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    with pytest.raises(ImportError, match=rf"{func} requires scikit-learn.*\[cluster\]"):
        getattr(cv, func)(*args)
