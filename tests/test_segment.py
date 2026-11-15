import numpy as np
import pytest

import cv2u as cv
from cv2u.imgproc.segment import (
    Component,
    connectedComponentsEx,
    floodFillEx,
    grabCutEx,
    keepComponents,
    maskCoverage,
    maskHull,
    maskOutline,
    mergeMasks,
    smoothMask,
    splitTouching,
    trimap,
    watershedEx,
)


@pytest.fixture
def discs():
    # Two touching discs of the same radius (centers 40 px apart, radius 25).
    mask = np.zeros((100, 140), np.uint8)
    cv.circle(mask, (45, 50), 25, 255, cv.FILLED)
    cv.circle(mask, (85, 50), 25, 255, cv.FILLED)
    return mask


@pytest.fixture
def blobs():
    # Three separate blobs: a big square (2500), a small square (100) and a 3x3 speck (9).
    mask = np.zeros((100, 120), np.uint8)
    mask[10:60, 10:60] = 255
    mask[70:80, 90:100] = 255
    mask[5:8, 110:113] = 255
    return mask


@pytest.fixture
def scene():
    # A red square on a light gray background.
    img = np.full((80, 80, 3), (200, 200, 200), np.uint8)
    img[20:60, 20:60] = (0, 0, 255)
    return img


def _sizes(labels):
    return sorted(int((labels == i).sum()) for i in range(1, int(labels.max()) + 1))


# --- grabCutEx ---------------------------------------------------------------------


def test_grabCutEx_rect_finds_the_square(scene):
    out = grabCutEx(scene, rect=(10, 10, 60, 60))
    assert out.dtype == np.uint8 and out.shape == (80, 80)
    assert set(np.unique(out)) <= {0, 255}
    expected = np.zeros((80, 80), np.uint8)
    expected[20:60, 20:60] = 255
    np.testing.assert_array_equal(out, expected)


def test_grabCutEx_rough_mask_and_sure_values(scene):
    rough = np.zeros((80, 80), np.uint8)
    rough[15:65, 15:65] = 255
    out = grabCutEx(scene, mask=rough, iterations=3)
    assert (out[20:60, 20:60] == 255).all()
    assert out[:15].sum() == 0
    # Sure background (1) and sure foreground (3) survive the cut.
    rough[25, 25] = 3
    rough[2, 2] = 1
    out = grabCutEx(scene, mask=rough)
    assert out[25, 25] == 255 and out[2, 2] == 0
    # A bool mask works too.
    np.testing.assert_array_equal(grabCutEx(scene, mask=rough > 0), grabCutEx(scene, mask=(rough > 0) * 255))


def test_grabCutEx_gray_bgra_and_full_rect(scene):
    gray = cv.cvtColor(scene, cv.COLOR_BGR2GRAY)
    out = grabCutEx(gray, rect=(10, 10, 60, 60))
    assert (out[20:60, 20:60] == 255).all() and out[:15].sum() == 0
    bgra = cv.cvtColor(scene, cv.COLOR_BGR2BGRA)
    np.testing.assert_array_equal(grabCutEx(bgra, rect=(10, 10, 60, 60)), grabCutEx(scene, rect=(10, 10, 60, 60)))
    # The full-image rect is shrunk by a pixel so GrabCut has some background to learn from.
    assert (grabCutEx(scene, rect=(0, 0, 80, 80))[20:60, 20:60] == 255).all()


def test_grabCutEx_trivial_masks(scene):
    assert grabCutEx(scene, mask=np.zeros((80, 80), np.uint8)).sum() == 0
    assert (grabCutEx(scene, mask=np.full((80, 80), 255, np.uint8)) == 255).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"rect": (0, 0, 10, 10), "mask": np.zeros((80, 80), np.uint8)},
        {"rect": (0, 0, 10)},
        {"rect": (70, 70, 20, 20)},
        {"rect": (0, 0, 0, 10)},
        {"mask": np.zeros((40, 80), np.uint8)},
        {"rect": (0, 0, 10, 10), "iterations": 0},
    ],
)
def test_grabCutEx_validation(scene, kwargs):
    with pytest.raises(ValueError):
        grabCutEx(scene, **kwargs)


def test_grabCutEx_rejects_float_and_empty_images():
    with pytest.raises(ValueError, match="src"):
        grabCutEx(np.zeros((8, 8, 3), np.float32), rect=(0, 0, 4, 4))
    with pytest.raises(ValueError, match="src"):
        grabCutEx(np.zeros((0, 8, 3), np.uint8), rect=(0, 0, 4, 4))


# --- watershedEx / splitTouching ---------------------------------------------------


def test_watershedEx_separates_touching_discs(discs):
    img = np.full((100, 140, 3), 200, np.uint8)
    img[discs > 0] = (30, 60, 200)
    labels = watershedEx(img, mask=discs, minDistance=10)
    assert labels.dtype == np.int32 and labels.shape == (100, 140)
    assert labels.min() == 0 and labels.max() == 2
    sizes = _sizes(labels)
    assert abs(sizes[0] - sizes[1]) < 100  # same-size discs, roughly the same halves
    assert sum(sizes) > 0.98 * np.count_nonzero(discs)  # only the one-pixel ridge is lost
    assert labels[discs == 0].sum() == 0  # nothing labeled outside the mask
    assert labels[50, 45] != labels[50, 85]


def test_watershedEx_otsu_fallback_and_gray(discs):
    gray = np.where(discs > 0, 220, 30).astype(np.uint8)  # bright objects on a dark background
    labels = watershedEx(gray, minDistance=10)
    assert labels.max() == 2
    np.testing.assert_array_equal(watershedEx(cv.cvtColor(gray, cv.COLOR_GRAY2BGRA), minDistance=10), labels)


def test_watershedEx_user_markers(discs):
    img = cv.cvtColor(discs, cv.COLOR_GRAY2BGR)
    markers = np.zeros((100, 140), np.int32)
    markers[50, 40] = 7  # labels need not be consecutive
    markers[50, 90] = 3
    labels = watershedEx(img, markers, mask=discs)
    assert set(np.unique(labels).tolist()) == {0, 3, 7}  # the caller's ids are kept
    assert labels[50, 40] == 7 and labels[50, 90] == 3
    # Without a mask the whole image is foreground, so the background gets a label too.
    labels = watershedEx(img, markers)
    assert {3, 7} <= set(np.unique(labels).tolist()) and (labels > 0).mean() > 0.9  # 0 only on the ridge


def test_grabCutEx_binary_01_mask_is_foreground(scene):
    rough = np.zeros((80, 80), np.uint8)
    rough[15:65, 15:65] = 1  # 0/1, not 0/255: 1 is foreground here, not the GC_BGD scribble code
    out = grabCutEx(scene, mask=rough)
    assert (out[20:60, 20:60] == 255).all() and out[:15].sum() == 0
    np.testing.assert_array_equal(out, grabCutEx(scene, mask=rough * 255))


def test_watershedEx_empty_results():
    img = np.zeros((20, 20, 3), np.uint8)
    assert watershedEx(img, mask=np.zeros((20, 20), np.uint8)).max() == 0
    assert watershedEx(img, np.zeros((20, 20), np.int32)).max() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minDistance": -1},
        {"threshold": 0},
        {"threshold": 1.5},
        {"threshold": float("nan")},
        {"mask": np.zeros((5, 5), np.uint8)},
        {"markers": np.zeros((5, 5), np.int32)},
        {"markers": np.zeros((20, 20), np.float32)},
    ],
)
def test_watershedEx_validation(kwargs):
    with pytest.raises(ValueError):
        watershedEx(np.zeros((20, 20, 3), np.uint8), **kwargs)


def test_splitTouching_same_discs(discs):
    labels = splitTouching(discs, minDistance=10)
    assert labels.dtype == np.int32 and labels.max() == 2
    sizes = _sizes(labels)
    assert sizes[0] == sizes[1]  # symmetric cut through the neck
    assert sum(sizes) > 0.98 * np.count_nonzero(discs)
    assert labels[discs == 0].sum() == 0
    assert labels[50, 45] != labels[50, 85]
    assert labels[50, 65] == 0  # the ridge pixel on the neck


def test_splitTouching_different_sizes_cut_at_the_neck():
    mask = np.zeros((100, 140), np.uint8)
    cv.circle(mask, (40, 50), 30, 255, cv.FILLED)
    cv.circle(mask, (90, 50), 18, 255, cv.FILLED)
    labels = splitTouching(mask, minDistance=10)
    assert labels.max() == 2
    big, small = labels[50, 40], labels[50, 90]
    assert big != small
    # The cut is on the neck (x ~ 70), so the big disc keeps almost all of its pixels.
    assert (labels == big).sum() == pytest.approx(np.pi * 30**2, rel=0.03)
    assert (labels == small).sum() > 0.9 * (np.pi * 18**2 - 300)


def test_split_small_blob_without_seed_keeps_its_own_label(discs):
    # A 4x4 blob far from the discs has a distance peak below the seed threshold: it still gets a label.
    mask = discs.copy()
    mask[5:9, 120:124] = 255
    for labels in (
        splitTouching(mask, minDistance=10),
        watershedEx(cv.cvtColor(mask, cv.COLOR_GRAY2BGR), mask=mask, minDistance=10),
    ):
        assert labels.max() == 3
        assert (labels[5:9, 120:124] == 3).all()
        assert labels[50, 45] != labels[50, 85]


def test_splitTouching_without_minDistance_keeps_merged_seeds(discs):
    # With the default threshold the two seeds touch through the neck and form one label.
    assert splitTouching(discs).max() == 1
    assert splitTouching(discs, threshold=0.8).max() == 2
    assert splitTouching(np.zeros((10, 10), np.uint8)).max() == 0
    assert splitTouching(discs > 0, minDistance=10).max() == 2


@pytest.mark.parametrize("kwargs", [{"minDistance": -1}, {"threshold": 0}, {"threshold": 2}])
def test_splitTouching_validation(discs, kwargs):
    with pytest.raises(ValueError):
        splitTouching(discs, **kwargs)


# --- floodFillEx ---------------------------------------------------------------------


def test_floodFillEx_fills_the_square_only(scene):
    out = floodFillEx(scene, (40, 40), 10)
    expected = np.zeros((80, 80), np.uint8)
    expected[20:60, 20:60] = 255
    np.testing.assert_array_equal(out, expected)
    assert (scene[20:60, 20:60] == (0, 0, 255)).all()  # src untouched
    # From the background with a huge tolerance everything is filled.
    assert (floodFillEx(scene, (0, 0), 255) == 255).all()


def test_floodFillEx_tolerance_on_a_gradient():
    ramp = np.tile(np.arange(100, dtype=np.uint8), (10, 1))
    fixed = floodFillEx(ramp, (50, 5), 5, fixedRange=True)
    assert fixed[5, 45:56].all() and not fixed[5, 44] and not fixed[5, 56]
    floating = floodFillEx(ramp, (50, 5), 5, fixedRange=False)
    assert floating.all()  # each step is within 5, so the fill drifts across the whole ramp
    assert floodFillEx(ramp, (50, 5), 0).sum() == 255 * 10  # one column


def test_floodFillEx_connectivity_per_channel_tolerance_and_dtypes():
    diag = np.zeros((6, 6), np.uint8)
    diag[np.arange(6), np.arange(6)] = 255
    assert np.count_nonzero(floodFillEx(diag, (0, 0), 0, connectivity=4)) == 1
    assert np.count_nonzero(floodFillEx(diag, (0, 0), 0, connectivity=8)) == 6
    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[:, 2:] = (0, 0, 20)
    assert np.count_nonzero(floodFillEx(bgr, (0, 0), (0, 0, 25))) == 16
    assert np.count_nonzero(floodFillEx(bgr, (0, 0), (25, 25, 0))) == 8
    bgra = cv.cvtColor(bgr, cv.COLOR_BGR2BGRA)
    np.testing.assert_array_equal(floodFillEx(bgra, (0, 0), (0, 0, 25, 0)), floodFillEx(bgr, (0, 0), 25))
    f32 = np.zeros((4, 4), np.float32)
    f32[:, 2:] = 0.5
    assert np.count_nonzero(floodFillEx(f32, (0, 0), 0.6)) == 16
    assert np.count_nonzero(floodFillEx(f32, (0, 0), 0.4)) == 8


@pytest.mark.parametrize(
    ("src", "seed", "tolerance", "kwargs"),
    [
        (np.zeros((0, 4), np.uint8), (0, 0), 0, {}),
        (np.zeros((4, 4), np.uint16), (0, 0), 0, {}),
        (np.zeros((4, 4, 2), np.uint8), (0, 0), 0, {}),
        (np.zeros((4, 4), np.uint8), (4, 0), 0, {}),
        (np.zeros((4, 4), np.uint8), (0,), 0, {}),
        (np.zeros((4, 4), np.uint8), (0, 0), -1, {}),
        (np.zeros((4, 4), np.uint8), (0, 0), float("inf"), {}),
        (np.zeros((4, 4, 3), np.uint8), (0, 0), (1, 2), {}),
        (np.zeros((4, 4), np.uint8), (0, 0), 0, {"connectivity": 6}),
    ],
)
def test_floodFillEx_validation(src, seed, tolerance, kwargs):
    with pytest.raises(ValueError):
        floodFillEx(src, seed, tolerance, **kwargs)


# --- connectedComponentsEx / keepComponents -------------------------------------------


def test_connectedComponentsEx_sorted_with_masks(blobs):
    components = connectedComponentsEx(blobs)
    assert [c.area for c in components] == [2500, 100, 9]
    big = components[0]
    assert isinstance(big, Component)
    assert big.box == (10, 10, 50, 50)
    assert big.centroid == pytest.approx((34.5, 34.5))
    assert big.mask.dtype == np.uint8 and big.mask.shape == blobs.shape
    assert np.count_nonzero(big.mask) == 2500 and big.mask[10, 10] == 255
    assert sorted(c.label for c in components) == [1, 2, 3]
    union = mergeMasks([c.mask for c in components])
    np.testing.assert_array_equal(union, blobs)


def test_connectedComponentsEx_minArea_connectivity_and_empty(blobs):
    assert [c.area for c in connectedComponentsEx(blobs, minArea=10)] == [2500, 100]
    diag = np.zeros((6, 6), bool)
    diag[np.arange(6), np.arange(6)] = True
    assert len(connectedComponentsEx(diag, connectivity=8)) == 1
    assert len(connectedComponentsEx(diag, connectivity=4)) == 6
    assert connectedComponentsEx(np.zeros((5, 5), np.uint8)) == []
    color = cv.cvtColor(blobs, cv.COLOR_GRAY2BGR)
    assert [c.area for c in connectedComponentsEx(color)] == [2500, 100, 9]


@pytest.mark.parametrize("kwargs", [{"connectivity": 5}, {"minArea": -1}])
def test_connectedComponentsEx_validation(blobs, kwargs):
    with pytest.raises(ValueError):
        connectedComponentsEx(blobs, **kwargs)


def test_keepComponents_filters(blobs):
    assert np.count_nonzero(keepComponents(blobs, minArea=50)) == 2600
    assert np.count_nonzero(keepComponents(blobs, maxArea=200)) == 109
    assert np.count_nonzero(keepComponents(blobs, minArea=50, maxArea=200)) == 100
    assert np.count_nonzero(keepComponents(blobs, keepTop=2)) == 2600
    assert np.count_nonzero(keepComponents(blobs, keepTop=1, maxArea=200)) == 100
    assert np.count_nonzero(keepComponents(blobs, keepTop=0)) == 0
    np.testing.assert_array_equal(keepComponents(blobs), blobs)
    np.testing.assert_array_equal(keepComponents(blobs, keepTop=1), cv.largestComponent(blobs))
    np.testing.assert_array_equal(keepComponents(blobs, minArea=50), cv.removeSmallObjects(blobs, 50))
    out = keepComponents(blobs > 0, keepTop=5)
    assert out.dtype == np.uint8 and set(np.unique(out)) == {0, 255}


def test_keepComponents_connectivity_and_ties():
    diag = np.zeros((6, 6), np.uint8)
    diag[np.arange(6), np.arange(6)] = 255
    assert np.count_nonzero(keepComponents(diag, keepTop=1, connectivity=4)) == 1
    assert np.count_nonzero(keepComponents(diag, keepTop=1, connectivity=8)) == 6
    assert keepComponents(diag, keepTop=1, connectivity=4)[0, 0] == 255  # ties keep the first label


@pytest.mark.parametrize(
    "kwargs", [{"minArea": -1}, {"maxArea": -1}, {"keepTop": -1}, {"minArea": 5, "maxArea": 2}, {"connectivity": 0}]
)
def test_keepComponents_validation(blobs, kwargs):
    with pytest.raises(ValueError):
        keepComponents(blobs, **kwargs)


# --- trimap / smoothMask / maskOutline -------------------------------------------------


def test_trimap_bands():
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = 255
    out = trimap(mask, 3)
    assert out.dtype == np.uint8 and set(np.unique(out)) == {0, 128, 255}
    assert out[20, 20] == 255 and out[13, 13] == 255
    assert out[20, 12] == 128 and out[20, 8] == 128 and out[20, 10] == 128
    assert out[20, 6] == 0 and out[20, 33] == 0
    assert trimap(mask, 2, unknown=77)[20, 9] == 77
    np.testing.assert_array_equal(trimap(mask, 0), mask)
    np.testing.assert_array_equal(trimap(mask > 0, 3), out)


@pytest.mark.parametrize("kwargs", [{"width": -1}, {"width": 2, "unknown": 0}, {"width": 2, "unknown": 255}])
def test_trimap_validation(kwargs):
    with pytest.raises(ValueError):
        trimap(np.ones((5, 5), np.uint8), **kwargs)


def test_smoothMask_removes_staircase_and_keeps_edges():
    rng = np.random.default_rng(0)
    mask = np.zeros((80, 80), np.uint8)
    cv.circle(mask, (40, 40), 25, 255, cv.FILLED)
    noisy = mask.copy()
    # Nicks and spikes along the edge.
    edge = maskOutline(mask, 2) > 0
    flips = edge & (rng.random(mask.shape) < 0.3)
    noisy[flips] = 255
    inner = maskOutline(mask, 2, inside=True) > 0
    noisy[inner & (rng.random(mask.shape) < 0.3)] = 0
    out = smoothMask(noisy, 2)
    assert out.dtype == np.uint8 and set(np.unique(out)) <= {0, 255}
    assert abs(np.count_nonzero(out) - np.count_nonzero(mask)) < 0.03 * np.count_nonzero(mask)
    assert np.count_nonzero(out != mask) < np.count_nonzero(noisy != mask)
    assert out[40, 40] == 255 and out[2, 2] == 0
    # The threshold grows (< 0.5) or shrinks (> 0.5) the mask.
    assert np.count_nonzero(smoothMask(mask, 3, threshold=0.2)) > np.count_nonzero(mask)
    assert np.count_nonzero(smoothMask(mask, 3, threshold=0.8)) < np.count_nonzero(mask)
    np.testing.assert_array_equal(smoothMask(mask > 0, 0), mask)


@pytest.mark.parametrize("kwargs", [{"radius": -1}, {"radius": 1, "threshold": 0}, {"radius": 1, "threshold": 1}])
def test_smoothMask_validation(kwargs):
    with pytest.raises(ValueError):
        smoothMask(np.ones((5, 5), np.uint8), **kwargs)


def test_maskOutline_outside_and_inside():
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = 255
    outer = maskOutline(mask, 2)
    assert outer.dtype == np.uint8 and set(np.unique(outer)) == {0, 255}
    assert (outer[mask > 0] == 0).all()
    assert outer[20, 9] == 255 and outer[20, 8] == 255 and outer[20, 7] == 0
    inner = maskOutline(mask, 2, inside=True)
    assert (inner[mask == 0] == 0).all()
    assert inner[20, 10] == 255 and inner[20, 11] == 255 and inner[20, 12] == 0
    assert inner[20, 20] == 0
    one = maskOutline(mask > 0)
    assert np.count_nonzero(one) == 4 * 20  # a 1 px ring (the 3x3 ellipse kernel is a cross: no corners)
    with pytest.raises(ValueError):
        maskOutline(mask, 0)


# --- mergeMasks / maskHull / maskCoverage ----------------------------------------------


def test_mergeMasks_modes():
    a = np.zeros((10, 10), np.uint8)
    a[:, :6] = 255
    b = np.zeros((10, 10), bool)
    b[:, 4:] = True
    c = np.zeros((10, 10, 3), np.uint8)
    c[:2] = (0, 0, 9)
    assert np.count_nonzero(mergeMasks([a, b])) == 100
    assert np.count_nonzero(mergeMasks([a, b], "intersection")) == 20
    assert np.count_nonzero(mergeMasks([a, b], "difference")) == 40
    assert np.count_nonzero(mergeMasks([a, b, c], "intersection")) == 4
    assert np.count_nonzero(mergeMasks([a, b, c], "difference")) == 40 - 8
    out = mergeMasks([a])
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, a)


@pytest.mark.parametrize(
    ("masks", "mode"),
    [
        ([], "union"),
        ([np.zeros((4, 4), np.uint8), np.zeros((4, 5), np.uint8)], "union"),
        ([np.zeros((4, 4), np.uint8)], "xor"),
        ([np.zeros((0, 4), np.uint8)], "union"),
    ],
)
def test_mergeMasks_validation(masks, mode):
    with pytest.raises(ValueError):
        mergeMasks(masks, mode)


def test_maskHull_fills_the_hull():
    mask = np.zeros((50, 50), np.uint8)
    cv.circle(mask, (25, 25), 20, 255, cv.FILLED)
    mask[25:, :] = 0  # half disc
    mask[25, 5] = 255
    mask[25, 45] = 255
    out = maskHull(mask)
    assert out.dtype == np.uint8 and set(np.unique(out)) == {0, 255}
    assert (out[mask > 0] == 255).all()
    assert out[20, 25] == 255 and out[24, 24] == 255
    assert out[30, 25] == 0  # below the half disc
    assert np.count_nonzero(out) > np.count_nonzero(mask)
    assert maskHull(np.zeros((5, 5), np.uint8)).sum() == 0
    np.testing.assert_array_equal(maskHull(cv.cvtColor(mask, cv.COLOR_GRAY2BGR)), out)


def test_maskCoverage_full_and_roi():
    mask = np.zeros((10, 20), np.uint8)
    mask[:, :5] = 255
    assert maskCoverage(mask) == 0.25
    assert maskCoverage(mask > 0) == 0.25
    assert maskCoverage(mask, roi=(0, 0, 5, 10)) == 1.0
    assert maskCoverage(mask, roi=(0, 0, 10, 10)) == 0.5
    assert maskCoverage(mask, roi=(-5, -5, 10, 10)) == 1.0  # clipped to the image
    assert maskCoverage(mask, roi=(30, 0, 5, 5)) == 0.0  # outside the image
    assert maskCoverage(np.zeros((4, 4), np.uint8)) == 0.0


@pytest.mark.parametrize("roi", [(0, 0, 5), (0, 0, 0, 5), (0, 0, 5, -1)])
def test_maskCoverage_validation(roi):
    with pytest.raises(ValueError):
        maskCoverage(np.ones((5, 5), np.uint8), roi=roi)


def test_foreground_validation():
    with pytest.raises(ValueError, match="mask"):
        maskHull(np.zeros(5, np.uint8))
    with pytest.raises(ValueError, match="mask"):
        maskHull(np.zeros((0, 5), np.uint8))
