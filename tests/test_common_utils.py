"""Boxes, detectors, binarization, morphology, feature matching, drawing helpers and crops added in 0.3.0."""

import numpy as np
import pytest

import cv2u as cv


@pytest.fixture
def scene():
    # Two black discs, a black square and a long bar on white.
    img = np.full((200, 300, 3), 255, np.uint8)
    cv.circle(img, (80, 80), 30, (0, 0, 0), -1)
    cv.circle(img, (220, 120), 20, (0, 0, 0), -1)
    cv.rectangle(img, (150, 20), (200, 60), (0, 0, 0), -1)
    cv.line(img, (10, 190), (290, 190), (0, 0, 0), 3)
    return img


BOXES = np.array([[10, 10, 50, 50], [12, 12, 50, 50], [100, 100, 20, 20]], np.float64)


# --- boxes ---------------------------------------------------------------------------


def test_xywh_xyxy_round_trip():
    np.testing.assert_array_equal(cv.xywhToXyxy(BOXES[0]), [10, 10, 60, 60])
    np.testing.assert_array_equal(cv.xyxyToXywh(cv.xywhToXyxy(BOXES)), BOXES)
    ints = cv.xywhToXyxy(np.array([[1, 2, 3, 4]], np.int32))
    assert ints.dtype == np.int32
    with pytest.raises(ValueError):
        cv.xywhToXyxy([1, 2, 3])


def test_nms_keeps_the_best_of_overlapping_boxes():
    np.testing.assert_array_equal(cv.nms(BOXES, [0.9, 0.8, 0.7]), [0, 2])
    np.testing.assert_array_equal(cv.nms(BOXES, [0.5, 0.9, 0.7]), [1, 2])  # best score wins, not first
    np.testing.assert_array_equal(cv.nms(BOXES, [0.9, 0.8, 0.7], iouThreshold=0.95), [0, 1, 2])
    np.testing.assert_array_equal(cv.nms(BOXES, [0.9, 0.8, 0.7], scoreThreshold=0.75), [0])
    np.testing.assert_array_equal(cv.nms(BOXES, [0.9, 0.8, 0.7], maxCount=1), [0])


def test_nms_empty_and_validation():
    out = cv.nms([], [])
    assert out.shape == (0,)
    assert out.dtype == np.int64
    with pytest.raises(ValueError, match="scores"):
        cv.nms(BOXES, [1, 2])
    with pytest.raises(ValueError):
        cv.nms(BOXES, [1, 2, 3], iouThreshold=float("nan"))


def test_clipBoxes():
    np.testing.assert_array_equal(cv.clipBoxes([-10, -10, 50, 50], (30, 30)), [0, 0, 30, 30])
    np.testing.assert_array_equal(cv.clipBoxes([[100, 100, 5, 5]], (30, 30)), [[30, 30, 0, 0]])
    assert cv.clipBoxes(BOXES, (300, 300)).shape == (3, 4)
    with pytest.raises(ValueError, match="size"):
        cv.clipBoxes(BOXES, (0, 10))


def test_scaleBoxes_undoes_a_letterbox(scene):
    _, scale, (left, top) = cv.letterbox(scene, (160, 160))
    box = np.array([80 - 30, 80 - 30, 60, 60], np.float64)  # the first disc in the original
    small = cv.scaleBoxes(box, scale, (left, top))
    np.testing.assert_allclose(cv.scaleBoxes(small, scale, (left, top), inverse=True), box)
    np.testing.assert_array_equal(cv.scaleBoxes([10, 20, 30, 40], (2, 0.5)), [20, 10, 60, 20])
    with pytest.raises(ValueError, match="scale"):
        cv.scaleBoxes(box, 0)


def test_expandBoxes():
    np.testing.assert_array_equal(cv.expandBoxes([10, 10, 20, 20], ratio=0.5), [0, 0, 40, 40])
    np.testing.assert_array_equal(cv.expandBoxes([10, 10, 20, 20], pixels=2), [8, 8, 24, 24])
    np.testing.assert_array_equal(cv.expandBoxes([10, 10, 20, 20], ratio=0.5, size=(25, 100)), [0, 0, 25, 40])
    np.testing.assert_array_equal(cv.expandBoxes([10, 10, 20, 20], ratio=-1), [20, 20, 0, 0])  # never negative
    assert cv.expandBoxes(BOXES, 0.1).shape == (3, 4)


# --- detectors -------------------------------------------------------------------------


def test_findLines_finds_the_bar_longest_first(scene):
    lines = cv.findLines(scene, minLength=100)
    assert lines.shape[1] == 4
    assert len(lines) >= 1
    x1, y1, x2, y2 = lines[0]
    assert abs(y1 - y2) <= 2 and abs(y1 - 190) <= 4
    assert abs(x2 - x1) > 250
    lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    assert list(lengths) == sorted(lengths, reverse=True)


def test_findLines_accepts_an_edge_image_and_returns_empty(scene):
    edges = cv.autoCanny(scene)
    assert len(cv.findLines(scene, edges=edges, minLength=100)) == len(cv.findLines(scene, minLength=100))
    assert cv.findLines(np.zeros((50, 50), np.uint8)).shape == (0, 4)


def test_findCircles(scene):
    circles = cv.findCircles(scene, minRadius=10, maxRadius=50)
    assert circles.shape == (2, 3)
    found = sorted((round(x), round(y), round(r)) for x, y, r in circles)
    assert abs(found[0][0] - 80) <= 2 and abs(found[0][1] - 80) <= 2 and abs(found[0][2] - 30) <= 2
    assert abs(found[1][0] - 220) <= 2 and abs(found[1][1] - 120) <= 3 and abs(found[1][2] - 20) <= 2
    assert cv.findCircles(np.full((50, 50), 255, np.uint8)).shape == (0, 3)


@pytest.mark.parametrize("kwargs", [{"blur": 4}, {"minRadius": -1}, {"sensitivity": 0}])
def test_findCircles_validation(scene, kwargs):
    with pytest.raises(ValueError):
        cv.findCircles(scene, **kwargs)


def test_detectBlobs(scene):
    blobs = cv.detectBlobs(scene, minArea=500, maxArea=5000, minCircularity=0.8)
    assert blobs.shape == (2, 3)
    assert blobs[0, 2] >= blobs[1, 2]  # largest first
    assert abs(blobs[0, 0] - 80) <= 2 and abs(blobs[0, 1] - 80) <= 2
    assert cv.detectBlobs(255 - scene, minArea=500, minCircularity=0.8).shape == (0, 3)  # the discs are light now
    assert cv.detectBlobs(255 - scene, minArea=500, dark=False, minCircularity=0.8).shape == (2, 3)


@pytest.mark.parametrize(
    "kwargs", [{"minArea": -1}, {"minArea": 10, "maxArea": 5}, {"minCircularity": 2}, {"minThreshold": 300}]
)
def test_detectBlobs_validation(scene, kwargs):
    with pytest.raises(ValueError):
        cv.detectBlobs(scene, **kwargs)


# --- binarize / morphology -----------------------------------------------------------


def test_binarize_methods(scene):
    otsu = cv.binarize(scene, invert=True)
    assert otsu.dtype == np.uint8 and otsu.shape == (200, 300)
    assert set(np.unique(otsu)) == {0, 255}
    assert otsu[80, 80] == 255 and otsu[5, 5] == 0
    assert cv.binarize(scene)[80, 80] == 0  # not inverted: dark objects are background
    np.testing.assert_array_equal(cv.binarize(scene, 128, invert=True), otsu)
    assert cv.binarize(scene, "triangle", invert=True)[80, 80] == 255
    adaptive = cv.binarize(scene, "adaptive", invert=True, blockSize=51, C=5)
    assert adaptive[80, 50] == 255  # the edge of the disc is darker than its surroundings
    assert cv.binarize(scene, blur=3).shape == (200, 300)


def test_binarize_float_image_with_fixed_threshold():
    img = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    mask = cv.binarize(img, 0.5)
    assert mask.dtype == np.uint8 and mask[0, 0] == 0 and mask[9, 9] == 255
    with pytest.raises(ValueError, match="8-bit"):
        cv.binarize(img, "otsu")


@pytest.mark.parametrize("kwargs", [{"method": "magic"}, {"blur": 2}, {"method": "adaptive", "blockSize": 4}])
def test_binarize_validation(scene, kwargs):
    with pytest.raises(ValueError):
        cv.binarize(scene, **kwargs)


def test_morphology(scene):
    mask = cv.binarize(scene, invert=True)
    assert cv.morphology(mask, "open", 9).sum() < mask.sum()  # the 3-px bar disappears
    assert cv.morphology(mask, "dilate", 3).sum() > mask.sum()
    assert cv.morphology(mask, "erode", (3, 1), "rect").sum() < mask.sum()
    assert cv.morphology(mask, "close", 3, "rect", iterations=2).shape == mask.shape
    assert (cv.morphology(mask, "gradient", 3) & mask)[80, 80] == 0  # the outline is hollow inside
    for kwargs in (
        {"op": "blur"},
        {"op": "open", "shape": "star"},
        {"op": "open", "ksize": 0},
        {"op": "open", "iterations": 0},
    ):
        with pytest.raises(ValueError):
            cv.morphology(mask, **kwargs)  # type: ignore[arg-type]


# --- masks and crops -------------------------------------------------------------------


def test_maskBoundingRect():
    mask = np.zeros((20, 30), np.uint8)
    mask[5:9, 10:14] = 1
    assert cv.maskBoundingRect(mask) == (10, 5, 4, 4)
    assert cv.maskBoundingRect(np.dstack([mask, mask, mask])) == (10, 5, 4, 4)
    assert cv.maskBoundingRect(mask.astype(bool)) == (10, 5, 4, 4)
    assert cv.maskBoundingRect(np.zeros((20, 30), np.uint8)) is None


def test_applyMask():
    img = np.full((4, 4, 3), 200, np.uint8)
    mask = np.zeros((4, 4), np.uint8)
    mask[1, 1] = 255
    out = cv.applyMask(img, mask, (1, 2, 3))
    assert tuple(out[1, 1]) == (200, 200, 200) and tuple(out[0, 0]) == (1, 2, 3)
    assert cv.applyMask(img.astype(np.float32), mask)[0, 0, 0] == 0
    assert cv.applyMask(img[..., 0], mask.astype(bool), 7)[0, 0] == 7
    with pytest.raises(ValueError, match="shape"):
        cv.applyMask(img, mask[:2])


def test_centerCrop():
    img = np.arange(100 * 60).reshape(60, 100).astype(np.uint8)
    out = cv.centerCrop(img, 20)
    assert out.shape == (20, 20)
    np.testing.assert_array_equal(out, img[20:40, 40:60])
    assert cv.centerCrop(img, (30, 10)).shape == (10, 30)
    assert cv.centerCrop(img, (200, 10)).shape == (10, 100)  # clipped
    assert cv.centerCrop(img, (200, 10), fill=0).shape == (10, 200)  # padded
    with pytest.raises(ValueError, match="size"):
        cv.centerCrop(img, 0)


# --- drawing -------------------------------------------------------------------------


def test_distinctColors():
    assert cv.distinctColors(3) == [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    many = cv.distinctColors(100)
    assert len(many) == 100 and len(set(many)) == 100
    assert cv.distinctColors(0) == []
    assert cv.distinctColors(5, seed=1) != cv.distinctColors(5) and sorted(cv.distinctColors(5, seed=1)) == sorted(
        cv.distinctColors(5)
    )
    with pytest.raises(ValueError):
        cv.distinctColors(-1)


def test_drawBoxes(scene):
    plain = cv.drawBoxes(scene.copy(), BOXES)
    assert (plain != scene).any()
    labeled = cv.drawBoxes(scene.copy(), BOXES, labels=["a", "b", "c"])
    assert (labeled != plain).any()
    one_color = cv.drawBoxes(scene.copy(), BOXES, colors=(255, 0, 0), thickness=cv.FILLED)
    assert tuple(one_color[30, 30]) == (255, 0, 0) and tuple(one_color[105, 105]) == (255, 0, 0)
    per_box = cv.drawBoxes(scene.copy(), BOXES, colors=[(1, 1, 1), (2, 2, 2), (3, 3, 3)], thickness=cv.FILLED)
    assert tuple(per_box[105, 105]) == (3, 3, 3)
    assert cv.drawBoxes(scene.copy(), []) is not None
    with pytest.raises(ValueError, match="labels"):
        cv.drawBoxes(scene.copy(), BOXES, labels=["a"])
    with pytest.raises(ValueError, match="colors"):
        cv.drawBoxes(scene.copy(), BOXES, colors=[(1, 1, 1)])


def test_drawPoints(scene):
    out = cv.drawPoints(np.zeros((50, 50, 3), np.uint8), [(10, 10), (30.4, 29.6)], radius=2)
    assert tuple(out[10, 10]) == (0, 0, 255) and tuple(out[30, 30]) == (0, 0, 255) and tuple(out[0, 0]) == (0, 0, 0)
    labeled = cv.drawPoints(np.zeros((50, 50, 3), np.uint8), [(10, 10)], labels=True)
    assert (labeled != cv.drawPoints(np.zeros((50, 50, 3), np.uint8), [(10, 10)])).any()
    contour = np.array([[[5, 5]], [[7, 7]]], np.int32)
    assert cv.drawPoints(np.zeros((10, 10), np.uint8), contour, color=9, radius=1)[5, 5] == 9
    with pytest.raises(ValueError):
        cv.drawPoints(scene, [(1, 1)], labels=["a", "b"])
    with pytest.raises(ValueError):
        cv.drawPoints(scene, [(1, 1)], radius=-1)


def test_drawGrid():
    out = cv.drawGrid(np.zeros((20, 30), np.uint8), 10, color=255)
    assert out[:, 0].all() and out[:, 10].all() and out[:, 20].all() and out[0].all() and out[10].all()
    assert out[5, 5] == 0
    shifted = cv.drawGrid(np.zeros((20, 30), np.uint8), (10, 5), color=255, offset=(3, 2))
    assert shifted[:, 3].all() and shifted[2].all() and shifted[7].all()
    with pytest.raises(ValueError):
        cv.drawGrid(out, 0)


# --- features ------------------------------------------------------------------------


@pytest.fixture
def photo():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    return cv.GaussianBlur(img, (0, 0), 1.5)  # texture with structure at several scales


@pytest.mark.parametrize("method", ["ORB", "SIFT"])
def test_matchFeatures_finds_the_shift(photo, method):
    moved = cv.translate(photo, 7, -4)
    found = cv.matchFeatures(photo, moved, method)
    assert len(found.matches) >= 20
    assert found.srcPoints.shape == found.dstPoints.shape == (len(found.matches), 2)
    shift = np.median(found.dstPoints - found.srcPoints, axis=0)
    np.testing.assert_allclose(shift, [7, -4], atol=1.0)
    assert found.srcKeypoints and found.dstKeypoints
    cv.drawMatches(photo, found.srcKeypoints, moved, found.dstKeypoints, found.matches[:10], None)


def test_matchFeatures_nothing_to_match():
    flat = np.zeros((50, 50), np.uint8)
    found = cv.matchFeatures(flat, flat)
    assert found.srcPoints.shape == (0, 2) and found.matches == []


@pytest.mark.parametrize("kwargs", [{"method": "AKAZE"}, {"ratio": 0}, {"ratio": 1.5}, {"maxFeatures": 0}])
def test_matchFeatures_validation(photo, kwargs):
    with pytest.raises(ValueError):
        cv.matchFeatures(photo, photo, **kwargs)


@pytest.mark.parametrize("motion", ["homography", "affine", "similarity"])
def test_alignImages_recovers_a_rotated_image(photo, motion):
    rotated = cv.rotateEx(photo, 6)
    aligned, matrix = cv.alignImages(rotated, photo, method="SIFT", motion=motion)
    assert aligned.shape == photo.shape
    assert matrix.shape == ((3, 3) if motion == "homography" else (2, 3))
    inner = (slice(40, 160), slice(60, 240))
    assert cv.ssim(cv.toGray(aligned)[inner], cv.toGray(photo)[inner]) > 0.6


def test_alignImages_validation(photo):
    with pytest.raises(ValueError, match="matches"):
        cv.alignImages(np.zeros((50, 50), np.uint8), photo)
    with pytest.raises(ValueError, match="minMatches"):
        cv.alignImages(photo, photo, minMatches=2)
