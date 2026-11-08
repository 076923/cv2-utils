import numpy as np
import pytest

import cv2u as cv


@pytest.fixture
def shapes():
    # A square on the left, a circle on the right, a thin bar at the bottom (in scan order: square, circle, bar).
    img = np.zeros((120, 160), np.uint8)
    cv.rectangle(img, (20, 20), (70, 70), 255, cv.FILLED)
    cv.circle(img, (115, 45), 25, 255, cv.FILLED)
    cv.rectangle(img, (10, 100), (150, 105), 255, cv.FILLED)
    return img


@pytest.fixture
def contours(shapes):
    # cv.findContours scans bottom-up; fix the order to square, circle, bar (by top edge, then left edge).
    return sorted(cv.findContoursEx(shapes, sort=False), key=lambda c: cv.boundingRect(c)[1::-1])


# --- asContour ---------------------------------------------------------------------


@pytest.mark.parametrize("points", [[(0, 0), (10, 0), (10, 10)], np.array([[0, 0], [10, 0], [10, 10]], np.float32)])
def test_asContour_layout_and_dtype(points):
    contour = cv.asContour(points)
    assert contour.shape == (3, 1, 2)
    assert contour.dtype == np.int32
    assert cv.contourArea(contour) == 50


def test_asContour_rounds_floats():
    np.testing.assert_array_equal(cv.asContour([(0.6, 1.4)]).reshape(-1), [1, 1])


def test_asContour_keeps_a_contour_as_is():
    contour = np.zeros((4, 1, 2), np.int32)
    assert cv.asContour(contour).shape == (4, 1, 2)


@pytest.mark.parametrize("points", [[], [(1, 2, 3)], [(float("nan"), 0)]])
def test_asContour_validation(points):
    with pytest.raises(ValueError):
        cv.asContour(points)


# --- boundingRects -----------------------------------------------------------------


def test_boundingRects(contours):
    rects = cv.boundingRects(contours)
    assert rects.shape == (3, 4)
    assert rects.dtype == np.int32
    assert [tuple(r) for r in rects] == [cv.boundingRect(c) for c in contours]


def test_boundingRects_empty():
    assert cv.boundingRects([]).shape == (0, 4)


# --- contourStats ------------------------------------------------------------------


def test_contourStats_square(contours):
    s = cv.contourStats(contours[0])
    assert s.rect == (20, 20, 51, 51)
    assert s.area == pytest.approx(50 * 50)
    assert s.perimeter == pytest.approx(200)
    assert s.center == pytest.approx((45, 45))
    assert s.aspectRatio == 1
    assert s.extent == pytest.approx(2500 / 51**2)
    assert s.solidity == pytest.approx(1)
    assert s.circularity == pytest.approx(np.pi / 4, abs=0.01)
    assert s.equivalentDiameter == pytest.approx(np.sqrt(4 * 2500 / np.pi))


def test_contourStats_circle_is_round(contours):
    assert cv.contourStats(contours[1]).circularity > 0.85
    assert cv.contourStats(contours[0]).circularity < 0.8


def test_contourStats_orientation_of_a_bar(contours):
    orientation = cv.contourStats(contours[2]).orientation
    assert orientation == pytest.approx(90, abs=1) or orientation == pytest.approx(0, abs=1) or orientation > 179


def test_contourStats_accepts_points_and_degenerate_shapes():
    line = cv.contourStats([(0, 0), (10, 0)])
    assert line.area == 0
    assert line.solidity == 0
    assert line.circularity == 0
    assert line.center == (5, 0)
    assert line.rect == (0, 0, 11, 1)


# --- sortContours ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("left-to-right", [2, 0, 1]),  # bar starts at x=10, square at 20, circle at 90
        ("right-to-left", [1, 0, 2]),
        ("top-to-bottom", [0, 1, 2]),  # square y=20, circle y=20, bar y=100 (stable for ties)
        ("bottom-to-top", [2, 0, 1]),
        ("area", [0, 1, 2]),
        ("perimeter", [2, 0, 1]),  # the bar is 140 wide: longest outline
    ],
)
def test_sortContours(contours, method, expected):
    sorted_ = cv.sortContours(contours, method)
    assert [cv.boundingRect(c) for c in sorted_] == [cv.boundingRect(contours[i]) for i in expected]


def test_sortContours_returns_a_new_list(contours):
    out = cv.sortContours(contours)
    assert out is not contours
    assert cv.sortContours([]) == []


def test_sortContours_validation(contours):
    with pytest.raises(ValueError, match="method"):
        cv.sortContours(contours, "diagonal")  # type: ignore[arg-type]


# --- filterContours ----------------------------------------------------------------


def test_filterContours_by_area_and_shape(contours):
    assert len(cv.filterContours(contours, minArea=1000)) == 2
    assert len(cv.filterContours(contours, maxArea=1000)) == 1
    assert len(cv.filterContours(contours, minCircularity=0.85)) == 1
    assert len(cv.filterContours(contours, minAspectRatio=5)) == 1
    assert len(cv.filterContours(contours, minExtent=0.9)) == 1  # only the square fills its box
    assert cv.filterContours(contours) == contours


def test_filterContours_keeps_the_original_order(contours):
    kept = cv.filterContours(contours, minArea=1000)
    assert [cv.boundingRect(c) for c in kept] == [cv.boundingRect(c) for c in contours[:2]]


@pytest.mark.parametrize("kwargs", [{"minArea": -1}, {"minArea": float("nan")}, {"minArea": 10, "maxArea": 5}])
def test_filterContours_validation(contours, kwargs):
    with pytest.raises(ValueError):
        cv.filterContours(contours, **kwargs)


# --- approxContour -----------------------------------------------------------------


def test_approxContour_finds_the_corners_of_a_square(contours):
    circle = contours[1]
    assert len(cv.approxContour(circle, 0.02)) < len(circle)
    assert len(cv.approxContour(circle, 0.5)) <= 4
    assert len(cv.approxContour(contours[0])) == 4


def test_approxContour_zero_keeps_the_contour(contours):
    assert len(cv.approxContour(contours[1], 0)) == len(contours[1])


def test_approxContour_validation(contours):
    with pytest.raises(ValueError, match="epsilon"):
        cv.approxContour(contours[0], -0.1)


# --- mergeContours -----------------------------------------------------------------


def test_mergeContours_is_the_hull_of_all_pieces(contours):
    merged = cv.mergeContours(contours[:2])
    assert cv.isContourConvex(merged)
    x, y, w, h = cv.boundingRect(merged)
    assert (x, y, x + w, y + h) == (20, 20, 141, 71)


def test_mergeContours_validation():
    with pytest.raises(ValueError, match="empty"):
        cv.mergeContours([])


# --- contoursToMask ----------------------------------------------------------------


def test_contoursToMask_round_trips_findContoursEx(shapes, contours):
    mask = cv.contoursToMask(contours, (160, 120))
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 255}
    np.testing.assert_array_equal(mask, shapes)


def test_contoursToMask_single_contour_and_outline(contours):
    filled = cv.contoursToMask(contours[0], (160, 120))
    outline = cv.contoursToMask(contours[0], (160, 120), thickness=1)
    assert filled.sum() > outline.sum() > 0
    np.testing.assert_array_equal(cv.contoursToMask(contours[0].reshape(-1, 2), (160, 120)), filled)


def test_contoursToMask_empty_and_validation():
    assert cv.contoursToMask([], (8, 6)).shape == (6, 8)
    with pytest.raises(ValueError, match="size"):
        cv.contoursToMask([], (0, 6))


# --- cropContour -------------------------------------------------------------------


def test_cropContour_blanks_the_outside(contours):
    img = np.full((120, 160, 3), 200, np.uint8)
    crop = cv.cropContour(img, contours[1])
    assert crop.shape == (51, 51, 3)
    assert tuple(crop[0, 0]) == (0, 0, 0)  # the corner of the box lies outside the circle
    assert tuple(crop[25, 25]) == (200, 200, 200)
    assert tuple(cv.cropContour(img, contours[1], fill=(1, 2, 3))[0, 0]) == (1, 2, 3)


def test_cropContour_margin_is_clipped_and_fill_none_keeps_pixels(contours):
    img = np.full((120, 160), 200, np.uint8)
    crop = cv.cropContour(img, contours[1], margin=100, fill=None)
    assert crop.shape == (120, 160)
    assert crop.min() == 200
    assert cv.cropContour(img, contours[0], margin=5).shape == (61, 61)


def test_cropContour_transparent(contours):
    img = np.full((120, 160, 3), 200, np.uint8)
    out = cv.cropContour(img, contours[1], transparent=True)
    assert out.shape == (51, 51, 4)
    assert out[0, 0, 3] == 0
    assert out[25, 25, 3] == 255
    assert tuple(out[0, 0, :3]) == (200, 200, 200)  # colors are untouched


def test_cropContour_does_not_modify_the_source(contours):
    img = np.full((120, 160), 200, np.uint8)
    cv.cropContour(img, contours[0])
    assert img.min() == 200


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"margin": -1}, "margin"),
        ({"transparent": True}, "8-bit"),
        ({"fill": float("nan")}, "fill"),
    ],
)
def test_cropContour_validation(contours, kwargs, match):
    img = np.zeros((120, 160), np.float32)
    with pytest.raises(ValueError, match=match):
        cv.cropContour(img, contours[0], **kwargs)


def test_cropContour_outside_the_image(contours):
    with pytest.raises(ValueError, match="outside"):
        cv.cropContour(np.zeros((10, 10), np.uint8), contours[1])


# --- drawContoursEx ----------------------------------------------------------------


def test_drawContoursEx_uses_a_different_color_per_contour(shapes, contours):
    img = cv.toBGR(shapes)
    out = cv.drawContoursEx(img, contours, thickness=cv.FILLED)
    assert out is img  # drawn in place, like cv.drawContours
    colors = {tuple(out[45, 45]), tuple(out[45, 115]), tuple(out[102, 80])}
    assert len(colors) == 3
    assert colors == {tuple(c) for c in cv.distinctColors(3)}


def test_drawContoursEx_gray_stays_gray(shapes, contours):
    img = shapes.copy()
    out = cv.drawContoursEx(img, contours, thickness=cv.FILLED)
    assert out is img and out.ndim == 2
    assert out[45, 45] != out[45, 115]  # the luminance of each color


def test_drawContoursEx_labels(shapes, contours):
    plain = cv.drawContoursEx(cv.toBGR(shapes), contours)
    numbered = cv.drawContoursEx(cv.toBGR(shapes), contours, labels=True)
    named = cv.drawContoursEx(cv.toBGR(shapes), contours, labels=["a", "b", "c"])
    assert (numbered != plain).any()
    assert (named != plain).any()


def test_drawContoursEx_custom_colors_and_bgra(shapes, contours):
    bgra = cv.toBGRA(shapes)
    out = cv.drawContoursEx(bgra, contours, colors=[(0, 0, 255)], thickness=cv.FILLED)
    assert out.shape[2] == 4
    assert tuple(out[45, 45]) == (0, 0, 255, 255)
    assert tuple(out[45, 115]) == (0, 0, 255, 255)  # one color: cycled


def test_drawContoursEx_validation(shapes, contours):
    with pytest.raises(ValueError, match="labels"):
        cv.drawContoursEx(shapes, contours, labels=["only one"])
    with pytest.raises(ValueError, match="8-bit"):
        cv.drawContoursEx(shapes.astype(np.float32), contours)
    with pytest.raises(ValueError, match="fontScale"):
        cv.drawContoursEx(shapes, contours, fontScale=0)
    with pytest.raises(ValueError, match="colors"):
        cv.drawContoursEx(shapes, contours, colors=[])
    with pytest.raises(ValueError, match="empty"):
        cv.drawContoursEx(np.zeros((0, 4), np.uint8), contours)
