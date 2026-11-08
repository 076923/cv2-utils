import numpy as np
import pytest

import cv2u as cv


@pytest.fixture
def shapes():
    img = np.full((120, 160), 40, np.uint8)
    cv.rectangle(img, (20, 20), (70, 90), 200, cv.FILLED)
    cv.circle(img, (115, 60), 25, 150, cv.FILLED)
    return img


# --- autoCanny ---------------------------------------------------------------------


def test_autoCanny_matches_canny_with_median_thresholds(shapes):
    median = float(np.median(shapes))
    expected = cv.Canny(shapes, int(0.67 * median), int(min(255, 1.33 * median)))
    np.testing.assert_array_equal(cv.autoCanny(shapes), expected)


def test_autoCanny_color_and_bgra(shapes):
    bgr = cv.cvtColor(shapes, cv.COLOR_GRAY2BGR)
    np.testing.assert_array_equal(cv.autoCanny(bgr), cv.autoCanny(shapes))
    np.testing.assert_array_equal(cv.autoCanny(cv.cvtColor(bgr, cv.COLOR_BGR2BGRA)), cv.autoCanny(shapes))


def test_autoCanny_finds_the_shapes(shapes):
    edges = cv.autoCanny(shapes)
    assert edges.dtype == np.uint8
    assert set(np.unique(edges)) <= {0, 255}
    assert edges[20:91, 18:23].any()  # left side of the rectangle
    assert not edges[100:, :].any()  # nothing below the shapes


def test_autoCanny_accepts_16_bit_and_float_images(shapes):
    # Stretched to 8 bits first: cv.Canny itself rejects anything but uint8.
    np.testing.assert_array_equal(cv.autoCanny(shapes.astype(np.uint16) * 257), cv.autoCanny(shapes))
    np.testing.assert_array_equal(cv.autoCanny(shapes.astype(np.float32) / 255), cv.autoCanny(shapes))


@pytest.mark.parametrize(
    ("src", "sigma"),
    [(np.zeros((0, 4), np.uint8), 0.33), (np.zeros((4, 4), np.uint8), 1.5), (np.zeros((4, 4, 2)), 0.3)],
)
def test_autoCanny_validation(src, sigma):
    with pytest.raises(ValueError):
        cv.autoCanny(src, sigma=sigma)


# --- fourPointTransform -------------------------------------------------------------


@pytest.fixture
def document():
    # A 200x100 "page" (white, with a black mark at its top-left corner), warped into a quadrilateral.
    page = np.full((100, 200), 255, np.uint8)
    page[:20, :20] = 0
    quad = np.array([[60, 40], [250, 70], [230, 190], [40, 150]], np.float32)
    matrix = cv.getPerspectiveTransform(np.array([[0, 0], [199, 0], [199, 99], [0, 99]], np.float32), quad)
    photo = cv.warpPerspective(page, matrix, (300, 230), borderValue=128)
    return photo, quad


def test_fourPointTransform_recovers_the_page(document):
    photo, quad = document
    page = cv.fourPointTransform(photo, quad, size=(200, 100))
    assert page.shape == (100, 200)
    assert page[5:15, 5:15].mean() < 60  # the mark is back at the top-left
    assert page[40:60, 90:110].mean() > 200


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]])
def test_fourPointTransform_point_order_does_not_matter(document, order):
    photo, quad = document
    np.testing.assert_array_equal(
        cv.fourPointTransform(photo, quad[order], size=(200, 100)), cv.fourPointTransform(photo, quad, size=(200, 100))
    )


def test_fourPointTransform_automatic_size_and_contour_input():
    src = np.zeros((50, 80, 3), np.uint8)
    contour = np.array([[[10, 5]], [[70, 5]], [[70, 45]], [[10, 45]]], np.int32)  # (4, 1, 2), like approxPolyDP
    assert cv.fourPointTransform(src, contour).shape == (40, 60, 3)


def test_fourPointTransform_rotated_square_is_ordered():
    # A diamond: the sum-of-coordinates trick alone cannot order these corners.
    img = np.zeros((101, 101), np.uint8)
    img[45:56, 45:56] = 255
    diamond = [[50, 0], [100, 50], [50, 100], [0, 50]]
    out = cv.fourPointTransform(img, diamond)
    assert out.shape[0] == out.shape[1]
    assert out[out.shape[0] // 2, out.shape[1] // 2] == 255


@pytest.mark.parametrize("pts", [[[0, 0], [1, 0], [1, 1]], [[5, 5]] * 4])
def test_fourPointTransform_validation(pts):
    with pytest.raises(ValueError):
        cv.fourPointTransform(np.zeros((10, 10), np.uint8), pts)
