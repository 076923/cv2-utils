import numpy as np
import pytest

import cv2u as cv


@pytest.fixture
def img():
    return np.arange(100, dtype=np.uint8).reshape(10, 10)


# --- cropRect ------------------------------------------------------------------------


def test_cropRect_inside_matches_slicing(img):
    np.testing.assert_array_equal(cv.cropRect(img, (2, 3, 4, 5)), img[3:8, 2:6])


def test_cropRect_returns_a_copy(img):
    crop = cv.cropRect(img, (0, 0, 2, 2))
    crop[:] = 0
    assert img[0, 1] == 1


def test_cropRect_negative_coordinates_are_clipped(img):
    # Plain slicing img[-2:3, -2:3] would return an empty array here.
    np.testing.assert_array_equal(cv.cropRect(img, (-2, -2, 5, 5)), img[0:3, 0:3])


def test_cropRect_fill_keeps_the_box_size(img):
    crop = cv.cropRect(img, (-2, 8, 4, 4), fill=255)
    assert crop.shape == (4, 4)
    np.testing.assert_array_equal(crop[:2, 2:], img[8:10, 0:2])
    assert (crop[:, :2] == 255).all()
    assert (crop[2:, :] == 255).all()


def test_cropRect_fill_color_image():
    bgr = np.zeros((5, 5, 3), np.uint8)
    crop = cv.cropRect(bgr, (3, 3, 4, 4), fill=(1, 2, 3))
    assert crop.shape == (4, 4, 3)
    assert tuple(crop[3, 3]) == (1, 2, 3)
    assert tuple(crop[0, 0]) == (0, 0, 0)


def test_cropRect_margin(img):
    np.testing.assert_array_equal(cv.cropRect(img, (4, 4, 2, 2), margin=1), img[3:7, 3:7])


def test_cropRect_outside_raises(img):
    # Before: an empty array (or a box full of fill), which failed later with an unrelated error.
    with pytest.raises(ValueError, match="outside"):
        cv.cropRect(img, (20, 20, 3, 3))
    with pytest.raises(ValueError, match="outside"):
        cv.cropRect(img, (20, 20, 3, 3), fill=7)
    with pytest.raises(ValueError, match="outside"):
        cv.cropRect(img, (-5, 0, 5, 5))  # touches the border without covering a pixel


def test_cropRect_validation(img):
    with pytest.raises(ValueError, match="rect"):
        cv.cropRect(img, (0, 0, 0, 5))
    with pytest.raises(ValueError, match="rect"):
        cv.cropRect(img, (0, 0, 4, 4), margin=-2)
    with pytest.raises(ValueError, match="rect"):
        cv.cropRect(img, (0, 0, float("nan"), 4))
    with pytest.raises(ValueError, match="empty"):
        cv.cropRect(np.zeros((0, 0), np.uint8), (0, 0, 4, 4))


# --- inRangeEx ---------------------------------------------------------------------


@pytest.fixture
def hues():
    return np.array([[[175, 200, 200], [5, 200, 200], [90, 200, 200], [178, 50, 200]]], np.uint8)


def test_inRangeEx_wraps_red(hues):
    assert cv.inRange(hues, (170, 100, 100), (10, 255, 255)).ravel().tolist() == [0, 0, 0, 0]  # the pitfall
    assert cv.inRangeEx(hues, (170, 100, 100), (10, 255, 255)).ravel().tolist() == [255, 255, 0, 0]


def test_inRangeEx_equals_inRange_without_wrap(hues):
    np.testing.assert_array_equal(
        cv.inRangeEx(hues, (80, 100, 100), (100, 255, 255)), cv.inRange(hues, (80, 100, 100), (100, 255, 255))
    )


def test_inRangeEx_float_hsv():
    hsv = np.array([[[350.0, 0.8, 0.8], [10.0, 0.8, 0.8], [180.0, 0.8, 0.8]]], np.float32)  # hue in degrees
    assert cv.inRangeEx(hsv, (340, 0.5, 0.5), (20, 1, 1)).ravel().tolist() == [255, 255, 0]


@pytest.mark.parametrize(
    ("src", "lo", "hi"),
    [(np.zeros((2, 2), np.uint8), (0, 0, 0), (1, 1, 1)), (np.zeros((2, 2, 3), np.uint8), (0, 0), (1, 1, 1))],
)
def test_inRangeEx_validation(src, lo, hi):
    with pytest.raises(ValueError):
        cv.inRangeEx(src, lo, hi)


# --- montage -----------------------------------------------------------------------


def test_montage_square_grid():
    tiles = [np.full((10, 20, 3), i * 40, np.uint8) for i in range(5)]
    grid = cv.montage(tiles, gap=2)
    assert grid.shape == (2 * 10 + 2, 3 * 20 + 2 * 2, 3)  # 5 images -> 3 columns x 2 rows
    assert tuple(grid[-1, -1]) == (0, 0, 0)  # the empty 6th cell is padding
    assert tuple(grid[0, 0]) == (0, 0, 0)
    assert tuple(grid[0, 22]) == (40, 40, 40)


def test_montage_cols_and_mixed_sizes():
    grid = cv.montage([np.zeros((5, 5), np.uint8), np.full((8, 3), 9, np.uint8)], cols=1, color=255)
    assert grid.shape == (13, 5)
    assert grid[5, 0] == 9
    assert grid[5, 4] == 255  # narrower image padded with color


def test_montage_cellSize_letterboxes():
    grid = cv.montage([np.zeros((10, 40, 3), np.uint8), np.zeros((40, 10, 3), np.uint8)], cellSize=(20, 20))
    assert grid.shape == (20, 40, 3)


def test_montage_gray_images_with_bgr_color():
    grid = cv.montage([np.zeros((2, 2), np.uint8)] * 2, gap=1, color=(255, 255, 255))
    assert grid.shape == (2, 5)
    assert grid[0, 2] == 255


def test_montage_validation():
    with pytest.raises(ValueError):
        cv.montage([])
    with pytest.raises(ValueError):
        cv.montage([np.zeros((2, 2), np.uint8)], cols=0)
