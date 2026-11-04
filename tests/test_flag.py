"""The named flags must mean exactly what the magic numbers mean in OpenCV."""

import numpy as np
import pytest

import cv2u as cv


@pytest.fixture
def img():
    return np.arange(12, dtype=np.uint8).reshape(3, 4)


def test_flip_modes(img):
    np.testing.assert_array_equal(cv.flip(img, cv.FLIP_MODE_X), img[::-1])
    np.testing.assert_array_equal(cv.flip(img, cv.FLIP_MODE_Y), img[:, ::-1])
    np.testing.assert_array_equal(cv.flip(img, cv.FLIP_MODE_XY), cv.rotate(img, cv.ROTATE_180))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_ddepth_same_keeps_the_source_depth(dtype):
    src = np.ones((5, 5), dtype)
    assert cv.filter2D(src, cv.DDEPTH_SAME, np.ones((3, 3), np.float32)).dtype == dtype
    assert cv.Sobel(src, cv.DDEPTH_SAME, 1, 0).dtype == dtype
    assert cv.Laplacian(src, cv.DDEPTH_SAME).dtype == dtype


def test_all_contours_draws_every_contour():
    contours = [np.array([[[2, 2]], [[8, 2]], [[8, 8]]]), np.array([[[22, 2]], [[30, 2]], [[30, 10]]])]
    canvas = np.zeros((20, 40), np.uint8)
    first = cv.drawContours(canvas.copy(), contours, 0, 255, cv.FILLED)
    every = cv.drawContours(canvas.copy(), contours, cv.ALL_CONTOURS, 255, cv.FILLED)
    assert not first[:, 20:].any()
    assert every[:, :20].any()
    assert every[:, 20:].any()


def test_reduce_dims(img):
    row = cv.reduce(img, cv.REDUCE_TO_ROW, cv.REDUCE_SUM, dtype=cv.CV_32S)
    col = cv.reduce(img, cv.REDUCE_TO_COL, cv.REDUCE_SUM, dtype=cv.CV_32S)
    assert row.shape == (1, 4)
    assert col.shape == (3, 1)
    np.testing.assert_array_equal(row.ravel(), img.sum(axis=0))
    np.testing.assert_array_equal(col.ravel(), img.sum(axis=1))


def test_key_codes_are_ascii():
    assert cv.KEY_ESC == 0x1B
    assert ord("\r") == cv.KEY_ENTER
    assert ord(" ") == cv.KEY_SPACE
    assert ord("\t") == cv.KEY_TAB
    assert ord("\b") == cv.KEY_BACKSPACE
    assert cv.WAIT_FOREVER == 0
