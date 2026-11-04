import numpy as np
import pytest

import cv2u as cv


@pytest.fixture
def img():
    return np.random.default_rng(2).integers(0, 256, (100, 200, 3), dtype=np.uint8)  # h=100, w=200


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"width": 100}, (50, 100)),
        ({"height": 50}, (50, 100)),
        ({"width": 400}, (200, 400)),
        ({"width": 30, "height": 70}, (70, 30)),  # both given: exact size, like cv.resize
        ({"width": 1}, (1, 1)),  # never collapses to 0 pixels
    ],
)
def test_resizeEx_shape(img, kwargs, expected):
    assert cv.resizeEx(img, **kwargs).shape[:2] == expected


def test_resizeEx_default_interpolation_matches_opencv_advice(img):
    np.testing.assert_array_equal(cv.resizeEx(img, width=100), cv.resize(img, (100, 50), interpolation=cv.INTER_AREA))
    np.testing.assert_array_equal(
        cv.resizeEx(img, width=400), cv.resize(img, (400, 200), interpolation=cv.INTER_LINEAR)
    )


def test_resizeEx_explicit_interpolation(img):
    np.testing.assert_array_equal(
        cv.resizeEx(img, width=100, interpolation=cv.INTER_NEAREST),
        cv.resize(img, (100, 50), interpolation=cv.INTER_NEAREST),
    )


@pytest.mark.parametrize("kwargs", [{}, {"width": 0}, {"height": -5}])
def test_resizeEx_validation(img, kwargs):
    with pytest.raises(ValueError):
        cv.resizeEx(img, **kwargs)


def test_letterbox_shape_scale_and_offset(img):
    dst, scale, (left, top) = cv.letterbox(img, (300, 300), color=(1, 2, 3))
    assert dst.shape == (300, 300, 3)
    assert scale == pytest.approx(1.5)
    assert (left, top) == (0, 75)
    assert tuple(dst[0, 0]) == (1, 2, 3)  # padding
    assert tuple(dst[-1, -1]) == (1, 2, 3)


def test_letterbox_maps_points(img):
    img = np.zeros((100, 200), np.uint8)
    img[20, 50] = 255
    dst, scale, (left, top) = cv.letterbox(img, (100, 100), interpolation=cv.INTER_NEAREST)
    x, y = round(50 * scale + left), round(20 * scale + top)
    assert dst[y, x] == 255


def test_letterbox_gray_scalar_color():
    dst, _, _ = cv.letterbox(np.zeros((10, 20), np.uint8), (20, 20), color=128)
    assert dst.shape == (20, 20)
    assert dst[0, 0] == 128


def test_letterbox_validation(img):
    with pytest.raises(ValueError):
        cv.letterbox(img, (0, 10))
