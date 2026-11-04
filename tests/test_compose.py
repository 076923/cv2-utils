import numpy as np
import pytest

import cv2u as cv


def test_hconcatEx_matches_hconcat_for_equal_images():
    a = np.full((10, 5, 3), 10, np.uint8)
    b = np.full((10, 7, 3), 20, np.uint8)
    np.testing.assert_array_equal(cv.hconcatEx([a, b]), cv.hconcat([a, b]))


def test_vconcatEx_matches_vconcat_for_equal_images():
    a = np.full((4, 6), 10, np.uint8)
    b = np.full((3, 6), 20, np.uint8)
    np.testing.assert_array_equal(cv.vconcatEx([a, b]), cv.vconcat([a, b]))


def test_hconcatEx_mixed_sizes_and_channels():
    gray = np.full((10, 4), 50, np.uint8)
    color = np.full((6, 3, 3), (1, 2, 3), np.uint8)
    out = cv.hconcatEx([gray, color], color=(9, 9, 9), gap=2)
    assert out.shape == (10, 4 + 2 + 3, 3)
    assert tuple(out[0, 0]) == (50, 50, 50)  # gray converted to BGR
    assert tuple(out[0, 5]) == (9, 9, 9)  # gap
    assert tuple(out[0, 6]) == (1, 2, 3)
    assert tuple(out[9, 6]) == (9, 9, 9)  # shorter image padded at the bottom


def test_vconcatEx_pads_right_and_keeps_gray():
    out = cv.vconcatEx([np.zeros((2, 5), np.uint8), np.full((3, 2), 7, np.uint8)], color=255)
    assert out.shape == (5, 5)
    assert out[2, 0] == 7
    assert out[2, 4] == 255


def test_hconcatEx_bgra_padding_is_opaque():
    bgra = np.zeros((4, 2, 4), np.uint8)
    out = cv.hconcatEx([bgra, np.zeros((2, 2, 3), np.uint8)], color=(1, 2, 3))
    assert out.shape == (4, 4, 4)
    assert tuple(out[3, 3]) == (1, 2, 3, 255)


@pytest.mark.parametrize(
    ("images", "kwargs", "match"),
    [
        ([], {}, "empty"),
        ([np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.float32)], {}, "dtype"),
        ([np.zeros((2, 2, 2), np.uint8)], {}, "channels"),
        ([np.zeros((2, 2), np.uint8)], {"gap": -1}, "gap"),
        ([np.zeros((2, 2, 3), np.uint8)], {"color": (1, 2)}, "color"),
    ],
)
def test_concat_validation(images, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cv.hconcatEx(images, **kwargs)


# --- overlayImage ------------------------------------------------------------------


@pytest.fixture
def background():
    return np.full((10, 10, 3), 100, np.uint8)


def test_overlayImage_opaque(background):
    out = cv.overlayImage(background, np.full((2, 3, 3), (1, 2, 3), np.uint8), org=(4, 5))
    assert tuple(out[5, 4]) == (1, 2, 3)
    assert tuple(out[6, 6]) == (1, 2, 3)
    assert tuple(out[7, 4]) == (100, 100, 100)
    assert (background == 100).all()  # src is not modified


def test_overlayImage_alpha_and_opacity(background):
    logo = np.zeros((1, 2, 4), np.uint8)
    logo[0, 0] = (200, 200, 200, 255)  # opaque
    logo[0, 1] = (200, 200, 200, 0)  # fully transparent
    out = cv.overlayImage(background, logo)
    assert tuple(out[0, 0]) == (200, 200, 200)
    assert tuple(out[0, 1]) == (100, 100, 100)
    half = cv.overlayImage(background, logo, opacity=0.5)
    assert tuple(half[0, 0]) == (150, 150, 150)


@pytest.mark.parametrize(("org", "expected_sum"), [((-1, -1), 1), ((9, 9), 1), ((20, 0), 0), ((-5, 0), 0)])
def test_overlayImage_clips_at_borders(background, org, expected_sum):
    out = cv.overlayImage(background, np.full((2, 2, 3), 255, np.uint8), org=org)
    assert int((out[..., 0] == 255).sum()) == expected_sum


def test_overlayImage_gray_and_bgra_targets():
    gray = cv.overlayImage(np.zeros((4, 4), np.uint8), np.full((2, 2, 3), (255, 255, 255), np.uint8))
    assert gray.shape == (4, 4)
    assert gray[0, 0] == 255

    target = np.zeros((4, 4, 4), np.uint8)  # fully transparent canvas
    patch = np.full((2, 2, 4), (10, 20, 30, 255), np.uint8)
    out = cv.overlayImage(target, patch)
    assert tuple(out[0, 0]) == (10, 20, 30, 255)
    assert tuple(out[3, 3]) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ("src", "overlay", "kwargs", "match"),
    [
        (np.zeros((4, 4), np.float32), np.zeros((2, 2), np.uint8), {}, "8-bit"),
        (np.zeros((4, 4), np.uint8), np.zeros((2, 2, 2), np.uint8), {}, "channels"),
        (np.zeros((4, 4), np.uint8), np.zeros((2, 2), np.uint8), {"opacity": 1.5}, "opacity"),
    ],
)
def test_overlayImage_validation(src, overlay, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cv.overlayImage(src, overlay, **kwargs)
