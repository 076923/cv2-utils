import numpy as np
import pytest

import cv2u as cv

# --- drawLabel ---------------------------------------------------------------------


def test_drawLabel_draws_in_place_like_rectangle():
    img = np.zeros((100, 120, 3), np.uint8)
    out = cv.drawLabel(img, (20, 40, 60, 40), "cat 0.92", (0, 255, 0))
    assert out is img  # in place, like cv.rectangle
    assert tuple(img[60, 20]) == (0, 255, 0)  # left edge of the box
    tag = img[20:39, 20:60]  # the tag above the box
    assert (tag == (0, 255, 0)).all(axis=2).mean() > 0.4
    assert (tag < 64).all(axis=2).any()  # dark text on the bright green tag


def test_drawLabel_moves_tag_inside_when_no_room_above():
    img = np.zeros((60, 80, 3), np.uint8)
    cv.drawLabel(img, (5, 0, 50, 40), "top", (255, 255, 255))
    assert img[5:15, 10:20].any()  # tag drawn inside the box, near its top


def test_drawLabel_text_color_contrasts():
    bright = cv.drawLabel(np.zeros((60, 80, 3), np.uint8), (5, 30, 50, 20), "A", (255, 255, 255))
    dark = cv.drawLabel(np.full((60, 80, 3), 255, np.uint8), (5, 30, 50, 20), "A", (40, 0, 0))
    assert (bright[5:29, 5:40] == 0).all(axis=2).any()  # black text on white
    assert (dark[5:29, 5:40] == 255).all(axis=2).any()  # white text on dark blue


def test_drawLabel_gray_image():
    img = np.zeros((60, 80), np.uint8)
    cv.drawLabel(img, (5, 30, 50, 20), "A", 200)
    assert img[40, 5] == 200


def test_drawLabel_gray_image_with_bgr_color_uses_luminance():
    img = np.zeros((60, 80), np.uint8)
    cv.drawLabel(img, (5, 30, 50, 20), "A", (0, 200, 255))  # OpenCV alone would draw with 0 (black)
    assert img[40, 5] == round(0.587 * 200 + 0.299 * 255)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_drawLabel_unicode_with_font(channels, font_path):
    font = font_path
    shape = (80, 140) if channels == 1 else (80, 140, channels)
    img = np.zeros(shape, np.uint8)
    out = cv.drawLabel(img, (10, 40, 100, 30), "고양이 cat", (0, 200, 255), fontPath=font, fontSize=18)
    assert out is img
    assert out.shape == shape
    assert img[5:40, 10:110].any()


def test_drawLabel_validation():
    with pytest.raises(ValueError):
        cv.drawLabel(np.zeros((10, 10), np.float32), (1, 1, 5, 5), "x", 1)


# --- overlayMask -------------------------------------------------------------------


def test_overlayMask_tints_only_masked_pixels():
    src = np.full((4, 4, 3), 100, np.uint8)
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    out = cv.overlayMask(src, mask, color=(0, 0, 255), alpha=0.5)
    assert tuple(out[0, 0]) == (50, 50, 178)
    assert tuple(out[1, 1]) == (100, 100, 100)
    assert (src == 100).all()  # src is not modified


def test_overlayMask_uint8_mask_gray_and_bgra():
    mask = np.zeros((3, 3), np.uint8)
    mask[1, 1] = 255
    gray = cv.overlayMask(np.zeros((3, 3), np.uint8), mask, color=(255, 255, 255), alpha=1.0)
    assert gray[1, 1] == 255
    assert gray[0, 0] == 0
    bgra = cv.overlayMask(np.zeros((3, 3, 4), np.uint8), mask, color=(10, 20, 30), alpha=1.0)
    assert tuple(bgra[1, 1]) == (10, 20, 30, 0)  # alpha channel untouched


def test_overlayMask_empty_mask_is_a_copy():
    src = np.full((3, 3, 3), 9, np.uint8)
    out = cv.overlayMask(src, np.zeros((3, 3), bool))
    np.testing.assert_array_equal(out, src)
    assert out is not src


@pytest.mark.parametrize(
    ("src", "mask", "kwargs"),
    [
        (np.zeros((3, 3), np.float32), np.zeros((3, 3)), {}),
        (np.zeros((3, 3), np.uint8), np.zeros((2, 3)), {}),
        (np.zeros((3, 3), np.uint8), np.zeros((3, 3)), {"alpha": 2}),
    ],
)
def test_overlayMask_validation(src, mask, kwargs):
    with pytest.raises(ValueError):
        cv.overlayMask(src, mask, **kwargs)
