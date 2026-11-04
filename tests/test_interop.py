import numpy as np
import pytest
from PIL import Image

import cv2u as cv


@pytest.fixture
def bgr():
    img = np.zeros((4, 6, 3), np.uint8)
    img[:] = (255, 0, 0)  # pure blue in BGR
    return img


# --- Pillow ------------------------------------------------------------------------


def test_toPillow_fixes_channel_order(bgr):
    pil = cv.toPillow(bgr)
    assert pil.mode == "RGB"
    assert pil.size == (6, 4)
    assert pil.getpixel((0, 0)) == (0, 0, 255)  # still blue


def test_toPillow_bgra_and_gray():
    bgra = np.zeros((2, 2, 4), np.uint8)
    bgra[:] = (255, 0, 0, 128)
    assert cv.toPillow(bgra).getpixel((0, 0)) == (0, 0, 255, 128)
    assert cv.toPillow(np.full((2, 2), 7, np.uint8)).mode == "L"
    assert cv.toPillow(np.full((2, 2), 1000, np.uint16)).getpixel((0, 0)) == 1000


@pytest.mark.parametrize("src", [np.zeros((2, 2, 3), np.float32), np.zeros((2, 2, 2), np.uint8)])
def test_toPillow_validation(src):
    with pytest.raises(ValueError):
        cv.toPillow(src)


@pytest.mark.parametrize("img", ["bgr", "bgra", "gray"])
def test_pillow_roundtrip(img, bgr):
    src = {
        "bgr": bgr,
        "bgra": np.dstack([bgr, np.full(bgr.shape[:2], 77, np.uint8)]),
        "gray": np.arange(24, dtype=np.uint8).reshape(4, 6),
    }[img]
    np.testing.assert_array_equal(cv.fromPillow(cv.toPillow(src)), src)


def test_fromPillow_palette_gives_colors_not_indices():
    pil = Image.new("RGB", (2, 1), (255, 0, 0)).convert("P")  # red, stored as a palette index
    assert np.asarray(pil).ndim == 2  # the classic pitfall: np.asarray returns palette indices
    assert tuple(cv.fromPillow(pil)[0, 0]) == (0, 0, 255)  # red in BGR


def test_fromPillow_cmyk_is_not_mistaken_for_bgra():
    pil = Image.new("RGB", (1, 1), (0, 255, 0)).convert("CMYK")
    assert np.asarray(pil).shape[2] == 4  # looks like BGRA to np.asarray
    out = cv.fromPillow(pil)
    assert out.shape == (1, 1, 3)
    assert tuple(out[0, 0]) == (0, 255, 0)


@pytest.mark.parametrize(
    ("mode", "shape", "dtype"),
    [("1", (2, 2), np.uint8), ("LA", (2, 2, 4), np.uint8), ("YCbCr", (2, 2, 3), np.uint8), ("F", (2, 2), np.float32)],
)
def test_fromPillow_other_modes(mode, shape, dtype):
    out = cv.fromPillow(Image.new(mode, (2, 2)))
    assert out.shape == shape
    assert out.dtype == dtype


def test_fromPillow_palette_with_transparency_gives_bgra():
    pil = Image.new("RGBA", (2, 1), (0, 0, 0, 0)).convert("P")
    pil.info["transparency"] = 0
    assert cv.fromPillow(pil).shape == (1, 2, 4)


# --- base64 ------------------------------------------------------------------------


def test_base64_roundtrip(bgr):
    text = cv.imencodeBase64(bgr)
    assert isinstance(text, str)
    np.testing.assert_array_equal(cv.imdecodeBase64(text), bgr)
    np.testing.assert_array_equal(cv.imdecodeBase64(text.encode()), bgr)  # bytes work too


def test_base64_data_uri(bgr):
    uri = cv.imencodeBase64(bgr, ".jpg", (cv.IMWRITE_JPEG_QUALITY, 95), dataUri=True)
    assert uri.startswith("data:image/jpeg;base64,")
    decoded = cv.imdecodeBase64(uri)
    assert decoded is not None
    assert decoded.shape == bgr.shape


def test_imdecodeBase64_ignores_whitespace_and_flags(bgr):
    text = cv.imencodeBase64(bgr)
    wrapped = "\n".join(text[i : i + 10] for i in range(0, len(text), 10))
    gray = cv.imdecodeBase64(wrapped, cv.IMREAD_GRAYSCALE)
    assert gray is not None
    assert gray.shape == bgr.shape[:2]


def test_imdecodeBase64_not_an_image_returns_none():
    import base64

    assert cv.imdecodeBase64(base64.b64encode(b"hello").decode()) is None
    assert cv.imdecodeBase64("") is None


def test_imdecodeBase64_invalid_base64():
    with pytest.raises(ValueError, match="base64"):
        cv.imdecodeBase64("not base64 !!!")


def test_imencodeBase64_unknown_format(bgr):
    with pytest.raises(ValueError, match="encode"):
        cv.imencodeBase64(bgr, ".nope")
