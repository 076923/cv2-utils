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


@pytest.mark.parametrize("src", [np.zeros((2, 2, 3), np.float32), np.zeros((2, 2, 5), np.uint8)])
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


# Every mode Pillow can create, and what an OpenCV user expects back.
_ALL_MODES = {
    "1": ((2, 3), np.uint8),
    "L": ((2, 3), np.uint8),
    "LA": ((2, 3, 4), np.uint8),
    "La": ((2, 3, 4), np.uint8),
    "P": ((2, 3, 3), np.uint8),
    "PA": ((2, 3, 4), np.uint8),
    "RGB": ((2, 3, 3), np.uint8),
    "RGBA": ((2, 3, 4), np.uint8),
    "RGBa": ((2, 3, 4), np.uint8),
    "RGBX": ((2, 3, 3), np.uint8),
    "CMYK": ((2, 3, 3), np.uint8),
    "YCbCr": ((2, 3, 3), np.uint8),
    "LAB": ((2, 3, 3), np.uint8),
    "HSV": ((2, 3, 3), np.uint8),
    "I": ((2, 3), np.int32),
    "I;16": ((2, 3), np.uint16),
    "I;16B": ((2, 3), np.uint16),
    "I;16L": ((2, 3), np.uint16),
    "I;16N": ((2, 3), np.uint16),
    "F": ((2, 3), np.float32),
}


@pytest.mark.parametrize("mode", sorted(_ALL_MODES))
def test_fromPillow_accepts_every_mode(mode):
    shape, dtype = _ALL_MODES[mode]
    out = cv.fromPillow(Image.new(mode, (3, 2)))
    assert out.shape == shape
    assert out.dtype == dtype


@pytest.mark.parametrize("mode", ["I;16B", "I;16L", "I;16N"])
def test_fromPillow_16bit_byte_orders_keep_values(mode):
    pil = Image.new(mode, (2, 1))
    pil.putpixel((0, 0), 1000)
    pil.putpixel((1, 0), 65535)
    out = cv.fromPillow(pil)
    assert out.dtype == np.uint16
    assert out.dtype.isnative  # not a big-endian view that OpenCV would reject
    assert out.tolist() == [[1000, 65535]]


def test_fromPillow_premultiplied_alpha_keeps_alpha():
    rgba = Image.new("RGBA", (1, 1), (200, 100, 0, 128)).convert("RGBa")
    out = cv.fromPillow(rgba)
    assert out.shape == (1, 1, 4)
    assert out[0, 0, 3] == 128
    la = Image.new("LA", (1, 1), (100, 64)).convert("La")
    assert cv.fromPillow(la).shape == (1, 1, 4)
    assert cv.fromPillow(la)[0, 0, 3] == 64


@pytest.mark.parametrize(
    ("src", "mode"),
    [
        (np.arange(6, dtype=np.uint8).reshape(2, 3), "L"),
        (np.arange(6, dtype=np.uint16).reshape(2, 3) * 10000, "I;16"),
        (np.arange(6, dtype=np.int32).reshape(2, 3) * -100000, "I"),
        (np.linspace(-1.5, 2.5, 6, dtype=np.float32).reshape(2, 3), "F"),
        (np.array([[True, False, True], [False, True, False]]), "1"),
        (np.arange(12, dtype=np.uint8).reshape(2, 3, 2), "LA"),
    ],
)
def test_toPillow_single_channel_dtypes_roundtrip(src, mode):
    pil = cv.toPillow(src)
    assert pil.mode == mode
    back = cv.fromPillow(pil)
    if mode == "1":
        np.testing.assert_array_equal(back, src.astype(np.uint8) * 255)
    elif mode == "LA":
        np.testing.assert_array_equal(back[..., 0], src[..., 0])  # gray
        np.testing.assert_array_equal(back[..., 3], src[..., 1])  # alpha
    else:
        assert back.dtype == src.dtype
        np.testing.assert_array_equal(back, src)


def test_toPillow_accepts_hxwx1_and_non_contiguous_views():
    src = np.random.default_rng(0).integers(0, 256, (6, 8, 3), np.uint8)
    view = src[::2, ::2]  # non-contiguous
    assert cv.toPillow(view).size == (4, 3)
    np.testing.assert_array_equal(cv.fromPillow(cv.toPillow(view)), view)
    gray = src[..., :1]  # (H, W, 1)
    assert cv.toPillow(gray).mode == "L"


@pytest.mark.parametrize(
    "src",
    [
        np.zeros((2, 2), np.float64),
        np.zeros((2, 2), np.int16),
        np.zeros((2, 2, 3), np.uint16),
        np.zeros((2, 2, 5), np.uint8),
        np.zeros((0, 4, 3), np.uint8),
        np.zeros((2, 2, 3, 1), np.uint8),
    ],
)
def test_toPillow_rejects_unsupported_with_value_error(src):
    with pytest.raises(ValueError):
        cv.toPillow(src)


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
