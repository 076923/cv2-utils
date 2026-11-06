"""Invalid input is rejected with a ValueError, not with an unrelated error from deep inside NumPy or OpenCV.

Policy (docs/principles.md): empty images, NaN / infinite parameters and out-of-range colors raise ``ValueError``
naming the argument. OpenCV flag values are passed through unchecked, as OpenCV reports those itself.
"""

import numpy as np
import pytest

import cv2u as cv

NAN = float("nan")
INF = float("inf")

IMG = np.zeros((40, 60, 3), np.uint8)
GRAY = np.zeros((40, 60), np.uint8)
EMPTY = np.zeros((0, 0, 3), np.uint8)


# --- NaN and infinite numeric parameters ------------------------------------------


@pytest.mark.parametrize("value", [NAN, INF, -INF])
@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("rotateEx angle", lambda v: cv.rotateEx(IMG, v)),
        ("rotateEx scale", lambda v: cv.rotateEx(IMG, 30, scale=v)),
        ("peakColors sigma", lambda v: cv.peakColors(IMG, sigma=v)),
        ("gammaCorrection gamma", lambda v: cv.gammaCorrection(IMG, gamma=v)),
        ("adjustBrightnessContrast brightness", lambda v: cv.adjustBrightnessContrast(IMG, brightness=v)),
        ("adjustBrightnessContrast contrast", lambda v: cv.adjustBrightnessContrast(IMG, contrast=v)),
        ("adjustHSV hue", lambda v: cv.adjustHSV(IMG, hue=v)),
        ("adjustHSV saturation", lambda v: cv.adjustHSV(IMG, saturation=v)),
        ("adjustHSV value", lambda v: cv.adjustHSV(IMG, value=v)),
        ("autoContrast clip", lambda v: cv.autoContrast(IMG, clip=v)),
        ("cropToContent tolerance", lambda v: cv.cropToContent(IMG, tolerance=v)),
        ("cropToContent background", lambda v: cv.cropToContent(IMG, background=v)),
        ("diffMask threshold", lambda v: cv.diffMask(IMG, IMG, threshold=v)),
        ("ssim dataRange", lambda v: cv.ssim(IMG, IMG, dataRange=v)),
        ("equalizeHistEx clipLimit", lambda v: cv.equalizeHistEx(IMG, clahe=True, clipLimit=v)),
        ("unsharpMask sigma", lambda v: cv.unsharpMask(IMG, sigma=v)),
        ("unsharpMask amount", lambda v: cv.unsharpMask(IMG, amount=v)),
        ("findContoursEx minArea", lambda v: cv.findContoursEx(GRAY, minArea=v)),
        ("cutout feather", lambda v: cv.cutout(IMG, GRAY, feather=v)),
        ("letterbox color", lambda v: cv.letterbox(IMG, (80, 80), color=v)),
        ("hconcatEx color", lambda v: cv.hconcatEx([IMG, GRAY], color=v)),
        ("vconcatEx color", lambda v: cv.vconcatEx([IMG, GRAY], color=v)),
        ("montage color", lambda v: cv.montage([IMG, IMG], color=v, gap=2)),
        ("padToSize color", lambda v: cv.padToSize(IMG, (80, 80), color=v)),
        ("padToMultiple color", lambda v: cv.padToMultiple(IMG, 32, color=v)),
        ("toBGRA alpha", lambda v: cv.toBGRA(IMG, alpha=v)),
        ("writeVideo fps", lambda v: cv.writeVideo("x.avi", [IMG], fps=v)),
        ("fourPointTransform pts", lambda v: cv.fourPointTransform(IMG, [[0, 0], [10, 0], [10, 10], [0, v]])),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_non_finite_numbers_raise_value_error(name, call, value):
    with pytest.raises(ValueError):
        call(value)


# --- zero and negative values that would silently produce garbage -----------------


@pytest.mark.parametrize("scale", [0, -1, -0.5])
def test_rotateEx_rejects_non_positive_scale(scale):
    # Before: an output canvas of size 0 or negative, which warpAffine quietly turned into an unscaled image.
    with pytest.raises(ValueError, match="scale"):
        cv.rotateEx(IMG, 30, scale=scale)


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("fillHoles maxHoleArea", lambda: cv.fillHoles(GRAY, maxHoleArea=-1)),
        ("removeSmallObjects minArea", lambda: cv.removeSmallObjects(GRAY, minArea=-1)),
        ("diffMask minArea", lambda: cv.diffMask(IMG, IMG, minArea=-1)),
        ("diffMask threshold", lambda: cv.diffMask(IMG, IMG, threshold=-1)),
        ("ssim dataRange", lambda: cv.ssim(IMG, IMG, dataRange=0)),
        ("equalizeHistEx clipLimit", lambda: cv.equalizeHistEx(IMG, clahe=True, clipLimit=0)),
        ("cutout feather", lambda: cv.cutout(IMG, GRAY, feather=-1)),
        ("urlread maxBytes", lambda: cv.urlread("file:///x.png", maxBytes=0)),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_negative_or_zero_parameters_raise_value_error(name, call):
    with pytest.raises(ValueError):
        call()


# --- colors outside the dtype range ----------------------------------------------


@pytest.mark.parametrize("color", [-1, 256, (0, 0, 300)])
@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("hconcatEx", lambda c: cv.hconcatEx([IMG, GRAY], color=c)),
        ("letterbox", lambda c: cv.letterbox(IMG, (80, 80), color=c)),
        ("padToSize", lambda c: cv.padToSize(IMG, (80, 80), color=c)),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_color_outside_uint8_range_raises_value_error(name, call, color):
    # Before: numpy's "Python integer -1 out of bounds for uint8" OverflowError, which names no argument.
    with pytest.raises(ValueError, match="color"):
        call(color)


def test_toBGRA_alpha_outside_uint8_range_raises_value_error():
    with pytest.raises(ValueError, match="alpha"):
        cv.toBGRA(IMG, alpha=300)


def test_float_images_accept_any_finite_color():
    out = cv.padToSize(IMG.astype(np.float32), (80, 80), color=-1.5)
    assert out[0, 0, 0] == -1.5


# --- empty images ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("adjustBrightnessContrast", lambda: cv.adjustBrightnessContrast(EMPTY)),
        ("adjustHSV", lambda: cv.adjustHSV(EMPTY)),
        ("autoContrast", lambda: cv.autoContrast(EMPTY)),
        ("whiteBalance", lambda: cv.whiteBalance(EMPTY)),
        ("matchHistograms", lambda: cv.matchHistograms(EMPTY, IMG)),
        ("colorTransfer", lambda: cv.colorTransfer(IMG, EMPTY)),
        ("cropToContent", lambda: cv.cropToContent(EMPTY)),
        ("fillHoles", lambda: cv.fillHoles(EMPTY)),
        ("removeSmallObjects", lambda: cv.removeSmallObjects(EMPTY, 5)),
        ("largestComponent", lambda: cv.largestComponent(EMPTY)),
        ("findContoursEx", lambda: cv.findContoursEx(EMPTY)),
        ("skeletonize", lambda: cv.skeletonize(EMPTY)),
        ("cutout", lambda: cv.cutout(EMPTY, np.zeros((0, 0), np.uint8))),
        ("letterbox", lambda: cv.letterbox(EMPTY, (10, 10))),
        ("resizeEx", lambda: cv.resizeEx(EMPTY, width=10)),
        ("equalizeHistEx", lambda: cv.equalizeHistEx(EMPTY)),
        ("unsharpMask", lambda: cv.unsharpMask(EMPTY)),
        ("pixelate", lambda: cv.pixelate(EMPTY)),
        ("ssim", lambda: cv.ssim(EMPTY, EMPTY)),
        ("diffMask", lambda: cv.diffMask(EMPTY, EMPTY)),
        ("toGray", lambda: cv.toGray(EMPTY)),
        ("toBGR", lambda: cv.toBGR(EMPTY)),
        ("toBGRA", lambda: cv.toBGRA(EMPTY)),
        ("peakColors", lambda: cv.peakColors(EMPTY)),
        ("hconcatEx", lambda: cv.hconcatEx([IMG, EMPTY])),
        ("overlayImage", lambda: cv.overlayImage(IMG, EMPTY)),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_empty_image_raises_value_error(name, call):
    # Before: IndexError, ZeroDivisionError, TypeError or an OpenCV C++ assertion, depending on the function.
    with pytest.raises(ValueError, match="empty"):
        call()


def test_mask_helpers_reject_1d_and_4d_arrays():
    for bad in (np.zeros(10, np.uint8), np.zeros((2, 2, 2, 2), np.uint8)):
        with pytest.raises(ValueError, match="2-D"):
            cv.fillHoles(bad)


def test_mask_helpers_accept_boolean_color_masks():
    mask = np.zeros((10, 10, 3), bool)
    mask[2:5, 2:5, 1] = True
    assert cv.largestComponent(mask).sum() == 9 * 255


# --- degenerate geometry -----------------------------------------------------------


@pytest.mark.parametrize(
    "pts",
    [
        [[0, 0], [10, 0], [20, 0], [30, 0]],  # collinear
        [[0, 0], [0, 0], [10, 10], [0, 10]],  # duplicate point
        [[0, 0], [10, 0], [10, 0.0001], [0, 0.0001]],  # a sliver with (almost) no area
    ],
)
def test_fourPointTransform_rejects_degenerate_quads(pts):
    # Before: cv.getPerspectiveTransform produced a singular matrix and a garbage image, even with an explicit size.
    with pytest.raises(ValueError, match="degenerate"):
        cv.fourPointTransform(np.zeros((40, 40), np.uint8), pts, size=(10, 10))


def test_fourPointTransform_accepts_a_thin_but_real_quad():
    assert cv.fourPointTransform(
        np.zeros((40, 40), np.uint8), [[0, 0], [10, 0], [10, 1], [0, 1]], size=(10, 10)
    ).shape == (
        10,
        10,
    )


def test_fourPointTransform_rejects_bad_size():
    with pytest.raises(ValueError, match="size"):
        cv.fourPointTransform(IMG, [[0, 0], [10, 0], [10, 10], [0, 10]], size=(0, 10))


# --- writeVideo frame channels -----------------------------------------------------


def test_writeVideo_rejects_frames_with_a_different_channel_count(tmp_path):
    # Before: the writer silently dropped the mismatched frame but the frame count still said it was written.
    frames = [np.zeros((48, 64, 3), np.uint8), np.zeros((48, 64), np.uint8)]
    with pytest.raises(ValueError, match="channel"):
        cv.writeVideo(tmp_path / "a.avi", frames, fps=10)


@pytest.mark.parametrize("shape", [(48, 64, 4), (48, 64, 2), (48, 64, 3, 1), (48,)])
def test_writeVideo_rejects_frames_that_are_not_gray_or_bgr(tmp_path, shape):
    with pytest.raises(ValueError, match="grayscale"):
        cv.writeVideo(tmp_path / "a.avi", [np.zeros(shape, np.uint8)], fps=10)


# --- urlread size limit ------------------------------------------------------------


def test_urlread_maxBytes_accepts_a_file_that_fits(tmp_path, bgr=IMG):
    path = tmp_path / "img.png"
    assert cv.imwrite(str(path), bgr)
    np.testing.assert_array_equal(cv.urlread(path.as_uri(), maxBytes=path.stat().st_size), bgr)


def test_urlread_maxBytes_rejects_a_larger_file(tmp_path, bgr=IMG):
    path = tmp_path / "img.png"
    assert cv.imwrite(str(path), bgr)
    with pytest.raises(ValueError, match="maxBytes"):
        cv.urlread(path.as_uri(), maxBytes=path.stat().st_size - 1)


def test_urlread_maxBytes_uses_content_length_before_downloading(monkeypatch):
    import io
    import sys
    from email.message import Message

    urlread_module = sys.modules["cv2u.imgcodecs.urlread"]

    class Response(io.BytesIO):
        headers = Message()
        headers["Content-Length"] = "1000000"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def read(self, n=-1):
            raise AssertionError("the body must not be read when Content-Length already exceeds maxBytes")

    monkeypatch.setattr(urlread_module, "urlopen", lambda request, timeout=None: Response())
    with pytest.raises(ValueError, match="maxBytes"):
        cv.urlread("https://example.invalid/big.png", maxBytes=1000)
