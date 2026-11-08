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
CONTOUR = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]], np.int32)


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
        ("translate dx", lambda v: cv.translate(IMG, v, 0)),
        ("translate dy", lambda v: cv.translate(IMG, 0, v)),
        ("imcolor color", lambda v: cv.imcolor((4, 4), v)),
        ("imcolorLike color", lambda v: cv.imcolorLike(IMG, v)),
        ("makeGradient start", lambda v: cv.makeGradient((4, 4), v, 255)),
        ("makeGradient end", lambda v: cv.makeGradient((4, 4), 0, (v, 0, 0))),
        ("cropRotatedRect rect", lambda v: cv.cropRotatedRect(IMG, ((v, 5), (4, 4), 0))),
        ("cropRect rect", lambda v: cv.cropRect(IMG, (0, 0, v, 4))),
        ("dbscanColors eps", lambda v: cv.dbscanColors(IMG, v)),
        ("clusterPoints eps", lambda v: cv.clusterPoints([[0, 0], [1, 1]], v)),
        ("meanShiftColors bandwidth", lambda v: cv.meanShiftColors(IMG, bandwidth=v)),
        ("peakColors minShare", lambda v: cv.peakColors(IMG, minShare=v)),
        ("overlayImage alpha", lambda v: cv.overlayImage(IMG, IMG, alpha=v)),
        ("overlayMask alpha", lambda v: cv.overlayMask(IMG, GRAY, alpha=v)),
        ("nms iouThreshold", lambda v: cv.nms([[0, 0, 1, 1]], [1.0], iouThreshold=v)),
        ("nms scoreThreshold", lambda v: cv.nms([[0, 0, 1, 1]], [1.0], scoreThreshold=v)),
        ("expandBoxes margin", lambda v: cv.expandBoxes([0, 0, 1, 1], margin=v)),
        ("findTemplate threshold", lambda v: cv.findTemplate(IMG, IMG[:4, :4], threshold=v)),
        ("autoCanny sigma", lambda v: cv.autoCanny(GRAY, sigma=v)),
        ("removeSmallObjects minArea", lambda v: cv.removeSmallObjects(GRAY, minArea=v)),
        ("diffMask minArea", lambda v: cv.diffMask(IMG, IMG, minArea=v)),
        ("applyMask fill", lambda v: cv.applyMask(IMG, GRAY, fill=v)),
        ("cropContour fill", lambda v: cv.cropContour(IMG, CONTOUR, fill=v)),
        ("flattenAlpha fill", lambda v: cv.flattenAlpha(np.zeros((4, 4, 4), np.uint8), fill=v)),
        ("autoContrast clip", lambda v: cv.autoContrast(IMG, clip=v)),
        ("drawLabel fontScale", lambda v: cv.drawLabel(IMG.copy(), (0, 0, 10, 10), "a", 255, fontScale=v)),
        ("drawContoursEx fontScale", lambda v: cv.drawContoursEx(IMG.copy(), [CONTOUR], fontScale=v)),
        ("deskew maxAngle", lambda v: cv.deskew(GRAY, maxAngle=v)),
        ("inRangeEx lowerb", lambda v: cv.inRangeEx(IMG, (v, 0, 0), (10, 255, 255))),
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
        ("imreadUrl maxBytes", lambda: cv.imreadUrl("file:///x.png", maxBytes=0)),
        ("kmeansColors maxIter", lambda: cv.kmeansColors(IMG, 2, maxIter=0)),
        ("dbscanColors eps", lambda: cv.dbscanColors(IMG, 0)),
        ("dbscanColors minSamples", lambda: cv.dbscanColors(IMG, 1.0, minSamples=0)),
        ("meanShiftColors bandwidth", lambda: cv.meanShiftColors(IMG, bandwidth=0)),
        ("nms iouThreshold", lambda: cv.nms([[0, 0, 1, 1]], [1.0], iouThreshold=1.5)),
        ("makeGradient size", lambda: cv.makeGradient((0, 4), 0, 255)),
        ("imcolor size", lambda: cv.imcolor((4, 0), 0)),
        ("drawHistogram size", lambda: cv.drawHistogram(IMG, size=(0, 10))),
        ("mergeTiles size", lambda: cv.mergeTiles([(IMG, (0, 0))], (0, 10))),
        ("findTemplate maxCount", lambda: cv.findTemplate(IMG, IMG[:4, :4], maxCount=-1)),
        ("findCircles minRadius", lambda: cv.findCircles(GRAY, minRadius=-1)),
        ("findCircles maxRadius", lambda: cv.findCircles(GRAY, maxRadius=-1)),
        ("cropRect rect", lambda: cv.cropRect(IMG, (0, 0, 0, 4))),
        ("drawContoursEx fontScale", lambda: cv.drawContoursEx(IMG.copy(), [CONTOUR], fontScale=0)),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_negative_or_zero_parameters_raise_value_error(name, call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: cv.findCircles(GRAY, minRadius=-1), "minRadius"),
        (lambda: cv.findCircles(GRAY, maxRadius=-1), "maxRadius"),
        (lambda: cv.filterContours([CONTOUR], minArea=float("nan")), "minArea must be a number"),
        (lambda: cv.filterContours([CONTOUR], minArea=10, maxArea=5), "minArea must not exceed maxArea"),
        (lambda: cv.filterContours([CONTOUR], minAspectRatio=2, maxAspectRatio=1), "minAspectRatio"),
        (lambda: cv.cropRect(IMG, (0, 0, 0, 4)), "rect"),
        (lambda: cv.cropRect(IMG, (100, 100, 4, 4)), "rect .* outside"),
    ],
)
def test_messages_name_the_argument(call, match):
    with pytest.raises(ValueError, match=match):
        call()


# --- colors outside the dtype range ----------------------------------------------


@pytest.mark.parametrize("color", [-1, 256, (0, 0, 300)])
@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("hconcatEx", lambda c: cv.hconcatEx([IMG, GRAY], color=c)),
        ("letterbox", lambda c: cv.letterbox(IMG, (80, 80), color=c)),
        ("padToSize", lambda c: cv.padToSize(IMG, (80, 80), color=c)),
        ("imcolor", lambda c: cv.imcolor((4, 4), c)),
        ("imcolorLike", lambda c: cv.imcolorLike(IMG, c)),
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
        ("rotateEx", lambda: cv.rotateEx(EMPTY, 30)),
        ("fourPointTransform", lambda: cv.fourPointTransform(EMPTY, [[0, 0], [10, 0], [10, 10], [0, 10]])),
        ("padToMultiple", lambda: cv.padToMultiple(EMPTY, 32)),
        ("padToSize", lambda: cv.padToSize(EMPTY, (10, 10))),
        ("splitTiles", lambda: cv.splitTiles(EMPTY, (10, 10))),
        ("mapTiles", lambda: cv.mapTiles(EMPTY, (10, 10), lambda t: t)),
        ("gammaCorrection", lambda: cv.gammaCorrection(EMPTY, 2.2)),
        ("boldEdges", lambda: cv.boldEdges(EMPTY)),
        ("drawHistogram", lambda: cv.drawHistogram(EMPTY)),
        ("drawLabel", lambda: cv.drawLabel(EMPTY, (0, 0, 1, 1), "a", 255)),
        ("drawBoxes", lambda: cv.drawBoxes(EMPTY, [[0, 0, 1, 1]])),
        ("drawPoints", lambda: cv.drawPoints(EMPTY, [[0, 0]])),
        ("drawGrid", lambda: cv.drawGrid(EMPTY, 10)),
        ("drawDashedLine", lambda: cv.drawDashedLine(EMPTY, (0, 0), (5, 5), 255)),
        ("drawDashedRectangle", lambda: cv.drawDashedRectangle(EMPTY, (0, 0, 5, 5), 255)),
        ("drawRoundedRectangle", lambda: cv.drawRoundedRectangle(EMPTY, (0, 0, 5, 5), 255)),
        ("drawContoursEx", lambda: cv.drawContoursEx(EMPTY, [CONTOUR])),
        ("overlayMask", lambda: cv.overlayMask(EMPTY, np.zeros((0, 0), np.uint8))),
        ("findTemplate src", lambda: cv.findTemplate(EMPTY, IMG)),
        ("findTemplate templ", lambda: cv.findTemplate(IMG, EMPTY)),
        ("inRangeEx", lambda: cv.inRangeEx(EMPTY, (0, 0, 0), (10, 10, 10))),
        ("autoCanny", lambda: cv.autoCanny(EMPTY)),
        ("findLines", lambda: cv.findLines(EMPTY)),
        ("findCircles", lambda: cv.findCircles(EMPTY)),
        ("findBlobs", lambda: cv.findBlobs(EMPTY)),
        ("imshowEx", lambda: cv.imshowEx("w", EMPTY)),
        ("imshowInline", lambda: cv.imshowInline(EMPTY)),
        ("imencodeBase64", lambda: cv.imencodeBase64(EMPTY)),
        ("translate", lambda: cv.translate(EMPTY, 1, 1)),
        ("cropRotatedRect", lambda: cv.cropRotatedRect(EMPTY, ((5, 5), (4, 4), 0))),
        ("deskew", lambda: cv.deskew(EMPTY)),
        ("cropRect", lambda: cv.cropRect(EMPTY, (0, 0, 4, 4))),
        ("cropCenter", lambda: cv.cropCenter(EMPTY, 4)),
        ("cropContour", lambda: cv.cropContour(EMPTY, CONTOUR)),
        ("applyMask", lambda: cv.applyMask(EMPTY, np.zeros((0, 0), np.uint8))),
        ("kmeansColors", lambda: cv.kmeansColors(EMPTY, 1)),
        ("meanShiftColors", lambda: cv.meanShiftColors(EMPTY)),
        ("dbscanColors", lambda: cv.dbscanColors(EMPTY, 1.0)),
        ("drawClusters", lambda: cv.drawClusters(EMPTY, np.zeros((0, 0), np.int32))),
        ("matchFeatures", lambda: cv.matchFeatures(EMPTY, IMG)),
        ("alignImages", lambda: cv.alignImages(IMG, EMPTY)),
        ("sharpnessScore", lambda: cv.sharpnessScore(EMPTY)),
        ("imageHash", lambda: cv.imageHash(EMPTY)),
        ("binarize", lambda: cv.binarize(EMPTY)),
        ("morphology", lambda: cv.morphology(EMPTY, "open")),
        ("flattenAlpha", lambda: cv.flattenAlpha(EMPTY)),
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


# --- imreadUrl size limit ------------------------------------------------------------


def test_imreadUrl_maxBytes_accepts_a_file_that_fits(tmp_path, bgr=IMG):
    path = tmp_path / "img.png"
    assert cv.imwrite(str(path), bgr)
    np.testing.assert_array_equal(cv.imreadUrl(path.as_uri(), maxBytes=path.stat().st_size), bgr)


def test_imreadUrl_maxBytes_rejects_a_larger_file(tmp_path, bgr=IMG):
    path = tmp_path / "img.png"
    assert cv.imwrite(str(path), bgr)
    with pytest.raises(ValueError, match="maxBytes"):
        cv.imreadUrl(path.as_uri(), maxBytes=path.stat().st_size - 1)


def test_imreadUrl_maxBytes_uses_content_length_before_downloading(monkeypatch):
    import io
    import sys
    from email.message import Message

    urlread_module = sys.modules["cv2u.imgcodecs.url"]

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
        cv.imreadUrl("https://example.invalid/big.png", maxBytes=1000)
