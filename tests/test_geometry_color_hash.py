import numpy as np
import pytest

import cv2u as cv

# --- transformPoints -------------------------------------------------------------


def test_transformPoints_affine_matches_opencv():
    matrix = cv.getRotationMatrix2D((50, 40), 30, 1.2)
    pts = np.array([[0, 0], [10, 5], [99, 79]], np.float64)
    expected = cv.transform(pts.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    np.testing.assert_allclose(cv.transformPoints(pts, matrix), expected)


def test_transformPoints_perspective_matches_opencv_and_keeps_shape():
    src = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    dst = np.float32([[10, 5], [90, 12], [95, 90], [3, 80]])
    matrix = cv.getPerspectiveTransform(src, dst)
    contour = np.array([[[20, 30]], [[70, 50]]], np.int32)
    out = cv.transformPoints(contour, matrix)
    assert out.shape == (2, 1, 2)
    expected = cv.perspectiveTransform(contour.astype(np.float64), matrix)
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(cv.transformPoints((20, 30), matrix), expected[0, 0])


@pytest.mark.parametrize(("points", "matrix"), [([[1, 2, 3]], np.eye(3)), ([[1, 2]], np.eye(4)), (5, np.eye(3))])
def test_transformPoints_validation(points, matrix):
    with pytest.raises(ValueError):
        cv.transformPoints(points, matrix)


# --- cropRotatedRect -------------------------------------------------------------


def test_cropRotatedRect_axis_aligned_equals_slice():
    img = np.random.default_rng(0).integers(0, 256, (60, 80, 3), np.uint8)
    rect = ((29.5, 24.5), (20, 10), 0)  # pixels x 20..39, y 20..29
    np.testing.assert_array_equal(cv.cropRotatedRect(img, rect, flags=cv.INTER_NEAREST), img[20:30, 20:40])


@pytest.mark.parametrize("angle", [15, 30, 60, -20])
def test_cropRotatedRect_extracts_the_box(angle):
    img = np.zeros((200, 200), np.uint8)
    box = cv.boxPoints(((100, 100), (80, 30), angle))
    cv.fillPoly(img, [np.round(box).astype(np.int32)], 255)
    rect = cv.minAreaRect(cv.findNonZero(img))
    crop = cv.cropRotatedRect(img, rect)
    assert crop.shape == (round(rect[1][1]), round(rect[1][0]))
    assert crop[2:-2, 2:-2].min() > 200  # the inside is all box, no background


def test_cropRotatedRect_validation():
    with pytest.raises(ValueError):
        cv.cropRotatedRect(np.zeros((10, 10), np.uint8), ((5, 5), (0.2, 4), 0))


# --- deskew ----------------------------------------------------------------------


def _page(angle: float) -> np.ndarray:
    page = np.full((400, 500, 3), 250, np.uint8)
    for row in range(60, 340, 28):
        cv.putText(page, "the quick brown fox jumps", (40, row), cv.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    return cv.rotateEx(page, angle, borderValue=(250, 250, 250))


@pytest.mark.parametrize("tilt", [-12.0, -3.0, 0.0, 4.5, 20.0])
def test_deskew_finds_the_tilt(tilt):
    dst, angle = cv.deskew(_page(tilt))
    assert angle == pytest.approx(-tilt, abs=0.3)
    assert dst.ndim == 3
    assert tuple(dst[0, 0]) == (250, 250, 250)  # new corners match the paper


def test_deskew_light_text_on_dark_and_empty_page():
    _, angle = cv.deskew(255 - _page(7))
    assert angle == pytest.approx(-7, abs=0.3)
    blank = np.full((50, 60), 200, np.uint8)
    dst, angle = cv.deskew(blank)
    assert angle == 0
    np.testing.assert_array_equal(dst, blank)


@pytest.mark.parametrize("maxAngle", [-1, 90])
def test_deskew_validation(maxAngle):
    with pytest.raises(ValueError):
        cv.deskew(_page(0), maxAngle=maxAngle)


# --- matchHistograms / colorTransfer ---------------------------------------------


def test_matchHistograms_matches_the_reference_distribution():
    rng = np.random.default_rng(1)
    src = rng.integers(0, 120, (80, 80, 3), np.uint8)
    ref = rng.integers(100, 256, (50, 70, 3), np.uint8)
    out = cv.matchHistograms(src, ref)
    assert out.shape == src.shape
    for c in range(3):
        assert abs(np.median(out[..., c]) - np.median(ref[..., c])) <= 3
        assert np.all(np.diff(out[..., c].ravel()[np.argsort(src[..., c].ravel(), kind="stable")].astype(int)) >= 0)


def test_matchHistograms_gray_keeps_alpha_and_validates():
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    np.testing.assert_array_equal(cv.matchHistograms(gray, gray), gray)
    bgra = np.dstack([np.zeros((4, 4, 3), np.uint8), np.full((4, 4), 7, np.uint8)])
    assert cv.matchHistograms(bgra, np.zeros((2, 2, 3), np.uint8))[..., 3].tolist() == [[7] * 4] * 4
    with pytest.raises(ValueError):
        cv.matchHistograms(gray, np.zeros((4, 4, 3), np.uint8))


def test_colorTransfer_moves_the_lab_statistics():
    rng = np.random.default_rng(2)
    src = rng.integers(0, 256, (60, 60, 3), np.uint8)
    ref = np.clip(rng.normal((40, 90, 200), 15, (40, 40, 3)), 0, 255).astype(np.uint8)  # orange-ish
    out = cv.colorTransfer(src, ref)
    lab = lambda img: cv.cvtColor(img.astype(np.float32) / 255, cv.COLOR_BGR2Lab).reshape(-1, 3)  # noqa: E731
    np.testing.assert_allclose(lab(out).mean(axis=0), lab(ref).mean(axis=0), atol=3)
    with pytest.raises(ValueError):
        cv.colorTransfer(src[..., 0], ref)


# --- flattenAlpha ----------------------------------------------------------------


def test_flattenAlpha_composites_on_the_background():
    bgra = np.zeros((2, 2, 4), np.uint8)
    bgra[0, 0] = (0, 0, 255, 255)  # opaque red
    bgra[0, 1] = (0, 0, 255, 0)  # transparent
    bgra[1, 0] = (0, 0, 0, 128)  # half black
    out = cv.flattenAlpha(bgra)
    assert out.shape == (2, 2, 3)
    assert tuple(out[0, 0]) == (0, 0, 255)
    assert tuple(out[0, 1]) == (255, 255, 255)
    assert tuple(out[1, 0]) == (127, 127, 127)
    assert tuple(cv.flattenAlpha(bgra, (0, 255, 0))[0, 1]) == (0, 255, 0)
    background = np.full((2, 2), 10, np.uint8)
    assert tuple(cv.flattenAlpha(bgra, background)[0, 1]) == (10, 10, 10)


def test_flattenAlpha_other_inputs():
    gray = np.full((3, 3), 9, np.uint8)
    assert cv.flattenAlpha(gray).shape == (3, 3, 3)
    floats = np.dstack([np.zeros((2, 2, 3), np.float32), np.full((2, 2), 0.25, np.float32)])
    np.testing.assert_allclose(cv.flattenAlpha(floats, 1.0), 0.75)
    with pytest.raises(ValueError):
        cv.flattenAlpha(np.zeros((2, 2, 4), np.uint8), np.zeros((3, 3, 3), np.uint8))


# --- imageHash / hashDistance ----------------------------------------------------


@pytest.mark.parametrize("method", ["difference", "average", "perceptual"])
def test_imageHash_near_duplicates_are_close(method):
    img = cv.makeGradient((120, 160, 3), (0, 0, 0), (255, 200, 100))
    cv.circle(img, (60, 50), 25, (0, 0, 255), -1)
    cv.rectangle(img, (100, 70), (150, 110), (255, 255, 0), -1)
    resized = cv.resize(img, (80, 60), interpolation=cv.INTER_AREA)
    _, jpeg = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 40])
    other = np.random.default_rng(3).integers(0, 256, (120, 160, 3), np.uint8)
    h = cv.imageHash(img, method=method)
    assert 0 <= h < 2**64
    assert cv.hashDistance(h, cv.imageHash(resized, method=method)) <= 6
    assert cv.hashDistance(h, cv.imageHash(cv.imdecode(jpeg, cv.IMREAD_COLOR), method=method)) <= 6
    assert cv.hashDistance(h, cv.imageHash(other, method=method)) >= 16


def test_imageHash_size_and_validation():
    img = np.random.default_rng(4).integers(0, 256, (40, 40), np.uint8)
    assert cv.imageHash(img, hashSize=16) < 2**256
    assert cv.hashDistance(0b1011, 0b0001) == 2
    with pytest.raises(ValueError):
        cv.imageHash(img, hashSize=1)
    with pytest.raises(ValueError):
        cv.imageHash(img, method="wavelet")
    with pytest.raises(ValueError):
        cv.hashDistance(-1, 0)


# --- boxIoU ----------------------------------------------------------------------


def test_boxIoU_values_and_shape():
    a = [(0, 0, 10, 10), (20, 20, 10, 10)]
    b = [(0, 0, 10, 10), (5, 0, 10, 10), (100, 100, 5, 5)]
    iou = cv.boxIoU(a, b)
    assert iou.shape == (2, 3)
    np.testing.assert_allclose(iou[0], [1, 50 / 150, 0])
    np.testing.assert_allclose(iou[1], [0, 0, 0])
    assert float(cv.boxIoU((0, 0, 4, 4), (2, 2, 4, 4))[0, 0]) == pytest.approx(4 / 28)
    assert cv.boxIoU((0, 0, 0, 0), (0, 0, 0, 0))[0, 0] == 0
    assert cv.boxIoU(np.zeros((0, 4)), a).shape == (0, 2)


def test_boxIoU_validation():
    with pytest.raises(ValueError):
        cv.boxIoU([(0, 0, 1)], [(0, 0, 1, 1)])


# --- drawRoundedRectangle --------------------------------------------------------


def test_drawRoundedRectangle_outline_and_fill():
    img = np.zeros((60, 80), np.uint8)
    assert cv.drawRoundedRectangle(img, (10, 10, 50, 30), 255, radius=8) is img
    assert img[10, 35] == 255  # top edge
    assert img[25, 10] == 255  # left edge
    assert img[10, 10] == 0  # the corner is rounded off
    assert img[25, 35] == 0  # outline only

    filled = np.zeros((60, 80, 3), np.uint8)
    cv.drawRoundedRectangle(filled, (10, 10, 50, 30), (0, 255, 0), cv.FILLED, cv.LINE_AA, radius=8)
    assert tuple(filled[25, 35]) == (0, 255, 0)
    assert tuple(filled[10, 10]) == (0, 0, 0)
    square = np.zeros((60, 80), np.uint8)
    cv.drawRoundedRectangle(square, (10, 10, 50, 30), 255, cv.FILLED, radius=0)
    assert np.count_nonzero(square) == 50 * 30


def test_drawRoundedRectangle_small_box_and_validation():
    img = np.zeros((20, 20), np.uint8)
    cv.drawRoundedRectangle(img, (5, 5, 3, 3), 255, cv.FILLED, radius=50)
    assert np.count_nonzero(img) > 0
    with pytest.raises(ValueError):
        cv.drawRoundedRectangle(img, (0, 0, 0, 5), 255)
    with pytest.raises(ValueError):
        cv.drawRoundedRectangle(img, (0, 0, 5, 5), 255, radius=-1)


# --- pixelate --------------------------------------------------------------------


def test_pixelate_whole_image_blocks():
    img = np.random.default_rng(5).integers(0, 256, (32, 48, 3), np.uint8)
    out = cv.pixelate(img, blockSize=16)
    assert out.shape == img.shape
    block = out[:16, :16].reshape(-1, 3)
    assert (block == block[0]).all()
    np.testing.assert_allclose(block[0], img[:16, :16].reshape(-1, 3).mean(axis=0), atol=1)


def test_pixelate_rect_only_and_clipping():
    img = np.random.default_rng(6).integers(0, 256, (40, 40), np.uint8)
    out = cv.pixelate(img, 8, rect=(30, -5, 30, 20))
    np.testing.assert_array_equal(out[:, :30], img[:, :30])
    np.testing.assert_array_equal(out[15:], img[15:])
    assert len(np.unique(out[:8, 30:38])) == 1
    np.testing.assert_array_equal(cv.pixelate(img, 8, rect=(100, 100, 5, 5)), img)
    assert cv.pixelate(img[..., None], 4).shape == (40, 40, 1)
    with pytest.raises(ValueError):
        cv.pixelate(img, 0)


# --- listImages ------------------------------------------------------------------


def test_listImages_natural_order_and_recursion(tmp_path):
    for name in ["img10.png", "img2.JPG", "img1.webp", "notes.txt", "sub/deep3.tif", "folder.png/x.txt"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    names = [p.name for p in cv.listImages(tmp_path)]
    assert names == ["img1.webp", "img2.JPG", "img10.png"]
    assert [p.name for p in cv.listImages(str(tmp_path), recursive=True)][-1] == "deep3.tif"
    with pytest.raises(NotADirectoryError):
        cv.listImages(tmp_path / "img1.webp")
