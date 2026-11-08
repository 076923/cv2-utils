import cv2
import numpy as np
import pytest

from cv2u.imgproc.document import (
    detectPage,
    detectTextBlocks,
    detectTextLines,
    enhanceDocument,
    estimateSkew,
    niblackThreshold,
    removeBorder,
    removeLines,
    removeShadow,
    removeSpeckles,
    sauvolaThreshold,
    scanDocument,
)

ROWS = ["Hello world text", "Second line here", "Third row ok"]


def page(rows: list[str] = ROWS, spacing: int = 60) -> np.ndarray:
    img = np.full((300, 400), 255, np.uint8)
    for i, text in enumerate(rows):
        cv2.putText(img, text, (20, 60 + i * spacing), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    return img


def shaded_page() -> np.ndarray:
    gradient = np.tile(np.linspace(60, 255, 400).astype(np.uint8), (300, 1))
    return cv2.min(page(), gradient)


def table_page() -> np.ndarray:
    img = page()
    for y in range(30, 300, 60):
        cv2.line(img, (0, y), (399, y), 0, 2)
    for x in range(0, 400, 100):
        cv2.line(img, (x, 0), (x, 299), 0, 2)
    return img


def photo() -> np.ndarray:
    img = np.full((400, 500, 3), 30, np.uint8)
    pts = np.array([[100, 50], [420, 80], [400, 350], [80, 320]], np.int32)
    cv2.fillPoly(img, [pts], (230, 230, 230))
    return img


# removeShadow ---------------------------------------------------------------------------------------------------


def test_remove_shadow_flattens_background():
    clean, shaded = page(), shaded_page()
    out = removeShadow(shaded)
    assert out.shape == shaded.shape and out.dtype == np.uint8
    assert shaded[clean == 255].std() > 40
    assert out[clean == 255].std() < 5
    assert out[clean == 255].mean() > 240
    assert out[clean == 0].mean() < 100


def test_remove_shadow_color_keeps_channels():
    bgr = cv2.cvtColor(shaded_page(), cv2.COLOR_GRAY2BGR)
    out = removeShadow(bgr)
    assert out.shape == bgr.shape
    assert out[..., 0].mean() > 200


@pytest.mark.parametrize("kwargs", [{"kernelSize": 0}, {"blurSize": 4}, {"blurSize": 1}])
def test_remove_shadow_rejects_bad_kernels(kwargs):
    with pytest.raises(ValueError):
        removeShadow(page(), **kwargs)


def test_remove_shadow_rejects_empty_and_float():
    with pytest.raises(ValueError, match="src"):
        removeShadow(np.zeros((0, 10), np.uint8))
    with pytest.raises(ValueError, match="8-bit"):
        removeShadow(np.zeros((10, 10), np.float32))


# local thresholds -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [sauvolaThreshold, niblackThreshold])
def test_local_threshold_recovers_text_under_shadow(fn):
    clean = page()
    out = fn(shaded_page())
    assert out.shape == clean.shape and out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}
    assert (out == clean).mean() > 0.95
    assert (fn(shaded_page(), invert=True) == 255 - clean).mean() > 0.95


def test_local_threshold_color_input_and_flat_page():
    bgr = cv2.cvtColor(shaded_page(), cv2.COLOR_GRAY2BGR)
    assert sauvolaThreshold(bgr).shape == (300, 400)
    assert np.unique(sauvolaThreshold(np.full((50, 50), 100, np.uint8))).tolist() == [255]
    # Niblack's threshold equals the mean on flat paper, so nothing is strictly above it (as with cv.threshold).
    assert np.unique(niblackThreshold(np.full((50, 50), 100, np.uint8))).tolist() == [0]
    assert np.unique(niblackThreshold(np.full((50, 50), 100, np.uint8), k=-0.2, invert=True)).tolist() == [255]


def test_local_threshold_matches_naive_mean():
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    out = niblackThreshold(gray, windowSize=5, k=0.0)
    padded = cv2.copyMakeBorder(gray, 2, 2, 2, 2, cv2.BORDER_REFLECT_101).astype(np.float64)
    mean = cv2.blur(padded, (5, 5), borderType=cv2.BORDER_ISOLATED)[2:-2, 2:-2]
    expect = (gray > mean).astype(np.uint8) * 255
    assert (out == expect).mean() > 0.99


@pytest.mark.parametrize("kwargs", [{"windowSize": 4}, {"windowSize": 1}, {"k": float("nan")}, {"r": 0}])
def test_sauvola_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        sauvolaThreshold(page(), **kwargs)


@pytest.mark.parametrize("kwargs", [{"windowSize": 4}, {"k": float("inf")}])
def test_niblack_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        niblackThreshold(page(), **kwargs)


# removeLines -----------------------------------------------------------------------------------------------------


def test_remove_lines_on_binary_keeps_text():
    clean, table = page(), table_page()
    out = removeLines(table)
    assert out.shape == table.shape and out.dtype == np.uint8
    line_pixels = (table == 0) & (clean == 255)
    assert (out[line_pixels] == 0).mean() < 0.05
    assert (out[clean == 0] == 0).mean() > 0.8


def test_remove_lines_binary_input_stays_binary():
    binary = (table_page() > 127).astype(np.uint8) * 255
    clean = (page() > 127).astype(np.uint8) * 255
    out = removeLines(binary)
    assert set(np.unique(out)) <= {0, 255}
    assert (out[(binary == 0) & (clean == 255)] == 0).mean() < 0.05
    assert (out[clean == 0] == 0).mean() > 0.8
    inverted = removeLines(255 - binary)
    assert (inverted[(binary == 0) & (clean == 255)] == 255).mean() < 0.05


def test_remove_lines_only_one_direction():
    table = table_page()
    horizontal_only = removeLines(table, vertical=False)
    assert (horizontal_only[:, 200] == 0).sum() < (table[:, 200] == 0).sum()  # horizontal rules gone
    assert (horizontal_only[:, 100] == 0).sum() >= (table[:, 100] == 0).sum() - 30  # vertical ones remain
    assert np.array_equal(removeLines(table, horizontal=False, vertical=False), table)


def test_remove_lines_gray_and_color_inpaint():
    gray = cv2.GaussianBlur(table_page(), (3, 3), 0)
    out = removeLines(gray, minLength=0.3)
    assert out.shape == gray.shape
    assert out[150, 50:90].mean() > 200  # horizontal rule at y=150 filled with paper
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert removeLines(bgr, minLength=150, thickness=1).shape == bgr.shape
    bgra = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
    out = removeLines(bgra)
    assert out.shape == bgra.shape and (out[..., 3] == 255).all()


def test_remove_lines_no_lines_returns_copy():
    src = page()
    out = removeLines(src)
    assert np.array_equal(out, src) and out is not src


@pytest.mark.parametrize("kwargs", [{"minLength": 0}, {"minLength": float("nan")}, {"thickness": 0}])
def test_remove_lines_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        removeLines(page(), **kwargs)


# detectTextLines / detectTextBlocks ------------------------------------------------------------------------------


def test_detect_text_lines_top_to_bottom():
    boxes = detectTextLines(page())
    assert boxes.shape == (3, 4) and boxes.dtype == np.int32
    assert np.all(np.diff(boxes[:, 1]) > 0)
    assert np.all(boxes[:, 3] >= 15) and np.all(boxes[:, 2] > 100)
    assert np.all(boxes[:, 0] >= 15)


def test_detect_text_lines_color_light_on_dark_and_explicit_width():
    inverted = cv2.cvtColor(255 - page(), cv2.COLOR_GRAY2BGR)
    assert detectTextLines(inverted).shape == (3, 4)
    assert detectTextLines(page(), dilateWidth=40).shape == (3, 4)
    # a tiny smear splits lines into words
    assert detectTextLines(page(), dilateWidth=1).shape[0] > 3


def test_detect_text_lines_blank_page():
    assert detectTextLines(np.full((50, 50), 255, np.uint8)).shape == (0, 4)
    assert detectTextLines(np.full((50, 50), 255, np.uint8), minHeight=100).shape == (0, 4)


def test_detect_text_blocks_merges_paragraphs():
    img = np.full((400, 400), 255, np.uint8)
    for i, y in enumerate([40, 70, 100, 250, 280, 310]):
        cv2.putText(img, ROWS[i % 3], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    assert detectTextLines(img).shape == (6, 4)
    blocks = detectTextBlocks(img)
    assert blocks.shape == (2, 4)
    assert blocks[0, 1] < 50 < blocks[0, 1] + blocks[0, 3] < 200 < blocks[1, 1]
    assert detectTextBlocks(img, dilateSize=(60, 300)).shape == (1, 4)


@pytest.mark.parametrize("kwargs", [{"minHeight": 0}, {"dilateWidth": 0}])
def test_detect_text_lines_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        detectTextLines(page(), **kwargs)


@pytest.mark.parametrize("kwargs", [{"minHeight": -1}, {"dilateSize": (0, 5)}])
def test_detect_text_blocks_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        detectTextBlocks(page(), **kwargs)


# removeBorder ----------------------------------------------------------------------------------------------------


def test_remove_border_crops_dark_frame():
    img = np.zeros((300, 400), np.uint8)
    img[20:280, 30:370] = 255
    cv2.putText(img, "text", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
    out = removeBorder(img)
    assert out.shape == (260, 340) and out.flags.c_contiguous
    assert removeBorder(img, margin=5).shape == (250, 330)
    assert removeBorder(img, margin=-10).shape == (280, 360)
    assert removeBorder(img, threshold=128).shape == (260, 340)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    assert removeBorder(bgr).shape == (260, 340, 3)


def test_remove_border_nothing_bright_or_too_much_margin():
    dark = np.zeros((30, 40), np.uint8)
    out = removeBorder(dark)
    assert out.shape == dark.shape and out is not dark
    img = np.zeros((300, 400), np.uint8)
    img[20:280, 30:370] = 255
    assert removeBorder(img, margin=200).shape == img.shape


def test_remove_border_rejects_bad_args():
    with pytest.raises(ValueError, match="threshold"):
        removeBorder(page(), threshold=float("nan"))
    with pytest.raises(ValueError, match="src"):
        removeBorder(np.zeros((0, 0), np.uint8))


# detectPage / scanDocument ---------------------------------------------------------------------------------------


def test_detect_page_finds_corners():
    corners = detectPage(photo())
    assert corners is not None and corners.shape == (4, 2) and corners.dtype == np.float32
    expect = np.array([[100, 50], [420, 80], [400, 350], [80, 320]], np.float32)
    assert np.abs(corners - expect).max() <= 3
    gray = cv2.cvtColor(photo(), cv2.COLOR_BGR2GRAY)
    corners = detectPage(gray)
    assert corners is not None and np.abs(corners - expect).max() <= 3


def test_detect_page_large_image_is_downscaled():
    big = cv2.resize(photo(), (1500, 1200), interpolation=cv2.INTER_NEAREST)
    corners = detectPage(big)
    assert corners is not None
    assert np.abs(corners - np.array([[300, 150], [1260, 240], [1200, 1050], [240, 960]])).max() <= 10


def test_detect_page_none_when_absent_or_too_small():
    assert detectPage(np.full((100, 100, 3), 30, np.uint8)) is None
    assert detectPage(photo(), minAreaRatio=0.9) is None
    circle = np.full((300, 300), 30, np.uint8)
    cv2.circle(circle, (150, 150), 120, 230, -1)
    assert detectPage(circle) is None


@pytest.mark.parametrize("ratio", [-0.1, 1.5, float("nan")])
def test_detect_page_rejects_bad_ratio(ratio):
    with pytest.raises(ValueError, match="minAreaRatio"):
        detectPage(photo(), minAreaRatio=ratio)


def test_scan_document_warps_and_enhances():
    out = scanDocument(photo())
    assert out.ndim == 3 and out.dtype == np.uint8
    assert 300 < out.shape[1] < 340 and 250 < out.shape[0] < 290
    assert out.mean() > 200
    sized = scanDocument(photo(), size=(200, 100), enhance=False)
    assert sized.shape == (100, 200, 3)
    assert scanDocument(cv2.cvtColor(photo(), cv2.COLOR_BGR2GRAY)).ndim == 2


def test_scan_document_without_page():
    src = np.full((60, 80, 3), 30, np.uint8)
    out = scanDocument(src, enhance=False)
    assert out.shape == src.shape and out is not src
    assert scanDocument(src, size=(40, 30), enhance=False).shape == (30, 40, 3)
    assert scanDocument(src).shape == src.shape


# enhanceDocument -------------------------------------------------------------------------------------------------


def test_enhance_document_modes():
    clean = page()
    gray = enhanceDocument(shaded_page())
    assert gray.shape == (300, 400) and gray.dtype == np.uint8
    assert gray[clean == 255].mean() > 230
    binary = enhanceDocument(shaded_page(), "binary")
    assert set(np.unique(binary)) <= {0, 255}
    assert (binary == clean).mean() > 0.9
    bgr = cv2.cvtColor(shaded_page(), cv2.COLOR_GRAY2BGR)
    assert enhanceDocument(bgr, "color").shape == bgr.shape
    assert enhanceDocument(bgr, "gray").shape == (300, 400)
    assert enhanceDocument(shaded_page(), "color").ndim == 2


def test_enhance_document_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode"):
        enhanceDocument(page(), "sepia")  # type: ignore[arg-type]


# estimateSkew ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tilt", [0.0, 4.0, -7.5])
def test_estimate_skew(tilt):
    img = page()
    matrix = cv2.getRotationMatrix2D((200, 150), tilt, 1)
    rotated = cv2.warpAffine(img, matrix, (400, 300), borderValue=255)
    angle = estimateSkew(rotated)
    assert isinstance(angle, float)
    assert abs(angle + tilt) <= 1.0
    assert abs(estimateSkew(cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)) + tilt) <= 1.0


def test_estimate_skew_blank_and_limits():
    assert estimateSkew(np.full((50, 50), 255, np.uint8)) == 0.0
    assert estimateSkew(page(), maxAngle=0) == 0.0
    with pytest.raises(ValueError, match="maxAngle"):
        estimateSkew(page(), maxAngle=90)
    with pytest.raises(ValueError, match="src"):
        estimateSkew(np.zeros((0, 5), np.uint8))


# removeSpeckles --------------------------------------------------------------------------------------------------


def test_remove_speckles_keeps_polarity():
    clean = page()
    noisy = clean.copy()
    rng = np.random.default_rng(1)
    ys, xs = rng.integers(0, 300, 40), rng.integers(0, 400, 40)
    noisy[ys, xs] = 0
    out = removeSpeckles(noisy, maxArea=4)
    assert out.shape == clean.shape and out.dtype == np.uint8
    assert (out == clean).mean() > 0.999
    assert (out[clean == 0] == 0).mean() > 0.95  # text survives
    inverted = removeSpeckles(255 - noisy, maxArea=4)
    assert (inverted == 255 - clean).mean() > 0.999


def test_remove_speckles_color_and_clean_input():
    bgr = cv2.cvtColor(page(), cv2.COLOR_GRAY2BGR)
    bgr[5, 5] = (0, 0, 0)
    out = removeSpeckles(bgr)
    assert out.shape == bgr.shape and tuple(out[5, 5]) == (255, 255, 255)
    clean = page()
    out = removeSpeckles(clean, maxArea=0)
    assert np.array_equal(out, clean) and out is not clean


def test_remove_speckles_rejects_bad_args():
    with pytest.raises(ValueError, match="maxArea"):
        removeSpeckles(page(), maxArea=-1)
    with pytest.raises(ValueError, match="src"):
        removeSpeckles(np.zeros((5, 5, 5, 5), np.uint8))


# names -----------------------------------------------------------------------------------------------------------


def test_names_do_not_exist_in_cv2():
    from cv2u.imgproc import document

    for name in document.__all__:
        assert not hasattr(cv2, name), name
