"""Regression guards: a cv2u wrapper must stay close to the OpenCV call it wraps.

Absolute timings depend on the machine, so these compare each wrapper with its plain OpenCV (or NumPy) equivalent in
the same process and fail when the wrapper is much slower. A failure means validation, copies or conversions were
added on the hot path.
"""

import base64
import timeit
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import cv2u as cv

# Allowed wrapper cost: ``RATIO`` times the raw call plus ``SLACK`` seconds for argument checks and small allocations.
RATIO = 1.5
SLACK = 0.0005


def _best(func: Callable[[], Any], repeat: int = 7, number: int = 5) -> float:
    """The fastest of ``repeat`` runs, in seconds per call; the minimum is the most stable statistic."""
    return min(timeit.repeat(func, number=number, repeat=repeat)) / number


def _assert_close(name: str, wrapper: Callable[[], Any], raw: Callable[[], Any]) -> None:
    wrapper()  # warm up (lazy imports, caches)
    raw()
    w, r = _best(wrapper), _best(raw)
    limit = r * RATIO + SLACK
    assert w <= limit, (
        f"{name}: {w * 1e3:.3f} ms, but the plain OpenCV call takes {r * 1e3:.3f} ms (limit {limit * 1e3:.3f} ms)"
    )


def test_resizeEx_vs_resize(bgr):
    _assert_close(
        "resizeEx",
        lambda: cv.resizeEx(bgr, width=640),
        lambda: cv.resize(bgr, (640, 360), interpolation=cv.INTER_AREA),
    )


def test_rotateEx_vs_warpAffine(bgr):
    h, w = bgr.shape[:2]
    M = cv.getRotationMatrix2D((w / 2, h / 2), 30, 1)
    _assert_close("rotateEx", lambda: cv.rotateEx(bgr, 30), lambda: cv.warpAffine(bgr, M, (w, h)))


def test_toGray_vs_cvtColor(bgr):
    _assert_close("toGray", lambda: cv.toGray(bgr), lambda: cv.cvtColor(bgr, cv.COLOR_BGR2GRAY))


def test_binarize_vs_threshold(gray):
    _assert_close(
        "binarize",
        lambda: cv.binarize(gray, "otsu"),
        lambda: cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1],
    )


def test_morphology_vs_morphologyEx(mask):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    _assert_close(
        "morphology",
        lambda: cv.morphology(mask, "open", 5),
        lambda: cv.morphologyEx(mask, cv.MORPH_OPEN, kernel),
    )


def test_findContoursEx_vs_findContours(mask):
    _assert_close(
        "findContoursEx",
        lambda: cv.findContoursEx(mask, sort=False),
        lambda: cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0],
    )


def test_cropRect_vs_slicing(bgr):
    _assert_close("cropRect", lambda: cv.cropRect(bgr, (100, 100, 800, 600)), lambda: bgr[100:700, 100:900].copy())


def test_padToMultiple_vs_copyMakeBorder(bgr):
    _assert_close(
        "padToMultiple",
        lambda: cv.padToMultiple(bgr, 64),
        lambda: cv.copyMakeBorder(bgr, 0, 8, 0, 0, cv.BORDER_CONSTANT, value=(0, 0, 0)),
    )


def test_nms_vs_dnn_NMSBoxes(boxes):
    b, s = boxes
    _assert_close(
        "nms",
        lambda: cv.nms(b, s, iouThreshold=0.5),
        lambda: cv.dnn.NMSBoxes(b.tolist(), s.tolist(), 0.0, 0.5),
    )


def test_imencodeBase64_vs_imencode(small):
    _assert_close(
        "imencodeBase64",
        lambda: cv.imencodeBase64(small, ".png"),
        lambda: base64.b64encode(cv.imencode(".png", small)[1].tobytes()),
    )


def test_imreadEx_vs_imread(small, tmp_path):
    path = tmp_path / "image.png"
    cv.imwriteEx(path, small)
    _assert_close("imreadEx", lambda: cv.imreadEx(path), lambda: cv.imread(str(path)))


def test_toPillow_vs_cvtColor(bgr):
    from PIL import Image

    _assert_close(
        "toPillow",
        lambda: cv.toPillow(bgr),
        lambda: Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB)),
    )


def test_mergeTiles_without_overlap_is_a_copy(bgr):
    """No overlaps: the tiles are just placed, so the cost is that of copying the image."""
    tiles = cv.splitTiles(bgr, (640, 540))
    size = (bgr.shape[1], bgr.shape[0])
    copy = _best(lambda: bgr.copy())
    merged = _best(lambda: cv.mergeTiles(tiles, size))
    assert merged <= copy * 4 + SLACK, f"mergeTiles: {merged * 1e3:.2f} ms vs {copy * 1e3:.2f} ms for a copy"


def test_mergeTiles_with_overlap_is_a_few_passes(bgr):
    """Averaging the overlaps is a float accumulation over the image, not a Python loop over pixels."""
    tiles = cv.splitTiles(bgr, (512, 512), overlap=32)
    size = (bgr.shape[1], bgr.shape[0])

    def reference():  # the same arithmetic on the whole image: accumulate, divide, round, convert back
        total = bgr.astype(np.float32)
        total += bgr
        total /= 2
        return np.clip(np.rint(total), 0, 255).astype(np.uint8)

    ref = _best(reference)
    merged = _best(lambda: cv.mergeTiles(tiles, size))
    assert merged <= ref * 4 + 0.005, (
        f"mergeTiles: {merged * 1e3:.2f} ms vs {ref * 1e3:.2f} ms for the plain arithmetic"
    )


def test_mapTiles_adds_nothing_to_its_parts(bgr):
    blur = lambda img: cv.GaussianBlur(img, (5, 5), 0)  # noqa: E731
    size = (bgr.shape[1], bgr.shape[0])
    tiles = cv.splitTiles(bgr, (512, 512), overlap=16)
    parts = _best(lambda: cv.mergeTiles([(blur(t), xy) for t, xy in tiles], size))
    whole = _best(lambda: cv.mapTiles(bgr, (512, 512), blur, overlap=16))
    assert whole <= parts * RATIO + SLACK, f"mapTiles: {whole * 1e3:.2f} ms vs {parts * 1e3:.2f} ms for split + merge"


@pytest.mark.parametrize("n", [10, 1000])
def test_boxIoU_scales_linearly_in_pairs(boxes, n):
    """Doubling the boxes on one side must not more than ~triple the time (no Python-level loops)."""
    b = boxes[0][:n]
    one = _best(lambda: cv.boxIoU(b, b[: max(1, n // 2)]))
    two = _best(lambda: cv.boxIoU(b, b))
    assert two <= one * 3 + SLACK, f"boxIoU: {two * 1e3:.3f} ms vs {one * 1e3:.3f} ms for half the pairs"


def test_contourStats_is_not_quadratic(mask):
    contours = cv.findContoursEx(mask)
    assert contours, "the benchmark mask must contain shapes"
    biggest = contours[0]
    small = cv.approxContour(biggest, 0.05)
    t_big, t_small = _best(lambda: cv.contourStats(biggest)), _best(lambda: cv.contourStats(small))
    assert len(biggest) > 2 * len(small)
    assert t_big <= t_small * (len(biggest) / len(small)) * 4 + SLACK, (
        "contourStats cost grows faster than the point count"
    )


def test_validation_helpers_are_cheap():
    """The checks every function runs must cost microseconds, not milliseconds."""
    from cv2u._validate import check_color, check_finite, check_image_not_empty

    img = np.zeros((1080, 1920, 3), np.uint8)
    t = _best(
        lambda: (
            check_image_not_empty("src", img),
            check_finite("angle", 30.0),
            check_color("c", (1, 2, 3), img.dtype),
        ),
        repeat=5,
        number=1000,
    )
    assert t < 50e-6, f"validation helpers take {t * 1e6:.1f} us per call"
