"""Timings of the most used cv2u functions on a 1080p image.

These are for ``pytest benchmarks --benchmark-compare`` (see CONTRIBUTING.md); the pass/fail regression guards are in
``test_overhead.py``.
"""

import numpy as np

import cv2u as cv

# --- core -----------------------------------------------------------------------------


def test_rotateEx(benchmark, bgr):
    benchmark(cv.rotateEx, bgr, 30)


def test_cropRect(benchmark, bgr):
    benchmark(cv.cropRect, bgr, (-100, -100, 800, 600))


def test_montage(benchmark, small):
    benchmark(cv.montage, [small] * 6, cols=3)


def test_padToMultiple(benchmark, bgr):
    benchmark(cv.padToMultiple, bgr, 32)


def test_splitTiles_mergeTiles(benchmark, bgr):
    def run():
        tiles = cv.splitTiles(bgr, (512, 512), overlap=32)
        return cv.mergeTiles(tiles, (bgr.shape[1], bgr.shape[0]))

    benchmark(run)


def test_mapTiles(benchmark, bgr):
    benchmark(cv.mapTiles, bgr, (512, 512), lambda t: t, overlap=32)


def test_nms(benchmark, boxes):
    benchmark(cv.nms, *boxes, iouThreshold=0.5)


def test_boxIoU(benchmark, boxes):
    benchmark(cv.boxIoU, boxes[0][:300], boxes[0][:300])


def test_dominantColors(benchmark, small):
    benchmark(cv.dominantColors, small, 5)


# --- imgproc --------------------------------------------------------------------------


def test_resizeEx(benchmark, bgr):
    benchmark(cv.resizeEx, bgr, width=640)


def test_letterbox(benchmark, bgr):
    benchmark(cv.letterbox, bgr, (640, 640))


def test_toGray(benchmark, bgr):
    benchmark(cv.toGray, bgr)


def test_autoCanny(benchmark, gray):
    benchmark(cv.autoCanny, gray)


def test_binarize_otsu(benchmark, gray):
    benchmark(cv.binarize, gray, "otsu")


def test_binarize_adaptive(benchmark, gray):
    benchmark(cv.binarize, gray, "adaptive")


def test_morphology(benchmark, mask):
    benchmark(cv.morphology, mask, "open", 5)


def test_findContoursEx(benchmark, mask):
    benchmark(cv.findContoursEx, mask, minArea=10)


def test_contourStats(benchmark, mask):
    contours = cv.findContoursEx(mask)
    benchmark(lambda: [cv.contourStats(c) for c in contours])


def test_adjustBrightnessContrast(benchmark, bgr):
    benchmark(cv.adjustBrightnessContrast, bgr, brightness=20, contrast=1.2)


def test_equalizeHistEx(benchmark, bgr):
    benchmark(cv.equalizeHistEx, bgr)


def test_unsharpMask(benchmark, bgr):
    benchmark(cv.unsharpMask, bgr)


def test_overlayMask(benchmark, bgr, mask):
    benchmark(cv.overlayMask, bgr, mask)


def test_drawBoxes(benchmark, bgr, boxes):
    canvas = bgr.copy()
    benchmark(cv.drawBoxes, canvas, boxes[0][:100], labels=[str(i) for i in range(100)])


def test_drawLabel(benchmark, small):
    canvas = small.copy()
    benchmark(cv.drawLabel, canvas, (20, 40, 200, 100), "person 0.93", (0, 255, 0))


def test_findTemplate(benchmark, bgr):
    template = bgr[480:560, 560:640]
    benchmark(cv.findTemplate, bgr, template)


def test_imageHash(benchmark, bgr):
    benchmark(cv.imageHash, bgr)


def test_ssim(benchmark, gray):
    other = cv.GaussianBlur(gray, (5, 5), 0)
    benchmark(cv.ssim, gray, other)


def test_matchFeatures(benchmark, small):
    other = cv.rotateEx(small, 5)
    benchmark(cv.matchFeatures, small, other, maxFeatures=500)


# --- imgcodecs ------------------------------------------------------------------------


def test_imencodeBase64(benchmark, small):
    benchmark(cv.imencodeBase64, small, ".jpg")


def test_toPillow_fromPillow(benchmark, bgr):
    benchmark(lambda: cv.fromPillow(cv.toPillow(bgr)))


def test_imwriteEx_imreadEx(benchmark, small, tmp_path):
    path = tmp_path / "이미지.png"

    def run():
        cv.imwriteEx(path, small)
        return cv.imreadEx(path)

    benchmark(run)


def test_fourPointTransform(benchmark, bgr):
    pts = np.array([(100, 100), (1500, 150), (1400, 900), (200, 1000)], np.float32)
    benchmark(cv.fourPointTransform, bgr, pts)
