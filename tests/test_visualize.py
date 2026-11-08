"""Tests for cv2u.imgproc.visualize."""

import cv2
import numpy as np
import pytest

from cv2u.imgproc import visualize
from cv2u.imgproc.visualize import (
    COCO_SKELETON,
    applyColorMapEx,
    blendImages,
    colorBar,
    compareImages,
    drawArrow,
    drawCrosshair,
    drawFPS,
    drawLegend,
    drawPolygon,
    drawProgressBar,
    drawRotatedBoxes,
    drawRotatedRect,
    drawSkeleton,
    drawTextBlock,
    drawTrajectory,
    heatmap,
    labeledMontage,
)


def bgr(h: int = 60, w: int = 80, value: int = 0) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def gray(h: int = 60, w: int = 80) -> np.ndarray:
    return np.zeros((h, w), np.uint8)


def test_no_public_name_shadows_opencv() -> None:
    for name in visualize.__all__:
        assert not hasattr(cv2, name), name


# --------------------------------------------------------------------------------------------- color maps


def test_applyColorMapEx_float_and_uint8() -> None:
    values = np.linspace(0, 1, 40 * 50, dtype=np.float32).reshape(40, 50)
    out = applyColorMapEx(values)
    assert out.shape == (40, 50, 3) and out.dtype == np.uint8
    raw = np.rint(values * 255).astype(np.uint8)
    assert np.array_equal(out, applyColorMapEx(raw, normalize=False))
    assert np.array_equal(out, applyColorMapEx(values, normalize=False))  # [0, 1] floats scaled by 255
    assert applyColorMapEx(values[..., None]).shape == (40, 50, 3)


def test_applyColorMapEx_constant_lut_and_mask() -> None:
    flat = np.full((10, 10), 3.0)
    out = applyColorMapEx(flat)  # constant map: no division by zero
    assert out.shape == (10, 10, 3)
    lut = np.zeros((256, 3), np.uint8)
    lut[:, 2] = np.arange(256)
    out = applyColorMapEx(np.arange(256, dtype=np.uint8).reshape(16, 16), lut, normalize=False)
    assert out[0, 0].tolist() == [0, 0, 0] and out[15, 15].tolist() == [0, 0, 255]
    mask = np.zeros((16, 16), np.uint8)
    mask[:8] = 1
    masked = applyColorMapEx(np.arange(256, dtype=np.uint8).reshape(16, 16), lut, normalize=False, mask=mask)
    assert masked[8:].max() == 0 and masked[:8].max() > 0


def test_applyColorMapEx_errors() -> None:
    with pytest.raises(ValueError, match="values"):
        applyColorMapEx(np.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="values"):
        applyColorMapEx(np.zeros((0, 4)))
    with pytest.raises(ValueError, match="values"):
        applyColorMapEx(np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError, match="mask"):
        applyColorMapEx(np.zeros((4, 4)), mask=np.zeros((2, 2)))


def test_heatmap_resizes_and_keeps_alpha() -> None:
    src = bgr(40, 60, 100)
    values = np.random.default_rng(0).random((8, 8))
    out = heatmap(src, values)
    assert out.shape == src.shape and out.dtype == np.uint8
    assert not np.array_equal(out, src) and np.array_equal(src, bgr(40, 60, 100))
    assert np.array_equal(heatmap(src, values, alpha=0.0), src)
    assert heatmap(gray(40, 60), values).shape == (40, 60, 3)
    bgra = np.dstack([src, np.full((40, 60), 7, np.uint8)])
    out4 = heatmap(bgra, values)
    assert out4.shape == (40, 60, 4) and np.all(out4[..., 3] == 7)


def test_heatmap_errors() -> None:
    with pytest.raises(ValueError, match="alpha"):
        heatmap(bgr(), np.zeros((4, 4)), alpha=1.5)
    with pytest.raises(ValueError, match="src"):
        heatmap(bgr().astype(np.float32), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="src"):
        heatmap(np.zeros((0, 4, 3), np.uint8), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="channels"):
        heatmap(np.zeros((4, 4, 2), np.uint8), np.zeros((4, 4)))


def test_colorBar() -> None:
    bar = colorBar(size=(100, 10))
    assert bar.shape == (10, 100, 3)
    assert not np.array_equal(bar[:, 0], bar[:, -1])
    vbar = colorBar(size=(10, 100), vertical=True)
    assert vbar.shape == (100, 10, 3) and np.array_equal(vbar[0, 0], bar[0, -1])
    labeled = colorBar(size=(100, 10), minVal=0, maxVal=1.5)
    assert labeled.shape[1] == 100 and labeled.shape[0] > 10
    vlabeled = colorBar(size=(10, 100), vertical=True, minVal=-1, maxVal=1)
    assert vlabeled.shape[0] == 100 and vlabeled.shape[1] > 10
    with pytest.raises(ValueError, match="size"):
        colorBar(size=(0, 10))
    with pytest.raises(ValueError, match="minVal"):
        colorBar(minVal=float("nan"))


# --------------------------------------------------------------------------------------------- shapes


def test_drawArrow_heads_and_in_place() -> None:
    img = bgr()
    assert drawArrow(img, (10, 30), (70, 30), (0, 255, 0)) is img
    assert img[30, 40].tolist() == [0, 255, 0]
    assert img[25, 60:64, 1].max() > 0 and img[35, 60:64, 1].max() > 0  # both wings of the head at the end
    both = drawArrow(bgr(), (10.4, 30.2), (70, 30), (0, 255, 0), heads="both")
    assert both[25, 16:20, 1].max() > 0
    none = drawArrow(bgr(), (10, 30), (70, 30), (0, 255, 0), heads="none")
    assert none[20:27, :, 1].max() == 0
    assert drawArrow(bgr(), (10, 30), (10, 30), (0, 255, 0)).max() > 0  # zero length: a dot, no error
    g = drawArrow(gray(), (10, 30), (70, 30), (0, 255, 0), heads="start")
    assert g[30, 40] > 0
    with pytest.raises(ValueError, match="heads"):
        drawArrow(bgr(), (0, 0), (10, 10), (0, 255, 0), heads="left")
    with pytest.raises(ValueError, match="headSize"):
        drawArrow(bgr(), (0, 0), (10, 10), (0, 255, 0), headSize=-1)
    with pytest.raises(ValueError, match="pt1"):
        drawArrow(bgr(), (float("nan"), 0), (10, 10), (0, 255, 0))
    with pytest.raises(ValueError, match="color"):
        drawArrow(bgr(), (0, 0), (10, 10), (0, 300, 0))


def test_drawRotatedRect() -> None:
    img = bgr(100, 100)
    rect = ((50.0, 50.0), (40.0, 20.0), 0.0)
    assert drawRotatedRect(img, rect, (255, 0, 0)) is img
    assert img[40, 50].tolist() == [255, 0, 0] and img[50, 50].tolist() == [0, 0, 0]
    filled = drawRotatedRect(bgr(100, 100), rect, (255, 0, 0), cv2.FILLED)
    assert filled[50, 50].tolist() == [255, 0, 0]
    rotated = drawRotatedRect(bgr(100, 100), ((50, 50), (40, 20), 90), (255, 0, 0))
    assert rotated[50, 40].tolist() == [255, 0, 0]
    assert drawRotatedRect(gray(100, 100), rect, 200)[40, 50] == 200
    with pytest.raises(ValueError, match="rect size"):
        drawRotatedRect(bgr(), ((5, 5), (-1, 2), 0), 255)
    with pytest.raises(ValueError, match="rect"):
        drawRotatedRect(bgr(), ((float("inf"), 5), (1, 2), 0), 255)


def test_drawRotatedBoxes() -> None:
    img = bgr(120, 120)
    rects = [((40.0, 40.0), (30.0, 10.0), 15.0), ((80, 80), (20, 20), 45.0)]
    assert drawRotatedBoxes(img, rects, labels=["a", "b"]) is img
    assert img.max() > 0
    colored = drawRotatedBoxes(bgr(120, 120), rects, colors=(0, 0, 255))
    assert colored[..., :2].max() == 0 and colored[..., 2].max() == 255  # red only (anti-aliased)
    assert drawRotatedBoxes(bgr(120, 120), rects, colors=[(1, 2, 3), (4, 5, 6)]).max() > 0
    near_top = drawRotatedBoxes(bgr(120, 120), [((20, 5), (20, 6), 0)], labels=["top"])  # label moves inside
    assert near_top.max() > 0
    assert drawRotatedBoxes(gray(120, 120), rects, labels=["a", "b"]).max() > 0
    assert drawRotatedBoxes(bgr(), []).max() == 0
    with pytest.raises(ValueError, match="labels"):
        drawRotatedBoxes(bgr(), rects, labels=["a"])
    with pytest.raises(ValueError, match="colors"):
        drawRotatedBoxes(bgr(), rects, colors=[(1, 2, 3)])


def test_drawPolygon_outline_fill_alpha() -> None:
    img = bgr(100, 100)
    pts = np.array([[10.2, 10.0], [89.6, 10.0], [50.0, 80.0]])
    assert drawPolygon(img, pts, (0, 255, 0)) is img
    assert img[10, 50].tolist() == [0, 255, 0] and img[30, 50].tolist() == [0, 0, 0]
    filled = drawPolygon(bgr(100, 100), pts, (0, 255, 0), fill=True)
    assert filled[30, 50].tolist() == [0, 255, 0]
    half = drawPolygon(bgr(100, 100), pts, (0, 200, 0), thickness=0, fill=True, alpha=0.5)
    assert 90 <= half[30, 50, 1] <= 110 and half[95, 95].tolist() == [0, 0, 0]
    open_ = drawPolygon(bgr(100, 100), [[10, 50], [90, 50], [90, 90]], (0, 255, 0), closed=False)
    assert open_[70, 50].tolist() == [0, 0, 0]
    assert drawPolygon(bgr(), [], (0, 255, 0)).max() == 0
    assert drawPolygon(gray(100, 100), pts, (0, 255, 0), fill=True)[30, 50] > 0
    with pytest.raises(ValueError, match="alpha"):
        drawPolygon(bgr(), pts, (0, 255, 0), alpha=2)
    with pytest.raises(ValueError, match="points"):
        drawPolygon(bgr(), [[1], [2]], (0, 255, 0))


def test_drawSkeleton_coco() -> None:
    assert len(COCO_SKELETON) == 16 and max(max(p) for p in COCO_SKELETON) == 16
    rng = np.random.default_rng(1)
    kps = rng.uniform(10, 90, (17, 2))
    img = bgr(100, 100)
    assert drawSkeleton(img, kps) is img
    assert img.max() > 0
    scored = np.hstack([kps, np.ones((17, 1))])
    scored[5:, 2] = 0.1
    low = drawSkeleton(bgr(100, 100), scored, minScore=0.5)
    full = drawSkeleton(bgr(100, 100), scored)
    assert np.count_nonzero(low) < np.count_nonzero(full)
    kps0 = kps.copy()
    kps0[0] = 0  # undetected marker
    assert drawSkeleton(bgr(100, 100), kps0, radius=0)[0, 0].tolist() == [0, 0, 0]
    partial = drawSkeleton(bgr(100, 100), kps[:5], colors=(0, 0, 255))  # out-of-range pairs skipped
    assert partial[..., :2].max() == 0 and partial[..., 2].max() == 255
    assert drawSkeleton(gray(100, 100), kps, connections=[(0, 1)], colors=[(255, 255, 255)]).max() > 0
    assert drawSkeleton(bgr(100, 100), kps, connections=[]).max() > 0  # joints only, default color
    with pytest.raises(ValueError, match="radius"):
        drawSkeleton(bgr(), kps, radius=-1)
    with pytest.raises(ValueError, match="colors"):
        drawSkeleton(bgr(), kps, connections=[(0, 1), (1, 2)], colors=[(1, 1, 1)])
    with pytest.raises(ValueError, match="keypoints"):
        drawSkeleton(bgr(), [[1], [2]])


def test_drawTrajectory_fade() -> None:
    pts = [(10, 30), (30, 30), (50, 30), (70, 30)]
    img = bgr()
    assert drawTrajectory(img, pts) is img
    assert img[30, 65, 1] > img[30, 15, 1] > 0  # brighter towards the newest point
    plain = drawTrajectory(bgr(), pts, color=(0, 0, 255), fade=False)
    assert plain[30, 15].tolist() == [0, 0, 255] and plain[30, 65].tolist() == [0, 0, 255]
    last = drawTrajectory(bgr(), pts, maxPoints=2)
    assert last[30, 15].max() == 0 and last[30, 65].max() > 0
    tail = drawTrajectory(bgr(), pts, color=(0, 0, 255), endColor=(255, 0, 0))
    assert tail[30, 15, 0] > tail[30, 65, 0]
    assert drawTrajectory(bgr(), [(1, 1)]).max() == 0
    assert drawTrajectory(gray(), pts).max() > 0
    with pytest.raises(ValueError, match="maxPoints"):
        drawTrajectory(bgr(), pts, maxPoints=0)
    with pytest.raises(ValueError, match="points"):
        drawTrajectory(bgr(), [(np.nan, 1), (2, 2)])


def test_drawCrosshair() -> None:
    img = bgr(61, 61)
    assert drawCrosshair(img, (30.3, 30), (255, 255, 255), size=20, gap=4) is img
    assert img[30, 30].max() == 0 and img[30, 32].max() == 0
    assert img[30, 40].max() > 0 and img[40, 30].max() > 0 and img[30, 50].max() > 0
    assert drawCrosshair(gray(61, 61), (30, 30), (255, 255, 255))[30, 45] > 0
    with pytest.raises(ValueError, match="size"):
        drawCrosshair(bgr(), (5, 5), 255, size=0)
    with pytest.raises(ValueError, match="gap"):
        drawCrosshair(bgr(), (5, 5), 255, size=5, gap=5)


# --------------------------------------------------------------------------------------------- HUD


def test_drawTextBlock() -> None:
    img = bgr(100, 200, 200)
    assert drawTextBlock(img, ["line one", "line two"], org=(10, 10)) is img
    assert img[12, 12].max() < 200  # darkened panel
    assert img[90, 190].tolist() == [200, 200, 200]
    single = drawTextBlock(bgr(100, 200, 200), "line one\nline two")
    assert np.array_equal(single, img)
    opaque = drawTextBlock(bgr(100, 200, 200), ["x"], alpha=1.0)
    assert opaque[12, 12].tolist() == [0, 0, 0]
    plain = drawTextBlock(bgr(100, 200, 200), ["x"], background=None)
    assert plain[12, 12].tolist() == [200, 200, 200]
    assert np.array_equal(drawTextBlock(bgr(), []), bgr())
    assert drawTextBlock(gray(100, 200), ["gray"], color=255).max() > 0
    clipped = drawTextBlock(bgr(100, 200), ["far"], org=(190, 95))
    assert clipped.shape == (100, 200, 3)
    assert np.array_equal(drawTextBlock(bgr(), [""], org=(500, 500)), bgr())  # panel fully outside
    with pytest.raises(ValueError, match="alpha"):
        drawTextBlock(bgr(), ["x"], alpha=-0.1)
    with pytest.raises(ValueError, match="padding"):
        drawTextBlock(bgr(), ["x"], padding=-1)
    with pytest.raises(ValueError, match="fontScale"):
        drawTextBlock(bgr(), ["x"], fontScale=0)
    with pytest.raises(ValueError, match="lineSpacing"):
        drawTextBlock(bgr(), ["x"], lineSpacing=float("nan"))
    with pytest.raises(ValueError, match="img"):
        drawTextBlock(bgr().astype(np.float32), ["x"])


def test_drawFPS() -> None:
    img = bgr(60, 120, 50)
    assert drawFPS(img, 29.97) is img
    assert img.max() > 50
    with pytest.raises(ValueError, match="fps"):
        drawFPS(bgr(), float("inf"))


def test_drawProgressBar() -> None:
    img = bgr(40, 240)
    assert drawProgressBar(img, 0.5, (20, 10), size=(200, 12)) is img
    assert img[15, 60].tolist() == [0, 200, 0] and img[15, 180].tolist() == [60, 60, 60]
    assert img[9, 60].tolist() == [0, 0, 0]
    empty = drawProgressBar(bgr(40, 240), -1.0, (20, 10), size=(200, 12), border=(255, 255, 255))
    assert empty[15, 60].tolist() == [60, 60, 60] and empty[10, 20].tolist() == [255, 255, 255]
    full = drawProgressBar(bgr(40, 240), 3.0, (20, 10), size=(200, 12))
    assert full[15, 219].tolist() == [0, 200, 0]
    assert drawProgressBar(gray(40, 240), 0.5, (20, 10))[15, 60] > 0
    with pytest.raises(ValueError, match="progress"):
        drawProgressBar(bgr(), float("nan"), (0, 0))
    with pytest.raises(ValueError, match="size"):
        drawProgressBar(bgr(), 0.5, (0, 0), size=(10, 0))


def test_drawLegend() -> None:
    img = bgr(120, 200, 128)
    assert drawLegend(img, ["cat", "dog"], [(0, 0, 255), (0, 255, 0)]) is img
    assert img[16 + 7, 16 + 7].tolist() == [0, 0, 255]  # first swatch (org 10 + padding 6, row centered)
    assert img[110, 190].tolist() == [128, 128, 128]
    auto = drawLegend(bgr(120, 200, 128), ["a", "b", "c"])
    assert auto.max() > 128
    assert np.array_equal(drawLegend(bgr(), []), bgr())
    assert drawLegend(gray(120, 200), ["x"], [(255, 255, 255)], background=None).max() > 0
    with pytest.raises(ValueError, match="colors"):
        drawLegend(bgr(), ["a", "b"], [(1, 1, 1)])
    with pytest.raises(ValueError, match="swatch"):
        drawLegend(bgr(), ["a"], swatch=0)
    with pytest.raises(ValueError, match="padding"):
        drawLegend(bgr(), ["a"], padding=-2)
    with pytest.raises(ValueError, match="alpha"):
        drawLegend(bgr(), ["a"], alpha=3)
    with pytest.raises(ValueError, match="img"):
        drawLegend(np.zeros((0, 0, 3), np.uint8), ["a"])


# --------------------------------------------------------------------------------------------- comparisons


def test_compareImages_modes() -> None:
    a = bgr(40, 60, 0)
    b = np.full((20, 30, 3), 200, np.uint8)  # different size: resized to a
    side = compareImages(a, b)
    assert side.shape == (40, 120, 3) and side[20, 10].max() == 0 and side[20, 100].tolist() == [200] * 3
    assert compareImages(a, b, gap=4, color=(255, 0, 0))[20, 62].tolist() == [255, 0, 0]
    labeled = compareImages(a, b, labels=["before", "after"], color=(255, 255, 255))
    assert labeled.shape[0] > 40 and labeled.shape[1] == 120
    vertical = compareImages(a, b, mode="vertical", labels=["x", "y"])
    assert vertical.shape[1] == 60 and vertical.shape[0] > 80
    split = compareImages(a, b, mode="split", split=0.5, gap=2, color=(0, 0, 255))
    assert split.shape == a.shape and split[20, 10].max() == 0 and split[20, 50].tolist() == [200] * 3
    assert split[20, 30].tolist() == [0, 0, 255]
    assert compareImages(a, b, mode="split", split=0.5)[20, 30].tolist() == [200] * 3
    blend = compareImages(a, b, mode="blend", split=0.5)
    assert blend[20, 20].tolist() == [100] * 3
    diff = compareImages(a, b, mode="diff")
    assert diff.shape == a.shape and diff.max() > 0
    assert compareImages(gray(40, 60), b).shape == (40, 120, 3)
    bgra = np.dstack([a, np.full((40, 60), 255, np.uint8)])
    assert compareImages(bgra, bgra, mode="blend").shape == (40, 60, 3)


def test_compareImages_errors() -> None:
    with pytest.raises(ValueError, match="mode"):
        compareImages(bgr(), bgr(), mode="overlay")
    with pytest.raises(ValueError, match="split"):
        compareImages(bgr(), bgr(), split=1.5)
    with pytest.raises(ValueError, match="gap"):
        compareImages(bgr(), bgr(), gap=-1)
    with pytest.raises(ValueError, match="labels"):
        compareImages(bgr(), bgr(), labels=["one"])
    with pytest.raises(ValueError, match="b"):
        compareImages(bgr(), bgr().astype(np.uint16))


def test_labeledMontage() -> None:
    images = [bgr(30, 40, 10), gray(20, 50), np.full((30, 30, 4), 255, np.uint8)]
    out = labeledMontage(images, ["a", "b", "c"], cols=3, gap=2)
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8
    assert out.shape[1] == 40 + 50 + 30 + 2 * 2 and out.shape[0] > 30
    fitted = labeledMontage(images, ["a", "b", "c"], cellSize=(32, 32), cols=2)
    assert fitted.shape[1] == 32 * 2 + 4
    assert labeledMontage(images[:1], ["only"]).shape[1] == 40
    with pytest.raises(ValueError, match="images"):
        labeledMontage([], [])
    with pytest.raises(ValueError, match="labels"):
        labeledMontage(images, ["a"])
    with pytest.raises(ValueError, match="cols"):
        labeledMontage(images, ["a", "b", "c"], cols=0)
    with pytest.raises(ValueError, match="images"):
        labeledMontage([bgr().astype(np.float32)], ["a"])


def test_blendImages() -> None:
    a, b = bgr(20, 30, 0), bgr(20, 30, 200)
    assert np.all(blendImages([a, b]) == 100)
    assert np.all(blendImages([a, b], weights=[1, 3]) == 150)
    assert np.all(blendImages([a, np.full((10, 10), 200, np.uint8)]) == 100)  # gray, other size
    assert np.all(blendImages([a[..., 0], b]) == 100)  # gray target from color source
    bgra = np.dstack([a, np.full((20, 30), 255, np.uint8)])
    out = blendImages([bgra, b])
    assert out.shape == (20, 30, 4) and np.all(out[..., 3] == 255)
    assert np.array_equal(blendImages([a]), a)
    with pytest.raises(ValueError, match="images"):
        blendImages([])
    with pytest.raises(ValueError, match="images"):
        blendImages([a.astype(np.float32)])
    with pytest.raises(ValueError, match="weights"):
        blendImages([a, b], weights=[1])
    with pytest.raises(ValueError, match="weights"):
        blendImages([a, b], weights=[1, -1])
    with pytest.raises(ValueError, match="weights"):
        blendImages([a, b], weights=[0, 0])
