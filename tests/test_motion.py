from itertools import pairwise

import cv2
import numpy as np
import pytest

from cv2u.videoio.motion import (
    MotionAccumulator,
    MultiTrackerEx,
    TrackerEx,
    backgroundMask,
    createBackgroundSubtractor,
    denseFlow,
    drawFlow,
    flowMagnitude,
    flowToColor,
    frameDifference,
    sparseFlow,
    trackPoints,
)

SIZE = 96


def _frame(x, y=30, size=20, *, gray=False, seed=0):
    """A textured square at (x, y) on a textured background."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(60, 100, (SIZE, SIZE), np.uint8)
    square = np.random.default_rng(seed + 1).integers(150, 255, (size, size), np.uint8)
    frame[y : y + size, x : x + size] = square
    return frame if gray else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def _moving(n=8, dx=4, start=10, **kw):
    return [_frame(start + i * dx, **kw) for i in range(n)]


def _has_tracker(name):
    return hasattr(cv2, f"Tracker{name}_create") or hasattr(getattr(cv2, f"Tracker{name}", None), "create")


# --- trackers ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["MIL", "KCF", "CSRT"])
def test_TrackerEx_follows_square(name):
    if not _has_tracker(name):
        pytest.skip(f"Tracker{name} is not available")
    frames = _moving()
    tracker = TrackerEx(name)
    assert tracker.lost and tracker.box is None
    tracker.init(frames[0], (10, 30, 20, 20))
    assert tracker.box == (10, 30, 20, 20) and not tracker.lost
    boxes = list(tracker.track(frames[1:]))
    assert len(boxes) == 7
    assert all(b is not None for b in boxes)
    xs = [b[0] for b in boxes]
    assert xs[-1] > xs[0] + 10
    assert all(isinstance(v, int) for v in boxes[-1])
    assert name in repr(tracker)


def test_TrackerEx_params_and_errors(monkeypatch):
    if not _has_tracker("MIL"):
        pytest.skip("TrackerMIL is not available")
    tracker = TrackerEx("MIL", samplerSearchWinSize=30)
    with pytest.raises(RuntimeError, match="init"):
        tracker.update(_frame(10))
    with pytest.raises(ValueError, match="box"):
        tracker.init(_frame(10), (1, 2, 3))
    with pytest.raises(ValueError, match="box"):
        tracker.init(_frame(10), (1, 2, 0, 3))
    with pytest.raises(ValueError, match="frame"):
        tracker.init(_frame(10).astype(np.float32), (10, 30, 20, 20))
    with pytest.raises(ValueError, match="parameter"):
        TrackerEx("MIL", nope=1)
    with pytest.raises(RuntimeError, match="available"):
        TrackerEx("NoSuchTracker")
    monkeypatch.delattr(cv2, "TrackerMIL_Params", raising=False)
    with pytest.raises(ValueError, match="no parameters"):
        TrackerEx("MIL", samplerSearchWinSize=30)


def test_TrackerEx_lost(monkeypatch):
    if not _has_tracker("MIL"):
        pytest.skip("TrackerMIL is not available")
    tracker = TrackerEx("MIL")
    tracker.init(_frame(10), (10, 30, 20, 20))

    class Lost:
        def update(self, frame):
            return False, (0, 0, 0, 0)

    monkeypatch.setattr(tracker, "_tracker", Lost())
    assert tracker.update(_frame(14)) is None
    assert tracker.lost and tracker.box is None


def test_MultiTrackerEx():
    if not _has_tracker("MIL"):
        pytest.skip("TrackerMIL is not available")
    frames = _moving(n=4)
    multi = MultiTrackerEx("MIL")
    assert len(multi) == 0
    first = multi.add(frames[0], (10, 30, 20, 20))
    second = multi.add(frames[0], (60, 60, 15, 15))
    assert (first, second) == (0, 1) and multi.ids == [0, 1] and len(multi) == 2
    assert multi.boxes == {0: (10, 30, 20, 20), 1: (60, 60, 15, 15)}
    result = multi.update(frames[1])
    assert set(result) == {0, 1}
    assert result[0] is not None and all(isinstance(v, int) for v in result[0])
    multi.remove(second)
    assert multi.ids == [0] and "targets=1" in repr(multi)
    with pytest.raises(KeyError):
        multi.remove(second)
    assert multi.add(frames[2], (20, 30, 20, 20)) == 2


# --- frame difference and background subtraction -------------------------------------------


def test_frameDifference_detects_motion():
    a, b = _frame(10), _frame(30)
    mask = frameDifference(a, b, 25)
    assert mask.dtype == np.uint8 and mask.shape == (SIZE, SIZE)
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[30:50, 10:30].mean() > 200 and mask[30:50, 30:50].mean() > 200
    assert mask[60:, :].max() == 0
    assert frameDifference(a, a).max() == 0
    gray = frameDifference(_frame(10, gray=True), _frame(30, gray=True), blur=3)
    assert gray[40, 15] == 255
    bgra = cv2.cvtColor(a, cv2.COLOR_BGR2BGRA)
    assert frameDifference(bgra, b)[40, 15] == 255


def test_frameDifference_float_input_and_errors():
    a = _frame(10, gray=True).astype(np.float32) / 255
    b = _frame(30, gray=True).astype(np.float32) / 255
    assert frameDifference(a, b)[40, 15] == 255
    with pytest.raises(ValueError, match="threshold"):
        frameDifference(a, b, 300)
    with pytest.raises(ValueError, match="threshold"):
        frameDifference(a, b, -1)
    with pytest.raises(ValueError, match="blur"):
        frameDifference(a, b, blur=4)
    with pytest.raises(ValueError, match="same size"):
        frameDifference(a, b[:50])
    with pytest.raises(ValueError, match="prev"):
        frameDifference(np.zeros((0, 4), np.uint8), b)
    with pytest.raises(ValueError, match="next"):
        frameDifference(a, np.zeros((SIZE, SIZE, 2), np.uint8))


@pytest.mark.parametrize("method", ["MOG2", "knn"])
def test_backgroundMask_finds_moving_square(method):
    subtractor = createBackgroundSubtractor(method, history=20, detectShadows=True)
    background = _frame(0, size=0)
    for _ in range(10):
        backgroundMask(subtractor, background)
    mask = backgroundMask(subtractor, _frame(40), minArea=20)
    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 255}
    assert mask[30:50, 40:60].mean() > 200
    assert mask[:20, :].max() == 0


def test_backgroundMask_shadows_and_minArea():
    class Fake:
        def apply(self, frame, learningRate=-1):
            raw = np.zeros(frame.shape[:2], np.uint8)
            raw[5:15, 5:15] = 255
            raw[20:22, 20:22] = 255  # 4-pixel speck
            raw[40:50, 40:50] = 127  # shadow
            return raw

    frame = _frame(0, size=0)
    mask = backgroundMask(Fake(), frame)
    assert mask[10, 10] == 255 and mask[21, 21] == 255 and mask[45, 45] == 0
    mask = backgroundMask(Fake(), frame, shadows=True, minArea=10)
    assert mask[10, 10] == 255 and mask[21, 21] == 0 and mask[45, 45] == 255
    assert backgroundMask(Fake(), frame, learningRate=0)[10, 10] == 255
    with pytest.raises(ValueError, match="learningRate"):
        backgroundMask(Fake(), frame, learningRate=2)
    with pytest.raises(ValueError, match="minArea"):
        backgroundMask(Fake(), frame, minArea=-1)
    with pytest.raises(ValueError, match="frame"):
        backgroundMask(Fake(), np.zeros((0, 0), np.uint8))


def test_createBackgroundSubtractor_validation():
    mog2 = createBackgroundSubtractor()
    assert mog2.getVarThreshold() == 16 and mog2.getHistory() == 500 and mog2.getDetectShadows()
    knn = createBackgroundSubtractor("KNN", threshold=100, detectShadows=False)
    assert knn.getDist2Threshold() == 100 and not knn.getDetectShadows()
    assert createBackgroundSubtractor("knn").getDist2Threshold() == 400
    with pytest.raises(ValueError, match="method"):
        createBackgroundSubtractor("GMG")
    with pytest.raises(ValueError, match="history"):
        createBackgroundSubtractor(history=0)
    with pytest.raises(ValueError, match="threshold"):
        createBackgroundSubtractor(threshold=0)


# --- optical flow --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["farneback", "DIS"])
def test_denseFlow_direction(method):
    a, b = _frame(20), _frame(25)
    flow = denseFlow(a, b, method)
    assert flow.shape == (SIZE, SIZE, 2) and flow.dtype == np.float32
    inside = flow[35:45, 25:35]
    assert inside[..., 0].mean() == pytest.approx(5, abs=1.5)
    assert abs(inside[..., 1].mean()) < 1.0


def test_denseFlow_options_and_errors():
    a, b = _frame(20, gray=True), _frame(25, gray=True)
    assert denseFlow(a, b, winsize=9, levels=2)[40, 30, 0] == pytest.approx(5, abs=2)
    assert denseFlow(a, b, "dis", preset="ultrafast").shape == (SIZE, SIZE, 2)
    with pytest.raises(ValueError, match="method"):
        denseFlow(a, b, "lucas")
    with pytest.raises(ValueError, match="unknown farneback"):
        denseFlow(a, b, window=3)
    with pytest.raises(ValueError, match="unknown dis"):
        denseFlow(a, b, "dis", levels=3)
    with pytest.raises(ValueError, match="preset"):
        denseFlow(a, b, "dis", preset="slow")


def test_denseFlow_dis_missing(monkeypatch):
    monkeypatch.delattr(cv2, "DISOpticalFlow_create", raising=False)
    monkeypatch.delattr(cv2, "DISOpticalFlow", raising=False)
    with pytest.raises(RuntimeError, match="DISOpticalFlow"):
        denseFlow(_frame(20), _frame(25), "dis")


def test_sparseFlow_tracks_corners():
    a, b = _frame(20), _frame(26)
    points = [(20, 30), (39, 49), (30, 40)]
    moved, status = sparseFlow(a, b, points)
    assert moved.shape == (3, 2) and moved.dtype == np.float32
    assert status.shape == (3,) and status.dtype == bool and status.all()
    assert np.allclose(moved[:, 0] - np.array([20, 39, 30]), 6, atol=1.0)
    assert np.allclose(moved[:, 1], [30, 49, 40], atol=1.0)
    outside = sparseFlow(a, b, np.array([[20, 30], [500, 500]]), winSize=(15, 15), maxLevel=1)
    assert bool(outside[1][0]) and not bool(outside[1][1])
    assert tuple(outside[0][1]) == (500, 500)
    empty = sparseFlow(a, b, [])
    assert empty[0].shape == (0, 2) and empty[1].shape == (0,)


def test_sparseFlow_errors():
    a, b = _frame(20), _frame(26)
    with pytest.raises(ValueError, match="points"):
        sparseFlow(a, b, [(np.nan, 1)])
    with pytest.raises(ValueError, match="winSize"):
        sparseFlow(a, b, [(1, 1)], winSize=(2, 2))
    with pytest.raises(ValueError, match="maxLevel"):
        sparseFlow(a, b, [(1, 1)], maxLevel=-1)


def test_trackPoints_drops_lost_points():
    frames = _moving(n=5, dx=3, start=20)
    results = list(trackPoints(frames, [(20, 30), (30, 40), (90, 90)], winSize=(15, 15)))
    assert len(results) == 4
    points, status = results[0]
    assert status.shape == (3,)
    assert points.shape[0] == int(status.sum())
    last_points, _ = results[-1]
    assert last_points.shape[0] >= 2
    assert np.allclose(last_points[:2, 0], [32, 42], atol=1.5)
    assert list(trackPoints([], [(1, 1)])) == []
    assert list(trackPoints(frames[:1], [(1, 1)])) == []
    assert next(iter(trackPoints(frames[:2], [])))[0].shape == (0, 2)


def test_flowMagnitude_and_flowToColor():
    flow = np.zeros((4, 6, 2), np.float32)
    flow[0, 0] = (3, 4)
    flow[1, 1] = (0, -2)
    magnitude, angle = flowMagnitude(flow)
    assert magnitude.shape == (4, 6) and magnitude[0, 0] == pytest.approx(5)
    assert angle[0, 0] == pytest.approx(53.13, abs=0.1) and angle[1, 1] == pytest.approx(270, abs=0.1)
    color = flowToColor(flow)
    assert color.shape == (4, 6, 3) and color.dtype == np.uint8
    assert color[2, 2].max() == 0 and color[0, 0].max() == 255
    fixed = flowToColor(flow, maxMagnitude=10)
    assert fixed[0, 0].max() == pytest.approx(128, abs=1) and fixed[1, 1].max() == pytest.approx(51, abs=1)
    assert flowToColor(np.zeros((2, 2, 2), np.float32)).max() == 0
    assert flowToColor(flow.astype(np.float64)).shape == (4, 6, 3)
    with pytest.raises(ValueError, match="flow"):
        flowMagnitude(np.zeros((4, 6), np.float32))
    with pytest.raises(ValueError, match="maxMagnitude"):
        flowToColor(flow, maxMagnitude=0)


def test_drawFlow():
    flow = np.zeros((32, 32, 2), np.float32)
    flow[..., 0] = 6
    img = np.zeros((32, 32, 3), np.uint8)
    assert drawFlow(img, flow, step=8) is img
    assert img[..., 1].max() > 200 and img[..., 0].max() == 0
    assert img[4, 4:11, 1].max() > 200
    gray = np.zeros((32, 32), np.uint8)
    drawFlow(gray, flow, step=8, color=(200, 200, 200), scale=0.5, thickness=2)
    assert gray[4, 4:9].max() > 0
    assert drawFlow(np.zeros((32, 32), np.uint8), flow, color=128).max() > 0
    with pytest.raises(ValueError, match="same size"):
        drawFlow(np.zeros((16, 32, 3), np.uint8), flow)
    with pytest.raises(ValueError, match="step"):
        drawFlow(img, flow, step=0)
    with pytest.raises(ValueError, match="scale"):
        drawFlow(img, flow, scale=float("nan"))
    with pytest.raises(ValueError, match="thickness"):
        drawFlow(img, flow, thickness=0)
    with pytest.raises(ValueError, match="color"):
        drawFlow(img, flow, color=(0, 0, 300))
    with pytest.raises(ValueError, match="src"):
        drawFlow(img.astype(np.float32), flow)


# --- motion accumulator --------------------------------------------------------------------


def test_MotionAccumulator():
    acc = MotionAccumulator((8, 10), decay=0.5)
    assert acc.shape == (8, 10) and acc.count == 0
    assert acc.heatmap().max() == 0
    mask = np.zeros((8, 10), np.uint8)
    mask[2, 2] = 255
    acc.update(mask)
    mask2 = np.zeros((8, 10), np.uint8)
    mask2[5, 5] = 255
    raw = acc.update(mask2)
    assert raw[2, 2] == pytest.approx(0.5) and raw[5, 5] == pytest.approx(1.0)
    heat = acc.heatmap()
    assert heat.dtype == np.float32 and heat.max() == 1.0 and heat[2, 2] == pytest.approx(0.5)
    assert acc.count == 2 and "count=2" in repr(acc)
    acc.update(np.ones((8, 10), bool))
    acc.update(np.full((8, 10), 2.0, np.float32))
    assert acc.heatmap().min() > 0
    acc.update(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    acc.reset()
    assert acc.count == 0 and acc.heatmap().max() == 0
    with pytest.raises(ValueError, match="mask"):
        acc.update(np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError, match="decay"):
        MotionAccumulator((8, 10), decay=1.5)
    with pytest.raises(ValueError, match="shape"):
        MotionAccumulator((8, 0))


def test_MotionAccumulator_with_frameDifference():
    frames = _moving(n=6)
    acc = MotionAccumulator((SIZE, SIZE), decay=0.9)
    for prev, nxt in pairwise(frames):
        acc.update(frameDifference(prev, nxt))
    heat = acc.heatmap()
    assert heat[30:50, 10:50].max() == 1.0 and heat[40, 20:40].max() > 0.5
    assert heat[80, :].max() == 0
