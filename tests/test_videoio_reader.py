import numpy as np
import pytest

import cv2u as cv
from cv2u.videoio.reader import (
    CameraStream,
    FPSCounter,
    VideoReader,
    VideoWriterEx,
    formatTimestamp,
    frameToTime,
    timeToFrame,
)


def _frames(n=12, size=(64, 48), gray=False):
    w, h = size
    frames = []
    for i in range(n):
        frame = np.zeros((h, w) if gray else (h, w, 3), np.uint8)
        frame[:, : (i + 1) * w // n] = 255
        frames.append(frame)
    return frames


def _bar_width(frame):
    gray = frame if frame.ndim == 2 else cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return int((gray[gray.shape[0] // 2] > 127).sum())


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "비디오.avi"
    try:
        assert cv.writeVideo(path, _frames(), fps=12) == 12
    except OSError:
        pytest.skip("MJPG codec is not available")
    return path


# --- VideoReader -------------------------------------------------------------------------


def test_VideoReader_properties_and_iteration(video):
    with VideoReader(video) as reader:
        assert (reader.width, reader.height, reader.frameCount) == (64, 48, 12)
        assert reader.fps == pytest.approx(12)
        assert reader.info.duration == pytest.approx(1.0)
        assert len(reader) == 12
        assert reader.isOpened
        assert "64x48" in repr(reader)
        frames = list(reader)
        assert len(frames) == 12
        assert reader.position == 12
        assert reader.read() is None
    assert not reader.isOpened
    widths = [_bar_width(f) for f in frames]
    assert widths == sorted(widths)


def test_VideoReader_read_and_seek(video):
    with VideoReader(video) as reader:
        first = reader.read()
        assert first is not None and reader.position == 1
        reader.seek(5)
        assert reader.position == 5
        frame5 = reader.read()
        reader.seekTime(0.5)  # 12 fps: frame 6
        assert reader.position == 6
        frame6 = reader.read()
        assert _bar_width(frame5) < _bar_width(frame6)
        reader.seek(0)
        np.testing.assert_array_equal(reader.read(), first)


def test_VideoReader_frames_range_and_step(video):
    with VideoReader(video) as reader:
        all_widths = [_bar_width(f) for f in reader.frames()]
        assert [_bar_width(f) for f in reader.frames(step=3)] == all_widths[::3]  # rewinds automatically
        assert [_bar_width(f) for f in reader.frames(start=4, end=8)] == all_widths[4:8]
        assert [_bar_width(f) for f in reader.frames(step=2, start=3, end=8)] == all_widths[3:8:2]
        assert list(reader.frames(start=100)) == []
        assert list(reader.frames(start=5, end=5)) == []


def test_VideoReader_frames_forward_uses_grab(video):
    with VideoReader(video) as reader:
        reader.read()
        got = [_bar_width(f) for f in reader.frames(start=3)]
        assert len(got) == 9


@pytest.mark.parametrize("kwargs", [{"step": 0}, {"start": -1}, {"start": 3, "end": 2}])
def test_VideoReader_frames_validation(video, kwargs):
    with VideoReader(video) as reader, pytest.raises(ValueError):
        list(reader.frames(**kwargs))


def test_VideoReader_seek_validation(video):
    with VideoReader(video) as reader:
        with pytest.raises(ValueError, match="index"):
            reader.seek(-1)
        with pytest.raises(ValueError, match="seconds"):
            reader.seekTime(-0.1)


def test_VideoReader_missing(tmp_path):
    with pytest.raises(OSError):
        VideoReader(tmp_path / "missing.avi")


# --- VideoWriterEx -----------------------------------------------------------------------


def test_VideoWriterEx_lazy_open_and_makeDirs(tmp_path):
    path = tmp_path / "새폴더" / "출력.avi"
    with VideoWriterEx(path, fps=10, makeDirs=True) as writer:
        assert writer.size is None and writer.isColor is None and not writer.isOpened
        for frame in _frames(5):
            writer.write(frame)
        assert writer.size == (64, 48) and writer.isColor is True and writer.isOpened
        assert writer.frameCount == 5
    assert not writer.isOpened
    assert cv.videoInfo(path).frameCount == 5


def test_VideoWriterEx_gray_to_color_conversion(tmp_path):
    path = tmp_path / "mixed.avi"
    with VideoWriterEx(path, fps=10) as writer:
        writer.write(_frames(1)[0])
        writer.write(_frames(1, gray=True)[0])  # gray frame converted to BGR
    assert cv.videoInfo(path).frameCount == 2


def test_VideoWriterEx_gray_file_accepts_color(tmp_path):
    path = tmp_path / "gray.avi"
    with VideoWriterEx(path, fps=10, isColor=False) as writer:
        writer.write(_frames(1, gray=True)[0])
        writer.write(_frames(1)[0])
    assert cv.videoInfo(path).frameCount == 2


def test_VideoWriterEx_no_frames_writes_nothing(tmp_path):
    path = tmp_path / "empty.avi"
    with VideoWriterEx(path, fps=10):
        pass
    assert not path.exists()


def test_VideoWriterEx_frame_validation(tmp_path):
    with VideoWriterEx(tmp_path / "a.avi", fps=10) as writer:
        with pytest.raises(ValueError, match="8-bit"):
            writer.write(np.zeros((10, 10, 3), np.float32))
        with pytest.raises(ValueError, match="grayscale"):
            writer.write(np.zeros((10, 10, 4), np.uint8))
        writer.write(_frames(1)[0])
        with pytest.raises(ValueError, match="size"):
            writer.write(np.zeros((10, 10, 3), np.uint8))


@pytest.mark.parametrize(
    ("name", "kwargs", "match"),
    [("a.avi", {"fps": 0}, "fps"), ("a.xyz", {"fps": 10}, "codec"), ("a.avi", {"fps": 10, "fourcc": "MJPEG"}, "4 ")],
)
def test_VideoWriterEx_init_validation(tmp_path, name, kwargs, match):
    with pytest.raises(ValueError, match=match):
        VideoWriterEx(tmp_path / name, **kwargs)


def test_VideoWriterEx_bad_codec_raises_OSError(tmp_path):
    with VideoWriterEx(tmp_path / "a.avi", fps=10, fourcc="ZZZZ") as writer, pytest.raises(OSError):
        writer.write(_frames(1)[0])


# --- CameraStream ------------------------------------------------------------------------


def test_CameraStream_on_a_file(video):
    with CameraStream(video) as stream:
        frame = stream.read()
        assert frame is not None and frame.shape == (48, 64, 3)
        assert stream.isRunning or stream.ended
        again = stream.read()
        assert again is not None
    assert not stream.isRunning
    assert stream.fps >= 0


def test_CameraStream_ends_and_delivers_last_frame_once(video):
    stream = CameraStream(video, maxWait=2.0).start()
    try:
        stream._thread.join(timeout=5)  # the file is short: the thread finishes
        assert stream.ended
        last = stream.read()
        assert last is not None and _bar_width(last) == 64
        assert stream.read() is None  # the docstring promise: None once the stream ended
        assert stream.read() is None
    finally:
        stream.stop()
        stream.stop()  # idempotent


def test_CameraStream_read_loop_terminates_on_a_file(video):
    import time

    deadline = time.monotonic() + 10
    last = None
    count = 0
    with CameraStream(video, maxWait=2.0) as stream:
        while (frame := stream.read()) is not None:  # the same frame may repeat while the thread still runs
            last, count = frame, count + 1
            if time.monotonic() > deadline:
                pytest.fail("read() never returned None after the file ended")
    assert count >= 1 and _bar_width(last) == 64


def test_CameraStream_missing(tmp_path):
    with pytest.raises(OSError):
        CameraStream(tmp_path / "missing.avi")


# --- FPSCounter --------------------------------------------------------------------------


def test_FPSCounter_moving_average():
    now = [0.0]
    counter = FPSCounter(window=4, clock=lambda: now[0])
    assert counter.fps == 0.0
    assert counter.tick() == 0.0  # one tick: no interval yet
    for _ in range(3):
        now[0] += 0.1
        fps = counter.tick()
    assert fps == pytest.approx(10.0)
    for _ in range(4):
        now[0] += 0.05  # faster: the old ticks fall out of the window
        fps = counter.tick()
    assert fps == pytest.approx(20.0)
    counter.reset()
    assert counter.fps == 0.0


def test_FPSCounter_zero_elapsed_and_validation():
    counter = FPSCounter(window=2, clock=lambda: 1.0)
    counter.tick()
    assert counter.tick() == 0.0
    with pytest.raises(ValueError):
        FPSCounter(window=1)


def test_FPSCounter_default_clock():
    counter = FPSCounter()
    counter.tick()
    assert counter.tick() >= 0.0


# --- time helpers ------------------------------------------------------------------------


def test_frame_time_conversion():
    assert frameToTime(30, 30) == pytest.approx(1.0)
    assert timeToFrame(1.0, 30) == 30
    assert timeToFrame(0.5, 12) == 6
    assert timeToFrame(0.1, 29.97) == 2
    with pytest.raises(ValueError):
        frameToTime(1, 0)
    with pytest.raises(ValueError):
        timeToFrame(1.0, -1)
    with pytest.raises(ValueError):
        timeToFrame(float("nan"), 30)


def test_formatTimestamp():
    assert formatTimestamp(0) == "00:00:00.000"
    assert formatTimestamp(3661.5) == "01:01:01.500"
    assert formatTimestamp(59.9996) == "00:01:00.000"
    assert formatTimestamp(3661.9, ms=False) == "01:01:01"
    with pytest.raises(ValueError):
        formatTimestamp(-1)
    with pytest.raises(ValueError):
        formatTimestamp(float("inf"))
