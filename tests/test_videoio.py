import numpy as np
import pytest

import cv2u as cv


def _frames(n=12, size=(64, 48), gray=False):
    w, h = size
    frames = []
    for i in range(n):
        frame = np.zeros((h, w) if gray else (h, w, 3), np.uint8)
        frame[:, : (i + 1) * w // n] = 255  # a white bar that grows, so frames are distinguishable
        frames.append(frame)
    return frames


def _bar_width(frame):
    gray = frame if frame.ndim == 2 else cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return int((gray[gray.shape[0] // 2] > 127).sum())


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "비디오.avi"  # non-ASCII paths work for video files
    assert cv.writeVideo(path, _frames(), fps=12) == 12
    return path


def test_roundtrip_and_info(video):
    info = cv.videoInfo(video)
    assert (info.width, info.height, info.frameCount) == (64, 48, 12)
    assert info.fps == pytest.approx(12)
    assert info.duration == pytest.approx(1.0)
    frames = list(cv.readFrames(video))
    assert len(frames) == 12
    assert frames[0].shape == (48, 64, 3)
    widths = [_bar_width(f) for f in frames]
    assert widths == sorted(widths)  # frame order is preserved


def test_readFrames_step_and_start(video):
    every_third = [_bar_width(f) for f in cv.readFrames(video, step=3)]
    all_widths = [_bar_width(f) for f in cv.readFrames(video)]
    assert every_third == all_widths[::3]
    assert [_bar_width(f) for f in cv.readFrames(video, start=10)] == all_widths[10:]
    assert list(cv.readFrames(video, start=100)) == []


def test_readFrames_releases_on_break(video):
    for _ in cv.readFrames(video):
        break  # generator is closed and the capture released; the file can be reopened right away
    assert len(list(cv.readFrames(video))) == 12


def test_readFrames_accepts_a_generator_for_writeVideo(video, tmp_path):
    copy = tmp_path / "copy.avi"
    assert cv.writeVideo(copy, cv.readFrames(video, step=2), fps=6) == 6
    assert cv.videoInfo(copy).frameCount == 6


def test_writeVideo_gray(tmp_path):
    path = tmp_path / "gray.avi"
    assert cv.writeVideo(path, _frames(gray=True), fps=10) == 12
    assert len(list(cv.readFrames(path))) == 12


def test_writeVideo_mp4(tmp_path):
    path = tmp_path / "clip.mp4"
    try:
        written = cv.writeVideo(path, _frames(), fps=12)
    except OSError:
        pytest.skip("mp4v codec is not available in this OpenCV build")
    assert written == 12
    assert len(list(cv.readFrames(path))) == 12


@pytest.mark.parametrize(
    ("name", "frames", "kwargs", "match"),
    [
        ("a.avi", [], {"fps": 10}, "empty"),
        ("a.avi", _frames(), {"fps": 0}, "fps"),
        ("a.xyz", _frames(), {"fps": 10}, "codec"),
        ("a.avi", _frames(), {"fps": 10, "fourcc": "MJPEG"}, "4 characters"),
        ("a.avi", [_frames()[0], np.zeros((10, 10, 3), np.uint8)], {"fps": 10}, "size"),
        ("a.avi", [np.zeros((10, 10, 3), np.float32)], {"fps": 10}, "8-bit"),
    ],
)
def test_writeVideo_validation(tmp_path, name, frames, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cv.writeVideo(tmp_path / name, frames, **kwargs)


def test_missing_source_raises(tmp_path):
    with pytest.raises(OSError):
        cv.readFrames(tmp_path / "missing.avi")  # raised on the call, not on the first frame
    with pytest.raises(OSError):
        cv.videoInfo(tmp_path / "missing.avi")


@pytest.mark.parametrize("kwargs", [{"step": 0}, {"start": -1}])
def test_readFrames_validation(video, kwargs):
    with pytest.raises(ValueError):
        cv.readFrames(video, **kwargs)


# --- readFrameAt -------------------------------------------------------------------------


def test_readFrameAt_index_and_seconds(video):
    frames = list(cv.readFrames(video))
    np.testing.assert_array_equal(cv.readFrameAt(video, index=5), frames[5])
    by_time = cv.readFrameAt(video, seconds=0.5)  # 12 fps: frame 6
    assert _bar_width(by_time) == _bar_width(frames[6])
    assert cv.readFrameAt(video, index=0) is not None
    assert cv.readFrameAt(video, index=100) is None  # past the end


@pytest.mark.parametrize("kwargs", [{}, {"index": 1, "seconds": 1.0}, {"index": -1}, {"seconds": -0.5}])
def test_readFrameAt_validation(video, kwargs):
    with pytest.raises(ValueError):
        cv.readFrameAt(video, **kwargs)


def test_readFrameAt_missing_file(tmp_path):
    with pytest.raises(OSError):
        cv.readFrameAt(tmp_path / "missing.avi", index=0)
