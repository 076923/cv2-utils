import numpy as np
import pytest

import cv2u as cv
from cv2u.videoio.edit import (
    concatVideos,
    detectScenes,
    extractFrames,
    imagesToVideo,
    resizeVideo,
    sampleFrames,
    trimVideo,
    videoThumbnail,
    videoToGif,
)


def _frames(n=12, size=(64, 48), color=(255, 255, 255)):
    w, h = size
    frames = []
    for i in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:, : (i + 1) * w // n] = color
        frames.append(frame)
    return frames


def _bar_width(frame):
    gray = frame if frame.ndim == 2 else cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return int((gray[gray.shape[0] // 2] > 60).sum())


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "비디오.avi"
    try:
        assert cv.writeVideo(path, _frames(), fps=12) == 12
    except OSError:
        pytest.skip("MJPG codec is not available")
    return path


def test_extractFrames(video, tmp_path):
    seen = []
    out = tmp_path / "프레임"
    paths = extractFrames(video, out, step=4, start=1, ext=".png", progress=lambda i, n: seen.append((i, n)))
    assert [p.name for p in paths] == ["frame_000001.png", "frame_000005.png", "frame_000009.png"]
    assert seen == [(1, 12), (5, 12), (9, 12)]
    assert all(p.exists() for p in paths)
    widths = [_bar_width(cv.imreadEx(p)) for p in paths]
    assert widths == sorted(widths)
    assert extractFrames(video, out, start=2, end=4, digits=3, prefix="f") == [out / "f002.jpg", out / "f003.jpg"]


def test_extractFrames_validation(video, tmp_path):
    with pytest.raises(ValueError):
        extractFrames(video, tmp_path, step=0)
    with pytest.raises(ValueError):
        extractFrames(video, tmp_path, start=5, end=2)
    with pytest.raises(OSError):
        extractFrames(tmp_path / "missing.avi", tmp_path)


def test_trimVideo_by_time_and_frames(video, tmp_path):
    out = tmp_path / "trim.avi"
    assert trimVideo(video, out, 0.25, 0.5) == 3  # frames 3, 4, 5
    assert [_bar_width(f) for f in cv.readFrames(out)] == [_bar_width(f) for f in _frames()[3:6]]
    assert trimVideo(video, out, 8, byTime=False) == 4
    assert trimVideo(video, out, 2, 5, byTime=False) == 3
    with pytest.raises(ValueError):
        trimVideo(video, out, 5, 2, byTime=False)
    with pytest.raises(ValueError):
        trimVideo(video, out, 100, byTime=False)  # no frames in range


def test_concatVideos_resizes_to_first(video, tmp_path):
    small = tmp_path / "small.avi"
    cv.writeVideo(small, _frames(4, size=(32, 24)), fps=12)
    out = tmp_path / "concat.avi"
    assert concatVideos([video, small], out) == 16
    info = cv.videoInfo(out)
    assert (info.width, info.height, info.frameCount) == (64, 48, 16)
    with pytest.raises(ValueError):
        concatVideos([], out)


def test_resizeVideo(video, tmp_path):
    out = tmp_path / "resized.avi"
    assert resizeVideo(video, out, width=32) == 12
    info = cv.videoInfo(out)
    assert (info.width, info.height) == (32, 24)
    resizeVideo(video, out, scale=0.5)
    assert (cv.videoInfo(out).width, cv.videoInfo(out).height) == (32, 24)
    resizeVideo(video, out, size=(20, 30))
    assert (cv.videoInfo(out).width, cv.videoInfo(out).height) == (20, 30)
    resizeVideo(video, out, height=96)  # upscaling path
    assert cv.videoInfo(out).width == 128


@pytest.mark.parametrize(
    "kwargs", [{}, {"size": (10, 10), "scale": 0.5}, {"scale": 0}, {"size": (0, 10)}, {"scale": float("nan")}]
)
def test_resizeVideo_validation(video, tmp_path, kwargs):
    with pytest.raises(ValueError):
        resizeVideo(video, tmp_path / "x.avi", **kwargs)


def test_videoToGif(video, tmp_path):
    from PIL import Image

    gif = tmp_path / "애니.gif"
    assert videoToGif(video, gif, step=2, size=(32, 24)) == 6
    with Image.open(gif) as image:
        assert image.n_frames == 6
        assert image.size == (32, 24)
    assert videoToGif(video, tmp_path / "out" / "b.gif", maxFrames=3, fps=5) == 3
    for kwargs in ({"step": 0}, {"maxFrames": 0}, {"fps": 0}):
        with pytest.raises(ValueError):
            videoToGif(video, gif, **kwargs)


def test_imagesToVideo_paths_and_arrays(tmp_path):
    frames = _frames(4)
    paths = []
    for i, frame in enumerate(frames[:2]):
        path = tmp_path / f"이미지_{i}.png"
        assert cv.imwriteEx(path, frame)
        paths.append(path)
    out = tmp_path / "out" / "images.avi"
    assert imagesToVideo([*paths, *frames[2:]], out, fps=4) == 4
    assert cv.videoInfo(out).frameCount == 4
    with pytest.raises(OSError):
        imagesToVideo([tmp_path / "missing.png"], out)
    with pytest.raises(ValueError):
        imagesToVideo([], out)


def test_detectScenes_finds_the_cut(tmp_path):
    frames = _frames(10, color=(40, 40, 40)) + _frames(10, color=(255, 0, 0))
    path = tmp_path / "scenes.avi"
    try:
        cv.writeVideo(path, frames, fps=10)
    except OSError:
        pytest.skip("MJPG codec is not available")
    assert detectScenes(path, threshold=30.0, minLength=5) == [10]
    assert detectScenes(path, threshold=30.0, minLength=5, step=2) == [10]
    assert detectScenes(path, threshold=250.0) == []
    with pytest.raises(ValueError):
        detectScenes(path, threshold=-1)
    with pytest.raises(ValueError):
        detectScenes(path, minLength=0)
    with pytest.raises(ValueError):
        detectScenes(path, step=0)


def test_sampleFrames_and_thumbnail(video):
    frames = sampleFrames(video, 4)
    assert len(frames) == 4
    widths = [_bar_width(f) for f in frames]
    assert widths == sorted(widths) and len(set(widths)) == 4
    assert len(sampleFrames(video, 100)) == 12  # capped at the frame count
    sheet = videoThumbnail(video, count=4, size=(32, 24), cols=2, gap=2)
    assert sheet.shape == (24 * 2 + 2, 32 * 2 + 2, 3)
    assert videoThumbnail(video, count=3).shape[2] == 3
    with pytest.raises(ValueError):
        sampleFrames(video, 0)
