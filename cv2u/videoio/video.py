import os
from collections.abc import Iterable, Iterator
from itertools import chain
from typing import NamedTuple

import cv2
import numpy as np

__all__ = ["VideoInfo", "readFrameAt", "readFrames", "videoInfo", "writeVideo"]

VideoSource = str | os.PathLike[str] | int
"""A file path or URL, or a camera index (as for ``cv.VideoCapture``)."""

# Codecs that ship with every opencv-python build for the common containers.
_FOURCC_BY_EXTENSION = {".avi": "MJPG", ".mp4": "mp4v", ".m4v": "mp4v", ".mov": "mp4v"}


def _open(source: VideoSource) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source if isinstance(source, int) else os.fspath(source))
    if not cap.isOpened():
        cap.release()
        raise OSError(f"cannot open video source {source!r}")
    return cap


class VideoInfo(NamedTuple):
    """Properties of a video, as reported by OpenCV. ``frameCount`` may be an estimate for some formats."""

    width: int
    height: int
    fps: float
    frameCount: int

    @property
    def duration(self) -> float:
        """Length in seconds (``frameCount / fps``), or ``0.0`` if the frame rate is unknown."""
        return self.frameCount / self.fps if self.fps > 0 else 0.0


def videoInfo(source: VideoSource) -> VideoInfo:
    """Return the size, frame rate and frame count of a video in one call.

    Raises:
        OSError: If the source cannot be opened.
    """
    cap = _open(source)
    try:
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            frameCount=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        )
    finally:
        cap.release()


def readFrames(source: VideoSource, step: int = 1, start: int = 0) -> Iterator[np.ndarray]:
    """Iterate over the frames of a video file, stream or camera.

    Replaces the usual ``cap = cv.VideoCapture(...)`` / ``while True: ok, frame = cap.read()`` / ``cap.release()``
    loop. The capture is released when the iteration ends, also on ``break``.

    Args:
        source: File path or URL, or a camera index.
        step: Yield every ``step``-th frame (skipped frames are not decoded).
        start: Number of frames to skip at the beginning.

    Raises:
        OSError: If the source cannot be opened.
        ValueError: If ``step < 1`` or ``start < 0``.

    Example:
        ```python
        for frame in cv.readFrames("video.mp4", step=5):
            ...
        ```
    """
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")
    if start < 0:
        raise ValueError(f"start must not be negative, got {start}")
    return _frames(_open(source), step, start)


def _frames(cap: cv2.VideoCapture, step: int, start: int) -> Iterator[np.ndarray]:
    try:
        # grab() skips without decoding, and works for cameras, unlike seeking with CAP_PROP_POS_FRAMES.
        for _ in range(start):
            if not cap.grab():
                return
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
            for _ in range(step - 1):
                if not cap.grab():
                    return
    finally:
        cap.release()


def writeVideo(
    filename: str | os.PathLike[str],
    frames: Iterable[np.ndarray],
    fps: float,
    fourcc: str | None = None,
) -> int:
    """Write frames to a video file.

    The frame size and color mode are taken from the first frame, and the codec from the file extension
    (``.avi``: MJPG, ``.mp4``/``.m4v``/``.mov``: mp4v) unless ``fourcc`` is given.

    Args:
        filename: Output path.
        frames: 8-bit BGR or grayscale frames, all of the same size. Any iterable works, including a generator
            such as :func:`readFrames`.
        fps: Frame rate.
        fourcc: Four-character codec code, e.g. ``"XVID"``.

    Returns:
        The number of frames written.

    Raises:
        ValueError: If there are no frames, a frame has a different size or is not 8-bit, ``fps`` is not positive,
            or no codec is known for the file extension.
        OSError: If the video writer cannot be opened (for example, the codec is not available).
    """
    path = os.fspath(filename)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    extension = os.path.splitext(path)[1].lower()
    code = fourcc or _FOURCC_BY_EXTENSION.get(extension)
    if code is None:
        raise ValueError(f"no default codec for {extension or 'files without an extension'!r}; pass fourcc")
    if len(code) != 4:
        raise ValueError(f"fourcc must have 4 characters, got {code!r}")

    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        raise ValueError("frames must not be empty")
    height, width = first.shape[:2]
    is_color = first.ndim == 3

    writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*code), fps, (width, height), is_color)
    if not writer.isOpened():
        raise OSError(f"cannot open a video writer for {path!r} with codec {code!r}")
    count = 0
    try:
        for frame in chain([first], iterator):
            if frame.dtype != np.uint8:
                raise ValueError(f"frame {count} must be 8-bit (uint8), got {frame.dtype}")
            if frame.shape[:2] != (height, width):
                raise ValueError(f"frame {count} has size {frame.shape[1]}x{frame.shape[0]}, expected {width}x{height}")
            writer.write(frame)
            count += 1
    finally:
        writer.release()
    return count


def readFrameAt(
    source: str | os.PathLike[str], index: int | None = None, seconds: float | None = None
) -> np.ndarray | None:
    """Read a single frame of a video file, by frame number or by time, e.g. for a thumbnail.

    Replaces opening a capture, seeking with ``cap.set(cv.CAP_PROP_POS_FRAMES, ...)``, reading and releasing.
    Seeking is exact for most files, but may land on a nearby key frame for some codecs.

    Args:
        source: Video file path or URL.
        index: Frame number, starting at 0.
        seconds: Time from the start, in seconds. Give either ``index`` or ``seconds``.

    Returns:
        The frame, or ``None`` if the video is shorter.

    Raises:
        OSError: If the video cannot be opened.
        ValueError: If not exactly one of ``index`` and ``seconds`` is given, or it is negative.
    """
    if (index is None) == (seconds is None):
        raise ValueError("give exactly one of index and seconds")
    position = index if index is not None else seconds
    assert position is not None
    if position < 0:
        raise ValueError(f"index and seconds must not be negative, got {position}")
    cap = _open(source)
    try:
        if index is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        else:
            assert seconds is not None
            cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()
