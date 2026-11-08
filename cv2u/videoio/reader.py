"""Video reading and writing objects: a seekable frame reader, a lazy writer, a threaded camera stream and
frame-rate helpers."""

import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from cv2u.videoio.video import _FOURCC_BY_EXTENSION, VideoInfo, VideoSource, _frame_channels, _open

__all__ = [
    "CameraStream",
    "FPSCounter",
    "VideoReader",
    "VideoWriterEx",
    "formatTimestamp",
    "frameToTime",
    "timeToFrame",
]


def _check_fps(fps: float) -> None:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")


class VideoReader:
    """A ``cv.VideoCapture`` that is a context manager, an iterator and knows its properties.

    ``cv.VideoCapture`` needs ``isOpened()`` checks, a ``while True: ok, frame = cap.read()`` loop, ``get()`` calls
    with ``CAP_PROP_*`` constants and a ``release()``. This class opens the source (or raises), iterates over the
    frames, exposes ``width``/``height``/``fps``/``frameCount`` and seeks by frame or by time.

    Args:
        source: File path or URL, or a camera index.

    Raises:
        OSError: If the source cannot be opened.

    Example:
        ```python
        with VideoReader("video.mp4") as video:
            for frame in video:
                ...
        ```
    """

    def __init__(self, source: VideoSource) -> None:
        self._cap = _open(source)
        self.source = source
        self.info = VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._cap.get(cv2.CAP_PROP_FPS)),
            frameCount=max(0, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        )
        """Size, frame rate and frame count (see :class:`VideoInfo`)."""
        self._position = 0

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self.info.width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self.info.height

    @property
    def fps(self) -> float:
        """Frame rate, or ``0.0`` when unknown (cameras and some streams)."""
        return self.info.fps

    @property
    def frameCount(self) -> int:
        """Number of frames (an estimate for some formats, ``0`` for cameras)."""
        return self.info.frameCount

    @property
    def position(self) -> int:
        """Index of the frame that the next :meth:`read` returns."""
        return self._position

    @property
    def isOpened(self) -> bool:
        """``False`` after :meth:`release`."""
        return self._cap.isOpened()

    def read(self) -> np.ndarray | None:
        """Read the next frame, or return ``None`` at the end (instead of a ``(False, None)`` tuple)."""
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._position += 1
        return frame

    def seek(self, index: int) -> None:
        """Move to frame ``index`` (0-based), so that the next :meth:`read` returns it.

        Seeking is exact for most files, but may land on a nearby key frame for some codecs. Not supported by
        cameras.

        Raises:
            ValueError: If ``index`` is negative.
        """
        if index < 0:
            raise ValueError(f"index must not be negative, got {index}")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self._position = index

    def seekTime(self, seconds: float) -> None:
        """Move to the frame at ``seconds`` from the start (see :meth:`seek`).

        Raises:
            ValueError: If ``seconds`` is negative or not finite.
        """
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"seconds must not be negative, got {seconds}")
        self._cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        self._position = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.fps > 0 else 0

    def frames(self, step: int = 1, start: int = 0, end: int | None = None) -> Iterator[np.ndarray]:
        """Iterate over the frames from ``start`` to ``end`` (exclusive), every ``step``-th one.

        Frames are skipped with ``grab()`` (no decoding), which also works for cameras; ``start`` seeks when it is
        before the current position.

        Raises:
            ValueError: If ``step < 1``, ``start < 0`` or ``end < start``.
        """
        if step < 1:
            raise ValueError(f"step must be at least 1, got {step}")
        if start < 0:
            raise ValueError(f"start must not be negative, got {start}")
        if end is not None and end < start:
            raise ValueError(f"end must not be before start, got start={start}, end={end}")
        if start < self._position:
            self.seek(start)
        while self._position < start:
            if not self._grab():
                return
        while end is None or self._position < end:
            frame = self.read()
            if frame is None:
                return
            yield frame
            for _ in range(step - 1):
                if (end is not None and self._position >= end) or not self._grab():
                    return

    def _grab(self) -> bool:
        if not self._cap.grab():
            return False
        self._position += 1
        return True

    def release(self) -> None:
        """Release the capture. Safe to call more than once."""
        self._cap.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.frames()

    def __len__(self) -> int:
        return self.frameCount

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"VideoReader({self.source!r}, {self.width}x{self.height}, {self.fps:g} fps, {self.frameCount} frames)"


class VideoWriterEx:
    """A ``cv.VideoWriter`` that takes its codec, size and color mode from the first frame written.

    ``cv.VideoWriter`` must be given a fourcc code, the frame size and ``isColor`` before the first frame exists,
    silently writes nothing when a frame has the wrong size, and fails on a missing folder. This writer is opened
    lazily on the first :meth:`write`, chooses the codec from the file extension (``.avi``: MJPG,
    ``.mp4``/``.m4v``/``.mov``: mp4v), converts grayscale frames to BGR when writing color, and validates every
    frame. Non-ASCII paths are supported on Windows (``cv.VideoWriter`` accepts them, unlike ``cv.imwrite``).

    Args:
        filename: Output path.
        fps: Frame rate.
        fourcc: Four-character codec code, e.g. ``"XVID"``. By default chosen from the file extension.
        isColor: ``True`` for a BGR file, ``False`` for grayscale; by default taken from the first frame.
        makeDirs: Create missing parent directories.

    Raises:
        ValueError: If ``fps`` is not positive, ``fourcc`` has not 4 characters or no codec is known for the
            extension.

    Example:
        ```python
        with VideoWriterEx("out.mp4", fps=30) as writer:
            for frame in frames:
                writer.write(frame)
        ```
    """

    def __init__(
        self,
        filename: str | os.PathLike[str],
        fps: float,
        fourcc: str | None = None,
        isColor: bool | None = None,
        makeDirs: bool = False,
    ) -> None:
        self.filename = os.fspath(filename)
        _check_fps(fps)
        extension = os.path.splitext(self.filename)[1].lower()
        code = fourcc or _FOURCC_BY_EXTENSION.get(extension)
        if code is None:
            raise ValueError(f"no default codec for {extension or 'files without an extension'!r}; pass fourcc")
        if len(code) != 4:
            raise ValueError(f"fourcc must have 4 characters, got {code!r}")
        self.fps = fps
        self.fourcc = code
        self._is_color = isColor
        self._make_dirs = makeDirs
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._count = 0

    @property
    def frameCount(self) -> int:
        """Number of frames written so far."""
        return self._count

    @property
    def size(self) -> tuple[int, int] | None:
        """Frame size ``(width, height)``, or ``None`` before the first frame."""
        return self._size

    @property
    def isColor(self) -> bool | None:
        """Whether the file is written in color, or ``None`` before the first frame when not given."""
        return self._is_color

    @property
    def isOpened(self) -> bool:
        """``True`` between the first :meth:`write` and :meth:`release`."""
        return self._writer is not None and self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        """Write one 8-bit grayscale or BGR frame; the first one opens the file.

        Raises:
            ValueError: If the frame is not 8-bit grayscale/BGR, or has a different size than the first frame,
                or is grayscale/color while the file is the other and cannot be converted.
            OSError: If the writer cannot be opened (for example, the codec is not available).
        """
        index = self._count
        if frame.dtype != np.uint8:
            raise ValueError(f"frame {index} must be 8-bit (uint8), got {frame.dtype}")
        channels = _frame_channels(frame, index)
        height, width = frame.shape[:2]
        if self._writer is None:
            if self._is_color is None:
                self._is_color = channels == 3
            if self._make_dirs:
                Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
            self._size = (width, height)
            self._writer = cv2.VideoWriter(
                self.filename, cv2.VideoWriter.fourcc(*self.fourcc), self.fps, self._size, self._is_color
            )
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise OSError(f"cannot open a video writer for {self.filename!r} with codec {self.fourcc!r}")
        elif (width, height) != self._size:
            assert self._size is not None
            raise ValueError(f"frame {index} has size {width}x{height}, expected {self._size[0]}x{self._size[1]}")
        if self._is_color and channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif not self._is_color and channels == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._writer.write(frame)
        self._count += 1

    def release(self) -> None:
        """Close the file. Safe to call more than once; nothing is written if no frame was."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriterEx":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.release()


class CameraStream:
    """Capture frames in a background thread, always keeping only the latest one.

    ``cv.VideoCapture.read()`` on a live camera or RTSP stream returns frames from an internal buffer, so a slow
    consumer gets stale frames that lag seconds behind. This reads continuously in a daemon thread and drops the
    frames nobody asked for, so :meth:`read` always returns the newest frame. Works on files too (where it plays
    them as fast as it can).

    Args:
        source: Camera index, URL or file path.
        maxWait: Seconds to wait in :meth:`read` for the first frame before giving up.

    Raises:
        OSError: If the source cannot be opened.

    Example:
        ```python
        with CameraStream(0) as camera:
            while True:
                frame = camera.read()
                ...
        ```
    """

    def __init__(self, source: VideoSource, maxWait: float = 5.0) -> None:
        self._cap = _open(source)
        self.source = source
        self.maxWait = maxWait
        self._frame: np.ndarray | None = None
        self._has_frame = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._counter = FPSCounter()
        self._ended = False

    @property
    def isRunning(self) -> bool:
        """``True`` while the capture thread runs."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def ended(self) -> bool:
        """``True`` once the source stopped delivering frames (end of file, camera lost)."""
        return self._ended

    @property
    def fps(self) -> float:
        """Measured capture frame rate (``0.0`` until a few frames arrived)."""
        return self._counter.fps

    def start(self) -> "CameraStream":
        """Start the capture thread. Returns ``self`` so it can be chained."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="cv2u.CameraStream", daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                self._ended = True
                break
            with self._lock:
                self._frame = frame
            self._counter.tick()
            self._has_frame.set()
        self._has_frame.set()

    def read(self) -> np.ndarray | None:
        """Return the latest frame, or ``None`` if none arrived within ``maxWait`` or the stream ended.

        Returns the same frame again when no new one arrived in the meantime. The thread is started if needed.
        """
        self.start()
        if not self._has_frame.wait(self.maxWait):
            return None
        with self._lock:
            return self._frame

    def stop(self) -> None:
        """Stop the thread and release the capture. Safe to call more than once."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._cap.release()

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.stop()


class FPSCounter:
    """Measure a loop's frame rate as a moving average over the last ``window`` ticks.

    Args:
        window: Number of ticks the average covers.
        clock: Time source returning seconds (``time.perf_counter`` by default); injectable for tests.

    Raises:
        ValueError: If ``window < 2``.

    Example:
        ```python
        counter = FPSCounter()
        for frame in cv.readFrames(0):
            fps = counter.tick()
        ```
    """

    def __init__(self, window: int = 30, clock: Callable[[], float] = time.perf_counter) -> None:
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        self.window = window
        self._clock = clock
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Record a frame and return the current frame rate (``0.0`` until two ticks happened)."""
        self._times.append(self._clock())
        return self.fps

    @property
    def fps(self) -> float:
        """Frames per second over the ticks in the window."""
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0

    def reset(self) -> None:
        """Forget all ticks."""
        self._times.clear()


def frameToTime(index: int, fps: float) -> float:
    """Convert a frame index to seconds from the start.

    Raises:
        ValueError: If ``fps`` is not positive.
    """
    _check_fps(fps)
    return index / fps


def timeToFrame(seconds: float, fps: float) -> int:
    """Convert seconds from the start to the frame index shown at that time (rounded down).

    Raises:
        ValueError: If ``fps`` is not positive or ``seconds`` is not finite.
    """
    _check_fps(fps)
    if not math.isfinite(seconds):
        raise ValueError(f"seconds must be finite, got {seconds}")
    return math.floor(seconds * fps + 1e-9)


def formatTimestamp(seconds: float, ms: bool = True) -> str:
    """Format seconds as ``"HH:MM:SS.mmm"`` (or ``"HH:MM:SS"`` with ``ms=False``), for overlays and file names.

    Raises:
        ValueError: If ``seconds`` is negative or not finite.
    """
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"seconds must not be negative, got {seconds}")
    total_ms = round(seconds * 1000) if ms else int(seconds) * 1000
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    text = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{text}.{millis:03d}" if ms else text
