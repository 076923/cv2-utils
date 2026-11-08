"""Video file utilities: frame extraction, trimming, concatenation, resizing, GIF export, scene detection and
thumbnails."""

import math
import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from cv2u.core.compose import montage
from cv2u.imgcodecs.files import imreadEx, imwriteEx
from cv2u.imgproc.resize import resizeEx
from cv2u.videoio.reader import VideoReader, VideoWriterEx
from cv2u.videoio.video import writeVideo

__all__ = [
    "concatVideos",
    "detectScenes",
    "extractFrames",
    "imagesToVideo",
    "resizeVideo",
    "sampleFrames",
    "trimVideo",
    "videoThumbnail",
    "videoToGif",
]

VideoPath = str | os.PathLike[str]


def _check_range(step: int, start: int, end: int | None) -> None:
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")
    if start < 0:
        raise ValueError(f"start must not be negative, got {start}")
    if end is not None and end < start:
        raise ValueError(f"end must not be before start, got start={start}, end={end}")


def extractFrames(
    source: VideoPath,
    outDir: VideoPath,
    step: int = 1,
    start: int = 0,
    end: int | None = None,
    ext: str = ".jpg",
    prefix: str = "frame_",
    digits: int = 6,
    progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Save the frames of a video as image files named by frame index, e.g. ``frame_000012.jpg``.

    The read loop, zero-padded names and the non-ASCII path problem of ``cv.imwrite`` on Windows are handled
    (frames are written with :func:`imwriteEx`). The directory is created if needed.

    Args:
        source: Video file path or URL.
        outDir: Directory to write the images into.
        step: Save every ``step``-th frame.
        start: First frame index to save.
        end: Stop before this frame index (``None``: until the end).
        ext: Image file extension, e.g. ``".png"``.
        prefix: File name prefix.
        digits: Number of digits of the zero-padded frame index.
        progress: Called as ``progress(index, frameCount)`` after each saved frame (``frameCount`` may be ``0``).

    Returns:
        The paths of the saved images, in frame order.

    Raises:
        OSError: If the video cannot be opened or an image cannot be written.
        ValueError: If ``step < 1``, ``start < 0`` or ``end < start``.
    """
    _check_range(step, start, end)
    directory = Path(outDir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with VideoReader(source) as video:
        for frame in video.frames(step=step, start=start, end=end):
            index = video.position - 1
            path = directory / f"{prefix}{index:0{digits}d}{ext}"
            if not imwriteEx(path, frame):
                raise OSError(f"cannot write {path}")
            paths.append(path)
            if progress is not None:
                progress(index, video.frameCount)
    return paths


def _write(dst: VideoPath, frames: Iterator[np.ndarray], fps: float, fourcc: str | None) -> int:
    with VideoWriterEx(dst, fps=fps, fourcc=fourcc, makeDirs=True) as writer:
        for frame in frames:
            writer.write(frame)
        count = writer.frameCount
    if count == 0:
        raise ValueError("no frames to write")
    return count


def trimVideo(
    source: VideoPath,
    dst: VideoPath,
    start: float,
    end: float | None = None,
    byTime: bool = True,
    fourcc: str | None = None,
) -> int:
    """Cut a part of a video into a new file.

    Args:
        source: Video file path.
        dst: Output path; the codec is chosen from its extension (see :func:`writeVideo`).
        start: Start, in seconds (or a frame index when ``byTime=False``).
        end: End (exclusive), in seconds or frames; ``None`` for the end of the video.
        byTime: Whether ``start``/``end`` are seconds (``True``) or frame indices.
        fourcc: Codec override.

    Returns:
        The number of frames written.

    Raises:
        OSError: If the source cannot be opened or the writer cannot be created.
        ValueError: If ``start`` is negative, ``end < start``, or the range contains no frames.
    """
    with VideoReader(source) as video:
        if byTime:
            if video.fps <= 0:
                raise ValueError("the video has no frame rate; use byTime=False")
            first = math.floor(start * video.fps + 1e-9)
            last = None if end is None else math.floor(end * video.fps + 1e-9)
        else:
            first, last = int(start), None if end is None else int(end)
        _check_range(1, first, last)
        return _write(dst, video.frames(start=first, end=last), video.fps or 30.0, fourcc)


def concatVideos(sources: Sequence[VideoPath], dst: VideoPath, fourcc: str | None = None) -> int:
    """Join videos one after another into one file, resized to the size of the first one.

    Args:
        sources: Video file paths, in order.
        dst: Output path.
        fourcc: Codec override.

    Returns:
        The number of frames written.

    Raises:
        OSError: If a source cannot be opened or the writer cannot be created.
        ValueError: If ``sources`` is empty.
    """
    if not sources:
        raise ValueError("sources must not be empty")

    def frames() -> Iterator[np.ndarray]:
        size: tuple[int, int] | None = None
        for path in sources:
            with VideoReader(path) as video:
                if size is None:
                    size = (video.width, video.height)
                for frame in video:
                    yield frame if frame.shape[1::-1] == size else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    with VideoReader(sources[0]) as first:
        fps = first.fps or 30.0
    return _write(dst, frames(), fps, fourcc)


def resizeVideo(
    source: VideoPath,
    dst: VideoPath,
    size: tuple[int, int] | None = None,
    scale: float | None = None,
    width: int | None = None,
    height: int | None = None,
    interpolation: int | None = None,
    fourcc: str | None = None,
) -> int:
    """Resize every frame of a video, keeping the aspect ratio when only one dimension is given.

    Give exactly one of ``size`` (``(width, height)``), ``scale``, or ``width`` and/or ``height`` (as for
    :func:`resizeEx`).

    Returns:
        The number of frames written.

    Raises:
        OSError: If the source cannot be opened or the writer cannot be created.
        ValueError: If no or conflicting target sizes are given, or they are not positive.
    """
    given = (size is not None) + (scale is not None) + (width is not None or height is not None)
    if given != 1:
        raise ValueError("give exactly one of size, scale, or width/height")
    if size is not None and (size[0] <= 0 or size[1] <= 0):
        raise ValueError(f"size must be positive, got {size}")
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        raise ValueError(f"scale must be positive, got {scale}")

    with VideoReader(source) as video:
        if size is not None:
            target_w, target_h = size
        elif scale is not None:
            target_w, target_h = max(1, round(video.width * scale)), max(1, round(video.height * scale))
        else:
            probe = resizeEx(np.zeros((video.height, video.width), np.uint8), width, height)
            target_h, target_w = probe.shape[:2]
        if interpolation is None:
            interpolation = cv2.INTER_AREA if target_w < video.width else cv2.INTER_LINEAR
        frames = (cv2.resize(f, (target_w, target_h), interpolation=interpolation) for f in video)
        return _write(dst, frames, video.fps or 30.0, fourcc)


def videoToGif(
    source: VideoPath,
    dst: VideoPath,
    fps: float | None = None,
    size: tuple[int, int] | None = None,
    step: int = 1,
    maxFrames: int | None = None,
    loop: int = 0,
) -> int:
    """Convert a video (or a part of it) to an animated GIF with Pillow.

    ``cv.VideoWriter`` cannot write GIFs. Frames are converted from BGR to RGB; ``step`` and ``maxFrames`` keep
    the file small.

    Args:
        source: Video file path.
        dst: Output ``.gif`` path.
        fps: Playback frame rate of the GIF (default: the video's rate divided by ``step``).
        size: Output ``(width, height)``; ``None`` keeps the frame size.
        step: Take every ``step``-th frame.
        maxFrames: Stop after this many frames.
        loop: Number of repetitions, ``0`` for forever.

    Returns:
        The number of frames written.

    Raises:
        OSError: If the source cannot be opened.
        ValueError: If the video has no frames, ``step < 1``, ``maxFrames < 1`` or ``fps`` is not positive.
    """
    from PIL import Image

    _check_range(step, 0, None)
    if maxFrames is not None and maxFrames < 1:
        raise ValueError(f"maxFrames must be at least 1, got {maxFrames}")
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError(f"fps must be positive, got {fps}")
    images: list[Image.Image] = []
    with VideoReader(source) as video:
        if fps is None:
            fps = video.fps / step if video.fps > 0 else 10.0
        for frame in video.frames(step=step):
            if size is not None:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame
            images.append(Image.fromarray(rgb))
            if maxFrames is not None and len(images) >= maxFrames:
                break
    if not images:
        raise ValueError("the video has no frames")
    path = Path(dst)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, format="GIF", save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=loop)
    return len(images)


def imagesToVideo(
    images: Sequence[VideoPath | np.ndarray], dst: VideoPath, fps: float = 30, fourcc: str | None = None
) -> int:
    """Write image files and/or arrays to a video file, e.g. to make a time-lapse from a folder.

    Paths are read with :func:`imreadEx` (any path, non-ASCII included); see :func:`listImages` to collect them.
    All images must have the same size, as for :func:`writeVideo`.

    Returns:
        The number of frames written.

    Raises:
        OSError: If an image file cannot be read.
        ValueError: As :func:`writeVideo` (no images, size mismatch, bad fps or codec).
    """

    def frames() -> Iterator[np.ndarray]:
        for item in images:
            if isinstance(item, np.ndarray):
                yield item
            else:
                image = imreadEx(item)
                if image is None:
                    raise OSError(f"cannot read image {os.fspath(item)!r}")
                yield image

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    return writeVideo(dst, frames(), fps=fps, fourcc=fourcc)


def detectScenes(source: VideoPath, threshold: float = 30.0, minLength: int = 15, step: int = 1) -> list[int]:
    """Find the frame indices where the scene changes (hard cuts).

    Frames are downscaled to about 64 pixels wide and converted to grayscale; a cut is reported where the mean
    absolute difference between consecutive compared frames exceeds ``threshold`` (0-255). Gradual fades and
    very fast motion may be missed or over-reported; for those use a dedicated library such as PySceneDetect.

    Args:
        source: Video file path.
        threshold: Mean absolute gray difference (0-255) that counts as a cut; lower is more sensitive.
        minLength: Minimum number of frames between two cuts.
        step: Compare every ``step``-th frame (faster, but cuts are reported at a multiple of ``step``).

    Returns:
        The indices of the first frame of each new scene, in increasing order (the first frame is not included).

    Raises:
        OSError: If the video cannot be opened.
        ValueError: If ``threshold`` is negative or not finite, ``minLength < 1`` or ``step < 1``.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must not be negative, got {threshold}")
    if minLength < 1:
        raise ValueError(f"minLength must be at least 1, got {minLength}")
    _check_range(step, 0, None)
    cuts: list[int] = []
    previous: np.ndarray | None = None
    last_cut = -minLength
    with VideoReader(source) as video:
        for frame in video.frames(step=step):
            index = video.position - 1
            small = _small_gray(frame)
            if previous is not None:
                score = float(np.mean(cv2.absdiff(small, previous)))
                if score > threshold and index - last_cut >= minLength:
                    cuts.append(index)
                    last_cut = index
            previous = small
    return cuts


def _small_gray(frame: np.ndarray, width: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    height = max(1, round(gray.shape[0] * width / gray.shape[1]))
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def sampleFrames(source: VideoPath, count: int) -> list[np.ndarray]:
    """Read ``count`` evenly spaced frames of a video, e.g. for a preview or a dataset sample.

    Seeks with ``CAP_PROP_POS_FRAMES`` (like :func:`readFrameAt`), so it is fast on long videos, but may land on
    a nearby key frame for some codecs. Fewer frames are returned when the video is shorter than ``count``.

    Raises:
        OSError: If the video cannot be opened.
        ValueError: If ``count < 1``.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    frames: list[np.ndarray] = []
    with VideoReader(source) as video:
        total = video.frameCount
        if total <= 0:
            frames.extend(video.frames())  # unknown length: decode everything, then pick
            indices = _spaced(len(frames), count)
            return [frames[i] for i in indices]
        for index in _spaced(total, count):
            video.seek(index)
            frame = video.read()
            if frame is None:
                break
            frames.append(frame)
    return frames


def _spaced(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    count = min(count, total)
    return sorted({int((i + 0.5) * total / count) for i in range(count)})


def videoThumbnail(
    source: VideoPath,
    count: int = 4,
    size: tuple[int, int] | None = None,
    cols: int | None = None,
    gap: int = 0,
) -> np.ndarray:
    """Make a contact-sheet image of ``count`` evenly spaced frames.

    Args:
        source: Video file path.
        count: Number of frames.
        size: ``(width, height)`` of each cell; ``None`` keeps the frame size.
        cols: Number of columns of the grid (default: as square as possible, see :func:`montage`).
        gap: Space in pixels between the cells.

    Returns:
        The montage image.

    Raises:
        OSError: If the video cannot be opened.
        ValueError: If ``count < 1`` or the video has no frames.
    """
    frames = sampleFrames(source, count)
    if not frames:
        raise ValueError("the video has no frames")
    if size is not None:
        frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
    return montage(frames, cols=cols, gap=gap)
