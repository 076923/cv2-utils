"""Batch image I/O: lazy image sequences, parallel reading, folder processing and a decoded-image cache."""

import glob
import os
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import TypeVar, cast, overload

import cv2
import numpy as np

from .._validate import check_image_not_empty, check_size
from .files import _IMAGE_EXTENSIONS, _natural_key, imreadEx, imwriteEx
from .meta import _prepare_for_format, convertImageFormat, imreadOriented

__all__ = [
    "ImageSequence",
    "clearImageCache",
    "convertImages",
    "imreadCached",
    "imreadParallel",
    "processImages",
    "stackImages",
    "unstackImages",
]

T = TypeVar("T")

PathLike = str | os.PathLike[str]


def _resolve_paths(source: PathLike | Iterable[PathLike]) -> list[Path]:
    """Expand a folder, a glob pattern or a list of paths into a naturally sorted path list."""
    if isinstance(source, (str, os.PathLike)):
        text = os.fspath(source)
        path = Path(text)
        if path.is_dir():
            files = [f for f in path.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS and f.is_file()]
            return sorted(files, key=_natural_key)
        if glob.has_magic(text):
            return sorted((Path(f) for f in glob.glob(text, recursive=True)), key=_natural_key)
        return [path]
    return sorted((Path(os.fspath(p)) for p in source), key=_natural_key)


class ImageSequence(Sequence[np.ndarray]):
    """A folder, glob or list of image files that reads like a list of images, decoding lazily.

    OpenCV has ``cv.VideoCapture`` for frame sequences (``img_%03d.png``), but it decodes everything in order, cannot
    be indexed or sliced, and sorts ``img10`` before ``img2``. ``ImageSequence`` lists the files once in natural
    order and decodes an image only when it is indexed, optionally keeping the last ``cache`` decoded images.

    Args:
        source: A folder (its image files), a glob pattern such as ``"data/*.png"`` or an iterable of paths.
        flags: ``cv.IMREAD_*`` flags, as for ``cv.imread``.
        cache: Number of decoded images to keep in a least-recently-used cache; ``0`` disables caching.

    Raises:
        ValueError: If ``cache`` is negative.

    Examples:
        >>> seq = ImageSequence("frames/")  # doctest: +SKIP
        >>> first, last = seq[0], seq[-1]  # doctest: +SKIP
        >>> for image in seq[::10]:  # doctest: +SKIP
        ...     pass
    """

    def __init__(self, source: PathLike | Iterable[PathLike], *, flags: int = cv2.IMREAD_COLOR, cache: int = 0):
        if cache < 0:
            raise ValueError(f"cache must be >= 0, got {cache}")
        self._paths = _resolve_paths(source)
        self._flags = flags
        self._cache_size = cache
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    @property
    def paths(self) -> list[Path]:
        """The image paths, in natural order."""
        return list(self._paths)

    @property
    def flags(self) -> int:
        """The ``cv.IMREAD_*`` flags used for decoding."""
        return self._flags

    def __len__(self) -> int:
        return len(self._paths)

    def _read(self, index: int) -> np.ndarray:
        if index < 0:
            index += len(self._paths)
        if not 0 <= index < len(self._paths):
            raise IndexError(f"index {index} out of range for {len(self._paths)} images")
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        path = self._paths[index]
        image = imreadEx(path, self._flags)
        if image is None:
            if not path.is_file():
                raise FileNotFoundError(f"image file not found: {path}")
            raise OSError(f"cannot decode image: {path}")
        if self._cache_size:
            self._cache[index] = image
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return image

    @overload
    def __getitem__(self, index: int) -> np.ndarray: ...

    @overload
    def __getitem__(self, index: slice) -> list[np.ndarray]: ...

    def __getitem__(self, index: int | slice) -> np.ndarray | list[np.ndarray]:
        if isinstance(index, slice):
            return [self._read(i) for i in range(*index.indices(len(self._paths)))]
        return self._read(int(index))

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self._paths)):
            yield self._read(i)

    def clearCache(self) -> None:
        """Drop the cached decoded images."""
        self._cache.clear()

    def __repr__(self) -> str:
        first = f", first={self._paths[0].name!r}" if self._paths else ""
        return f"ImageSequence(len={len(self._paths)}, flags={self._flags}, cache={self._cache_size}{first})"


def imreadParallel(
    filenames: Iterable[PathLike], *, flags: int = cv2.IMREAD_COLOR, workers: int | None = None
) -> list[np.ndarray | None]:
    """Read many images with threads, in the order given.

    ``cv.imread`` releases the GIL, so decoding a folder of JPEGs is several times faster with a thread pool; the
    result lines up with ``filenames`` like :func:`imreadAll`.

    Args:
        filenames: The paths to read.
        flags: ``cv.IMREAD_*`` flags, as for ``cv.imread``.
        workers: Thread count; ``None`` uses the ``ThreadPoolExecutor`` default.

    Returns:
        The images in the order of ``filenames``; ``None`` for files that cannot be read (like ``cv.imread``).

    Raises:
        ValueError: If ``workers`` is given and less than 1.
    """
    paths = [os.fspath(p) for p in filenames]
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: imreadEx(p, flags), paths))


def processImages(
    filenames: Iterable[PathLike],
    fn: Callable[[np.ndarray, Path], T],
    *,
    outDir: PathLike | None = None,
    workers: int | None = None,
    ext: str | None = None,
    skipExisting: bool = False,
    flags: int = cv2.IMREAD_COLOR,
) -> list[T | None]:
    """Read each image, apply ``fn(image, path)`` and optionally write the result, with threads.

    The read / process / write loop over a folder is rewritten in every project. With ``outDir`` the result of ``fn``
    (an image) is written under the same file name (``ext`` replaces the extension) with :func:`imwriteEx`; without
    it the results are only returned.

    Args:
        filenames: The paths to process.
        fn: ``fn(image, path) -> result``; the result must be an image when ``outDir`` is given.
        outDir: Folder to write the results to (created if missing).
        workers: Thread count; ``None`` uses the ``ThreadPoolExecutor`` default.
        ext: Output extension such as ``".png"``, with or without the dot; ``None`` keeps each file's extension.
        skipExisting: Skip files whose output already exists (their result is ``None``). Only with ``outDir``.
        flags: ``cv.IMREAD_*`` flags for reading.

    Returns:
        The results of ``fn`` in the order of ``filenames``; ``None`` for unreadable or skipped files.

    Raises:
        ValueError: If ``workers`` is given and less than 1.
        OSError: If a result cannot be written to ``outDir``; the message names the file.
    """
    paths = [Path(os.fspath(p)) for p in filenames]
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    out = Path(os.fspath(outDir)) if outDir is not None else None
    suffix = None if ext is None else ("." + ext.lstrip("."))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    def target_of(path: Path) -> Path:
        assert out is not None
        return out / (path.stem + (suffix or path.suffix))

    def work(path: Path) -> T | None:
        target = None
        if out is not None:
            target = target_of(path)
            if skipExisting and target.exists():
                return None
        image = imreadEx(path, flags)
        if image is None:
            return None
        result = fn(image, path)
        if target is not None and not imwriteEx(target, cast(np.ndarray, result), makeDirs=True):
            raise OSError(f"cannot write image: {str(target)!r}")
        return result

    if not paths:
        return []
    if out is not None:
        _check_unique_targets([target_of(p) for p in paths])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, paths))


def _check_unique_targets(targets: Sequence[Path]) -> None:
    """Two inputs with the same file name in different folders would overwrite each other's output."""
    seen: set[Path] = set()
    for target in targets:
        if target in seen:
            raise ValueError(f"two input files map to the same output file: {str(target)!r}")
        seen.add(target)


def convertImages(
    source: PathLike | Iterable[PathLike],
    outDir: PathLike,
    ext: str,
    *,
    size: tuple[int, int] | None = None,
    quality: int | None = None,
    recursive: bool = False,
    workers: int | None = None,
) -> list[Path]:
    """Convert a folder (or list) of images to another format, optionally resizing them.

    Builds on :func:`convertImageFormat` (alpha flattening and depth scaling for the target format), so a folder of
    transparent PNGs becomes correct JPEGs. The folder structure is kept under ``outDir`` with ``recursive``.

    Args:
        source: A folder, a glob pattern or an iterable of paths.
        outDir: Folder to write to (created if missing).
        ext: Target extension such as ``"jpg"`` or ``".webp"``.
        size: Resize every image to ``(width, height)`` before writing.
        quality: JPEG / WebP / AVIF quality (0-100).
        recursive: Also convert the images in subfolders, keeping the relative paths. Only for a folder ``source``.
        workers: Thread count; ``None`` uses the ``ThreadPoolExecutor`` default.

    Returns:
        The written paths, in natural order of the sources. Files that could not be read are skipped.

    Raises:
        ValueError: If ``ext`` is empty, ``size`` is not a positive ``(width, height)`` or ``workers`` is less than 1.
    """
    suffix = "." + ext.lstrip(".").lower()
    if suffix == ".":
        raise ValueError("ext must be a file extension such as 'jpg', got ''")
    if size is not None:
        size = check_size("size", size)
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    out = Path(os.fspath(outDir))
    root: Path | None = None
    if isinstance(source, (str, os.PathLike)) and Path(os.fspath(source)).is_dir():
        root = Path(os.fspath(source))
        if recursive:
            files = [f for f in root.rglob("*") if f.suffix.lower() in _IMAGE_EXTENSIONS and f.is_file()]
            paths = sorted(files, key=_natural_key)
        else:
            paths = _resolve_paths(root)
    else:
        paths = _resolve_paths(source)

    def target_of(path: Path) -> Path:
        relative = path.relative_to(root) if root is not None else Path(path.name)
        return out / relative.with_suffix(suffix)

    def work(path: Path) -> Path | None:
        target = target_of(path)
        if size is None:
            ok = convertImageFormat(path, target, quality, makeDirs=True)
        else:
            image = imreadOriented(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                return None
            interpolation = cv2.INTER_AREA if size[0] * size[1] < image.shape[0] * image.shape[1] else cv2.INTER_LINEAR
            resized = cv2.resize(image, size, interpolation=interpolation)
            prepared, params = _prepare_for_format(resized, suffix, quality, (255, 255, 255))
            ok = imwriteEx(target, prepared, params, makeDirs=True)
        return target if ok else None

    if not paths:
        return []
    _check_unique_targets([target_of(p) for p in paths])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [t for t in pool.map(work, paths) if t is not None]


@lru_cache(maxsize=64)
def _cached_read(path: str, flags: int, _mtime_ns: int) -> np.ndarray | None:
    return imreadEx(path, flags)


def imreadCached(filename: PathLike, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image through a process-wide cache of the last 64 decoded files.

    Reading the same reference image, template or lookup table repeatedly (e.g. per frame) decodes it every time;
    OpenCV has no cache. The cache key is the path, the flags and the file's modification time, so an updated file
    is re-read. The returned array is a **copy**, so modifying it does not corrupt the cached image.

    Args:
        filename: Path to the image file.
        flags: ``cv.IMREAD_*`` flags, as for ``cv.imread``.

    Returns:
        A copy of the image, or ``None`` if the file does not exist or cannot be decoded (not cached).
    """
    path = os.fspath(filename)
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return None
    image = _cached_read(path, flags, mtime)
    return None if image is None else image.copy()


def clearImageCache() -> None:
    """Empty the cache used by :func:`imreadCached`."""
    _cached_read.cache_clear()


def stackImages(images: Iterable[np.ndarray], *, size: tuple[int, int] | None = None, pad: bool = False) -> np.ndarray:
    """Stack images into one ``(N, H, W[, C])`` batch array, resizing or padding them to a common size.

    ``np.stack`` fails as soon as two images differ in size; model inputs and batched operations need one array.
    With ``size`` every image is resized to ``(width, height)``; with ``pad`` every image is padded with zeros at the
    bottom and right to the largest width and height.

    Args:
        images: Images with the same number of channels and dtype.
        size: Resize every image to ``(width, height)``.
        pad: Pad every image to the largest size instead of resizing.

    Returns:
        The batch array with the images along the first axis.

    Raises:
        ValueError: If ``images`` is empty, an image is empty, the channel counts or dtypes differ, both ``size`` and
            ``pad`` are given, or the sizes differ while neither ``size`` nor ``pad`` is given.
    """
    items = [np.asarray(img) for img in images]
    if not items:
        raise ValueError("images must not be empty")
    if size is not None and pad:
        raise ValueError("size and pad are mutually exclusive")
    for i, img in enumerate(items):
        check_image_not_empty(f"images[{i}]", img)
        if img.ndim not in (2, 3):
            raise ValueError(f"images[{i}] must be 2-D or 3-D, got shape {img.shape}")
    first = items[0]
    channels = 1 if first.ndim == 2 else first.shape[2]
    for i, img in enumerate(items):
        if (1 if img.ndim == 2 else img.shape[2]) != channels or img.ndim != first.ndim:
            raise ValueError(f"images[{i}] has shape {img.shape}, expected the layout of images[0] {first.shape}")
        if img.dtype != first.dtype:
            raise ValueError(f"images[{i}] has dtype {img.dtype}, expected {first.dtype}")
    if size is not None:
        width, height = check_size("size", size)
        items = [_resize_keep_dims(img, (width, height)) for img in items]
    elif pad:
        height = max(img.shape[0] for img in items)
        width = max(img.shape[1] for img in items)
        items = [
            np.pad(img, [(0, height - img.shape[0]), (0, width - img.shape[1])] + [(0, 0)] * (img.ndim - 2))
            for img in items
        ]
    elif any(img.shape != first.shape for img in items):
        shapes = sorted({img.shape[:2] for img in items})
        raise ValueError(f"images have different sizes {shapes}; pass size=... or pad=True")
    return np.stack(items)


def _resize_keep_dims(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if img.shape[1::-1] == size:
        return img
    resized: np.ndarray = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return resized.reshape(size[1], size[0], *img.shape[2:])


def unstackImages(batch: np.ndarray) -> list[np.ndarray]:
    """Split an ``(N, H, W[, C])`` batch array back into a list of images (the inverse of :func:`stackImages`).

    Args:
        batch: Batch array with the images along the first axis.

    Returns:
        The images (views into ``batch``).

    Raises:
        ValueError: If ``batch`` is not 3-D or 4-D.
    """
    batch = np.asarray(batch)
    if batch.ndim not in (3, 4):
        raise ValueError(f"batch must be (N, H, W) or (N, H, W, C), got shape {batch.shape}")
    return list(batch)
