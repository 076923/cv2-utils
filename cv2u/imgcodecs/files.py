import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

__all__ = ["imreadAll", "imreadEx", "imwriteAll", "imwriteEx", "listImages"]


def imreadEx(filename: str | os.PathLike[str], flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image like ``cv.imread``, but from any path, including non-ASCII paths on Windows.

    ``cv.imread`` cannot open paths such as ``C:/사진/이미지.png`` on Windows. This reads the file with
    NumPy and decodes it with ``cv.imdecode``.

    Args:
        filename: Path to the image file.
        flags: ``cv.IMREAD_*`` flags, as for ``cv.imread``.

    Returns:
        The image, or ``None`` if the file does not exist, cannot be read or cannot be decoded (like ``cv.imread``).
    """
    try:
        data = np.fromfile(os.fspath(filename), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwriteEx(
    filename: str | os.PathLike[str], img: np.ndarray, params: Sequence[int] = (), makeDirs: bool = False
) -> bool:
    """Write an image like ``cv.imwrite``, but to any path, including non-ASCII paths on Windows.

    The format is chosen from the file extension, as in ``cv.imwrite``.

    Args:
        filename: Path to write to.
        img: Image to save.
        params: Format-specific ``cv.IMWRITE_*`` parameter pairs, e.g. ``(cv.IMWRITE_JPEG_QUALITY, 90)``.
        makeDirs: Create missing parent directories. ``cv.imwrite`` (and this function by default) just returns
            ``False`` when the directory does not exist.

    Returns:
        ``True`` on success, ``False`` if the image could not be encoded or the file could not be written.

    Raises:
        ValueError: If ``filename`` has no file extension.
        cv2.error: If no encoder exists for the extension (e.g. ``.xyz``), like ``cv.imwrite``.
    """
    path = os.fspath(filename)
    ext = Path(path).suffix
    if not ext:
        raise ValueError(f"cannot determine the image format: {path!r} has no file extension")
    ok, buffer = cv2.imencode(ext, img, list(params))
    if not ok:
        return False
    try:
        if makeDirs:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        buffer.tofile(path)
    except OSError:
        return False
    return True


_IMAGE_EXTENSIONS = frozenset(
    {
        ".avif", ".bmp", ".dib", ".exr", ".gif", ".hdr", ".jp2", ".jpe", ".jpeg", ".jpg", ".pbm", ".pfm", ".pgm",
        ".pic", ".png", ".pnm", ".ppm", ".pxm", ".ras", ".sr", ".tif", ".tiff", ".webp",
    }
)  # fmt: skip


def _natural_key(path: Path) -> list[str | int]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.as_posix())]


def listImages(path: str | os.PathLike[str], recursive: bool = False) -> list[Path]:
    """List the image files in a folder, in natural order (``img2.png`` before ``img10.png``).

    Files are recognized by the extensions OpenCV can read (``.jpg``, ``.png``, ``.webp``, ``.tif``, ... in any
    letter case). Use with ``imreadEx`` to process a folder of images.

    Args:
        path: Folder to list.
        recursive: Also list the images in subfolders.

    Returns:
        The image paths.

    Raises:
        NotADirectoryError: If ``path`` is not a folder.
    """
    folder = Path(path)
    if not folder.is_dir():
        raise NotADirectoryError(f"not a folder: {folder}")
    files = folder.rglob("*") if recursive else folder.iterdir()
    return sorted((f for f in files if f.suffix.lower() in _IMAGE_EXTENSIONS and f.is_file()), key=_natural_key)


def imreadAll(
    filenames: Iterable[str | os.PathLike[str]], flags: int = cv2.IMREAD_COLOR, strict: bool = True
) -> list[np.ndarray | None]:
    """Read many images with :func:`imreadEx`, e.g. a whole folder from :func:`listImages`.

    Args:
        filenames: The paths to read.
        flags: ``cv.IMREAD_*`` flags, as for ``cv.imread``.
        strict: Raise on the first file that cannot be read. With ``False``, unreadable files give ``None`` in the
            result (as ``cv.imread`` would), so the result lines up with ``filenames``.

    Returns:
        The images, in the order of ``filenames``.

    Raises:
        OSError: If ``strict`` and a file does not exist or cannot be decoded; the message names the file.
    """
    images: list[np.ndarray | None] = []
    for filename in filenames:
        image = imreadEx(filename, flags)
        if image is None and strict:
            raise OSError(f"cannot read image: {os.fspath(filename)!r}")
        images.append(image)
    return images


def imwriteAll(
    filenames: Iterable[str | os.PathLike[str]],
    images: Iterable[np.ndarray],
    params: Sequence[int] = (),
    makeDirs: bool = False,
    strict: bool = True,
) -> list[bool]:
    """Write many images with :func:`imwriteEx`, pairing each path with the image at the same position.

    Args:
        filenames: The paths to write to; the format comes from each extension.
        images: The images, one per path.
        params: Format-specific ``cv.IMWRITE_*`` parameter pairs, used for every file.
        makeDirs: Create missing parent directories.
        strict: Raise on the first file that cannot be written. With ``False``, failures give ``False`` in the result.

    Returns:
        One ``True`` / ``False`` per file, in order.

    Raises:
        ValueError: If there are more paths than images or more images than paths, or a path has no extension.
        OSError: If ``strict`` and a file could not be encoded or written; the message names the file.
    """
    paths, imgs = list(filenames), list(images)
    if len(paths) != len(imgs):
        raise ValueError(f"filenames and images must have the same length, got {len(paths)} and {len(imgs)}")
    results = []
    for filename, image in zip(paths, imgs, strict=True):
        ok = imwriteEx(filename, image, params, makeDirs)
        if not ok and strict:
            raise OSError(f"cannot write image: {os.fspath(filename)!r}")
        results.append(ok)
    return results
