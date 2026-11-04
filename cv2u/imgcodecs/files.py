import os
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

__all__ = ["imreadEx", "imwriteEx"]


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
