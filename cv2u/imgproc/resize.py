from collections.abc import Sequence

import cv2
import numpy as np

__all__ = ["letterbox", "resizeEx"]


def _default_interpolation(scale: float) -> int:
    # OpenCV's own recommendation: INTER_AREA to shrink, INTER_LINEAR (or CUBIC) to enlarge.
    return cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR


def resizeEx(
    src: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    interpolation: int | None = None,
) -> np.ndarray:
    """Resize an image, keeping the aspect ratio when only ``width`` or ``height`` is given.

    ``cv.resize`` needs the full target size, so resizing "to 640 pixels wide" means computing the height yourself.

    Args:
        src: Image to resize.
        width: Target width in pixels.
        height: Target height in pixels. If both are given, the image is resized to exactly ``(width, height)``.
        interpolation: ``cv.INTER_*`` flag. By default ``cv.INTER_AREA`` when shrinking and ``cv.INTER_LINEAR``
            when enlarging, as recommended by OpenCV.

    Raises:
        ValueError: If neither ``width`` nor ``height`` is given, or a size is not positive.
    """
    if width is None and height is None:
        raise ValueError("give width, height or both")
    for name, value in (("width", width), ("height", height)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    h, w = src.shape[:2]
    if width is not None and height is not None:
        size = (width, height)
        scale = min(width / w, height / h)
    elif width is not None:
        scale = width / w
        size = (width, max(1, round(h * scale)))
    else:
        assert height is not None
        scale = height / h
        size = (max(1, round(w * scale)), height)

    flag = _default_interpolation(scale) if interpolation is None else interpolation
    dst: np.ndarray = cv2.resize(src, size, interpolation=flag)
    return dst


def letterbox(
    src: np.ndarray,
    size: tuple[int, int],
    color: float | Sequence[float] = 0,
    interpolation: int | None = None,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Fit an image into ``size`` keeping its aspect ratio, and pad the rest with ``color``.

    Common as preprocessing for neural networks with a fixed input size.

    Args:
        src: Image to fit.
        size: Output size as ``(width, height)``, like ``cv.resize``.
        color: Padding color (BGR for color images).
        interpolation: ``cv.INTER_*`` flag; chosen like in :func:`resizeEx` by default.

    Returns:
        ``(dst, scale, (left, top))``. A point ``(x, y)`` in ``src`` is at ``(x * scale + left, y * scale + top)``
        in ``dst``.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    out_w, out_h = size
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"size must be positive, got {size}")
    h, w = src.shape[:2]
    scale = min(out_w / w, out_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    flag = _default_interpolation(scale) if interpolation is None else interpolation
    resized = cv2.resize(src, (new_w, new_h), interpolation=flag)

    left, top = (out_w - new_w) // 2, (out_h - new_h) // 2
    right, bottom = out_w - new_w - left, out_h - new_h - top
    value = [float(color)] * 4 if isinstance(color, (int, float)) else [float(c) for c in color]
    dst: np.ndarray = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
    return dst, scale, (left, top)
