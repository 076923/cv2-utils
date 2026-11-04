"""Create new images (OpenCV only offers ``np.zeros``/``np.full`` style allocation through NumPy)."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import DTypeLike

__all__ = ["imcolor", "imcolorLike", "makeGradient"]

GradientDirection = Literal["horizontal", "vertical", "raster"]


def makeGradient(
    shape: Sequence[int],
    start: float | Sequence[float],
    end: float | Sequence[float],
    direction: GradientDirection = "horizontal",
) -> np.ndarray:
    """Create an 8-bit image filled with a linear gradient from ``start`` to ``end``.

    Args:
        shape: ``(height, width)``, ``(height, width, 1)`` or ``(height, width, 3)``.
        start: Start value. A scalar, or one value per channel (BGR).
        end: End value. A scalar, or one value per channel (BGR).
        direction: ``"horizontal"`` (left to right), ``"vertical"`` (top to bottom), or
            ``"raster"`` (through every pixel in reading order, row by row).

    Raises:
        ValueError: If ``shape`` does not have 1 or 3 channels, or ``direction`` is unknown.
    """
    height, width = shape[:2]
    channels = shape[2] if len(shape) == 3 else 1
    if channels not in (1, 3):
        raise ValueError(f"shape must have 1 or 3 channels, got {channels}")

    def _per_channel(value: float | Sequence[float]) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        return arr[:1] if channels == 1 else np.broadcast_to(arr, (channels,))

    lo, hi = _per_channel(start), _per_channel(end)
    if direction == "raster":
        img = np.linspace(lo, hi, num=height * width, dtype=np.uint8).reshape(height, width, channels)
    elif direction == "horizontal":
        row = np.linspace(lo, hi, num=width, dtype=np.uint8)
        img = np.broadcast_to(row, (height, width, channels)).copy()
    elif direction == "vertical":
        column = np.linspace(lo, hi, num=height, dtype=np.uint8)
        img = np.broadcast_to(column[:, None, :], (height, width, channels)).copy()
    else:
        raise ValueError(f"direction must be 'horizontal', 'vertical' or 'raster', got {direction!r}")
    return img.reshape(tuple(shape))


def imcolor(shape: Sequence[int], color: float | Sequence[float], dtype: DTypeLike = np.uint8) -> np.ndarray:
    """Create an image of ``shape`` filled with ``color`` (8-bit by default)."""
    return np.full(shape, color, dtype=dtype)


def imcolorLike(src: np.ndarray, color: float | Sequence[float], dtype: DTypeLike | None = None) -> np.ndarray:
    """Create an image with the same shape (and, by default, dtype) as ``src`` filled with ``color``."""
    return np.full_like(src, color, dtype=dtype)
