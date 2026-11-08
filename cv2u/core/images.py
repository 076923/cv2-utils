"""Create new images (OpenCV only offers ``np.zeros``/``np.full`` style allocation through NumPy)."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import DTypeLike

from .._validate import check_color

__all__ = ["imcolor", "imcolorLike", "makeGradient"]

GradientDirection = Literal["horizontal", "vertical", "raster"]


def _size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    if width < 1 or height < 1:
        raise ValueError(f"size must be at least 1x1, got {size}")
    return int(width), int(height)


def _color_values(name: str, color: float | Sequence[float], channels: int, dtype: DTypeLike) -> np.ndarray:
    values = np.atleast_1d(np.asarray(color, dtype=np.float64)).ravel()
    if values.size not in (1, channels):
        raise ValueError(f"{name} must be a scalar or have {channels} values, got {color!r}")
    check_color(name, values.tolist(), np.dtype(dtype))
    return np.broadcast_to(values, (channels,))


def makeGradient(
    size: tuple[int, int],
    start: float | Sequence[float],
    end: float | Sequence[float],
    direction: GradientDirection = "horizontal",
    channels: int = 3,
) -> np.ndarray:
    """Create an 8-bit image filled with a linear gradient from ``start`` to ``end``.

    Args:
        size: Image size as ``(width, height)``, like ``cv.resize``.
        start: Start value. A scalar, or one value per channel (BGR).
        end: End value. A scalar, or one value per channel (BGR).
        direction: ``"horizontal"`` (left to right), ``"vertical"`` (top to bottom), or
            ``"raster"`` (through every pixel in reading order, row by row).
        channels: ``3`` for a BGR image (default) or ``1`` for a grayscale image of shape ``(height, width)``.

    Returns:
        A ``uint8`` image of shape ``(height, width, 3)``, or ``(height, width)`` for ``channels=1``.

    Raises:
        ValueError: If ``size`` is not positive, ``channels`` is not 1 or 3, a value is outside ``[0, 255]`` or
            does not match the channel count, or ``direction`` is unknown.
    """
    width, height = _size(size)
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels}")
    lo = _color_values("start", start, channels, np.uint8)
    hi = _color_values("end", end, channels, np.uint8)
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
    return img.reshape(height, width) if channels == 1 else img


def imcolor(
    size: tuple[int, int], color: float | Sequence[float], channels: int = 3, dtype: DTypeLike = np.uint8
) -> np.ndarray:
    """Create an image of ``size`` filled with ``color`` (a BGR ``uint8`` image by default).

    Args:
        size: Image size as ``(width, height)``, like ``cv.resize``.
        color: A scalar, or one value per channel (BGR, or BGRA for ``channels=4``).
        channels: ``3`` (BGR, default), ``4`` (BGRA) or ``1`` for a grayscale image of shape ``(height, width)``.
        dtype: Pixel type.

    Raises:
        ValueError: If ``size`` is not positive, ``channels`` is not 1, 3 or 4, or ``color`` is not finite, does not
            match the channel count, or is outside the range of ``dtype``.
    """
    width, height = _size(size)
    if channels not in (1, 3, 4):
        raise ValueError(f"channels must be 1, 3 or 4, got {channels}")
    values = _color_values("color", color, channels, dtype)
    shape = (height, width) if channels == 1 else (height, width, channels)
    return np.full(shape, values[0] if channels == 1 else values, dtype=dtype)


def imcolorLike(src: np.ndarray, color: float | Sequence[float], dtype: DTypeLike | None = None) -> np.ndarray:
    """Create an image with the same shape (and, by default, dtype) as ``src`` filled with ``color``.

    Args:
        src: Image whose shape (and dtype) to copy.
        color: A scalar, or one value per channel of ``src``.
        dtype: Pixel type of the result; by default that of ``src``.

    Raises:
        ValueError: If ``color`` is not finite, does not match the channels of ``src``, or is outside the range of
            the result dtype.
    """
    out_dtype = src.dtype if dtype is None else np.dtype(dtype)
    channels = 1 if src.ndim < 3 else int(src.shape[-1])
    values = _color_values("color", color, channels, out_dtype)
    return np.full_like(src, values[0] if src.ndim < 3 else values, dtype=out_dtype)
