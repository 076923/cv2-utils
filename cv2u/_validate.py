"""Argument checks shared by the extensions: one wording for every function."""

import math
import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "as_color",
    "check_channels",
    "check_color",
    "check_finite",
    "check_image_not_empty",
    "check_size",
    "check_uint8",
    "is_single_color",
    "to_uint8",
]


def check_finite(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> None:
    """Raise ``ValueError`` unless ``value`` is a finite number (and positive / non-negative when asked).

    ``NaN`` passes ``< 0`` comparisons, so a plain sign check lets it through to fail later with an unrelated
    message deep inside NumPy or OpenCV.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must not be negative, got {value}")


def check_color(name: str, values: list[float], dtype: np.dtype[Any] | None = None) -> None:
    """Raise ``ValueError`` unless every component of a color is finite and, for an integer ``dtype``, within its range.

    NumPy raises ``OverflowError`` for ``np.array([-1], np.uint8)``, which says nothing about which argument was wrong.
    """
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must be finite numbers, got {values}")
    if dtype is not None and np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if not all(info.min <= v <= info.max for v in values):
            raise ValueError(
                f"{name} must be within [{info.min}, {info.max}] for {np.dtype(dtype).name} images, got {values}"
            )


def check_image_not_empty(name: str, image: np.ndarray) -> None:
    """Raise ``ValueError`` for an image without pixels (OpenCV would fail with an assertion from its C++ code)."""
    if image.size == 0:
        raise ValueError(f"{name} must not be empty, got shape {image.shape}")


def check_channels(name: str, image: np.ndarray) -> int:
    """Return the channel count (1, 3 or 4) of a grayscale, BGR or BGRA image, or raise ``ValueError``.

    Shared by every function that accepts "any image": a 2-D array counts as one channel, and anything else
    (a 1-D array, a 2-channel image, a batch) is rejected with a message naming the argument.
    """
    channels = 1 if image.ndim == 2 else int(image.shape[2]) if image.ndim == 3 else 0
    if channels not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3 or 4 channels, got shape {image.shape}")
    return channels


def check_uint8(name: str, image: np.ndarray) -> int:
    """Return the channel count of a non-empty 8-bit grayscale, BGR or BGRA image, or raise ``ValueError``.

    OpenCV's drawing and document functions accept only ``uint8`` and fail (or draw garbage) on other depths.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must be an 8-bit (uint8) image, got dtype {image.dtype}")
    check_image_not_empty(name, image)
    return check_channels(name, image)


def check_size(name: str, size: Sequence[float]) -> tuple[int, int]:
    """Return a ``(width, height)`` size as positive ints, or raise ``ValueError``.

    ``NaN``, infinities, zeros and negative values would surface later as an OpenCV assertion or an empty array.
    """
    if len(size) != 2:
        raise ValueError(f"{name} must be (width, height), got {size!r}")
    width, height = float(size[0]), float(size[1])
    if not (math.isfinite(width) and math.isfinite(height)) or width < 1 or height < 1:
        raise ValueError(f"{name} must be a positive (width, height) size, got {size!r}")
    return int(width), int(height)


def is_single_color(value: object) -> bool:
    """Whether ``value`` is one color (a number, or 1, 3 or 4 numbers) rather than a sequence of colors.

    Accepts Python and NumPy scalars and 1-D sequences or arrays of numbers, so a row of ``Clusters.centers``
    (``np.array([0, 0, 255], np.uint8)``) counts as one color and is not mistaken for a palette of three. Two
    numbers are never one color (no image has 2 channels): ``[200, 100]`` is two gray levels.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (numbers.Real, np.number)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 1 and value.size in (1, 3, 4) and np.issubdtype(value.dtype, np.number)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return False
    return len(value) in (1, 3, 4) and all(
        isinstance(v, (numbers.Real, np.number)) and not isinstance(v, (bool, np.bool_)) for v in value
    )


def as_color(value: Any) -> float | tuple[float, ...]:
    """One color as a Python ``float`` or tuple of floats, so OpenCV accepts it.

    OpenCV rejects NumPy integer scalars and ``uint8`` arrays as colors ("Scalar value for argument 'color' is not
    numeric"); a row of ``Clusters.centers`` or a ``np.uint8`` level has to be converted first.
    """
    if isinstance(value, (np.ndarray, np.generic)):
        value = value.tolist()
    if isinstance(value, numbers.Real):
        return float(value)
    return tuple(float(v) for v in value)


def to_uint8(src: np.ndarray) -> np.ndarray:
    """8-bit version of an image for formats and functions that only take ``uint8``, scaled by the dtype range.

    Integer images are divided by their dtype peak (``65535`` for 16-bit). Float images are assumed to be in
    ``[0, 1]``; when their maximum exceeds 1 they are scaled by that maximum instead. A plain ``astype(np.uint8)``
    wraps 16-bit values around and ``cv.imwrite`` saturates them to white.
    """
    if src.dtype == np.uint8:
        return src
    values = src.astype(np.float64)
    if np.issubdtype(src.dtype, np.integer):
        peak = float(np.iinfo(src.dtype).max)
    else:
        finite = values[np.isfinite(values)]
        top = float(np.max(finite)) if finite.size else 1.0
        peak = top if top > 1.0 else 1.0
    scaled = np.clip(np.nan_to_num(values) * (255.0 / peak), 0, 255)
    out: np.ndarray = np.rint(scaled).astype(np.uint8)
    return out
