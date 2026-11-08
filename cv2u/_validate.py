"""Argument checks shared by the extensions: one wording for every function."""

import math
from typing import Any

import numpy as np

__all__ = ["check_channels", "check_color", "check_finite", "check_image_not_empty"]


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
