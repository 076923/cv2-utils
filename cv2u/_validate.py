"""Argument checks shared by the extensions: one wording for every function."""

import math
from typing import Any

import numpy as np

__all__ = ["check_color", "check_finite", "check_image_not_empty"]


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
