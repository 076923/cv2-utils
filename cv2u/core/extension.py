import math
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .._validate import check_finite

__all__ = [
    "DHEdge",
    "cartesianToPolar",
    "centerPoint",
    "gammaCorrection",
    "isInside",
    "polarToCartesian",
]


def gammaCorrection(src: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction to an 8-bit image using a lookup table.

    Each value ``v`` becomes ``255 * (v / 255) ** (1 / gamma)``: ``gamma > 1`` brightens the image,
    ``gamma < 1`` darkens it, and ``gamma == 1`` leaves it unchanged.

    Raises:
        ValueError: If ``src`` is not 8-bit or ``gamma`` is not positive.
    """
    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    check_finite("gamma", gamma, positive=True)
    table = (np.arange(256) / 255.0) ** (1.0 / gamma) * 255
    dst: np.ndarray = cv2.LUT(src, np.clip(table, 0, 255).astype(np.uint8))
    return dst


def centerPoint(contour: np.ndarray) -> tuple[float, float]:
    """Return the centroid ``(x, y)`` of a contour.

    Raises:
        ValueError: If the contour has zero area.
    """
    moment = cv2.moments(contour)
    if moment["m00"] == 0:
        raise ValueError("contour has zero area; centroid is undefined")
    return moment["m10"] / moment["m00"], moment["m01"] / moment["m00"]


def cartesianToPolar(x: ArrayLike, y: ArrayLike, angleInDegrees: bool = False) -> tuple[Any, Any]:
    """Convert Cartesian coordinates to polar coordinates.

    Unlike ``cv.cartToPolar``, this accepts scalars as well as arrays of any shape, returns the angle
    **first** (``(theta, rho)``), and the angle is in ``[-pi, pi]`` (``[-180, 180]`` in degrees)
    instead of ``[0, 2*pi)``.

    Args:
        x: x coordinate(s).
        y: y coordinate(s).
        angleInDegrees: Return the angle in degrees instead of radians (same name as in OpenCV).

    Returns:
        ``(theta, rho)``: the angle and the distance from the origin.
    """
    theta = np.arctan2(y, x)
    return (np.degrees(theta) if angleInDegrees else theta), np.hypot(x, y)


def polarToCartesian(theta: ArrayLike, rho: ArrayLike, angleInDegrees: bool = False) -> tuple[Any, Any]:
    """Convert polar coordinates ``(theta, rho)`` to Cartesian ``(x, y)``.

    The inverse of :func:`cartesianToPolar`. Unlike ``cv.polarToCart``, this accepts scalars as well as
    arrays, and takes the angle **first**.

    Args:
        theta: Angle(s), in radians unless ``angleInDegrees`` is true.
        rho: Distance(s) from the origin.
        angleInDegrees: ``theta`` is in degrees (same name as in OpenCV).
    """
    t = np.radians(theta) if angleInDegrees else np.asarray(theta)
    r = np.asarray(rho)
    return r * np.cos(t), r * np.sin(t)


def isInside(p: Sequence[float], polygon: ArrayLike) -> bool:
    """Return whether point ``p`` lies inside ``polygon`` (even-odd ray casting).

    ``polygon`` may be an OpenCV contour of shape ``(N, 1, 2)`` or an array of shape ``(N, 2)``.
    """
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    x1, y1 = pts.T
    x2, y2 = np.roll(pts, -1, axis=0).T
    px, py = p[0], p[1]

    straddles = (y1 > py) != (y2 > py)
    with np.errstate(divide="ignore", invalid="ignore"):  # horizontal edges never straddle
        at_x = (x2 - x1) * (py - y1) / (y2 - y1) + x1
    crosses = np.count_nonzero(straddles & (px < at_x))
    return bool(crosses % 2 == 1)


_GRADIENT_KERNEL = np.ones((3, 3), np.uint8)
_SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)


def DHEdge(src: np.ndarray) -> np.ndarray:
    """Bold, high-contrast edge map for drawings and logos.

    The 3x3 morphological gradient (``cv.morphologyEx(src, cv.MORPH_GRADIENT, ...)``) is sharpened and then
    contrast-stretched, so edges come out as thick white lines on black.

    Args:
        src: 8-bit image with 1 or 3 channels. Color images are processed per channel.

    Raises:
        ValueError: If ``src`` is not 8-bit.
    """
    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    gradient = cv2.morphologyEx(src, cv2.MORPH_GRADIENT, _GRADIENT_KERNEL)
    sharpen = cv2.filter2D(gradient, -1, _SHARPEN_KERNEL)
    return np.clip(math.pi * sharpen.astype(np.float64) - 255, 0, 255).astype(np.uint8)
