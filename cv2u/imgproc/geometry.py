import math

import cv2
import numpy as np
from numpy.typing import ArrayLike

__all__ = ["fourPointTransform", "translate"]


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left (clockwise on screen)."""
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    clockwise = pts[np.argsort(angles)]  # image y points down, so increasing angle is clockwise
    first = int(np.argmin(clockwise.sum(axis=1)))  # top-left has the smallest x + y
    return np.roll(clockwise, -first, axis=0)


def fourPointTransform(
    src: np.ndarray,
    pts: ArrayLike,
    size: tuple[int, int] | None = None,
    flags: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Warp the quadrilateral ``pts`` in ``src`` to an upright rectangle (e.g. a photographed document).

    The four points may be given in any order. Without ``size``, the output size follows the longest edges of the
    quadrilateral, so the content keeps its proportions.

    Args:
        src: Image to warp.
        pts: The 4 corners, as an array of shape ``(4, 2)`` or an OpenCV contour of shape ``(4, 1, 2)``,
            e.g. from ``cv.approxPolyDP``.
        size: Output size as ``(width, height)``.
        flags: Interpolation, as for ``cv.warpPerspective``.

    Returns:
        The warped image.

    Raises:
        ValueError: If ``pts`` does not hold exactly 4 points, or the points are degenerate.
    """
    corners = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if corners.shape != (4, 2):
        raise ValueError(f"pts must contain exactly 4 points, got {corners.shape[0]}")
    tl, tr, br, bl = _order_corners(corners)

    if size is None:
        width = round(max(math.dist(tl, tr), math.dist(bl, br)))
        height = round(max(math.dist(tl, bl), math.dist(tr, br)))
    else:
        width, height = size
    if width < 1 or height < 1:
        raise ValueError(f"the points span no area (output size {width}x{height})")

    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), target)
    dst: np.ndarray = cv2.warpPerspective(src, matrix, (width, height), flags=flags)
    return dst


def translate(
    src: np.ndarray,
    dx: float,
    dy: float,
    borderMode: int = cv2.BORDER_CONSTANT,
    borderValue: float | tuple[float, ...] = 0,
) -> np.ndarray:
    """Shift an image by ``(dx, dy)`` pixels, keeping its size (OpenCV has no function for this).

    Args:
        src: Image to shift.
        dx: Shift to the right (negative: to the left). Fractions are interpolated.
        dy: Shift down (negative: up).
        borderMode: How to fill the uncovered area, as for ``cv.warpAffine`` (e.g. ``cv.BORDER_REPLICATE``).
        borderValue: Fill color for ``cv.BORDER_CONSTANT``.
    """
    h, w = src.shape[:2]
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)
    dst: np.ndarray = cv2.warpAffine(src, matrix, (w, h), borderMode=borderMode, borderValue=borderValue)
    return dst
