import math
from typing import Any

import cv2
import numpy as np
from numpy.typing import ArrayLike

from ..core.rotate import rotateEx
from .color import toGray

__all__ = ["cropRotatedRect", "deskew", "fourPointTransform", "transformPoints", "translate"]


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
    if not np.all(np.isfinite(corners)):
        raise ValueError("pts must be finite")
    tl, tr, br, bl = _order_corners(corners)
    # Shoelace area of the ordered quad: 0 for collinear or duplicate points, which have no perspective transform.
    quad = np.array([tl, tr, br, bl], dtype=np.float64)
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < 1e-3 or len(np.unique(quad, axis=0)) < 4:
        raise ValueError("pts are degenerate: duplicate or collinear points do not form a quadrilateral")

    if size is None:
        width = round(max(math.dist(tl, tr), math.dist(bl, br)))
        height = round(max(math.dist(tl, bl), math.dist(tr, br)))
    else:
        width, height = size
        if width < 1 or height < 1:
            raise ValueError(f"size must be at least 1x1, got {width}x{height}")
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


def transformPoints(points: ArrayLike, M: ArrayLike) -> np.ndarray:
    """Apply a 2x3 affine or a 3x3 perspective matrix to points.

    ``cv.transform`` and ``cv.perspectiveTransform`` need float arrays of shape ``(N, 1, 2)`` and a different
    function for each kind of matrix. This takes points in any common layout and either matrix, e.g. to map
    keypoints or box corners through the matrix used by ``cv.warpAffine`` or ``cv.warpPerspective``.

    Args:
        points: One point ``(x, y)``, an array of shape ``(N, 2)`` or a contour of shape ``(N, 1, 2)``.
        M: A 2x3 affine matrix (``cv.getRotationMatrix2D``, ``cv.getAffineTransform``) or a 3x3 perspective
            matrix (``cv.getPerspectiveTransform``, ``cv.findHomography``).

    Returns:
        The transformed points as float64, in the same shape as ``points``.

    Raises:
        ValueError: If the points do not have 2 coordinates or ``M`` is neither 2x3 nor 3x3.
    """
    pts = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(M, dtype=np.float64)
    if pts.ndim == 0 or pts.shape[-1] != 2:
        raise ValueError(f"points must have 2 coordinates (x, y), got shape {pts.shape}")
    flat = pts.reshape(-1, 2)
    if matrix.shape == (2, 3):
        out = flat @ matrix[:, :2].T + matrix[:, 2]
    elif matrix.shape == (3, 3):
        homogeneous = flat @ matrix[:, :2].T + matrix[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = homogeneous[:, :2] / homogeneous[:, 2:]
    else:
        raise ValueError(f"M must be a 2x3 affine or a 3x3 perspective matrix, got shape {matrix.shape}")
    result: np.ndarray = out.reshape(pts.shape)
    return result


def cropRotatedRect(
    src: np.ndarray,
    rect: tuple[tuple[float, float], tuple[float, float], float],
    flags: int = cv2.INTER_LINEAR,
    borderMode: int = cv2.BORDER_REPLICATE,
    borderValue: float | tuple[float, ...] = 0,
) -> np.ndarray:
    """Cut out a rotated rectangle as an upright image (e.g. a tilted label or license plate).

    OpenCV describes rotated boxes as ``((cx, cy), (width, height), angle)`` (``cv.minAreaRect``,
    ``cv.RotatedRect``) but has no function to extract one.

    Args:
        src: Image to crop from.
        rect: The rotated box, as returned by ``cv.minAreaRect``.
        flags: Interpolation, as for ``cv.warpAffine``.
        borderMode: How to fill the parts of the box outside the image, as for ``cv.warpAffine``.
        borderValue: Fill color for ``cv.BORDER_CONSTANT``.

    Returns:
        An image of size ``(round(width), round(height))`` whose x axis runs along the box's width side.

    Raises:
        ValueError: If the box is smaller than one pixel.
    """
    (cx, cy), (w, h), angle = rect
    width, height = round(w), round(h)
    if width < 1 or height < 1:
        raise ValueError(f"rect must be at least 1x1 pixel, got {w}x{h}")
    matrix = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle), 1.0)
    matrix[0, 2] += (width - 1) / 2 - cx
    matrix[1, 2] += (height - 1) / 2 - cy
    dst: np.ndarray = cv2.warpAffine(
        src, matrix, (width, height), flags=flags, borderMode=borderMode, borderValue=borderValue
    )
    return dst


def _ink(src: np.ndarray) -> np.ndarray:
    """Foreground (text) of a page as a binary image: the minority class of an Otsu threshold."""
    gray = toGray(src)
    if gray.dtype != np.uint8:
        values = gray.astype(np.float64)
        span = float(np.ptp(values)) or 1.0
        gray = np.rint((values - float(np.min(values))) / span * 255).astype(np.uint8)
    scale = 600 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if np.count_nonzero(binary) > binary.size / 2:  # light text on a dark background
        binary = cv2.bitwise_not(binary)
    return binary


def _line_score(binary: np.ndarray, angle: float) -> float:
    """How strongly the ink concentrates in rows after rotating by ``angle`` (peaks when text lines are level)."""
    rows = rotateEx(binary, angle, flags=cv2.INTER_NEAREST, borderValue=0).sum(axis=1, dtype=np.float64)
    return float(np.dot(rows, rows))


def deskew(
    src: np.ndarray,
    maxAngle: float = 45.0,
    borderMode: int = cv2.BORDER_CONSTANT,
    borderValue: Any = None,
) -> tuple[np.ndarray, float]:
    """Straighten a scanned or photographed page whose text lines are tilted.

    The angle is found from the text itself: the image is rotated until the ink is concentrated in as few rows as
    possible (a projection-profile search, first in 1° and then in 0.1° steps). Works for dark text on a light
    background and the other way around. The page is rotated with ``rotateEx``, so no corner is cut off.

    Args:
        src: Image of a page (grayscale, BGR or BGRA).
        maxAngle: Largest tilt to look for, in degrees.
        borderMode: How to fill the corners uncovered by the rotation, as for ``cv.warpAffine``.
        borderValue: Fill color for ``cv.BORDER_CONSTANT``. By default the median color of the image border,
            so the new corners match the paper.

    Returns:
        ``(dst, angle)``: the straightened image and the rotation applied to it, in degrees
        (positive is counter-clockwise, as for ``cv.getRotationMatrix2D``). ``angle`` is 0 for an empty page.

    Raises:
        ValueError: If ``maxAngle`` is negative or not below 90.
    """
    if not 0 <= maxAngle < 90:
        raise ValueError(f"maxAngle must be in [0, 90), got {maxAngle}")
    binary = _ink(src)
    angle = 0.0
    if np.count_nonzero(binary) and maxAngle > 0:
        limit = math.floor(maxAngle)
        coarse = [float(a) for a in range(-limit, limit + 1)]
        best = max(coarse, key=lambda a: _line_score(binary, a))
        fine = {min(max(round(best + step / 10, 1), -maxAngle), maxAngle) for step in range(-10, 11)}
        angle = max(sorted(fine), key=lambda a: _line_score(binary, a))
    if borderValue is None and borderMode == cv2.BORDER_CONSTANT:
        edge = np.concatenate([src[0], src[-1], src[:, 0], src[:, -1]]).reshape(-1, *src.shape[2:3])
        borderValue = tuple(float(v) for v in np.median(edge, axis=0).ravel())
    return rotateEx(src, angle, borderMode=borderMode, borderValue=borderValue), angle
