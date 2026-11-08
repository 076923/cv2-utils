"""2-D point sets: transform, normalize, clip, measure and order ``(x, y)`` points.

Keypoints, landmarks and polygon corners go through the same arithmetic in every project: rotate them with the
image, map them to ``[0, 1]`` and back, flip them with ``cv.flip``, find the nearest matches. OpenCV transforms
points only through matrices (``cv.transform``, ``cv.perspectiveTransform``) and has no distance matrix, ordering
or line intersection. Points are ``(x, y)`` pairs: one point, an array of shape ``(N, 2)`` or an OpenCV contour of
shape ``(N, 1, 2)``; results come back as float64 in the same shape.
"""

import math

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .._validate import check_finite
from ..imgproc.geometry import _order_corners

__all__ = [
    "angleBetween",
    "clipPoints",
    "denormalizePoints",
    "distancePoints",
    "flipPoints",
    "lineIntersection",
    "nearestPoints",
    "normalizePoints",
    "orderPoints",
    "pointLineDistance",
    "pointsBoundingBox",
    "polygonCentroid",
    "rotatePoints",
    "scalePoints",
    "translatePoints",
]


def _points(points: ArrayLike, name: str = "points") -> tuple[np.ndarray, tuple[int, ...]]:
    """Validate points and return them as a float64 ``(N, 2)`` array plus the shape to give the result."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 2:
        raise ValueError(f"{name} must be (x, y) points: shape (2,), (N, 2) or (N, 1, 2), got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.reshape(-1, 2), array.shape


def _size(size: tuple[int, int]) -> tuple[float, float]:
    width, height = float(size[0]), float(size[1])
    if not (math.isfinite(width) and math.isfinite(height)) or width <= 0 or height <= 0:
        raise ValueError(f"size must be positive (width, height), got {size}")
    return width, height


def _point(point: ArrayLike, name: str) -> tuple[float, float]:
    pts, shape = _points(point, name)
    if pts.shape[0] != 1:
        raise ValueError(f"{name} must be one (x, y) point, got shape {shape}")
    return float(pts[0, 0]), float(pts[0, 1])


def rotatePoints(
    points: ArrayLike, angle: float, center: tuple[float, float] = (0, 0), scale: float = 1.0
) -> np.ndarray:
    """Rotate points around ``center`` by ``angle`` degrees, with the convention of ``cv.getRotationMatrix2D``.

    Positive angles turn counter-clockwise on screen (image y points down), so points rotated with the same
    ``angle`` and ``center`` as ``cv.warpAffine(img, cv.getRotationMatrix2D(center, angle, scale), ...)`` land on
    the rotated image.

    Args:
        points: One point, ``(N, 2)`` or ``(N, 1, 2)``.
        angle: Rotation in degrees.
        center: Center of rotation ``(x, y)``; the origin by default.
        scale: Isotropic scale applied with the rotation.

    Returns:
        The rotated points as float64, in the shape of ``points``.

    Raises:
        ValueError: If a value is not finite or ``scale`` is not positive.
    """
    pts, shape = _points(points)
    check_finite("angle", angle)
    check_finite("scale", scale, positive=True)
    cx, cy = _point(center, "center")
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    out: np.ndarray = pts @ matrix[:, :2].T + matrix[:, 2]
    return out.reshape(shape)


def scalePoints(
    points: ArrayLike, scale: float | tuple[float, float], center: tuple[float, float] = (0, 0)
) -> np.ndarray:
    """Scale points about ``center``, e.g. to map keypoints onto a resized image.

    Args:
        points: One point, ``(N, 2)`` or ``(N, 1, 2)``.
        scale: One factor, or ``(sx, sy)`` for different factors per axis (``new_w / old_w, new_h / old_h``).
        center: Fixed point ``(x, y)``; the origin by default.

    Returns:
        The scaled points as float64, in the shape of ``points``.

    Raises:
        ValueError: If a value is not finite.
    """
    pts, shape = _points(points)
    sx, sy = (float(scale), float(scale)) if isinstance(scale, (int, float)) else (float(scale[0]), float(scale[1]))
    check_finite("scale", sx)
    check_finite("scale", sy)
    c = np.array(_point(center, "center"))
    out: np.ndarray = (pts - c) * (sx, sy) + c
    return out.reshape(shape)


def translatePoints(points: ArrayLike, dx: float, dy: float) -> np.ndarray:
    """Shift points by ``(dx, dy)``, e.g. to move keypoints from a crop back into the full image.

    Returns:
        The shifted points as float64, in the shape of ``points``.

    Raises:
        ValueError: If a value is not finite.
    """
    pts, shape = _points(points)
    check_finite("dx", dx)
    check_finite("dy", dy)
    out: np.ndarray = pts + np.array([dx, dy])
    return out.reshape(shape)


def flipPoints(points: ArrayLike, size: tuple[int, int], flipCode: int) -> np.ndarray:
    """Flip points like ``cv.flip`` flips an image of ``size``, so landmarks stay on the flipped image.

    Pixel ``x`` maps to ``width - 1 - x`` (and ``y`` to ``height - 1 - y``), the pixel-center convention of
    ``cv.flip``. Note that a horizontal flip swaps left / right landmarks (e.g. the eyes) in meaning, which is up
    to the caller.

    Args:
        points: One point, ``(N, 2)`` or ``(N, 1, 2)``.
        size: Image size as ``(width, height)``.
        flipCode: As for ``cv.flip``: ``0`` vertical, positive horizontal, negative both.

    Returns:
        The flipped points as float64, in the shape of ``points``.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    pts, shape = _points(points)
    width, height = _size(size)
    out = pts.copy()
    if flipCode != 0:
        out[:, 0] = width - 1 - out[:, 0]
    if flipCode <= 0:
        out[:, 1] = height - 1 - out[:, 1]
    return out.reshape(shape)


def normalizePoints(points: ArrayLike, size: tuple[int, int]) -> np.ndarray:
    """Divide pixel coordinates by the image size, giving ``[0, 1]`` coordinates independent of the resolution.

    Args:
        points: One point, ``(N, 2)`` or ``(N, 1, 2)`` in pixels.
        size: Image size as ``(width, height)``.

    Returns:
        The normalized points as float64, in the shape of ``points``. Points outside the image fall outside ``[0, 1]``.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    pts, shape = _points(points)
    out: np.ndarray = pts / _size(size)
    return out.reshape(shape)


def denormalizePoints(points: ArrayLike, size: tuple[int, int]) -> np.ndarray:
    """Multiply ``[0, 1]`` coordinates by the image size; the inverse of :func:`normalizePoints`.

    Args:
        points: One point, ``(N, 2)`` or ``(N, 1, 2)`` in ``[0, 1]``.
        size: Image size as ``(width, height)``.

    Returns:
        The pixel coordinates as float64, in the shape of ``points``.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    pts, shape = _points(points)
    out: np.ndarray = pts * _size(size)
    return out.reshape(shape)


def clipPoints(points: ArrayLike, size: tuple[int, int]) -> np.ndarray:
    """Clamp points into an image of ``size``: ``x`` to ``[0, width - 1]`` and ``y`` to ``[0, height - 1]``.

    Args:
        points: One point, ``(N, 2)`` or ``(N, 1, 2)``.
        size: Image size as ``(width, height)``.

    Returns:
        The clipped points as float64, in the shape of ``points``.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    pts, shape = _points(points)
    width, height = _size(size)
    out: np.ndarray = np.clip(pts, 0, (width - 1, height - 1))
    return out.reshape(shape)


def pointsBoundingBox(points: ArrayLike) -> tuple[int, int, int, int]:
    """Integer bounding box ``(x, y, width, height)`` of points, like ``cv.boundingRect`` but for float input too.

    The box starts at the floor of the minimum and ends at the ceiling of the maximum, so every point lies on a
    pixel inside it; for integer points this equals ``cv.boundingRect``.

    Raises:
        ValueError: If ``points`` is empty or not finite.
    """
    pts, _ = _points(points)
    if pts.shape[0] == 0:
        raise ValueError("points must not be empty")
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def distancePoints(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Euclidean distance from every point in ``a`` to every point in ``b``.

    Args:
        a: One point, ``(N, 2)`` or ``(N, 1, 2)``.
        b: One point, ``(M, 2)`` or ``(M, 1, 2)``.

    Returns:
        A float64 array of shape ``(N, M)``; ``float(distancePoints(p, q)[0, 0])`` for two single points.
    """
    pa, _ = _points(a, "a")
    pb, _ = _points(b, "b")
    diff = pa[:, None, :] - pb[None, :, :]
    out: np.ndarray = np.sqrt(np.sum(diff * diff, axis=2))
    return out


def nearestPoints(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """For every point in ``a``, the index of the closest point in ``b`` and the distance to it.

    Simple matching of detections between frames or of landmarks to a template.

    Args:
        a: One point, ``(N, 2)`` or ``(N, 1, 2)``.
        b: ``(M, 2)`` or ``(M, 1, 2)``, with at least one point.

    Returns:
        ``(indices, distances)``: an ``int64`` array and a float64 array, both of length ``N``.

    Raises:
        ValueError: If ``b`` is empty.
    """
    dist = distancePoints(a, b)
    if dist.shape[1] == 0:
        raise ValueError("b must not be empty")
    indices = np.argmin(dist, axis=1)
    return indices, dist[np.arange(dist.shape[0]), indices]


def orderPoints(points: ArrayLike) -> np.ndarray:
    """Order 4 corner points as top-left, top-right, bottom-right, bottom-left (clockwise on screen).

    ``cv.approxPolyDP`` and ``cv.minAreaRect`` give the corners of a quadrilateral in an arbitrary starting
    order; ``cv.getPerspectiveTransform`` needs them matched to the output corners. This is the ordering
    :func:`fourPointTransform` uses.

    Args:
        points: 4 points as ``(4, 2)`` or a contour of shape ``(4, 1, 2)``.

    Returns:
        A float64 array of shape ``(4, 2)``.

    Raises:
        ValueError: If there are not exactly 4 points.
    """
    pts, _ = _points(points)
    if pts.shape[0] != 4:
        raise ValueError(f"points must contain exactly 4 points, got {pts.shape[0]}")
    ordered: np.ndarray = _order_corners(pts)
    return ordered


def polygonCentroid(points: ArrayLike) -> tuple[float, float]:
    """Area centroid ``(x, y)`` of a polygon given by its vertices in order, like ``cv.moments`` on a contour.

    For a polygon with no area (fewer than 3 points, or collinear ones) the mean of the points is returned instead
    of failing, so it also works for a line or a single point.

    Raises:
        ValueError: If ``points`` is empty or not finite.
    """
    pts, _ = _points(points)
    if pts.shape[0] == 0:
        raise ValueError("points must not be empty")
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-12:
        return float(x.mean()), float(y.mean())
    return float(((x + x1) * cross).sum() / (6 * area)), float(((y + y1) * cross).sum() / (6 * area))


def angleBetween(p1: ArrayLike, p2: ArrayLike, degrees: bool = True) -> float:
    """Direction of the vector from ``p1`` to ``p2``, as ``cv.getRotationMatrix2D`` or :func:`rotateEx` expect it.

    Zero points right; because image y points down, positive angles turn clockwise on screen, so
    ``rotateEx(img, angleBetween(p1, p2))`` makes the ``p1``-``p2`` line horizontal (e.g. the eye line for face
    alignment).

    Args:
        p1: Start point ``(x, y)``.
        p2: End point ``(x, y)``.
        degrees: Return degrees (``(-180, 180]``) instead of radians.

    Raises:
        ValueError: If a point is not one finite ``(x, y)`` pair.
    """
    x1, y1 = _point(p1, "p1")
    x2, y2 = _point(p2, "p2")
    angle = math.atan2(y2 - y1, x2 - x1)
    return math.degrees(angle) if degrees else angle


Line = tuple[tuple[float, float], tuple[float, float]]


def lineIntersection(line1: Line, line2: Line) -> tuple[float, float] | None:
    """Intersection of two infinite lines, each given by two points ``((x1, y1), (x2, y2))``.

    Extends the segments, so it also finds the corner where two edges from ``cv.HoughLinesP`` would meet.

    Returns:
        The point ``(x, y)``, or ``None`` if the lines are parallel (including coincident).

    Raises:
        ValueError: If a line is not two finite points, or its two points coincide.
    """
    (x1, y1), (x2, y2) = _line(line1, "line1")
    (x3, y3), (x4, y4) = _line(line2, "line2")
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def _line(line: Line, name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    pts, _ = _points(line, name)
    if pts.shape[0] != 2:
        raise ValueError(f"{name} must be two points ((x1, y1), (x2, y2)), got {pts.shape[0]} points")
    if np.array_equal(pts[0], pts[1]):
        raise ValueError(f"{name} must have two different points, got {line!r}")
    return (float(pts[0, 0]), float(pts[0, 1])), (float(pts[1, 0]), float(pts[1, 1]))


def pointLineDistance(point: ArrayLike, line: Line, segment: bool = False) -> float:
    """Distance from a point to an infinite line (or, with ``segment``, to the segment between its two points).

    Args:
        point: The point ``(x, y)``.
        line: Two points ``((x1, y1), (x2, y2))`` on the line.
        segment: Measure to the segment: beyond its ends, the distance to the nearer end point.

    Raises:
        ValueError: If the point or line is not finite, or the line's two points coincide.
    """
    px, py = _point(point, "point")
    (x1, y1), (x2, y2) = _line(line, "line")
    dx, dy = x2 - x1, y2 - y1
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    if segment:
        t = min(1.0, max(0.0, t))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
