"""Measurement and shape analysis: shape descriptors, Feret diameters, curvature, intensity profiles, real-world scale.

OpenCV measures a contour one property at a time and stops at the raw numbers: there is no circularity, Feret
diameter, curvature, line profile or pixel-to-millimeter conversion. These helpers compute the descriptors that
particle analysis, metrology and quality inspection use, from contours or masks, with plain NumPy on top of
``cv.moments``, ``cv.convexHull``, ``cv.fitEllipse`` and ``cv.remap``.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .._validate import check_channels, check_finite, check_image_not_empty
from .color import toGray

__all__ = [
    "Ellipse",
    "Feret",
    "PixelScale",
    "ShapeDescriptors",
    "contourCurvature",
    "contourDistance",
    "feretDiameter",
    "fitEllipseEx",
    "huMomentsEx",
    "intensityProfile",
    "matchShapesEx",
    "measureArea",
    "measureLength",
    "radialProfile",
    "shapeDescriptors",
]


def _polygon(contour: ArrayLike, name: str = "contour", minPoints: int = 3) -> np.ndarray:
    """Validate a contour and return its points as a float64 ``(N, 2)`` array (sub-pixel coordinates kept)."""
    array = np.asarray(contour, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.shape[-1] != 2:
        raise ValueError(f"{name} must have 2 coordinates (x, y), got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    points = array.reshape(-1, 2)
    if len(points) < minPoints:
        raise ValueError(f"{name} must have at least {minPoints} points, got {len(points)}")
    return points


def _point(point: ArrayLike, name: str) -> tuple[float, float]:
    """Validate one ``(x, y)`` point."""
    array = np.asarray(point, dtype=np.float64).reshape(-1)
    if array.size != 2:
        raise ValueError(f"{name} must be an (x, y) point, got {point!r}")
    check_finite(name, float(array[0]))
    check_finite(name, float(array[1]))
    return float(array[0]), float(array[1])


def _is_contour(array: np.ndarray) -> bool:
    """Whether an array is a point list ``(N, 2)`` / ``(N, 1, 2)`` rather than an image."""
    return (array.ndim == 3 and array.shape[1] == 1 and array.shape[2] == 2) or (
        array.ndim == 2 and array.shape[1] == 2
    )


def _segment_distances(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Distance from every point to every segment, shape ``(len(points), len(starts))``."""
    d = ends - starts
    length2 = np.maximum((d**2).sum(axis=1), 1e-300)
    rel = points[:, None, :] - starts[None, :, :]
    t = np.clip((rel * d[None, :, :]).sum(axis=2) / length2[None, :], 0.0, 1.0)
    nearest = starts[None, :, :] + t[:, :, None] * d[None, :, :]
    out: np.ndarray = np.linalg.norm(points[:, None, :] - nearest, axis=2)
    return out


def _segments_cross(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether any edge of closed polygon ``a`` properly crosses any edge of closed polygon ``b``."""
    a0, a1 = a, np.roll(a, -1, axis=0)
    b0, b1 = b, np.roll(b, -1, axis=0)

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        cross = (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
        out: np.ndarray = np.sign(cross)
        return out

    p, q = a0[:, None, :], a1[:, None, :]
    r, s = b0[None, :, :], b1[None, :, :]
    o1, o2 = orient(p, q, r), orient(p, q, s)
    o3, o4 = orient(r, s, p), orient(r, s, q)
    return bool(np.any((o1 * o2 < 0) & (o3 * o4 < 0)))


class ShapeDescriptors(NamedTuple):
    """Dimensionless shape descriptors of a contour (see :func:`shapeDescriptors`)."""

    area: float
    """``cv.contourArea``: pixels inside the contour."""
    perimeter: float
    """``cv.arcLength`` of the closed contour."""
    circularity: float
    """``4 * pi * area / perimeter ** 2``: 1 for a circle, ``pi / 4`` for a square, lower for ragged shapes."""
    solidity: float
    """``area / hull area``: 1 for convex shapes, lower with dents or holes."""
    convexity: float
    """``hull perimeter / perimeter``: 1 for convex shapes, lower the more the outline wiggles."""
    extent: float
    """``area / (w * h)`` of the upright bounding box."""
    aspectRatio: float
    """``long side / short side`` of ``cv.minAreaRect`` (>= 1, rotation-invariant)."""
    orientation: float
    """Angle in degrees ``[0, 180)`` of the major axis from the second-order central moments."""
    eccentricity: float
    """``sqrt(1 - minor / major)`` of the moment eigenvalues: 0 for a circle, close to 1 for a line."""
    equivalentDiameter: float
    """Diameter of the circle with the same area."""
    compactness: float
    """``perimeter ** 2 / (4 * pi * area)``: the inverse of circularity, 1 for a circle and larger otherwise."""


def shapeDescriptors(contour: ArrayLike) -> ShapeDescriptors:
    """Dimensionless shape descriptors of a contour: circularity, solidity, convexity, eccentricity, orientation ...

    :func:`contourStats` returns the sizes (area, box, centroid). This adds the rotation- and scale-invariant
    descriptors used to classify particles and defects, which OpenCV leaves to the user: convexity needs
    ``cv.convexHull`` + ``cv.arcLength``, eccentricity and orientation need the eigenvalues of the moment tensor,
    and the aspect ratio of the rotated box needs ``cv.minAreaRect`` and sorting its sides.

    Args:
        contour: Contour of shape ``(N, 1, 2)`` or ``(N, 2)`` with at least 3 points.

    Raises:
        ValueError: If the contour has fewer than 3 points or is not finite.
    """
    points = _polygon(contour)
    cnt = points.astype(np.float32)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))
    _, _, w, h = cv2.boundingRect(cnt)
    (rw, rh) = cv2.minAreaRect(cnt)[1]
    long_side, short_side = max(rw, rh), min(rw, rh)
    m = cv2.moments(cnt)
    orientation = eccentricity = 0.0
    if m["m00"] > 0:
        mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
        orientation = math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02)) % 180.0
        common = math.sqrt(4 * mu11**2 + (mu20 - mu02) ** 2)
        lam_max, lam_min = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2
        eccentricity = math.sqrt(max(1 - lam_min / lam_max, 0.0))
    return ShapeDescriptors(
        area=area,
        perimeter=perimeter,
        circularity=4 * math.pi * area / perimeter**2 if perimeter else 0.0,
        solidity=area / hull_area if hull_area else 0.0,
        convexity=hull_perimeter / perimeter if perimeter else 0.0,
        extent=area / (w * h) if w and h else 0.0,
        aspectRatio=long_side / short_side if short_side else 0.0,
        orientation=orientation,
        eccentricity=eccentricity,
        equivalentDiameter=math.sqrt(4 * area / math.pi),
        compactness=perimeter**2 / (4 * math.pi * area) if area else 0.0,
    )


class Feret(NamedTuple):
    """Feret (caliper) diameters of a contour (see :func:`feretDiameter`)."""

    max: float
    """Largest distance between two points of the contour (the maximum caliper width)."""
    min: float
    """Smallest width of the shape between two parallel lines (the minimum caliper width)."""
    maxAngle: float
    """Direction in degrees ``[0, 180)`` of the maximum diameter."""
    minAngle: float
    """Direction in degrees ``[0, 180)`` along which the minimum width is measured."""


def feretDiameter(contour: ArrayLike) -> Feret:
    """Maximum and minimum Feret (caliper) diameters of a shape and the directions they are measured along.

    The caliper widths are the standard particle-size measures (ImageJ's "Feret" and "MinFeret"). OpenCV has no
    equivalent: ``cv.minAreaRect`` gives the minimum-area box, whose short side is not the minimum width in
    general. The computation runs on the convex hull with the rotating-calipers idea in NumPy.

    Args:
        contour: Contour of shape ``(N, 1, 2)`` or ``(N, 2)`` with at least 2 points.

    Raises:
        ValueError: If the contour has fewer than 2 points or is not finite.
    """
    points = _polygon(contour, minPoints=2)
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    diff = hull[:, None, :] - hull[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    i, j = np.unravel_index(int(np.argmax(dist)), dist.shape)
    max_d = float(dist[i, j])
    dx, dy = hull[j] - hull[i]
    max_angle = math.degrees(math.atan2(dy, dx)) % 180.0 if max_d > 0 else 0.0
    if len(hull) < 3:
        return Feret(max=max_d, min=0.0, maxAngle=max_angle, minAngle=(max_angle + 90.0) % 180.0)
    starts, ends = hull, np.roll(hull, -1, axis=0)
    edges = ends - starts
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 0
    starts, edges, lengths = starts[keep], edges[keep], lengths[keep]
    # Distance of every hull point from every edge line: (points, edges).
    rel = hull[:, None, :] - starts[None, :, :]
    cross = np.abs(rel[:, :, 0] * edges[None, :, 1] - rel[:, :, 1] * edges[None, :, 0]) / lengths[None, :]
    widths = cross.max(axis=0)
    k = int(np.argmin(widths))
    min_d = float(widths[k])
    min_angle = (math.degrees(math.atan2(edges[k, 1], edges[k, 0])) + 90.0) % 180.0
    return Feret(max=max_d, min=min_d, maxAngle=max_angle, minAngle=min_angle)


class Ellipse(NamedTuple):
    """A fitted ellipse (see :func:`fitEllipseEx`)."""

    center: tuple[float, float]
    """Center ``(x, y)``."""
    axes: tuple[float, float]
    """Full lengths ``(major, minor)``, ``major >= minor``."""
    angle: float
    """Direction in degrees ``[0, 180)`` of the major axis."""


def fitEllipseEx(contour: ArrayLike) -> Ellipse:
    """``cv.fitEllipse`` with validation and a fixed convention: ``(major, minor)`` axes and the major-axis angle.

    ``cv.fitEllipse`` asserts from C++ on fewer than 5 points, returns the axes in an arbitrary order and an angle
    that refers to whichever axis came first, so "the orientation of the blob" needs a sort and a ``+ 90`` fix-up.

    Args:
        contour: Contour of shape ``(N, 1, 2)`` or ``(N, 2)`` with at least 5 points.

    Raises:
        ValueError: If the contour has fewer than 5 points, is not finite, or the fit is degenerate.
    """
    points = _polygon(contour, minPoints=5)
    (cx, cy), (w, h), angle = cv2.fitEllipse(points.astype(np.float32))
    if not all(math.isfinite(v) for v in (cx, cy, w, h, angle)):  # pragma: no cover - depends on the OpenCV build
        raise ValueError("contour is degenerate (collinear points), no ellipse fits it")
    # OpenCV's angle is that of the ``w`` axis; the major axis is ``h`` when ``h > w``.
    direction = float(angle) + (90.0 if h > w else 0.0)
    return Ellipse(center=(float(cx), float(cy)), axes=(float(max(w, h)), float(min(w, h))), angle=direction % 180.0)


def huMomentsEx(src: np.ndarray, *, log: bool = True) -> np.ndarray:
    """Hu moments of a contour or of an image, as a flat ``(7,)`` vector, log-scaled so they are comparable.

    ``cv.HuMoments`` needs ``cv.moments`` first, returns a ``(7, 1)`` array and raw values that span 20 orders of
    magnitude; the usual ``-sign(h) * log10(|h|)`` scaling is left to the user.

    Args:
        src: A contour of shape ``(N, 1, 2)`` / ``(N, 2)``, or a grayscale/BGR/BGRA mask image whose non-zero
            pixels are the shape (``cv.moments`` with ``binaryImage=True``, so the values do not matter).
        log: Return ``-sign(h) * log10(|h|)`` (zeros stay 0) instead of the raw values.

    Returns:
        ``float64`` array of shape ``(7,)``.

    Raises:
        ValueError: If the contour has no points or is not finite, or the image is empty or has an unsupported
            channel count.
    """
    array = np.asarray(src)
    if _is_contour(array):
        moments = cv2.moments(_polygon(array, "src", minPoints=1).astype(np.float32))
    else:
        check_image_not_empty("src", array)
        moments = cv2.moments(_mask(array), binaryImage=True)
    hu = np.asarray(cv2.HuMoments(moments), dtype=np.float64).reshape(7)
    if not log:
        return hu
    out = np.zeros(7, dtype=np.float64)
    nonzero = hu != 0
    out[nonzero] = -np.sign(hu[nonzero]) * np.log10(np.abs(hu[nonzero]))
    return out


def _mask(image: np.ndarray) -> np.ndarray:
    """A ``uint8`` 0/1 mask of the non-zero pixels of any grayscale, BGR or BGRA image."""
    gray = toGray(image)
    out: np.ndarray = (gray != 0).astype(np.uint8)
    return out


def _shape_input(value: np.ndarray, name: str) -> np.ndarray:
    """A contour as ``float32 (N, 2)`` or a 0/1 ``uint8`` mask for ``cv.matchShapes``."""
    array = np.asarray(value)
    if _is_contour(array):
        return _polygon(array, name, minPoints=1).astype(np.float32)
    check_image_not_empty(name, array)
    return _mask(array)


def matchShapesEx(a: np.ndarray, b: np.ndarray, method: int = cv2.CONTOURS_MATCH_I1) -> float:
    """``cv.matchShapes`` for two contours or two binary masks (or one of each), with validation.

    ``cv.matchShapes`` takes a mysterious unused ``parameter`` argument, asserts from C++ on empty input,
    rejects a mask that is not 8-bit, and weights image pixels by their value, so a 0/255 mask does not match the
    contour of the same shape. Masks are binarized here (non-zero = shape). 0 means identical shapes (up to
    translation, scale and rotation).

    Args:
        a: Contour ``(N, 1, 2)`` / ``(N, 2)`` or a mask image (non-zero pixels are the shape).
        b: Same as ``a``.
        method: ``cv.CONTOURS_MATCH_I1`` (default), ``cv.CONTOURS_MATCH_I2`` or ``cv.CONTOURS_MATCH_I3``.

    Raises:
        ValueError: If an input is empty or not finite, or ``method`` is unknown.
    """
    if method not in (cv2.CONTOURS_MATCH_I1, cv2.CONTOURS_MATCH_I2, cv2.CONTOURS_MATCH_I3):
        raise ValueError(f"method must be CONTOURS_MATCH_I1, CONTOURS_MATCH_I2 or CONTOURS_MATCH_I3, got {method!r}")
    return float(cv2.matchShapes(_shape_input(a, "a"), _shape_input(b, "b"), method, 0.0))


def contourDistance(a: ArrayLike, b: ArrayLike) -> float:
    """Smallest distance between the outlines of two contours; 0 when they touch, cross or one contains the other.

    OpenCV only measures a point against a contour (``cv.pointPolygonTest``). The gap between two objects, the
    usual "are these particles separated by at least d pixels" check, needs the point-to-segment minimum both ways
    plus an overlap test, which this does in NumPy.

    Args:
        a: Contour of shape ``(N, 1, 2)`` or ``(N, 2)`` with at least 1 point.
        b: Same as ``a``.

    Returns:
        The distance in pixels (``0.0`` when the shapes overlap).

    Raises:
        ValueError: If a contour has no points or is not finite.
    """
    pa, pb = _polygon(a, "a", minPoints=1), _polygon(b, "b", minPoints=1)
    ca, cb = pa.astype(np.float32), pb.astype(np.float32)
    if len(pa) >= 3 and any(cv2.pointPolygonTest(ca, (float(x), float(y)), False) >= 0 for x, y in pb):
        return 0.0
    if len(pb) >= 3 and any(cv2.pointPolygonTest(cb, (float(x), float(y)), False) >= 0 for x, y in pa):
        return 0.0
    if len(pa) >= 2 and len(pb) >= 2 and _segments_cross(pa, pb):
        return 0.0
    d_ab = _segment_distances(pa, pb, np.roll(pb, -1, axis=0)).min()
    d_ba = _segment_distances(pb, pa, np.roll(pa, -1, axis=0)).min()
    return float(min(d_ab, d_ba))


def contourCurvature(contour: ArrayLike, step: int = 3) -> np.ndarray:
    """Signed curvature at every point of a closed contour (``1 / radius``; positive where the outline turns left).

    Corners, dents and the roundness of a tip are curvature questions; OpenCV offers ``cv.approxPolyDP`` and
    ``cv.convexityDefects`` but no curvature. Each point is compared with its neighbours ``step`` points before
    and after (Menger curvature of the three points), which smooths over pixel staircases.

    Args:
        contour: Contour of shape ``(N, 1, 2)`` or ``(N, 2)`` with at least 3 points.
        step: Neighbour offset along the contour; larger values smooth more.

    Returns:
        ``float32`` array of shape ``(N,)`` in 1/pixel. The sign follows the traversal direction: for the
        counter-clockwise outer contours of ``cv.findContours`` in image coordinates, convex parts are negative.

    Raises:
        ValueError: If the contour has fewer than 3 points or is not finite, or ``step`` is not positive.
    """
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")
    points = _polygon(contour)
    prev_pt = np.roll(points, step, axis=0)
    next_pt = np.roll(points, -step, axis=0)
    d1, d2, d3 = points - prev_pt, next_pt - points, next_pt - prev_pt
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    denominator = np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1) * np.linalg.norm(d3, axis=1)
    curvature = np.zeros(len(points), dtype=np.float64)
    ok = denominator > 0
    curvature[ok] = 2 * cross[ok] / denominator[ok]
    return curvature.astype(np.float32)


def intensityProfile(src: np.ndarray, p1: ArrayLike, p2: ArrayLike, *, width: int = 1) -> np.ndarray:
    """Pixel values sampled along the line from ``p1`` to ``p2``, with bilinear interpolation.

    The line profile is the first tool for measuring an edge, a stripe or a ruler. OpenCV's ``cv.LineIterator``
    is not exposed to Python, so this samples through ``cv.remap``; ``width > 1`` averages that many parallel
    lines, which reduces noise the way ImageJ's "line width" does.

    Args:
        src: Grayscale, BGR or BGRA image of any dtype.
        p1: Start point ``(x, y)`` (sub-pixel allowed).
        p2: End point ``(x, y)``.
        width: Number of parallel 1-pixel lines, centred on the line, to average (keyword-only).

    Returns:
        ``float32`` array of shape ``(M,)`` for grayscale or ``(M, C)`` for color, with ``M = ceil(length) + 1``
        samples spaced evenly from ``p1`` to ``p2``. Samples outside the image take the nearest border pixel.

    Raises:
        ValueError: If ``src`` is empty or has an unsupported channel count, a point is not finite, or
            ``width`` is not positive.
    """
    check_image_not_empty("src", src)
    check_channels("src", src)
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
    x1, y1 = _point(p1, "p1")
    x2, y2 = _point(p2, "p2")
    length = math.hypot(x2 - x1, y2 - y1)
    samples = math.ceil(length) + 1
    t = np.linspace(0.0, 1.0, samples)
    xs, ys = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    if length > 0:
        nx, ny = -(y2 - y1) / length, (x2 - x1) / length
    else:
        nx, ny = 0.0, 1.0
    offsets = np.arange(width, dtype=np.float64) - (width - 1) / 2
    map_x = (xs[None, :] + offsets[:, None] * nx).astype(np.float32)
    map_y = (ys[None, :] + offsets[:, None] * ny).astype(np.float32)
    image = src.astype(np.float32)
    sampled = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out: np.ndarray = sampled.reshape(width, samples, -1).mean(axis=0, dtype=np.float32)
    return out[:, 0] if src.ndim == 2 else out


def radialProfile(src: np.ndarray, center: ArrayLike | None = None, *, maxRadius: int | None = None) -> np.ndarray:
    """Mean intensity at each integer distance from a center: the radial (azimuthally averaged) profile.

    Used for lens vignetting and MTF checks, diffraction rings and the fall-off of a spot. OpenCV has
    ``cv.warpPolar`` but averaging its rows gets the sampling density wrong near the center; this bins every pixel
    by its rounded distance.

    Args:
        src: Grayscale, BGR or BGRA image of any dtype.
        center: ``(x, y)`` to measure from; default the image center.
        maxRadius: Largest radius to include; default the distance to the farthest corner.

    Returns:
        ``float64`` array of shape ``(maxRadius + 1,)`` for grayscale or ``(maxRadius + 1, C)`` for color, where
        index ``r`` is the mean of the pixels whose distance rounds to ``r``. Radii with no pixels are ``NaN``.

    Raises:
        ValueError: If ``src`` is empty or has an unsupported channel count, ``center`` is not finite, or
            ``maxRadius`` is negative.
    """
    check_image_not_empty("src", src)
    channels = check_channels("src", src)
    height, width = src.shape[:2]
    cx, cy = ((width - 1) / 2, (height - 1) / 2) if center is None else _point(center, "center")
    ys, xs = np.mgrid[0:height, 0:width]
    radius = np.rint(np.hypot(xs - cx, ys - cy)).astype(np.int64).ravel()
    if maxRadius is None:
        maxRadius = int(radius.max())
    elif maxRadius < 0:
        raise ValueError(f"maxRadius must not be negative, got {maxRadius}")
    keep = radius <= maxRadius
    radius = radius[keep]
    values = src.reshape(height * width, channels).astype(np.float64)[keep]
    counts = np.bincount(radius, minlength=maxRadius + 1).astype(np.float64)
    sums = np.stack(
        [np.bincount(radius, weights=values[:, c], minlength=maxRadius + 1) for c in range(channels)], axis=1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = sums / counts[:, None]
    return profile[:, 0] if src.ndim == 2 else profile


@dataclass(frozen=True)
class PixelScale:
    """Conversion between pixels and real-world units, from a known reference length.

    OpenCV measures everything in pixels. A ruler, a calibration target or a known part width gives the scale
    once; this object then turns lengths and areas into millimetres (or any unit) and back.

    Args:
        pixelsPerUnit: How many pixels one unit spans.
        unit: Name of the unit, for display only.

    Raises:
        ValueError: If ``pixelsPerUnit`` is not a positive finite number.
    """

    pixelsPerUnit: float
    unit: str = "mm"

    def __post_init__(self) -> None:
        check_finite("pixelsPerUnit", self.pixelsPerUnit, positive=True)

    @classmethod
    def fromReference(cls, pixelLength: float, realLength: float, unit: str = "mm") -> "PixelScale":
        """Scale from an object of known size: ``pixelLength`` pixels span ``realLength`` units.

        Raises:
            ValueError: If either length is not a positive finite number.
        """
        check_finite("pixelLength", pixelLength, positive=True)
        check_finite("realLength", realLength, positive=True)
        return cls(pixelLength / realLength, unit)

    def toReal(self, px: float) -> float:
        """Length in real units for a length in pixels."""
        return px / self.pixelsPerUnit

    def toPixels(self, real: float) -> float:
        """Length in pixels for a length in real units."""
        return real * self.pixelsPerUnit

    def area(self, pxArea: float) -> float:
        """Area in square real units for an area in square pixels."""
        return pxArea / self.pixelsPerUnit**2


def measureLength(p1: ArrayLike, p2: ArrayLike, scale: PixelScale) -> float:
    """Real-world distance between two image points, through a :class:`PixelScale`.

    Args:
        p1: Point ``(x, y)`` in pixels.
        p2: Point ``(x, y)`` in pixels.
        scale: The pixel-to-unit conversion.

    Raises:
        ValueError: If a point is not finite.
    """
    x1, y1 = _point(p1, "p1")
    x2, y2 = _point(p2, "p2")
    return scale.toReal(math.hypot(x2 - x1, y2 - y1))


def measureArea(contour: ArrayLike, scale: PixelScale) -> float:
    """Real-world area enclosed by a contour, through a :class:`PixelScale`.

    Args:
        contour: Contour of shape ``(N, 1, 2)`` or ``(N, 2)`` with at least 3 points.
        scale: The pixel-to-unit conversion.

    Raises:
        ValueError: If the contour has fewer than 3 points or is not finite.
    """
    points = _polygon(contour)
    return scale.area(float(cv2.contourArea(points.astype(np.float32))))
