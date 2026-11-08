"""Contours: shape statistics, sorting, filtering, masks, crops and debugging drawings.

``cv.findContours`` gives a list of ``(N, 1, 2)`` arrays and OpenCV measures them one property at a time
(``cv.contourArea``, ``cv.arcLength``, ``cv.boundingRect``, ``cv.convexHull`` ...). Everything after that, from
"sort left to right" to "crop the object with its background removed", is rewritten in every project.
"""

import math
from collections.abc import Sequence
from typing import Literal, NamedTuple

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .._validate import check_channels, check_finite, check_image_not_empty

__all__ = [
    "ContourStats",
    "approxContour",
    "asContour",
    "boundingRects",
    "contourStats",
    "contoursToMask",
    "cropContour",
    "drawContoursEx",
    "filterContours",
    "mergeContours",
    "sortContours",
]

SortMethod = Literal["left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top", "area", "perimeter"]


def asContour(points: ArrayLike) -> np.ndarray:
    """Turn any list of points into an OpenCV contour: an ``int32`` array of shape ``(N, 1, 2)``.

    ``cv.drawContours``, ``cv.contourArea``, ``cv.boundingRect`` and friends want exactly that layout and
    integer type, while points usually come as ``[(x, y), ...]`` or an ``(N, 2)`` float array.

    Args:
        points: ``(N, 2)``, ``(N, 1, 2)`` or a sequence of ``(x, y)`` pairs. Floats are rounded.

    Raises:
        ValueError: If the points are not 2-D, are not finite, or there are none.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        raise ValueError("points must not be empty")
    if array.shape[-1] != 2:
        raise ValueError(f"points must have 2 coordinates (x, y), got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must be finite")
    return np.rint(array).astype(np.int32).reshape(-1, 1, 2)


def boundingRects(contours: Sequence[np.ndarray]) -> np.ndarray:
    """``cv.boundingRect`` of every contour, as an ``(N, 4)`` ``int32`` array of ``(x, y, w, h)``.

    Ready for :func:`boxIoU`, for slicing, or for sorting with NumPy. Empty for no contours.
    """
    if len(contours) == 0:
        return np.zeros((0, 4), dtype=np.int32)
    return np.array([cv2.boundingRect(contour) for contour in contours], dtype=np.int32)


class ContourStats(NamedTuple):
    """The usual shape measurements of a contour (see :func:`contourStats`)."""

    area: float
    """``cv.contourArea``: pixels inside the contour."""
    perimeter: float
    """``cv.arcLength`` of the closed contour."""
    rect: tuple[int, int, int, int]
    """Upright bounding box ``(x, y, w, h)``."""
    center: tuple[float, float]
    """Centroid ``(x, y)`` from the image moments (the box center if the area is 0)."""
    aspectRatio: float
    """``w / h`` of the bounding box."""
    extent: float
    """``area / (w * h)``: how much of the bounding box the shape fills (1 for an upright rectangle)."""
    solidity: float
    """``area / hull area``: 1 for convex shapes, lower for shapes with dents or holes."""
    circularity: float
    """``4 * pi * area / perimeter ** 2``: 1 for a circle, lower for elongated or ragged shapes."""
    equivalentDiameter: float
    """Diameter of the circle with the same area."""
    orientation: float
    """Angle in degrees of the major axis from ``cv.fitEllipse`` (``0`` if there are fewer than 5 points)."""


def contourStats(contour: np.ndarray) -> ContourStats:
    """Measure a contour once: area, perimeter, box, centroid, aspect ratio, extent, solidity, circularity, ...

    OpenCV computes each of these with a different call (and ``solidity`` needs ``cv.convexHull`` first). This
    returns them all as a named tuple, so filtering shapes is one comparison per property.

    Args:
        contour: Contour of shape ``(N, 1, 2)`` or ``(N, 2)``, as from ``cv.findContours``.

    Raises:
        ValueError: If the contour has no points.
    """
    contour = asContour(contour)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    else:
        center = (x + (w - 1) / 2, y + (h - 1) / 2)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    orientation = 0.0
    if len(contour) >= 5:
        orientation = float(cv2.fitEllipse(contour)[2])
    return ContourStats(
        area=area,
        perimeter=perimeter,
        rect=(int(x), int(y), int(w), int(h)),
        center=center,
        aspectRatio=w / h if h else 0.0,
        extent=area / (w * h) if w and h else 0.0,
        solidity=area / hull_area if hull_area else 0.0,
        circularity=4 * math.pi * area / perimeter**2 if perimeter else 0.0,
        equivalentDiameter=math.sqrt(4 * area / math.pi),
        orientation=orientation,
    )


def sortContours(contours: Sequence[np.ndarray], method: SortMethod = "left-to-right") -> list[np.ndarray]:
    """Sort contours by position or size.

    Reading order ("left-to-right", "top-to-bottom") uses the bounding box; "area" and "perimeter" put the
    largest first. ``cv.findContours`` returns them in scan order, which is neither.

    Args:
        contours: Contours as from ``cv.findContours`` or :func:`findContoursEx`.
        method: ``"left-to-right"``, ``"right-to-left"``, ``"top-to-bottom"``, ``"bottom-to-top"``, ``"area"`` or
            ``"perimeter"``.

    Returns:
        A new list, in the requested order.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    items = list(contours)
    if not items:
        return items
    if method in ("left-to-right", "right-to-left"):
        keys = [float(cv2.boundingRect(c)[0]) for c in items]
        reverse = method == "right-to-left"
    elif method in ("top-to-bottom", "bottom-to-top"):
        keys = [float(cv2.boundingRect(c)[1]) for c in items]
        reverse = method == "bottom-to-top"
    elif method == "area":
        keys = [float(cv2.contourArea(c)) for c in items]
        reverse = True
    elif method == "perimeter":
        keys = [float(cv2.arcLength(c, True)) for c in items]
        reverse = True
    else:
        raise ValueError(
            f"method must be 'left-to-right', 'right-to-left', 'top-to-bottom', 'bottom-to-top', 'area' or 'perimeter',"
            f" got {method!r}"
        )
    order = sorted(range(len(items)), key=lambda i: keys[i], reverse=reverse)
    return [items[i] for i in order]


def filterContours(
    contours: Sequence[np.ndarray],
    minArea: float = 0.0,
    maxArea: float = math.inf,
    minAspectRatio: float = 0.0,
    maxAspectRatio: float = math.inf,
    minSolidity: float = 0.0,
    minCircularity: float = 0.0,
    minExtent: float = 0.0,
    minPerimeter: float = 0.0,
) -> list[np.ndarray]:
    """Keep the contours whose measurements fall within the given ranges.

    Each limit defaults to "no limit", so ``filterContours(cnts, minArea=100, minCircularity=0.8)`` keeps the
    round shapes of at least 100 pixels. See :class:`ContourStats` for what the properties mean.

    Returns:
        A new list, in the original order.

    Raises:
        ValueError: If a limit is ``NaN`` or negative, or a minimum exceeds its maximum.
    """
    limits = {
        "minArea": minArea,
        "maxArea": maxArea,
        "minAspectRatio": minAspectRatio,
        "maxAspectRatio": maxAspectRatio,
        "minSolidity": minSolidity,
        "minCircularity": minCircularity,
        "minExtent": minExtent,
        "minPerimeter": minPerimeter,
    }
    for name, value in limits.items():
        if math.isnan(value):
            raise ValueError(f"{name} must be a number, got {value}")
        if value < 0:
            raise ValueError(f"{name} must not be negative, got {value}")
    if minArea > maxArea:
        raise ValueError(f"minArea must not exceed maxArea, got {minArea} > {maxArea}")
    if minAspectRatio > maxAspectRatio:
        raise ValueError(f"minAspectRatio must not exceed maxAspectRatio, got {minAspectRatio} > {maxAspectRatio}")
    kept = []
    for contour in contours:
        s = contourStats(contour)
        if (
            minArea <= s.area <= maxArea
            and minAspectRatio <= s.aspectRatio <= maxAspectRatio
            and s.solidity >= minSolidity
            and s.circularity >= minCircularity
            and s.extent >= minExtent
            and s.perimeter >= minPerimeter
        ):
            kept.append(contour)
    return kept


def approxContour(contour: np.ndarray, epsilon: float = 0.02, closed: bool = True) -> np.ndarray:
    """``cv.approxPolyDP`` with the tolerance given as a fraction of the perimeter.

    The usual way to find the corners of a document or a sign is ``approxPolyDP(c, 0.02 * arcLength(c, True))``;
    the absolute tolerance OpenCV takes depends on the image size, the fraction does not.

    Args:
        contour: Contour of shape ``(N, 1, 2)``.
        epsilon: Tolerance as a fraction of the perimeter, e.g. ``0.02`` (2%). Larger values give fewer corners.
        closed: Whether the contour is closed (as for ``cv.arcLength`` and ``cv.approxPolyDP``).

    Returns:
        The simplified contour, shape ``(M, 1, 2)``.

    Raises:
        ValueError: If ``epsilon`` is negative or not finite, or the contour has no points or is not finite.
    """
    check_finite("epsilon", epsilon, nonnegative=True)
    contour = asContour(contour)
    tolerance = epsilon * cv2.arcLength(contour, closed)
    dst: np.ndarray = cv2.approxPolyDP(contour, tolerance, closed)
    return dst


def mergeContours(contours: Sequence[np.ndarray]) -> np.ndarray:
    """Merge contours into one: the convex hull around all their points.

    When a detection breaks into pieces (a letter and its dot, a logo split by a highlight), the hull of all the
    pieces is the usual single outline. For a tight, concave outline, draw the pieces with :func:`contoursToMask`,
    dilate, and run :func:`findContoursEx` again.

    Returns:
        A contour of shape ``(M, 1, 2)``.

    Raises:
        ValueError: If there are no contours, or a contour has no points or is not finite.
    """
    if len(contours) == 0:
        raise ValueError("contours must not be empty")
    points = np.concatenate([asContour(c) for c in contours])
    hull: np.ndarray = cv2.convexHull(points)
    return hull


def contoursToMask(
    contours: np.ndarray | Sequence[np.ndarray],
    size: tuple[int, int],
    thickness: int = cv2.FILLED,
    lineType: int = cv2.LINE_8,
) -> np.ndarray:
    """Draw contours into a new mask of ``size``: filled (default) or as outlines.

    The inverse of :func:`findContoursEx`. ``cv.drawContours`` needs an existing image, the ``-1`` for "all",
    a color and ``cv.FILLED``, and it rejects a bare ``(N, 2)`` array.

    Args:
        contours: One contour or a sequence of contours, each ``(N, 1, 2)`` or ``(N, 2)``.
        size: ``(width, height)`` of the mask.
        thickness: ``cv.FILLED`` (default) fills the shapes; a positive value draws outlines that thick.
        lineType: As for ``cv.drawContours``.

    Returns:
        ``uint8`` mask with values 0 and 255.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    width, height = size
    if width < 1 or height < 1:
        raise ValueError(f"size must be at least 1x1, got {size}")
    if isinstance(contours, np.ndarray) and contours.ndim <= 3 and contours.shape[-1] == 2:
        items = [contours]
    else:
        items = list(contours)
    mask = np.zeros((height, width), np.uint8)
    if items:
        cv2.drawContours(mask, [asContour(c) for c in items], -1, 255, thickness, lineType)
    return mask


def cropContour(
    src: np.ndarray,
    contour: np.ndarray,
    margin: int = 0,
    fill: float | Sequence[float] | None = 0,
    *,
    transparent: bool = False,
) -> np.ndarray:
    """Crop the bounding box of a contour, with the pixels outside the contour blanked out.

    Args:
        src: Image to crop (any dtype, 1, 3 or 4 channels).
        contour: Contour of shape ``(N, 1, 2)`` or ``(N, 2)``.
        margin: Extra pixels around the bounding box (clipped at the image border).
        fill: Value for the pixels outside the contour: a scalar or one value per channel (as ``fill`` in
            :func:`cropRect` and :func:`applyMask`). ``None`` keeps the original pixels (a plain bounding-box crop).
        transparent: Return BGRA with the outside transparent instead of filled with ``fill``
            (8-bit images only).

    Returns:
        The cropped image. Its size is the bounding box plus margins, clipped to the image.

    Raises:
        ValueError: If ``src`` is empty, ``margin`` is negative, the contour has no points, is not finite or lies
            outside the image, ``fill`` is not finite, or ``transparent`` is used on a non-8-bit image.
    """
    from ..core.roi import cropRect
    from .color import toBGRA

    check_image_not_empty("src", src)
    if margin < 0:
        raise ValueError(f"margin must not be negative, got {margin}")
    if transparent and src.dtype != np.uint8:
        raise ValueError(f"transparent output needs an 8-bit image, got dtype {src.dtype}")
    if fill is not None:
        for component in np.ravel(np.asarray(fill, dtype=np.float64)).tolist():
            check_finite("fill", component)
    contour = asContour(contour)
    x, y, w, h = cv2.boundingRect(contour)
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + w + margin, src.shape[1]), min(y + h + margin, src.shape[0])
    if x1 <= x0 or y1 <= y0:
        raise ValueError("the contour lies outside the image")
    crop = cropRect(src, (x0, y0, x1 - x0, y1 - y0))
    if fill is None and not transparent:
        return crop
    mask = contoursToMask(contour - (x0, y0), (x1 - x0, y1 - y0))
    if transparent:
        out = toBGRA(crop)
        out[..., 3] = (out[..., 3].astype(np.uint16) * mask // 255).astype(np.uint8)
        return out
    value = np.asarray(fill, dtype=crop.dtype)
    if crop.ndim == 3 and value.ndim == 0:
        value = np.full(crop.shape[2], value, dtype=crop.dtype)
    crop[mask == 0] = value
    return crop


def drawContoursEx(
    img: np.ndarray,
    contours: Sequence[np.ndarray],
    colors: Sequence[Sequence[float]] | None = None,
    labels: bool | Sequence[str] = False,
    thickness: int = 2,
    fontScale: float = 0.5,
) -> np.ndarray:
    """Draw every contour in its own color, optionally with an index or text label at its centroid.

    For checking what a segmentation found. ``cv.drawContours`` draws them all in one color, so touching shapes
    cannot be told apart. Like ``cv.drawContours``, this draws **in place** and returns ``img``.

    Args:
        img: 8-bit grayscale, BGR or BGRA image to draw on. On a grayscale image the luminance of each color
            is used (convert with :func:`toBGR` first to see the colors).
        contours: Contours as from :func:`findContoursEx`.
        colors: BGR colors, cycled through. By default :func:`distinctColors`.
        labels: ``True`` writes the index of each contour at its centroid; a sequence gives the texts.
        thickness: Outline thickness, or ``cv.FILLED``.
        fontScale: Hershey font size for the labels.

    Returns:
        ``img``, with the contours drawn on it.

    Raises:
        ValueError: If ``img`` is not 8-bit, is empty or has an unsupported channel count, a contour has no points
            or is not finite, ``fontScale`` is not positive, or ``labels`` or ``colors`` has a different length
            than ``contours``.
    """
    from .color import distinctColors

    if img.dtype != np.uint8:
        raise ValueError(f"img must be an 8-bit (uint8) image, got dtype {img.dtype}")
    check_image_not_empty("img", img)
    channels = check_channels("img", img)
    check_finite("fontScale", fontScale, positive=True)
    items = [asContour(c) for c in contours]
    if not isinstance(labels, bool) and len(labels) != len(items):
        raise ValueError(f"labels must have one entry per contour, got {len(labels)} for {len(items)} contours")
    palette: list[tuple[float, ...]]
    if colors is None:
        palette = [tuple(float(c) for c in color) for color in distinctColors(max(len(items), 1))]
    else:
        palette = [tuple(float(c) for c in color) for color in colors]
        if not palette:
            raise ValueError("colors must not be empty")
    for i, contour in enumerate(items):
        color = palette[i % len(palette)]
        if channels == 1:
            b, g, r = color[:3] if len(color) >= 3 else (color[0],) * 3
            color = (0.114 * b + 0.587 * g + 0.299 * r,)
        elif channels == 4 and len(color) == 3:
            color = (*color, 255.0)
        cv2.drawContours(img, [contour], -1, color, thickness)
        if labels is False:
            continue
        text = str(i) if labels is True else str(labels[i])
        cx, cy = contourStats(contour).center
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)
        org = (round(cx - tw / 2), round(cy + th / 2))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 1, cv2.LINE_AA)
    return img
