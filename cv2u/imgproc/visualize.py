"""Visualization: heatmaps, HUD overlays, skeletons, trajectories, rotated boxes and image comparisons."""

import math
from collections.abc import Sequence

import cv2
import numpy as np

from .._validate import check_color, check_finite, check_image_not_empty
from ..core.compose import hconcatEx, vconcatEx
from .color import distinctColors, toBGR, toBGRA, toGray
from .draw import Color, _bgr, _luminance

__all__ = [
    "COCO_SKELETON",
    "applyColorMapEx",
    "blendImages",
    "colorBar",
    "compareImages",
    "drawArrow",
    "drawCrosshair",
    "drawFPS",
    "drawLegend",
    "drawPolygon",
    "drawProgressBar",
    "drawRotatedBoxes",
    "drawRotatedRect",
    "drawSkeleton",
    "drawTextBlock",
    "drawTrajectory",
    "heatmap",
    "labeledMontage",
]

COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),  # nose - eyes - ears
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # shoulders and arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
)  # fmt: skip
"""Keypoint pairs of the 17-keypoint COCO skeleton (nose, eyes, ears, shoulders, elbows, wrists, hips, knees,
ankles)."""

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _channels(img: np.ndarray) -> int:
    return 1 if img.ndim == 2 else int(img.shape[2])


def _check_uint8(name: str, img: np.ndarray) -> None:
    if img.dtype != np.uint8:
        raise ValueError(f"{name} must be an 8-bit (uint8) image, got dtype {img.dtype}")
    check_image_not_empty(name, img)
    if img.ndim not in (2, 3) or _channels(img) not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3 or 4 channels, got shape {img.shape}")


def _check_unit(name: str, value: float) -> None:
    check_finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def _draw_color(img: np.ndarray, color: Color) -> Color:
    """A color usable by ``cv2`` drawing on ``img`` (luminance for grayscale images)."""
    values = [float(color)] if isinstance(color, (int, float)) else [float(c) for c in color]
    check_color("color", values, img.dtype)
    if _channels(img) == 1 and not isinstance(color, (int, float)):
        return _luminance(color)
    return color


def _points2d(
    name: str, points: Sequence[Sequence[float]] | Sequence[float] | np.ndarray, columns: int = 2
) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, columns))
    array = array.reshape(-1, array.shape[-1]) if array.ndim >= 2 else array.reshape(1, -1)
    if array.shape[1] < 2:
        raise ValueError(f"{name} must have at least 2 columns (x, y), got shape {np.shape(points)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite numbers")
    return array


def _xy(point: Sequence[float] | np.ndarray) -> tuple[int, int]:
    return round(float(point[0])), round(float(point[1]))


def _blend_region(img: np.ndarray, layer: np.ndarray, alpha: float, x0: int, y0: int, x1: int, y1: int) -> None:
    """Blend ``layer`` (same shape as ``img``) into ``img`` over a clipped rectangle, in place."""
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    region = img[y0:y1, x0:x1]
    cv2.addWeighted(region, 1 - alpha, layer[y0:y1, x0:x1], alpha, 0, dst=region)


# ----------------------------------------------------------------------------------------------- color maps


def applyColorMapEx(
    values: np.ndarray,
    colormap: int | np.ndarray = cv2.COLORMAP_JET,
    normalize: bool = True,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Colorize a 2-D map of any dtype, like ``cv.applyColorMap`` which only takes 8-bit input.

    Float maps (saliency, depth, attention, probabilities) are scaled to ``[0, 255]`` first.

    Args:
        values: 2-D array (or 3-D with one channel) of any numeric dtype.
        colormap: An OpenCV ``cv.COLORMAP_*`` constant, or a ``(256, 1, 3)`` / ``(256, 3)`` 8-bit lookup table.
        normalize: Stretch the minimum and maximum of ``values`` to 0 and 255. When ``False``, the values are
            clipped to ``[0, 255]`` (float maps in ``[0, 1]`` are scaled by 255 first).
        mask: Optional 2-D mask with the shape of ``values``; pixels outside it are black.

    Returns:
        An 8-bit BGR image with the shape of ``values``.

    Raises:
        ValueError: If ``values`` is empty or not 2-D, contains NaN/inf, or ``mask`` has a different shape.
    """
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"values must be a 2-D array, got shape {array.shape}")
    check_image_not_empty("values", array)
    if mask is not None and mask.shape[:2] != array.shape:
        raise ValueError(f"mask must have the shape {array.shape}, got {mask.shape}")

    if array.dtype == np.uint8 and not normalize:
        scaled = array
    else:
        floats = array.astype(np.float64)
        if not np.all(np.isfinite(floats)):
            raise ValueError("values must be finite numbers")
        if normalize:
            lo, hi = float(np.min(floats)), float(np.max(floats))
            floats = (floats - lo) / (hi - lo) if hi > lo else np.zeros_like(floats)
            floats *= 255.0
        elif np.issubdtype(array.dtype, np.floating) and np.max(floats) <= 1.0 and np.min(floats) >= 0.0:
            floats *= 255.0
        scaled = np.clip(np.rint(floats), 0, 255).astype(np.uint8)

    if isinstance(colormap, np.ndarray):
        lut = np.asarray(colormap, dtype=np.uint8).reshape(256, 1, 3)
        colored: np.ndarray = cv2.applyColorMap(scaled, lut)
    else:
        colored = cv2.applyColorMap(scaled, colormap)
    if mask is not None:
        colored[np.asarray(mask) == 0] = 0
    return colored


def heatmap(
    src: np.ndarray,
    values: np.ndarray,
    alpha: float = 0.5,
    colormap: int | np.ndarray = cv2.COLORMAP_JET,
    normalize: bool = True,
) -> np.ndarray:
    """Overlay a colorized map (attention, saliency, density, CAM) on an image.

    ``cv.applyColorMap`` needs an 8-bit map of the image size; this takes a float map of any size, resizes it to
    ``src`` and blends it in.

    Args:
        src: 8-bit grayscale, BGR or BGRA image. It is not modified.
        values: 2-D map of any numeric dtype and size (resized to ``src`` with linear interpolation).
        alpha: Opacity of the color map, from 0 to 1.
        colormap: As for :func:`applyColorMapEx`.
        normalize: As for :func:`applyColorMapEx`.

    Returns:
        A new BGR image (BGRA keeps the alpha channel of ``src``) with the size of ``src``.

    Raises:
        ValueError: If ``src`` is not 8-bit, ``values`` is not 2-D, or ``alpha`` is outside ``[0, 1]``.
    """
    _check_uint8("src", src)
    _check_unit("alpha", alpha)
    colored = applyColorMapEx(values, colormap, normalize)
    height, width = src.shape[:2]
    if colored.shape[:2] != (height, width):
        colored = cv2.resize(colored, (width, height), interpolation=cv2.INTER_LINEAR)
    base = toBGR(src)
    blended: np.ndarray = cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)
    if _channels(src) == 4:
        blended = np.dstack([blended, src[..., 3]])
    return blended


def colorBar(
    colormap: int | np.ndarray = cv2.COLORMAP_JET,
    size: tuple[int, int] = (256, 20),
    vertical: bool = False,
    minVal: float | None = None,
    maxVal: float | None = None,
    fontScale: float = 0.4,
) -> np.ndarray:
    """Make a legend strip for a color map, optionally labeled with its value range, as a new BGR image.

    Args:
        colormap: As for :func:`applyColorMapEx`.
        size: Size of the strip as ``(width, height)`` (``vertical`` swaps the gradient direction, not the size).
        vertical: Run the gradient from the bottom (minimum) to the top instead of left to right.
        minVal: Value written at the low end; ``None`` writes nothing.
        maxVal: Value written at the high end; ``None`` writes nothing.
        fontScale: Hershey font size of the labels.

    Returns:
        A BGR image of ``size``; with labels, the strip grows to make room for the text.

    Raises:
        ValueError: If ``size`` is not positive or a label value is not finite.
    """
    width, height = size
    if width < 1 or height < 1:
        raise ValueError(f"size must be positive, got {size}")
    for name, value in (("minVal", minVal), ("maxVal", maxVal)):
        if value is not None:
            check_finite(name, value)
    if vertical:
        ramp = np.linspace(255, 0, height).reshape(height, 1).repeat(width, axis=1)
    else:
        ramp = np.linspace(0, 255, width).reshape(1, width).repeat(height, axis=0)
    strip = applyColorMapEx(ramp.astype(np.uint8), colormap, normalize=False)
    labels = [(name, f"{value:g}") for name, value in (("min", minVal), ("max", maxVal)) if value is not None]
    if not labels:
        return strip
    pad = 4
    sizes = {name: cv2.getTextSize(text, _FONT, fontScale, 1)[0] for name, text in labels}
    text_h = max(h for _, h in sizes.values())
    if vertical:
        text_w = max(w for w, _ in sizes.values())
        canvas = np.full((height, width + text_w + pad, 3), 255, np.uint8)
        canvas[:, :width] = strip
        for name, text in labels:
            y = height - 1 if name == "min" else sizes[name][1]
            cv2.putText(canvas, text, (width + pad, y), _FONT, fontScale, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        canvas = np.full((height + text_h + pad, width, 3), 255, np.uint8)
        canvas[:height] = strip
        for name, text in labels:
            x = 0 if name == "min" else width - sizes[name][0]
            cv2.putText(canvas, text, (x, height + pad + text_h - 1), _FONT, fontScale, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


# ----------------------------------------------------------------------------------------------- shapes


def drawArrow(
    img: np.ndarray,
    pt1: Sequence[float],
    pt2: Sequence[float],
    color: Color,
    thickness: int = 2,
    headSize: int = 10,
    heads: str = "end",
    lineType: int = cv2.LINE_AA,
) -> np.ndarray:
    """Draw an arrow with a head of a fixed size in pixels, in place.

    ``cv.arrowedLine`` sizes the head as a fraction of the arrow length, so short arrows (e.g. optical flow)
    get invisible heads and long ones get huge heads; it also draws a head at one end only.

    Args:
        img: Image to draw on.
        pt1: Start point ``(x, y)``; floats are rounded.
        pt2: End point ``(x, y)``.
        color: Arrow color.
        thickness: Line thickness.
        headSize: Length of the head's sides in pixels.
        heads: ``"end"`` (head at ``pt2``), ``"start"``, ``"both"`` or ``"none"``.
        lineType: As for ``cv.line``.

    Returns:
        ``img``.

    Raises:
        ValueError: If ``heads`` is unknown, ``headSize`` is negative, or a point is not finite.
    """
    if heads not in ("end", "start", "both", "none"):
        raise ValueError(f"heads must be 'end', 'start', 'both' or 'none', got {heads!r}")
    if headSize < 0:
        raise ValueError(f"headSize must not be negative, got {headSize}")
    start, end = _points2d("pt1", pt1)[0], _points2d("pt2", pt2)[0]
    color = _draw_color(img, color)
    cv2.line(img, _xy(start), _xy(end), color, thickness, lineType)
    length = float(np.hypot(*(end - start)))
    if length == 0 or headSize == 0:
        return img
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    spread = math.pi / 6
    tips = []
    if heads in ("end", "both"):
        tips.append((end, angle))
    if heads in ("start", "both"):
        tips.append((start, angle + math.pi))
    for tip, direction in tips:
        for sign in (1, -1):
            a = direction + math.pi + sign * spread
            wing = (tip[0] + headSize * math.cos(a), tip[1] + headSize * math.sin(a))
            cv2.line(img, _xy(tip), _xy(wing), color, thickness, lineType)
    return img


def drawRotatedRect(
    img: np.ndarray,
    rect: tuple[Sequence[float], Sequence[float], float],
    color: Color,
    thickness: int = 2,
    lineType: int = cv2.LINE_AA,
) -> np.ndarray:
    """Draw a rotated rectangle ``((cx, cy), (w, h), angle)`` as returned by ``cv.minAreaRect``, in place.

    OpenCV has ``cv.boxPoints`` but no drawing function for its own ``RotatedRect``.

    Args:
        img: Image to draw on.
        rect: The rotated rectangle.
        color: Line or fill color.
        thickness: Line thickness; ``cv.FILLED`` fills the rectangle.
        lineType: As for ``cv.polylines``.

    Returns:
        ``img``.

    Raises:
        ValueError: If ``rect`` has a negative size or non-finite values.
    """
    (cx, cy), (w, h), angle = rect
    for name, value in (("rect center", cx), ("rect center", cy), ("rect angle", angle)):
        check_finite(name, float(value))
    for value in (w, h):
        check_finite("rect size", float(value), nonnegative=True)
    color = _draw_color(img, color)
    corners = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(angle)))
    points = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    if thickness < 0:
        cv2.fillPoly(img, [points], color, lineType)
    else:
        cv2.polylines(img, [points], True, color, thickness, lineType)
    return img


def drawRotatedBoxes(
    img: np.ndarray,
    rects: Sequence[tuple[Sequence[float], Sequence[float], float]],
    colors: Sequence[Color] | Color | None = None,
    labels: Sequence[str] | None = None,
    thickness: int = 2,
    fontScale: float = 0.5,
) -> np.ndarray:
    """Draw many rotated rectangles, each with an optional label at its top corner, in place.

    The rotated counterpart of :func:`drawBoxes` for oriented detectors and ``cv.minAreaRect`` results.

    Args:
        img: Image to draw on.
        rects: Rotated rectangles ``((cx, cy), (w, h), angle)``.
        colors: One color per box, one color for all, or ``None`` for :func:`distinctColors`.
        labels: One text per box, or ``None``.
        thickness: Line thickness.
        fontScale: Hershey font size of the labels.

    Returns:
        ``img``.

    Raises:
        ValueError: If ``labels`` or ``colors`` has a different length than ``rects``.
    """
    count = len(rects)
    if labels is not None and len(labels) != count:
        raise ValueError(f"labels must have one entry per box, got {len(labels)} for {count} boxes")
    palette = _palette(colors, count)
    for i, rect in enumerate(rects):
        color = palette[i]
        drawRotatedRect(img, rect, color, thickness)
        if labels is None:
            continue
        corners = cv2.boxPoints(
            ((float(rect[0][0]), float(rect[0][1])), (float(rect[1][0]), float(rect[1][1])), float(rect[2]))
        )
        top = corners[np.argmin(corners[:, 1])]
        (text_w, text_h), baseline = cv2.getTextSize(labels[i], _FONT, fontScale, 1)
        x, y = round(float(top[0])), round(float(top[1])) - 4
        if y - text_h - baseline < 0:
            y = text_h + baseline + 2
        cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w + 4, y + 2), color, cv2.FILLED)
        level = 0 if _luminance(color) > 140 else 255
        text_color: Color = level if _channels(img) == 1 else (level, level, level)
        cv2.putText(img, labels[i], (x + 2, y), _FONT, fontScale, text_color, 1, cv2.LINE_AA)
    return img


def _palette(colors: Sequence[Color] | Color | None, count: int) -> list[Color]:
    if colors is None:
        return list(distinctColors(max(count, 1)))
    if isinstance(colors, (int, float)) or (len(colors) in (3, 4) and all(isinstance(c, (int, float)) for c in colors)):
        return [colors] * max(count, 1)  # type: ignore[list-item]
    palette = list(colors)
    if len(palette) != count:
        raise ValueError(f"colors must have one entry per item, got {len(palette)} for {count} items")
    return palette


def drawPolygon(
    img: np.ndarray,
    points: Sequence[Sequence[float]] | np.ndarray,
    color: Color,
    thickness: int = 2,
    closed: bool = True,
    fill: bool = False,
    alpha: float = 1.0,
    lineType: int = cv2.LINE_AA,
) -> np.ndarray:
    """Draw a polygon from float points, optionally filled with transparency, in place.

    ``cv.polylines`` and ``cv.fillPoly`` need an ``int32`` ``(N, 1, 2)`` array and draw opaque only; a translucent
    fill (segmentation polygons, zones) takes a copy and ``cv.addWeighted`` every time.

    Args:
        img: Image to draw on.
        points: ``(N, 2)`` vertices; floats are rounded.
        color: Outline and fill color.
        thickness: Outline thickness; ``0`` draws no outline.
        closed: Connect the last vertex to the first (ignored when ``fill`` is set).
        fill: Fill the polygon.
        alpha: Opacity of the drawing, from 0 to 1 (applied to the outline and the fill).

        lineType: As for ``cv.polylines``.

    Returns:
        ``img``.

    Raises:
        ValueError: If ``points`` is not 2-D or not finite, or ``alpha`` is outside ``[0, 1]``.
    """
    _check_unit("alpha", alpha)
    array = _points2d("points", points)
    if len(array) == 0:
        return img
    color = _draw_color(img, color)
    pts = np.rint(array[:, :2]).astype(np.int32).reshape(-1, 1, 2)
    target = img if alpha >= 1.0 else img.copy()
    if fill:
        cv2.fillPoly(target, [pts], color, lineType)
    if thickness > 0:
        cv2.polylines(target, [pts], closed or fill, color, thickness, lineType)
    if target is not img:
        x0, y0 = int(np.min(pts[:, 0, 0])) - thickness - 1, int(np.min(pts[:, 0, 1])) - thickness - 1
        x1, y1 = int(np.max(pts[:, 0, 0])) + thickness + 2, int(np.max(pts[:, 0, 1])) + thickness + 2
        _blend_region(img, target, alpha, x0, y0, x1, y1)
    return img


def drawSkeleton(
    img: np.ndarray,
    keypoints: Sequence[Sequence[float]] | np.ndarray,
    connections: Sequence[tuple[int, int]] = COCO_SKELETON,
    colors: Sequence[Color] | Color | None = None,
    radius: int = 4,
    thickness: int = 2,
    minScore: float = 0.0,
) -> np.ndarray:
    """Draw pose keypoints and the limbs between them, in place.

    Args:
        img: Image to draw on.
        keypoints: ``(K, 2)`` coordinates or ``(K, 3)`` with a confidence score in the third column.
        connections: Pairs of keypoint indices to join; :data:`COCO_SKELETON` by default. Pairs with an index
            outside ``keypoints`` are skipped, so partial detections draw fine.
        colors: One color per connection, one color for all, or ``None`` for :func:`distinctColors`. Each joint
            takes the color of its first connection.
        radius: Joint circle radius; ``0`` draws limbs only.
        thickness: Limb line thickness.
        minScore: Keypoints with a score below this are not drawn (and neither are their limbs). Keypoints at
            ``(0, 0)`` with no score column are also skipped, the usual marker for "not detected".

    Returns:
        ``img``.

    Raises:
        ValueError: If ``keypoints`` has fewer than 2 columns, ``radius`` is negative, or ``colors`` has a
            different length than ``connections``.
    """
    if radius < 0:
        raise ValueError(f"radius must not be negative, got {radius}")
    check_finite("minScore", minScore)
    array = _points2d("keypoints", keypoints)
    count = len(array)
    undetected = (array[:, 0] == 0) & (array[:, 1] == 0)  # without a score column, (0, 0) marks a missing keypoint
    visible = array[:, 2] >= minScore if array.shape[1] >= 3 else ~undetected
    palette = _palette(colors, len(connections))
    joint_colors: dict[int, Color] = {}
    for n, (a, b) in enumerate(connections):
        if not (0 <= a < count and 0 <= b < count) or not (visible[a] and visible[b]):
            continue
        color = _draw_color(img, palette[n])
        cv2.line(img, _xy(array[a]), _xy(array[b]), color, thickness, cv2.LINE_AA)
        joint_colors.setdefault(a, color)
        joint_colors.setdefault(b, color)
    if radius == 0:
        return img
    default = _draw_color(img, palette[0]) if palette else (0, 0, 255)
    for k in range(count):
        if visible[k]:
            cv2.circle(img, _xy(array[k]), radius, joint_colors.get(k, default), cv2.FILLED, cv2.LINE_AA)
    return img


def drawTrajectory(
    img: np.ndarray,
    points: Sequence[Sequence[float]] | np.ndarray,
    color: Color | None = None,
    thickness: int = 2,
    fade: bool = True,
    maxPoints: int | None = None,
    endColor: Color | None = None,
) -> np.ndarray:
    """Draw a track (object path, mouse trail) as a polyline that fades out towards its oldest points, in place.

    Args:
        img: Image to draw on.
        points: ``(N, 2)`` positions from oldest to newest; floats are rounded.
        color: Color at the newest point; ``None`` uses green.
        thickness: Line thickness at the newest segment.
        fade: Darken and thin the line towards the oldest point. ``False`` draws a plain polyline.
        maxPoints: Only draw the last ``maxPoints`` positions.
        endColor: Color at the oldest point when fading; ``None`` fades towards black (or the image's zero level).

    Returns:
        ``img``.

    Raises:
        ValueError: If ``points`` is not 2-D or not finite, or ``maxPoints`` is not positive.
    """
    if maxPoints is not None and maxPoints < 1:
        raise ValueError(f"maxPoints must be positive, got {maxPoints}")
    array = _points2d("points", points)[:, :2]
    if maxPoints is not None:
        array = array[-maxPoints:]
    if len(array) < 2:
        return img
    head = (0, 255, 0) if color is None else color
    head_bgr = _bgr(head)
    if not fade:
        pts = np.rint(array).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, _draw_color(img, head), thickness, cv2.LINE_AA)
        return img
    tail_bgr = (0.0, 0.0, 0.0) if endColor is None else _bgr(endColor)
    segments = len(array) - 1
    for i in range(segments):
        t = (i + 1) / segments  # 1 at the newest segment
        mixed = tuple(tail + (h - tail) * t for h, tail in zip(head_bgr, tail_bgr, strict=True))
        width = max(1, round(thickness * (0.3 + 0.7 * t)))
        cv2.line(img, _xy(array[i]), _xy(array[i + 1]), _draw_color(img, mixed), width, cv2.LINE_AA)
    return img


def drawCrosshair(
    img: np.ndarray,
    center: Sequence[float],
    color: Color,
    size: int = 20,
    gap: int = 4,
    thickness: int = 1,
    lineType: int = cv2.LINE_AA,
) -> np.ndarray:
    """Draw a crosshair (four ticks around a center with a gap), in place.

    ``cv.drawMarker`` draws a full cross that covers the point of interest.

    Args:
        img: Image to draw on.
        center: ``(x, y)``; floats are rounded.
        color: Line color.
        size: Half length of the crosshair in pixels, from the center to the tip of a tick.
        gap: Empty radius around the center.
        thickness: Line thickness.
        lineType: As for ``cv.line``.

    Returns:
        ``img``.

    Raises:
        ValueError: If ``size`` is not positive, ``gap`` is negative or not smaller than ``size``.
    """
    if size < 1:
        raise ValueError(f"size must be positive, got {size}")
    if gap < 0 or gap >= size:
        raise ValueError(f"gap must be between 0 and size - 1, got {gap} for size {size}")
    cx, cy = _xy(_points2d("center", center)[0])
    color = _draw_color(img, color)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img, (cx + dx * gap, cy + dy * gap), (cx + dx * size, cy + dy * size), color, thickness, lineType)
    return img


# ----------------------------------------------------------------------------------------------- HUD


def drawTextBlock(
    img: np.ndarray,
    lines: Sequence[str] | str,
    org: tuple[int, int] = (10, 10),
    fontScale: float = 0.5,
    color: Color = (255, 255, 255),
    background: Color | None = (0, 0, 0),
    alpha: float = 0.6,
    padding: int = 6,
    lineSpacing: float = 1.3,
    thickness: int = 1,
) -> np.ndarray:
    """Write several lines of text on a translucent panel (an on-screen display), in place.

    ``cv.putText`` writes one line, with no background, and needs ``getTextSize`` math for the line height.

    Args:
        img: 8-bit image to draw on.
        lines: Text lines (a single string is split at newlines). The Hershey font covers ASCII only.
        org: Top-left corner ``(x, y)`` of the panel.
        fontScale: Hershey font size.
        color: Text color.
        background: Panel color; ``None`` draws text only.
        alpha: Opacity of the panel, from 0 to 1.
        padding: Space in pixels between the text and the panel border.
        lineSpacing: Line height as a multiple of the text height.
        thickness: Text stroke thickness.

    Returns:
        ``img``.

    Raises:
        ValueError: If ``img`` is not 8-bit, ``alpha`` is outside ``[0, 1]``, ``padding`` is negative, or
            ``fontScale`` / ``lineSpacing`` is not positive.
    """
    _check_uint8("img", img)
    _check_unit("alpha", alpha)
    check_finite("fontScale", fontScale, positive=True)
    check_finite("lineSpacing", lineSpacing, positive=True)
    if padding < 0:
        raise ValueError(f"padding must not be negative, got {padding}")
    texts = lines.split("\n") if isinstance(lines, str) else list(lines)
    if not texts:
        return img
    sizes = [cv2.getTextSize(t or " ", _FONT, fontScale, thickness) for t in texts]
    text_h = max(h for (_, h), _ in sizes)
    baseline = max(b for _, b in sizes)
    line_h = max(1, round((text_h + baseline) * lineSpacing))
    width = max(w for (w, _), _ in sizes) + 2 * padding
    height = line_h * len(texts) + 2 * padding
    x, y = int(org[0]), int(org[1])
    if background is not None:
        panel = img.copy()
        cv2.rectangle(panel, (x, y), (x + width - 1, y + height - 1), _draw_color(img, background), cv2.FILLED)
        _blend_region(img, panel, alpha, x, y, x + width, y + height)
    text_color = _draw_color(img, color)
    for i, text in enumerate(texts):
        base = (x + padding, y + padding + i * line_h + text_h + (line_h - text_h - baseline) // 2)
        cv2.putText(img, text, base, _FONT, fontScale, text_color, thickness, cv2.LINE_AA)
    return img


def drawFPS(img: np.ndarray, fps: float, org: tuple[int, int] = (10, 10), fontScale: float = 0.6) -> np.ndarray:
    """Write ``"FPS: 29.9"`` on a translucent panel, in place (see :func:`drawTextBlock`).

    Raises:
        ValueError: If ``fps`` is not finite.
    """
    check_finite("fps", fps)
    return drawTextBlock(img, f"FPS: {fps:.1f}", org, fontScale)


def drawProgressBar(
    img: np.ndarray,
    progress: float,
    org: tuple[int, int],
    size: tuple[int, int] = (200, 12),
    color: Color = (0, 200, 0),
    background: Color = (60, 60, 60),
    border: Color | None = None,
) -> np.ndarray:
    """Draw a horizontal progress bar filled to ``progress`` (0 to 1), in place.

    Args:
        img: Image to draw on.
        progress: Fraction done, clipped to ``[0, 1]``.
        org: Top-left corner ``(x, y)``.
        size: Bar size as ``(width, height)``.
        color: Fill color.
        background: Track color.
        border: Outline color; ``None`` draws no outline.

    Returns:
        ``img``.

    Raises:
        ValueError: If ``progress`` is not finite or ``size`` is not positive.
    """
    check_finite("progress", progress)
    width, height = size
    if width < 1 or height < 1:
        raise ValueError(f"size must be positive, got {size}")
    x, y = int(org[0]), int(org[1])
    x1, y1 = x + width - 1, y + height - 1
    cv2.rectangle(img, (x, y), (x1, y1), _draw_color(img, background), cv2.FILLED)
    filled = round(width * min(max(progress, 0.0), 1.0))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled - 1, y1), _draw_color(img, color), cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x, y), (x1, y1), _draw_color(img, border), 1)
    return img


def drawLegend(
    img: np.ndarray,
    labels: Sequence[str],
    colors: Sequence[Color] | None = None,
    org: tuple[int, int] = (10, 10),
    swatch: int = 14,
    fontScale: float = 0.5,
    color: Color = (255, 255, 255),
    background: Color | None = (0, 0, 0),
    alpha: float = 0.6,
    padding: int = 6,
) -> np.ndarray:
    """Draw a legend panel with a color swatch next to each label, in place.

    Pairs with :func:`drawBoxes` / :func:`distinctColors` to explain which color is which class.

    Args:
        img: 8-bit image to draw on.
        labels: One text per entry.
        colors: One color per entry; ``None`` uses :func:`distinctColors`.
        org: Top-left corner ``(x, y)`` of the panel.
        swatch: Side of the color squares in pixels.
        fontScale: Hershey font size.
        color: Text color.
        background: Panel color; ``None`` draws no panel.
        alpha: Opacity of the panel.
        padding: Space between the entries and the panel border.

    Returns:
        ``img``.

    Raises:
        ValueError: If ``colors`` has a different length than ``labels``, ``swatch`` is not positive, or see
            :func:`drawTextBlock`.
    """
    _check_uint8("img", img)
    _check_unit("alpha", alpha)
    if swatch < 1:
        raise ValueError(f"swatch must be positive, got {swatch}")
    if padding < 0:
        raise ValueError(f"padding must not be negative, got {padding}")
    palette = list(distinctColors(len(labels))) if colors is None else list(colors)
    if len(palette) != len(labels):
        raise ValueError(f"colors must have one entry per label, got {len(palette)} for {len(labels)} labels")
    if not labels:
        return img
    sizes = [cv2.getTextSize(t or " ", _FONT, fontScale, 1) for t in labels]
    text_h = max(h for (_, h), _ in sizes)
    row_h = max(swatch, text_h) + 6
    gap = 6
    width = swatch + gap + max(w for (w, _), _ in sizes) + 2 * padding
    height = row_h * len(labels) + 2 * padding
    x, y = int(org[0]), int(org[1])
    if background is not None:
        panel = img.copy()
        cv2.rectangle(panel, (x, y), (x + width - 1, y + height - 1), _draw_color(img, background), cv2.FILLED)
        _blend_region(img, panel, alpha, x, y, x + width, y + height)
    text_color = _draw_color(img, color)
    for i, (text, entry) in enumerate(zip(labels, palette, strict=True)):
        top = y + padding + i * row_h + (row_h - swatch) // 2
        cv2.rectangle(
            img, (x + padding, top), (x + padding + swatch - 1, top + swatch - 1), _draw_color(img, entry), -1
        )
        base = (x + padding + swatch + gap, y + padding + i * row_h + (row_h + text_h) // 2)
        cv2.putText(img, text, base, _FONT, fontScale, text_color, 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------------------------- comparisons


def _fit(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both images as BGR (BGRA alpha dropped), with ``b`` resized to ``a``."""
    a3, b3 = toBGR(a), toBGR(b)
    if b3.shape[:2] != a3.shape[:2]:
        b3 = cv2.resize(b3, (a3.shape[1], a3.shape[0]), interpolation=cv2.INTER_AREA)
    return a3, b3


def _title(img: np.ndarray, text: str, fontScale: float, color: Color) -> np.ndarray:
    (w, h), baseline = cv2.getTextSize(text, _FONT, fontScale, 1)
    strip = np.zeros((h + baseline + 8, img.shape[1], 3), np.uint8)
    strip[:] = np.array(_bgr(color), np.uint8)
    level = 0 if _luminance(color) > 140 else 255
    x = max((img.shape[1] - w) // 2, 2)
    cv2.putText(strip, text, (x, h + 4), _FONT, fontScale, (level, level, level), 1, cv2.LINE_AA)
    return np.concatenate([strip, img], axis=0)


def compareImages(
    a: np.ndarray,
    b: np.ndarray,
    mode: str = "side",
    split: float = 0.5,
    labels: Sequence[str] | None = None,
    gap: int = 0,
    color: Color = (0, 0, 0),
    fontScale: float = 0.5,
) -> np.ndarray:
    """Put two images next to each other for a before/after comparison, as a new BGR image.

    Args:
        a: 8-bit grayscale, BGR or BGRA image (the reference size).
        b: 8-bit image; resized to the size of ``a`` when it differs.
        mode: ``"side"`` (left/right), ``"vertical"`` (top/bottom), ``"split"`` (left part of ``a`` and right
            part of ``b`` with a divider), ``"blend"`` (``cv.addWeighted`` with ``split`` as the weight of ``b``)
            or ``"diff"`` (absolute difference, colorized with ``cv.COLORMAP_JET``).
        split: Position of the divider as a fraction of the width for ``"split"``; blend weight for ``"blend"``.
        labels: Two titles drawn above the images for ``"side"`` / ``"vertical"``; ignored by the other modes.
        gap: Space between the images for ``"side"`` / ``"vertical"``; divider thickness for ``"split"``.
        color: Color of the gap, the divider and the title strips.
        fontScale: Hershey font size of the labels.

    Returns:
        A new BGR image.

    Raises:
        ValueError: If an image is not 8-bit, ``mode`` is unknown, ``split`` is outside ``[0, 1]``, ``gap`` is
            negative or ``labels`` does not have two entries.
    """
    _check_uint8("a", a)
    _check_uint8("b", b)
    if mode not in ("side", "vertical", "split", "blend", "diff"):
        raise ValueError(f"mode must be 'side', 'vertical', 'split', 'blend' or 'diff', got {mode!r}")
    _check_unit("split", split)
    if gap < 0:
        raise ValueError(f"gap must not be negative, got {gap}")
    if labels is not None and len(labels) != 2:
        raise ValueError(f"labels must have two entries, got {len(labels)}")
    left, right = _fit(a, b)
    if mode in ("side", "vertical"):
        if labels is not None:
            left, right = _title(left, labels[0], fontScale, color), _title(right, labels[1], fontScale, color)
        concat = hconcatEx if mode == "side" else vconcatEx
        return concat([left, right], color=_bgr(color), gap=gap)
    if mode == "blend":
        blended: np.ndarray = cv2.addWeighted(left, 1 - split, right, split, 0)
        return blended
    if mode == "diff":
        diff: np.ndarray = cv2.absdiff(left, right)
        return applyColorMapEx(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET, normalize=False)
    out = right.copy()
    x = round(left.shape[1] * split)
    out[:, :x] = left[:, :x]
    if gap > 0:
        x0, x1 = max(x - gap // 2, 0), min(x - gap // 2 + gap, out.shape[1])
        out[:, x0:x1] = np.array(_bgr(color), np.uint8)
    return out


def labeledMontage(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    cols: int | None = None,
    cellSize: tuple[int, int] | None = None,
    gap: int = 4,
    color: Color = (0, 0, 0),
    fontScale: float = 0.5,
) -> np.ndarray:
    """Arrange images in a grid with a title strip above each one, as a new BGR image.

    :func:`montage` places the images only; naming each tile ("input", "blur 3", "blur 5") is the usual next step.

    Args:
        images: 8-bit images; sizes and channels may differ.
        labels: One title per image.
        cols: Number of columns; by default as square as possible.
        cellSize: Fit every image into ``(width, height)`` first with :func:`letterbox`.
        gap: Space between the cells.
        color: Color of the gaps, the padding and the title strips.
        fontScale: Hershey font size of the titles.

    Returns:
        A new BGR image.

    Raises:
        ValueError: If ``images`` is empty, ``labels`` has a different length, or ``cols`` is not positive.
    """
    from .resize import letterbox

    if len(images) == 0:
        raise ValueError("images must not be empty")
    if len(labels) != len(images):
        raise ValueError(f"labels must have one entry per image, got {len(labels)} for {len(images)} images")
    columns = math.ceil(math.sqrt(len(images))) if cols is None else cols
    if columns < 1:
        raise ValueError(f"cols must be positive, got {cols}")
    fill = _bgr(color)
    cells = []
    for img, text in zip(images, labels, strict=True):
        _check_uint8("images", img)
        tile = toBGR(img)
        if cellSize is not None:
            tile = letterbox(tile, cellSize, color=fill)[0]
        cells.append(_title(tile, text, fontScale, color))
    rows = [hconcatEx(cells[i : i + columns], color=fill, gap=gap) for i in range(0, len(cells), columns)]
    return vconcatEx(rows, color=fill, gap=gap)


def blendImages(images: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    """Average any number of images with optional weights (``cv.addWeighted`` takes exactly two).

    Images are resized to the first one and converted to its channel count; the weights are normalized, so
    ``[1, 1, 2]`` means the last image counts twice.

    Args:
        images: 8-bit grayscale, BGR or BGRA images.
        weights: One non-negative weight per image; ``None`` gives equal weights.

    Returns:
        A new 8-bit image with the shape of ``images[0]``.

    Raises:
        ValueError: If ``images`` is empty, an image is not 8-bit, or ``weights`` has a different length, a
            negative or non-finite entry, or sums to zero.
    """
    if len(images) == 0:
        raise ValueError("images must not be empty")
    for img in images:
        _check_uint8("images", img)
    if weights is None:
        weights = [1.0] * len(images)
    if len(weights) != len(images):
        raise ValueError(f"weights must have one entry per image, got {len(weights)} for {len(images)} images")
    for w in weights:
        check_finite("weights", w, nonnegative=True)
    total = float(sum(weights))
    if total == 0:
        raise ValueError("weights must not all be zero")
    first = images[0]
    height, width = first.shape[:2]
    channels = _channels(first)
    convert = {1: toGray, 3: toBGR, 4: toBGRA}[channels]
    acc = np.zeros(first.shape, np.float32)
    for img, w in zip(images, weights, strict=True):
        if _channels(img) != channels:
            img = convert(img)
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        acc += img.astype(np.float32) * (float(w) / total)
    return np.clip(np.rint(acc), 0, 255).astype(np.uint8)
