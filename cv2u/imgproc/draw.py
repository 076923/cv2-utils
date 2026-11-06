"""Draw annotations: labeled boxes and segmentation masks."""

import os
from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .puttext import _load_font

__all__ = [
    "drawBoxes",
    "drawDashedLine",
    "drawDashedRectangle",
    "drawGrid",
    "drawHistogram",
    "drawLabel",
    "drawPoints",
    "drawRoundedRectangle",
    "overlayMask",
]

Color = float | Sequence[float]


def _bgr(color: Color) -> tuple[float, float, float]:
    if isinstance(color, (int, float)):
        return (float(color),) * 3
    values = [float(c) for c in color]
    if len(values) == 1:
        return (values[0],) * 3
    if len(values) < 3:
        raise ValueError(f"color must be a scalar or have at least 3 values (BGR), got {color!r}")
    return values[0], values[1], values[2]


def _luminance(color: Color) -> float:
    b, g, r = _bgr(color)
    return 0.114 * b + 0.587 * g + 0.299 * r


def _channels(img: np.ndarray) -> int:
    return 1 if img.ndim == 2 else int(img.shape[2])


def _pil_fill(color: Color, channels: int) -> int | tuple[int, ...]:
    """Color for drawing with Pillow directly on an OpenCV array (channel order is kept as is)."""
    if channels == 1:
        return round(_luminance(color)) if not isinstance(color, (int, float)) else round(color)
    bgr = tuple(round(c) for c in _bgr(color))
    return (*bgr, 255) if channels == 4 else bgr


def drawLabel(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    text: str,
    color: Color,
    thickness: int = 2,
    fontScale: float = 0.5,
    textColor: Color | None = None,
    fontPath: str | os.PathLike[str] | None = None,
    fontSize: int = 16,
) -> np.ndarray:
    """Draw a box with a filled label tag above it, like the output of an object detector.

    Like ``cv.rectangle`` and the other OpenCV drawing functions, this draws **in place** and returns ``img``.
    The tag moves inside the box when there is no room above it.

    Args:
        img: 8-bit grayscale, BGR or BGRA image to draw on.
        rect: The box as ``(x, y, width, height)``, e.g. from ``cv.boundingRect``.
        text: Label text. The built-in OpenCV font supports ASCII only; pass ``fontPath`` for any Unicode text
            (e.g. Korean).
        color: Box and tag color (BGR).
        thickness: Box line thickness.
        fontScale: Size of the built-in OpenCV font.
        textColor: Text color. By default black or white, whichever contrasts better with ``color``.
        fontPath: TrueType/OpenType font for the label; then ``fontSize`` (pixels) is used instead of ``fontScale``.
        fontSize: Font size in pixels when ``fontPath`` is given.

    Returns:
        ``img``, with the box and label drawn on it.

    Raises:
        ValueError: If ``img`` is not 8-bit.
    """
    if img.dtype != np.uint8:
        raise ValueError(f"img must be an 8-bit (uint8) image, got dtype {img.dtype}")
    channels = _channels(img)
    if channels == 1 and not isinstance(color, (int, float)):
        # OpenCV would use only color[0] on a grayscale image, e.g. (0, 200, 255) would draw black.
        color = _luminance(color)
    x, y, w, h = (int(v) for v in rect)
    cv2.rectangle(img, (x, y, w, h), color, thickness)
    if textColor is None:
        level = 0 if _luminance(color) > 140 else 255
        textColor = level if channels == 1 else (level, level, level)

    pad = 3
    font = None if fontPath is None else _load_font(os.fspath(fontPath), fontSize)
    if font is None:
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)
        tag_h = text_h + baseline + 2 * pad
        left = top = 0
    else:
        left, top, right, bottom = (int(v) for v in font.getbbox(text))
        text_w, text_h = right - left, bottom - top
        tag_h = text_h + 2 * pad
    tag_w = text_w + 2 * pad
    edge = thickness // 2
    tag_x = x - edge
    tag_y = y - edge - tag_h if y - edge - tag_h >= 0 else y + edge  # above the box, or inside when no room
    cv2.rectangle(img, (tag_x, tag_y), (tag_x + tag_w - 1, tag_y + tag_h - 1), color, cv2.FILLED)

    if font is None:
        origin = (tag_x + pad, tag_y + pad + text_h)
        cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, fontScale, textColor, 1, cv2.LINE_AA)
        return img

    # Draw the text with Pillow on the tag area only, then copy it back, so that drawing stays in place.
    x0, y0 = max(tag_x, 0), max(tag_y, 0)
    x1, y1 = min(tag_x + tag_w, img.shape[1]), min(tag_y + tag_h, img.shape[0])
    if x0 >= x1 or y0 >= y1:
        return img
    region = img[y0:y1, x0:x1]
    patch = Image.fromarray(np.ascontiguousarray(region.reshape(region.shape[:2]) if channels == 1 else region))
    position = (tag_x + pad - x0 - left, tag_y + pad - y0 - top)
    ImageDraw.Draw(patch).text(position, text, font=font, fill=_pil_fill(textColor, channels))
    region[...] = np.asarray(patch).reshape(region.shape)
    return img


def overlayMask(src: np.ndarray, mask: np.ndarray, color: Color = (0, 0, 255), alpha: float = 0.5) -> np.ndarray:
    """Tint the pixels selected by ``mask`` with ``color``, e.g. to visualize a segmentation.

    Args:
        src: 8-bit grayscale, BGR or BGRA image. It is not modified.
        mask: 2-D array with the shape ``src.shape[:2]``; nonzero (or ``True``) pixels are tinted.
        color: Tint color (BGR). For grayscale images, its luminance is used.
        alpha: Opacity of the tint, from 0 to 1.

    Returns:
        A new image with the same shape as ``src``.

    Raises:
        ValueError: If ``src`` is not 8-bit, the mask shape does not match, or ``alpha`` is outside ``[0, 1]``.
    """
    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    if mask.ndim != 2 or mask.shape != src.shape[:2]:
        raise ValueError(f"mask must have the shape {src.shape[:2]}, got {mask.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    dst = src.copy()
    selected = mask.astype(bool)
    channels = _channels(src)
    tint = np.array([_luminance(color)] if channels == 1 else _bgr(color), dtype=np.float32)

    pixels = dst[selected]  # (N,) for grayscale, (N, C) for color
    if channels == 1:
        pixels = pixels.reshape(-1, 1)
    blended = pixels[:, : tint.size].astype(np.float32) * (1 - alpha) + tint * alpha
    pixels[:, : tint.size] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    dst[selected] = pixels.reshape(dst[selected].shape)
    return dst


def drawDashedLine(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: Color,
    thickness: int = 1,
    dashLength: int = 10,
    gapLength: int | None = None,
    lineType: int = cv2.LINE_8,
) -> np.ndarray:
    """Draw a dashed line, in place like ``cv.line`` (which only draws solid lines).

    Args:
        img: Image to draw on.
        pt1: Start point ``(x, y)``.
        pt2: End point ``(x, y)``.
        color: Line color.
        thickness: Line thickness.
        dashLength: Length of each dash in pixels.
        gapLength: Length of each gap; by default equal to ``dashLength``.
        lineType: As for ``cv.line``.

    Returns:
        ``img``.

    Raises:
        ValueError: If ``dashLength`` or ``gapLength`` is not positive.
    """
    gap = dashLength if gapLength is None else gapLength
    if dashLength < 1 or gap < 1:
        raise ValueError(f"dashLength and gapLength must be positive, got {dashLength} and {gap}")
    start, end = np.array(pt1, np.float64), np.array(pt2, np.float64)
    length = float(np.hypot(*(end - start)))
    if length == 0:
        return img
    direction = (end - start) / length
    position = 0.0
    while position < length:
        a = start + direction * position
        b = start + direction * min(position + dashLength, length)
        cv2.line(img, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), color, thickness, lineType)
        position += dashLength + gap
    return img


def drawDashedRectangle(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    color: Color,
    thickness: int = 1,
    dashLength: int = 10,
    gapLength: int | None = None,
) -> np.ndarray:
    """Draw a dashed rectangle ``(x, y, width, height)``, in place like ``cv.rectangle``.

    Returns:
        ``img``.
    """
    x, y, w, h = (int(v) for v in rect)
    x1, y1 = x + w - 1, y + h - 1
    for p, q in (((x, y), (x1, y)), ((x1, y), (x1, y1)), ((x1, y1), (x, y1)), ((x, y1), (x, y))):
        drawDashedLine(img, p, q, color, thickness, dashLength, gapLength)
    return img


def drawHistogram(src: np.ndarray, size: tuple[int, int] = (512, 200), log: bool = False) -> np.ndarray:
    """Plot the histogram of an 8-bit image (one curve per channel, in its own color) as a new image.

    ``cv.calcHist`` only returns the numbers.

    Args:
        src: 8-bit grayscale, BGR or BGRA image (alpha is ignored).
        size: Plot size as ``(width, height)``.
        log: Use a logarithmic vertical scale, which shows small counts next to large peaks.
            On a linear scale, the few tallest spikes are clipped at the top.

    Returns:
        A BGR image with a dark background.

    Raises:
        ValueError: If ``src`` is not 8-bit.
    """
    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    width, height = size
    plot = np.full((height, width, 3), 24, np.uint8)
    channels = 1 if src.ndim == 2 else min(src.shape[2], 3)
    colors = [(230, 230, 230)] if channels == 1 else [(255, 120, 40), (60, 200, 60), (40, 60, 255)]
    images = [src] if channels == 1 else [np.ascontiguousarray(src[..., c]) for c in range(channels)]
    hists = [cv2.calcHist([img], [0], None, [256], [0, 256]).ravel() for img in images]
    if log:
        hists = [np.log1p(h) for h in hists]
    # Scale to the 98th percentile of the bins, so that a few tall spikes (e.g. a black background) are clipped
    # at the top instead of flattening the rest of the histogram.
    top = float(np.percentile(np.concatenate(hists), 98)) * 1.1 or max(float(np.max(h)) for h in hists) or 1.0
    xs = np.linspace(0, width - 1, 256)
    for hist, color in zip(hists, colors, strict=True):
        ys = height - 1 - np.minimum(hist / top, 1.0) * (height - 1)
        points = np.stack([xs, ys], axis=1).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(plot, [points], False, color, 1, cv2.LINE_AA)
    return plot


def drawRoundedRectangle(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    color: Color,
    thickness: int = 1,
    lineType: int = cv2.LINE_8,
    radius: int = 8,
) -> np.ndarray:
    """Draw a rectangle ``(x, y, width, height)`` with rounded corners, in place like ``cv.rectangle``.

    Args:
        img: Image to draw on.
        rect: The box, as for ``drawDashedRectangle``.
        color: Line or fill color.
        thickness: Line thickness; ``cv.FILLED`` (negative) fills the shape.
        lineType: As for ``cv.rectangle`` (``cv.LINE_AA`` for smooth corners).
        radius: Corner radius in pixels, reduced to fit small boxes.

    Returns:
        ``img``.

    Raises:
        ValueError: If the box is empty or ``radius`` is negative.
    """
    x, y, w, h = (int(v) for v in rect)
    if w < 1 or h < 1:
        raise ValueError(f"rect must have a positive size, got {w}x{h}")
    if radius < 0:
        raise ValueError(f"radius must not be negative, got {radius}")
    x1, y1 = x + w - 1, y + h - 1
    r = min(int(radius), (w - 1) // 2, (h - 1) // 2)
    corners = (((x + r, y + r), 180), ((x1 - r, y + r), 270), ((x1 - r, y1 - r), 0), ((x + r, y1 - r), 90))
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x1 - r, y1), color, cv2.FILLED, lineType)
        cv2.rectangle(img, (x, y + r), (x1, y1 - r), color, cv2.FILLED, lineType)
        for center, start in corners:
            cv2.ellipse(img, center, (r, r), 0, start, start + 90, color, cv2.FILLED, lineType)
        return img
    for p, q in (
        ((x + r, y), (x1 - r, y)),
        ((x1, y + r), (x1, y1 - r)),
        ((x + r, y1), (x1 - r, y1)),
        ((x, y + r), (x, y1 - r)),
    ):
        cv2.line(img, p, q, color, thickness, lineType)
    for center, start in corners:
        cv2.ellipse(img, center, (r, r), 0, start, start + 90, color, thickness, lineType)
    return img


def drawBoxes(
    img: np.ndarray,
    boxes: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[str] | None = None,
    colors: Sequence[Color] | Color | None = None,
    thickness: int = 2,
    fontScale: float = 0.5,
    fontPath: str | os.PathLike[str] | None = None,
    fontSize: int = 16,
) -> np.ndarray:
    """Draw many boxes ``(x, y, width, height)`` at once, each with an optional label tag, in place.

    The loop around :func:`drawLabel` that every detection demo writes, with one color per box from a fixed
    palette by default.

    Args:
        img: Image to draw on.
        boxes: Boxes as ``(N, 4)``; floats are rounded.
        labels: One text per box, or ``None`` for plain boxes.
        colors: One color per box, one color for all, or ``None`` for :func:`distinctColors`.
        thickness: Line thickness.
        fontScale: Hershey font size for the labels.
        fontPath: TrueType font for Unicode labels (see :func:`drawLabel`).
        fontSize: Font size in pixels for ``fontPath``.

    Returns:
        ``img``.

    Raises:
        ValueError: If ``labels`` or ``colors`` has a different length than ``boxes``.
    """
    from .color import distinctColors

    array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4) if np.size(boxes) else np.zeros((0, 4))
    count = len(array)
    if labels is not None and len(labels) != count:
        raise ValueError(f"labels must have one entry per box, got {len(labels)} for {count} boxes")
    palette: list[Color]
    if colors is None:
        palette = list(distinctColors(max(count, 1)))
    elif isinstance(colors, (int, float)) or (
        len(colors) in (3, 4) and all(isinstance(c, (int, float)) for c in colors)
    ):
        palette = [colors] * max(count, 1)  # type: ignore[list-item]
    else:
        palette = list(colors)
        if len(palette) != count:
            raise ValueError(f"colors must have one entry per box, got {len(palette)} for {count} boxes")
    for i, box in enumerate(array):
        x, y, w, h = (round(v) for v in box)
        if w < 1 or h < 1:
            continue
        color = palette[i % len(palette)]
        if labels is None:
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, thickness)
        else:
            drawLabel(img, (x, y, w, h), labels[i], color, thickness, fontScale, fontPath=fontPath, fontSize=fontSize)
    return img


def drawPoints(
    img: np.ndarray,
    points: Sequence[Sequence[float]] | np.ndarray,
    color: Color = (0, 0, 255),
    radius: int = 3,
    thickness: int = cv2.FILLED,
    labels: bool | Sequence[str] = False,
    fontScale: float = 0.4,
) -> np.ndarray:
    """Draw points (keypoints, corners, landmarks) as circles, optionally numbered, in place.

    Args:
        img: Image to draw on.
        points: ``(N, 2)`` coordinates, or an ``(N, 1, 2)`` contour; floats are rounded.
        color: Circle color.
        radius: Circle radius in pixels.
        thickness: ``cv.FILLED`` (default) or an outline thickness.
        labels: ``True`` writes the index next to each point; a sequence gives the texts.
        fontScale: Hershey font size for the labels.

    Returns:
        ``img``.

    Raises:
        ValueError: If the points are not 2-D, ``radius`` is negative, or ``labels`` has a different length.
    """
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2) if np.size(points) else np.zeros((0, 2))
    if not isinstance(labels, bool) and len(labels) != len(array):
        raise ValueError(f"labels must have one entry per point, got {len(labels)} for {len(array)} points")
    if radius < 0:
        raise ValueError(f"radius must not be negative, got {radius}")
    for i, (x, y) in enumerate(array):
        center = (round(x), round(y))
        cv2.circle(img, center, radius, color, thickness, cv2.LINE_AA)
        if labels is False:
            continue
        text = str(i) if labels is True else str(labels[i])
        org = (center[0] + radius + 2, center[1] + radius)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, 1, cv2.LINE_AA)
    return img


def drawGrid(
    img: np.ndarray,
    step: int | tuple[int, int],
    color: Color = (0, 255, 0),
    thickness: int = 1,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Draw a grid of lines every ``step`` pixels, in place (for tile layouts, calibration or alignment checks).

    Args:
        img: Image to draw on.
        step: Cell size, one number or ``(width, height)``.
        color: Line color.
        thickness: Line thickness.
        offset: ``(x, y)`` of the first vertical and horizontal line.

    Returns:
        ``img``.

    Raises:
        ValueError: If ``step`` is not positive.
    """
    step_x, step_y = (step, step) if isinstance(step, int) else step
    if step_x < 1 or step_y < 1:
        raise ValueError(f"step must be positive, got {step}")
    height, width = img.shape[:2]
    for x in range(offset[0] % step_x, width, step_x):
        cv2.line(img, (x, 0), (x, height - 1), color, thickness)
    for y in range(offset[1] % step_y, height, step_y):
        cv2.line(img, (0, y), (width - 1, y), color, thickness)
    return img
