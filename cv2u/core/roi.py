"""Regions of interest: cropping and color ranges."""

from collections.abc import Sequence

import cv2
import numpy as np

__all__ = ["cropEx", "cropToContent", "inRangeEx"]


def cropEx(
    src: np.ndarray,
    rect: tuple[int, int, int, int],
    margin: int = 0,
    fill: float | Sequence[float] | None = None,
) -> np.ndarray:
    """Crop ``rect`` from ``src`` safely, even when the box extends outside the image.

    Plain slicing (``src[y:y+h, x:x+w]``) silently returns the wrong region for negative coordinates, which is common
    with boxes from detectors.

    Args:
        src: Image to crop.
        rect: The box as ``(x, y, width, height)``, e.g. from ``cv.boundingRect`` or a detector.
        margin: Grow the box by this many pixels on every side (negative values shrink it).
        fill: ``None`` (default) clips the box to the image, so the result may be smaller than ``rect``.
            Otherwise the result has exactly the box size, and the parts outside the image are filled with this color.

    Returns:
        A new array (not a view of ``src``). It is empty (size 0) when the box is entirely outside the image and
        ``fill`` is ``None``.

    Raises:
        ValueError: If the box (after ``margin``) has no area.
    """
    x, y, w, h = (int(v) for v in rect)
    x, y, w, h = x - margin, y - margin, w + 2 * margin, h + 2 * margin
    if w <= 0 or h <= 0:
        raise ValueError(f"the box has no area: {(x, y, w, h)}")

    height, width = src.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, width), min(y + h, height)
    if fill is None:
        return src[y0 : max(y0, y1), x0 : max(x0, x1)].copy()

    out = np.empty((h, w, *src.shape[2:]), dtype=src.dtype)
    out[...] = fill  # a scalar, or one value per channel
    if x0 < x1 and y0 < y1:
        out[y0 - y : y1 - y, x0 - x : x1 - x] = src[y0:y1, x0:x1]
    return out


def cropToContent(
    src: np.ndarray,
    background: float | Sequence[float] | None = None,
    tolerance: float = 0,
    margin: int = 0,
) -> np.ndarray:
    """Crop away a uniform border, e.g. the white margin around a scan or the black bars around a video frame.

    Args:
        src: Image to trim.
        background: Border color. By default the color of the top-left pixel.
        tolerance: Pixels whose every channel differs from ``background`` by at most this much count as border,
            which absorbs JPEG noise.
        margin: Keep this many pixels of border around the content (limited by the image size).

    Returns:
        A new array with the content. If the whole image is border, a copy of ``src`` is returned unchanged.

    Raises:
        ValueError: If ``tolerance`` or ``margin`` is negative.
    """
    if tolerance < 0 or margin < 0:
        raise ValueError(f"tolerance and margin must not be negative, got {tolerance} and {margin}")
    pixels = src.astype(np.float64)
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    reference = (
        pixels[0, 0] if background is None else np.broadcast_to(np.asarray(background, np.float64), pixels.shape[2:])
    )
    content = (np.abs(pixels - reference) > tolerance).any(axis=2)
    rows, cols = np.nonzero(content.any(axis=1))[0], np.nonzero(content.any(axis=0))[0]
    if rows.size == 0:
        return src.copy()
    y0, y1 = max(int(rows[0]) - margin, 0), min(int(rows[-1]) + 1 + margin, src.shape[0])
    x0, x1 = max(int(cols[0]) - margin, 0), min(int(cols[-1]) + 1 + margin, src.shape[1])
    return src[y0:y1, x0:x1].copy()


def inRangeEx(src: np.ndarray, lowerb: Sequence[float], upperb: Sequence[float]) -> np.ndarray:
    """``cv.inRange`` for HSV images where the hue range may wrap around (e.g. red: hue 170 to 10).

    Hue is circular, so a range like red crosses the end of the scale. With ``cv.inRange``, ``lowerb[0] > upperb[0]``
    silently selects nothing; here it selects the hues ``>= lowerb[0]`` **or** ``<= upperb[0]``. Otherwise this is
    exactly ``cv.inRange``. Works with ``cv.COLOR_BGR2HSV`` (hue 0-179) and ``cv.COLOR_BGR2HSV_FULL`` (hue 0-255).

    Args:
        src: 3-channel HSV image.
        lowerb: Lower bound ``(h, s, v)``.
        upperb: Upper bound ``(h, s, v)``.

    Returns:
        Single-channel mask (0 or 255).

    Raises:
        ValueError: If ``src`` does not have 3 channels or the bounds do not have 3 values.
    """
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError(f"src must be a 3-channel (HSV) image, got shape {src.shape}")
    if len(lowerb) != 3 or len(upperb) != 3:
        raise ValueError("lowerb and upperb must have 3 values (h, s, v)")
    lo, hi = tuple(float(v) for v in lowerb), tuple(float(v) for v in upperb)
    if lo[0] <= hi[0]:
        mask: np.ndarray = cv2.inRange(src, lo, hi)
        return mask
    hue_max = float(np.iinfo(src.dtype).max) if np.issubdtype(src.dtype, np.integer) else 360.0
    upper_part = cv2.inRange(src, lo, (hue_max, hi[1], hi[2]))
    lower_part = cv2.inRange(src, (0.0, lo[1], lo[2]), hi)
    combined: np.ndarray = cv2.bitwise_or(upper_part, lower_part)
    return combined
