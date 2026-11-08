import os
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .._validate import check_channels, check_color, check_image_not_empty

__all__ = ["putTextEx"]


@lru_cache(maxsize=32)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def _fill(color: float | Sequence[float], channels: int) -> int | tuple[int, ...]:
    """The Pillow fill for an OpenCV color, drawn directly on the array so the channel order is kept."""
    values = [float(color)] if isinstance(color, (int, float)) else [float(c) for c in color]
    check_color("color", values, np.dtype(np.uint8))
    if channels == 1:
        if len(values) >= 3:
            b, g, r = values[:3]
            values = [0.114 * b + 0.587 * g + 0.299 * r]  # a BGR color on a grayscale image: its luminance
        return round(values[0])
    if len(values) == 1:
        values = values * 3
    if len(values) < 3:
        raise ValueError(f"color must be a scalar or have at least 3 values (BGR), got {color!r}")
    bgr = tuple(round(v) for v in values[:3])
    return (*bgr, round(values[3]) if len(values) > 3 else 255) if channels == 4 else bgr


def putTextEx(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    fontPath: str | os.PathLike[str],
    fontSize: int,
    color: float | Sequence[float],
) -> np.ndarray:
    """Draw text with a TrueType/OpenType font, so any Unicode text (e.g. Korean) can be rendered.

    ``cv.putText`` only supports the built-in Hershey fonts, which cover ASCII only. Like ``cv.putText``, this
    draws **in place** and returns ``img``.

    Args:
        img: 8-bit grayscale, BGR or BGRA image to draw on. Grayscale images stay grayscale.
        text: Text to draw. ``\\n`` starts a new line.
        org: Top-left corner of the text, as ``(x, y)``. Note that ``cv.putText`` uses the
            bottom-left corner instead.
        fontPath: Path to a ``.ttf``/``.otf``/``.ttc`` font file. Loaded fonts are cached.
        fontSize: Font size in pixels.
        color: Text color (BGR, or a scalar). On grayscale images, the luminance of a BGR color is used.

    Returns:
        ``img``, with the text drawn on it.

    Raises:
        ValueError: If ``img`` is not 8-bit, is empty or has an unsupported channel count, ``fontSize`` is not
            positive, or ``color`` is outside ``[0, 255]``.
    """
    if img.dtype != np.uint8:
        raise ValueError(f"img must be an 8-bit (uint8) image, got dtype {img.dtype}")
    check_image_not_empty("img", img)
    channels = check_channels("img", img)
    if fontSize < 1:
        raise ValueError(f"fontSize must be positive, got {fontSize}")
    fill = _fill(color, channels)
    font = _load_font(os.fspath(fontPath), fontSize)
    canvas = Image.fromarray(np.ascontiguousarray(img.reshape(img.shape[:2]) if channels == 1 else img))
    ImageDraw.Draw(canvas).text(org, text, font=font, fill=fill)
    img[...] = np.asarray(canvas).reshape(img.shape)
    return img
