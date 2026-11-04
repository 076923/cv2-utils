import os
from collections.abc import Sequence
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["putTextEx"]


@lru_cache(maxsize=32)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def putTextEx(
    src: np.ndarray,
    text: str,
    org: tuple[int, int],
    fontPath: str | os.PathLike[str],
    fontSize: int,
    color: Sequence[int],
) -> np.ndarray:
    """Draw text with a TrueType/OpenType font, so any Unicode text (e.g. Korean) can be rendered.

    ``cv.putText`` only supports the built-in Hershey fonts, which cover ASCII only.

    Args:
        src: 8-bit image. Grayscale input is converted to 3 channels.
        text: Text to draw. ``\\n`` starts a new line.
        org: Top-left corner of the text, as ``(x, y)``. Note that ``cv.putText`` uses the
            bottom-left corner instead.
        fontPath: Path to a ``.ttf``/``.otf``/``.ttc`` font file. Loaded fonts are cached.
        fontSize: Font size in pixels.
        color: Text color in the image's channel order (BGR for OpenCV images).

    Returns:
        A new image with the text drawn; ``src`` is not modified.
    """
    if src.ndim == 2 or src.shape[2] == 1:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    font = _load_font(os.fspath(fontPath), fontSize)
    dst = Image.fromarray(src)
    ImageDraw.Draw(dst).text(org, text, font=font, fill=tuple(color))
    return np.array(dst)
