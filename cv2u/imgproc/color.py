"""Channel conversions that work whatever the input already is, and color notations."""

import string
from collections.abc import Sequence

import cv2
import numpy as np

from .._validate import check_color, check_image_not_empty

__all__ = [
    "bgrToHex",
    "distinctColors",
    "flattenAlpha",
    "hexToBGR",
    "toBGR",
    "toBGRA",
    "toGray",
]


def _channels(src: np.ndarray) -> int:
    channels = 1 if src.ndim == 2 else int(src.shape[2]) if src.ndim == 3 else 0
    if channels not in (1, 3, 4):
        raise ValueError(f"src must have 1, 3 or 4 channels, got shape {src.shape}")
    check_image_not_empty("src", src)
    return channels


def _gray2d(src: np.ndarray) -> np.ndarray:
    return src.reshape(src.shape[:2])


def toGray(src: np.ndarray) -> np.ndarray:
    """Convert a grayscale, BGR or BGRA image to single-channel grayscale.

    ``cv.cvtColor(img, cv.COLOR_BGR2GRAY)`` raises an error when ``img`` is already grayscale, so code that accepts
    any image needs a channel check first. Grayscale input is returned as a copy.

    Raises:
        ValueError: If ``src`` does not have 1, 3 or 4 channels.
    """
    channels = _channels(src)
    if channels == 1:
        return _gray2d(src).copy()
    out: np.ndarray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY if channels == 3 else cv2.COLOR_BGRA2GRAY)
    return out


def toBGR(src: np.ndarray) -> np.ndarray:
    """Convert a grayscale, BGR or BGRA image to 3-channel BGR (the alpha channel is dropped).

    Raises:
        ValueError: If ``src`` does not have 1, 3 or 4 channels.
    """
    channels = _channels(src)
    if channels == 3:
        return src.copy()
    if channels == 1:
        gray: np.ndarray = cv2.cvtColor(_gray2d(src), cv2.COLOR_GRAY2BGR)
        return gray
    bgr: np.ndarray = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
    return bgr


def toBGRA(src: np.ndarray, alpha: float | None = None) -> np.ndarray:
    """Convert a grayscale, BGR or BGRA image to 4-channel BGRA.

    Args:
        src: Image to convert.
        alpha: Alpha value for images without an alpha channel. By default fully opaque (255 for 8-bit images,
            65535 for 16-bit, 1.0 for float). For BGRA input, the existing alpha is kept unless this is given.

    Raises:
        ValueError: If ``src`` does not have 1, 3 or 4 channels, or ``alpha`` is not a finite number.
    """
    if alpha is not None:
        check_color("alpha", [alpha], src.dtype)
    channels = _channels(src)
    if channels == 4:
        out = src.copy()
    elif channels == 1:
        out = cv2.cvtColor(_gray2d(src), cv2.COLOR_GRAY2BGRA)
    else:
        out = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
    if alpha is not None:
        out[..., 3] = alpha
    return out


def hexToBGR(hexColor: str) -> tuple[int, ...]:
    """Convert a CSS-style hex color to an OpenCV BGR tuple, e.g. ``"#ff8800"`` to ``(0, 136, 255)``.

    Accepts ``"#rrggbb"``, ``"#rgb"`` and, with alpha, ``"#rrggbbaa"`` / ``"#rgba"`` (giving ``(b, g, r, a)``);
    the ``#`` is optional. Saves reversing the channel order by hand, which is easy to get wrong.

    Raises:
        ValueError: If the text is not a hex color.
    """
    text = hexColor.strip().removeprefix("#")
    if len(text) in (3, 4):
        text = "".join(c * 2 for c in text)
    if len(text) not in (6, 8) or any(c not in string.hexdigits for c in text):
        raise ValueError(f"not a hex color: {hexColor!r}")
    r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r, int(text[6:8], 16)) if len(text) == 8 else (b, g, r)


def bgrToHex(color: Sequence[float]) -> str:
    """Convert an OpenCV BGR (or BGRA) color to a hex string, e.g. ``(0, 136, 255)`` to ``"#ff8800"``.

    Raises:
        ValueError: If ``color`` does not have 3 or 4 values in ``[0, 255]``.
    """
    values = [round(float(c)) for c in color]
    if len(values) not in (3, 4) or not all(0 <= v <= 255 for v in values):
        raise ValueError(f"color must have 3 or 4 values in [0, 255], got {color!r}")
    b, g, r = values[:3]
    return f"#{r:02x}{g:02x}{b:02x}" + (f"{values[3]:02x}" if len(values) == 4 else "")


def flattenAlpha(src: np.ndarray, background: float | Sequence[float] | np.ndarray = (255, 255, 255)) -> np.ndarray:
    """Composite a transparent (BGRA) image onto a background and return 3-channel BGR.

    ``cv.cvtColor(img, cv.COLOR_BGRA2BGR)`` just drops the alpha channel, so transparent areas show whatever color
    the hidden pixels happen to have (often black). Use this before saving as JPEG or passing to code that expects BGR.

    Args:
        src: Image to flatten. Images without an alpha channel are converted with ``toBGR``.
        background: Background color (white by default), or a background image of the same height and width.

    Returns:
        A BGR image with the dtype of ``src``.

    Raises:
        ValueError: If ``src`` does not have 1, 3 or 4 channels, or a background image has a different size.
    """
    if _channels(src) != 4:
        return toBGR(src)
    peak = float(np.iinfo(src.dtype).max) if np.issubdtype(src.dtype, np.integer) else 1.0
    alpha = src[..., 3:4].astype(np.float32) / peak
    if isinstance(background, np.ndarray) and background.ndim >= 2:
        if background.shape[:2] != src.shape[:2]:
            raise ValueError(f"background image must be {src.shape[1]}x{src.shape[0]}, got shape {background.shape}")
        bg = toBGR(background).astype(np.float32)
    else:
        values = np.atleast_1d(np.asarray(background, dtype=np.float32)).ravel()
        bg = np.resize(values, 3) if values.size < 3 else values[:3]
    out = src[..., :3].astype(np.float32) * alpha + bg * (1 - alpha)
    if np.issubdtype(src.dtype, np.integer):
        out = np.clip(np.rint(out), 0, peak)
    flat: np.ndarray = out.astype(src.dtype)
    return flat


def distinctColors(count: int, seed: int | None = None) -> list[tuple[int, int, int]]:
    """``count`` BGR colors that are easy to tell apart, for drawing classes, tracks or clusters.

    The first 12 are a fixed palette (red, green, blue, yellow, magenta, cyan, ...); further colors step around
    the hue circle by the golden ratio, so the same ``count`` always gives the same colors and no two are equal.

    Args:
        count: How many colors.
        seed: Shuffle the colors with this random seed; ``None`` keeps the fixed order.

    Raises:
        ValueError: If ``count`` is negative.
    """
    if count < 0:
        raise ValueError(f"count must not be negative, got {count}")
    colors = list(_PALETTE[:count])
    index = 0
    while len(colors) < count:
        # Golden-ratio hue steps never repeat, with three brightness levels for more contrast between neighbors.
        hue = (index * 0.618033988749895) % 1.0
        value = (255, 180, 120)[index % 3]
        hsv = np.array([[[hue * 180, 255, value]]], dtype=np.uint8)
        b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        color = (int(b), int(g), int(r))
        if color not in colors:
            colors.append(color)
        index += 1
    if seed is not None:
        np.random.default_rng(seed).shuffle(colors)
    return colors


_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 128, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 255, 128),
    (128, 255, 0),
    (255, 0, 128),
]
