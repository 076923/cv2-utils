"""Color arithmetic on BGR tuples: lighten, blend, compare, name and generate colors for drawing.

OpenCV converts whole images between color spaces but has nothing for a single color: lightening a box color
for its fill, picking black or white text for a background, or naming the dominant color all start with a 1x1
``cv.cvtColor`` and some arithmetic. Colors here are BGR tuples of ints in ``[0, 255]``, as drawn by ``cv.rectangle``.
"""

from collections.abc import Sequence

import cv2
import numpy as np

from .._validate import check_finite
from .color import hexToBGR
from .draw import _luminance

__all__ = [
    "CSS_COLORS",
    "blendColors",
    "colorDistance",
    "colorName",
    "colormapPalette",
    "complementaryColor",
    "darkenColor",
    "gradientPalette",
    "invertColor",
    "lightenColor",
    "nearestColor",
    "randomColors",
    "textColorFor",
]

Color = tuple[int, int, int]

_CSS_HEX = {
    "aliceblue": "f0f8ff", "antiquewhite": "faebd7", "aqua": "00ffff", "aquamarine": "7fffd4", "azure": "f0ffff",
    "beige": "f5f5dc", "bisque": "ffe4c4", "black": "000000", "blanchedalmond": "ffebcd", "blue": "0000ff",
    "blueviolet": "8a2be2", "brown": "a52a2a", "burlywood": "deb887", "cadetblue": "5f9ea0", "chartreuse": "7fff00",
    "chocolate": "d2691e", "coral": "ff7f50", "cornflowerblue": "6495ed", "cornsilk": "fff8dc", "crimson": "dc143c",
    "cyan": "00ffff", "darkblue": "00008b", "darkcyan": "008b8b", "darkgoldenrod": "b8860b", "darkgray": "a9a9a9",
    "darkgreen": "006400", "darkgrey": "a9a9a9", "darkkhaki": "bdb76b", "darkmagenta": "8b008b",
    "darkolivegreen": "556b2f", "darkorange": "ff8c00", "darkorchid": "9932cc", "darkred": "8b0000",
    "darksalmon": "e9967a", "darkseagreen": "8fbc8f", "darkslateblue": "483d8b", "darkslategray": "2f4f4f",
    "darkslategrey": "2f4f4f", "darkturquoise": "00ced1", "darkviolet": "9400d3", "deeppink": "ff1493",
    "deepskyblue": "00bfff", "dimgray": "696969", "dimgrey": "696969", "dodgerblue": "1e90ff", "firebrick": "b22222",
    "floralwhite": "fffaf0", "forestgreen": "228b22", "fuchsia": "ff00ff", "gainsboro": "dcdcdc",
    "ghostwhite": "f8f8ff", "gold": "ffd700", "goldenrod": "daa520", "gray": "808080", "green": "008000",
    "greenyellow": "adff2f", "grey": "808080", "honeydew": "f0fff0", "hotpink": "ff69b4", "indianred": "cd5c5c",
    "indigo": "4b0082", "ivory": "fffff0", "khaki": "f0e68c", "lavender": "e6e6fa", "lavenderblush": "fff0f5",
    "lawngreen": "7cfc00", "lemonchiffon": "fffacd", "lightblue": "add8e6", "lightcoral": "f08080",
    "lightcyan": "e0ffff", "lightgoldenrodyellow": "fafad2", "lightgray": "d3d3d3", "lightgreen": "90ee90",
    "lightgrey": "d3d3d3", "lightpink": "ffb6c1", "lightsalmon": "ffa07a", "lightseagreen": "20b2aa",
    "lightskyblue": "87cefa", "lightslategray": "778899", "lightslategrey": "778899", "lightsteelblue": "b0c4de",
    "lightyellow": "ffffe0", "lime": "00ff00", "limegreen": "32cd32", "linen": "faf0e6", "magenta": "ff00ff",
    "maroon": "800000", "mediumaquamarine": "66cdaa", "mediumblue": "0000cd", "mediumorchid": "ba55d3",
    "mediumpurple": "9370db", "mediumseagreen": "3cb371", "mediumslateblue": "7b68ee",
    "mediumspringgreen": "00fa9a", "mediumturquoise": "48d1cc", "mediumvioletred": "c71585",
    "midnightblue": "191970", "mintcream": "f5fffa", "mistyrose": "ffe4e1", "moccasin": "ffe4b5",
    "navajowhite": "ffdead", "navy": "000080", "oldlace": "fdf5e6", "olive": "808000", "olivedrab": "6b8e23",
    "orange": "ffa500", "orangered": "ff4500", "orchid": "da70d6", "palegoldenrod": "eee8aa", "palegreen": "98fb98",
    "paleturquoise": "afeeee", "palevioletred": "db7093", "papayawhip": "ffefd5", "peachpuff": "ffdab9",
    "peru": "cd853f", "pink": "ffc0cb", "plum": "dda0dd", "powderblue": "b0e0e6", "purple": "800080",
    "rebeccapurple": "663399", "red": "ff0000", "rosybrown": "bc8f8f", "royalblue": "4169e1",
    "saddlebrown": "8b4513", "salmon": "fa8072", "sandybrown": "f4a460", "seagreen": "2e8b57", "seashell": "fff5ee",
    "sienna": "a0522d", "silver": "c0c0c0", "skyblue": "87ceeb", "slateblue": "6a5acd", "slategray": "708090",
    "slategrey": "708090", "snow": "fffafa", "springgreen": "00ff7f", "steelblue": "4682b4", "tan": "d2b48c",
    "teal": "008080", "thistle": "d8bfd8", "tomato": "ff6347", "turquoise": "40e0d0", "violet": "ee82ee",
    "wheat": "f5deb3", "white": "ffffff", "whitesmoke": "f5f5f5", "yellow": "ffff00", "yellowgreen": "9acd32",
}  # fmt: skip

CSS_COLORS: dict[str, Color] = {name: (hexToBGR(h)[0], hexToBGR(h)[1], hexToBGR(h)[2]) for name, h in _CSS_HEX.items()}
"""The 148 CSS named colors (``"red"``, ``"steelblue"``, ...) as BGR tuples, for :func:`colorName` and drawing."""


def _color(name: str, color: Sequence[float]) -> Color:
    """Validate a BGR color and return it as a tuple of 3 ints in [0, 255]."""
    values = [float(c) for c in color]
    if len(values) < 3:
        raise ValueError(f"{name} must be a BGR color with 3 values, got {color!r}")
    if not all(np.isfinite(v) and 0 <= v <= 255 for v in values[:3]):
        raise ValueError(f"{name} must have values in [0, 255], got {color!r}")
    return round(values[0]), round(values[1]), round(values[2])


def _convert(color: Color, code: int) -> np.ndarray:
    """Convert one 8-bit BGR color through ``cv.cvtColor`` in float32 (full-range H 0-360, L/S/Lab unscaled)."""
    pixel = np.array([[color]], dtype=np.float32) / 255.0
    out: np.ndarray = cv2.cvtColor(pixel, code)[0, 0]
    return out


def _from_hls(hls: np.ndarray) -> Color:
    bgr = cv2.cvtColor(np.array([[hls]], dtype=np.float32), cv2.COLOR_HLS2BGR)[0, 0] * 255.0
    b, g, r = np.clip(np.rint(bgr), 0, 255)
    return int(b), int(g), int(r)


def _amount(amount: float) -> float:
    check_finite("amount", amount)
    if not 0 <= amount <= 1:
        raise ValueError(f"amount must be in [0, 1], got {amount}")
    return float(amount)


def lightenColor(color: Sequence[float], amount: float) -> Color:
    """Lighten a BGR color by moving its HLS lightness toward white, e.g. for the fill of a box drawn in ``color``.

    Args:
        color: BGR color.
        amount: 0 leaves the color as is, 1 gives white.

    Raises:
        ValueError: If ``color`` is not a BGR color in ``[0, 255]`` or ``amount`` is not in ``[0, 1]``.
    """
    bgr, t = _color("color", color), _amount(amount)
    hls = _convert(bgr, cv2.COLOR_BGR2HLS)
    hls[1] = hls[1] + (1.0 - hls[1]) * t
    return _from_hls(hls)


def darkenColor(color: Sequence[float], amount: float) -> Color:
    """Darken a BGR color by moving its HLS lightness toward black, e.g. for the border of a box filled in ``color``.

    Args:
        color: BGR color.
        amount: 0 leaves the color as is, 1 gives black.

    Raises:
        ValueError: If ``color`` is not a BGR color in ``[0, 255]`` or ``amount`` is not in ``[0, 1]``.
    """
    bgr, t = _color("color", color), _amount(amount)
    hls = _convert(bgr, cv2.COLOR_BGR2HLS)
    hls[1] = hls[1] * (1.0 - t)
    return _from_hls(hls)


def blendColors(a: Sequence[float], b: Sequence[float], t: float = 0.5) -> Color:
    """Mix two BGR colors: ``t = 0`` gives ``a``, ``t = 1`` gives ``b``, ``0.5`` the midpoint.

    Raises:
        ValueError: If a color is not a BGR color in ``[0, 255]`` or ``t`` is not in ``[0, 1]``.
    """
    ca, cb = _color("a", a), _color("b", b)
    check_finite("t", t)
    if not 0 <= t <= 1:
        raise ValueError(f"t must be in [0, 1], got {t}")
    mixed = [round(x + (y - x) * t) for x, y in zip(ca, cb, strict=True)]
    return mixed[0], mixed[1], mixed[2]


def complementaryColor(color: Sequence[float]) -> Color:
    """The color on the opposite side of the hue circle (hue + 180°), keeping lightness and saturation.

    Different from :func:`invertColor`: the complement of a dark red is a dark cyan, not a light cyan.

    Raises:
        ValueError: If ``color`` is not a BGR color in ``[0, 255]``.
    """
    hls = _convert(_color("color", color), cv2.COLOR_BGR2HLS)
    hls[0] = (hls[0] + 180.0) % 360.0
    return _from_hls(hls)


def invertColor(color: Sequence[float]) -> Color:
    """The negative of a BGR color: ``255 - value`` per channel, like ``cv.bitwise_not`` on a pixel.

    Raises:
        ValueError: If ``color`` is not a BGR color in ``[0, 255]``.
    """
    b, g, r = _color("color", color)
    return 255 - b, 255 - g, 255 - r


def colorDistance(a: Sequence[float], b: Sequence[float], space: str = "lab") -> float:
    """Distance between two BGR colors: CIE76 ΔE in Lab (how different they look) or Euclidean in BGR.

    In Lab a ΔE of about 2 is the smallest difference most people notice, and 100 separates black from white.

    Args:
        a: First BGR color.
        b: Second BGR color.
        space: ``"lab"`` for CIE76 ΔE (perceptual) or ``"bgr"`` for plain Euclidean distance in ``[0, 441]``.

    Raises:
        ValueError: If a color is not a BGR color in ``[0, 255]`` or ``space`` is unknown.
    """
    ca, cb = _color("a", a), _color("b", b)
    if space == "lab":
        la, lb = _convert(ca, cv2.COLOR_BGR2Lab), _convert(cb, cv2.COLOR_BGR2Lab)
        return float(np.linalg.norm(la - lb))
    if space == "bgr":
        return float(np.linalg.norm(np.subtract(ca, cb, dtype=np.float64)))
    raise ValueError(f"space must be 'lab' or 'bgr', got {space!r}")


def _lab_table(colors: np.ndarray) -> np.ndarray:
    """Lab values of an ``(N, 3)`` 8-bit BGR array, as float32 ``(N, 3)``."""
    lab: np.ndarray = cv2.cvtColor(colors.reshape(1, -1, 3).astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return lab.reshape(-1, 3)


def nearestColor(color: Sequence[float], palette: Sequence[Sequence[float]] | np.ndarray, space: str = "lab") -> int:
    """Index of the palette color closest to ``color`` (CIE76 in Lab by default), e.g. to snap to a color legend.

    Args:
        color: BGR color.
        palette: BGR colors, as a sequence of tuples or an array of shape ``(N, 3)``.
        space: ``"lab"`` or ``"bgr"``, as for :func:`colorDistance`.

    Raises:
        ValueError: If ``color`` or a palette entry is not a BGR color in ``[0, 255]``, the palette is empty, or
            ``space`` is unknown.
    """
    if space not in ("lab", "bgr"):
        raise ValueError(f"space must be 'lab' or 'bgr', got {space!r}")
    target = np.array(_color("color", color), dtype=np.float64)
    entries = np.asarray(palette, dtype=np.float64).reshape(-1, 3) if len(palette) else np.empty((0, 3))
    if entries.shape[0] == 0:
        raise ValueError("palette must not be empty")
    if not (np.all(np.isfinite(entries)) and np.all((entries >= 0) & (entries <= 255))):
        raise ValueError("palette must hold BGR colors with values in [0, 255]")
    if space == "lab":
        table = _lab_table(np.rint(entries).astype(np.uint8))
        target = _lab_table(np.rint(target).astype(np.uint8)[None])[0]
    else:
        table = entries
    return int(np.argmin(np.linalg.norm(table - target, axis=1)))


_CSS_NAMES = list(CSS_COLORS)
_CSS_BGR = np.array(list(CSS_COLORS.values()), dtype=np.uint8)
_CSS_LAB = _lab_table(_CSS_BGR)


def colorName(color: Sequence[float]) -> str:
    """Name of the CSS color closest to a BGR color (CIE76 in Lab), e.g. ``(30, 20, 220)`` gives ``"crimson"``.

    Useful for logging or labeling the result of ``dominantColors``. For colors with two CSS names
    (``aqua`` / ``cyan``, ``gray`` / ``grey``, ...) the alphabetically first name is returned.

    Raises:
        ValueError: If ``color`` is not a BGR color in ``[0, 255]``.
    """
    target = _lab_table(np.array([_color("color", color)], dtype=np.uint8))[0]
    return _CSS_NAMES[int(np.argmin(np.linalg.norm(_CSS_LAB - target, axis=1)))]


def textColorFor(background: Sequence[float]) -> Color:
    """Black or white, whichever reads better on ``background``; the rule used by :func:`drawLabel`.

    Returns:
        ``(0, 0, 0)`` for light backgrounds (luminance above 140) and ``(255, 255, 255)`` for dark ones.

    Raises:
        ValueError: If ``background`` is not a BGR color in ``[0, 255]``.
    """
    return (0, 0, 0) if _luminance(_color("background", background)) > 140 else (255, 255, 255)


def randomColors(count: int, seed: int | None = None, saturation: float = 1.0, value: float = 1.0) -> list[Color]:
    """``count`` random BGR colors with random hues and the given saturation and brightness.

    Unlike :func:`distinctColors` (a fixed, maximally different sequence) these are random, so two calls with
    different seeds give different palettes, e.g. for per-instance colors in a demo.

    Args:
        count: How many colors.
        seed: Random seed for reproducible colors.
        saturation: HSV saturation in ``[0, 1]``; lower is more pastel.
        value: HSV brightness in ``[0, 1]``; lower is darker.

    Raises:
        ValueError: If ``count`` is negative or ``saturation`` / ``value`` is not in ``[0, 1]``.
    """
    if count < 0:
        raise ValueError(f"count must not be negative, got {count}")
    for name, v in (("saturation", saturation), ("value", value)):
        check_finite(name, v)
        if not 0 <= v <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    if count == 0:
        return []
    hues = np.random.default_rng(seed).uniform(0, 360, size=count).astype(np.float32)
    hsv = np.stack([hues, np.full(count, saturation, np.float32), np.full(count, value, np.float32)], axis=1)
    bgr = cv2.cvtColor(hsv.reshape(1, -1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3) * 255.0
    return [(int(b), int(g), int(r)) for b, g, r in np.clip(np.rint(bgr), 0, 255)]


def gradientPalette(start: Sequence[float], end: Sequence[float], count: int) -> list[Color]:
    """``count`` BGR colors interpolated linearly from ``start`` to ``end`` (both included).

    Args:
        start: First color.
        end: Last color.
        count: How many colors; 1 gives ``[start]``.

    Raises:
        ValueError: If a color is not a BGR color in ``[0, 255]`` or ``count`` is negative.
    """
    a, b = np.array(_color("start", start), np.float64), np.array(_color("end", end), np.float64)
    if count < 0:
        raise ValueError(f"count must not be negative, got {count}")
    steps = np.linspace(0.0, 1.0, count) if count > 1 else np.zeros(count)
    colors = np.rint(a + (b - a) * steps[:, None]).astype(int)
    return [(int(c[0]), int(c[1]), int(c[2])) for c in colors]


def colormapPalette(colormap: int, count: int) -> list[Color]:
    """``count`` BGR colors sampled evenly from an OpenCV colormap, e.g. ``cv.COLORMAP_VIRIDIS``.

    ``cv.applyColorMap`` only maps whole images; this gives the colors themselves, for drawing classes or a
    legend in the colormap's order.

    Args:
        colormap: ``cv.COLORMAP_*`` constant.
        count: How many colors, from the darkest end (0) to the brightest (255) of the map.

    Raises:
        ValueError: If ``count`` is negative.
    """
    if count < 0:
        raise ValueError(f"count must not be negative, got {count}")
    if count == 0:
        return []
    levels = np.rint(np.linspace(0, 255, count)).astype(np.uint8) if count > 1 else np.array([0], np.uint8)
    bgr = cv2.applyColorMap(levels.reshape(1, -1), colormap).reshape(-1, 3)
    return [(int(b), int(g), int(r)) for b, g, r in bgr]
