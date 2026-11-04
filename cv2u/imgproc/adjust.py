"""Photo-style color adjustments."""

import cv2
import numpy as np

__all__ = ["adjustBrightnessContrast", "adjustHSV", "autoContrast", "whiteBalance"]


def _split_alpha(src: np.ndarray, name: str = "src") -> tuple[np.ndarray, np.ndarray | None]:
    if src.dtype != np.uint8:
        raise ValueError(f"{name} must be an 8-bit (uint8) image, got dtype {src.dtype}")
    if src.ndim == 3 and src.shape[2] == 4:
        return src[..., :3], src[..., 3]
    if src.ndim == 2 or (src.ndim == 3 and src.shape[2] in (1, 3)):
        return src, None
    raise ValueError(f"{name} must have 1, 3 or 4 channels, got shape {src.shape}")


def _join_alpha(color: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    return color if alpha is None else np.dstack([color, alpha])


def _lut(color: np.ndarray, table: np.ndarray) -> np.ndarray:
    out: np.ndarray = cv2.LUT(color, np.clip(np.rint(table), 0, 255).astype(np.uint8))
    return out


def adjustBrightnessContrast(src: np.ndarray, brightness: float = 0, contrast: float = 1.0) -> np.ndarray:
    """Change brightness and contrast like the sliders of a photo editor.

    Each value ``v`` becomes ``(v - 128) * contrast + 128 + brightness``, so contrast pivots around mid-gray instead of
    black (``cv.convertScaleAbs(src, alpha, beta)`` scales around 0 and takes the absolute value).

    Args:
        src: 8-bit grayscale, BGR or BGRA image (the alpha channel is kept).
        brightness: Added to every value, e.g. ``-50`` to ``50``.
        contrast: ``1`` keeps the contrast, ``> 1`` increases it, ``< 1`` reduces it, ``0`` gives flat gray.

    Raises:
        ValueError: If ``src`` is not 8-bit or ``contrast`` is negative.
    """
    if contrast < 0:
        raise ValueError(f"contrast must not be negative, got {contrast}")
    color, alpha = _split_alpha(src)
    table = (np.arange(256) - 128.0) * contrast + 128.0 + brightness
    return _join_alpha(_lut(color, table), alpha)


def adjustHSV(src: np.ndarray, hue: float = 0, saturation: float = 1.0, value: float = 1.0) -> np.ndarray:
    """Shift the hue and scale the saturation and brightness (value) of a BGR image.

    Saves the HSV round trip and its pitfalls: OpenCV stores 8-bit hue as degrees / 2 (0 to 179), and hue must wrap
    around instead of being clipped.

    Args:
        src: 8-bit BGR or BGRA image (the alpha channel is kept).
        hue: Hue rotation in degrees, e.g. ``180`` for complementary colors.
        saturation: Saturation factor; ``0`` gives grayscale, ``> 1`` more vivid colors.
        value: Brightness factor.

    Raises:
        ValueError: If ``src`` is not an 8-bit color image, or a factor is negative.
    """
    if saturation < 0 or value < 0:
        raise ValueError(f"saturation and value must not be negative, got {saturation} and {value}")
    color, alpha = _split_alpha(src)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"src must be a color (BGR or BGRA) image, got shape {src.shape}")
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h = (hsv[..., 0].astype(np.int32) + round(hue / 2)) % 180
    s = np.clip(np.rint(hsv[..., 1] * saturation), 0, 255)
    v = np.clip(np.rint(hsv[..., 2] * value), 0, 255)
    adjusted = np.dstack([h, s, v]).astype(np.uint8)
    return _join_alpha(cv2.cvtColor(adjusted, cv2.COLOR_HSV2BGR), alpha)


def autoContrast(src: np.ndarray, clip: float = 0.01, perChannel: bool = False) -> np.ndarray:
    """Stretch the levels so the darkest pixels become black and the brightest white (like "Auto Levels").

    ``cv.normalize(..., cv.NORM_MINMAX)`` uses the exact minimum and maximum, so a single noisy pixel prevents any
    stretch. Here a fraction ``clip`` of the pixels at each end is allowed to saturate.

    Args:
        src: 8-bit grayscale, BGR or BGRA image (the alpha channel is kept).
        clip: Fraction of pixels clipped at each end, e.g. ``0.01`` for 1%.
        perChannel: Stretch each channel separately. This also removes color casts, but can shift colors; by default
            all channels share one stretch, which keeps the colors.

    Raises:
        ValueError: If ``src`` is not 8-bit or ``clip`` is outside ``[0, 0.5)``.
    """
    if not 0.0 <= clip < 0.5:
        raise ValueError(f"clip must be in [0, 0.5), got {clip}")
    color, alpha = _split_alpha(src)

    def stretch(values: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(values, [100 * clip, 100 * (1 - clip)])
        if hi <= lo:
            return np.arange(256, dtype=np.float64)  # flat: leave unchanged
        table: np.ndarray = (np.arange(256) - lo) * 255.0 / (hi - lo)
        return table

    if perChannel and color.ndim == 3:
        channels = [_lut(np.ascontiguousarray(color[..., c]), stretch(color[..., c])) for c in range(color.shape[2])]
        return _join_alpha(np.dstack(channels), alpha)
    return _join_alpha(_lut(color, stretch(color)), alpha)


def whiteBalance(src: np.ndarray, method: str = "whitePatch") -> np.ndarray:
    """Remove a color cast, so that white and gray objects look neutral.

    OpenCV's white balancing lives in the ``xphoto`` contrib module, which ``opencv-python`` does not include.

    Args:
        src: 8-bit BGR or BGRA image (the alpha channel is kept).
        method: ``"whitePatch"`` (default) assumes that the brightest pixels (the top 0.5%, ignoring blown-out ones)
            are white. ``"grayWorld"`` assumes that the average color of the scene is gray; it also works without white
            objects, but shifts the colors of scenes dominated by one color (e.g. a green object on black).

    Raises:
        ValueError: If ``src`` is not an 8-bit color image, or ``method`` is unknown.
    """
    color, alpha = _split_alpha(src)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"src must be a color (BGR or BGRA) image, got shape {src.shape}")
    pixels = color.reshape(-1, 3).astype(np.float64)
    if method == "whitePatch":
        # Blown-out pixels (255 in every channel) carry no color information; partly clipped ones still do.
        unclipped = pixels[(pixels < 255).any(axis=1)]
        sample = unclipped if len(unclipped) >= max(1, len(pixels) // 100) else pixels
        reference = np.percentile(sample, 99.5, axis=0)
        gains = float(np.max(reference)) / np.maximum(reference, 1e-6)
    elif method == "grayWorld":
        means = pixels.mean(axis=0)
        gains = means.mean() / np.maximum(means, 1e-6)
    else:
        raise ValueError(f"method must be 'whitePatch' or 'grayWorld', got {method!r}")
    balanced = np.clip(np.rint(color.astype(np.float64) * gains), 0, 255).astype(np.uint8)
    return _join_alpha(balanced, alpha)
