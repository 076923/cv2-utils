"""Image statistics and quality checks: brightness, contrast, noise, entropy, histograms, blank/blurry tests."""

import math
from typing import Any, NamedTuple

import cv2
import numpy as np

from .._validate import check_finite, check_image_not_empty
from .color import toBGR, toGray
from .quality import sharpnessScore

__all__ = [
    "ImageStats",
    "brightnessScore",
    "colorfulness",
    "contrastScore",
    "dominantHue",
    "entropy",
    "estimateNoise",
    "exposureScore",
    "histogramCompare",
    "histogramEx",
    "imageStats",
    "isBlank",
    "isBlurry",
    "isGrayscale",
    "mae",
    "meanColor",
    "mse",
]


def _peak(dtype: np.dtype[Any]) -> float:
    """Largest value of the dtype: 255 for 8-bit, 65535 for 16-bit, 1.0 for float images."""
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0


def _gray64(src: np.ndarray) -> np.ndarray:
    """Grayscale ``float64`` copy of any supported image (validates channels and emptiness)."""
    return toGray(src).astype(np.float64)


def _color_channels(src: np.ndarray) -> np.ndarray:
    """The image without its alpha channel, as a ``(h, w, c)`` array."""
    check_image_not_empty("src", src)
    if src.ndim == 2:
        return src[..., np.newaxis]
    if src.ndim != 3 or src.shape[2] not in (1, 3, 4):
        raise ValueError(f"src must have 1, 3 or 4 channels, got shape {src.shape}")
    return src[..., :3]


def _to_uint8(src: np.ndarray) -> np.ndarray:
    """8-bit version of the image for OpenCV functions that only accept ``uint8`` (scaled by the dtype peak)."""
    if src.dtype == np.uint8:
        return src
    scaled = np.clip(src.astype(np.float64) * (255.0 / _peak(src.dtype)), 0, 255)
    out: np.ndarray = np.rint(scaled).astype(np.uint8)
    return out


def _check_pair(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")
    check_image_not_empty("a", a)


def _check_mask(src: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.shape[:2] != src.shape[:2]:
        raise ValueError(f"mask must be {src.shape[1]}x{src.shape[0]}, got shape {mask.shape}")
    keep: np.ndarray = mask.reshape(mask.shape[:2]) != 0
    return keep


class ImageStats(NamedTuple):
    """Summary of an image as returned by ``imageStats``."""

    width: int
    height: int
    channels: int
    dtype: str
    mean: tuple[float, ...]
    """Mean of each channel (alpha included for BGRA)."""
    std: tuple[float, ...]
    """Standard deviation of each channel."""
    minVal: float
    maxVal: float
    brightness: float
    """Mean gray level in ``[0, 1]`` (see ``brightnessScore``)."""
    contrast: float
    """RMS contrast in ``[0, 1]`` (see ``contrastScore``)."""
    sharpness: float
    """Variance of the Laplacian (see ``sharpnessScore``)."""
    entropy: float
    """Shannon entropy of the gray histogram in bits (see ``entropy``)."""
    isGray: bool
    """``True`` when the image is 2-D or all its color channels are equal."""


def isGrayscale(src: np.ndarray, tolerance: float = 0) -> bool:
    """Whether an image carries no color: it is single-channel, or its B, G and R channels are (nearly) equal.

    ``img.ndim == 2`` misses the common case of a gray picture stored as 3-channel BGR (JPEG scans, frames from
    ``cv.VideoCapture``). The alpha channel of BGRA images is ignored.

    Args:
        src: Grayscale, BGR or BGRA image.
        tolerance: Largest allowed difference between the channels of any pixel (0 means exactly equal; a few
            levels absorb JPEG chroma noise).

    Raises:
        ValueError: If ``src`` is empty, has a wrong channel count, or ``tolerance`` is negative.
    """
    check_finite("tolerance", tolerance, nonnegative=True)
    color = _color_channels(src)
    if color.shape[2] == 1:
        return True
    values = color.astype(np.float64)
    spread = values.max(axis=2) - values.min(axis=2)
    return bool(np.max(spread) <= tolerance)


def isBlank(src: np.ndarray, tolerance: float = 2.0) -> bool:
    """Whether an image is uniform (a single flat color), e.g. a black frame, a white page or a failed capture.

    Args:
        src: Grayscale, BGR or BGRA image.
        tolerance: An image is blank when the standard deviation of every color channel is at most this value,
            in the units of the image (gray levels for 8-bit).

    Raises:
        ValueError: If ``src`` is empty, has a wrong channel count, or ``tolerance`` is negative.
    """
    check_finite("tolerance", tolerance, nonnegative=True)
    color = _color_channels(src).astype(np.float64)
    return bool(np.max(color.reshape(-1, color.shape[2]).std(axis=0)) <= tolerance)


def isBlurry(src: np.ndarray, threshold: float = 100.0) -> bool:
    """Whether an image is out of focus: its ``sharpnessScore`` (variance of the Laplacian) is below ``threshold``.

    The score depends on the content and the image size as much as on the focus, so ``threshold`` is only a
    starting point (100 suits typical 8-bit photos); calibrate it on sharp and blurry samples of your own data.

    Args:
        src: Grayscale, BGR or BGRA image.
        threshold: Sharpness score below which the image counts as blurry.

    Raises:
        ValueError: If ``src`` is empty, has a wrong channel count, or ``threshold`` is not finite or negative.
    """
    check_finite("threshold", threshold, nonnegative=True)
    return sharpnessScore(src) < threshold


def brightnessScore(src: np.ndarray) -> float:
    """Mean gray level of an image normalized to ``[0, 1]`` (0 is black, 1 is white).

    ``cv.mean`` returns a 4-tuple in the units of the dtype; this gives one number that is comparable between
    8-bit, 16-bit and float images.

    Args:
        src: Grayscale, BGR or BGRA image.

    Raises:
        ValueError: If ``src`` is empty or has a wrong channel count.
    """
    return float(_gray64(src).mean() / _peak(src.dtype))


def contrastScore(src: np.ndarray) -> float:
    """RMS contrast: standard deviation of the gray levels normalized to ``[0, 1]``.

    0 for a flat image, 0.5 for a half black / half white image. Useful for sorting images by how washed out they
    are or for choosing which ones need ``autoContrast``.

    Args:
        src: Grayscale, BGR or BGRA image.

    Raises:
        ValueError: If ``src`` is empty or has a wrong channel count.
    """
    return float(_gray64(src).std() / _peak(src.dtype))


def colorfulness(src: np.ndarray) -> float:
    """Colorfulness metric of Hasler and Süsstrunk (2003): 0 for gray images, roughly 20-40 for average photos
    and above 60 for very vivid ones.

    Computed on the opponent color axes ``rg = R - G`` and ``yb = (R + G) / 2 - B`` as
    ``sqrt(std_rg² + std_yb²) + 0.3 * sqrt(mean_rg² + mean_yb²)``. Non-8-bit images are scaled to the 0-255 range
    first so the numbers stay comparable.

    Args:
        src: Grayscale, BGR or BGRA image.

    Raises:
        ValueError: If ``src`` is empty or has a wrong channel count.
    """
    color = _color_channels(src)
    if color.shape[2] == 1:
        return 0.0
    values = color.astype(np.float64) * (255.0 / _peak(src.dtype))
    b, g, r = values[..., 0], values[..., 1], values[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_root = math.hypot(float(rg.std()), float(yb.std()))
    mean_root = math.hypot(float(rg.mean()), float(yb.mean()))
    return std_root + 0.3 * mean_root


def exposureScore(src: np.ndarray, lowThreshold: float = 0.05, highThreshold: float = 0.95) -> tuple[float, float]:
    """Fractions of under-exposed and over-exposed pixels, for flagging too dark or blown-out images.

    Args:
        src: Grayscale, BGR or BGRA image.
        lowThreshold: Gray level (normalized to ``[0, 1]``) at or below which a pixel counts as under-exposed.
        highThreshold: Gray level (normalized to ``[0, 1]``) at or above which a pixel counts as over-exposed.

    Returns:
        ``(underExposed, overExposed)``, each in ``[0, 1]``.

    Raises:
        ValueError: If ``src`` is empty or has a wrong channel count, or a threshold is outside ``[0, 1]`` or
            ``lowThreshold`` exceeds ``highThreshold``.
    """
    for name, value in (("lowThreshold", lowThreshold), ("highThreshold", highThreshold)):
        check_finite(name, value, nonnegative=True)
        if value > 1:
            raise ValueError(f"{name} must be within [0, 1], got {value}")
    if lowThreshold > highThreshold:
        raise ValueError(f"lowThreshold must not exceed highThreshold, got {lowThreshold} and {highThreshold}")
    gray = _gray64(src) / _peak(src.dtype)
    return float((gray <= lowThreshold).mean()), float((gray >= highThreshold).mean())


def _gray_histogram(src: np.ndarray, bins: int) -> np.ndarray:
    gray = toGray(src)
    if gray.dtype == np.uint8 and bins == 256:
        return np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    counts, _ = np.histogram(gray.astype(np.float64), bins=bins, range=(0.0, _peak(src.dtype)))
    return counts.astype(np.float64)


def entropy(src: np.ndarray, bins: int = 256) -> float:
    """Shannon entropy of the gray-level histogram in bits: 0 for a flat image, up to 8 for 8-bit images.

    A rough measure of how much detail or texture an image holds; near-zero values flag blank or almost blank
    images, and it is a common feature for picking informative frames.

    Args:
        src: Grayscale, BGR or BGRA image.
        bins: Number of histogram bins over the dtype range (``[0, 255]`` for 8-bit, ``[0, 1]`` for float).

    Raises:
        ValueError: If ``src`` is empty or has a wrong channel count, or ``bins`` is below 1.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    counts = _gray_histogram(src, bins)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def estimateNoise(src: np.ndarray) -> float:
    """Estimate the standard deviation of additive Gaussian noise in an image (Immerkær 1996, fast noise variance
    estimation).

    Works without a clean reference: a Laplacian-like 3x3 kernel cancels smooth structure and edges contribute
    little on average. Strongly textured images (fine patterns, hair) overestimate the noise.

    Args:
        src: Grayscale, BGR or BGRA image, at least 3x3 pixels.

    Returns:
        The noise sigma in the units of the image (gray levels for 8-bit).

    Raises:
        ValueError: If ``src`` is empty, has a wrong channel count or is smaller than 3x3.
    """
    gray = _gray64(src)
    height, width = gray.shape
    if height < 3 or width < 3:
        raise ValueError(f"src must be at least 3x3 pixels, got {width}x{height}")
    response: np.ndarray = cv2.filter2D(gray, cv2.CV_64F, _NOISE_KERNEL, borderType=cv2.BORDER_REPLICATE)
    total = float(np.abs(response[1:-1, 1:-1]).sum())
    return float(total * math.sqrt(math.pi / 2) / (6 * (width - 2) * (height - 2)))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two images of the same shape (0 for identical images).

    ``cv.PSNR`` gives a logarithmic score, and ``cv.norm(a, b, cv.NORM_L2)`` an unnormalized sum; this is the plain
    per-value mean in ``float64``, so 8-bit inputs do not overflow.

    Raises:
        ValueError: If the shapes differ or the images are empty.
    """
    _check_pair(a, b)
    difference = a.astype(np.float64) - b.astype(np.float64)
    return float((difference * difference).mean())


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error between two images of the same shape (0 for identical images).

    Less sensitive to a few large outliers than ``mse``; the per-value mean of ``cv.absdiff`` computed in ``float64``.

    Raises:
        ValueError: If the shapes differ or the images are empty.
    """
    _check_pair(a, b)
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def histogramEx(
    src: np.ndarray,
    bins: int = 256,
    mask: np.ndarray | None = None,
    channels: list[int] | tuple[int, ...] | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Per-channel histogram of an image as a ``(channels, bins)`` array.

    ``cv.calcHist`` takes lists of images, channels and ranges, needs the ranges spelled out per dtype and returns a
    ``(bins, 1)`` column per call; this takes any image and returns one row per channel over the dtype range
    (``[0, 255]`` for 8-bit, ``[0, 65535]`` for 16-bit, ``[0, 1]`` for float).

    Args:
        src: Grayscale, BGR or BGRA image.
        bins: Number of bins per channel.
        mask: Optional ``uint8`` mask of the same height and width; only pixels where it is non-zero are counted.
        channels: Channel indices to include (all by default).
        normalize: Divide each row by its sum so the rows are probability distributions.

    Returns:
        ``float64`` array of shape ``(len(channels), bins)``.

    Raises:
        ValueError: If ``src`` is empty or has a wrong channel count, ``bins`` is below 1, a channel index is out of
            range, or the mask has a different size.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    check_image_not_empty("src", src)
    image = src[..., np.newaxis] if src.ndim == 2 else src
    if image.ndim != 3:
        raise ValueError(f"src must be a 2-D or 3-D image, got shape {src.shape}")
    count = int(image.shape[2])
    indices = list(range(count)) if channels is None else list(channels)
    if any(not 0 <= c < count for c in indices):
        raise ValueError(f"channels must be within [0, {count - 1}], got {indices}")
    keep = _check_mask(src, mask)
    peak = _peak(src.dtype)
    rows = []
    for c in indices:
        values = image[..., c].astype(np.float64)
        if keep is not None:
            values = values[keep]
        counts, _ = np.histogram(values, bins=bins, range=(0.0, peak))
        row = counts.astype(np.float64)
        if normalize and row.sum() > 0:
            row /= row.sum()
        rows.append(row)
    return np.stack(rows)


def histogramCompare(
    a: np.ndarray,
    b: np.ndarray,
    method: int = cv2.HISTCMP_CORREL,
    bins: int = 32,
    colorSpace: str = "hsv",
) -> float:
    """Compare the color distributions of two images (any sizes) with ``cv.compareHist``.

    ``cv.compareHist`` needs two histograms built with the same ``cv.calcHist`` setup; this builds and normalizes them
    for you. Images with different channel counts are brought to the same color space first.

    Args:
        a: First image (grayscale, BGR or BGRA).
        b: Second image; may have a different size or channel count.
        method: ``cv.HISTCMP_*`` method. For ``HISTCMP_CORREL`` and ``HISTCMP_INTERSECT`` higher means more similar;
            for ``HISTCMP_CHISQR``, ``HISTCMP_BHATTACHARYYA`` and ``HISTCMP_HELLINGER`` lower means more similar.
        bins: Bins per channel.
        colorSpace: ``"hsv"`` (joint hue-saturation histogram, robust to brightness changes), ``"bgr"`` (joint
            3-D color histogram) or ``"gray"`` (1-D intensity histogram).

    Raises:
        ValueError: If an image is empty or has a wrong channel count, ``bins`` is below 1 or ``colorSpace`` is unknown.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    if colorSpace not in ("hsv", "bgr", "gray"):
        raise ValueError(f'colorSpace must be "hsv", "bgr" or "gray", got {colorSpace!r}')

    def histogram(src: np.ndarray) -> np.ndarray:
        image = _to_uint8(src)
        if colorSpace == "gray":
            image, channels, ranges = toGray(image), [0], [0.0, 256.0]
        elif colorSpace == "hsv":
            image = cv2.cvtColor(toBGR(image), cv2.COLOR_BGR2HSV)
            channels, ranges = [0, 1], [0.0, 180.0, 0.0, 256.0]
        else:
            image, channels, ranges = toBGR(image), [0, 1, 2], [0.0, 256.0] * 3
        hist: np.ndarray = cv2.calcHist([image], channels, None, [bins] * len(channels), ranges)
        return hist / max(float(hist.sum()), 1.0)

    return float(cv2.compareHist(histogram(a), histogram(b), method))


def dominantHue(src: np.ndarray, bins: int = 36) -> float:
    """The most common hue of an image in degrees ``[0, 360)``, weighted by saturation so gray pixels do not vote.

    0 is red, 120 green, 240 blue. The hue histogram peak is refined by the circular mean of the hues inside the
    peak bin. Returns 0.0 for images without any saturated pixel (grayscale images).

    Args:
        src: Grayscale, BGR or BGRA image.
        bins: Number of hue bins over the circle.

    Raises:
        ValueError: If ``src`` is empty or has a wrong channel count, or ``bins`` is below 1.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    color = _color_channels(src)
    if color.shape[2] == 1:
        return 0.0
    hsv = cv2.cvtColor(toBGR(_to_uint8(src)), cv2.COLOR_BGR2HSV_FULL)
    hue = hsv[..., 0].astype(np.float64) * (360.0 / 256.0)
    weight = hsv[..., 1].astype(np.float64) * (hsv[..., 2].astype(np.float64) / 255.0)
    if weight.sum() == 0:
        return 0.0
    counts, edges = np.histogram(hue, bins=bins, range=(0.0, 360.0), weights=weight)
    peak = int(np.argmax(counts))
    inside = (hue >= edges[peak]) & (hue < edges[peak + 1])
    radians = np.deg2rad(hue[inside])
    w = weight[inside]
    angle = math.degrees(math.atan2(float((w * np.sin(radians)).sum()), float((w * np.cos(radians)).sum())))
    return angle % 360.0


def meanColor(src: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, ...]:
    """Mean value of each channel, optionally over a mask: one number for gray, ``(b, g, r)`` for BGR images.

    ``cv.mean`` always returns four values (zeros for missing channels) and requires a ``uint8`` mask.

    Args:
        src: Image of any dtype with 1, 3 or 4 channels.
        mask: Optional mask of the same height and width; only pixels where it is non-zero are averaged.

    Returns:
        A tuple with one ``float`` per channel of ``src`` (zeros when the mask selects no pixel).

    Raises:
        ValueError: If ``src`` is empty or has a wrong channel count, or the mask has a different size.
    """
    _color_channels(src)
    image = src[..., np.newaxis] if src.ndim == 2 else src
    keep = _check_mask(src, mask)
    values = image.reshape(-1, image.shape[2]) if keep is None else image[keep]
    if values.shape[0] == 0:
        return tuple(0.0 for _ in range(image.shape[2]))
    return tuple(float(v) for v in values.astype(np.float64).mean(axis=0))


def imageStats(src: np.ndarray) -> ImageStats:
    """Summarize an image in one call: size, dtype, per-channel mean and std, range, brightness, contrast,
    sharpness, entropy and whether it is grayscale.

    Handy for quick inspection in a notebook or for logging dataset statistics; ``cv.meanStdDev`` and
    ``cv.minMaxLoc`` cover only parts of this and the latter rejects color images.

    Args:
        src: Grayscale, BGR or BGRA image.

    Raises:
        ValueError: If ``src`` is empty or has a wrong channel count.
    """
    _color_channels(src)
    image = src[..., np.newaxis] if src.ndim == 2 else src
    flat = image.reshape(-1, image.shape[2]).astype(np.float64)
    height, width = src.shape[:2]
    return ImageStats(
        width=int(width),
        height=int(height),
        channels=int(image.shape[2]),
        dtype=str(src.dtype),
        mean=tuple(float(v) for v in flat.mean(axis=0)),
        std=tuple(float(v) for v in flat.std(axis=0)),
        minVal=float(np.min(flat)),
        maxVal=float(np.max(flat)),
        brightness=brightnessScore(src),
        contrast=contrastScore(src),
        sharpness=sharpnessScore(src),
        entropy=entropy(src),
        isGray=isGrayscale(src),
    )
