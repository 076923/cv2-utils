"""Lightweight data augmentation for training detectors and classifiers, without an extra dependency.

Every random function takes ``seed`` and draws from ``np.random.default_rng(seed)``, so the same seed gives the same
output. Geometric functions accept detection boxes ``(x, y, width, height)`` and return them transformed along with the
image, so labels stay aligned.
"""

import math
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .._validate import check_color, check_finite, check_image_not_empty
from ..core.boxes import clipBoxes
from .adjust import adjustBrightnessContrast, adjustHSV

__all__ = [
    "addNoise",
    "cutMix",
    "jpegArtifacts",
    "mixup",
    "mosaic",
    "motionBlur",
    "posterize",
    "randomBlur",
    "randomBrightnessContrast",
    "randomCrop",
    "randomErase",
    "randomFlip",
    "randomHSV",
    "randomPerspective",
    "randomRotate",
    "randomScale",
    "solarize",
    "vignette",
]


# --- helpers -------------------------------------------------------------------------------------------------------


def _check_image(src: np.ndarray, name: str = "src") -> None:
    if src.ndim not in (2, 3) or (src.ndim == 3 and src.shape[2] not in (1, 3, 4)):
        raise ValueError(f"{name} must be a grayscale, BGR or BGRA image, got shape {src.shape}")
    check_image_not_empty(name, src)


def _value_range(src: np.ndarray) -> tuple[float, float]:
    """The nominal value range of an image: the dtype range for integers, ``[0, 1]`` for floats."""
    if np.issubdtype(src.dtype, np.integer):
        info = np.iinfo(src.dtype)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def _to_dtype(values: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Clip float values to the range of ``src`` and convert back to its dtype."""
    low, high = _value_range(src)
    clipped = np.clip(values, low, high)
    if np.issubdtype(src.dtype, np.integer):
        clipped = np.rint(clipped)
    return np.asarray(clipped, dtype=src.dtype)


def _box_array(boxes: ArrayLike, name: str = "boxes") -> np.ndarray:
    array = np.asarray(boxes, dtype=np.float64)
    if array.size == 0:
        return array.reshape(0, 4)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4) as (x, y, width, height), got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite numbers")
    return array


def _box_corners(boxes: np.ndarray) -> np.ndarray:
    """``(N, 4, 2)`` corner points of xywh boxes."""
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack(
        [np.stack([x, y], 1), np.stack([x + w, y], 1), np.stack([x + w, y + h], 1), np.stack([x, y + h], 1)], 1
    )


def _corners_to_boxes(corners: np.ndarray) -> np.ndarray:
    """Axis-aligned xywh boxes around ``(N, 4, 2)`` corner points."""
    if len(corners) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    low, high = corners.min(axis=1), corners.max(axis=1)
    return np.concatenate([low, high - low], axis=1)


def _fill_value(fill: Any, src: np.ndarray) -> list[float]:
    values = [float(fill)] if isinstance(fill, (int, float)) else [float(v) for v in fill]
    check_color("fill", values, src.dtype)
    channels = 1 if src.ndim == 2 else src.shape[2]
    if len(values) == 1:
        values = values * channels
    if len(values) != channels:
        raise ValueError(f"fill must have 1 or {channels} components for this image, got {values}")
    return values


def _size(size: tuple[int, int], name: str = "size") -> tuple[int, int]:
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must be positive, got {size}")
    return width, height


# --- deterministic degradations ------------------------------------------------------------------------------------


def addNoise(
    src: np.ndarray, kind: str = "gaussian", amount: float | None = None, seed: int | None = None
) -> np.ndarray:
    """Add synthetic sensor noise to an image.

    OpenCV has no noise generator except ``cv.randn``/``cv.randu``, which need a preallocated array of the right type
    and leave the clipping and the salt-and-pepper logic to you. The result keeps the dtype of ``src`` and is clipped
    to its range (``[0, 255]`` for 8-bit, ``[0, 1]`` for float images).

    Args:
        src: Grayscale, BGR or BGRA image of any dtype (noise is added to every channel, including alpha).
        kind: ``"gaussian"`` adds normal noise with standard deviation ``amount`` (as a fraction of the value range,
            default ``0.1``); ``"salt_pepper"`` sets a fraction ``amount`` (default ``0.05``) of the pixels to black or
            white; ``"poisson"`` applies shot noise whose strength depends on the brightness (``amount`` scales the
            photon count, default ``1``, larger is less noisy); ``"speckle"`` multiplies every value by
            ``1 + N(0, amount)`` (default ``0.1``).
        amount: Noise strength, see ``kind``.
        seed: Random seed for reproducible noise.

    Returns:
        A new image with the same shape and dtype as ``src``.

    Raises:
        ValueError: If ``src`` is empty, ``kind`` is unknown, or ``amount`` is negative or not finite (for
            ``"salt_pepper"`` it must also be at most 1).
    """
    _check_image(src)
    rng = np.random.default_rng(seed)
    low, high = _value_range(src)
    span = high - low
    values = src.astype(np.float64)
    if kind == "gaussian":
        sigma = 0.1 if amount is None else amount
        check_finite("amount", sigma, nonnegative=True)
        noisy = values + rng.normal(0.0, sigma * span, src.shape)
    elif kind == "salt_pepper":
        fraction = 0.05 if amount is None else amount
        check_finite("amount", fraction, nonnegative=True)
        if fraction > 1:
            raise ValueError(f"amount must be at most 1 for salt_pepper noise, got {fraction}")
        noisy = values.copy()
        draw = rng.random(src.shape[:2])
        noisy[draw < fraction / 2] = low
        noisy[(draw >= fraction / 2) & (draw < fraction)] = high
    elif kind == "poisson":
        scale = 1.0 if amount is None else amount
        check_finite("amount", scale, positive=True)
        # Photon counts in the 0..255 scale, so that 8-bit and float images get the same look.
        photons = (values - low) / span * 255.0 * scale
        noisy = rng.poisson(np.maximum(photons, 0)) / (255.0 * scale) * span + low
    elif kind == "speckle":
        sigma = 0.1 if amount is None else amount
        check_finite("amount", sigma, nonnegative=True)
        noisy = values * (1.0 + rng.normal(0.0, sigma, src.shape))
    else:
        raise ValueError(f"kind must be 'gaussian', 'salt_pepper', 'poisson' or 'speckle', got {kind!r}")
    return _to_dtype(noisy, src)


def motionBlur(src: np.ndarray, size: int = 15, angle: float = 0.0) -> np.ndarray:
    """Blur along a line, like a camera or object moving during the exposure.

    ``cv.blur`` and ``cv.GaussianBlur`` only blur isotropically; a directional kernel must be built by hand.

    Args:
        src: Image of any dtype and channel count.
        size: Length of the motion in pixels (the kernel is ``size x size``).
        angle: Direction of the motion in degrees, counter-clockwise from the x axis; ``0`` is horizontal.

    Returns:
        The blurred image, with the dtype of ``src``.

    Raises:
        ValueError: If ``src`` is empty, ``size`` is smaller than 1, or ``angle`` is not finite.
    """
    _check_image(src)
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    check_finite("angle", angle)
    if size == 1:
        return src.copy()
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2
    dx, dy = math.cos(math.radians(angle)), -math.sin(math.radians(angle))
    half = (size - 1) / 2
    p0 = (round(center - dx * half), round(center - dy * half))
    p1 = (round(center + dx * half), round(center + dy * half))
    cv2.line(kernel, p0, p1, 1.0, 1)
    kernel /= kernel.sum()
    blurred: np.ndarray = cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    return blurred


def vignette(src: np.ndarray, strength: float = 0.5, radius: float = 1.0) -> np.ndarray:
    """Darken the image towards the corners with a smooth Gaussian falloff, like a cheap lens.

    Args:
        src: Grayscale, BGR or BGRA image (the alpha channel is not changed).
        strength: Darkening at the corners: ``0`` leaves the image unchanged, ``1`` makes the corners black.
        radius: Size of the bright area relative to the half diagonal; smaller values darken a larger area.

    Returns:
        A new image with the same shape and dtype as ``src``.

    Raises:
        ValueError: If ``src`` is empty, ``strength`` is outside ``[0, 1]``, or ``radius`` is not positive.
    """
    _check_image(src)
    check_finite("strength", strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    check_finite("radius", radius, positive=True)
    h, w = src.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    half_diagonal = math.hypot((w - 1) / 2, (h - 1) / 2) or 1.0
    distance = np.hypot(xs - (w - 1) / 2, ys - (h - 1) / 2) / half_diagonal
    gain = 1.0 - strength * (1.0 - np.exp(-(distance**2) / (2 * radius**2)))
    values = src.astype(np.float64)
    if src.ndim == 3:
        gain = gain[..., None]
        if src.shape[2] == 4:
            gain = np.concatenate([np.repeat(gain, 3, axis=2), np.ones_like(gain)], axis=2)
    return _to_dtype(values * gain, src)


def jpegArtifacts(src: np.ndarray, quality: int = 30) -> np.ndarray:
    """Add JPEG compression artifacts (blocking, ringing) by an encode/decode round trip in memory.

    Args:
        src: 8-bit grayscale, BGR or BGRA image (the alpha channel is kept as is).
        quality: JPEG quality from 1 (worst) to 100.

    Returns:
        The re-encoded image with the shape of ``src``.

    Raises:
        ValueError: If ``src`` is not 8-bit or is empty, or ``quality`` is outside ``[1, 100]``.
    """
    _check_image(src)
    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be in [1, 100], got {quality}")
    color = src[..., :3] if src.ndim == 3 and src.shape[2] == 4 else src
    ok, encoded = cv2.imencode(".jpg", np.ascontiguousarray(color), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:  # pragma: no cover - imencode only fails on unsupported inputs, which are rejected above
        raise ValueError("src could not be encoded as JPEG")
    decoded = np.asarray(cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)).reshape(color.shape)
    if src.ndim == 3 and src.shape[2] == 4:
        return np.dstack([decoded, src[..., 3]])
    return decoded


def randomErase(
    src: np.ndarray,
    size: int | tuple[int, int] | None = None,
    count: int = 1,
    fill: float | Sequence[float] | str = 0,
    seed: int | None = None,
) -> np.ndarray:
    """Blank out random rectangles ("cutout" / random erasing) so a model cannot rely on one region.

    Args:
        src: Grayscale, BGR or BGRA image of any dtype.
        size: Patch size as ``(width, height)`` or one side; by default a quarter of the image in each direction.
        count: Number of patches.
        fill: Fill value or color, or ``"random"`` for uniform noise in the value range of the image.
        seed: Random seed for the patch positions (and the noise).

    Returns:
        A new image with the patches erased.

    Raises:
        ValueError: If ``src`` is empty, ``size`` is not positive or larger than the image, ``count`` is negative, or
            ``fill`` is not a valid color for the dtype.
    """
    _check_image(src)
    h, w = src.shape[:2]
    if size is None:
        patch = (max(1, w // 4), max(1, h // 4))
    elif isinstance(size, int):
        patch = (size, size)
    else:
        patch = (int(size[0]), int(size[1]))
    pw, ph = _size(patch)
    if pw > w or ph > h:
        raise ValueError(f"size must not exceed the image size ({w}x{h}), got {patch}")
    if count < 0:
        raise ValueError(f"count must not be negative, got {count}")
    random_fill = isinstance(fill, str)
    if random_fill and fill != "random":
        raise ValueError(f"fill must be a value, a color or 'random', got {fill!r}")
    value = None if random_fill else _fill_value(fill, src)
    rng = np.random.default_rng(seed)
    low, high = _value_range(src)
    dst = src.copy()
    for _ in range(count):
        x = int(rng.integers(0, w - pw + 1))
        y = int(rng.integers(0, h - ph + 1))
        region = dst[y : y + ph, x : x + pw]
        if value is None:
            region[...] = _to_dtype(np.asarray(rng.uniform(low, high, region.shape)), src)
        else:
            region[...] = value[0] if src.ndim == 2 else np.asarray(value, dtype=src.dtype)
    return dst


def solarize(src: np.ndarray, threshold: float = 128) -> np.ndarray:
    """Invert every value above ``threshold`` (the darkroom "Sabattier" effect).

    A standard augmentation (AutoAugment, RandAugment) that Pillow has in ``ImageOps`` but OpenCV lacks.

    Args:
        src: Grayscale, BGR or BGRA image of any dtype.
        threshold: Values at or above it are inverted within the value range (``[0, 255]`` for 8-bit, ``[0, 1]``
            for float images).

    Returns:
        A new image with the same shape and dtype as ``src``.

    Raises:
        ValueError: If ``src`` is empty or ``threshold`` is not finite.
    """
    _check_image(src)
    check_finite("threshold", threshold)
    low, high = _value_range(src)
    values = src.astype(np.float64)
    return _to_dtype(np.where(values >= threshold, high + low - values, values), src)


def posterize(src: np.ndarray, bits: int = 4) -> np.ndarray:
    """Reduce the number of tone levels per channel to ``2 ** bits`` (flat, poster-like colors).

    A standard augmentation (AutoAugment, RandAugment) that Pillow has in ``ImageOps`` but OpenCV lacks.

    Args:
        src: Grayscale, BGR or BGRA image of any dtype. Integer images keep their ``bits`` most significant bits;
            float images in ``[0, 1]`` are quantized to ``2 ** bits`` levels.
        bits: Bits to keep per channel, from 1 to the bit depth of the dtype (8 for 8-bit images).

    Returns:
        A new image with the same shape and dtype as ``src``.

    Raises:
        ValueError: If ``src`` is empty or ``bits`` is outside the valid range.
    """
    _check_image(src)
    if np.issubdtype(src.dtype, np.integer):
        depth = np.iinfo(src.dtype).bits
        if not 1 <= bits <= depth:
            raise ValueError(f"bits must be in [1, {depth}] for {src.dtype} images, got {bits}")
        mask = ~np.array((1 << (depth - bits)) - 1, dtype=src.dtype)
        return np.asarray(src & mask, dtype=src.dtype)
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be in [1, 16] for float images, got {bits}")
    levels = float(2**bits)
    return _to_dtype(np.floor(src.astype(np.float64) * levels) / levels, src)


# --- random geometric transforms -----------------------------------------------------------------------------------


def randomCrop(
    src: np.ndarray,
    size: tuple[int, int],
    boxes: ArrayLike | None = None,
    seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Crop a window of ``size`` at a random position.

    Args:
        src: Image of any dtype and channel count.
        size: Crop size as ``(width, height)``, at most the image size.
        boxes: Optional boxes ``(N, 4)`` as ``(x, y, width, height)``; they are shifted and clipped to the crop
            (boxes outside become zero-sized, so the rows stay aligned with your labels).
        seed: Random seed.

    Returns:
        The crop, or ``(crop, boxes)`` when ``boxes`` is given.

    Raises:
        ValueError: If ``src`` is empty, ``size`` is not positive or exceeds the image, or ``boxes`` is malformed.
    """
    _check_image(src)
    cw, ch = _size(size)
    h, w = src.shape[:2]
    if cw > w or ch > h:
        raise ValueError(f"size must not exceed the image size ({w}x{h}), got {size}")
    rng = np.random.default_rng(seed)
    x = int(rng.integers(0, w - cw + 1))
    y = int(rng.integers(0, h - ch + 1))
    crop = src[y : y + ch, x : x + cw].copy()
    if boxes is None:
        return crop
    moved = _box_array(boxes)
    if len(moved):
        moved = moved - np.array([x, y, 0, 0], dtype=np.float64)
    return crop, clipBoxes(moved, (cw, ch)).reshape(-1, 4)


def randomFlip(
    src: np.ndarray,
    horizontal: float = 0.5,
    vertical: float = 0.0,
    boxes: ArrayLike | None = None,
    seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Flip the image horizontally and/or vertically with the given probabilities.

    ``cv.flip`` always flips; the coin toss and the box bookkeeping are left to you.

    Args:
        src: Image of any dtype and channel count.
        horizontal: Probability of a horizontal (left-right) flip.
        vertical: Probability of a vertical (top-bottom) flip.
        boxes: Optional boxes ``(N, 4)`` as ``(x, y, width, height)`` to flip along.
        seed: Random seed.

    Returns:
        The (possibly) flipped image, or ``(image, boxes)`` when ``boxes`` is given.

    Raises:
        ValueError: If ``src`` is empty, a probability is outside ``[0, 1]``, or ``boxes`` is malformed.
    """
    _check_image(src)
    for name, p in (("horizontal", horizontal), ("vertical", vertical)):
        check_finite(name, p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be a probability in [0, 1], got {p}")
    rng = np.random.default_rng(seed)
    flip_h = bool(rng.random() < horizontal)
    flip_v = bool(rng.random() < vertical)
    h, w = src.shape[:2]
    dst = src.copy()
    moved = None if boxes is None else _box_array(boxes)
    if flip_h:
        dst = cv2.flip(dst, 1)
        if moved is not None and len(moved):
            moved[:, 0] = w - moved[:, 0] - moved[:, 2]
    if flip_v:
        dst = cv2.flip(dst, 0)
        if moved is not None and len(moved):
            moved[:, 1] = h - moved[:, 1] - moved[:, 3]
    if moved is None:
        return dst
    return dst, moved


def _warp_boxes(boxes: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Transform box corners by a 2x3 or 3x3 matrix and return the clipped axis-aligned boxes."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    corners = _box_corners(boxes).reshape(-1, 1, 2)
    if matrix.shape[0] == 2:
        warped: np.ndarray = cv2.transform(corners, matrix)
    else:
        warped = cv2.perspectiveTransform(corners, matrix)
    return clipBoxes(_corners_to_boxes(warped.reshape(-1, 4, 2)), size).reshape(-1, 4)


def randomRotate(
    src: np.ndarray,
    maxAngle: float = 15,
    boxes: ArrayLike | None = None,
    seed: int | None = None,
    crop: bool = False,
    borderValue: float | Sequence[float] = 0,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Rotate around the center by a random angle in ``[-maxAngle, maxAngle]`` degrees.

    Like :func:`rotateEx`, the canvas grows so no corner is cut off unless ``crop`` is set; boxes are rotated with the
    image and replaced by the axis-aligned box around their rotated corners (they grow a little for large angles).

    Args:
        src: Image of any dtype and channel count.
        maxAngle: Largest rotation in degrees (either direction).
        boxes: Optional boxes ``(N, 4)`` as ``(x, y, width, height)``.
        seed: Random seed.
        crop: Keep the size of ``src`` and cut off the corners, instead of enlarging the canvas.
        borderValue: Fill value or color for the uncovered areas.

    Returns:
        The rotated image, or ``(image, boxes)`` when ``boxes`` is given.

    Raises:
        ValueError: If ``src`` is empty, ``maxAngle`` is negative or not finite, ``borderValue`` is not a valid color,
            or ``boxes`` is malformed.
    """
    _check_image(src)
    check_finite("maxAngle", maxAngle, nonnegative=True)
    fill = _fill_value(borderValue, src)
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-maxAngle, maxAngle))
    h, w = src.shape[:2]
    center = (w / 2, h / 2)
    matrix: np.ndarray = cv2.getRotationMatrix2D(center, angle, 1.0)
    if crop:
        out_w, out_h = w, h
    else:
        radians = math.radians(angle)
        sin, cos = abs(math.sin(radians)), abs(math.cos(radians))
        out_w, out_h = round(h * sin + w * cos), round(h * cos + w * sin)
        matrix[0, 2] += out_w / 2 - center[0]
        matrix[1, 2] += out_h / 2 - center[1]
    dst: np.ndarray = cv2.warpAffine(src, matrix, (out_w, out_h), borderValue=(fill + [0.0] * 4)[:4])
    dst = dst.reshape((out_h, out_w, *src.shape[2:]))
    if boxes is None:
        return dst
    return dst, _warp_boxes(_box_array(boxes), matrix, (out_w, out_h))


def randomScale(
    src: np.ndarray,
    scaleRange: tuple[float, float] = (0.8, 1.2),
    boxes: ArrayLike | None = None,
    seed: int | None = None,
    interpolation: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Resize by a random factor drawn uniformly from ``scaleRange`` (the output size changes).

    Args:
        src: Image of any dtype and channel count.
        scaleRange: ``(low, high)`` factors, both positive.
        boxes: Optional boxes ``(N, 4)`` as ``(x, y, width, height)``; they are scaled by the same factor.
        seed: Random seed.
        interpolation: ``cv.INTER_*`` flag; ``INTER_AREA`` to shrink and ``INTER_LINEAR`` to enlarge by default.

    Returns:
        The resized image, or ``(image, boxes)`` when ``boxes`` is given.

    Raises:
        ValueError: If ``src`` is empty, ``scaleRange`` is not positive or ``low > high``, or ``boxes`` is malformed.
    """
    _check_image(src)
    low, high = float(scaleRange[0]), float(scaleRange[1])
    check_finite("scaleRange", low, positive=True)
    check_finite("scaleRange", high, positive=True)
    if low > high:
        raise ValueError(f"scaleRange must be (low, high) with low <= high, got {scaleRange}")
    rng = np.random.default_rng(seed)
    scale = float(rng.uniform(low, high))
    h, w = src.shape[:2]
    out_w, out_h = max(1, round(w * scale)), max(1, round(h * scale))
    flag = (cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR) if interpolation is None else interpolation
    dst: np.ndarray = cv2.resize(src, (out_w, out_h), interpolation=flag)
    dst = dst.reshape((out_h, out_w, *src.shape[2:]))
    if boxes is None:
        return dst
    # Use the realized factor so boxes match the rounded output size exactly.
    factors = np.array([out_w / w, out_h / h, out_w / w, out_h / h], dtype=np.float64)
    return dst, _box_array(boxes) * factors


def randomPerspective(
    src: np.ndarray,
    distortion: float = 0.1,
    boxes: ArrayLike | None = None,
    seed: int | None = None,
    borderValue: float | Sequence[float] = 0,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Warp the image as if seen from a slightly different viewpoint, by jittering its four corners.

    Args:
        src: Image of any dtype and channel count.
        distortion: Largest corner displacement as a fraction of the width and height, e.g. ``0.1`` for 10%.
        boxes: Optional boxes ``(N, 4)`` as ``(x, y, width, height)``; each is replaced by the axis-aligned box
            around its warped corners, clipped to the image.
        seed: Random seed.
        borderValue: Fill value or color for the uncovered areas.

    Returns:
        The warped image (same size as ``src``), or ``(image, boxes)`` when ``boxes`` is given.

    Raises:
        ValueError: If ``src`` is empty, ``distortion`` is outside ``[0, 0.5)``, ``borderValue`` is not a valid color,
            or ``boxes`` is malformed.
    """
    _check_image(src)
    check_finite("distortion", distortion)
    if not 0.0 <= distortion < 0.5:
        raise ValueError(f"distortion must be in [0, 0.5), got {distortion}")
    fill = _fill_value(borderValue, src)
    rng = np.random.default_rng(seed)
    h, w = src.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    jitter = rng.uniform(-distortion, distortion, (4, 2)) * np.array([w, h], dtype=np.float64)
    target = (corners + jitter).astype(np.float32)
    matrix: np.ndarray = cv2.getPerspectiveTransform(corners, target)
    dst: np.ndarray = cv2.warpPerspective(src, matrix, (w, h), borderValue=(fill + [0.0] * 4)[:4])
    dst = dst.reshape(src.shape)
    if boxes is None:
        return dst
    return dst, _warp_boxes(_box_array(boxes), matrix, (w, h))


# --- random photometric transforms ---------------------------------------------------------------------------------


def randomBrightnessContrast(
    src: np.ndarray, brightness: float = 0.2, contrast: float = 0.2, seed: int | None = None
) -> np.ndarray:
    """Apply :func:`adjustBrightnessContrast` with random amounts.

    Args:
        src: 8-bit grayscale, BGR or BGRA image (the alpha channel is kept).
        brightness: Largest brightness shift as a fraction of 255 (drawn from ``[-brightness, brightness]``).
        contrast: Largest contrast change; the factor is drawn from ``[1 - contrast, 1 + contrast]``.
        seed: Random seed.

    Returns:
        A new image with the same shape and dtype as ``src``.

    Raises:
        ValueError: If ``src`` is not 8-bit or is empty, or a limit is negative, above 1, or not finite.
    """
    for name, limit in (("brightness", brightness), ("contrast", contrast)):
        check_finite(name, limit, nonnegative=True)
        if limit > 1:
            raise ValueError(f"{name} must be at most 1, got {limit}")
    rng = np.random.default_rng(seed)
    shift = float(rng.uniform(-brightness, brightness)) * 255.0
    factor = float(rng.uniform(1.0 - contrast, 1.0 + contrast))
    return adjustBrightnessContrast(src, shift, factor)


def randomHSV(
    src: np.ndarray, hue: float = 10, saturation: float = 0.3, value: float = 0.3, seed: int | None = None
) -> np.ndarray:
    """Apply :func:`adjustHSV` with random amounts (hue shift, saturation and brightness factors).

    Args:
        src: 8-bit BGR or BGRA image (the alpha channel is kept).
        hue: Largest hue shift in degrees (drawn from ``[-hue, hue]``).
        saturation: Largest saturation change; the factor is drawn from ``[1 - saturation, 1 + saturation]``.
        value: Largest brightness change; the factor is drawn from ``[1 - value, 1 + value]``.
        seed: Random seed.

    Returns:
        A new image with the same shape and dtype as ``src``.

    Raises:
        ValueError: If ``src`` is not an 8-bit color image or is empty, or a limit is negative, not finite, or (for
            ``saturation`` and ``value``) above 1.
    """
    check_finite("hue", hue, nonnegative=True)
    for name, limit in (("saturation", saturation), ("value", value)):
        check_finite(name, limit, nonnegative=True)
        if limit > 1:
            raise ValueError(f"{name} must be at most 1, got {limit}")
    rng = np.random.default_rng(seed)
    shift = float(rng.uniform(-hue, hue))
    s = float(rng.uniform(1.0 - saturation, 1.0 + saturation))
    v = float(rng.uniform(1.0 - value, 1.0 + value))
    return adjustHSV(src, shift, s, v)


def randomBlur(src: np.ndarray, maxKernel: int = 7, seed: int | None = None) -> np.ndarray:
    """Gaussian-blur with a random odd kernel size from 1 (no blur) to ``maxKernel``.

    Args:
        src: Image of any dtype and channel count.
        maxKernel: Largest kernel size; rounded down to an odd number.
        seed: Random seed.

    Returns:
        A new image with the same shape and dtype as ``src``.

    Raises:
        ValueError: If ``src`` is empty or ``maxKernel`` is smaller than 1.
    """
    _check_image(src)
    if maxKernel < 1:
        raise ValueError(f"maxKernel must be at least 1, got {maxKernel}")
    rng = np.random.default_rng(seed)
    largest = maxKernel if maxKernel % 2 else maxKernel - 1
    kernel = int(rng.integers(0, (largest - 1) // 2 + 1)) * 2 + 1
    if kernel == 1:
        return src.copy()
    blurred: np.ndarray = cv2.GaussianBlur(src, (kernel, kernel), 0)
    return blurred.reshape(src.shape)


# --- multi-image mixing --------------------------------------------------------------------------------------------


def _same_layout(a: np.ndarray, b: np.ndarray, nameA: str, nameB: str) -> None:
    _check_image(a, nameA)
    _check_image(b, nameB)
    if a.dtype != b.dtype or a.ndim != b.ndim or a.shape[2:] != b.shape[2:]:
        raise ValueError(
            f"{nameA} and {nameB} must have the same dtype and channels, "
            f"got {a.dtype} {a.shape} and {b.dtype} {b.shape}"
        )


def _resize_like(src: np.ndarray, like: np.ndarray) -> np.ndarray:
    if src.shape[:2] == like.shape[:2]:
        return src
    h, w = like.shape[:2]
    resized: np.ndarray = cv2.resize(
        src, (w, h), interpolation=cv2.INTER_LINEAR if src.shape[0] < h else cv2.INTER_AREA
    )
    return resized.reshape(like.shape)


def mosaic(
    images: Sequence[np.ndarray],
    size: tuple[int, int],
    boxes: Sequence[ArrayLike] | None = None,
    seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Combine four images into one YOLO-style mosaic around a random center point.

    Each image is scaled to cover its quadrant (keeping its aspect ratio) and cropped so the quadrants meet at the
    center. Boxes are scaled, shifted and clipped to their quadrant.

    Args:
        images: Exactly four images with the same dtype and channel count (sizes may differ).
        size: Output size as ``(width, height)``.
        boxes: Optional list of four box arrays ``(N_i, 4)`` as ``(x, y, width, height)``, one per image.
        seed: Random seed for the center point.

    Returns:
        The mosaic, or ``(mosaic, boxes)`` when ``boxes`` is given: the transformed boxes of all four images
        concatenated in order, so labels concatenate the same way (boxes outside their quadrant are zero-sized).

    Raises:
        ValueError: If there are not four images, they differ in dtype or channels, ``size`` is not positive, or
            ``boxes`` does not have four entries.
    """
    if len(images) != 4:
        raise ValueError(f"images must contain exactly 4 images, got {len(images)}")
    if boxes is not None and len(boxes) != 4:
        raise ValueError(f"boxes must contain one array per image (4), got {len(boxes)}")
    first = images[0]
    for i, image in enumerate(images):
        _same_layout(first, image, "images[0]", f"images[{i}]")
    out_w, out_h = _size(size)
    rng = np.random.default_rng(seed)
    cx = int(rng.integers(out_w // 4, out_w - out_w // 4 + 1))
    cy = int(rng.integers(out_h // 4, out_h - out_h // 4 + 1))
    # (x0, y0, x1, y1) of each quadrant and which corner of the scaled image anchors at the center.
    quadrants = [(0, 0, cx, cy), (cx, 0, out_w, cy), (0, cy, cx, out_h), (cx, cy, out_w, out_h)]
    anchors = [(1, 1), (0, 1), (1, 0), (0, 0)]  # 1: take the far end (bottom/right) of the scaled image
    dst = np.zeros((out_h, out_w, *first.shape[2:]), dtype=first.dtype)
    all_boxes: list[np.ndarray] = []
    for image, (x0, y0, x1, y1), (ax, ay), index in zip(images, quadrants, anchors, range(4), strict=True):
        qw, qh = x1 - x0, y1 - y0
        if qw <= 0 or qh <= 0:
            if boxes is not None:
                all_boxes.append(np.zeros((len(_box_array(boxes[index])), 4), dtype=np.float64))
            continue
        h, w = image.shape[:2]
        scale = max(qw / w, qh / h)
        sw, sh = max(qw, round(w * scale)), max(qh, round(h * scale))
        scaled: np.ndarray = cv2.resize(
            image, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        )
        scaled = scaled.reshape((sh, sw, *first.shape[2:]))
        ox, oy = (sw - qw) if ax else 0, (sh - qh) if ay else 0
        dst[y0:y1, x0:x1] = scaled[oy : oy + qh, ox : ox + qw]
        if boxes is not None:
            moved = _box_array(boxes[index], f"boxes[{index}]")
            if len(moved):
                moved = moved * np.array([sw / w, sh / h, sw / w, sh / h]) - np.array([ox - x0, oy - y0, 0, 0])
                moved = clipBoxes(moved, (out_w, out_h)).reshape(-1, 4)
                # Clip to the quadrant, not only to the image.
                qx2 = np.clip(moved[:, 0] + moved[:, 2], x0, x1)
                qy2 = np.clip(moved[:, 1] + moved[:, 3], y0, y1)
                qx1 = np.clip(moved[:, 0], x0, x1)
                qy1 = np.clip(moved[:, 1], y0, y1)
                moved = np.stack([qx1, qy1, np.maximum(qx2 - qx1, 0), np.maximum(qy2 - qy1, 0)], axis=1)
            all_boxes.append(moved)
    if boxes is None:
        return dst
    return dst, np.concatenate(all_boxes, axis=0)


def mixup(
    a: np.ndarray, b: np.ndarray, alpha: float = 0.5, lam: float | None = None, seed: int | None = None
) -> tuple[np.ndarray, float]:
    """Blend two images, ``lam * a + (1 - lam) * b`` (mixup), with ``lam`` drawn from ``Beta(alpha, alpha)``.

    ``b`` is resized to the size of ``a``. Use the returned ``lam`` to mix the labels the same way.

    Args:
        a: First image (grayscale, BGR or BGRA, any dtype).
        b: Second image with the same dtype and channels; any size.
        alpha: Beta distribution parameter; smaller values favor one image, ``1`` is uniform.
        lam: Fixed mixing weight in ``[0, 1]`` instead of a random one.
        seed: Random seed.

    Returns:
        ``(image, lam)`` with the image in the dtype of ``a``.

    Raises:
        ValueError: If an image is empty, the images differ in dtype or channels, ``alpha`` is not positive, or
            ``lam`` is outside ``[0, 1]``.
    """
    _same_layout(a, b, "a", "b")
    check_finite("alpha", alpha, positive=True)
    if lam is None:
        lam = float(np.random.default_rng(seed).beta(alpha, alpha))
    check_finite("lam", lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    other = _resize_like(b, a).astype(np.float64)
    return _to_dtype(a.astype(np.float64) * lam + other * (1.0 - lam), a), lam


def cutMix(a: np.ndarray, b: np.ndarray, alpha: float = 1.0, seed: int | None = None) -> tuple[np.ndarray, float]:
    """Paste a random rectangle of ``b`` into ``a`` (CutMix).

    ``b`` is resized to the size of ``a``; the patch covers a fraction ``1 - lam`` of the area, with ``lam`` drawn
    from ``Beta(alpha, alpha)``. The returned ``lam`` is the exact fraction of ``a`` that remains, for label mixing.

    Args:
        a: Image that receives the patch (grayscale, BGR or BGRA, any dtype).
        b: Image the patch is taken from, with the same dtype and channels; any size.
        alpha: Beta distribution parameter; ``1`` draws the area fraction uniformly.
        seed: Random seed.

    Returns:
        ``(image, lam)`` with the image in the dtype of ``a``.

    Raises:
        ValueError: If an image is empty, the images differ in dtype or channels, or ``alpha`` is not positive.
    """
    _same_layout(a, b, "a", "b")
    check_finite("alpha", alpha, positive=True)
    rng = np.random.default_rng(seed)
    lam = float(rng.beta(alpha, alpha))
    h, w = a.shape[:2]
    ratio = math.sqrt(1.0 - lam)
    pw, ph = round(w * ratio), round(h * ratio)
    cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
    x0, x1 = max(0, cx - pw // 2), min(w, cx + (pw - pw // 2))
    y0, y1 = max(0, cy - ph // 2), min(h, cy + (ph - ph // 2))
    dst = a.copy()
    dst[y0:y1, x0:x1] = _resize_like(b, a)[y0:y1, x0:x1]
    return dst, 1.0 - (x1 - x0) * (y1 - y0) / (w * h)
