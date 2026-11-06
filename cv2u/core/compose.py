"""Combine images: side-by-side layouts, grids and alpha-blended overlays."""

import math
from collections.abc import Sequence

import cv2
import numpy as np

from .._validate import check_color, check_image_not_empty
from ..imgproc.resize import letterbox

__all__ = ["hconcatEx", "montage", "overlayImage", "vconcatEx"]

_UPCONVERT = {(1, 3): cv2.COLOR_GRAY2BGR, (1, 4): cv2.COLOR_GRAY2BGRA, (3, 4): cv2.COLOR_BGR2BGRA}


def _channels(img: np.ndarray) -> int:
    return 1 if img.ndim == 2 else int(img.shape[2])


def _as3d(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 3 else img[..., None]


def _fill_value(color: float | Sequence[float], channels: int) -> list[float]:
    if isinstance(color, (int, float)):
        values = [float(color)] * channels
        check_color("color", values)
        return values
    values = [float(c) for c in color]
    if channels == 1 and len(values) in (3, 4):
        b, g, r = values[:3]
        values = [0.114 * b + 0.587 * g + 0.299 * r]  # a BGR color on grayscale images: use its luminance
    if channels == 4 and len(values) == 3:
        values.append(255.0)  # opaque padding for BGRA images
    if len(values) != channels:
        raise ValueError(f"color must be a scalar or have {channels} values, got {color!r}")
    check_color("color", values)
    return values


def _concat(images: Sequence[np.ndarray], axis: int, color: float | Sequence[float], gap: int) -> np.ndarray:
    if len(images) == 0:
        raise ValueError("images must not be empty")
    if gap < 0:
        raise ValueError(f"gap must not be negative, got {gap}")
    dtypes = {img.dtype for img in images}
    if len(dtypes) != 1:
        raise ValueError(f"all images must have the same dtype, got {sorted(str(d) for d in dtypes)}")
    for img in images:
        if img.ndim not in (2, 3) or _channels(img) not in (1, 3, 4):
            raise ValueError(f"images must have 1, 3 or 4 channels, got shape {img.shape}")
        check_image_not_empty("images", img)

    channels = max(_channels(img) for img in images)
    unified = [
        _as3d(img if _channels(img) == channels else cv2.cvtColor(img, _UPCONVERT[(_channels(img), channels)]))
        for img in images
    ]
    dtype = unified[0].dtype
    values = _fill_value(color, channels)
    check_color("color", values, dtype)
    fill = np.array(values, dtype=dtype)

    cross = 1 - axis  # the axis that is padded: height for hconcat, width for vconcat
    extent = max(img.shape[cross] for img in unified)

    def block(length: int) -> np.ndarray:
        shape = [0, 0, channels]
        shape[axis], shape[cross] = length, extent
        return np.broadcast_to(fill, shape)

    parts: list[np.ndarray] = []
    for i, img in enumerate(unified):
        if i and gap:
            parts.append(block(gap))
        missing = extent - img.shape[cross]
        if missing:
            pad_shape = list(img.shape)
            pad_shape[cross] = missing
            img = np.concatenate([img, np.broadcast_to(fill, pad_shape)], axis=cross)
        parts.append(img)
    result = np.concatenate(parts, axis=axis)
    return result[..., 0] if channels == 1 else result


def hconcatEx(images: Sequence[np.ndarray], color: float | Sequence[float] = 0, gap: int = 0) -> np.ndarray:
    """Place images side by side, like ``cv.hconcat`` but for images of different sizes and channels.

    Shorter images are padded at the bottom with ``color``. Grayscale and color images can be mixed: all images are
    converted to the largest channel count (gray to BGR, BGR to BGRA).

    Args:
        images: Images with the same dtype.
        color: Color of the padding and the gaps: a scalar, or one value per channel (BGR, or BGRA).
        gap: Space in pixels between the images.

    Raises:
        ValueError: If ``images`` is empty, the dtypes differ, an image has an unsupported channel count,
            or ``color`` does not match the channel count.
    """
    return _concat(images, axis=1, color=color, gap=gap)


def vconcatEx(images: Sequence[np.ndarray], color: float | Sequence[float] = 0, gap: int = 0) -> np.ndarray:
    """Stack images vertically, like ``cv.vconcat`` but for images of different sizes and channels.

    Narrower images are padded on the right with ``color``. See :func:`hconcatEx` for the details.

    Raises:
        ValueError: If ``images`` is empty, the dtypes differ, an image has an unsupported channel count,
            or ``color`` does not match the channel count.
    """
    return _concat(images, axis=0, color=color, gap=gap)


def montage(
    images: Sequence[np.ndarray],
    cols: int | None = None,
    gap: int = 0,
    color: float | Sequence[float] = 0,
    cellSize: tuple[int, int] | None = None,
) -> np.ndarray:
    """Arrange images in a grid, e.g. to compare many results at once.

    Args:
        images: Images with the same dtype; sizes and channel counts may differ (see :func:`hconcatEx`).
        cols: Number of columns. By default the grid is as square as possible.
        gap: Space in pixels between the cells.
        color: Color of the gaps and padding.
        cellSize: If given, every image is fitted into ``(width, height)`` with :func:`letterbox` first, so all
            cells have the same size.

    Raises:
        ValueError: If ``images`` is empty or ``cols`` is not positive (and see :func:`hconcatEx`).
    """
    if len(images) == 0:
        raise ValueError("images must not be empty")
    columns = math.ceil(math.sqrt(len(images))) if cols is None else cols
    if columns < 1:
        raise ValueError(f"cols must be positive, got {cols}")
    cells = list(images) if cellSize is None else [letterbox(img, cellSize, color=color)[0] for img in images]
    rows = [hconcatEx(cells[i : i + columns], color=color, gap=gap) for i in range(0, len(cells), columns)]
    return vconcatEx(rows, color=color, gap=gap)


def overlayImage(
    src: np.ndarray, overlay: np.ndarray, org: tuple[int, int] = (0, 0), opacity: float = 1.0
) -> np.ndarray:
    """Draw ``overlay`` on top of ``src`` at ``org``, using the overlay's alpha channel if it has one.

    Typical use: putting a transparent PNG (read with ``cv.IMREAD_UNCHANGED``) such as a logo onto a photo.
    Parts of ``overlay`` outside ``src`` are clipped, so ``org`` may also be negative.

    Args:
        src: 8-bit grayscale, BGR or BGRA image. It is not modified.
        overlay: 8-bit grayscale, BGR or BGRA image. Its alpha channel (if any) controls the blending per pixel.
        org: Position of the overlay's top-left corner in ``src``, as ``(x, y)``.
        opacity: Overall opacity from 0 (invisible) to 1, multiplied with the overlay's alpha.

    Returns:
        A new image with the same shape as ``src``.

    Raises:
        ValueError: If an image is not 8-bit or has an unsupported channel count, or ``opacity`` is outside ``[0, 1]``.
    """
    if src.dtype != np.uint8 or overlay.dtype != np.uint8:
        raise ValueError(f"src and overlay must be 8-bit (uint8), got {src.dtype} and {overlay.dtype}")
    check_image_not_empty("src", src)
    check_image_not_empty("overlay", overlay)
    for name, img in (("src", src), ("overlay", overlay)):
        if _channels(img) not in (1, 3, 4):
            raise ValueError(f"{name} must have 1, 3 or 4 channels, got shape {img.shape}")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"opacity must be between 0 and 1, got {opacity}")

    dst = src.copy()
    x, y = org
    oh, ow = overlay.shape[:2]
    # Intersection of the overlay rectangle with src.
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + ow, src.shape[1]), min(y + oh, src.shape[0])
    if x0 >= x1 or y0 >= y1:
        return dst

    patch = _as3d(overlay[y0 - y : y1 - y, x0 - x : x1 - x])
    if patch.shape[2] == 4:
        alpha = patch[..., 3:4].astype(np.float32) / 255.0
        fg = patch[..., :3]
    else:
        alpha = np.ones((*patch.shape[:2], 1), np.float32)
        fg = patch
    alpha *= opacity

    region = _as3d(dst[y0:y1, x0:x1])  # a view into dst
    src_channels = region.shape[2]
    color_channels = 1 if src_channels == 1 else 3
    if fg.shape[2] != color_channels:
        fg = _as3d(cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY if color_channels == 1 else cv2.COLOR_GRAY2BGR))

    base = region[..., :color_channels].astype(np.float32)
    blended = base * (1 - alpha) + fg.astype(np.float32) * alpha
    region[..., :color_channels] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    if src_channels == 4:
        # "Over" compositing for the destination alpha.
        dst_alpha = region[..., 3:4].astype(np.float32) / 255.0
        region[..., 3:4] = np.rint((alpha + dst_alpha * (1 - alpha)) * 255).astype(np.uint8)
    return dst
