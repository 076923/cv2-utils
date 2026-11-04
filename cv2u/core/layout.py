"""Image layout: padding to a size or multiple, and splitting into overlapping tiles."""

import math
from collections.abc import Sequence

import cv2
import numpy as np

__all__ = ["mergeTiles", "padToMultiple", "padToSize", "splitTiles"]

Color = float | Sequence[float]


def _border_value(color: Color) -> list[float]:
    return [float(color)] * 4 if isinstance(color, (int, float)) else [float(c) for c in color]


def padToMultiple(src: np.ndarray, multiple: int, color: Color = 0) -> np.ndarray:
    """Pad the right and bottom so that the width and height become multiples of ``multiple``.

    Typical for neural networks that need sizes divisible by 32. The original pixels stay at the top-left, so
    coordinates do not change; crop the result back with ``dst[:h, :w]``.

    Raises:
        ValueError: If ``multiple`` is not positive.
    """
    if multiple < 1:
        raise ValueError(f"multiple must be positive, got {multiple}")
    h, w = src.shape[:2]
    bottom, right = -h % multiple, -w % multiple
    if bottom == right == 0:
        return src.copy()
    dst: np.ndarray = cv2.copyMakeBorder(src, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=_border_value(color))
    return dst


def padToSize(src: np.ndarray, size: tuple[int, int], color: Color = 0, center: bool = True) -> np.ndarray:
    """Pad an image to exactly ``size`` without scaling it (use :func:`letterbox` to also scale).

    Args:
        src: Image to pad.
        size: Output size as ``(width, height)``; it must not be smaller than the image.
        color: Padding color.
        center: Center the image; otherwise it stays at the top-left.

    Raises:
        ValueError: If ``size`` is smaller than the image.
    """
    out_w, out_h = size
    h, w = src.shape[:2]
    if out_w < w or out_h < h:
        raise ValueError(f"size {size} is smaller than the image ({w}, {h}); use letterbox or resizeEx to shrink")
    left = (out_w - w) // 2 if center else 0
    top = (out_h - h) // 2 if center else 0
    dst: np.ndarray = cv2.copyMakeBorder(
        src, top, out_h - h - top, left, out_w - w - left, cv2.BORDER_CONSTANT, value=_border_value(color)
    )
    return dst


def _starts(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    count = math.ceil((length - tile) / step) + 1
    starts = [min(i * step, length - tile) for i in range(count)]  # the last tile is moved inside the image
    return sorted(set(starts))


def splitTiles(
    src: np.ndarray, tileSize: tuple[int, int], overlap: int = 0
) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Split a large image into tiles, e.g. to run a model with a fixed input size over a big image.

    Every tile has exactly ``tileSize`` (unless the image itself is smaller): the last row and column of tiles are
    shifted inside the image instead of being cut off, so they overlap their neighbors a little more.

    Args:
        src: Image to split.
        tileSize: Tile size as ``(width, height)``.
        overlap: Pixels shared by neighboring tiles, to avoid artifacts at tile borders.

    Returns:
        ``(tile, (x, y))`` pairs in row-major order, where ``(x, y)`` is the tile's top-left corner in ``src``.
        Tiles are views of ``src``; use :func:`mergeTiles` to put processed tiles back together.

    Raises:
        ValueError: If ``tileSize`` is not positive or ``overlap`` is not smaller than the tile.
    """
    tile_w, tile_h = tileSize
    if tile_w < 1 or tile_h < 1:
        raise ValueError(f"tileSize must be positive, got {tileSize}")
    if not 0 <= overlap < min(tile_w, tile_h):
        raise ValueError(f"overlap must be in [0, {min(tile_w, tile_h)}), got {overlap}")
    h, w = src.shape[:2]
    xs = _starts(w, tile_w, tile_w - overlap)
    ys = _starts(h, tile_h, tile_h - overlap)
    return [(src[y : y + tile_h, x : x + tile_w], (x, y)) for y in ys for x in xs]


def mergeTiles(tiles: Sequence[tuple[np.ndarray, tuple[int, int]]], size: tuple[int, int]) -> np.ndarray:
    """Assemble tiles from :func:`splitTiles` (possibly processed) into one image; overlaps are averaged.

    Args:
        tiles: ``(tile, (x, y))`` pairs. Tiles may have any channel count, as long as all are the same.
        size: Output size as ``(width, height)``, normally the size of the original image.

    Returns:
        The assembled image, with the dtype of the tiles. Pixels covered by no tile are 0.

    Raises:
        ValueError: If ``tiles`` is empty.
    """
    if len(tiles) == 0:
        raise ValueError("tiles must not be empty")
    width, height = size
    first = tiles[0][0]
    shape = (height, width, *first.shape[2:])
    total = np.zeros(shape, np.float64)
    count = np.zeros((height, width), np.float64)
    for tile, (x, y) in tiles:
        h, w = tile.shape[:2]
        h, w = min(h, height - y), min(w, width - x)
        total[y : y + h, x : x + w] += tile[:h, :w]
        count[y : y + h, x : x + w] += 1
    divisor = np.maximum(count, 1)
    averaged = total / (divisor[..., None] if total.ndim == 3 else divisor)
    if np.issubdtype(first.dtype, np.integer):
        info = np.iinfo(first.dtype)
        averaged = np.clip(np.rint(averaged), info.min, info.max)
    return averaged.astype(first.dtype)
