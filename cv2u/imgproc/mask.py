"""Binary masks: clean-up, contours, skeletons and cut-outs."""

from collections.abc import Sequence

import cv2
import numpy as np

from .._validate import check_finite, check_image_not_empty

__all__ = [
    "applyMask",
    "cutout",
    "fillHoles",
    "findContoursEx",
    "largestComponent",
    "maskBoundingRect",
    "removeSmallObjects",
    "skeletonize",
]


def _foreground(mask: np.ndarray) -> np.ndarray:
    """Nonzero pixels as a uint8 0/1 array (any channel count; for color images, any nonzero channel)."""
    if mask.ndim not in (2, 3):
        raise ValueError(f"mask must be a 2-D array or an image, got shape {mask.shape}")
    check_image_not_empty("mask", mask)
    if mask.ndim == 3:
        return np.asarray((mask != 0).any(axis=2), dtype=np.uint8)
    return np.asarray(mask != 0, dtype=np.uint8)


def fillHoles(mask: np.ndarray, maxHoleArea: int | None = None) -> np.ndarray:
    """Fill the holes of the objects in a binary mask, i.e. background regions not connected to the image border.

    Args:
        mask: Binary mask; nonzero (or ``True``) pixels are foreground. Color images count any nonzero channel.
        maxHoleArea: If given, only holes up to this many pixels are filled.

    Returns:
        ``uint8`` mask with values 0 and 255.

    Raises:
        ValueError: If ``mask`` is not a 2-D array or an image, is empty, or ``maxHoleArea`` is negative.
    """
    if maxHoleArea is not None and maxHoleArea < 0:
        raise ValueError(f"maxHoleArea must not be negative, got {maxHoleArea}")
    fg = _foreground(mask)
    # Holes are 4-connected background regions (the complement of 8-connected foreground).
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - fg, connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    is_hole = np.ones(count, dtype=bool)
    is_hole[0] = False  # label 0 is the foreground itself
    is_hole[border] = False
    if maxHoleArea is not None:
        is_hole &= stats[:, cv2.CC_STAT_AREA] <= maxHoleArea
    filled = fg.astype(bool) | is_hole[labels]
    return np.asarray(filled, dtype=np.uint8) * 255


def removeSmallObjects(mask: np.ndarray, minArea: int, connectivity: int = 8) -> np.ndarray:
    """Remove connected foreground regions smaller than ``minArea`` pixels from a binary mask.

    Args:
        mask: Binary mask; nonzero (or ``True``) pixels are foreground.
        minArea: Regions with fewer pixels are removed.
        connectivity: 8 (diagonal neighbors connect) or 4.

    Returns:
        ``uint8`` mask with values 0 and 255.

    Raises:
        ValueError: If ``connectivity`` is not 4 or 8, ``minArea`` is negative, or ``mask`` is not a 2-D array or an
            image, or is empty.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if minArea < 0:
        raise ValueError(f"minArea must not be negative, got {minArea}")
    fg = _foreground(mask)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=connectivity)
    keep = stats[:, cv2.CC_STAT_AREA] >= minArea
    keep[0] = False  # background
    return np.asarray(keep[labels], dtype=np.uint8) * 255


def findContoursEx(
    image: np.ndarray,
    mode: int = cv2.RETR_EXTERNAL,
    method: int = cv2.CHAIN_APPROX_SIMPLE,
    minArea: float = 0.0,
    sort: bool = True,
) -> list[np.ndarray]:
    """``cv.findContours`` for any image, with the contours sorted by area and small ones filtered out.

    ``cv.findContours`` only accepts single-channel 8-bit images and returns the contours in scan order. Here any
    image works (nonzero pixels are foreground, so pass a thresholded image or mask), and the result is a plain
    list, largest contour first. Use ``cv.findContours`` directly when you need the hierarchy.

    Args:
        image: Mask or image; nonzero (or ``True``) pixels are foreground (for color images, any nonzero channel).
        mode: Retrieval mode, as for ``cv.findContours`` (default ``cv.RETR_EXTERNAL``: outer contours only).
        method: Approximation, as for ``cv.findContours``.
        minArea: Drop contours whose ``cv.contourArea`` is smaller than this.
        sort: Sort by area, largest first. If false, keep OpenCV's order.

    Returns:
        A list of contours, each of shape ``(N, 1, 2)``, ready for ``cv.drawContours``.

    Raises:
        ValueError: If ``image`` is not a 2-D array or an image, is empty, or ``minArea`` is not a finite number.
    """
    check_finite("minArea", minArea)
    fg = _foreground(image)
    contours: Sequence[np.ndarray]
    contours, _ = cv2.findContours(fg, mode, method)
    if not sort and minArea <= 0:
        return list(contours)  # nothing to measure
    items = [(c, cv2.contourArea(c)) for c in contours]
    items = [(c, a) for c, a in items if a >= minArea]
    if sort:
        items.sort(key=lambda item: item[1], reverse=True)
    return [c for c, _ in items]


def largestComponent(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Keep only the largest connected foreground region of a binary mask (e.g. the main object of a segmentation).

    Returns:
        ``uint8`` mask with values 0 and 255; all zeros if the mask is empty.

    Raises:
        ValueError: If ``connectivity`` is not 4 or 8, or ``mask`` is not a 2-D array or an image, or is empty.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    fg = _foreground(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=connectivity)
    if count <= 1:
        return np.zeros_like(fg)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.asarray(labels == largest, dtype=np.uint8) * 255


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Thin the shapes of a binary mask to 1-pixel-wide lines that keep their topology (Zhang-Suen thinning).

    Useful for measuring line lengths, finding junctions or vectorizing drawings. OpenCV's thinning lives in the
    ``ximgproc`` contrib module, which ``opencv-python`` does not include.

    Returns:
        ``uint8`` mask with values 0 and 255.

    Raises:
        ValueError: If ``mask`` is not a 2-D array or an image, or is empty.
    """
    img = np.pad(_foreground(mask), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            center = img[1:-1, 1:-1]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbors = sum(n.astype(np.int32) for n in ring[:8])
            transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int32) for i in range(8))
            if step == 0:
                first, second = p2 * p4 * p6, p4 * p6 * p8
            else:
                first, second = p2 * p4 * p8, p2 * p6 * p8
            remove = (center == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1)
            remove &= (first == 0) & (second == 0)
            if remove.any():
                center[remove] = 0
                changed = True
        if not changed:
            return np.asarray(img[1:-1, 1:-1], dtype=np.uint8) * 255


def cutout(src: np.ndarray, mask: np.ndarray, feather: float = 0) -> np.ndarray:
    """Cut an object out of an image: a BGRA image whose alpha channel is the mask, ready to save as PNG.

    Args:
        src: Grayscale, BGR or BGRA image (an existing alpha channel is combined with the mask).
        mask: 2-D mask with the shape ``src.shape[:2]``. ``bool`` and 0/255 masks give hard edges; other 8-bit values
            are used as partial transparency.
        feather: Soften the edge with a Gaussian blur of this sigma, in pixels.

    Returns:
        ``uint8`` BGRA image.

    Raises:
        ValueError: If ``src`` is not 8-bit or is empty, the mask shape does not match, or ``feather`` is negative.
    """
    from .color import toBGRA

    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    if mask.ndim != 2 or mask.shape != src.shape[:2]:
        raise ValueError(f"mask must have the shape {src.shape[:2]}, got {mask.shape}")
    check_image_not_empty("src", src)
    check_finite("feather", feather, nonnegative=True)
    alpha = (mask.astype(np.uint8) * 255) if mask.dtype == bool else np.clip(mask, 0, 255).astype(np.uint8)
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
    out = toBGRA(src)
    out[..., 3] = (out[..., 3].astype(np.uint16) * alpha // 255).astype(np.uint8)
    return out


def maskBoundingRect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box ``(x, y, width, height)`` of the nonzero pixels of a mask, or ``None`` if it is empty.

    Simpler than ``cv.boundingRect(cv.findNonZero(mask))``, which fails on color masks and on an all-zero mask.

    Raises:
        ValueError: If ``mask`` is not a 2-D array or an image, or has no pixels.
    """
    fg = _foreground(mask)
    rows = np.flatnonzero(fg.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(fg.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)


def applyMask(src: np.ndarray, mask: np.ndarray, background: float | Sequence[float] = 0) -> np.ndarray:
    """Keep the pixels where ``mask`` is nonzero and fill the rest with ``background``.

    ``cv.bitwise_and(src, src, mask=mask)`` only gives a black background, wants an 8-bit single-channel mask,
    and fails on float images. This works for any dtype and channel count and any background color.

    Args:
        src: Image.
        mask: 2-D array (or image) of the shape ``src.shape[:2]``; nonzero (or ``True``) pixels are kept.
        background: A scalar, or one value per channel of ``src``.

    Returns:
        A new array with the dtype and shape of ``src``.

    Raises:
        ValueError: If the mask shape does not match, or ``src`` is empty.
    """
    check_image_not_empty("src", src)
    fg = _foreground(mask)
    if fg.shape != src.shape[:2]:
        raise ValueError(f"mask must have the shape {src.shape[:2]}, got {mask.shape[:2]}")
    out = np.empty_like(src)
    out[...] = background
    keep = fg.astype(bool)
    out[keep] = src[keep]
    return out
