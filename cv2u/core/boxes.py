"""Bounding boxes ``(x, y, width, height)``, as used by OpenCV and detectors.

Detector output needs the same handful of steps in every project: convert between ``(x, y, w, h)`` and
``(x1, y1, x2, y2)``, suppress overlapping boxes, clip to the image, undo a letterbox, pad a box before cropping.
OpenCV has ``cv.dnn.NMSBoxes`` for one of those and nothing for the rest.
"""

import numpy as np
from numpy.typing import ArrayLike

from .._validate import check_finite

__all__ = ["boxIoU", "clipBoxes", "expandBoxes", "nms", "scaleBoxes", "xywhToXyxy", "xyxyToXywh"]


def _boxes(boxes: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(boxes, dtype=np.float64)
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must be one box (x, y, w, h) or an array of shape (N, 4), got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def boxIoU(boxesA: ArrayLike, boxesB: ArrayLike) -> np.ndarray:
    """Intersection over union of every box in ``boxesA`` with every box in ``boxesB``.

    The standard overlap measure for matching detections to ground truth or tracking boxes between frames.
    OpenCV only uses it inside ``cv.dnn.NMSBoxes``.

    Args:
        boxesA: One box ``(x, y, width, height)`` or an array of shape ``(N, 4)``.
        boxesB: One box or an array of shape ``(M, 4)``.

    Returns:
        A float64 array of shape ``(N, M)`` with values from 0 (no overlap) to 1 (identical boxes);
        ``float(boxIoU(a, b)[0, 0])`` for two single boxes.

    Raises:
        ValueError: If the boxes do not have 4 values.
    """
    a, b = _boxes(boxesA, "boxesA"), _boxes(boxesB, "boxesB")
    ax0, ay0, aw, ah = (a[:, i, None] for i in range(4))
    bx0, by0, bw, bh = (b[None, :, i] for i in range(4))
    iw = np.clip(np.minimum(ax0 + aw, bx0 + bw) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay0 + ah, by0 + bh) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    iou: np.ndarray = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return iou


def xywhToXyxy(boxes: ArrayLike) -> np.ndarray:
    """Convert boxes from ``(x, y, width, height)`` to corners ``(x1, y1, x2, y2)``.

    OpenCV (``cv.boundingRect``, ``cv.rectangle`` with a rect, ``cv.dnn.NMSBoxes``) uses ``(x, y, w, h)``;
    most detector APIs and slicing code want corners. The input shape and dtype are kept.
    """
    array = np.array(boxes, copy=True)
    if array.shape[-1:] != (4,):
        raise ValueError(f"boxes must end in 4 values (x, y, w, h), got shape {array.shape}")
    array[..., 2] += array[..., 0]
    array[..., 3] += array[..., 1]
    return array


def xyxyToXywh(boxes: ArrayLike) -> np.ndarray:
    """Convert boxes from corners ``(x1, y1, x2, y2)`` to ``(x, y, width, height)`` (inverse of :func:`xywhToXyxy`)."""
    array = np.array(boxes, copy=True)
    if array.shape[-1:] != (4,):
        raise ValueError(f"boxes must end in 4 values (x1, y1, x2, y2), got shape {array.shape}")
    array[..., 2] -= array[..., 0]
    array[..., 3] -= array[..., 1]
    return array


def nms(
    boxes: ArrayLike,
    scores: ArrayLike,
    iouThreshold: float = 0.5,
    scoreThreshold: float = 0.0,
    maxCount: int | None = None,
) -> np.ndarray:
    """Non-maximum suppression: the indices of the boxes to keep, best score first.

    ``cv.dnn.NMSBoxes`` does the same, but wants Python lists of a specific type, returns an empty tuple
    instead of an empty array when nothing survives, and has no size limit.

    Args:
        boxes: Boxes ``(x, y, width, height)``, shape ``(N, 4)``.
        scores: One score per box.
        iouThreshold: Boxes overlapping a kept box by more than this IoU are dropped.
        scoreThreshold: Boxes scoring below this are dropped first.
        maxCount: Keep at most this many boxes.

    Returns:
        An ``int64`` array of indices into ``boxes``, in descending score order (empty if none survive).

    Raises:
        ValueError: If ``boxes`` and ``scores`` disagree in length, or a threshold is out of range.
    """
    array = _boxes(boxes, "boxes") if np.size(boxes) else np.zeros((0, 4))
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(score_array) != len(array):
        raise ValueError(f"scores must have one value per box, got {len(score_array)} for {len(array)} boxes")
    check_finite("iouThreshold", iouThreshold, nonnegative=True)
    check_finite("scoreThreshold", scoreThreshold)
    if maxCount is not None and maxCount < 0:
        raise ValueError(f"maxCount must not be negative, got {maxCount}")
    if len(array) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-score_array, kind="stable")
    order = order[score_array[order] >= scoreThreshold]
    keep: list[int] = []
    while len(order) and (maxCount is None or len(keep) < maxCount):
        best = int(order[0])
        keep.append(best)
        rest = order[1:]
        if len(rest) == 0:
            break
        overlap = boxIoU(array[best], array[rest])[0]
        order = rest[overlap <= iouThreshold]
    return np.array(keep, dtype=np.int64)


def clipBoxes(boxes: ArrayLike, size: tuple[int, int]) -> np.ndarray:
    """Clip boxes ``(x, y, width, height)`` to an image of ``size`` ``(width, height)``.

    Boxes that lie completely outside become zero-sized at the border. Useful before :func:`cropEx` or
    :func:`drawLabel` on detections near the edge.

    Returns:
        A float64 array of shape ``(N, 4)``, or ``(4,)`` for a single input box.
    """
    width, height = size
    if width < 1 or height < 1:
        raise ValueError(f"size must be at least 1x1, got {size}")
    single = np.ndim(boxes) == 1
    array = _boxes(boxes, "boxes")
    x1 = np.clip(array[:, 0], 0, width)
    y1 = np.clip(array[:, 1], 0, height)
    x2 = np.clip(array[:, 0] + array[:, 2], 0, width)
    y2 = np.clip(array[:, 1] + array[:, 3], 0, height)
    out = np.stack([x1, y1, np.maximum(x2 - x1, 0), np.maximum(y2 - y1, 0)], axis=1)
    return out[0] if single else out


def scaleBoxes(
    boxes: ArrayLike,
    scale: float | tuple[float, float],
    offset: tuple[float, float] = (0, 0),
    inverse: bool = False,
) -> np.ndarray:
    """Scale (and shift) boxes, e.g. to map detections on a resized or letterboxed image back to the original.

    With ``scale`` and ``offset`` from :func:`letterbox`, ``scaleBoxes(boxes, scale, offset, inverse=True)``
    returns the boxes in the coordinates of the original image. Without ``inverse``, the boxes are mapped the
    other way: ``x * scale + offset``.

    Args:
        boxes: Boxes ``(x, y, width, height)``.
        scale: One factor, or ``(sx, sy)``.
        offset: ``(dx, dy)`` added after scaling (the padding of a letterbox).
        inverse: Undo the transform instead: ``(x - offset) / scale``.

    Returns:
        A float64 array of shape ``(N, 4)``, or ``(4,)`` for a single input box.

    Raises:
        ValueError: If a factor is not positive and finite.
    """
    sx, sy = (scale, scale) if isinstance(scale, (int, float)) else scale
    check_finite("scale", sx, positive=True)
    check_finite("scale", sy, positive=True)
    dx, dy = offset
    check_finite("offset", dx)
    check_finite("offset", dy)
    single = np.ndim(boxes) == 1
    array = _boxes(boxes, "boxes")
    if inverse:
        out = np.stack([(array[:, 0] - dx) / sx, (array[:, 1] - dy) / sy, array[:, 2] / sx, array[:, 3] / sy], axis=1)
    else:
        out = np.stack([array[:, 0] * sx + dx, array[:, 1] * sy + dy, array[:, 2] * sx, array[:, 3] * sy], axis=1)
    return out[0] if single else out


def expandBoxes(
    boxes: ArrayLike,
    ratio: float = 0.0,
    pixels: float = 0.0,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Grow (or shrink) boxes around their center, e.g. to add context before cropping a face or an object.

    Args:
        boxes: Boxes ``(x, y, width, height)``.
        ratio: Fraction of the width and height to add on each side (``0.1`` adds 10% left and right, 10% top and
            bottom). Negative values shrink.
        pixels: Fixed number of pixels to add on each side, after ``ratio``.
        size: ``(width, height)`` of the image; when given, the result is clipped to it.

    Returns:
        A float64 array of shape ``(N, 4)``, or ``(4,)`` for a single input box. Widths and heights never go
        below 0.

    Raises:
        ValueError: If ``ratio`` or ``pixels`` is not finite.
    """
    check_finite("ratio", ratio)
    check_finite("pixels", pixels)
    single = np.ndim(boxes) == 1
    array = _boxes(boxes, "boxes")
    # Shrinking stops at the center point, so the box never flips.
    grow_x = np.maximum(array[:, 2] * ratio + pixels, -array[:, 2] / 2)
    grow_y = np.maximum(array[:, 3] * ratio + pixels, -array[:, 3] / 2)
    out = np.stack(
        [
            array[:, 0] - grow_x,
            array[:, 1] - grow_y,
            np.maximum(array[:, 2] + 2 * grow_x, 0),
            np.maximum(array[:, 3] + 2 * grow_y, 0),
        ],
        axis=1,
    )
    if size is not None:
        out = clipBoxes(out, size)
    return out[0] if single else out
