"""Bounding boxes ``(x, y, width, height)``, as used by OpenCV and detectors."""

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["boxIoU"]


def _boxes(boxes: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(boxes, dtype=np.float64)
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must be one box (x, y, w, h) or an array of shape (N, 4), got shape {array.shape}")
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
