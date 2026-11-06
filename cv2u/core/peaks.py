"""``peakColors``: cv2u's own color clustering, by climbing the peaks of the color histogram.

How it works:

1. Count the pixels in a 3-D Lab color histogram (``bins`` per axis). Identical colors, which are very common in 8-bit
   images, simply add up instead of distorting the result.
2. Smooth the histogram with a Gaussian, so that noise does not create small peaks.
3. Every histogram cell points to its highest neighbor (26-neighborhood); following the pointers ends at a local
   maximum, a *peak*. Each peak is one color cluster, so the number of colors is found automatically. Ties are broken
   by a strict order, so the climb always ends.
4. Peaks holding less than ``minShare`` of the pixels are merged into the nearest strong peak.
5. Optionally, image regions smaller than ``minRegion`` pixels are absorbed by the surrounding cluster.

It needs only NumPy and OpenCV, is fully deterministic (no sampling, no random initialization), and runs in time
linear in the number of pixels.
"""

import itertools
import math

import cv2
import numpy as np

__all__ = ["peakColors"]

_NEIGHBORS = [d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)]


def _smooth(hist: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of a 3-D array (zero padding)."""
    if sigma <= 0:
        return hist
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    out = hist
    for axis in range(3):
        padded = np.pad(out, [(radius, radius) if a == axis else (0, 0) for a in range(3)])
        out = sum(
            kernel[i] * np.take(padded, np.arange(i, i + hist.shape[axis]), axis=axis) for i in range(kernel.size)
        )
    return np.asarray(out)


def _climb(height: np.ndarray) -> np.ndarray:
    """For every cell, the flat index of the peak reached by steepest ascent."""
    bins = height.shape[0]
    size = height.size
    index = np.arange(size).reshape(height.shape)
    best_value = height.copy()
    best_index = index.copy()
    padded_value = np.pad(height, 1, constant_values=-1.0)
    padded_index = np.pad(index, 1, constant_values=-1)
    for dz, dy, dx in _NEIGHBORS:
        window = (slice(1 + dz, 1 + dz + bins), slice(1 + dy, 1 + dy + bins), slice(1 + dx, 1 + dx + bins))
        value, idx = padded_value[window], padded_index[window]
        # Strict total order (value, index): the climb can never loop on plateaus.
        better = (value > best_value) | ((value == best_value) & (idx > best_index))
        best_value = np.where(better, value, best_value)
        best_index = np.where(better, idx, best_index)
    parent = best_index.ravel()
    while True:  # pointer jumping: O(log length) rounds
        grand = parent[parent]
        if np.array_equal(grand, parent):
            return parent
        parent = grand


def _absorb_small_regions(maps: np.ndarray, minRegion: int) -> np.ndarray:
    """Relabel 4-connected regions smaller than ``minRegion`` with the most common label around them."""
    out = maps.copy()
    kernel = np.ones((3, 3), np.uint8)
    for label in np.unique(maps):
        count, components, stats, _ = cv2.connectedComponentsWithStats((maps == label).astype(np.uint8), connectivity=4)
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] >= minRegion:
                continue
            x, y, w, h = stats[component, :4]
            x0, y0 = max(x - 1, 0), max(y - 1, 0)
            x1, y1 = min(x + w + 1, maps.shape[1]), min(y + h + 1, maps.shape[0])
            region = (components[y0:y1, x0:x1] == component).astype(np.uint8)
            ring = cv2.dilate(region, kernel).astype(bool) & ~region.astype(bool)
            around = out[y0:y1, x0:x1][ring]
            around = around[around != label]
            if around.size:
                out[y0:y1, x0:x1][region.astype(bool)] = int(np.argmax(np.bincount(around)))
    return out


def peakColors(
    src: np.ndarray,
    bins: int = 32,
    sigma: float = 0.75,
    minShare: float = 0.01,
    minRegion: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the pixel colors at the peaks of the color histogram, finding the number of colors by itself.

    cv2u's own method (see the module documentation for the algorithm). Compared with the other color clusterings:

    - like :func:`meanShiftColors`, the number of colors is found automatically, but no extra is needed;
    - unlike :func:`kmeansColors` and :func:`meanShiftColors`, every pixel is used (no sampling) and the result is
      fully deterministic, while the run time stays linear in the image size.

    Args:
        src: 8-bit BGR image.
        bins: Histogram cells per Lab axis. More bins separate similar colors; fewer bins merge them.
        sigma: Histogram smoothing, in cells. Larger values give fewer, broader colors.
        minShare: Peaks with less than this fraction of the pixels are merged into the nearest stronger color.
        minRegion: If positive, connected regions smaller than this many pixels take the label of the
            surrounding region. This removes speckles and gives cleaner segments, but also removes small details
            such as thin text.

    Returns:
        ``(maps, centers)`` as for :func:`kmeansColors`: ``maps`` holds labels ``0 .. n-1``, ordered from the most
        to the least common color, and ``centers`` the mean BGR color of each cluster.

    Raises:
        ValueError: If ``src`` is not an 8-bit BGR image, or a parameter is out of range.
    """
    if src.dtype != np.uint8 or src.ndim != 3 or src.shape[2] != 3:
        raise ValueError(f"src must be an 8-bit BGR image, got shape {src.shape} and dtype {src.dtype}")
    if not 2 <= bins <= 64:
        raise ValueError(f"bins must be between 2 and 64, got {bins}")
    if src.size == 0:
        raise ValueError("src must not be empty")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be a non-negative finite number, got {sigma}")
    if not 0.0 <= minShare < 1.0:
        raise ValueError(f"minShare must be in [0, 1), got {minShare}")
    if minRegion < 0:
        raise ValueError(f"minRegion must not be negative, got {minRegion}")

    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.int64)
    cell = lab * bins // 256
    flat = (cell[:, 0] * bins + cell[:, 1]) * bins + cell[:, 2]
    hist = np.bincount(flat, minlength=bins**3).astype(np.float64).reshape(bins, bins, bins)

    peak_of_cell = _climb(_smooth(hist, sigma))
    peak_of_pixel = peak_of_cell[flat]

    # Merge weak peaks into the nearest strong one (distance between peak cells in Lab).
    peaks, mass = np.unique(peak_of_pixel, return_counts=True)
    strong = peaks[mass >= minShare * len(flat)]
    if strong.size == 0:
        strong = peaks[[int(np.argmax(mass))]]
    coords = np.stack(np.unravel_index(peaks, (bins, bins, bins)), axis=1).astype(np.float64)
    strong_coords = np.stack(np.unravel_index(strong, (bins, bins, bins)), axis=1).astype(np.float64)
    nearest = ((coords[:, None, :] - strong_coords[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    target = np.empty(bins**3, dtype=np.int64)
    target[peaks] = nearest
    labels = target[peak_of_pixel]

    # Order labels by size: 0 is the most common color.
    sizes = np.bincount(labels, minlength=strong.size)
    rank = np.empty(strong.size, dtype=np.int64)
    rank[np.argsort(-sizes, kind="stable")] = np.arange(strong.size)
    maps = rank[labels].astype(np.int32).reshape(src.shape[:2])
    if minRegion > 0:
        maps = _absorb_small_regions(maps, minRegion)
        _, maps_flat = np.unique(maps.ravel(), return_inverse=True)  # drop labels that disappeared
        sizes = np.bincount(maps_flat)
        order = np.argsort(-sizes, kind="stable")
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = np.arange(order.size)
        maps = rank[maps_flat].astype(np.int32).reshape(src.shape[:2])

    count = int(maps.max()) + 1
    pixels = src.reshape(-1, 3).astype(np.float64)
    label_flat = maps.ravel()
    totals = np.bincount(label_flat, minlength=count)
    sums = np.stack([np.bincount(label_flat, weights=pixels[:, c], minlength=count) for c in range(3)], axis=1)
    centers = np.clip(np.rint(sums / np.maximum(totals, 1)[:, None]), 0, 255).astype(np.uint8)
    return maps, centers
