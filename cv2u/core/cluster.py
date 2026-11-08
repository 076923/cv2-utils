"""Clustering of pixel colors and of points.

``kmeansColors``, ``dominantColors`` and ``colorQuantize`` use OpenCV's ``cv.kmeans`` and need nothing else.
``dbscanColors``, ``meanShiftColors`` and ``clusterPoints`` need the ``cluster`` extra (scikit-learn).
"""

from typing import Any, NamedTuple

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .._validate import check_finite, check_image_not_empty

__all__ = [
    "Clusters",
    "clusterPoints",
    "colorQuantize",
    "dbscanColors",
    "dominantColors",
    "drawClusters",
    "kmeansColors",
    "meanShiftColors",
]


class Clusters(NamedTuple):
    """The result of :func:`kmeansColors`, :func:`meanShiftColors`, :func:`peakColors` and :func:`dbscanColors`."""

    maps: np.ndarray
    """Label image of shape ``src.shape[:2]`` (``int32``): the cluster of every pixel, ``0 .. len(centers) - 1``
    (``-1`` for the noise pixels of :func:`dbscanColors`)."""
    centers: np.ndarray
    """The color of every cluster as ``uint8``, shape ``(k, channels)`` (BGR, or gray for grayscale input)."""


_INSTALL_HINT = '{name} requires scikit-learn. Install it with: pip install "python-opencv-utils[cluster]"'


def _sklearn_cluster(name: str) -> Any:
    try:
        import sklearn.cluster
    except ImportError as e:
        raise ImportError(_INSTALL_HINT.format(name=name)) from e
    return sklearn.cluster


def _check_bgr(src: np.ndarray, name: str = "src") -> None:
    if not isinstance(src, np.ndarray) or src.ndim != 3 or src.shape[2] != 3:
        shape = getattr(src, "shape", None)
        raise ValueError(f"{name} must be a 3-channel BGR image with shape (h, w, 3), got shape {shape}")


def _check_gray_or_bgr(src: np.ndarray) -> int:
    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    channels = 1 if src.ndim == 2 else src.shape[2] if src.ndim == 3 else 0
    if channels not in (1, 3):
        raise ValueError(f"src must be a grayscale or BGR image, got shape {src.shape}")
    check_image_not_empty("src", src)
    return int(channels)


def _mean_colors(src: np.ndarray, maps: np.ndarray, count: int) -> np.ndarray:
    """Truncated mean color of every label ``0 .. count - 1`` (like ``int(cv.mean())``); negative labels are skipped."""
    labels = maps.ravel().astype(np.intp)
    pixels = src.reshape(len(labels), -1).astype(np.float64)
    valid = labels >= 0
    labels, pixels = labels[valid], pixels[valid]
    sizes = np.bincount(labels, minlength=count)
    sums = np.stack([np.bincount(labels, weights=pixels[:, c], minlength=count) for c in range(pixels.shape[1])], 1)
    return np.floor(sums / np.maximum(sizes, 1)[:, None]).astype(np.uint8)


# --- DBSCAN / clusterPoints (scikit-learn) ---------------------------------------------


def dbscanColors(src: np.ndarray, eps: float, minSamples: int = 4) -> Clusters:
    """Cluster the pixel colors of a BGR image with DBSCAN (in the Lab color space), noise included.

    Requires the ``cluster`` extra: ``pip install "python-opencv-utils[cluster]"``. Unlike :func:`kmeansColors`,
    the number of colors is found by DBSCAN itself, and pixels that belong to no dense color group are reported
    as noise instead of being forced into a cluster.

    Every pixel is a sample, so time and memory grow quickly with the image size. Downscale large images first
    (e.g. ``cv.resizeEx(src, width=200)``) and upscale ``maps`` with ``cv.INTER_NEAREST`` if needed.

    Args:
        src: 8-bit BGR image.
        eps: Maximum Lab color distance between two pixels in the same neighborhood.
        minSamples: Minimum number of pixels in a neighborhood for a core point.

    Returns:
        :class:`Clusters` ``(maps, centers)`` as for :func:`kmeansColors`: ``maps`` holds the labels
        ``0 .. len(centers) - 1`` and ``-1`` for noise pixels; ``centers`` holds the mean BGR color of every
        cluster. Paint the result with :func:`drawClusters`.

    Raises:
        ImportError: If scikit-learn is not installed.
        ValueError: If ``src`` is not an 8-bit BGR image or is empty, ``eps`` is not positive, or ``minSamples``
            is below 1.
    """
    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    _check_bgr(src)
    check_image_not_empty("src", src)
    check_finite("eps", eps, positive=True)
    if minSamples < 1:
        raise ValueError(f"minSamples must be at least 1, got {minSamples}")
    cluster = _sklearn_cluster("dbscanColors")
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    labels: np.ndarray = cluster.DBSCAN(eps=eps, min_samples=minSamples).fit(lab).labels_
    maps = labels.astype(np.int32).reshape(src.shape[:2])
    count = int(np.max(maps)) + 1
    return Clusters(maps, _mean_colors(src, maps, count).reshape(count, 3))


def clusterPoints(points: ArrayLike, eps: float, minSamples: int = 2) -> np.ndarray:
    """Group nearby points with DBSCAN, e.g. keypoints, blob centers or detected text boxes.

    Requires the ``cluster`` extra.

    Args:
        points: Points as an array of shape ``(N, 2)`` (or ``(N, d)``), or an OpenCV contour of shape ``(N, 1, 2)``.
        eps: Maximum distance between two points of the same group.
        minSamples: Minimum number of points that forms a group; ``1`` puts every point in some group.

    Returns:
        Labels of shape ``(N,)`` (``int32``): ``0, 1, ...`` for the groups and ``-1`` for noise (points in no
        group), as in :func:`dbscanColors` and scikit-learn.

    Raises:
        ImportError: If scikit-learn is not installed.
        ValueError: If ``points`` is empty or not finite, ``eps`` is not positive, or ``minSamples`` is below 1.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts.reshape(-1, pts.shape[2])
    if pts.ndim != 2 or len(pts) == 0:
        raise ValueError(f"points must have the shape (N, d) with N > 0, got {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must be finite")
    check_finite("eps", eps, positive=True)
    if minSamples < 1:
        raise ValueError(f"minSamples must be at least 1, got {minSamples}")
    cluster = _sklearn_cluster("clusterPoints")
    labels: np.ndarray = cluster.DBSCAN(eps=eps, min_samples=minSamples).fit(pts).labels_
    return labels.astype(np.int32)


def drawClusters(src: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """Paint every cluster of ``maps`` with its mean color in ``src``, as a new image.

    For looking at the result of :func:`dbscanColors`, :func:`meanShiftColors`, :func:`peakColors` or
    :func:`kmeansColors`. Noise pixels (label ``-1``) keep their original color. To posterize an image with
    k-means in one call, use :func:`colorQuantize` instead.

    Args:
        src: 8-bit grayscale or BGR image, the one that was clustered.
        maps: Label image of shape ``src.shape[:2]``, as returned in :class:`Clusters`.

    Returns:
        A new image with the shape and dtype of ``src``; ``src`` is not modified.

    Raises:
        ValueError: If ``src`` is not an 8-bit grayscale or BGR image or is empty, or ``maps`` has another shape.
    """
    _check_gray_or_bgr(src)
    if maps.shape != src.shape[:2]:
        raise ValueError(f"maps must have the shape {src.shape[:2]}, got {maps.shape}")
    labels = maps.ravel().astype(np.intp)
    count = int(np.max(labels)) + 1
    out = src.reshape(len(labels), -1).copy()
    if count > 0:
        colors = _mean_colors(src, maps, count)
        painted = labels >= 0
        out[painted] = colors[labels[painted]]
    return out.reshape(src.shape)


# --- k-means (OpenCV only) -----------------------------------------------------------------


def _nearest(data: np.ndarray, centers: np.ndarray, chunk: int = 1 << 16) -> np.ndarray:
    """Index of the nearest center for every row, in chunks so memory stays bounded for large images."""
    labels = np.empty(len(data), dtype=np.int64)
    for start in range(0, len(data), chunk):
        block = data[start : start + chunk]
        labels[start : start + chunk] = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return labels


def _kmeanspp(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = [data[rng.integers(len(data))]]
    closest = ((data - centers[0]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        total = closest.sum()
        index = rng.integers(len(data)) if total == 0 else rng.choice(len(data), p=closest / total)
        centers.append(data[index])
        closest = np.minimum(closest, ((data - data[index]) ** 2).sum(axis=1))
    return np.array(centers, dtype=np.float32)


def kmeansColors(
    src: np.ndarray,
    k: int,
    sampleSize: int = 20000,
    seed: int = 0,
    maxIter: int = 30,
) -> Clusters:
    """Cluster the pixel colors of an image into ``k`` groups with k-means (``cv.kmeans``).

    Takes care of the ``cv.kmeans`` boilerplate (reshaping, float conversion, criteria), clusters in the Lab color
    space so that groups match perceived colors, and is **deterministic**: the result depends only on ``seed``,
    without changing OpenCV's global random state. Large images are fitted on a random sample of pixels, then every
    pixel is assigned to the nearest center.

    Args:
        src: 8-bit grayscale or BGR image.
        k: Number of colors.
        sampleSize: Number of pixels used to fit the centers.
        seed: Seed for the initial centers and the sampling.
        maxIter: Maximum number of k-means iterations.

    Returns:
        :class:`Clusters` ``(maps, centers)``: ``maps`` has shape ``src.shape[:2]`` with labels ``0 .. k-1``;
        ``centers`` has shape ``(k, channels)`` and holds the colors (BGR, or gray) as ``uint8``.

    Raises:
        ValueError: If ``src`` is not an 8-bit grayscale or BGR image or is empty, ``k`` is not between 1 and the
            pixel count, ``sampleSize`` is below ``k``, or ``maxIter`` is below 1.
    """
    channels = _check_gray_or_bgr(src)
    pixel_count = src.shape[0] * src.shape[1]
    if not 1 <= k <= pixel_count:
        raise ValueError(f"k must be between 1 and the number of pixels ({pixel_count}), got {k}")
    if sampleSize < k:
        raise ValueError(f"sampleSize must be at least k ({k}), got {sampleSize}")
    if maxIter < 1:
        raise ValueError(f"maxIter must be at least 1, got {maxIter}")

    space = src if channels == 1 else cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    data = space.reshape(-1, channels).astype(np.float32)
    rng = np.random.default_rng(seed)
    sample = data if len(data) <= sampleSize else data[rng.choice(len(data), size=sampleSize, replace=False)]

    initial = _kmeanspp(sample, k, rng)
    labels = _nearest(sample, initial).astype(np.int32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, maxIter, 0.1)
    _, _, centers = cv2.kmeans(sample, k, labels, criteria, 1, cv2.KMEANS_USE_INITIAL_LABELS)

    maps = _nearest(data, centers).astype(np.int32).reshape(src.shape[:2])
    colors = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
    if channels == 3:
        colors = np.asarray(cv2.cvtColor(colors.reshape(-1, 1, 3), cv2.COLOR_LAB2BGR), dtype=np.uint8).reshape(-1, 3)
    return Clusters(maps, colors)


def dominantColors(
    src: np.ndarray, k: int = 5, sampleSize: int = 20000, seed: int = 0
) -> list[tuple[tuple[int, ...], float]]:
    """Find the ``k`` most common colors of an image (a color palette), most common first.

    Args:
        src: 8-bit grayscale or BGR image.
        k: Number of colors to find.
        sampleSize: Number of pixels used to fit the colors (see :func:`kmeansColors`).
        seed: Seed, for reproducible results.

    Returns:
        A list of ``(color, ratio)`` pairs sorted by ``ratio``, where ``color`` is a BGR tuple (or a 1-tuple for
        grayscale) and ``ratio`` is the share of pixels, summing to 1. Colors that no pixel ends up with are omitted.

    Raises:
        ValueError: As for :func:`kmeansColors`.
    """
    maps, centers = kmeansColors(src, k, sampleSize=sampleSize, seed=seed)
    counts = np.bincount(maps.ravel(), minlength=k)
    order = np.argsort(-counts, kind="stable")
    total = float(counts.sum())
    return [(tuple(int(c) for c in centers[i]), counts[i] / total) for i in order if counts[i] > 0]


def colorQuantize(src: np.ndarray, k: int, sampleSize: int = 20000, seed: int = 0) -> np.ndarray:
    """Reduce an image to ``k`` colors (posterize), using the colors found by :func:`kmeansColors`.

    Raises:
        ValueError: As for :func:`kmeansColors`.
    """
    maps, centers = kmeansColors(src, k, sampleSize=sampleSize, seed=seed)
    dst: np.ndarray = centers[maps].reshape(src.shape)
    return dst


def meanShiftColors(
    src: np.ndarray,
    bandwidth: float | None = None,
    quantile: float = 0.2,
    sampleSize: int = 2000,
    seed: int = 0,
) -> Clusters:
    """Cluster the pixel colors with mean shift, which finds the number of colors by itself.

    The counterpart of :func:`kmeansColors` when you do not know ``k``: same result format, clustering in the Lab
    color space, deterministic for a given ``seed``. The centers are fitted on a sample of pixels, so large images are
    fast. Requires the ``cluster`` extra.

    Args:
        src: 8-bit BGR image.
        bandwidth: Kernel size in Lab units; larger values merge more colors. By default it is estimated from the
            image with ``sklearn.cluster.estimate_bandwidth``.
        quantile: Used to estimate ``bandwidth`` when it is not given; larger values give fewer colors.
        sampleSize: Number of pixels used to fit the centers.
        seed: Seed for the sampling.

    Returns:
        :class:`Clusters` ``(maps, centers)`` as for :func:`kmeansColors`; ``len(centers)`` is the number of
        colors found.

    Raises:
        ImportError: If scikit-learn is not installed.
        ValueError: If ``src`` is not an 8-bit BGR image or is empty, ``bandwidth`` is not positive, or
            ``quantile`` is outside ``(0, 1]``.
    """
    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    _check_bgr(src)
    check_image_not_empty("src", src)
    if bandwidth is not None:
        check_finite("bandwidth", bandwidth, positive=True)
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    cluster = _sklearn_cluster("meanShiftColors")

    data = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    sample = data if len(data) <= sampleSize else data[rng.choice(len(data), size=sampleSize, replace=False)]
    if bandwidth is None:
        bandwidth = float(cluster.estimate_bandwidth(sample, quantile=quantile, random_state=seed))
    if bandwidth <= 0:  # e.g. a single flat color: every sample is identical
        bandwidth = 1.0
    model = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(sample)
    centers = model.cluster_centers_.astype(np.float32)

    maps = _nearest(data, centers).astype(np.int32).reshape(src.shape[:2])
    lab = np.clip(np.rint(centers), 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    colors = np.asarray(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), dtype=np.uint8).reshape(-1, 3)
    return Clusters(maps, colors)
