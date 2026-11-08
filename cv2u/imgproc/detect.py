"""Classic detectors with usable output: line segments, circles and blobs as plain arrays."""

import cv2
import numpy as np

from .._validate import check_finite, check_image_not_empty
from .edges import _gray8, autoCanny

__all__ = ["findBlobs", "findCircles", "findLines"]


def findLines(
    src: np.ndarray,
    minLength: float = 30,
    maxGap: float = 10,
    threshold: int = 50,
    rho: float = 1.0,
    theta: float = np.pi / 180,
    edges: np.ndarray | None = None,
) -> np.ndarray:
    """Line segments in an image, as an ``(N, 4)`` array of ``(x1, y1, x2, y2)``, longest first.

    ``cv.HoughLinesP`` needs an edge image first and returns an ``(N, 1, 4)`` array or ``None``. Here any image
    works (edges come from :func:`autoCanny` unless ``edges`` is given) and the result is always an array.

    Args:
        src: Grayscale, BGR or BGRA image (16-bit and float images are stretched to 8 bits).
        minLength: Shortest segment to report, in pixels.
        maxGap: Largest gap bridged between collinear edge pixels.
        threshold: Accumulator votes a line needs (``cv.HoughLinesP``).
        rho: Distance resolution in pixels.
        theta: Angle resolution in radians.
        edges: A binary edge image to use instead of running ``autoCanny``.

    Returns:
        A float64 array of shape ``(N, 4)``; empty ``(0, 4)`` when nothing is found.

    Raises:
        ValueError: If ``src`` is empty or a parameter is not positive.
    """
    check_image_not_empty("src", src)
    check_finite("minLength", minLength, nonnegative=True)
    check_finite("maxGap", maxGap, nonnegative=True)
    check_finite("rho", rho, positive=True)
    check_finite("theta", theta, positive=True)
    if threshold < 1:
        raise ValueError(f"threshold must be positive, got {threshold}")
    binary = autoCanny(_gray8(src)) if edges is None else np.asarray(edges != 0, np.uint8) * 255
    found = cv2.HoughLinesP(binary, rho, theta, threshold, minLineLength=minLength, maxLineGap=maxGap)
    if found is None or len(found) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    lines = found.reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    return lines[np.argsort(-lengths, kind="stable")]


def findCircles(
    src: np.ndarray,
    minRadius: int = 0,
    maxRadius: int = 0,
    minDist: float | None = None,
    sensitivity: float = 30,
    edgeThreshold: float = 100,
    blur: int = 5,
) -> np.ndarray:
    """Circles in an image, as an ``(N, 3)`` array of ``(x, y, radius)``, strongest first.

    ``cv.HoughCircles`` needs a blurred single-channel image, has parameters named ``dp``, ``param1`` and
    ``param2``, and returns ``(1, N, 3)`` or ``None``. Here any image works, the accumulator resolution (``dp=1.5``)
    suits clean filled or outlined circles, and the result is always an array.

    Args:
        src: Grayscale, BGR or BGRA image.
        minRadius: Smallest radius to look for (0: no limit).
        maxRadius: Largest radius (0: no limit).
        minDist: Minimum distance between circle centers; by default a quarter of the smaller image side.
        sensitivity: Accumulator threshold (``param2``): lower finds more circles, including false ones.
        edgeThreshold: Upper Canny threshold (``param1``).
        blur: Median blur kernel (odd) applied first; 0 to skip.

    Returns:
        A float64 array of shape ``(N, 3)``; empty ``(0, 3)`` when nothing is found.

    Raises:
        ValueError: If ``src`` is empty, ``minRadius`` or ``maxRadius`` is negative, ``blur`` is even, or a
            threshold is not positive.
    """
    check_image_not_empty("src", src)
    if minRadius < 0:
        raise ValueError(f"minRadius must not be negative, got {minRadius}")
    if maxRadius < 0:
        raise ValueError(f"maxRadius must not be negative, got {maxRadius}")
    if blur < 0 or (blur and blur % 2 == 0):
        raise ValueError(f"blur must be 0 or an odd kernel size, got {blur}")
    check_finite("sensitivity", sensitivity, positive=True)
    check_finite("edgeThreshold", edgeThreshold, positive=True)
    gray = _gray8(src)
    if blur:
        gray = cv2.medianBlur(gray, blur)
    if minDist is None:
        minDist = max(1.0, min(gray.shape[:2]) / 4)
    check_finite("minDist", minDist, positive=True)
    found = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=minDist,
        param1=edgeThreshold,
        param2=sensitivity,
        minRadius=minRadius,
        maxRadius=maxRadius,
    )
    if found is None:
        return np.zeros((0, 3), dtype=np.float64)
    return found.reshape(-1, 3).astype(np.float64)


def findBlobs(
    src: np.ndarray,
    minArea: float = 10,
    maxArea: float | None = None,
    *,
    dark: bool = True,
    minCircularity: float | None = None,
    minConvexity: float | None = None,
    minInertia: float | None = None,
    minThreshold: int = 10,
    maxThreshold: int = 220,
) -> np.ndarray:
    """Blobs (round-ish spots) in an image, as an ``(N, 3)`` array of ``(x, y, diameter)``, largest first.

    ``cv.SimpleBlobDetector`` needs a ``Params`` object with a dozen fields, each filter enabled separately,
    and returns ``KeyPoint`` objects. Here the common knobs are keyword arguments and the rest keep OpenCV's
    defaults.

    Args:
        src: Grayscale, BGR or BGRA image.
        minArea: Smallest blob, in pixels.
        maxArea: Largest blob; ``None`` for no limit.
        dark: Look for dark blobs on a light background (default); ``False`` for light blobs.
        minCircularity: ``0`` to ``1``; ``None`` disables the filter.
        minConvexity: ``0`` to ``1``; ``None`` disables the filter.
        minInertia: ``0`` (lines) to ``1`` (circles); ``None`` disables the filter.
        minThreshold: Lowest of the thresholds the detector scans.
        maxThreshold: Highest threshold scanned.

    Returns:
        A float64 array of shape ``(N, 3)``; empty ``(0, 3)`` when nothing is found.

    Raises:
        ValueError: If ``src`` is empty or a limit is out of range.
    """
    check_image_not_empty("src", src)
    check_finite("minArea", minArea, nonnegative=True)
    if maxArea is not None:
        check_finite("maxArea", maxArea, positive=True)
        if maxArea < minArea:
            raise ValueError(f"maxArea must be at least minArea, got {maxArea} < {minArea}")
    for name, value in (("minCircularity", minCircularity), ("minConvexity", minConvexity), ("minInertia", minInertia)):
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if not 0 <= minThreshold < maxThreshold <= 255:
        raise ValueError(
            f"thresholds must satisfy 0 <= minThreshold < maxThreshold <= 255, got {minThreshold}, {maxThreshold}"
        )
    params = cv2.SimpleBlobDetector.Params()
    params.minThreshold = minThreshold
    params.maxThreshold = maxThreshold
    params.filterByColor = True
    params.blobColor = 0 if dark else 255
    params.filterByArea = True
    params.minArea = float(minArea)
    params.maxArea = float(maxArea) if maxArea is not None else float(np.prod(src.shape[:2]))
    params.filterByCircularity = minCircularity is not None
    if minCircularity is not None:
        params.minCircularity = float(minCircularity)
    params.filterByConvexity = minConvexity is not None
    if minConvexity is not None:
        params.minConvexity = float(minConvexity)
    params.filterByInertia = minInertia is not None
    if minInertia is not None:
        params.minInertiaRatio = float(minInertia)
    gray = _gray8(src)
    keypoints = cv2.SimpleBlobDetector.create(params).detect(gray)
    if not keypoints:
        return np.zeros((0, 3), dtype=np.float64)
    blobs = np.array([(k.pt[0], k.pt[1], k.size) for k in keypoints], dtype=np.float64)
    return blobs[np.argsort(-blobs[:, 2], kind="stable")]
