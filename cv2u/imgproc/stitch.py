"""Image stitching: ``cv.Stitcher`` with exceptions instead of status codes, and a two-image homography stitch."""

from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np

from .._validate import check_finite, check_image_not_empty
from .color import toBGR
from .features import Detector, matchFeatures

__all__ = ["stitchImages", "stitchPair"]

_STATUS_NAMES = {
    cv2.STITCHER_ERR_NEED_MORE_IMGS: "ERR_NEED_MORE_IMGS",
    cv2.STITCHER_ERR_HOMOGRAPHY_EST_FAIL: "ERR_HOMOGRAPHY_EST_FAIL",
    cv2.STITCHER_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
}


def stitchImages(
    images: Sequence[np.ndarray], mode: Literal["panorama", "scans"] = "panorama", confidence: float | None = None
) -> np.ndarray:
    """Stitch overlapping images into one with ``cv.Stitcher``, raising on failure instead of returning a code.

    ``cv.Stitcher.stitch`` returns ``(status, pano)`` where ``pano`` is ``None`` on failure and ``status`` is a
    bare integer. This raises a ``RuntimeError`` naming the status (``ERR_NEED_MORE_IMGS``, ...) and accepts
    grayscale images (the stitcher needs BGR).

    Args:
        images: Two or more overlapping images, grayscale or BGR.
        mode: ``"panorama"`` for photos taken by rotating the camera (perspective warps), ``"scans"`` for
            flat content such as scans or aerial images (affine warps).
        confidence: ``setPanoConfidenceThresh``; lower (e.g. ``0.3``) accepts images with less overlap or
            texture. ``None`` keeps OpenCV's default (``1.0``).

    Returns:
        The stitched BGR image.

    Raises:
        ValueError: If fewer than 2 images are given, an image is empty, or ``mode`` is unknown.
        RuntimeError: If the stitcher fails; the message contains the status name.
    """
    if len(images) < 2:
        raise ValueError(f"images must contain at least 2 images, got {len(images)}")
    for index, image in enumerate(images):
        check_image_not_empty(f"images[{index}]", image)
    if mode == "panorama":
        stitcher = cv2.Stitcher.create(cv2.STITCHER_PANORAMA)
    elif mode == "scans":
        stitcher = cv2.Stitcher.create(cv2.STITCHER_SCANS)
    else:
        raise ValueError(f"mode must be 'panorama' or 'scans', got {mode!r}")
    if confidence is not None:
        check_finite("confidence", confidence, nonnegative=True)
        stitcher.setPanoConfidenceThresh(confidence)
    status, pano = stitcher.stitch([toBGR(image) for image in images])
    if status != cv2.STITCHER_OK or pano is None:
        raise RuntimeError(f"stitching failed with status {_STATUS_NAMES.get(status, status)}")
    return pano


def stitchPair(
    a: np.ndarray,
    b: np.ndarray,
    method: Detector = "ORB",
    ratio: float = 0.75,
    maxFeatures: int = 2000,
    ransacThreshold: float = 5.0,
    minMatches: int = 10,
) -> np.ndarray:
    """Stitch two overlapping images by a homography, on a canvas large enough to hold both.

    The simple recipe: :func:`matchFeatures`, ``cv.findHomography`` with RANSAC, and ``cv.warpPerspective``
    of ``b`` onto the plane of ``a``. Unlike the bare warp, the canvas is enlarged (and the origin shifted) so
    that nothing is cut off; ``a`` is pasted on top. No blending, no cropping: the background is black.

    Args:
        a: Reference image (grayscale or BGR); its plane is kept.
        b: Image to warp onto ``a``.
        method: Feature detector, as for :func:`matchFeatures`.
        ratio: Lowe's ratio for the matches.
        maxFeatures: Keypoint limit.
        ransacThreshold: Reprojection error in pixels above which a match counts as an outlier.
        minMatches: Fail when fewer matches pass the ratio test.

    Returns:
        The stitched image, same dtype and channel count as ``a``.

    Raises:
        ValueError: If an image is empty, too few matches are found, or the homography cannot be estimated.
    """
    check_image_not_empty("a", a)
    check_image_not_empty("b", b)
    check_finite("ransacThreshold", ransacThreshold, positive=True)
    if minMatches < 4:
        raise ValueError(f"minMatches must be at least 4, got {minMatches}")
    if a.ndim == 3 and b.ndim == 2:
        b = toBGR(b)
    elif a.ndim == 2 and b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    found = matchFeatures(b, a, method, maxFeatures, ratio)
    if len(found.matches) < minMatches:
        raise ValueError(f"only {len(found.matches)} matches found, need at least {minMatches}; try method='SIFT'")
    matrix, _ = cv2.findHomography(found.srcPoints, found.dstPoints, cv2.RANSAC, ransacThreshold)
    if matrix is None:
        raise ValueError("the homography could not be estimated (matches are degenerate)")
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    corners_b = np.array([[0, 0], [wb, 0], [wb, hb], [0, hb]], np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners_b, matrix).reshape(-1, 2)
    corners_a = np.array([[0, 0], [wa, 0], [wa, ha], [0, ha]], np.float32)
    all_corners = np.vstack([corners_a, warped])
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    shift = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], np.float64)
    canvas = cv2.warpPerspective(b, shift @ matrix, (int(x_max - x_min), int(y_max - y_min)))
    canvas[-y_min : -y_min + ha, -x_min : -x_min + wa] = a
    return canvas
