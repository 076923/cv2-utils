"""Feature matching and image alignment: the detector + matcher + ratio test + RANSAC recipe in one call."""

from typing import Literal, NamedTuple

import cv2
import numpy as np

from .._validate import check_finite, check_image_not_empty
from .color import toGray

__all__ = ["FeatureMatches", "alignImages", "matchFeatures"]

Detector = Literal["ORB", "SIFT"]


class FeatureMatches(NamedTuple):
    """The result of :func:`matchFeatures`."""

    srcPoints: np.ndarray
    """Matched keypoint locations in ``src``, shape ``(N, 2)`` float32."""
    dstPoints: np.ndarray
    """The corresponding locations in ``reference``, shape ``(N, 2)`` float32 (``dst`` as in ``cv.findHomography``)."""
    srcKeypoints: list[cv2.KeyPoint]
    """All keypoints found in ``src`` (for ``cv.drawMatches``)."""
    dstKeypoints: list[cv2.KeyPoint]
    """All keypoints found in ``reference``."""
    matches: list[cv2.DMatch]
    """The matches that passed the ratio test (for ``cv.drawMatches``), one per row of the point arrays."""


def _detector(method: str, maxFeatures: int) -> cv2.Feature2D:
    if method == "ORB":
        return cv2.ORB.create(nfeatures=maxFeatures)
    if method == "SIFT":
        return cv2.SIFT.create(nfeatures=maxFeatures)
    raise ValueError(f"method must be 'ORB' or 'SIFT', got {method!r}")


def matchFeatures(
    src: np.ndarray,
    reference: np.ndarray,
    method: Detector = "ORB",
    maxFeatures: int = 2000,
    ratio: float = 0.75,
    mask: np.ndarray | None = None,
    referenceMask: np.ndarray | None = None,
) -> FeatureMatches:
    """Find corresponding points between two images (detect, describe, match, Lowe's ratio test).

    The usual twelve lines before ``cv.findHomography``: create the detector, compute both descriptor sets,
    pick the right matcher norm (Hamming for binary descriptors, L2 for SIFT), ``knnMatch`` with k=2, keep the
    matches whose best distance is clearly below the second best, and unpack the keypoints to point arrays.

    Args:
        src: First image (any channel count; converted to grayscale).
        reference: Second image, the one ``src`` is matched to (as in :func:`alignImages`).
        method: ``"ORB"`` (default, fast, binary descriptors) or ``"SIFT"`` (slower, more robust to scale and
            lighting). Both ship with every OpenCV distribution from 4.4 on.
        maxFeatures: Keypoint limit for ORB and SIFT.
        ratio: Lowe's ratio; lower keeps fewer but more reliable matches.
        mask: Optional 8-bit mask of where to look for keypoints in ``src``.
        referenceMask: The same for ``reference``.

    Returns:
        A :class:`FeatureMatches`. ``srcPoints`` and ``dstPoints`` are empty ``(0, 2)`` arrays if nothing
        matches; pass them to ``cv.findHomography`` or ``cv.estimateAffinePartial2D`` for the transform.

    Raises:
        ValueError: If an image is empty or has an unsupported channel count, ``method`` is unknown, ``ratio`` is
            not in ``(0, 1]``, or ``maxFeatures`` is not positive.
    """
    check_image_not_empty("src", src)
    check_image_not_empty("reference", reference)
    check_finite("ratio", ratio, positive=True)
    if ratio > 1:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if maxFeatures < 1:
        raise ValueError(f"maxFeatures must be positive, got {maxFeatures}")
    detector = _detector(method, maxFeatures)
    kp1, des1 = detector.detectAndCompute(toGray(src), mask)
    kp2, des2 = detector.detectAndCompute(toGray(reference), referenceMask)
    empty = np.zeros((0, 2), np.float32)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return FeatureMatches(empty, empty, list(kp1), list(kp2), [])
    norm = cv2.NORM_L2 if method == "SIFT" else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher.create(norm)
    good = [
        pair[0]
        for pair in matcher.knnMatch(des1, des2, k=2)
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]
    good.sort(key=lambda m: m.distance)
    if not good:
        return FeatureMatches(empty, empty, list(kp1), list(kp2), [])
    src_pts = np.array([kp1[m.queryIdx].pt for m in good], np.float32)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good], np.float32)
    return FeatureMatches(src_pts, dst_pts, list(kp1), list(kp2), good)


def alignImages(
    src: np.ndarray,
    reference: np.ndarray,
    method: Detector = "ORB",
    motion: Literal["homography", "affine", "similarity"] = "homography",
    maxFeatures: int = 2000,
    ratio: float = 0.75,
    ransacThreshold: float = 5.0,
    minMatches: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp ``src`` onto ``reference`` so that the same content lands on the same pixels.

    For registering a scanned form to its template, stacking photos, or comparing before/after shots taken
    from slightly different positions. Combines :func:`matchFeatures`, ``cv.findHomography`` (or
    ``cv.estimateAffine2D`` / ``cv.estimateAffinePartial2D``) with RANSAC and the warp.

    Args:
        src: Image to align.
        reference: Image to align to; the output has its size.
        method: Feature detector, as for :func:`matchFeatures`.
        motion: ``"homography"`` (perspective, 8 degrees of freedom), ``"affine"`` (6) or ``"similarity"``
            (rotation, scale and translation only; the most stable when the images are nearly aligned already).
        maxFeatures: Keypoint limit.
        ratio: Lowe's ratio for the matches.
        ransacThreshold: Reprojection error in pixels above which a match counts as an outlier.
        minMatches: Fail when fewer matches pass the ratio test.

    Returns:
        ``(aligned, M)``: the warped image (same dtype and channels as ``src``, size of ``reference``) and the
        transform, ``3x3`` for a homography or ``2x3`` otherwise, mapping ``src`` coordinates to ``reference``.

    Raises:
        ValueError: If an image is empty, ``ransacThreshold`` is not positive, ``minMatches`` is too small for
            ``motion``, too few matches are found, or the transform cannot be estimated.
    """
    check_image_not_empty("src", src)
    check_image_not_empty("reference", reference)
    check_finite("ransacThreshold", ransacThreshold, positive=True)
    if minMatches < 4 if motion == "homography" else minMatches < 3:
        raise ValueError(f"minMatches must be at least {4 if motion == 'homography' else 3}, got {minMatches}")
    found = matchFeatures(src, reference, method, maxFeatures, ratio)
    if len(found.matches) < minMatches:
        raise ValueError(f"only {len(found.matches)} matches found, need at least {minMatches}; try method='SIFT'")
    height, width = reference.shape[:2]
    if motion == "homography":
        matrix, _ = cv2.findHomography(found.srcPoints, found.dstPoints, cv2.RANSAC, ransacThreshold)
        if matrix is None:
            raise ValueError("the homography could not be estimated (matches are degenerate)")
        aligned = cv2.warpPerspective(src, matrix, (width, height))
    else:
        estimate = cv2.estimateAffine2D if motion == "affine" else cv2.estimateAffinePartial2D
        matrix, _ = estimate(found.srcPoints, found.dstPoints, method=cv2.RANSAC, ransacReprojThreshold=ransacThreshold)
        if matrix is None:
            raise ValueError("the affine transform could not be estimated (matches are degenerate)")
        aligned = cv2.warpAffine(src, matrix, (width, height))
    return aligned, matrix
