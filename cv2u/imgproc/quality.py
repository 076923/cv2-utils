"""Measure and compare images: sharpness, structural similarity, differences."""

import cv2
import numpy as np

from .color import toGray

__all__ = ["diffMask", "sharpnessScore", "ssim"]


def sharpnessScore(src: np.ndarray) -> float:
    """Variance of the Laplacian, the standard measure of focus: low values mean a blurry image.

    Only comparable between images of the same scene and size. A common starting threshold for "blurry" is around
    100 for 8-bit photos, but it depends on the content.

    Args:
        src: Grayscale, BGR or BGRA image.
    """
    laplacian = cv2.Laplacian(toGray(src).astype(np.float64), cv2.CV_64F)
    return float(laplacian.var())


def _check_pair(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")


def ssim(a: np.ndarray, b: np.ndarray, dataRange: float | None = None) -> float:
    """Structural similarity (SSIM) of two images: 1 for identical images, lower for more different ones.

    ``cv.PSNR`` exists, but SSIM, which follows perceived quality much better, is only in the ``quality`` contrib
    module. This implements the standard definition (Wang et al. 2004: Gaussian window with sigma 1.5,
    ``K1 = 0.01``, ``K2 = 0.03``); color images are averaged over the channels.

    Args:
        a: First image.
        b: Second image with the same shape and dtype.
        dataRange: Value range (255 for 8-bit). By default taken from the dtype; 1.0 for float images.

    Raises:
        ValueError: If the shapes differ.
    """
    _check_pair(a, b)
    if dataRange is None:
        dataRange = float(np.iinfo(a.dtype).max) if np.issubdtype(a.dtype, np.integer) else 1.0
    c1, c2 = (0.01 * dataRange) ** 2, (0.03 * dataRange) ** 2
    x, y = a.astype(np.float64), b.astype(np.float64)

    def blur(img: np.ndarray) -> np.ndarray:
        out: np.ndarray = cv2.GaussianBlur(img, (11, 11), 1.5)
        return out

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x * mu_x
    var_y = blur(y * y) - mu_y * mu_y
    cov = blur(x * y) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim_map.mean())


def diffMask(a: np.ndarray, b: np.ndarray, threshold: float = 25, minArea: int = 0) -> np.ndarray:
    """Mask of the pixels that differ between two images, e.g. to spot changes between two frames or versions.

    Args:
        a: First image.
        b: Second image with the same shape.
        threshold: A pixel differs when any channel differs by more than this.
        minArea: Remove differing regions smaller than this many pixels (noise, compression artifacts).

    Returns:
        ``uint8`` mask with values 0 and 255, of shape ``a.shape[:2]``.

    Raises:
        ValueError: If the shapes differ.
    """
    _check_pair(a, b)
    difference = cv2.absdiff(a, b)
    if difference.ndim == 3:
        difference = difference.max(axis=2)
    mask = np.asarray(difference > threshold, dtype=np.uint8) * 255
    if minArea > 0:
        from .mask import removeSmallObjects

        mask = removeSmallObjects(mask, minArea)
    return mask
