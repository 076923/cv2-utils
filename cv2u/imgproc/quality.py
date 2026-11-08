"""Measure and compare images: sharpness, structural similarity, differences."""

import cv2
import numpy as np

from .._validate import check_finite, check_image_not_empty
from .color import toGray

__all__ = ["diffMask", "hashDistance", "imageHash", "sharpnessScore", "ssim"]


def sharpnessScore(src: np.ndarray) -> float:
    """Variance of the Laplacian, the standard measure of focus: low values mean a blurry image.

    Only comparable between images of the same scene and size. A common starting threshold for "blurry" is around
    100 for 8-bit photos, but it depends on the content.

    Args:
        src: Grayscale, BGR or BGRA image.

    Raises:
        ValueError: If ``src`` is empty or has an unsupported channel count.
    """
    laplacian = cv2.Laplacian(toGray(src).astype(np.float64), cv2.CV_64F)
    return float(laplacian.var())


def _check_pair(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")
    check_image_not_empty("a", a)


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
        ValueError: If the shapes differ, the images are empty, or ``dataRange`` is not positive.
    """
    _check_pair(a, b)
    if dataRange is not None:
        check_finite("dataRange", dataRange, positive=True)
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


def diffMask(a: np.ndarray, b: np.ndarray, threshold: float = 25, minArea: float = 0) -> np.ndarray:
    """Mask of the pixels that differ between two images, e.g. to spot changes between two frames or versions.

    Args:
        a: First image.
        b: Second image with the same shape.
        threshold: A pixel differs when any channel differs by more than this.
        minArea: Remove differing regions smaller than this many pixels (noise, compression artifacts).

    Returns:
        ``uint8`` mask with values 0 and 255, of shape ``a.shape[:2]``.

    Raises:
        ValueError: If the shapes differ, the images are empty, or ``threshold`` or ``minArea`` is negative.
    """
    _check_pair(a, b)
    check_finite("threshold", threshold, nonnegative=True)
    check_finite("minArea", minArea, nonnegative=True)
    difference = cv2.absdiff(a, b)
    if difference.ndim == 3:
        difference = difference.max(axis=2)
    mask = np.asarray(difference > threshold, dtype=np.uint8) * 255
    if minArea > 0:
        from .mask import removeSmallObjects

        mask = removeSmallObjects(mask, minArea)
    return mask


def imageHash(src: np.ndarray, hashSize: int = 8, method: str = "difference") -> int:
    """Perceptual hash of an image, for finding duplicates and near-duplicates.

    Similar images (resized, recompressed, slightly edited) get hashes that differ in only a few bits; compare them
    with ``hashDistance``. OpenCV's ``img_hash`` module is in contrib, which ``opencv-python`` does not include.

    Args:
        src: Image to hash (grayscale, BGR or BGRA).
        hashSize: The hash has ``hashSize * hashSize`` bits (64 by default).
        method: ``"difference"`` (dHash: brightness gradients, the default), ``"average"`` (aHash: brighter
            than the mean) or ``"perceptual"`` (pHash: low DCT frequencies, the most robust to edits).

    Returns:
        The hash as a non-negative integer (e.g. ``format(h, "016x")`` for a 64-bit hex string).

    Raises:
        ValueError: If ``src`` is empty or has an unsupported channel count, ``hashSize`` is below 2, or ``method``
            is unknown.
    """
    if hashSize < 2:
        raise ValueError(f"hashSize must be at least 2, got {hashSize}")
    gray = toGray(src).astype(np.float32)
    if method == "difference":
        small = cv2.resize(gray, (hashSize + 1, hashSize), interpolation=cv2.INTER_AREA)
        bits = small[:, 1:] > small[:, :-1]
    elif method == "average":
        small = cv2.resize(gray, (hashSize, hashSize), interpolation=cv2.INTER_AREA)
        bits = small > small.mean()
    elif method == "perceptual":
        small = cv2.resize(gray, (4 * hashSize, 4 * hashSize), interpolation=cv2.INTER_AREA)
        low = cv2.dct(small)[:hashSize, :hashSize]
        bits = low > np.median(low.ravel()[1:])  # the DC term is the mean brightness, not structure
    else:
        raise ValueError(f'method must be "difference", "average" or "perceptual", got {method!r}')
    return int("".join("1" if b else "0" for b in bits.ravel()), 2)


def hashDistance(a: int, b: int) -> int:
    """Number of differing bits between two ``imageHash`` values (Hamming distance).

    For 64-bit hashes, 0 means identical or nearly so, up to about 10 usually means the same picture, and
    unrelated images are around 32.

    Raises:
        ValueError: If a hash is negative.
    """
    if a < 0 or b < 0:
        raise ValueError(f"hashes must be non-negative, got {a} and {b}")
    return (a ^ b).bit_count()
