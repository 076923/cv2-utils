import cv2
import numpy as np

from .._validate import check_finite, check_image_not_empty
from .color import toGray

__all__ = ["autoCanny"]


def _gray8(src: np.ndarray) -> np.ndarray:
    """Grayscale 8-bit view of any image (float or 16-bit images are stretched to 0..255)."""
    gray = toGray(src)
    if gray.dtype == np.uint8:
        return gray
    values = gray.astype(np.float64)
    low, high = float(np.min(values)), float(np.max(values))
    scaled = (values - low) * (255.0 / (high - low)) if high > low else np.zeros_like(values)
    return np.asarray(np.rint(scaled), dtype=np.uint8)


def autoCanny(src: np.ndarray, sigma: float = 0.33, apertureSize: int = 3, L2gradient: bool = False) -> np.ndarray:
    """``cv.Canny`` with the two thresholds chosen from the image itself.

    The thresholds are ``(1 - sigma) * median`` and ``(1 + sigma) * median`` of the pixel values, a common heuristic
    that works well across lighting conditions without manual tuning.

    Args:
        src: Grayscale, BGR or BGRA image. Color images are converted to grayscale; 16-bit and float images are
            stretched to 8 bits first (``cv.Canny`` accepts only 8-bit input).
        sigma: Width of the threshold band around the median; smaller values give fewer edges.
        apertureSize: Sobel aperture, as in ``cv.Canny``.
        L2gradient: Use the more accurate L2 gradient norm, as in ``cv.Canny``.

    Returns:
        Single-channel edge map (0 or 255).

    Raises:
        ValueError: If ``src`` is empty or has an unsupported channel count, or ``sigma`` is outside ``[0, 1]``.
    """
    check_image_not_empty("src", src)
    check_finite("sigma", sigma)
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be between 0 and 1, got {sigma}")
    gray = _gray8(src)
    median = float(np.median(gray))
    lower = int(max(0.0, (1.0 - sigma) * median))
    upper = int(min(255.0, (1.0 + sigma) * median))
    edges: np.ndarray = cv2.Canny(gray, lower, upper, apertureSize=apertureSize, L2gradient=L2gradient)
    return edges
