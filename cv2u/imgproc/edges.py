import cv2
import numpy as np

__all__ = ["autoCanny"]

_TO_GRAY = {3: cv2.COLOR_BGR2GRAY, 4: cv2.COLOR_BGRA2GRAY}


def autoCanny(src: np.ndarray, sigma: float = 0.33, apertureSize: int = 3, L2gradient: bool = False) -> np.ndarray:
    """``cv.Canny`` with the two thresholds chosen from the image itself.

    The thresholds are ``(1 - sigma) * median`` and ``(1 + sigma) * median`` of the pixel values, a common heuristic
    that works well across lighting conditions without manual tuning.

    Args:
        src: 8-bit grayscale, BGR or BGRA image. Color images are converted to grayscale.
        sigma: Width of the threshold band around the median; smaller values give fewer edges.
        apertureSize: Sobel aperture, as in ``cv.Canny``.
        L2gradient: Use the more accurate L2 gradient norm, as in ``cv.Canny``.

    Returns:
        Single-channel edge map (0 or 255).

    Raises:
        ValueError: If ``src`` is not 8-bit or ``sigma`` is outside ``[0, 1]``.
    """
    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be between 0 and 1, got {sigma}")
    gray = src if src.ndim == 2 else cv2.cvtColor(src, _TO_GRAY[src.shape[2]])
    median = float(np.median(gray))
    lower = int(max(0.0, (1.0 - sigma) * median))
    upper = int(min(255.0, (1.0 + sigma) * median))
    edges: np.ndarray = cv2.Canny(gray, lower, upper, apertureSize=apertureSize, L2gradient=L2gradient)
    return edges
