import math
from typing import Any

import cv2
import numpy as np

__all__ = ["rotateEx"]


def rotateEx(
    src: np.ndarray,
    angle: float,
    scale: float = 1.0,
    flags: int | None = None,
    borderMode: int | None = None,
    borderValue: Any = None,
) -> np.ndarray:
    """Rotate an image around its center without cropping the corners.

    The output canvas is enlarged so the whole rotated (and scaled) image fits.
    ``flags``, ``borderMode`` and ``borderValue`` are forwarded to ``cv.warpAffine``.

    Raises:
        ValueError: If ``angle`` is not finite or ``scale`` is not a positive finite number.
    """
    if not math.isfinite(angle):
        raise ValueError(f"angle must be finite, got {angle}")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be a positive finite number, got {scale}")
    height, width = src.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    radians = math.radians(angle)
    sin = abs(math.sin(radians))
    cos = abs(math.cos(radians))
    # round(), not int(): 59.99999 from floating point error must not lose a pixel row.
    bound_width = round((height * sin + width * cos) * scale)
    bound_height = round((height * cos + width * sin) * scale)

    matrix[0, 2] += (bound_width / 2) - center[0]
    matrix[1, 2] += (bound_height / 2) - center[1]

    kwargs: dict[str, Any] = {}
    if flags is not None:
        kwargs["flags"] = flags
    if borderMode is not None:
        kwargs["borderMode"] = borderMode
    if borderValue is not None:
        kwargs["borderValue"] = borderValue

    return cv2.warpAffine(src, matrix, (bound_width, bound_height), **kwargs)
