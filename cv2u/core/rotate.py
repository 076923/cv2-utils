import math
from collections.abc import Sequence

import cv2
import numpy as np

from .._validate import check_finite, check_image_not_empty

__all__ = ["rotateEx"]


def rotateEx(
    src: np.ndarray,
    angle: float,
    scale: float = 1.0,
    flags: int = cv2.INTER_LINEAR,
    borderMode: int = cv2.BORDER_CONSTANT,
    borderValue: float | Sequence[float] = 0,
) -> np.ndarray:
    """Rotate an image around its center without cropping the corners.

    ``cv.rotate`` only rotates by multiples of 90°, and ``cv.warpAffine`` with ``cv.getRotationMatrix2D`` keeps
    the input size, so the corners are cut off. Here the output canvas is enlarged so the whole rotated (and
    scaled) image fits.

    Args:
        src: Image to rotate.
        angle: Rotation in degrees, counter-clockwise (as for ``cv.getRotationMatrix2D``).
        scale: Isotropic scale factor applied with the rotation.
        flags: Interpolation, as for ``cv.warpAffine``.
        borderMode: How to fill the corners uncovered by the rotation, as for ``cv.warpAffine``.
        borderValue: Fill color for ``cv.BORDER_CONSTANT``.

    Returns:
        The rotated image; its size is the bounding box of the rotated input.

    Raises:
        ValueError: If ``src`` is empty, ``angle`` is not finite, or ``scale`` is not a positive finite number.
    """
    check_image_not_empty("src", src)
    check_finite("angle", angle)
    check_finite("scale", scale, positive=True)
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

    dst: np.ndarray = cv2.warpAffine(
        src, matrix, (bound_width, bound_height), flags=flags, borderMode=borderMode, borderValue=borderValue
    )
    return dst
