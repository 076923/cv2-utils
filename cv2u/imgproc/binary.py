"""Binarization and morphology without the ceremony: one call from any image to a clean 0/255 mask."""

from typing import Literal

import cv2
import numpy as np

from .._validate import check_finite, check_image_not_empty
from .color import toGray

__all__ = ["binarize", "morphology"]

Method = Literal["otsu", "adaptive", "triangle"]
Operation = Literal["erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"]

_OPERATIONS = {
    "erode": cv2.MORPH_ERODE,
    "dilate": cv2.MORPH_DILATE,
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "tophat": cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}
_SHAPES = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}


def binarize(
    src: np.ndarray,
    method: Method | float = "otsu",
    invert: bool = False,
    blockSize: int = 31,
    C: float = 10,
    blur: int = 0,
) -> np.ndarray:
    """Turn any image into a 0/255 mask: Otsu (default), adaptive, triangle, or a fixed threshold.

    ``cv.threshold`` wants a grayscale 8-bit image and returns ``(value, image)``; the adaptive variant is a
    different function with its own flags. Here the input can be color, and the common choices are one argument.

    Args:
        src: Grayscale, BGR or BGRA image (8-bit, or any dtype for a fixed threshold).
        method: ``"otsu"`` (automatic global threshold), ``"adaptive"`` (local mean, for uneven lighting such as
            photographed documents), ``"triangle"`` (for images with one dominant background peak), or a number
            to use as the fixed threshold.
        invert: Foreground is dark instead of light (``cv.THRESH_BINARY_INV``), e.g. black text on white paper.
        blockSize: Neighborhood size for ``"adaptive"``, odd and at least 3.
        C: Constant subtracted from the local mean for ``"adaptive"``.
        blur: Gaussian blur kernel (odd) applied before thresholding; 0 to skip. A little blur (3 or 5) removes
            speckles from Otsu and adaptive results.

    Returns:
        ``uint8`` mask with values 0 and 255, of shape ``src.shape[:2]``.

    Raises:
        ValueError: If ``src`` is empty, ``method`` is unknown, or ``blockSize``/``blur`` is not odd.
    """
    check_image_not_empty("src", src)
    if blur < 0 or (blur and blur % 2 == 0):
        raise ValueError(f"blur must be 0 or an odd kernel size, got {blur}")
    gray = toGray(src)
    if blur:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if isinstance(method, (int, float)):
        check_finite("method", method)
        _, out = cv2.threshold(gray, float(method), 255, mode)
        return np.asarray(out, dtype=np.uint8) if out.dtype == np.uint8 else np.asarray(out > 0, np.uint8) * 255
    if gray.dtype != np.uint8:
        raise ValueError(f"'{method}' needs an 8-bit image, got dtype {src.dtype}; pass a numeric threshold instead")
    if method == "otsu":
        _, out = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
    elif method == "triangle":
        _, out = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_TRIANGLE)
    elif method == "adaptive":
        if blockSize < 3 or blockSize % 2 == 0:
            raise ValueError(f"blockSize must be odd and at least 3, got {blockSize}")
        check_finite("C", C)
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, blockSize, C)
    else:
        raise ValueError(f"method must be 'otsu', 'adaptive', 'triangle' or a number, got {method!r}")
    return np.asarray(out, dtype=np.uint8)


def morphology(
    src: np.ndarray,
    op: Operation,
    ksize: int | tuple[int, int] = 3,
    shape: Literal["ellipse", "rect", "cross"] = "ellipse",
    iterations: int = 1,
) -> np.ndarray:
    """``cv.morphologyEx`` with the kernel built for you: ``morphology(mask, "open", 5)``.

    OpenCV needs ``cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))`` and the ``cv.MORPH_*`` constant for
    every call. Here the operation is a word and the kernel a size.

    Args:
        src: Mask or image (any dtype and channel count that ``cv.morphologyEx`` accepts).
        op: ``"erode"``, ``"dilate"``, ``"open"`` (remove specks), ``"close"`` (fill small gaps), ``"gradient"``
            (outline), ``"tophat"`` or ``"blackhat"``.
        ksize: Kernel size, one number or ``(width, height)``.
        shape: Kernel shape: ``"ellipse"`` (default, direction-neutral), ``"rect"`` or ``"cross"``.
        iterations: How many times to apply it.

    Raises:
        ValueError: If ``op`` or ``shape`` is unknown, ``ksize`` is not positive, or ``src`` is empty.
    """
    check_image_not_empty("src", src)
    if op not in _OPERATIONS:
        raise ValueError(f"op must be one of {', '.join(_OPERATIONS)}, got {op!r}")
    if shape not in _SHAPES:
        raise ValueError(f"shape must be 'ellipse', 'rect' or 'cross', got {shape!r}")
    width, height = (ksize, ksize) if isinstance(ksize, int) else ksize
    if width < 1 or height < 1:
        raise ValueError(f"ksize must be positive, got {ksize}")
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")
    kernel = cv2.getStructuringElement(_SHAPES[shape], (width, height))
    dst: np.ndarray = cv2.morphologyEx(src, _OPERATIONS[op], kernel, iterations=iterations)
    return dst
