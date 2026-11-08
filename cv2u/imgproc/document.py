"""Document and OCR preprocessing: shadows, local thresholds, ruling lines, text lines, page detection and skew.

OpenCV has the pieces (``cv.integral``, ``cv.morphologyEx``, ``cv.inpaint``, ``cv.approxPolyDP``, ...) but no
function that takes a photographed or scanned page and gives back something an OCR engine likes. The local
thresholds (``cv.ximgproc.niBlackThreshold``) live in the contrib module that ``opencv-python`` does not ship.
"""

import math
from typing import Literal

import cv2
import numpy as np

from .._validate import check_finite, check_image_not_empty
from .adjust import autoContrast
from .binary import binarize
from .color import toGray
from .enhance import unsharpMask
from .geometry import _ink, _line_score, _order_corners, fourPointTransform

__all__ = [
    "detectPage",
    "detectTextBlocks",
    "detectTextLines",
    "enhanceDocument",
    "estimateSkew",
    "niblackThreshold",
    "removeBorder",
    "removeLines",
    "removeShadow",
    "removeSpeckles",
    "sauvolaThreshold",
    "scanDocument",
]

EnhanceMode = Literal["gray", "color", "binary"]


def _check_uint8(name: str, src: np.ndarray) -> None:
    check_image_not_empty(name, src)
    if src.dtype != np.uint8:
        raise ValueError(f"{name} must be an 8-bit (uint8) image, got dtype {src.dtype}")
    if src.ndim not in (2, 3):
        raise ValueError(f"{name} must be a grayscale or color image, got shape {src.shape}")


def _check_odd(name: str, value: int, minimum: int = 3) -> None:
    if value < minimum or value % 2 == 0:
        raise ValueError(f"{name} must be odd and at least {minimum}, got {value}")


def _ink_mask(src: np.ndarray) -> np.ndarray:
    """Text pixels (dark on light or light on dark) as a full-resolution 0/255 mask."""
    gray = toGray(src)
    mask = binarize(gray, "otsu", invert=True)
    if np.count_nonzero(mask) > mask.size / 2:  # light text on a dark page
        mask = cv2.bitwise_not(mask)
    return mask


def removeShadow(src: np.ndarray, kernelSize: int = 21, blurSize: int = 21) -> np.ndarray:
    """Flatten uneven lighting on a photographed page so the paper becomes uniformly white.

    The background is estimated per channel by dilating away the text and smoothing with a median blur; the
    image is then divided by it. A classic recipe that OpenCV does not package.

    Args:
        src: 8-bit grayscale, BGR or BGRA image of a page.
        kernelSize: Dilation kernel; must be larger than the text stroke width so the text vanishes.
        blurSize: Median blur kernel (odd) that smooths the background estimate.

    Returns:
        An image of the same shape and dtype with a flat, bright background.

    Raises:
        ValueError: If ``src`` is empty or not 8-bit, or ``kernelSize``/``blurSize`` is not a valid kernel size.
    """
    _check_uint8("src", src)
    if kernelSize < 1:
        raise ValueError(f"kernelSize must be positive, got {kernelSize}")
    _check_odd("blurSize", blurSize)
    kernel = np.ones((kernelSize, kernelSize), dtype=np.uint8)
    planes = [src] if src.ndim == 2 else list(cv2.split(src))
    out = []
    for plane in planes:
        background = cv2.medianBlur(cv2.dilate(plane, kernel), blurSize)
        diff = 255 - cv2.absdiff(plane, background)
        out.append(cv2.normalize(diff, diff, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    result: np.ndarray = out[0] if src.ndim == 2 else cv2.merge(out)
    return result


def _local_stats(gray: np.ndarray, windowSize: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of every ``windowSize`` neighborhood, from integral images."""
    r = windowSize // 2
    padded = cv2.copyMakeBorder(gray, r, r + 1, r, r + 1, cv2.BORDER_REFLECT_101)
    integral, sq_integral = cv2.integral2(padded, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
    h, w = gray.shape
    n = float(windowSize * windowSize)
    s = (
        integral[windowSize : windowSize + h, windowSize : windowSize + w]
        - integral[0:h, windowSize : windowSize + w]
        - integral[windowSize : windowSize + h, 0:w]
        + integral[0:h, 0:w]
    )
    sq = (
        sq_integral[windowSize : windowSize + h, windowSize : windowSize + w]
        - sq_integral[0:h, windowSize : windowSize + w]
        - sq_integral[windowSize : windowSize + h, 0:w]
        + sq_integral[0:h, 0:w]
    )
    mean = s / n
    std = np.sqrt(np.maximum(sq / n - mean * mean, 0.0))
    return mean, std


def _apply_local_threshold(gray: np.ndarray, threshold: np.ndarray, invert: bool) -> np.ndarray:
    above = gray.astype(np.float64) > threshold
    return np.asarray(above != invert, dtype=np.uint8) * 255


def sauvolaThreshold(
    src: np.ndarray, windowSize: int = 25, k: float = 0.2, r: float = 128.0, invert: bool = False
) -> np.ndarray:
    """Sauvola local threshold: the standard binarization for OCR of unevenly lit or stained documents.

    Threshold ``T = m * (1 + k * (s / r - 1))`` from the local mean ``m`` and standard deviation ``s``, computed
    with integral images. OpenCV only has it in the ``ximgproc`` contrib module (``niBlackThreshold`` with
    ``BINARIZATION_SAUVOLA``).

    Args:
        src: 8-bit grayscale, BGR or BGRA image.
        windowSize: Side of the square neighborhood (odd); 2-3 times the text height works well.
        k: Sensitivity, usually 0.2-0.5; larger values thin the text.
        r: Dynamic range of the standard deviation, 128 for 8-bit images.
        invert: Return the text (pixels below the threshold) as 255 instead of the paper, like
            ``cv.THRESH_BINARY_INV``.

    Returns:
        ``uint8`` mask with values 0 and 255: 255 where the pixel is above its local threshold (the paper).

    Raises:
        ValueError: If ``src`` is empty or not 8-bit, ``windowSize`` is not odd, or ``k``/``r`` is not finite.
    """
    _check_uint8("src", src)
    _check_odd("windowSize", windowSize)
    check_finite("k", k)
    check_finite("r", r, positive=True)
    gray = toGray(src)
    mean, std = _local_stats(gray, windowSize)
    return _apply_local_threshold(gray, mean * (1 + k * (std / r - 1)), invert)


def niblackThreshold(src: np.ndarray, windowSize: int = 25, k: float = -0.2, invert: bool = False) -> np.ndarray:
    """Niblack local threshold ``T = m + k * s`` from the local mean and standard deviation.

    Simpler than :func:`sauvolaThreshold` and noisier on blank paper, but useful when the text contrast is low.
    OpenCV only has it in the ``ximgproc`` contrib module.

    Args:
        src: 8-bit grayscale, BGR or BGRA image.
        windowSize: Side of the square neighborhood (odd).
        k: Weight of the standard deviation; negative values (-0.2) favor dark text on light paper.
        invert: Return the text (pixels below the threshold) as 255 instead of the paper.

    Returns:
        ``uint8`` mask with values 0 and 255: 255 where the pixel is above its local threshold.

    Raises:
        ValueError: If ``src`` is empty or not 8-bit, ``windowSize`` is not odd, or ``k`` is not finite.
    """
    _check_uint8("src", src)
    _check_odd("windowSize", windowSize)
    check_finite("k", k)
    gray = toGray(src)
    mean, std = _local_stats(gray, windowSize)
    return _apply_local_threshold(gray, mean + k * std, invert)


def _is_binary(src: np.ndarray) -> bool:
    return bool(np.isin(np.unique(src), (0, 255)).all())


def removeLines(
    src: np.ndarray,
    horizontal: bool = True,
    vertical: bool = True,
    minLength: float | None = None,
    thickness: int = 3,
) -> np.ndarray:
    """Erase table rules and underlines from a document so OCR sees only the text.

    Long runs of ink are found by opening with a thin horizontal / vertical structuring element. On a binary
    image the line pixels are set to the paper value; on grayscale or color images they are filled with
    ``cv.inpaint``.

    Args:
        src: 8-bit grayscale, BGR or BGRA document (binarized or not).
        horizontal: Remove horizontal lines.
        vertical: Remove vertical lines.
        minLength: Shortest line to remove, in pixels, or as a fraction of the image size when below 1.
            By default a tenth of the image width / height.
        thickness: Width of the strip erased around each line, in pixels (covers anti-aliased edges).

    Returns:
        A new image of the same shape and dtype.

    Raises:
        ValueError: If ``src`` is empty or not 8-bit, ``minLength`` is not positive, or ``thickness`` is not
            positive.
    """
    _check_uint8("src", src)
    if minLength is not None:
        check_finite("minLength", minLength, positive=True)
    if thickness < 1:
        raise ValueError(f"thickness must be positive, got {thickness}")
    h, w = src.shape[:2]

    def length(extent: int) -> int:
        if minLength is None:
            return max(int(extent * 0.1), 1)
        return max(round(minLength * extent if minLength < 1 else minLength), 1)

    ink = _ink_mask(src)
    lines = np.zeros_like(ink)
    if horizontal:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length(w), 1))
        lines |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    if vertical:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length(h)))
        lines |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    if thickness > 1:
        lines = cv2.dilate(lines, np.ones((thickness, thickness), dtype=np.uint8))
    if not np.any(lines):
        return src.copy()
    if _is_binary(src):
        out = src.copy()
        paper = np.array(255 if np.count_nonzero(src) > src.size / 2 else 0, dtype=np.uint8)
        out[lines > 0] = paper
        return out
    if src.ndim == 3 and src.shape[2] == 4:
        color: np.ndarray = cv2.inpaint(src[..., :3], lines, 3, cv2.INPAINT_TELEA)
        return np.dstack([color, src[..., 3]])
    dst: np.ndarray = cv2.inpaint(src, lines, 3, cv2.INPAINT_TELEA)
    return dst


def _text_regions(
    src: np.ndarray, ksize: tuple[int, int] | None, minHeight: int, defaults: tuple[float, float]
) -> np.ndarray:
    """Boxes of ink regions after dilating with ``ksize``, sorted top to bottom; the kernel defaults to
    ``defaults`` times the median character height."""
    _check_uint8("src", src)
    if minHeight < 1:
        raise ValueError(f"minHeight must be positive, got {minHeight}")
    ink = _ink_mask(src)
    if ksize is None:
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        heights = heights[heights >= minHeight]
        char = float(np.median(heights)) if heights.size else float(minHeight)
        ksize = (max(int(char * defaults[0]), 1), max(int(char * defaults[1]), 1))
    merged = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, ksize))
    _, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    boxes = stats[1:, :4].astype(np.int32)
    boxes = boxes[boxes[:, 3] >= minHeight]
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.int32)
    # Shrink the dilation back out of the boxes so they hug the ink.
    pad_x, pad_y = ksize[0] // 2, ksize[1] // 2
    boxes[:, 0] += pad_x
    boxes[:, 1] += pad_y
    boxes[:, 2] -= 2 * pad_x
    boxes[:, 3] -= 2 * pad_y
    boxes[:, 2:] = np.maximum(boxes[:, 2:], 1)
    order = np.lexsort((boxes[:, 0], boxes[:, 1]))
    return np.asarray(boxes[order], dtype=np.int32)


def detectTextLines(src: np.ndarray, minHeight: int = 5, dilateWidth: int | None = None) -> np.ndarray:
    """Bounding boxes of the text lines of a page, top to bottom, ready for cropping into an OCR engine.

    The text is binarized, smeared horizontally so the characters of a line join, and the connected regions are
    measured. Deskew first (:func:`cv2u.deskew`) if the lines are tilted.

    Args:
        src: 8-bit grayscale, BGR or BGRA image of a page.
        minHeight: Ignore regions lower than this many pixels (noise, rules).
        dilateWidth: Horizontal smear that joins characters and words, in pixels. By default about twice the
            median character height, which bridges word spaces but not columns.

    Returns:
        ``int32`` array of shape ``(N, 4)`` with one ``(x, y, w, h)`` box per line, sorted top to bottom (then
        left to right). Empty for a blank page.

    Raises:
        ValueError: If ``src`` is empty or not 8-bit, or ``minHeight``/``dilateWidth`` is not positive.
    """
    if dilateWidth is not None and dilateWidth < 1:
        raise ValueError(f"dilateWidth must be positive, got {dilateWidth}")
    ksize = None if dilateWidth is None else (dilateWidth, 1)
    return _text_regions(src, ksize, minHeight, (2.0, 0.0))


def detectTextBlocks(src: np.ndarray, minHeight: int = 5, dilateSize: tuple[int, int] | None = None) -> np.ndarray:
    """Bounding boxes of the text blocks (paragraphs, columns) of a page, top to bottom.

    Like :func:`detectTextLines`, but the ink is smeared both ways so that neighboring lines merge into one
    region while blocks separated by a wider gap stay apart.

    Args:
        src: 8-bit grayscale, BGR or BGRA image of a page.
        minHeight: Ignore regions lower than this many pixels.
        dilateSize: Smear as ``(width, height)`` in pixels. By default about 3 and 1.5 times the median
            character height.

    Returns:
        ``int32`` array of shape ``(N, 4)`` with one ``(x, y, w, h)`` box per block, sorted top to bottom
        (then left to right). Empty for a blank page.

    Raises:
        ValueError: If ``src`` is empty or not 8-bit, or ``minHeight``/``dilateSize`` is not positive.
    """
    if dilateSize is not None and (dilateSize[0] < 1 or dilateSize[1] < 1):
        raise ValueError(f"dilateSize must be positive, got {dilateSize}")
    return _text_regions(src, dilateSize, minHeight, (3.0, 1.5))


def removeBorder(src: np.ndarray, threshold: float | None = None, margin: int = 0) -> np.ndarray:
    """Crop away the dark frame a scanner or copier leaves around a page.

    The page is the largest bright region; the image is cropped to its bounding box. For a uniform border of
    any color see :func:`cv2u.cropToContent`.

    Args:
        src: 8-bit grayscale, BGR or BGRA image.
        threshold: Gray level above which a pixel counts as paper; by default chosen by Otsu's method.
        margin: Extra pixels trimmed from each side of the page (positive) or kept around it (negative).

    Returns:
        The cropped page. The whole image if nothing bright is found.

    Raises:
        ValueError: If ``src`` is empty or not 8-bit, or ``threshold`` is not finite.
    """
    _check_uint8("src", src)
    if threshold is not None:
        check_finite("threshold", threshold)
    gray = toGray(src)
    bright = binarize(gray, "otsu" if threshold is None else threshold)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=4)
    if count <= 1:
        return src.copy()
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[largest, :4])
    h_img, w_img = gray.shape
    x0, y0 = max(x + margin, 0), max(y + margin, 0)
    x1, y1 = min(x + w - margin, w_img), min(y + h - margin, h_img)
    if x1 <= x0 or y1 <= y0:
        return src.copy()
    return np.ascontiguousarray(src[y0:y1, x0:x1])


def detectPage(src: np.ndarray, minAreaRatio: float = 0.2) -> np.ndarray | None:
    """Find the four corners of a photographed page (or card, receipt, whiteboard) for :func:`fourPointTransform`.

    Edges are detected on a blurred grayscale copy, closed, and the largest contour that approximates to a
    quadrilateral covering at least ``minAreaRatio`` of the image is returned.

    Args:
        src: 8-bit grayscale, BGR or BGRA photo with the page against a contrasting background.
        minAreaRatio: Smallest fraction of the image area the page may cover.

    Returns:
        ``float32`` array of shape ``(4, 2)`` with the corners ordered top-left, top-right, bottom-right,
        bottom-left, or ``None`` if no quadrilateral is found.

    Raises:
        ValueError: If ``src`` is empty or not 8-bit, or ``minAreaRatio`` is not in ``[0, 1]``.
    """
    _check_uint8("src", src)
    check_finite("minAreaRatio", minAreaRatio)
    if not 0 <= minAreaRatio <= 1:
        raise ValueError(f"minAreaRatio must be in [0, 1], got {minAreaRatio}")
    gray = toGray(src)
    h, w = gray.shape
    scale = min(1.0, 800 / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    median = float(np.median(blurred))
    edges = cv2.Canny(blurred, max(0.0, 0.67 * median), min(255.0, 1.33 * median) or 50.0)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = minAreaRatio * small.shape[0] * small.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            break
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        for epsilon in (0.02, 0.05, 0.1):
            approx = cv2.approxPolyDP(hull, epsilon * cv2.arcLength(hull, True), True)
            # A quadrilateral must keep nearly all of the hull's area (a circle loses a third of it).
            if len(approx) == 4 and cv2.contourArea(approx) >= max(min_area, 0.9 * hull_area):
                corners = approx.reshape(4, 2).astype(np.float32) / np.float32(scale)
                return np.asarray(_order_corners(corners), dtype=np.float32)
    return None


def enhanceDocument(src: np.ndarray, mode: EnhanceMode = "gray") -> np.ndarray:
    """Give a photographed page the look of a flatbed scan: flat white paper and crisp dark text.

    Removes shadows (:func:`removeShadow`), stretches the contrast and sharpens; ``"binary"`` ends with an
    adaptive threshold for OCR or fax-style output.

    Args:
        src: 8-bit grayscale, BGR or BGRA image of a page.
        mode: ``"gray"`` (grayscale result), ``"color"`` (keeps the colors, e.g. stamps and highlights) or
            ``"binary"`` (0/255 mask).

    Returns:
        ``uint8`` image: 2-D for ``"gray"`` and ``"binary"``, the same channels as ``src`` for ``"color"``
        (grayscale input stays grayscale).

    Raises:
        ValueError: If ``src`` is empty or not 8-bit, or ``mode`` is unknown.
    """
    _check_uint8("src", src)
    if mode not in ("gray", "color", "binary"):
        raise ValueError(f"mode must be 'gray', 'color' or 'binary', got {mode!r}")
    image = src if mode == "color" else toGray(src)
    flat = removeShadow(image)
    if mode == "binary":
        return binarize(flat, "adaptive", blockSize=31, C=15)
    return unsharpMask(autoContrast(flat, clip=0.005), sigma=1.0, amount=0.5)


def scanDocument(
    src: np.ndarray,
    size: tuple[int, int] | None = None,
    enhance: bool = True,
    minAreaRatio: float = 0.2,
) -> np.ndarray:
    """Photo of a page in, scan out: :func:`detectPage` + :func:`fourPointTransform` + :func:`enhanceDocument`.

    Args:
        src: 8-bit grayscale, BGR or BGRA photo of a page.
        size: Output size ``(width, height)``; by default the size of the page in the photo.
        enhance: Flatten the lighting and sharpen the result (``"gray"`` for grayscale input, ``"color"``
            otherwise).
        minAreaRatio: Passed to :func:`detectPage`.

    Returns:
        The upright page. If no page outline is found, the whole image (enhanced if asked, resized to ``size``).

    Raises:
        ValueError: If ``src`` is empty or not 8-bit, or ``minAreaRatio`` is not in ``[0, 1]``.
    """
    corners = detectPage(src, minAreaRatio)
    if corners is None:
        page = src if size is None else cv2.resize(src, size, interpolation=cv2.INTER_AREA)
    else:
        page = fourPointTransform(src, corners, size)
    if enhance:
        return enhanceDocument(page, "gray" if page.ndim == 2 else "color")
    return page.copy() if page is src else page


def estimateSkew(src: np.ndarray, maxAngle: float = 45.0) -> float:
    """Tilt of the text lines of a page, in degrees, without rotating it (the angle :func:`cv2u.deskew` applies).

    Uses the same projection-profile search as ``deskew``: rotate the ink until it concentrates in as few rows as
    possible, first in 1° and then in 0.1° steps.

    Args:
        src: Image of a page (grayscale, BGR or BGRA).
        maxAngle: Largest tilt to look for, in degrees.

    Returns:
        The counter-clockwise rotation (as for ``cv.getRotationMatrix2D``) that levels the lines; 0 for a blank
        page.

    Raises:
        ValueError: If ``src`` is empty, or ``maxAngle`` is negative or not below 90.
    """
    check_image_not_empty("src", src)
    if not 0 <= maxAngle < 90:
        raise ValueError(f"maxAngle must be in [0, 90), got {maxAngle}")
    binary = _ink(src)
    if not np.count_nonzero(binary) or maxAngle == 0:
        return 0.0
    limit = math.floor(maxAngle)
    coarse = [float(a) for a in range(-limit, limit + 1)]
    best = max(coarse, key=lambda a: _line_score(binary, a))
    fine = {min(max(round(best + step / 10, 1), -maxAngle), maxAngle) for step in range(-10, 11)}
    return float(max(sorted(fine), key=lambda a: _line_score(binary, a)))


def removeSpeckles(src: np.ndarray, maxArea: int = 10) -> np.ndarray:
    """Remove the tiny ink specks (dust, noise) from a binarized document, keeping its polarity.

    :func:`cv2u.removeSmallObjects` treats nonzero pixels as foreground, which on black-on-white text removes
    the paper. This works on the ink, whichever value it has, and also accepts grayscale or color pages (the
    specks are painted over with the paper color).

    Args:
        src: 8-bit grayscale, BGR or BGRA document.
        maxArea: Ink regions with up to this many pixels are removed.

    Returns:
        A new image of the same shape and dtype.

    Raises:
        ValueError: If ``src`` is empty or not 8-bit, or ``maxArea`` is negative.
    """
    _check_uint8("src", src)
    if maxArea < 0:
        raise ValueError(f"maxArea must not be negative, got {maxArea}")
    ink = _ink_mask(src)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] <= maxArea
    small[0] = False
    speck = small[labels]
    out = src.copy()
    if not speck.any():
        return out
    paper = src[ink == 0]
    fill = np.median(paper, axis=0) if paper.size else np.full(src.shape[2:], 255)
    out[speck] = np.rint(fill).astype(np.uint8)
    return out
