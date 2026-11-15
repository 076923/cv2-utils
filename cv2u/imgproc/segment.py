"""Segmentation and mask editing: GrabCut, watershed, flood fill, components, trimaps, outlines and set operations."""

from collections.abc import Sequence
from typing import Literal, NamedTuple

import cv2
import numpy as np

from .._validate import check_channels, check_finite, check_image_not_empty

__all__ = [
    "Component",
    "connectedComponentsEx",
    "floodFillEx",
    "grabCutEx",
    "keepComponents",
    "maskCoverage",
    "maskHull",
    "maskOutline",
    "mergeMasks",
    "smoothMask",
    "splitTouching",
    "trimap",
    "watershedEx",
]

MergeMode = Literal["union", "intersection", "difference"]


def _foreground(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    """Nonzero pixels as a uint8 0/1 array (any channel count; for color images, any nonzero channel)."""
    if mask.ndim not in (2, 3):
        raise ValueError(f"{name} must be a 2-D array or an image, got shape {mask.shape}")
    check_image_not_empty(name, mask)
    if mask.ndim == 3:
        return np.asarray((mask != 0).any(axis=2), dtype=np.uint8)
    return np.asarray(mask != 0, dtype=np.uint8)


def _check_connectivity(connectivity: int) -> None:
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def _to_bgr(src: np.ndarray, name: str = "src") -> np.ndarray:
    """8-bit BGR version of a gray/BGR/BGRA image, as ``cv.grabCut`` and ``cv.watershed`` require."""
    if src.dtype != np.uint8:
        raise ValueError(f"{name} must be an 8-bit (uint8) image, got dtype {src.dtype}")
    check_image_not_empty(name, src)
    channels = check_channels(name, src)
    if channels == 1:
        return cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        return cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(src)


def grabCutEx(
    src: np.ndarray,
    rect: Sequence[int] | None = None,
    mask: np.ndarray | None = None,
    *,
    iterations: int = 5,
) -> np.ndarray:
    """``cv.grabCut`` from a box or a rough mask, returning a plain 0/255 foreground mask.

    ``cv.grabCut`` needs two preallocated ``(1, 65)`` float64 model arrays, a mask filled with ``GC_*`` codes, and
    its result has to be decoded with ``np.where((mask == GC_FGD) | (mask == GC_PR_FGD), ...)``. This hides all of
    that: pass either a bounding box or a rough mask, and get the cut-out mask.

    Args:
        src: 8-bit grayscale, BGR or BGRA image (converted to BGR for GrabCut).
        rect: ``(x, y, w, h)`` box around the object; everything outside is sure background.
        mask: Rough mask of the shape ``src.shape[:2]``. ``0`` is probable background and ``255`` (any other
            nonzero value except ``1`` and ``3``) is probable foreground; ``1`` means sure background and ``3``
            sure foreground, as in ``cv.GC_BGD`` / ``cv.GC_FGD``, for a user's scribbles.
        iterations: GrabCut iterations.

    Returns:
        ``uint8`` mask with values 0 and 255 (255 where GrabCut found foreground).

    Raises:
        ValueError: If neither or both of ``rect`` and ``mask`` are given, ``src`` is not 8-bit or is empty,
            ``rect`` is not a positive box inside the image, the mask shape does not match, or ``iterations`` is
            not positive.
    """
    if (rect is None) == (mask is None):
        raise ValueError("exactly one of rect and mask must be given")
    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")
    bgr = _to_bgr(src)
    height, width = bgr.shape[:2]
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    if rect is not None:
        if len(rect) != 4:
            raise ValueError(f"rect must be (x, y, w, h), got {rect!r}")
        x, y, w, h = (int(v) for v in rect)
        if w < 1 or h < 1 or x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError(f"rect must be a positive box inside the {width}x{height} image, got {rect!r}")
        if w == width and h == height:
            # GrabCut needs some sure background; with no margin it asserts. Shrink the box by one pixel.
            x, y, w, h = 1, 1, max(1, width - 2), max(1, height - 2)
        gc_mask = np.zeros((height, width), np.uint8)
        cv2.grabCut(bgr, gc_mask, (x, y, w, h), bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    else:
        assert mask is not None
        if mask.ndim != 2 or mask.shape != (height, width):
            raise ValueError(f"mask must have the shape {(height, width)}, got {mask.shape}")
        values = mask.astype(np.uint8) if mask.dtype == bool else np.asarray(mask)
        gc_mask = np.where(values != 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        if mask.dtype != bool:
            gc_mask[values == 1] = cv2.GC_BGD
            gc_mask[values == 3] = cv2.GC_FGD
        if not (gc_mask == cv2.GC_PR_FGD).any() and not (gc_mask == cv2.GC_FGD).any():
            return np.zeros((height, width), np.uint8)  # nothing marked as foreground
        if not (gc_mask == cv2.GC_PR_BGD).any() and not (gc_mask == cv2.GC_BGD).any():
            return np.full((height, width), 255, np.uint8)  # everything is foreground
        cv2.grabCut(bgr, gc_mask, (0, 0, width, height), bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    return np.asarray(fg, dtype=np.uint8) * 255


def _distance_markers(fg: np.ndarray, minDistance: float, threshold: float) -> np.ndarray:
    """Seed markers for the watershed: one label per local peak of the distance transform."""
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    peak = float(np.max(dist))
    if peak <= 0:
        return np.zeros(fg.shape, np.int32)
    seeds = np.asarray(dist >= threshold * peak, dtype=np.uint8)
    if minDistance > 0:
        # Keep only peaks that are the maximum within minDistance: a dilation window on the distance map.
        size = 2 * round(minDistance) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        local_max = cv2.dilate(dist, kernel)
        seeds &= np.asarray(dist >= local_max - 1e-6, dtype=np.uint8)
        # Grow the seeds back so that one object gives one connected seed instead of a cluster of isolated peaks.
        seeds = cv2.dilate(seeds, kernel) & np.asarray(dist >= threshold * peak, dtype=np.uint8)
    _, markers = cv2.connectedComponents(seeds, connectivity=8)
    return np.asarray(markers, dtype=np.int32)


def _watershed_labels(bgr: np.ndarray, fg: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Run ``cv.watershed`` with ``markers`` (0 = unknown) and clean the result into a label map."""
    count = int(markers.max())
    if count == 0:
        return np.zeros(fg.shape, np.int32)
    ws = markers.copy()
    # The background gets its own marker so the objects do not flood the whole image.
    sure_bg = cv2.dilate(fg, np.ones((3, 3), np.uint8), iterations=2) == 0
    ws[sure_bg] = count + 1
    cv2.watershed(bgr, ws)
    ws[ws == count + 1] = 0
    ws[fg == 0] = 0  # never label outside the mask
    # Watershed lines (-1) also run along the object/background border; give those pixels back to the one object
    # next to them and keep only the lines between two different objects as the boundary.
    kernel = np.ones((3, 3), np.uint8)
    positive = np.where(ws > 0, ws, 0).astype(np.float32)
    hi = cv2.dilate(positive, kernel)
    lo = -cv2.dilate(-np.where(ws > 0, ws, np.inf).astype(np.float32), kernel)
    line = ws == -1
    single = line & (hi > 0) & (hi == lo)
    ws[single] = hi[single].astype(np.int32)
    ws[ws == -1] = 0
    labels: np.ndarray = ws.astype(np.int32)
    # Relabel 1..N without gaps (a marker may vanish when a watershed basin swallows it).
    present = np.unique(labels)
    present = present[present > 0]
    lut = np.zeros(int(labels.max()) + 1, np.int32)
    lut[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    return np.asarray(lut[labels], dtype=np.int32)


def watershedEx(
    src: np.ndarray,
    markers: np.ndarray | None = None,
    *,
    mask: np.ndarray | None = None,
    minDistance: float = 0,
    threshold: float = 0.5,
) -> np.ndarray:
    """``cv.watershed`` with automatic markers: separate touching objects into a label map.

    ``cv.watershed`` wants an 8-bit BGR image and an ``int32`` marker image you have to build yourself (distance
    transform, threshold, ``connectedComponents``, a background marker), and it returns ``-1`` on the ridge lines
    and leaves the background label arbitrary. This builds the markers from the distance transform of the
    foreground when ``markers`` is ``None`` (the classic recipe for touching coins or cells) and normalizes the
    result: background is ``0``, objects are ``1..N`` with no gaps, the ``-1`` lines along the object borders are
    given back to the object, and the ``-1`` lines between two objects are mapped to ``0`` (so neighbouring
    objects are separated by a one-pixel gap). With automatic markers, blobs too small to get a seed keep one
    label each.

    Args:
        src: 8-bit grayscale, BGR or BGRA image.
        markers: ``int32`` (or any integer) label image of the shape ``src.shape[:2]``: ``0`` is unknown, positive
            values are seeds. When ``None``, seeds come from the distance transform of ``mask``.
        mask: Binary foreground mask (nonzero = object). When given with ``markers``, labels outside it are
            cleared. Without a mask and without ``markers``, ``src`` is thresholded with Otsu (bright objects on a
            dark background; invert the image first for dark objects).
        minDistance: Minimum distance in pixels between two seeds (automatic markers only). ``0`` takes every
            region where the distance transform exceeds ``threshold`` as a seed.
        threshold: Fraction of the distance-transform peak above which pixels become seeds (automatic markers).

    Returns:
        ``int32`` label map with the shape ``src.shape[:2]``: ``0`` background, ``1..N`` objects.

    Raises:
        ValueError: If ``src`` is not 8-bit or is empty, shapes do not match, ``minDistance`` is negative, or
            ``threshold`` is not in ``(0, 1]``.
    """
    check_finite("minDistance", minDistance, nonnegative=True)
    check_finite("threshold", threshold)
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    bgr = _to_bgr(src)
    shape = bgr.shape[:2]
    if mask is not None:
        fg = _foreground(mask)
        if fg.shape != shape:
            raise ValueError(f"mask must have the shape {shape}, got {mask.shape}")
    elif markers is not None:
        fg = np.ones(shape, np.uint8)
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        fg = np.asarray(otsu > 0, dtype=np.uint8)
    if markers is None:
        seeds = _distance_markers(fg, minDistance, threshold)
    else:
        if markers.ndim != 2 or markers.shape != shape:
            raise ValueError(f"markers must have the shape {shape}, got {markers.shape}")
        if not np.issubdtype(markers.dtype, np.integer):
            raise ValueError(f"markers must be an integer label image, got dtype {markers.dtype}")
        seeds = np.maximum(markers.astype(np.int32), 0)
    labels = _watershed_labels(bgr, fg, seeds)
    return _labelUnseeded(fg, labels) if markers is None else labels


def splitTouching(mask: np.ndarray, *, minDistance: float = 0, threshold: float = 0.5) -> np.ndarray:
    """Split touching blobs of a binary mask into separate labels (distance-transform watershed, no image needed).

    The usual "count the touching coins" recipe needs a color image for ``cv.watershed``; here the distance
    transform itself drives the flooding, so a mask alone is enough, and the cut lands on the neck between two
    blobs regardless of their sizes. The flooding runs level by level, so the time grows with the largest blob
    radius (a few hundred pixels is still fast).

    Args:
        mask: Binary mask; nonzero (or ``True``) pixels are foreground.
        minDistance: Minimum distance in pixels between two seeds (see :func:`watershedEx`).
        threshold: Fraction of the distance-transform peak above which pixels become seeds.

    Returns:
        ``int32`` label map: ``0`` background, ``1..N`` blobs (touching blobs are separated by a one-pixel gap).
        Blobs too small to get a seed of their own keep one label each.

    Raises:
        ValueError: If ``mask`` is not a 2-D array or an image, is empty, ``minDistance`` is negative, or
            ``threshold`` is not in ``(0, 1]``.
    """
    check_finite("minDistance", minDistance, nonnegative=True)
    check_finite("threshold", threshold)
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    fg = _foreground(mask)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    if float(np.max(dist)) <= 0:
        return np.zeros(fg.shape, np.int32)
    labels = _floodDistance(fg, dist, _distance_markers(fg, minDistance, threshold))
    return _labelUnseeded(fg, labels)


def _labelUnseeded(fg: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Give every foreground blob that received no seed (its peak is below the seed threshold) a label of its own.

    The one-pixel ridge lines are unlabeled too, but they touch a labeled region, so they are left alone.
    """
    rest = np.asarray((fg > 0) & (labels == 0), dtype=np.uint8)
    count, parts = cv2.connectedComponents(rest, connectivity=8)
    if count <= 1:
        return labels
    touched = cv2.dilate(np.asarray(labels > 0, dtype=np.uint8), np.ones((3, 3), np.uint8)) > 0
    unseeded = np.setdiff1d(np.unique(parts[rest > 0]), np.unique(parts[touched]))
    if unseeded.size == 0:
        return labels
    out = labels.copy()
    next_label = int(np.max(labels)) + 1
    for part in unseeded.tolist():
        out[parts == part] = next_label
        next_label += 1
    return out


def _floodDistance(fg: np.ndarray, dist: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Grow the markers downhill on the distance transform, level by level, until the mask is covered.

    ``cv.watershed`` floods by the gradient between neighbours, which on a smooth distance relief degenerates to a
    breadth-first race from the seeds. Flooding by distance level instead puts the ridge where the two blobs meet
    (their neck). Pixels reached by two labels in the same step stay 0 (the one-pixel boundary).
    """
    labels = markers.astype(np.float32)
    labels[fg == 0] = 0
    blocked = np.zeros(fg.shape, dtype=bool)
    kernel = np.ones((3, 3), np.uint8)
    peak = float(dist.max())
    step = max(1.0, peak / 256.0)
    level = peak
    inside = fg > 0
    while level > 0:
        level -= step
        allowed = inside & (dist > level) & (labels == 0) & ~blocked
        while allowed.any():
            hi = cv2.dilate(labels, kernel)
            front = allowed & (hi > 0)
            if not front.any():
                break  # the rest of this level is not reachable yet; wait for a lower level
            lo = -cv2.dilate(-np.where(labels > 0, labels, np.inf).astype(np.float32), kernel)
            conflict = front & (hi != lo)
            grow = front & ~conflict
            labels[grow] = hi[grow]
            blocked |= conflict
            allowed &= ~front
    return labels.astype(np.int32)


def floodFillEx(
    src: np.ndarray,
    seed: Sequence[int],
    tolerance: float | Sequence[float] = 0,
    *,
    connectivity: int = 4,
    fixedRange: bool = True,
) -> np.ndarray:
    """The region ``cv.floodFill`` would fill from ``seed``, as a 0/255 mask, without touching ``src``.

    ``cv.floodFill`` paints into the image, needs a mask two pixels larger than the image, packs the connectivity
    and the "mask only" flag into one bit field and takes the tolerance twice (``loDiff``/``upDiff``). This returns
    just the mask of the connected region whose pixels are within ``tolerance`` of the seed (or of their
    neighbors, with ``fixedRange=False``).

    Args:
        src: Grayscale, BGR or BGRA image, 8-bit or float32.
        seed: ``(x, y)`` start pixel.
        tolerance: Maximum difference from the seed, as one number for all channels or one value per channel.
        connectivity: 4 or 8.
        fixedRange: Compare each pixel with the seed (``True``) or with its already-filled neighbor (``False``,
            lets the region drift across smooth gradients).

    Returns:
        ``uint8`` mask with values 0 and 255 and the shape ``src.shape[:2]``.

    Raises:
        ValueError: If ``src`` is empty or not 8-bit / float32, the seed is outside the image, ``tolerance`` is
            negative or not finite, or ``connectivity`` is not 4 or 8.
    """
    check_image_not_empty("src", src)
    channels = check_channels("src", src)
    if src.dtype not in (np.uint8, np.float32):
        raise ValueError(f"src must be an 8-bit (uint8) or float32 image, got dtype {src.dtype}")
    _check_connectivity(connectivity)
    if len(seed) != 2:
        raise ValueError(f"seed must be (x, y), got {seed!r}")
    height, width = src.shape[:2]
    x, y = int(seed[0]), int(seed[1])
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"seed must be inside the {width}x{height} image, got {seed!r}")
    diffs = [float(v) for v in np.ravel(np.asarray(tolerance, dtype=np.float64)).tolist()]
    for value in diffs:
        check_finite("tolerance", value, nonnegative=True)
    if len(diffs) == 1:
        diffs = diffs * 4
    elif len(diffs) != channels:
        raise ValueError(f"tolerance must be one value or one per channel ({channels}), got {tolerance!r}")
    else:
        diffs = diffs + [0.0] * (4 - len(diffs))
    image = src if channels != 4 else cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
    image = np.ascontiguousarray(image)
    fill_mask = np.zeros((height + 2, width + 2), np.uint8)
    flags = connectivity | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    if fixedRange:
        flags |= cv2.FLOODFILL_FIXED_RANGE
    cv2.floodFill(image.copy(), fill_mask, (x, y), 0, tuple(diffs), tuple(diffs), flags)
    return np.ascontiguousarray(fill_mask[1:-1, 1:-1])


class Component(NamedTuple):
    """One connected region of a mask (see :func:`connectedComponentsEx`)."""

    label: int
    """Label in the ``cv.connectedComponents`` label map (1-based; 0 is the background)."""
    area: int
    """Number of pixels."""
    box: tuple[int, int, int, int]
    """Bounding box ``(x, y, w, h)``."""
    centroid: tuple[float, float]
    """Center of mass ``(x, y)``."""
    mask: np.ndarray
    """``uint8`` 0/255 mask of the full image size with only this component."""


def connectedComponentsEx(mask: np.ndarray, *, connectivity: int = 8, minArea: int = 0) -> list[Component]:
    """``cv.connectedComponentsWithStats`` as a list of components, largest first, each with its own mask.

    OpenCV returns four parallel arrays (count, labels, stats, centroids) indexed by ``CC_STAT_*`` constants, with
    the background in row 0. This gives one named tuple per region, sorted by area, so "the second largest blob"
    is ``components[1]`` and its box is ``components[1].box``.

    Args:
        mask: Binary mask; nonzero (or ``True``) pixels are foreground.
        connectivity: 8 (diagonal neighbors connect) or 4.
        minArea: Skip regions with fewer pixels.

    Returns:
        A list of :class:`Component`, sorted by area (largest first, ties by label); empty for an empty mask.

    Raises:
        ValueError: If ``connectivity`` is not 4 or 8, ``minArea`` is negative, or ``mask`` is not a 2-D array or
            an image, or is empty.
    """
    _check_connectivity(connectivity)
    if minArea < 0:
        raise ValueError(f"minArea must not be negative, got {minArea}")
    fg = _foreground(mask)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=connectivity)
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minArea:
            continue
        box = (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
        centroid = (float(centroids[label, 0]), float(centroids[label, 1]))
        component_mask = np.asarray(labels == label, dtype=np.uint8) * 255
        components.append(Component(label, area, box, centroid, component_mask))
    components.sort(key=lambda c: (-c.area, c.label))
    return components


def keepComponents(
    mask: np.ndarray,
    *,
    minArea: int | None = None,
    maxArea: int | None = None,
    keepTop: int | None = None,
    connectivity: int = 8,
) -> np.ndarray:
    """Keep the connected regions of a mask by size (``minArea``/``maxArea``) and/or the ``keepTop`` largest.

    One call for the usual clean-up chain: ``cv.connectedComponentsWithStats``, a boolean over ``CC_STAT_AREA``, and
    a look-up back into the label map. Generalizes :func:`removeSmallObjects` (minimum only) and
    :func:`largestComponent` (``keepTop=1``).

    Args:
        mask: Binary mask; nonzero (or ``True``) pixels are foreground.
        minArea: Drop regions with fewer pixels.
        maxArea: Drop regions with more pixels.
        keepTop: After the size filter, keep only the N largest regions.
        connectivity: 8 (diagonal neighbors connect) or 4.

    Returns:
        ``uint8`` mask with values 0 and 255.

    Raises:
        ValueError: If a limit is negative, ``minArea > maxArea``, ``connectivity`` is not 4 or 8, or ``mask`` is
            not a 2-D array or an image, or is empty.
    """
    _check_connectivity(connectivity)
    for name, value in (("minArea", minArea), ("maxArea", maxArea), ("keepTop", keepTop)):
        if value is not None and value < 0:
            raise ValueError(f"{name} must not be negative, got {value}")
    if minArea is not None and maxArea is not None and minArea > maxArea:
        raise ValueError(f"minArea must not exceed maxArea, got {minArea} > {maxArea}")
    fg = _foreground(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=connectivity)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = np.ones(count, dtype=bool)
    keep[0] = False
    if minArea is not None:
        keep &= areas >= minArea
    if maxArea is not None:
        keep &= areas <= maxArea
    if keepTop is not None:
        kept = np.flatnonzero(keep)
        order = kept[np.argsort(-areas[kept], kind="stable")]
        keep[:] = False
        keep[order[:keepTop]] = True
    return np.asarray(keep[labels], dtype=np.uint8) * 255


def trimap(mask: np.ndarray, width: int, *, unknown: int = 128) -> np.ndarray:
    """Trimap for matting from a binary mask: sure background 0, unknown band ``unknown``, sure foreground 255.

    Matting models (and ``cv.alphamat`` from contrib) want a trimap, which is the mask eroded for the foreground
    and dilated for the background, with the band in between marked unknown. Two morphology calls and an
    ``np.where``, but easy to get wrong (an odd kernel, the band width counted once or twice).

    Args:
        mask: Binary mask; nonzero (or ``True``) pixels are foreground.
        width: Half width of the unknown band, in pixels, on each side of the mask edge (``0`` returns the mask
            as 0/255).
        unknown: Value of the unknown band.

    Returns:
        ``uint8`` image with the values ``0``, ``unknown`` and ``255``.

    Raises:
        ValueError: If ``width`` is negative, ``unknown`` is not in ``[1, 254]``, or ``mask`` is not a 2-D array or
            an image, or is empty.
    """
    if width < 0:
        raise ValueError(f"width must not be negative, got {width}")
    if not 1 <= unknown <= 254:
        raise ValueError(f"unknown must be in [1, 254], got {unknown}")
    fg = _foreground(mask)
    if width == 0:
        return fg * 255
    size = 2 * int(width) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    sure_fg = cv2.erode(fg, kernel)
    maybe = cv2.dilate(fg, kernel)
    out = np.where(maybe > 0, unknown, 0).astype(np.uint8)
    out[sure_fg > 0] = 255
    return out


def smoothMask(mask: np.ndarray, radius: float, *, threshold: float = 0.5) -> np.ndarray:
    """Smooth the jagged edges of a binary mask: Gaussian blur, then threshold back to 0/255.

    Morphological opening/closing round corners unevenly and shift the edge; blurring the mask as a float image
    and re-thresholding it at ``0.5`` keeps the edge where it is while removing staircase pixels, nicks and spikes
    smaller than ``radius``.

    Args:
        mask: Binary mask; nonzero (or ``True``) pixels are foreground.
        radius: Sigma of the Gaussian blur, in pixels (``0`` returns the mask as 0/255).
        threshold: Fraction in ``(0, 1)`` above which the blurred mask counts as foreground; below ``0.5`` grows
            the mask, above shrinks it.

    Returns:
        ``uint8`` mask with values 0 and 255.

    Raises:
        ValueError: If ``radius`` is negative, ``threshold`` is not in ``(0, 1)``, or ``mask`` is not a 2-D array or
            an image, or is empty.
    """
    check_finite("radius", radius, nonnegative=True)
    check_finite("threshold", threshold)
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    fg = _foreground(mask)
    if radius == 0:
        return fg * 255
    blurred = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), radius)
    return np.asarray(blurred > threshold, dtype=np.uint8) * 255


def maskOutline(mask: np.ndarray, thickness: int = 1, *, inside: bool = False) -> np.ndarray:
    """Band of ``thickness`` pixels along the boundary of a mask, outside the shape (or inside with ``inside``).

    ``cv.drawContours`` with a thickness draws a band centered on the contour (half inside, half outside) and
    misses one-pixel details; ``cv.morphologyEx(..., MORPH_GRADIENT)`` gives both sides at once. This is the
    dilated minus the original (or the original minus the eroded), which is what a halo, a stroke or an
    "edge pixels only" measurement needs.

    Args:
        mask: Binary mask; nonzero (or ``True``) pixels are foreground.
        thickness: Width of the band, in pixels.
        inside: Take the band from the inside of the shape (the shape minus its erosion).

    Returns:
        ``uint8`` mask with values 0 and 255.

    Raises:
        ValueError: If ``thickness`` is not positive, or ``mask`` is not a 2-D array or an image, or is empty.
    """
    if thickness < 1:
        raise ValueError(f"thickness must be positive, got {thickness}")
    fg = _foreground(mask)
    size = 2 * int(thickness) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    band = fg & (1 - cv2.erode(fg, kernel)) if inside else cv2.dilate(fg, kernel) & (1 - fg)
    return np.asarray(band, dtype=np.uint8) * 255


def mergeMasks(masks: Sequence[np.ndarray], mode: MergeMode = "union") -> np.ndarray:
    """Combine masks with a set operation: union, intersection, or the first minus all the others.

    ``cv.bitwise_or``/``cv.bitwise_and`` take exactly two masks of the same dtype, and "difference" is two calls
    with ``cv.bitwise_not`` in between. This accepts any number of masks (``bool``, 0/255, color) and returns a
    0/255 result.

    Args:
        masks: Masks of the same ``(height, width)``; nonzero (or ``True``) pixels are foreground.
        mode: ``"union"`` (any mask), ``"intersection"`` (every mask) or ``"difference"`` (the first mask with
            the pixels of all later masks removed).

    Returns:
        ``uint8`` mask with values 0 and 255.

    Raises:
        ValueError: If no masks are given, their shapes differ, ``mode`` is unknown, or a mask is not a 2-D array or
            an image, or is empty.
    """
    if mode not in ("union", "intersection", "difference"):
        raise ValueError(f"mode must be 'union', 'intersection' or 'difference', got {mode!r}")
    if len(masks) == 0:
        raise ValueError("masks must not be empty")
    layers = [_foreground(m, "masks").astype(bool) for m in masks]
    shape = layers[0].shape
    for layer in layers[1:]:
        if layer.shape != shape:
            raise ValueError(f"masks must all have the shape {shape}, got {layer.shape}")
    stack = np.stack(layers)
    if mode == "union":
        out = stack.any(axis=0)
    elif mode == "intersection":
        out = stack.all(axis=0)
    else:
        out = layers[0] & ~stack[1:].any(axis=0)
    return np.asarray(out, dtype=np.uint8) * 255


def maskHull(mask: np.ndarray) -> np.ndarray:
    """Filled convex hull of all the foreground pixels of a mask.

    ``cv.convexHull`` wants a point list, so the usual route is ``cv.findNonZero`` (which fails on an empty mask
    and on color masks), the hull, and ``cv.fillConvexPoly``. One call here; an empty mask gives an empty mask.

    Args:
        mask: Binary mask; nonzero (or ``True``) pixels are foreground.

    Returns:
        ``uint8`` mask with values 0 and 255.

    Raises:
        ValueError: If ``mask`` is not a 2-D array or an image, or is empty.
    """
    fg = _foreground(mask)
    points = cv2.findNonZero(fg)
    out = np.zeros(fg.shape, np.uint8)
    if points is None:
        return out
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(out, hull, 255)
    return out


def maskCoverage(mask: np.ndarray, roi: Sequence[int] | None = None) -> float:
    """Fraction of the mask (or of a box inside it) that is foreground, from 0 to 1.

    ``cv.countNonZero`` wants a single-channel 8-bit array and the division by the area is left to you; this
    also counts ``bool`` and color masks and takes an ``(x, y, w, h)`` box.

    Args:
        mask: Binary mask; nonzero (or ``True``) pixels are foreground.
        roi: Optional ``(x, y, w, h)`` box; the fraction is measured inside the part of the box that lies in
            the image.

    Returns:
        ``float`` in ``[0, 1]`` (``0.0`` when the box does not overlap the image).

    Raises:
        ValueError: If ``roi`` is not a box with positive size, or ``mask`` is not a 2-D array or an image, or is
            empty.
    """
    fg = _foreground(mask)
    if roi is not None:
        if len(roi) != 4:
            raise ValueError(f"roi must be (x, y, w, h), got {roi!r}")
        x, y, w, h = (int(v) for v in roi)
        if w < 1 or h < 1:
            raise ValueError(f"roi must have a positive size, got {roi!r}")
        height, width = fg.shape
        fg = fg[max(y, 0) : min(y + h, height), max(x, 0) : min(x + w, width)]
        if fg.size == 0:
            return 0.0
    return float(np.count_nonzero(fg)) / float(fg.size)
