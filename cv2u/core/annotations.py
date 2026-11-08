"""Annotation formats: polygons, RLE masks, YOLO labels, label maps and the conversions between them.

Detection and segmentation datasets store the same objects in different shapes: a binary mask, a list of polygons
(COCO), a run-length encoding (COCO RLE), normalized ``class cx cy w h`` rows (YOLO) or a label map of class ids.
OpenCV has ``cv.findContours`` and ``cv.fillPoly`` for two of the steps and nothing for the rest; this module
holds the small, well-defined conversions and measures (area, IoU, flips) that every dataset script rewrites.

Conventions: boxes are ``(x, y, width, height)``, polygons are ``(N, 2)`` arrays (or lists) of ``(x, y)`` points,
sizes are ``(width, height)``, label maps are 2-D integer arrays of class ids with ``background`` (default 0) for
"no object". Coordinates are continuous (a pixel spans ``[x, x + 1)``), as in COCO and YOLO.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .._validate import check_color, check_finite, check_image_not_empty
from ..imgproc.color import distinctColors
from ..imgproc.contour import approxContour, asContour, contoursToMask
from ..imgproc.mask import _foreground
from .boxes import _boxes

__all__ = [
    "boxesToPolygons",
    "boxesToYolo",
    "colorToLabelMap",
    "drawPolygons",
    "drawSegmentation",
    "flipBoxes",
    "flipPolygons",
    "instanceMasksToLabelMap",
    "labelMapToColor",
    "labelMapToMasks",
    "maskIoU",
    "maskToPolygons",
    "polygonArea",
    "polygonIoU",
    "polygonsToBoxes",
    "polygonsToMask",
    "readYoloLabels",
    "rleDecode",
    "rleEncode",
    "writeYoloLabels",
    "yoloToBoxes",
]

Color = float | Sequence[float]


def _size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    if width < 1 or height < 1:
        raise ValueError(f"size must be at least 1x1, got {size}")
    return int(width), int(height)


def _polygon(polygon: ArrayLike, name: str = "polygon") -> np.ndarray:
    """One polygon as a float64 ``(N, 2)`` array (accepts ``(N, 1, 2)`` contours and point lists)."""
    array = np.asarray(polygon, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.shape[-1] != 2:
        raise ValueError(f"{name} must have 2 coordinates (x, y) per point, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.reshape(-1, 2)


def _polygons(polygons: Sequence[ArrayLike] | np.ndarray, name: str = "polygons") -> list[np.ndarray]:
    """A sequence of polygons (or one polygon, as an ``(N, 2)`` array or point list) as float64 ``(N, 2)`` arrays."""
    if not isinstance(polygons, np.ndarray):
        try:
            array = np.asarray(polygons, dtype=np.float64)
        except (ValueError, TypeError):  # ragged: polygons of different lengths
            return [_polygon(p, name) for p in polygons]
    else:
        array = polygons
    if array.size == 0 and array.ndim <= 1:
        return []
    if array.ndim == 2 and array.shape[-1] == 2:
        return [_polygon(array, name)]  # one polygon
    if array.ndim == 3 and array.shape[-1] == 2:
        return [_polygon(p, name) for p in array]  # equal-length polygons, or one (N, 1, 2) contour
    raise ValueError(f"{name} must be a sequence of (N, 2) polygons, got shape {array.shape}")


def _label_map(labels: np.ndarray, name: str = "labels") -> np.ndarray:
    if labels.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array of class ids, got shape {labels.shape}")
    check_image_not_empty(name, labels)
    if labels.dtype == bool:
        return labels.astype(np.int32)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array of class ids, got dtype {labels.dtype}")
    if labels.size and int(labels.min()) < 0:
        raise ValueError(f"{name} must not contain negative class ids, got {int(labels.min())}")
    return labels


def _flip_code(flipCode: int) -> int:
    if flipCode not in (0, 1, -1):
        raise ValueError(f"flipCode must be 0 (vertical), 1 (horizontal) or -1 (both), got {flipCode}")
    return flipCode


# --- masks and polygons ----------------------------------------------------------


def maskToPolygons(mask: np.ndarray, epsilon: float = 0.0, minArea: float = 0) -> list[np.ndarray]:
    """Outer contours of a binary mask as ``(N, 2)`` polygons, optionally simplified (COCO-style segmentation).

    ``cv.findContours`` wants an 8-bit single-channel image, returns ``(N, 1, 2)`` arrays and keeps one-pixel
    specks that no annotation format can store. Here any mask works (nonzero is foreground), polygons with fewer
    than 3 points are dropped, and ``epsilon`` simplifies each polygon like :func:`approxContour`.

    Args:
        mask: Binary mask; nonzero (or ``True``) pixels are foreground (color masks count any nonzero channel).
        epsilon: Simplification tolerance as a fraction of each polygon's perimeter (``0.01`` = 1%); 0 keeps every
            contour point.
        minArea: Drop polygons whose ``cv.contourArea`` is smaller than this.

    Returns:
        A list of ``int32`` arrays of shape ``(N, 2)``, largest area first. The inverse is :func:`polygonsToMask`.

    Raises:
        ValueError: If ``mask`` is not a 2-D array or an image, is empty, or ``epsilon``/``minArea`` is negative.
    """
    check_finite("epsilon", epsilon, nonnegative=True)
    check_finite("minArea", minArea, nonnegative=True)
    fg = _foreground(mask)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items = []
    for contour in contours:
        if epsilon > 0:
            contour = approxContour(contour, epsilon)
        area = cv2.contourArea(contour)
        if len(contour) >= 3 and area >= minArea:
            items.append((contour.reshape(-1, 2).astype(np.int32), area))
    items.sort(key=lambda item: item[1], reverse=True)
    return [polygon for polygon, _ in items]


def polygonsToMask(polygons: Sequence[ArrayLike] | np.ndarray, size: tuple[int, int], value: int = 255) -> np.ndarray:
    """Rasterize polygons into a new ``uint8`` mask of ``size`` (the inverse of :func:`maskToPolygons`).

    ``cv.fillPoly`` needs a pre-allocated image and ``int32`` ``(N, 1, 2)`` arrays; this takes point lists and
    float coordinates (rounded) and returns the mask.

    Args:
        polygons: One polygon ``(N, 2)`` or a sequence of polygons (an empty sequence gives an all-zero mask).
        size: ``(width, height)`` of the mask.
        value: Value of the polygon pixels, 1 to 255.

    Returns:
        ``uint8`` mask with values 0 and ``value``.

    Raises:
        ValueError: If ``size`` is not positive, ``value`` is outside ``[1, 255]`` or a polygon is empty / not 2-D.
    """
    if not 1 <= value <= 255:
        raise ValueError(f"value must be between 1 and 255, got {value}")
    mask = contoursToMask([asContour(p) for p in _polygons(polygons)], _size(size))
    if value != 255:
        mask[mask != 0] = value
    return mask


def boxesToPolygons(boxes: ArrayLike) -> list[np.ndarray]:
    """The four corners of each box ``(x, y, width, height)`` as a ``(4, 2)`` polygon (clockwise from top-left).

    Lets boxes go through the polygon functions (:func:`polygonsToMask`, :func:`polygonIoU`, :func:`flipPolygons`)
    and into COCO ``segmentation`` fields.

    Returns:
        A list with one float64 ``(4, 2)`` array per box.

    Raises:
        ValueError: If the boxes do not have 4 finite values.
    """
    array = _boxes(boxes, "boxes") if np.size(boxes) else np.zeros((0, 4))
    out = []
    for x, y, w, h in array:
        out.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64))
    return out


def polygonsToBoxes(polygons: Sequence[ArrayLike] | np.ndarray) -> np.ndarray:
    """Bounding box ``(x, y, width, height)`` of each polygon, in continuous coordinates.

    ``cv.boundingRect`` works on one integer contour at a time and counts pixels (``width = xmax - xmin + 1``);
    this keeps float coordinates and the COCO/YOLO convention ``width = xmax - xmin``.

    Args:
        polygons: One polygon ``(N, 2)`` or a sequence of polygons.

    Returns:
        A float64 array of shape ``(N, 4)`` (``(0, 4)`` for no polygons).

    Raises:
        ValueError: If a polygon is empty, not 2-D or not finite.
    """
    items = _polygons(polygons)
    if not items:
        return np.zeros((0, 4), dtype=np.float64)
    boxes = np.empty((len(items), 4), dtype=np.float64)
    for i, polygon in enumerate(items):
        low, high = polygon.min(axis=0), polygon.max(axis=0)
        boxes[i] = (low[0], low[1], high[0] - low[0], high[1] - low[1])
    return boxes


def polygonArea(polygon: ArrayLike) -> float:
    """Area of a simple polygon (shoelace formula), in continuous coordinates.

    ``cv.contourArea`` rounds the points to integers first, which is wrong for normalized or sub-pixel polygons.

    Args:
        polygon: ``(N, 2)`` points (or a point list); the polygon is closed implicitly.

    Returns:
        The absolute area (0 for fewer than 3 points).

    Raises:
        ValueError: If ``polygon`` is empty, not 2-D or not finite.
    """
    points = _polygon(polygon)
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def polygonIoU(polygonA: ArrayLike, polygonB: ArrayLike, resolution: int = 1024) -> float:
    """Intersection over union of two polygons, by rasterizing both over their common bounding box.

    OpenCV has no polygon clipping (``cv.intersectConvexConvex`` handles convex shapes only). Rasterization works
    for any polygons, including concave and self-overlapping ones. Precision: the shared bounding box is scaled so
    its longer side has ``resolution`` pixels and the polygons are drawn with 1/256 sub-pixel accuracy, so the
    error is about ``1 / resolution`` of the box side, enough for matching and metrics.

    Args:
        polygonA: ``(N, 2)`` points.
        polygonB: ``(M, 2)`` points.
        resolution: Raster size of the longer side of the common bounding box, in pixels.

    Returns:
        A value from 0 (disjoint) to 1 (identical); 0 when either polygon has no area.

    Raises:
        ValueError: If a polygon is empty, not 2-D or not finite, or ``resolution`` is less than 1.
    """
    if resolution < 1:
        raise ValueError(f"resolution must be at least 1, got {resolution}")
    a, b = _polygon(polygonA, "polygonA"), _polygon(polygonB, "polygonB")
    if len(a) < 3 or len(b) < 3:
        return 0.0
    both = np.concatenate([a, b])
    low, high = both.min(axis=0), both.max(axis=0)
    extent = float(max(high[0] - low[0], high[1] - low[1]))
    if extent <= 0:
        return 0.0
    scale = resolution / extent
    shift = 8
    width = int(np.ceil((high[0] - low[0]) * scale)) + 1
    height = int(np.ceil((high[1] - low[1]) * scale)) + 1
    masks = []
    for polygon in (a, b):
        fixed = np.rint((polygon - low) * scale * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [fixed], 1, cv2.LINE_8, shift)
        masks.append(mask.astype(bool))
    union = int(np.count_nonzero(masks[0] | masks[1]))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(masks[0] & masks[1])) / union


def maskIoU(maskA: np.ndarray, maskB: np.ndarray) -> float:
    """Intersection over union of two binary masks of the same shape (the segmentation counterpart of :func:`boxIoU`).

    Args:
        maskA: Binary mask; nonzero (or ``True``) pixels are foreground.
        maskB: Binary mask of the same height and width.

    Returns:
        A value from 0 to 1; 0 when both masks are empty.

    Raises:
        ValueError: If a mask is not a 2-D array or an image, is empty, or the shapes differ.
    """
    a, b = _foreground(maskA), _foreground(maskB)
    if a.shape != b.shape:
        raise ValueError(f"maskB must have the shape {a.shape}, got {b.shape}")
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(a & b)) / union


# --- run-length encoding ---------------------------------------------------------


def rleEncode(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as COCO uncompressed RLE: ``{"counts": [...], "size": [height, width]}``.

    The runs alternate background/foreground starting with background, in column-major (Fortran) order, exactly
    as ``pycocotools.mask.encode`` / ``frPyObjects`` read them. ``pycocotools.mask.frPyObjects(rle, h, w)`` turns
    the result into the compressed string form when a dataset needs it; this module stays pure NumPy.

    Args:
        mask: Binary mask; nonzero (or ``True``) pixels are foreground.

    Returns:
        A dict with ``"counts"`` (a list of ``int``) and ``"size"`` (``[height, width]``).

    Raises:
        ValueError: If ``mask`` is not a 2-D array or an image, or is empty.
    """
    fg2 = _foreground(mask)
    fg = fg2.ravel(order="F")
    changes = np.flatnonzero(fg[1:] != fg[:-1]) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [fg.size]])
    counts = ends - starts
    if fg[0]:
        counts = np.concatenate([[0], counts])  # the first run is background; COCO stores it as a 0-length run
    height, width = fg2.shape
    return {"counts": [int(c) for c in counts], "size": [int(height), int(width)]}


def rleDecode(rle: dict[str, Any]) -> np.ndarray:
    """Decode COCO uncompressed RLE (see :func:`rleEncode`) into a ``uint8`` mask with values 0 and 255.

    Args:
        rle: ``{"counts": [...], "size": [height, width]}`` with integer counts (compressed string counts are not
            supported; decode those with ``pycocotools.mask.decode``).

    Raises:
        ValueError: If ``size`` is not positive, ``counts`` is not a list of non-negative integers, or the counts do
            not add up to ``height * width``.
    """
    height, width = (int(v) for v in rle["size"])
    if height < 1 or width < 1:
        raise ValueError(f"rle['size'] must be positive [height, width], got {rle['size']}")
    counts = rle["counts"]
    if isinstance(counts, (str, bytes)):
        raise ValueError("rle['counts'] must be a list of run lengths; compressed RLE strings are not supported")
    runs = np.asarray(counts, dtype=np.int64).reshape(-1)
    if runs.size and runs.min() < 0:
        raise ValueError("rle['counts'] must not contain negative run lengths")
    if int(runs.sum()) != height * width:
        raise ValueError(f"rle['counts'] must add up to {height * width} pixels, got {int(runs.sum())}")
    values = np.arange(runs.size) % 2  # background, foreground, background, ...
    flat = np.repeat(values.astype(np.uint8), runs)
    return np.asarray(flat.reshape((height, width), order="F"), dtype=np.uint8) * 255


# --- YOLO labels -----------------------------------------------------------------


def boxesToYolo(boxes: ArrayLike, size: tuple[int, int], classIds: ArrayLike | None = None) -> np.ndarray:
    """Convert boxes ``(x, y, width, height)`` in pixels to YOLO rows: normalized ``cx cy w h``, with the class first.

    Args:
        boxes: Boxes of shape ``(N, 4)`` (or one box).
        size: ``(width, height)`` of the image.
        classIds: One class id per box; when given, the rows are ``class cx cy w h`` as in a YOLO label file.

    Returns:
        A float64 array of shape ``(N, 4)``, or ``(N, 5)`` with ``classIds``. Values are not clipped to ``[0, 1]``.

    Raises:
        ValueError: If ``size`` is not positive, the boxes are invalid, or ``classIds`` has the wrong length.
    """
    width, height = _size(size)
    array = _boxes(boxes, "boxes") if np.size(boxes) else np.zeros((0, 4))
    rows = np.stack(
        [
            (array[:, 0] + array[:, 2] / 2) / width,
            (array[:, 1] + array[:, 3] / 2) / height,
            array[:, 2] / width,
            array[:, 3] / height,
        ],
        axis=1,
    )
    if classIds is None:
        return rows
    ids = np.asarray(classIds, dtype=np.float64).reshape(-1)
    if len(ids) != len(array):
        raise ValueError(f"classIds must have one value per box, got {len(ids)} for {len(array)} boxes")
    return np.concatenate([ids[:, None], rows], axis=1)


def yoloToBoxes(rows: ArrayLike, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert YOLO rows (``class cx cy w h`` or ``cx cy w h``, normalized) to pixel boxes ``(x, y, width, height)``.

    Args:
        rows: Array of shape ``(N, 5)`` (class first) or ``(N, 4)``; extra columns (confidence) are ignored.
        size: ``(width, height)`` of the image.

    Returns:
        ``(boxes, classIds)``: float64 boxes of shape ``(N, 4)`` and ``int64`` ids of shape ``(N,)``, or ``None``
        for ``classIds`` when the rows have no class column.

    Raises:
        ValueError: If ``size`` is not positive or the rows do not have 4 or more values.
    """
    width, height = _size(size)
    array = np.asarray(rows, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.int64)
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2 or array.shape[1] < 4:
        raise ValueError(f"rows must have shape (N, 4) or (N, 5+), got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("rows must be finite")
    has_class = array.shape[1] >= 5
    values = array[:, 1:5] if has_class else array[:, :4]
    w, h = values[:, 2] * width, values[:, 3] * height
    boxes = np.stack([values[:, 0] * width - w / 2, values[:, 1] * height - h / 2, w, h], axis=1)
    return boxes, (np.rint(array[:, 0]).astype(np.int64) if has_class else None)


def readYoloLabels(filename: str | os.PathLike[str], size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO label file (``class cx cy w h`` per line) into pixel boxes ``(x, y, width, height)`` and class ids.

    Any path works, including non-ASCII ones; blank lines are skipped and a missing or empty file means "no
    objects", as YOLO training tools treat it.

    Args:
        filename: Path of the ``.txt`` label file.
        size: ``(width, height)`` of the matching image.

    Returns:
        ``(boxes, classIds)``: float64 boxes of shape ``(N, 4)`` and ``int64`` ids of shape ``(N,)``.

    Raises:
        ValueError: If ``size`` is not positive, or a line does not have at least 5 numbers.
    """
    _size(size)
    path = Path(filename)
    rows: list[list[float]] = []
    if path.is_file():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                raise ValueError(f"{path}: line {number} must have 'class cx cy w h', got {line!r}")
            rows.append([float(v) for v in parts[:5]])
    if not rows:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.int64)
    boxes, ids = yoloToBoxes(np.array(rows), size)
    assert ids is not None
    return boxes, ids


def writeYoloLabels(
    filename: str | os.PathLike[str],
    boxes: ArrayLike,
    classIds: ArrayLike,
    size: tuple[int, int],
    makeDirs: bool = False,
) -> None:
    """Write pixel boxes ``(x, y, width, height)`` and class ids as a YOLO label file (``class cx cy w h`` per line).

    Args:
        filename: Path of the ``.txt`` file (any path, including non-ASCII); an empty box list writes an empty file.
        boxes: Boxes of shape ``(N, 4)``.
        classIds: One integer class id per box.
        size: ``(width, height)`` of the matching image.
        makeDirs: Create missing parent directories.

    Raises:
        ValueError: If ``size`` is not positive, the boxes are invalid, or ``classIds`` has the wrong length.
        OSError: If the file cannot be written.
    """
    rows = boxesToYolo(boxes, size, classIds)
    path = Path(filename)
    if makeDirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{round(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# --- label maps ------------------------------------------------------------------


def labelMapToColor(labels: np.ndarray, palette: ArrayLike | None = None, background: int = 0) -> np.ndarray:
    """Colorize a label map of class ids into a BGR image, for viewing or saving a segmentation.

    Args:
        labels: 2-D integer array of class ids (``bool`` counts as 0/1).
        palette: One BGR color per class id, indexed by id (``(K, 3)``); ``None`` uses
            ``distinctColors(labels.max() + 1)``, so the same ids always get the same colors.
        background: The id drawn black (set to ``-1`` to color every id from the palette).

    Returns:
        ``uint8`` BGR image with the shape of ``labels``.

    Raises:
        ValueError: If ``labels`` is not a 2-D integer array, is empty, has negative ids, or an id has no palette
            entry.
    """
    ids = _label_map(labels)
    top = int(ids.max()) if ids.size else 0
    if palette is None:
        lut = np.array(distinctColors(top + 1), dtype=np.uint8).reshape(-1, 3)
    else:
        given = np.asarray(palette, dtype=np.float64).reshape(-1, 3)
        check_color("palette", [float(v) for v in given.ravel()], np.dtype(np.uint8))
        lut = given.astype(np.uint8)
        if top >= len(lut):
            raise ValueError(f"palette must have a color for every class id up to {top}, got {len(lut)} colors")
    if 0 <= background < len(lut):
        lut = lut.copy()
        lut[background] = 0
    out: np.ndarray = lut[ids]
    return out


def colorToLabelMap(image: np.ndarray, palette: ArrayLike, background: int = 0) -> np.ndarray:
    """Recover a label map from a colorized segmentation image (the inverse of :func:`labelMapToColor`).

    Args:
        image: 8-bit BGR (or BGRA) image whose colors come from ``palette``.
        palette: One BGR color per class id, indexed by id (``(K, 3)``).
        background: The id for pixels whose color is not in the palette.

    Returns:
        ``int32`` array of class ids with the shape ``image.shape[:2]``.

    Raises:
        ValueError: If ``image`` is not an 8-bit color image or is empty, or ``palette`` is not ``(K, 3)``.
    """
    if image.ndim != 3 or image.shape[2] < 3 or image.dtype != np.uint8:
        raise ValueError(f"image must be an 8-bit BGR image, got shape {image.shape} and dtype {image.dtype}")
    check_image_not_empty("image", image)
    lut = np.asarray(palette, dtype=np.float64)
    if lut.ndim != 2 or lut.shape[1] != 3 or lut.size == 0:
        raise ValueError(f"palette must have shape (K, 3), got shape {lut.shape}")
    check_color("palette", [float(v) for v in lut.ravel()], np.dtype(np.uint8))

    def pack(bgr: np.ndarray) -> np.ndarray:
        values = bgr.astype(np.int64)
        return (values[..., 0] << 16) | (values[..., 1] << 8) | values[..., 2]

    codes = pack(lut.astype(np.uint8))
    unique, first = np.unique(codes, return_index=True)  # duplicate colors map to the lowest id
    pixels = pack(image[..., :3])
    position = np.searchsorted(unique, pixels)
    position = np.clip(position, 0, len(unique) - 1)
    found = unique[position] == pixels
    out = np.full(pixels.shape, background, dtype=np.int32)
    out[found] = first[position[found]]
    return out


def instanceMasksToLabelMap(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Stack binary instance masks into one label map: mask ``i`` gets id ``i + 1``, later masks on top.

    Args:
        masks: Binary masks of the same shape; nonzero (or ``True``) pixels are foreground.

    Returns:
        ``int32`` array of ids (0 = background) with the shape of the masks.

    Raises:
        ValueError: If ``masks`` is empty, or a mask is not 2-D, is empty or has a different shape.
    """
    if len(masks) == 0:
        raise ValueError("masks must contain at least one mask")
    first = _foreground(masks[0])
    out = np.zeros(first.shape, dtype=np.int32)
    for i, mask in enumerate(masks):
        fg = _foreground(mask)
        if fg.shape != first.shape:
            raise ValueError(f"masks must all have the shape {first.shape}, got {fg.shape} for mask {i}")
        out[fg.astype(bool)] = i + 1
    return out


def labelMapToMasks(labels: np.ndarray, background: int = 0) -> list[np.ndarray]:
    """Split a label map into one binary mask per id (the inverse of :func:`instanceMasksToLabelMap`).

    Args:
        labels: 2-D integer array of ids.
        background: The id that gets no mask.

    Returns:
        A list of ``uint8`` masks (0/255), one for each id from 1 to ``labels.max()`` except ``background``, so
        ``masks[i]`` is id ``i + 1`` when ``background`` is 0 (unused ids give all-zero masks).

    Raises:
        ValueError: If ``labels`` is not a 2-D integer array, is empty or has negative ids.
    """
    ids = _label_map(labels)
    top = int(ids.max())
    return [np.asarray(ids == i, dtype=np.uint8) * 255 for i in range(1, top + 1) if i != background]


# --- drawing ---------------------------------------------------------------------


def drawPolygons(
    src: np.ndarray,
    polygons: Sequence[ArrayLike] | np.ndarray,
    color: Sequence[Color] | Color | None = None,
    thickness: int = 2,
    fill: bool = False,
    alpha: float = 1.0,
    labels: Sequence[str] | None = None,
    fontScale: float = 0.5,
    lineType: int = cv2.LINE_AA,
) -> np.ndarray:
    """Draw polygons (outlines or filled, optionally translucent) with optional labels, in place.

    ``cv.polylines``/``cv.fillPoly`` want ``int32`` ``(N, 1, 2)`` arrays in a list, one color for all, and have no
    transparency; this takes any point lists and gives each polygon its own color by default.

    Args:
        src: 8-bit image to draw on.
        polygons: One polygon ``(N, 2)`` or a sequence of polygons (floats are rounded).
        color: One color for all polygons, one color per polygon, or ``None`` for :func:`distinctColors`.
        thickness: Outline thickness (ignored with ``fill``).
        fill: Fill the polygons instead of outlining them.
        alpha: Opacity of the drawing, from 0 to 1.
        labels: One text per polygon, drawn at its top-most point.
        fontScale: Hershey font size for the labels.
        lineType: As for ``cv.polylines``.

    Returns:
        ``src``.

    Raises:
        ValueError: If ``src`` is not 8-bit or is empty, ``alpha`` is outside ``[0, 1]``, ``thickness`` is not
            positive, or ``labels``/``color`` has a different length than ``polygons``.
    """
    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    check_image_not_empty("src", src)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if thickness < 1:
        raise ValueError(f"thickness must be positive, got {thickness}")
    items = _polygons(polygons)
    count = len(items)
    if labels is not None and len(labels) != count:
        raise ValueError(f"labels must have one entry per polygon, got {len(labels)} for {count} polygons")
    palette: list[Color]
    if color is None:
        palette = list(distinctColors(max(count, 1)))
    elif isinstance(color, (int, float)) or (len(color) in (3, 4) and all(isinstance(c, (int, float)) for c in color)):
        palette = [color] * max(count, 1)  # type: ignore[list-item]
    else:
        palette = list(color)
        if len(palette) != count:
            raise ValueError(f"color must have one entry per polygon, got {len(palette)} for {count} polygons")
    for entry in palette:
        check_color(
            "color", [float(entry)] if isinstance(entry, (int, float)) else [float(c) for c in entry], src.dtype
        )
    if count == 0 or alpha == 0:
        return src
    canvas = src.copy() if alpha < 1 else src
    for i, polygon in enumerate(items):
        contour = asContour(polygon)
        if fill:
            cv2.fillPoly(canvas, [contour], palette[i], lineType)
        else:
            cv2.polylines(canvas, [contour], True, palette[i], thickness, lineType)
        if labels is not None:
            x, y = (int(v) for v in contour[int(np.argmin(contour[:, 0, 1])), 0])
            (tw, th), baseline = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)
            top = max(y - th - baseline - 2, 0)
            cv2.rectangle(canvas, (x, top), (x + tw + 2, top + th + baseline + 2), palette[i], cv2.FILLED)
            text_color = 0 if _luminance(palette[i]) > 140 else 255
            cv2.putText(
                canvas, labels[i], (x + 1, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, fontScale, text_color, 1, lineType
            )
    if alpha < 1:
        cv2.addWeighted(src, 1 - alpha, canvas, alpha, 0, dst=src)
    return src


def _luminance(color: Color) -> float:
    if isinstance(color, (int, float)):
        return float(color)
    values = [float(c) for c in color]
    if len(values) < 3:
        return values[0]
    return 0.114 * values[0] + 0.587 * values[1] + 0.299 * values[2]


def drawSegmentation(
    src: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.5,
    palette: ArrayLike | None = None,
    background: int = 0,
) -> np.ndarray:
    """Overlay a label map (or a binary mask) on an image in class colors; returns a new BGR image.

    :func:`overlayMask` tints one mask in one color; this colors every class of a segmentation at once, leaving
    ``background`` pixels untouched.

    Args:
        src: 8-bit grayscale, BGR or BGRA image (converted to BGR). It is not modified.
        labels: 2-D integer label map with the shape ``src.shape[:2]``. A ``bool`` array, or an array whose nonzero
            values are all 255, is treated as a binary mask (one class, id 1).
        alpha: Opacity of the colors, from 0 to 1.
        palette: One BGR color per class id, as for :func:`labelMapToColor`.
        background: The id left uncolored.

    Returns:
        A new ``uint8`` BGR image.

    Raises:
        ValueError: If ``src`` is not 8-bit or is empty, the label shape does not match, or ``alpha`` is outside
            ``[0, 1]``.
    """
    from ..imgproc.color import toBGR

    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    check_image_not_empty("src", src)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if labels.ndim != 2 or labels.shape != src.shape[:2]:
        raise ValueError(f"labels must have the shape {src.shape[:2]}, got {labels.shape}")
    ids = labels
    if labels.dtype == bool or (np.issubdtype(labels.dtype, np.integer) and np.all((labels == 0) | (labels == 255))):
        ids = (labels != 0).astype(np.int32)
    colors = labelMapToColor(ids, palette, background)
    out = toBGR(src)
    blended = cv2.addWeighted(out, 1 - alpha, colors, alpha, 0)
    selected = np.ascontiguousarray(ids != background).view(np.uint8)
    dst: np.ndarray = cv2.copyTo(blended, selected, out)
    return dst


# --- flips -----------------------------------------------------------------------


def flipBoxes(boxes: ArrayLike, size: tuple[int, int], flipCode: int) -> np.ndarray:
    """Flip boxes ``(x, y, width, height)`` with an image, with the ``flipCode`` of ``cv.flip``.

    ``cv.flip`` moves the pixels but not the annotations; this keeps the boxes aligned (continuous coordinates,
    so a box at ``x`` with width ``w`` lands at ``width - x - w``).

    Args:
        boxes: Boxes of shape ``(N, 4)`` (or one box).
        size: ``(width, height)`` of the image.
        flipCode: 0 (``cv.FLIP_MODE_X``) flips vertically, 1 (``cv.FLIP_MODE_Y``) horizontally, -1 both.

    Returns:
        A float64 array of shape ``(N, 4)``, or ``(4,)`` for a single input box.

    Raises:
        ValueError: If ``size`` is not positive, the boxes are invalid, or ``flipCode`` is not 0, 1 or -1.
    """
    width, height = _size(size)
    code = _flip_code(flipCode)
    single = np.ndim(boxes) == 1 and np.size(boxes) > 0
    out = _boxes(boxes, "boxes").copy() if np.size(boxes) else np.zeros((0, 4))
    if code in (1, -1):
        out[:, 0] = width - out[:, 0] - out[:, 2]
    if code in (0, -1):
        out[:, 1] = height - out[:, 1] - out[:, 3]
    return out[0] if single else out


def _flip_polygon(points: ArrayLike, size: tuple[int, int], flipCode: int) -> np.ndarray:
    """Flip one polygon in continuous coordinates (``x`` becomes ``width - x``).

    Continuous coordinates: ``x`` becomes ``width - x``. For integer pixel indices (the center convention of
    ``cv.findContours``), subtract 1 from the flipped coordinates.

    Args:
        points: ``(N, 2)`` points (or a point list / ``(N, 1, 2)`` contour).
        size: ``(width, height)`` of the image.
        flipCode: 0 flips vertically, 1 horizontally, -1 both.

    Returns:
        A float64 array of shape ``(N, 2)``.

    Raises:
        ValueError: If ``size`` is not positive, the points are empty or not 2-D, or ``flipCode`` is not 0, 1 or -1.
    """
    width, height = _size(size)
    code = _flip_code(flipCode)
    out = _polygon(points, "points").copy()
    if code in (1, -1):
        out[:, 0] = width - out[:, 0]
    if code in (0, -1):
        out[:, 1] = height - out[:, 1]
    return out


def flipPolygons(polygons: Sequence[ArrayLike], size: tuple[int, int], flipCode: int) -> list[np.ndarray]:
    """Flip polygons with an image, with the ``flipCode`` of ``cv.flip``.

    Continuous coordinates like :func:`flipBoxes` (``x`` becomes ``width - x``); for pixel-center landmarks
    use :func:`flipPoints`, which maps ``x`` to ``width - 1 - x`` like ``cv.flip`` moves pixels.

    Args:
        polygons: A list of ``(N, 2)`` polygons (or point lists / ``(N, 1, 2)`` contours).
        size: ``(width, height)`` of the image.
        flipCode: 0 flips vertically, 1 horizontally, -1 both.

    Returns:
        A list of float64 ``(N, 2)`` arrays. The vertex order is kept (so the winding direction reverses).

    Raises:
        ValueError: If ``size`` is not positive, a polygon is empty or not 2-D, or ``flipCode`` is not 0, 1 or -1.
    """
    _size(size)
    _flip_code(flipCode)
    return [_flip_polygon(p, size, flipCode) for p in _polygons(polygons)]
