"""Run a ``cv.dnn`` model end to end: load, preprocess, forward, decode and draw.

``cv.dnn`` stops at ``blobFromImage`` and ``Net.forward``: it does not remember how the image was letterboxed,
does not decode YOLO-style outputs, does not read Unicode paths on Windows and fails with a C++ assertion for a
missing file. These helpers fill those gaps with plain NumPy and the box utilities of :mod:`cv2u.core`.
"""

import os
from collections.abc import Sequence
from typing import Any, NamedTuple, Protocol

import cv2
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .._validate import check_finite, check_image_not_empty
from ..core.boxes import clipBoxes, nms, scaleBoxes, xyxyToXywh
from ..imgproc.color import distinctColors
from ..imgproc.draw import drawBoxes
from ..imgproc.resize import letterbox

__all__ = [
    "Detections",
    "PreprocessInfo",
    "drawDetections",
    "inferImage",
    "loadModel",
    "postprocessClassification",
    "postprocessDetections",
    "preprocessImage",
    "readClassNames",
    "sigmoid",
    "softmax",
]

Color = float | Sequence[float]


class _Net(Protocol):
    """What :func:`inferImage` needs from ``cv.dnn.Net`` (any object with these methods works)."""

    def setInput(self, blob: np.ndarray) -> None: ...

    def forward(self, outputNames: Sequence[str]) -> Any: ...

    def getUnconnectedOutLayersNames(self) -> Sequence[str]: ...


class PreprocessInfo(NamedTuple):
    """How :func:`preprocessImage` mapped the image into the network input; needed to map boxes back."""

    scale: tuple[float, float]
    """``(sx, sy)``: a point ``(x, y)`` of the original image is at ``(x * sx + left, y * sy + top)`` in the blob."""
    pad: tuple[int, int]
    """``(left, top)`` letterbox padding in pixels (``(0, 0)`` without letterboxing)."""
    inputSize: tuple[int, int]
    """Network input ``(width, height)``."""
    originalSize: tuple[int, int]
    """Original image ``(width, height)``."""


class Detections(NamedTuple):
    """Decoded detections in the coordinates of the original image, best score first."""

    boxes: np.ndarray
    """``(N, 4)`` float32 boxes ``(x, y, width, height)``, clipped to the original image."""
    scores: np.ndarray
    """``(N,)`` float32 confidences."""
    classIds: np.ndarray
    """``(N,)`` int32 class indices."""

    @property
    def size(self) -> int:
        """Number of detections (``len()`` of a named tuple is always its field count, 3)."""
        return len(self.scores)


def _require_dnn() -> Any:
    if not hasattr(cv2, "dnn"):
        raise RuntimeError("this OpenCV build has no cv2.dnn module; install opencv-python with the dnn module")
    return cv2.dnn


def softmax(x: ArrayLike, axis: int = -1) -> np.ndarray:
    """Softmax along ``axis``, numerically stable (the maximum is subtracted first).

    Raw classifier outputs (logits) need it to become probabilities; OpenCV has no NumPy-level softmax.

    Returns:
        A float32 array of the same shape, summing to 1 along ``axis``.
    """
    array = np.asarray(x, dtype=np.float32)
    if array.size == 0:
        return array
    shifted = np.exp(array - np.max(array, axis=axis, keepdims=True))
    result: np.ndarray = shifted / np.sum(shifted, axis=axis, keepdims=True)
    return result


def sigmoid(x: ArrayLike) -> np.ndarray:
    """Logistic function ``1 / (1 + exp(-x))``, for objectness and multi-label class logits.

    Returns:
        A float32 array of the same shape, without overflow warnings for large negative inputs.
    """
    array = np.asarray(x, dtype=np.float32)
    result: np.ndarray = 0.5 * (1.0 + np.tanh(0.5 * array))
    return result.astype(np.float32)


_softmax = softmax  # the ``softmax`` parameter below shadows the function


def preprocessImage(
    src: np.ndarray,
    size: tuple[int, int],
    letterboxPad: bool = True,
    color: Color = (114, 114, 114),
    scale: float = 1 / 255,
    mean: Sequence[float] = (0, 0, 0),
    swapRB: bool = True,
    dtype: DTypeLike = np.float32,
) -> tuple[np.ndarray, PreprocessInfo]:
    """Turn an image into an NCHW blob for ``Net.setInput`` and remember how to map results back.

    ``cv.dnn.blobFromImage`` resizes without keeping the aspect ratio and forgets the geometry, so boxes from the
    network cannot be mapped back to the image. This letterboxes (like YOLO), normalizes and returns the
    :class:`PreprocessInfo` that :func:`postprocessDetections` needs.

    Args:
        src: 8-bit grayscale, BGR or BGRA image (always fed to the network as 3 channels).
        size: Network input ``(width, height)``.
        letterboxPad: Keep the aspect ratio and pad with ``color`` (``True``), or stretch like ``blobFromImage``.
        color: Padding color (BGR).
        scale: Multiplier applied after ``mean`` subtraction (``1/255`` maps 8-bit pixels to ``[0, 1]``).
        mean: Per-channel values subtracted first, in the channel order of the blob (RGB when ``swapRB``).
        swapRB: Convert BGR to RGB, as most trained models expect.
        dtype: Blob data type.

    Returns:
        ``(blob, info)``: the ``(1, 3, height, width)`` blob and the mapping info.

    Raises:
        ValueError: If ``src`` is empty, ``size`` is not positive, or ``scale``/``mean`` are not finite.
    """
    check_image_not_empty("src", src)
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"size must be positive, got {size}")
    check_finite("scale", scale)
    mean_values = [float(m) for m in mean]
    if len(mean_values) not in (1, 3):
        raise ValueError(f"mean must have 1 or 3 values, got {mean!r}")
    for value in mean_values:
        check_finite("mean", value)

    if src.ndim == 2:
        bgr = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    elif src.shape[2] == 4:
        bgr = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
    elif src.shape[2] == 1:
        bgr = cv2.cvtColor(src[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        bgr = src
    h, w = bgr.shape[:2]
    if letterboxPad:
        resized, factor, pad = letterbox(bgr, size, color)
        scales = (factor, factor)
    else:
        resized = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA if width < w else cv2.INTER_LINEAR)
        scales, pad = (width / w, height / h), (0, 0)

    array = resized.astype(np.float32)
    if swapRB:
        array = array[:, :, ::-1]
    array = (array - np.asarray(mean_values, dtype=np.float32)) * np.float32(scale)
    blob = np.ascontiguousarray(array.transpose(2, 0, 1)[None]).astype(dtype)
    return blob, PreprocessInfo(scales, pad, (width, height), (w, h))


def _rows(output: ArrayLike, transpose: bool | None) -> np.ndarray:
    array = np.asarray(output, dtype=np.float32)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"output must have shape (1, N, K) or (1, K, N), got {np.shape(output)}")
    if transpose is None:
        # YOLOv8-style (1, 4 + C, 8400): the channel axis comes first and is the smaller one.
        transpose = array.shape[0] < array.shape[1]
    return array.T if transpose else array


def postprocessDetections(
    output: ArrayLike,
    info: PreprocessInfo,
    scoreThreshold: float = 0.25,
    iouThreshold: float = 0.45,
    maxCount: int | None = None,
    boxFormat: str = "cxcywh",
    hasObjectness: bool = False,
    classAgnostic: bool = False,
    normalized: bool = False,
    transpose: bool | None = None,
) -> Detections:
    """Decode a raw YOLO-style detector output into boxes on the original image.

    Handles the ``(1, N, 4 + C)`` and ``(1, 4 + C, N)`` layouts (YOLOv5 to YOLOv11 and similar), an optional
    objectness column, threshold filtering, undoing the letterbox of :func:`preprocessImage`, clipping and
    per-class NMS. ``cv.dnn`` leaves all of this to the user.

    Args:
        output: Network output with one row per candidate: 4 box values, optionally an objectness score, then
            one score per class.
        info: From :func:`preprocessImage`.
        scoreThreshold: Candidates whose best class score (times objectness) is below this are dropped.
        iouThreshold: NMS overlap threshold.
        maxCount: Keep at most this many detections.
        boxFormat: ``"cxcywh"`` (YOLO), ``"xywh"`` or ``"xyxy"``.
        hasObjectness: Column 4 is an objectness score multiplied into the class scores (YOLOv5/v7).
        classAgnostic: Run NMS over all classes at once instead of per class.
        normalized: Box values are in ``[0, 1]`` relative to the network input instead of pixels.
        transpose: Force the ``(4 + C, N)`` layout (``True``) or ``(N, 4 + C)`` (``False``); ``None`` picks
            the layout whose first axis is the longer one, which is right for every real model but may be wrong
            for tiny synthetic outputs.

    Returns:
        :class:`Detections` sorted by descending score.

    Raises:
        ValueError: If the output has no class columns, ``boxFormat`` is unknown, or a threshold is not finite.
    """
    if boxFormat not in ("cxcywh", "xywh", "xyxy"):
        raise ValueError(f"boxFormat must be 'cxcywh', 'xywh' or 'xyxy', got {boxFormat!r}")
    check_finite("scoreThreshold", scoreThreshold)
    check_finite("iouThreshold", iouThreshold, nonnegative=True)
    rows = _rows(output, transpose)
    first_class = 5 if hasObjectness else 4
    if rows.shape[1] <= first_class:
        needed = "4 box values, an objectness score" if hasObjectness else "4 box values"
        raise ValueError(f"output rows need {needed} and at least one class score, got {rows.shape[1]} columns")

    empty = Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32))
    if len(rows) == 0:
        return empty
    class_scores = rows[:, first_class:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(rows)), class_ids]
    if hasObjectness:
        scores = scores * rows[:, 4]
    keep = np.flatnonzero(scores >= scoreThreshold)
    if len(keep) == 0:
        return empty
    boxes = rows[keep, :4].astype(np.float64)
    scores, class_ids = scores[keep], class_ids[keep]

    if normalized:
        boxes *= np.array([*info.inputSize, *info.inputSize], dtype=np.float64)
    if boxFormat == "cxcywh":
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
    elif boxFormat == "xyxy":
        boxes = xyxyToXywh(boxes)
    boxes = clipBoxes(scaleBoxes(boxes, info.scale, info.pad, inverse=True), info.originalSize)

    if classAgnostic:
        offset_boxes = boxes
    else:
        # Shift each class to its own region so one NMS call never compares boxes of different classes.
        offset_boxes = boxes.copy()
        offset_boxes[:, 0] += class_ids * (max(info.originalSize) + 1.0)
    order = nms(offset_boxes, scores, iouThreshold, maxCount=maxCount)
    return Detections(
        boxes[order].astype(np.float32), scores[order].astype(np.float32), class_ids[order].astype(np.int32)
    )


def postprocessClassification(output: ArrayLike, topK: int = 5, softmax: bool = True) -> list[tuple[int, float]]:
    """Top-``k`` ``(classId, probability)`` pairs from a classifier output.

    Args:
        output: Output of shape ``(1, C)`` or ``(C,)``.
        topK: How many classes to return (fewer when the model has fewer classes).
        softmax: Apply :func:`softmax` first (``False`` when the model already outputs probabilities).

    Returns:
        A list sorted by descending probability.

    Raises:
        ValueError: If ``output`` is not one vector or ``topK`` is not positive.
    """
    if topK < 1:
        raise ValueError(f"topK must be positive, got {topK}")
    vector = np.asarray(output, dtype=np.float32).squeeze()
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"output must be one vector of class scores, got shape {np.shape(output)}")
    probs = _softmax(vector) if softmax else vector
    order = np.argsort(-probs, kind="stable")[:topK]
    return [(int(i), float(probs[i])) for i in order]


def loadModel(
    path: str | os.PathLike[str],
    config: str | os.PathLike[str] | None = None,
    backend: int | None = None,
    target: int | None = None,
    preferGpu: bool = False,
) -> Any:
    """Load a model with ``cv.dnn.readNet``, from any path, with clear errors.

    ``cv.dnn.readNet`` cannot open non-ASCII paths on Windows and reports a missing file as a C++ assertion. ONNX,
    TFLite and TensorFlow models are read from memory instead (``readNetFromONNX(buffer)`` etc.), so the path
    only has to be readable by Python; other formats fall back to ``readNet(path, config)``.

    Args:
        path: Model file (``.onnx``, ``.tflite``, ``.pb``, or anything ``readNet`` understands).
        config: Optional text config for the model (e.g. ``.pbtxt``).
        backend: ``cv.dnn.DNN_BACKEND_*`` to set, or ``None`` for OpenCV's default.
        target: ``cv.dnn.DNN_TARGET_*`` to set.
        preferGpu: Use the CUDA backend and target when this OpenCV build has it; silently keeps the default
            otherwise. Ignored when ``backend`` is given.

    Returns:
        A ``cv.dnn.Net``.

    Raises:
        FileNotFoundError: If ``path`` or ``config`` does not exist.
        RuntimeError: If this OpenCV build has no ``dnn`` module.
    """
    dnn = _require_dnn()
    model_path = os.fspath(path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")
    config_path = None if config is None else os.fspath(config)
    if config_path is not None and not os.path.isfile(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")

    ext = os.path.splitext(model_path)[1].lower()
    # Only the formats with a buffer reader (unicode-safe paths) are read into memory; the others let OpenCV
    # open the path itself, so a large Caffe or Darknet model is not loaded twice.
    if ext == ".onnx" and hasattr(dnn, "readNetFromONNX"):
        net = dnn.readNetFromONNX(np.fromfile(model_path, dtype=np.uint8))
    elif ext == ".tflite" and hasattr(dnn, "readNetFromTFLite"):
        net = dnn.readNetFromTFLite(np.fromfile(model_path, dtype=np.uint8))
    elif ext == ".pb" and hasattr(dnn, "readNetFromTensorflow"):
        config_buffer = np.zeros(0, np.uint8) if config_path is None else np.fromfile(config_path, dtype=np.uint8)
        net = dnn.readNetFromTensorflow(np.fromfile(model_path, dtype=np.uint8), config_buffer)
    else:
        net = dnn.readNet(model_path) if config_path is None else dnn.readNet(model_path, config_path)

    if backend is None and preferGpu and hasattr(dnn, "DNN_BACKEND_CUDA"):
        try:
            targets = list(dnn.getAvailableTargets(dnn.DNN_BACKEND_CUDA))
        except cv2.error:
            targets = []
        if dnn.DNN_TARGET_CUDA in targets:
            backend, target = dnn.DNN_BACKEND_CUDA, dnn.DNN_TARGET_CUDA if target is None else target
    if backend is not None:
        net.setPreferableBackend(backend)
    if target is not None:
        net.setPreferableTarget(target)
    return net


def inferImage(
    net: _Net,
    src: np.ndarray,
    outputNames: Sequence[str] | None = None,
    size: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """``setInput`` + ``forward`` for every output layer, as a list of arrays.

    ``Net.forward()`` returns only the first output; getting all of them means asking for
    ``getUnconnectedOutLayersNames()`` every time. ``cv.dnn`` also does not accept a plain image.

    Args:
        net: A ``cv.dnn.Net`` (or any object with ``setInput``, ``forward`` and ``getUnconnectedOutLayersNames``).
        src: A ready ``(1, C, H, W)`` blob, or an image which is then passed through :func:`preprocessImage`
            with ``size`` and default options (use :func:`preprocessImage` yourself when you need the
            :class:`PreprocessInfo` to map boxes back).
        outputNames: Layers to return; all unconnected output layers by default.
        size: Network input ``(width, height)``, required when ``src`` is an image.

    Returns:
        One array per output layer.

    Raises:
        ValueError: If ``src`` is an image and ``size`` is missing, or ``src`` is neither an image nor a blob.
    """
    if src.ndim == 4:
        blob = src
    elif src.ndim in (2, 3):
        if size is None:
            raise ValueError("size is required when src is an image instead of a blob")
        blob, _ = preprocessImage(src, size)
    else:
        raise ValueError(f"src must be an image or a 4-D blob, got shape {src.shape}")
    names = list(net.getUnconnectedOutLayersNames()) if outputNames is None else list(outputNames)
    net.setInput(blob)
    outputs = net.forward(names)
    if isinstance(outputs, np.ndarray):
        return [outputs]
    return [np.asarray(o) for o in outputs]


def readClassNames(path: str | os.PathLike[str]) -> list[str]:
    """Read class names from a text file, one per line (``coco.names`` style), skipping blank lines.

    Opened by Python, so Unicode paths and UTF-8 names (with or without BOM) work everywhere.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(path, encoding="utf-8-sig") as file:
        return [line.strip() for line in file if line.strip()]


def drawDetections(
    src: np.ndarray,
    detections: Detections,
    classNames: Sequence[str] | None = None,
    colors: Sequence[Color] | None = None,
    thickness: int = 2,
    fontScale: float = 0.5,
    showScore: bool = True,
    fontPath: str | os.PathLike[str] | None = None,
    fontSize: int = 16,
) -> np.ndarray:
    """Draw :class:`Detections` as labeled boxes, one color per class, in place.

    Args:
        src: 8-bit image to draw on (modified and returned, like ``cv.rectangle``).
        detections: From :func:`postprocessDetections`.
        classNames: Label text per class id; ids without a name are shown as numbers.
        colors: One color per class id; :func:`distinctColors` by default.
        thickness: Box line thickness.
        fontScale: Hershey font size for the labels.
        showScore: Append the score (``"cat 0.87"``).
        fontPath: TrueType font for Unicode class names (see :func:`drawLabel`).
        fontSize: Font size in pixels for ``fontPath``.

    Returns:
        ``src``.

    Raises:
        ValueError: If ``src`` is empty.
    """
    check_image_not_empty("src", src)
    class_ids = np.asarray(detections.classIds).reshape(-1)
    if len(class_ids) == 0:
        return src
    palette: Sequence[Color] = colors if colors is not None else distinctColors(int(class_ids.max()) + 1)
    labels = []
    for class_id, score in zip(class_ids, detections.scores, strict=True):
        name = classNames[class_id] if classNames is not None and 0 <= class_id < len(classNames) else str(class_id)
        labels.append(f"{name} {float(score):.2f}" if showScore else name)
    box_colors = [palette[int(c) % len(palette)] for c in class_ids]
    return drawBoxes(src, detections.boxes, labels, box_colors, thickness, fontScale, fontPath, fontSize)
