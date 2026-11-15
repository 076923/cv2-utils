"""Tracking and motion: object trackers with ids, frame differencing, background subtraction, optical flow helpers and
a motion heat map accumulator."""

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import cv2
import numpy as np

from .._validate import (
    as_color,
    check_channels,
    check_color,
    check_finite,
    check_image_not_empty,
    check_uint8,
    to_uint8,
)
from ..imgproc.color import toGray

__all__ = [
    "MotionAccumulator",
    "MultiTrackerEx",
    "TrackerEx",
    "backgroundMask",
    "createBackgroundSubtractor",
    "denseFlow",
    "drawFlow",
    "flowMagnitude",
    "flowToColor",
    "frameDifference",
    "sparseFlow",
    "trackPoints",
]

Box = tuple[int, int, int, int]
PointFlow = tuple[np.ndarray, np.ndarray]
"""``(points, status)`` of :func:`sparseFlow`: ``(N, 2)`` float32 positions and an ``(N,)`` bool status."""
_TRACKER_NAMES = ("CSRT", "KCF", "MIL", "Nano", "Vit", "DaSiamRPN", "GOTURN", "MOSSE", "Boosting", "MedianFlow", "TLD")


# --- validation ----------------------------------------------------------------------------


def _check_frame(name: str, frame: np.ndarray) -> None:
    check_image_not_empty(name, frame)
    check_channels(name, frame)


def _check_pair(prev: np.ndarray, next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grayscale uint8 versions of two frames of the same size."""
    _check_frame("prev", prev)
    _check_frame("next", next)
    if prev.shape[:2] != next.shape[:2]:
        raise ValueError(f"prev and next must have the same size, got shapes {prev.shape} and {next.shape}")
    return _gray8(prev), _gray8(next)


def _gray8(frame: np.ndarray) -> np.ndarray:
    return to_uint8(toGray(frame))


def _check_box(name: str, box: Sequence[float]) -> Box:
    if len(box) != 4:
        raise ValueError(f"{name} must be (x, y, w, h), got {box!r}")
    values = [float(v) for v in box]
    if not all(math.isfinite(v) for v in values) or values[2] <= 0 or values[3] <= 0:
        raise ValueError(f"{name} must be a finite box with a positive size, got {box!r}")
    return int(values[0]), int(values[1]), int(values[2]), int(values[3])


def _check_flow(name: str, flow: np.ndarray) -> np.ndarray:
    if flow.ndim != 3 or flow.shape[2] != 2 or flow.size == 0:
        raise ValueError(f"{name} must be a non-empty (H, W, 2) flow field, got shape {flow.shape}")
    return flow.astype(np.float32, copy=False)


# --- trackers ------------------------------------------------------------------------------


def _available_trackers() -> list[str]:
    return [n for n in _TRACKER_NAMES if hasattr(cv2, f"Tracker{n}_create") or hasattr(cv2, f"Tracker{n}")]


def _create_tracker(name: str, params: dict[str, Any]) -> Any:
    cls = getattr(cv2, f"Tracker{name}", None)
    factory = getattr(cv2, f"Tracker{name}_create", None) or getattr(cls, "create", None)
    if factory is None:
        raise RuntimeError(f"Tracker{name} is not available in this OpenCV build; available: {_available_trackers()}")
    if not params:
        return factory()
    params_cls = getattr(cv2, f"Tracker{name}_Params", None)
    if params_cls is None:
        raise ValueError(f"Tracker{name} takes no parameters, got {sorted(params)}")
    options = params_cls()
    for key, value in params.items():
        if not hasattr(options, key):
            raise ValueError(
                f"unknown Tracker{name} parameter {key!r}; known: {[a for a in dir(options) if a[0] != '_']}"
            )
        setattr(options, key, value)
    return factory(options)


class TrackerEx:
    """A ``cv.Tracker`` chosen by name, with a remembered box and lost state.

    OpenCV's trackers are created through names that change between versions and builds (``TrackerKCF_create`` in
    4.x, ``TrackerKCF.create`` in others, CSRT/KCF only with opencv-contrib) and return ``(ok, box)`` tuples with
    float boxes. This class resolves the factory, keeps the last box, converts it to ints and yields ``None`` when
    the target is lost.

    Args:
        name: Tracker name without the ``Tracker`` prefix: ``"CSRT"``, ``"KCF"``, ``"MIL"``, ``"Nano"``, ``"Vit"``,
            ``"DaSiamRPN"``...
        **params: Fields of ``cv.Tracker<name>_Params`` (e.g. ``samplerSearchWinSize=30`` for MIL).

    Raises:
        RuntimeError: If the tracker is not available in the installed OpenCV (the message lists the available ones).
        ValueError: If a parameter name is unknown.
    """

    def __init__(self, name: str = "MIL", **params: Any) -> None:
        self.name = name
        self._params = params
        self._tracker = _create_tracker(name, params)
        self._box: Box | None = None
        self._lost = True
        self._initialized = False

    @property
    def box(self) -> Box | None:
        """The last ``(x, y, w, h)`` box, or ``None`` before :meth:`init` or after the target was lost."""
        return self._box

    @property
    def lost(self) -> bool:
        """``True`` when the last :meth:`update` failed (or before :meth:`init`)."""
        return self._lost

    def init(self, frame: np.ndarray, box: Sequence[float]) -> None:
        """Start tracking ``box`` (``x, y, w, h``) in ``frame``.

        Raises:
            ValueError: If ``frame`` is not an 8-bit image or ``box`` has no positive size.
        """
        check_uint8("frame", frame)
        self._box = _check_box("box", box)
        self._tracker = _create_tracker(self.name, self._params)
        self._tracker.init(frame, self._box)
        self._lost = False
        self._initialized = True

    def update(self, frame: np.ndarray) -> Box | None:
        """Track into the next frame and return the new box, or ``None`` when the target is lost.

        Raises:
            RuntimeError: If called before :meth:`init`.
        """
        if not self._initialized:
            raise RuntimeError("TrackerEx.init must be called before update")
        check_uint8("frame", frame)
        ok, box = self._tracker.update(frame)
        if not ok:
            self._lost = True
            self._box = None
            return None
        self._lost = False
        x, y, w, h = (round(float(v)) for v in box)
        self._box = (x, y, w, h)
        return self._box

    def track(self, frames: Iterable[np.ndarray]) -> Iterator[Box | None]:
        """Yield the box for every frame of ``frames`` (``None`` once the target is lost), stopping at the end."""
        for frame in frames:
            yield self.update(frame)

    def __repr__(self) -> str:
        return f"TrackerEx({self.name!r}, box={self._box})"


class MultiTrackerEx:
    """Several :class:`TrackerEx` with stable integer ids.

    ``cv.legacy.MultiTracker`` is contrib-only, has no ids and cannot remove a target. This class assigns an id to
    every added box and updates all targets in one call.

    Args:
        name: Tracker name passed to :class:`TrackerEx`.
        **params: Tracker parameters passed to :class:`TrackerEx`.
    """

    def __init__(self, name: str = "MIL", **params: Any) -> None:
        self.name = name
        self._params = params
        self._trackers: dict[int, TrackerEx] = {}
        self._next_id = 0

    @property
    def boxes(self) -> dict[int, Box | None]:
        """Current boxes by id (``None`` for lost targets)."""
        return {i: t.box for i, t in self._trackers.items()}

    @property
    def ids(self) -> list[int]:
        """Ids of the tracked targets."""
        return list(self._trackers)

    def add(self, frame: np.ndarray, box: Sequence[float]) -> int:
        """Start tracking ``box`` in ``frame`` and return its id."""
        tracker = TrackerEx(self.name, **self._params)
        tracker.init(frame, box)
        target = self._next_id
        self._trackers[target] = tracker
        self._next_id += 1
        return target

    def remove(self, target: int) -> None:
        """Stop tracking ``target``.

        Raises:
            KeyError: If the id is unknown.
        """
        del self._trackers[target]

    def update(self, frame: np.ndarray) -> dict[int, Box | None]:
        """Update every target and return ``{id: box or None}``."""
        return {i: t.update(frame) for i, t in self._trackers.items()}

    def __len__(self) -> int:
        return len(self._trackers)

    def __repr__(self) -> str:
        return f"MultiTrackerEx({self.name!r}, targets={len(self._trackers)})"


# --- motion masks --------------------------------------------------------------------------


def frameDifference(prev: np.ndarray, next: np.ndarray, threshold: float = 25, *, blur: int = 0) -> np.ndarray:
    """Binary mask (0/255) of the pixels that changed between two frames.

    ``cv.absdiff`` needs grayscale conversion, an optional blur and a threshold around it; this does the three steps
    for frames of any channel count.

    Args:
        prev: First frame (grayscale, BGR or BGRA).
        next: Second frame of the same size.
        threshold: Minimum gray-level difference (0..255) counted as motion.
        blur: Odd Gaussian kernel size applied before differencing (``0`` for none) to suppress sensor noise.

    Returns:
        ``uint8`` mask with 255 where the frames differ.

    Raises:
        ValueError: If the frames differ in size, ``threshold`` is outside ``[0, 255]`` or ``blur`` is even.
    """
    check_finite("threshold", threshold, nonnegative=True)
    if threshold > 255:
        raise ValueError(f"threshold must be between 0 and 255, got {threshold}")
    if blur < 0 or (blur % 2 == 0 and blur != 0):
        raise ValueError(f"blur must be 0 or an odd kernel size, got {blur}")
    a, b = _check_pair(prev, next)
    if blur:
        a = cv2.GaussianBlur(a, (blur, blur), 0)
        b = cv2.GaussianBlur(b, (blur, blur), 0)
    _, mask = cv2.threshold(cv2.absdiff(a, b), threshold, 255, cv2.THRESH_BINARY)
    out: np.ndarray = mask
    return out


def createBackgroundSubtractor(
    method: str = "MOG2", history: int = 500, threshold: float | None = None, *, detectShadows: bool = True
) -> Any:
    """Create a MOG2 or KNN background subtractor by name with validated arguments.

    ``cv.createBackgroundSubtractorMOG2`` and ``...KNN`` name their threshold differently (``varThreshold`` vs
    ``dist2Threshold``, with defaults 16 and 400) and accept nonsense silently. This factory takes one ``threshold``
    argument, uses the right default for each method and rejects invalid values.

    Args:
        method: ``"MOG2"`` or ``"KNN"`` (case-insensitive).
        history: Number of frames that affect the background model.
        threshold: ``varThreshold`` for MOG2 (default 16) or ``dist2Threshold`` for KNN (default 400).
        detectShadows: Mark shadows with 127 in the mask (see :func:`backgroundMask` to drop them).

    Returns:
        A ``cv.BackgroundSubtractorMOG2`` or ``cv.BackgroundSubtractorKNN``.

    Raises:
        ValueError: If ``method`` is unknown, ``history < 1`` or ``threshold`` is not positive.
    """
    key = method.upper()
    if key not in ("MOG2", "KNN"):
        raise ValueError(f"method must be 'MOG2' or 'KNN', got {method!r}")
    if history < 1:
        raise ValueError(f"history must be at least 1, got {history}")
    if threshold is None:
        threshold = 16.0 if key == "MOG2" else 400.0
    check_finite("threshold", threshold, positive=True)
    if key == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=threshold, detectShadows=detectShadows)
    return cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=threshold, detectShadows=detectShadows)


def backgroundMask(
    subtractor: Any, frame: np.ndarray, *, learningRate: float = -1, shadows: bool = False, minArea: int = 0
) -> np.ndarray:
    """Foreground mask (0/255) from a background subtractor, without shadows and tiny blobs.

    ``subtractor.apply`` returns a mask with 127 for shadows and speckles from noise that need a threshold and a
    connected-components pass before the mask is usable.

    Args:
        subtractor: A ``cv.BackgroundSubtractor`` (see :func:`createBackgroundSubtractor`).
        frame: Next frame (grayscale, BGR or BGRA).
        learningRate: Model update rate (``-1`` for automatic, ``0`` to freeze the background, ``1`` to reset).
        shadows: Keep shadow pixels (127) as foreground.
        minArea: Remove connected blobs with fewer pixels than this.

    Returns:
        ``uint8`` mask with 255 on foreground pixels.

    Raises:
        ValueError: If ``learningRate`` is outside ``[-1, 1]`` or ``minArea`` is negative.
    """
    check_finite("learningRate", learningRate)
    if not -1 <= learningRate <= 1:
        raise ValueError(f"learningRate must be between -1 and 1, got {learningRate}")
    if minArea < 0:
        raise ValueError(f"minArea must not be negative, got {minArea}")
    _check_frame("frame", frame)
    raw = subtractor.apply(frame, learningRate=learningRate)
    _, mask = cv2.threshold(raw, 100 if shadows else 200, 255, cv2.THRESH_BINARY)
    if minArea > 1:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= minArea
        keep[0] = False
        mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    out: np.ndarray = mask
    return out


# --- optical flow --------------------------------------------------------------------------


def denseFlow(prev: np.ndarray, next: np.ndarray, method: str = "farneback", **kw: Any) -> np.ndarray:
    """Dense optical flow ``(H, W, 2)`` float32 between two frames of any channel count.

    ``cv.calcOpticalFlowFarneback`` needs grayscale input, a ``None`` placeholder and six positional parameters;
    ``cv.DISOpticalFlow`` needs an object and a preset constant. This converts the frames and picks the method by
    name.

    Args:
        prev: First frame.
        next: Second frame of the same size.
        method: ``"farneback"`` or ``"dis"``.
        **kw: Farneback: ``pyrScale``, ``levels``, ``winsize``, ``iterations``, ``polyN``, ``polySigma``, ``flags``.
            DIS: ``preset`` (``"ultrafast"``, ``"fast"``, ``"medium"``).

    Returns:
        Flow field where ``flow[y, x] = (dx, dy)``.

    Raises:
        ValueError: If ``method`` or a keyword is unknown.
        RuntimeError: If ``"dis"`` is requested and the OpenCV build lacks ``DISOpticalFlow``.
    """
    key = method.lower()
    a, b = _check_pair(prev, next)
    if key == "farneback":
        options = {
            "pyrScale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "polyN": 5,
            "polySigma": 1.2,
            "flags": 0,
        }
        _check_keywords("farneback", kw, options)
        options.update(kw)
        flow: np.ndarray = cv2.calcOpticalFlowFarneback(a, b, None, *options.values())  # type: ignore[call-overload]
        return flow
    if key == "dis":
        _check_keywords("dis", kw, {"preset": "medium"})
        factory = getattr(cv2, "DISOpticalFlow_create", None) or getattr(
            getattr(cv2, "DISOpticalFlow", None), "create", None
        )
        if factory is None:
            raise RuntimeError("DISOpticalFlow is not available in this OpenCV build")
        presets = {
            "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }
        preset = str(kw.get("preset", "medium")).lower()
        if preset not in presets:
            raise ValueError(f"preset must be one of {list(presets)}, got {kw.get('preset')!r}")
        dis_flow: np.ndarray = factory(presets[preset]).calc(a, b, None)
        return dis_flow
    raise ValueError(f"method must be 'farneback' or 'dis', got {method!r}")


def _check_keywords(method: str, kw: dict[str, Any], known: dict[str, Any]) -> None:
    unknown = sorted(set(kw) - set(known))
    if unknown:
        raise ValueError(f"unknown {method} keyword(s) {unknown}; known: {list(known)}")


def sparseFlow(
    prev: np.ndarray,
    next: np.ndarray,
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    winSize: tuple[int, int] = (21, 21),
    maxLevel: int = 3,
) -> PointFlow:
    """Track ``(N, 2)`` points from ``prev`` to ``next`` with pyramidal Lucas-Kanade.

    ``cv.calcOpticalFlowPyrLK`` wants ``(N, 1, 2)`` float32 points, returns ``(N, 1)`` uint8 status and errors, and
    fails on an empty point list. This takes any ``(N, 2)`` sequence, converts the frames and returns flat arrays.

    Args:
        prev: First frame.
        next: Second frame of the same size.
        points: ``(N, 2)`` point positions in ``prev`` (``(x, y)``).
        winSize: Search window ``(width, height)``.
        maxLevel: Number of pyramid levels (``0`` for a single level).

    Returns:
        ``(points, status)``: the new positions ``(N, 2)`` float32 (unchanged where not found) and a bool status
        ``(N,)`` that is ``True`` where the point was found.

    Raises:
        ValueError: If ``points`` is not ``(N, 2)``, ``winSize`` is smaller than 3 or ``maxLevel`` is negative.
    """
    a, b = _check_pair(prev, next)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2) if np.size(points) else np.zeros((0, 2), np.float32)
    if pts.size and not np.all(np.isfinite(pts)):
        raise ValueError("points must be finite numbers")
    if len(winSize) != 2 or min(winSize) < 3:
        raise ValueError(f"winSize must be (width, height) of at least 3, got {winSize!r}")
    if maxLevel < 0:
        raise ValueError(f"maxLevel must not be negative, got {maxLevel}")
    if pts.shape[0] == 0:
        return pts, np.zeros(0, bool)
    new, status, _ = cv2.calcOpticalFlowPyrLK(
        a,
        b,
        pts.reshape(-1, 1, 2),
        np.empty(0, np.float32),
        winSize=(int(winSize[0]), int(winSize[1])),
        maxLevel=maxLevel,
    )
    found = status.reshape(-1).astype(bool)
    new = new.reshape(-1, 2).astype(np.float32)
    new[~found] = pts[~found]
    return new, found


def trackPoints(
    frames: Iterable[np.ndarray], points: Sequence[Sequence[float]] | np.ndarray, **kw: Any
) -> Iterator[PointFlow]:
    """Follow ``points`` through a frame iterable with :func:`sparseFlow`, dropping the ones that are lost.

    The first frame is the reference; for every later frame ``(points, status)`` is yielded, where ``points`` are
    the positions of the still-tracked points and ``status`` says which of the *previous* points survived.

    Args:
        frames: Frame iterable (the first frame is where ``points`` are located).
        points: ``(N, 2)`` initial positions.
        **kw: ``winSize`` and ``maxLevel`` for :func:`sparseFlow`.

    Yields:
        ``(points, status)`` per frame after the first.
    """
    iterator = iter(frames)
    prev = next(iterator, None)
    if prev is None:
        return
    current = np.asarray(points, dtype=np.float32).reshape(-1, 2) if np.size(points) else np.zeros((0, 2), np.float32)
    for frame in iterator:
        moved, status = sparseFlow(prev, frame, current, **kw)
        current = moved[status]
        prev = frame
        yield current, status


def flowMagnitude(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel magnitude and direction (degrees, ``0..360``) of a flow field.

    ``cv.cartToPolar`` takes the two components separately and returns radians by default.

    Args:
        flow: ``(H, W, 2)`` flow field.

    Returns:
        ``(magnitude, angleDeg)``, both ``(H, W)`` float32.

    Raises:
        ValueError: If ``flow`` is not ``(H, W, 2)``.
    """
    flow = _check_flow("flow", flow)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    return magnitude, angle


def flowToColor(flow: np.ndarray, *, maxMagnitude: float | None = None) -> np.ndarray:
    """Visualize a flow field as a BGR image: hue is the direction, brightness the magnitude.

    The standard HSV flow coloring takes ``cartToPolar``, a hue scaling by 2 (OpenCV hue is 0..180), a
    normalization and ``cvtColor``; this does it in one call with an optional fixed scale so frames of a video share
    one color mapping.

    Args:
        flow: ``(H, W, 2)`` flow field.
        maxMagnitude: Magnitude mapped to full brightness (``None`` for the maximum in ``flow``).

    Returns:
        ``uint8`` BGR image of the flow's size.

    Raises:
        ValueError: If ``flow`` is not ``(H, W, 2)`` or ``maxMagnitude`` is not positive.
    """
    magnitude, angle = flowMagnitude(flow)
    if maxMagnitude is None:
        peak = float(magnitude.max())
        maxMagnitude = peak if peak > 0 else 1.0
    check_finite("maxMagnitude", maxMagnitude, positive=True)
    hsv = np.zeros((*magnitude.shape, 3), np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude * (255.0 / maxMagnitude), 0, 255).astype(np.uint8)
    out: np.ndarray = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out


def drawFlow(
    src: np.ndarray,
    flow: np.ndarray,
    step: int = 16,
    color: float | Sequence[float] = (0, 255, 0),
    scale: float = 1.0,
    *,
    thickness: int = 1,
) -> np.ndarray:
    """Draw a flow field as a grid of arrows on ``src`` (in place) and return it.

    OpenCV has no flow drawing function; the usual snippet samples the field on a grid and calls
    ``cv.arrowedLine`` per cell.

    Args:
        src: Image of the flow's size (8-bit).
        flow: ``(H, W, 2)`` flow field.
        step: Grid spacing in pixels.
        color: Arrow color (BGR, or a gray level for grayscale images).
        scale: Multiplier applied to the vectors before drawing.
        thickness: Line thickness.

    Returns:
        ``src``.

    Raises:
        ValueError: If the sizes differ, ``step < 1``, ``scale`` is not finite or ``thickness < 1``.
    """
    channels = check_uint8("src", src)
    flow = _check_flow("flow", flow)
    if src.shape[:2] != flow.shape[:2]:
        raise ValueError(f"src and flow must have the same size, got shapes {src.shape} and {flow.shape}")
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")
    check_finite("scale", scale)
    if thickness < 1:
        raise ValueError(f"thickness must be at least 1, got {thickness}")
    values = [float(color)] if isinstance(color, (int, float)) else [float(c) for c in color]
    check_color("color", values, src.dtype)
    draw_color = as_color(values[0] if channels == 1 and len(values) > 1 else color)
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = flow[y, x]
            end = (round(float(x + dx * scale)), round(float(y + dy * scale)))
            cv2.arrowedLine(src, (x, y), end, draw_color, thickness, cv2.LINE_AA, tipLength=0.3)
    return src


# --- heat map ------------------------------------------------------------------------------


class MotionAccumulator:
    """Exponentially decaying accumulation of motion masks into a ``0..1`` heat map.

    ``cv.accumulateWeighted`` needs a float accumulator of the right shape, a mask in ``0..1`` and a normalization
    before display. This class owns the buffer: feed it masks (from :func:`frameDifference` or
    :func:`backgroundMask`) and read :meth:`heatmap`.

    Args:
        shape: ``(height, width)`` of the masks.
        decay: Weight of the previous accumulation per update (``0.95`` keeps motion visible for a while, ``0``
            shows only the last mask).

    Raises:
        ValueError: If ``decay`` is outside ``[0, 1]`` or ``shape`` is not two positive ints.
    """

    def __init__(self, shape: Sequence[int], decay: float = 0.95) -> None:
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
            raise ValueError(f"shape must be (height, width) with positive values, got {shape!r}")
        check_finite("decay", decay)
        if not 0 <= decay <= 1:
            raise ValueError(f"decay must be between 0 and 1, got {decay}")
        self.decay = decay
        self._acc = np.zeros((int(shape[0]), int(shape[1])), np.float32)
        self._count = 0

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` of the accumulator."""
        return self._acc.shape[0], self._acc.shape[1]

    @property
    def count(self) -> int:
        """Number of masks accumulated so far."""
        return self._count

    def update(self, mask: np.ndarray) -> np.ndarray:
        """Add a mask (``uint8`` 0/255, bool or float ``0..1``) and return the raw accumulator.

        Raises:
            ValueError: If ``mask`` does not match :attr:`shape`.
        """
        if mask.shape[:2] != self._acc.shape:
            raise ValueError(f"mask must have shape {self._acc.shape}, got {mask.shape}")
        values = toGray(mask) if mask.ndim == 3 else mask
        if values.dtype == np.uint8:
            unit = values.astype(np.float32) / 255.0
        else:
            unit = np.clip(values.astype(np.float32), 0, 1)
        self._acc = self._acc * self.decay + unit
        self._count += 1
        return self._acc

    def heatmap(self) -> np.ndarray:
        """Accumulator normalized to ``0..1`` float32 (all zeros before any motion)."""
        peak = float(np.max(self._acc))
        out: np.ndarray = self._acc / peak if peak > 0 else self._acc.copy()
        return out

    def reset(self) -> None:
        """Clear the accumulation."""
        self._acc[:] = 0
        self._count = 0

    def __repr__(self) -> str:
        return f"MotionAccumulator(shape={self.shape}, decay={self.decay}, count={self._count})"
