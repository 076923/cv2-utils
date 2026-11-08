"""Camera calibration with a chessboard: corner detection, calibration, undistortion and a JSON file format."""

import json
import os
from collections.abc import Sequence
from typing import Any, NamedTuple

import cv2
import numpy as np

from .._validate import check_finite, check_image_not_empty

__all__ = [
    "CameraCalibration",
    "calibrateCameraEx",
    "findChessboardEx",
    "loadCalibration",
    "makeChessboard",
    "objectPointsGrid",
    "reprojectionErrors",
    "saveCalibration",
    "undistortEx",
]


class CameraCalibration(NamedTuple):
    """The result of :func:`calibrateCameraEx` (everything ``cv.calibrateCamera`` returns, plus the image size)."""

    cameraMatrix: np.ndarray
    """Intrinsic matrix, shape ``(3, 3)`` float64."""
    distCoeffs: np.ndarray
    """Distortion coefficients ``(k1, k2, p1, p2, k3, ...)``, shape ``(1, N)`` float64."""
    imageSize: tuple[int, int]
    """Size of the calibration images as ``(width, height)``."""
    rms: float
    """Overall RMS reprojection error in pixels (below 1.0 is a good calibration)."""
    rvecs: tuple[np.ndarray, ...]
    """Rotation vector of the board in each view, shape ``(3, 1)`` each."""
    tvecs: tuple[np.ndarray, ...]
    """Translation vector of the board in each view, shape ``(3, 1)`` each."""
    usedViews: tuple[int, ...] = ()
    """Indices into the ``views`` given to :func:`calibrateCameraEx` of the views where the board was found;
    ``rvecs[i]`` / ``tvecs[i]`` belong to ``views[usedViews[i]]``. Empty for calibrations loaded from files
    written before this field existed."""


def _check_pattern(patternSize: tuple[int, int]) -> tuple[int, int]:
    cols, rows = int(patternSize[0]), int(patternSize[1])
    if cols < 2 or rows < 2:
        raise ValueError(f"patternSize must have at least 2 inner corners per side, got {patternSize}")
    return cols, rows


def objectPointsGrid(patternSize: tuple[int, int], squareSize: float = 1.0) -> np.ndarray:
    """Return the 3D coordinates of the inner chessboard corners, lying in the ``z = 0`` plane.

    The ``objp = np.zeros((cols * rows, 3)); objp[:, :2] = np.mgrid[...].T.reshape(-1, 2)`` boilerplate from the
    OpenCV tutorial, in the row-major order ``cv.findChessboardCorners`` returns the corners in.

    Args:
        patternSize: Inner corners per row and column as ``(columns, rows)``, like ``cv.findChessboardCorners``.
        squareSize: Side of one square in your unit (mm, cm, ...); the translations come out in the same unit.

    Returns:
        Points of shape ``(columns * rows, 3)`` float32.

    Raises:
        ValueError: If ``patternSize`` has fewer than 2 corners per side or ``squareSize`` is not positive.
    """
    cols, rows = _check_pattern(patternSize)
    check_finite("squareSize", squareSize, positive=True)
    grid = np.zeros((cols * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * squareSize
    return grid


def makeChessboard(patternSize: tuple[int, int], squareSize: int = 50, border: int = 1) -> np.ndarray:
    """Render a chessboard image with the given number of inner corners, for printing or for tests.

    OpenCV detects a board by its *inner* corners, so a board with ``(7, 6)`` inner corners has 8 x 7 squares.
    The top-left square is black.

    Args:
        patternSize: Inner corners as ``(columns, rows)``, the value to pass to :func:`findChessboardEx` later.
        squareSize: Square side in pixels.
        border: White margin around the board, in squares (the detector needs some white around the board).

    Returns:
        8-bit grayscale image with ``columns + 1 + 2 * border`` by ``rows + 1 + 2 * border`` squares of
        ``squareSize`` pixels.

    Raises:
        ValueError: If ``patternSize`` has fewer than 2 corners per side, ``squareSize`` is not positive or ``border``
            is negative.
    """
    cols, rows = _check_pattern(patternSize)
    if squareSize < 1:
        raise ValueError(f"squareSize must be positive, got {squareSize}")
    if border < 0:
        raise ValueError(f"border must not be negative, got {border}")
    squares_x, squares_y = cols + 1, rows + 1
    cells = (np.indices((squares_y, squares_x)).sum(axis=0) % 2 * 255).astype(np.uint8)
    board = np.kron(cells, np.ones((squareSize, squareSize), np.uint8))
    margin = border * squareSize
    return cv2.copyMakeBorder(board, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)


def _gray(src: np.ndarray) -> np.ndarray:
    if src.ndim == 2:
        return src
    if src.shape[2] == 4:
        return cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


def findChessboardEx(
    src: np.ndarray, patternSize: tuple[int, int], refine: bool = True, flags: int | None = None
) -> np.ndarray | None:
    """Find and refine the inner corners of a chessboard in one call.

    ``cv.findChessboardCorners`` returns ``(found, corners)`` with ``corners`` of shape ``(N, 1, 2)`` that still
    need ``cv.cornerSubPix`` on a grayscale copy for sub-pixel accuracy. This converts to grayscale, uses the
    more robust ``cv.findChessboardCornersSB`` when the build has it, refines, and returns a flat point array.

    Args:
        src: Grayscale or BGR(A) image.
        patternSize: Inner corners as ``(columns, rows)``.
        refine: Run ``cv.cornerSubPix`` on the corners (not needed after ``findChessboardCornersSB``, which is
            already sub-pixel accurate, but harmless).
        flags: ``cv.CALIB_CB_*`` flags; ``None`` picks the detector's default.

    Returns:
        The corners as ``(columns * rows, 2)`` float32, or ``None`` if the board was not found.

    Raises:
        ValueError: If ``src`` is empty or ``patternSize`` is invalid.
    """
    check_image_not_empty("src", src)
    cols, rows = _check_pattern(patternSize)
    gray = _gray(src)
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags or 0)
    else:  # pragma: no cover - OpenCV < 4.0
        found, corners = cv2.findChessboardCorners(
            gray,
            (cols, rows),
            flags=flags if flags is not None else cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
    if not found or corners is None:
        return None
    corners = np.ascontiguousarray(corners, np.float32).reshape(-1, 1, 2)
    if refine:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return corners.reshape(-1, 2)


def _as_corners(cornersList: Sequence[np.ndarray], count: int) -> list[np.ndarray]:
    views = []
    for index, corners in enumerate(cornersList):
        points = np.asarray(corners, np.float32).reshape(-1, 2)
        if len(points) != count:
            raise ValueError(f"view {index} has {len(points)} corners, expected {count} for the pattern size")
        views.append(points.reshape(-1, 1, 2))
    return views


def calibrateCameraEx(
    views: Sequence[np.ndarray],
    patternSize: tuple[int, int],
    squareSize: float = 1.0,
    imageSize: tuple[int, int] | None = None,
    flags: int = 0,
) -> CameraCalibration:
    """Calibrate a camera from chessboard images (or already detected corners).

    The whole tutorial loop: detect the board in every image with :func:`findChessboardEx` (views where it is not
    found are skipped and listed by their absence in ``usedViews``), build the object points with
    :func:`objectPointsGrid`, and run ``cv.calibrateCamera``.

    Args:
        views: Images of the chessboard, or corner arrays of shape ``(N, 2)`` such as :func:`findChessboardEx`
            returns (then ``imageSize`` is required). Mixing the two is not supported.
        patternSize: Inner corners as ``(columns, rows)``.
        squareSize: Side of one square in your unit; ``tvecs`` come out in the same unit.
        imageSize: ``(width, height)`` of the images, needed only when ``views`` are corner arrays.
        flags: ``cv.CALIB_*`` flags for ``cv.calibrateCamera``.

    Returns:
        A :class:`CameraCalibration`. Its ``rvecs`` / ``tvecs`` have one entry per *used* view, so when a board was
        missed they are shorter than ``views``; ``usedViews`` maps them back to the input indices.

    Raises:
        ValueError: If fewer than 3 views contain the board, ``imageSize`` is missing for corner input, a corner
            array has the wrong number of points, or ``patternSize`` / ``squareSize`` are invalid.
    """
    cols, rows = _check_pattern(patternSize)
    objp = objectPointsGrid((cols, rows), squareSize)
    if len(views) == 0:
        raise ValueError("views must not be empty")
    first = np.asarray(views[0])
    is_corners = (first.ndim == 2 and first.shape[1] == 2) or (first.ndim == 3 and first.shape[1:] == (1, 2))
    if not is_corners:
        height, width = views[0].shape[:2]
        size = (width, height)
        found = [(i, findChessboardEx(view, (cols, rows))) for i, view in enumerate(views)]
        used = [i for i, c in found if c is not None]
        corners_list = [c for _, c in found if c is not None]
    else:
        if imageSize is None:
            raise ValueError("imageSize is required when views are corner arrays")
        size = (int(imageSize[0]), int(imageSize[1]))
        used = list(range(len(views)))
        corners_list = list(views)
    if len(corners_list) < 3:
        raise ValueError(f"the chessboard was found in only {len(corners_list)} view(s), need at least 3")
    image_points = _as_corners(corners_list, cols * rows)
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [objp] * len(image_points), image_points, size, None, None, flags=flags
    )
    return CameraCalibration(
        np.asarray(camera_matrix, np.float64),
        np.asarray(dist_coeffs, np.float64).reshape(1, -1),
        size,
        float(rms),
        tuple(np.asarray(r, np.float64).reshape(3, 1) for r in rvecs),
        tuple(np.asarray(t, np.float64).reshape(3, 1) for t in tvecs),
        tuple(used),
    )


def reprojectionErrors(
    calib: CameraCalibration,
    patternSize: tuple[int, int],
    squareSize: float,
    cornersList: Sequence[np.ndarray | None],
) -> np.ndarray:
    """Return the RMS reprojection error of every view, to spot the bad images in a calibration set.

    ``cv.calibrateCamera`` reports a single overall error; per-view errors need ``cv.projectPoints`` and
    ``cv.norm`` in a loop. Views with an error well above the others are usually blurred or mis-detected: drop
    them and calibrate again.

    Args:
        calib: The calibration whose ``rvecs`` / ``tvecs`` belong to ``cornersList``.
        patternSize: Inner corners as ``(columns, rows)``.
        squareSize: Square size used for the calibration.
        cornersList: The detected corners of each view, ``(N, 2)`` each: either one per used view (same order as
            ``calib.rvecs``), or one per *original* view given to :func:`calibrateCameraEx` (then the views listed
            in ``calib.usedViews`` are picked, so the output of :func:`findChessboardEx` for every image can be
            passed as is, ``None`` entries included).

    Returns:
        One RMS error in pixels per used view, shape ``(len(calib.rvecs),)`` float64.

    Raises:
        ValueError: If the number of views matches neither ``calib.rvecs`` nor ``calib.usedViews``, a selected
            entry is ``None``, or a corner array has the wrong size.
    """
    cols, rows = _check_pattern(patternSize)
    objp = objectPointsGrid((cols, rows), squareSize)
    count = len(calib.rvecs)
    used = calib.usedViews
    if len(cornersList) == count:
        selected = list(cornersList)
    elif len(used) == count and len(cornersList) > max(used, default=-1):
        selected = [cornersList[i] for i in used]
    else:
        raise ValueError(f"cornersList has {len(cornersList)} views, but calib has {count} (usedViews: {len(used)})")
    for index, corners in enumerate(selected):
        if corners is None:
            raise ValueError(f"view {index} has no corners (None), but it was used for the calibration")
    errors = []
    views = _as_corners([c for c in selected if c is not None], cols * rows)
    for corners, rvec, tvec in zip(views, calib.rvecs, calib.tvecs, strict=True):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, calib.cameraMatrix, calib.distCoeffs)
        diff = projected.reshape(-1, 2) - corners.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))))
    return np.array(errors, np.float64)


def undistortEx(
    src: np.ndarray,
    calib: CameraCalibration | np.ndarray,
    distCoeffs: np.ndarray | None = None,
    alpha: float = 0.0,
    crop: bool = True,
) -> np.ndarray:
    """Undistort an image with ``cv.getOptimalNewCameraMatrix`` + ``cv.undistort`` and crop the valid area.

    ``cv.undistort`` alone keeps the original camera matrix, which leaves black curved borders or cuts off
    content; the tutorial recipe computes a new matrix and crops to the returned ROI. This does both.

    Args:
        src: Image taken with the calibrated camera (any channel count).
        calib: A :class:`CameraCalibration`, or the ``(3, 3)`` camera matrix (then pass ``distCoeffs``).
        distCoeffs: Distortion coefficients when ``calib`` is a matrix.
        alpha: Free scaling parameter of ``cv.getOptimalNewCameraMatrix``: ``0`` keeps only valid pixels (no black
            corners), ``1`` keeps every source pixel (black corners).
        crop: Crop to the valid ROI. With ``False`` the result has the size of ``src``.

    Returns:
        The undistorted image, same dtype and channels as ``src``.

    Raises:
        ValueError: If ``src`` is empty, ``distCoeffs`` is missing for a matrix input, ``alpha`` is not in
            ``[0, 1]``, or ``src`` has a different size than the calibration images (the matrix would not apply).
    """
    check_image_not_empty("src", src)
    check_finite("alpha", alpha, nonnegative=True)
    if alpha > 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    height, width = src.shape[:2]
    if isinstance(calib, CameraCalibration):
        camera_matrix, coeffs = calib.cameraMatrix, calib.distCoeffs
        if calib.imageSize != (width, height):
            raise ValueError(f"src has size {(width, height)}, but calib is for {calib.imageSize}")
    else:
        if distCoeffs is None:
            raise ValueError("distCoeffs is required when calib is a camera matrix")
        camera_matrix, coeffs = np.asarray(calib, np.float64), np.asarray(distCoeffs, np.float64)
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, coeffs, (width, height), alpha, (width, height))
    result: np.ndarray = cv2.undistort(src, camera_matrix, coeffs, None, new_matrix)
    if crop:
        x, y, w, h = (int(v) for v in roi)
        if w > 0 and h > 0:
            result = result[y : y + h, x : x + w]
    return result


def saveCalibration(filename: str | os.PathLike[str], calib: CameraCalibration) -> None:
    """Save a calibration as a JSON file.

    OpenCV's ``cv.FileStorage`` writes YAML/XML and cannot open non-ASCII paths on Windows; JSON is readable
    everywhere and works with any path. Keys: ``cameraMatrix``, ``distCoeffs``, ``imageSize``, ``rms``,
    ``rvecs``, ``tvecs``, ``usedViews``.

    Args:
        filename: Path to write to (``.json``).
        calib: The calibration to save.
    """
    data: dict[str, Any] = {
        "cameraMatrix": calib.cameraMatrix.tolist(),
        "distCoeffs": calib.distCoeffs.reshape(-1).tolist(),
        "imageSize": list(calib.imageSize),
        "rms": calib.rms,
        "rvecs": [r.reshape(-1).tolist() for r in calib.rvecs],
        "tvecs": [t.reshape(-1).tolist() for t in calib.tvecs],
        "usedViews": list(calib.usedViews),
    }
    with open(os.fspath(filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def loadCalibration(filename: str | os.PathLike[str]) -> CameraCalibration:
    """Load a calibration written by :func:`saveCalibration`.

    Args:
        filename: Path of the JSON file.

    Returns:
        The :class:`CameraCalibration`.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is not valid JSON or lacks a required key.
    """
    with open(os.fspath(filename), encoding="utf-8") as f:
        data = json.load(f)
    try:
        return CameraCalibration(
            np.asarray(data["cameraMatrix"], np.float64).reshape(3, 3),
            np.asarray(data["distCoeffs"], np.float64).reshape(1, -1),
            (int(data["imageSize"][0]), int(data["imageSize"][1])),
            float(data["rms"]),
            tuple(np.asarray(r, np.float64).reshape(3, 1) for r in data.get("rvecs", [])),
            tuple(np.asarray(t, np.float64).reshape(3, 1) for t in data.get("tvecs", [])),
            tuple(int(i) for i in data.get("usedViews", [])),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"not a calibration file: {os.fspath(filename)!r} ({exc})") from exc
