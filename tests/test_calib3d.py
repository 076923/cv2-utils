import json

import cv2
import numpy as np
import pytest

from cv2u.calib3d.calibrate import (
    CameraCalibration,
    calibrateCameraEx,
    findChessboardEx,
    loadCalibration,
    makeChessboard,
    objectPointsGrid,
    reprojectionErrors,
    saveCalibration,
    undistortEx,
)

PATTERN = (6, 5)
SQUARE = 30
SIZE = (400, 320)


TRUE_K = np.array([[420.0, 0.0, 200.0], [0.0, 420.0, 160.0], [0.0, 0.0, 1.0]])


def _views(count: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Render ``count`` views of a synthetic board as seen by a pinhole camera with TRUE_K and no distortion."""
    board = makeChessboard(PATTERN, SQUARE, border=1)
    h, w = board.shape
    rng = np.random.default_rng(seed)
    # board image corners in world units (1 world unit = 1 board pixel), centered on the board
    corners = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], np.float64) - [w / 2, h / 2, 0]
    views = []
    for _ in range(count):
        rvec = rng.uniform(-0.35, 0.35, 3)
        tvec = np.array([rng.uniform(-30, 30), rng.uniform(-25, 25), rng.uniform(380, 460)])
        projected, _ = cv2.projectPoints(corners, rvec, tvec, TRUE_K, None)
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        matrix = cv2.getPerspectiveTransform(src, projected.reshape(-1, 2).astype(np.float32))
        view = cv2.warpPerspective(board, matrix, SIZE, flags=cv2.INTER_LINEAR, borderValue=255)
        views.append(view)
    return views


@pytest.fixture(scope="module")
def views() -> list[np.ndarray]:
    return _views()


@pytest.fixture(scope="module")
def calib(views: list[np.ndarray]) -> CameraCalibration:
    return calibrateCameraEx(views, PATTERN, SQUARE)


def test_object_points_grid() -> None:
    grid = objectPointsGrid((3, 2), 10)
    assert grid.shape == (6, 3) and grid.dtype == np.float32
    assert grid[:, 2].max() == 0
    assert grid[1].tolist() == [10, 0, 0] and grid[3].tolist() == [0, 10, 0]
    with pytest.raises(ValueError, match="patternSize"):
        objectPointsGrid((1, 5))
    with pytest.raises(ValueError, match="squareSize"):
        objectPointsGrid((3, 3), 0)


def test_make_chessboard() -> None:
    board = makeChessboard((7, 6), 10, border=2)
    assert board.shape == ((6 + 1 + 4) * 10, (7 + 1 + 4) * 10) and board.dtype == np.uint8
    assert board[20, 20] == 0 and board[20, 30] == 255 and board[0, 0] == 255
    assert makeChessboard((2, 2), 5, border=0).shape == (15, 15)
    with pytest.raises(ValueError, match="squareSize"):
        makeChessboard((3, 3), 0)
    with pytest.raises(ValueError, match="border"):
        makeChessboard((3, 3), 5, border=-1)


def test_find_chessboard(views: list[np.ndarray]) -> None:
    corners = findChessboardEx(views[0], PATTERN)
    assert corners is not None and corners.shape == (30, 2) and corners.dtype == np.float32
    color = cv2.cvtColor(views[0], cv2.COLOR_GRAY2BGR)
    assert findChessboardEx(color, PATTERN, refine=False) is not None
    bgra = cv2.cvtColor(views[0], cv2.COLOR_GRAY2BGRA)
    assert findChessboardEx(bgra, PATTERN, flags=0) is not None
    assert findChessboardEx(np.full((100, 100), 128, np.uint8), PATTERN) is None
    with pytest.raises(ValueError, match="src"):
        findChessboardEx(np.zeros((0, 0), np.uint8), PATTERN)
    with pytest.raises(ValueError, match="patternSize"):
        findChessboardEx(views[0], (6, 1))


def test_calibrate_from_images(calib: CameraCalibration) -> None:
    assert isinstance(calib, CameraCalibration)
    assert calib.cameraMatrix.shape == (3, 3) and calib.distCoeffs.shape[0] == 1
    assert calib.imageSize == SIZE
    assert calib.rms < 0.5
    assert np.allclose(calib.cameraMatrix[:2, :2], TRUE_K[:2, :2], rtol=0.05)
    assert len(calib.rvecs) == 5 and calib.rvecs[0].shape == (3, 1) and calib.tvecs[0].shape == (3, 1)


def test_calibrate_from_corners(views: list[np.ndarray], calib: CameraCalibration) -> None:
    corners = [findChessboardEx(v, PATTERN) for v in views]
    assert all(c is not None for c in corners)
    again = calibrateCameraEx([c for c in corners if c is not None], PATTERN, SQUARE, imageSize=SIZE)
    assert np.allclose(again.cameraMatrix, calib.cameraMatrix, rtol=1e-3)
    errors = reprojectionErrors(again, PATTERN, SQUARE, [c for c in corners if c is not None])
    assert errors.shape == (5,) and errors.max() < 0.5


def test_calibrate_errors(views: list[np.ndarray]) -> None:
    with pytest.raises(ValueError, match="empty"):
        calibrateCameraEx([], PATTERN)
    with pytest.raises(ValueError, match="imageSize"):
        calibrateCameraEx([np.zeros((30, 2), np.float32)] * 3, PATTERN)
    with pytest.raises(ValueError, match="at least 3"):
        calibrateCameraEx(views[:2], PATTERN)
    blank = np.full(SIZE[::-1], 255, np.uint8)
    with pytest.raises(ValueError, match="only 2 view"):
        calibrateCameraEx([*views[:2], blank, blank], PATTERN)
    with pytest.raises(ValueError, match="view 1 has 4 corners"):
        calibrateCameraEx([np.zeros((30, 2)), np.zeros((4, 2)), np.zeros((30, 2))], PATTERN, imageSize=SIZE)


def test_reprojection_errors_mismatch(calib: CameraCalibration) -> None:
    with pytest.raises(ValueError, match="views"):
        reprojectionErrors(calib, PATTERN, SQUARE, [np.zeros((30, 2))])


def test_undistort(views: list[np.ndarray], calib: CameraCalibration) -> None:
    out = undistortEx(views[0], calib)
    assert out.dtype == np.uint8 and out.ndim == 2
    assert 0 < out.shape[0] <= SIZE[1] and 0 < out.shape[1] <= SIZE[0]
    full = undistortEx(cv2.cvtColor(views[0], cv2.COLOR_GRAY2BGR), calib, crop=False, alpha=1.0)
    assert full.shape == (SIZE[1], SIZE[0], 3)
    raw = undistortEx(views[0], calib.cameraMatrix, calib.distCoeffs, crop=False)
    assert raw.shape == views[0].shape


def test_undistort_errors(views: list[np.ndarray], calib: CameraCalibration) -> None:
    with pytest.raises(ValueError, match="src"):
        undistortEx(np.zeros((0, 0), np.uint8), calib)
    with pytest.raises(ValueError, match="distCoeffs"):
        undistortEx(views[0], calib.cameraMatrix)
    with pytest.raises(ValueError, match="alpha"):
        undistortEx(views[0], calib, alpha=2.0)
    with pytest.raises(ValueError, match="alpha"):
        undistortEx(views[0], calib, alpha=float("nan"))
    with pytest.raises(ValueError, match="size"):
        undistortEx(views[0][:100], calib)


def test_save_load_roundtrip(tmp_path, calib: CameraCalibration) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "보정" / "camera.json"
    path.parent.mkdir()
    saveCalibration(path, calib)
    loaded = loadCalibration(str(path))
    assert np.allclose(loaded.cameraMatrix, calib.cameraMatrix)
    assert np.allclose(loaded.distCoeffs, calib.distCoeffs) and loaded.distCoeffs.shape == calib.distCoeffs.shape
    assert loaded.imageSize == calib.imageSize and loaded.rms == calib.rms
    assert len(loaded.rvecs) == 5 and np.allclose(loaded.tvecs[2], calib.tvecs[2])
    assert loaded.usedViews == calib.usedViews == (0, 1, 2, 3, 4)
    assert set(json.loads(path.read_text("utf-8"))) >= {"cameraMatrix", "distCoeffs", "imageSize", "rms", "usedViews"}
    old = {k: v for k, v in json.loads(path.read_text("utf-8")).items() if k != "usedViews"}
    (tmp_path / "old.json").write_text(json.dumps(old), "utf-8")
    assert loadCalibration(tmp_path / "old.json").usedViews == ()  # files from before the field existed


def test_load_errors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "bad.json"
    bad.write_text("{}", "utf-8")
    with pytest.raises(ValueError, match="not a calibration file"):
        loadCalibration(bad)
    with pytest.raises(OSError):
        loadCalibration(tmp_path / "missing.json")


def test_calibrate_skipped_views_are_listed_in_usedViews(views: list[np.ndarray]) -> None:
    blank = np.full_like(views[0], 255)
    mixed = [views[0], blank, views[1], views[2], blank, views[3]]
    calib = calibrateCameraEx(mixed, PATTERN, SQUARE)
    assert calib.usedViews == (0, 2, 3, 5)
    assert len(calib.rvecs) == len(calib.tvecs) == 4
    corners = [findChessboardEx(v, PATTERN) for v in mixed]
    assert corners[1] is None and corners[4] is None
    full = reprojectionErrors(calib, PATTERN, SQUARE, corners)  # one entry per original view, None included
    used = reprojectionErrors(calib, PATTERN, SQUARE, [c for c in corners if c is not None])  # per used view
    assert full.shape == (4,) and np.allclose(full, used)
    with pytest.raises(ValueError, match="cornersList"):
        reprojectionErrors(calib, PATTERN, SQUARE, corners[:5])
    with pytest.raises(ValueError, match="None"):
        reprojectionErrors(calib, PATTERN, SQUARE, [corners[0], None, corners[2], corners[3]])
    from_corners = calibrateCameraEx([c for c in corners if c is not None], PATTERN, SQUARE, imageSize=SIZE)
    assert from_corners.usedViews == (0, 1, 2, 3)
