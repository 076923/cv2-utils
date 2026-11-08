import sys

import numpy as np
import pytest

import cv2u as cv


@pytest.fixture
def bgr():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)


@pytest.fixture
def square():
    return np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)


# --- package / cv2 passthrough -------------------------------------------------


def test_version():
    assert isinstance(cv.__version__, str)


def test_requires_supported_opencv():
    import cv2

    assert tuple(int(p) for p in cv2.__version__.split(".")[:2]) >= cv._opencv.MIN_VERSION == (4, 10)


def test_missing_opencv_says_how_to_install():
    import subprocess

    code = "import sys; sys.modules['cv2'] = None\ntry:\n import cv2u\nexcept ImportError as e:\n print(e)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert "python-opencv-utils[opencv]" in out
    assert "[headless]" in out


def test_cv2_passthrough():
    import cv2

    assert cv is not cv2
    assert cv.imread is cv2.imread
    assert cv.COLOR_BGR2GRAY == cv2.COLOR_BGR2GRAY
    assert cv.dnn is not cv2.dnn  # cv2u.dnn adds helpers and passes the rest through
    assert cv.dnn.readNet is cv2.dnn.readNet
    assert cv.dnn.DNN_BACKEND_OPENCV == cv2.dnn.DNN_BACKEND_OPENCV
    assert "imread" in dir(cv)


def test_missing_attribute():
    with pytest.raises(AttributeError, match="cv2u"):
        cv.definitely_not_an_opencv_function  # noqa: B018


def test_star_import_exports_only_extensions():
    ns = {}
    exec("from cv2u import *", ns)
    assert "rotateEx" in ns
    assert "np" not in ns
    assert "cv2" not in ns
    assert "cv" not in ns


# --- design principles (see docs/principles.md) ------------------------------------


def test_extensions_do_not_shadow_opencv():
    # Principle: cv2u adds names, it never replaces an OpenCV function or constant.
    import cv2

    assert [name for name in cv.__all__ if hasattr(cv2, name)] == []


def test_extension_names_follow_opencv_style():
    # Principle: functions use camelCase (like OpenCV), constants use UPPER_CASE.
    import re

    for name in cv.__all__:
        assert re.fullmatch(r"[A-Za-z][A-Za-z0-9]*|[A-Z][A-Z0-9_]*", name), name


def test_top_level_exports_every_submodule_name():
    # Principle: ``import cv2u as cv`` reaches every extension; nothing is reachable only from a submodule.
    from cv2u import calib3d, core, flag, highgui, imgcodecs, imgproc, videoio

    submodule_names = set()
    for module in (calib3d, core, flag, highgui, imgcodecs, imgproc, videoio):
        submodule_names |= set(module.__all__)
    assert submodule_names == set(cv.__all__)


def test_readme_api_names_exist():
    # Every ``cv.<name>`` in the README resolves on cv2u (an extension or a cv2 passthrough).
    import re
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    names = set(re.findall(r"\bcv\.([A-Za-z_][A-Za-z0-9_]*)", readme))
    assert names
    assert [name for name in sorted(names) if not hasattr(cv, name)] == []


def test_extensions_are_documented():
    for name in cv.__all__:
        obj = getattr(cv, name)
        if callable(obj):
            assert obj.__doc__, f"{name} has no docstring"


# --- core ------------------------------------------------------------------------


@pytest.mark.parametrize(("angle", "expected"), [(0, (40, 60)), (90, (60, 40)), (180, (40, 60))])
def test_rotateEx_shape(bgr, angle, expected):
    assert cv.rotateEx(bgr, angle).shape[:2] == expected


def test_rotateEx_does_not_crop(bgr):
    h, w = cv.rotateEx(bgr, 45).shape[:2]
    assert h > 40
    assert w > 60


def test_rotateEx_scale(bgr):
    assert cv.rotateEx(bgr, 0, scale=2).shape[:2] == (80, 120)


def test_rotateEx_forwards_border_value():
    src = np.zeros((10, 10, 3), np.uint8)
    dst = cv.rotateEx(src, 45, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 255))
    assert tuple(dst[0, 0]) == (0, 0, 255)


def test_dbscan_roundtrip():
    pytest.importorskip("sklearn")
    src = np.zeros((8, 8, 3), np.uint8)
    src[:, 4:] = (200, 50, 10)
    maps, centers = cv.dbscanColors(src, 1.7, 2)
    assert isinstance(cv.dbscanColors(src, 1.7, 2), cv.Clusters)
    assert maps.shape == (8, 8) and maps.dtype == np.int32
    assert sorted(np.unique(maps).tolist()) == [0, 1]  # labels start at 0; -1 would be noise
    assert centers.shape == (2, 3) and centers.dtype == np.uint8
    assert {tuple(c) for c in centers} == {(0, 0, 0), (200, 50, 10)}
    dst = cv.drawClusters(src, maps)
    assert dst.shape == src.shape
    np.testing.assert_array_equal(dst, src)


def test_dbscan_noise_is_minus_one():
    pytest.importorskip("sklearn")
    src = np.zeros((6, 6, 3), np.uint8)
    src[2, 2] = (255, 255, 255)  # a lone pixel is noise with minSamples=4
    maps, centers = cv.dbscanColors(src, 1.7)
    assert maps[2, 2] == -1
    assert len(centers) == 1 and maps.max() == 0
    painted = cv.drawClusters(src, maps)
    assert tuple(painted[2, 2]) == (255, 255, 255)  # noise keeps its original color


def test_dbscan_without_cluster_extra(monkeypatch, bgr):
    # Simulate an install without the ``cluster`` extra.
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    with pytest.raises(ImportError, match=r"python-opencv-utils\[cluster\]"):
        cv.dbscanColors(bgr, 1.7, 2)


@pytest.mark.parametrize("src", [np.zeros((8, 8), np.uint8), np.zeros((8, 8, 4), np.uint8), np.zeros((1, 3), np.uint8)])
def test_dbscan_rejects_non_bgr(src):
    with pytest.raises(ValueError, match="3-channel"):
        cv.dbscanColors(src, 1.7, 2)


def test_dbscan_validation(bgr):
    with pytest.raises(ValueError, match="uint8"):
        cv.dbscanColors(bgr.astype(np.float32), 1.7)
    with pytest.raises(ValueError, match="empty"):
        cv.dbscanColors(np.zeros((0, 0, 3), np.uint8), 1.7)
    with pytest.raises(ValueError, match="eps"):
        cv.dbscanColors(bgr, 0)
    with pytest.raises(ValueError, match="minSamples"):
        cv.dbscanColors(bgr, 1.7, minSamples=0)


def test_drawClusters_rejects_mismatched_maps(bgr):
    with pytest.raises(ValueError, match="maps"):
        cv.drawClusters(bgr, np.zeros((2, 5), dtype=np.int64))


def test_gammaCorrection(bgr):
    np.testing.assert_array_equal(cv.gammaCorrection(bgr, 1.0), bgr)
    assert cv.gammaCorrection(bgr, 2.2).mean() > bgr.mean()


@pytest.mark.parametrize(
    ("src", "gamma", "match"),
    [(np.zeros((4, 4), np.float32), 1.0, "uint8"), (np.zeros((4, 4), np.uint8), 0, "positive")],
)
def test_gammaCorrection_validation(src, gamma, match):
    with pytest.raises(ValueError, match=match):
        cv.gammaCorrection(src, gamma)


def test_centerPoint(square):
    assert cv.centerPoint(square) == pytest.approx((5.0, 5.0))


def test_centerPoint_zero_area():
    with pytest.raises(ValueError):
        cv.centerPoint(np.array([[[1, 1]]], dtype=np.int32))


def test_polar_roundtrip():
    theta, rho = cv.cartesianToPolar(3.0, 4.0)
    assert rho == pytest.approx(5.0)
    assert cv.polarToCartesian(theta, rho) == pytest.approx((3.0, 4.0))


def test_polar_angleInDegrees():
    theta, rho = cv.cartesianToPolar(0.0, 2.0, angleInDegrees=True)
    assert (theta, rho) == pytest.approx((90.0, 2.0))
    assert cv.polarToCartesian(90.0, 2.0, angleInDegrees=True) == pytest.approx((0.0, 2.0))


def test_polar_arrays_match_opencv():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(2, 50))
    theta, rho = cv.cartesianToPolar(x, y)
    magnitude, angle = cv.cartToPolar(x, y)  # OpenCV: (magnitude, angle in [0, 2pi))
    np.testing.assert_allclose(rho, magnitude.ravel(), rtol=1e-6)
    # OpenCV computes the angle with a fast approximation (about 0.3 degrees accurate);
    # compare the wrapped angular difference so 0 and 2pi count as equal.
    diff = (theta - angle.ravel() + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(diff).max() < np.radians(0.5)


@pytest.mark.parametrize("reshape", [False, True])
def test_isInside(square, reshape):
    poly = square.reshape(-1, 2) if reshape else square
    assert cv.isInside((5, 5), poly)
    assert not cv.isInside((15, 5), poly)


def test_isInside_matches_pointPolygonTest():
    rng = np.random.default_rng(42)
    # A non-convex polygon (star) to exercise multiple crossings.
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    radii = np.where(np.arange(10) % 2 == 0, 40, 15)
    star = np.stack([50 + radii * np.cos(angles), 50 + radii * np.sin(angles)], axis=1).astype(np.float32)
    for x, y in rng.uniform(0, 100, (500, 2)):
        expected = cv.pointPolygonTest(star, (float(x), float(y)), measureDist=True)
        if abs(expected) < 1e-6:  # on an edge: both conventions are valid
            continue
        assert cv.isInside((x, y), star) == (expected > 0), (x, y)


def test_boldEdges(bgr):
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    dst = cv.boldEdges(gray)
    assert dst.shape == gray.shape
    assert dst.dtype == np.uint8


def test_boldEdges_flat_image_has_no_edges():
    # Including the 1-pixel border, which must not keep the source intensity.
    assert not cv.boldEdges(np.full((10, 12), 100, np.uint8)).any()


def test_boldEdges_color(bgr):
    assert cv.boldEdges(bgr).shape == bgr.shape


def test_boldEdges_validation():
    with pytest.raises(ValueError, match="uint8"):
        cv.boldEdges(np.zeros((8, 8), np.float32))
    with pytest.raises(ValueError, match="empty"):
        cv.boldEdges(np.zeros((0, 8), np.uint8))


def test_makeGradient_bgr():
    img = cv.makeGradient((20, 10), (0, 0, 0), (255, 128, 64))
    assert img.shape == (10, 20, 3)
    assert img.dtype == np.uint8
    assert tuple(img[0, 0]) == (0, 0, 0)
    assert tuple(img[-1, -1]) == (255, 128, 64)


@pytest.mark.parametrize(("start", "end"), [(0, 255), ((0,), (255,))])
def test_makeGradient_gray(start, end):
    img = cv.makeGradient((20, 10), start, end, channels=1)
    assert img.shape == (10, 20)
    assert img.flat[0] == 0
    assert img.flat[-1] == 255


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": (10, 10), "start": 0, "end": 255, "channels": 4},
        {"size": (0, 10), "start": 0, "end": 255},
        {"size": (10, 10), "start": -1, "end": 255},
        {"size": (10, 10), "start": 0, "end": float("nan")},
        {"size": (10, 10), "start": (0, 0), "end": 255},
    ],
)
def test_makeGradient_validation(kwargs):
    with pytest.raises(ValueError):
        cv.makeGradient(**kwargs)


def test_makeGradient_horizontal():
    img = cv.makeGradient((256, 4), (0, 0, 0), (255, 255, 255), direction="horizontal")
    assert img.shape == (4, 256, 3)
    np.testing.assert_array_equal(img[0], img[-1])  # every row is the same
    assert img[0, 0, 0] == 0
    assert img[0, -1, 0] == 255
    assert (np.diff(img[0, :, 0].astype(int)) >= 0).all()


def test_makeGradient_vertical():
    img = cv.makeGradient((4, 256), 0, 255, direction="vertical", channels=1)
    assert img.shape == (256, 4)
    np.testing.assert_array_equal(img[:, 0], img[:, -1])  # every column is the same
    assert img[0, 0] == 0
    assert img[-1, 0] == 255


def test_makeGradient_horizontal_is_default():
    np.testing.assert_array_equal(
        cv.makeGradient((20, 10), 0, 255), cv.makeGradient((20, 10), 0, 255, direction="horizontal")
    )


def test_makeGradient_raster():
    img = cv.makeGradient((3, 2), 0, 250, direction="raster", channels=1)
    np.testing.assert_array_equal(img, [[0, 50, 100], [150, 200, 250]])


def test_makeGradient_invalid_direction():
    with pytest.raises(ValueError, match="direction"):
        cv.makeGradient((10, 10), 0, 255, direction="diagonal")


def test_imcolor(bgr):
    img = cv.imcolor((5, 4), (1, 2, 3))
    assert img.shape == (4, 5, 3) and img.dtype == np.uint8
    assert tuple(img[2, 2]) == (1, 2, 3)
    assert cv.imcolor((2, 3), 0.5, channels=1, dtype=np.float32).shape == (3, 2)
    assert cv.imcolor((2, 2), 0.5, channels=1, dtype=np.float32).dtype == np.float32
    assert tuple(cv.imcolor((2, 2), (1, 2, 3, 4), channels=4)[0, 0]) == (1, 2, 3, 4)
    assert tuple(cv.imcolor((2, 2), 9)[0, 0]) == (9, 9, 9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": (0, 2), "color": 0},
        {"size": (2, 2), "color": 0, "channels": 2},
        {"size": (2, 2), "color": 256},
        {"size": (2, 2), "color": float("inf")},
        {"size": (2, 2), "color": (1, 2)},
    ],
)
def test_imcolor_validation(kwargs):
    with pytest.raises(ValueError):
        cv.imcolor(**kwargs)


def test_imcolorLike(bgr):
    like = cv.imcolorLike(bgr, (1, 2, 3))
    assert like.shape == bgr.shape
    assert like.dtype == bgr.dtype
    assert tuple(like[0, 0]) == (1, 2, 3)
    assert cv.imcolorLike(np.zeros((2, 2), np.float32), 0.5).dtype == np.float32
    assert cv.imcolorLike(np.zeros((2, 2), np.float32), 7, dtype=np.uint8).dtype == np.uint8
    with pytest.raises(ValueError, match="color"):
        cv.imcolorLike(bgr, 300)
    with pytest.raises(ValueError, match="color"):
        cv.imcolorLike(bgr, (1, 2))
    with pytest.raises(ValueError, match="color"):
        cv.imcolorLike(np.zeros((2, 2), np.float32), float("nan"))


# --- imgcodecs -------------------------------------------------------------------


def test_imreadUrl_file_url(tmp_path, bgr):
    path = tmp_path / "img.png"
    assert cv.imwrite(str(path), bgr)
    np.testing.assert_array_equal(cv.imreadUrl(path.as_uri()), bgr)


def test_imreadUrl_reads_color_by_default_like_imread(tmp_path):
    path = tmp_path / "img.png"
    assert cv.imwrite(str(path), np.zeros((4, 4, 4), np.uint8))
    assert cv.imreadUrl(path.as_uri()).shape == (4, 4, 3)
    assert cv.imreadUrl(path.as_uri(), cv.IMREAD_UNCHANGED).shape == (4, 4, 4)


def test_imreadUrl_undecodable_returns_none(tmp_path):
    path = tmp_path / "not-an-image.png"
    path.write_bytes(b"hello")
    assert cv.imreadUrl(path.as_uri()) is None


def test_imreadUrl_sends_headers(monkeypatch, tmp_path, bgr):
    import urllib.request

    urlread_module = sys.modules["cv2u.imgcodecs.url"]

    path = tmp_path / "img.png"
    assert cv.imwrite(str(path), bgr)
    seen = {}
    real_urlopen = urllib.request.urlopen

    def spy(request, timeout=None):
        seen.update(request.header_items())
        seen["timeout"] = timeout
        return real_urlopen(request, timeout=timeout)

    monkeypatch.setattr(urlread_module, "urlopen", spy)
    cv.imreadUrl(path.as_uri(), timeout=5, headers={"X-Test": "1"})
    assert seen["User-agent"].startswith("cv2u")
    assert seen["X-test"] == "1"
    assert seen["timeout"] == 5

    cv.imreadUrl(path.as_uri(), headers={"User-Agent": "custom"})
    assert seen["User-agent"] == "custom"


def test_flip_flags(bgr):
    np.testing.assert_array_equal(cv.flip(bgr, cv.FLIP_MODE_X), bgr[::-1])
    np.testing.assert_array_equal(cv.flip(bgr, cv.FLIP_MODE_Y), bgr[:, ::-1])
    np.testing.assert_array_equal(cv.flip(bgr, cv.FLIP_MODE_XY), bgr[::-1, ::-1])


# --- imgproc ---------------------------------------------------------------------


@pytest.mark.parametrize("gray_input", [False, True])
def test_putTextEx(bgr, gray_input, font_path):
    font = font_path
    src = np.zeros((40, 60), np.uint8) if gray_input else np.zeros_like(bgr)
    dst = cv.putTextEx(src, "Hi 한글", (2, 2), font, 20, (255, 255, 255))  # pathlib.Path is accepted
    assert dst is src  # drawn in place, like cv.putText
    assert dst.shape == src.shape  # grayscale stays grayscale
    assert dst.any()
