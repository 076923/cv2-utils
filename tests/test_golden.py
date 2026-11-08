"""Regression tests against golden images in ``tests/data``.

Regenerate them with ``python tools/regenerate_goldens.py`` only when an output change is intended.
Small tolerances absorb rounding differences between OpenCV's SIMD code paths on different CPUs and between
OpenCV 4 and 5, whose bilinear warps round differently along edges.
"""

from pathlib import Path

import numpy as np
import pytest

import cv2u as cv

DATA = Path(__file__).parent / "data"


def _read(name, flags=cv.IMREAD_UNCHANGED):
    img = cv.imread(str(DATA / name), flags)
    assert img is not None, f"missing golden image {name}; run tools/regenerate_goldens.py"
    return img


def _assert_close(actual, expected, *, max_abs=1, max_mismatch_ratio=0.0):
    assert actual.shape == expected.shape
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    mismatch = np.count_nonzero(diff > max_abs) / diff.size
    assert mismatch <= max_mismatch_ratio, f"{mismatch:.2%} of values differ by more than {max_abs}"


@pytest.fixture(scope="module")
def rotate_src():
    return _read("rotate_src.png")


def test_rotateEx_45(rotate_src):
    _assert_close(cv.rotateEx(rotate_src, 45), _read("rotateEx_45.png"), max_abs=2, max_mismatch_ratio=0.03)


def test_rotateEx_30_scale(rotate_src):
    _assert_close(
        cv.rotateEx(rotate_src, 30, scale=0.5), _read("rotateEx_30_scale0.5.png"), max_abs=2, max_mismatch_ratio=0.03
    )


def test_gammaCorrection(rotate_src):
    _assert_close(cv.gammaCorrection(rotate_src, 2.2), _read("gammaCorrection_2.2.png"), max_abs=0)


def test_boldEdges(rotate_src):
    gray = cv.cvtColor(rotate_src, cv.COLOR_BGR2GRAY)
    # boldEdges amplifies differences by pi, so allow a little more slack.
    _assert_close(cv.boldEdges(gray), _read("boldEdges.png"), max_abs=4, max_mismatch_ratio=0.001)


def test_drawClusters():
    pytest.importorskip("sklearn")
    src = _read("dbscan_src.png")
    maps, _ = cv.dbscanColors(src, 1.7, 2)
    # A +-1 difference in the Lab conversion can move a few border pixels to another cluster.
    _assert_close(cv.drawClusters(src, maps), _read("drawClusters.png"), max_abs=2, max_mismatch_ratio=0.01)
