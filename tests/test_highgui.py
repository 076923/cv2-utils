import sys
import types

import cv2
import numpy as np
import pytest

import cv2u as cv


@pytest.fixture
def shown(monkeypatch):
    calls = []
    monkeypatch.setattr(cv2, "imshow", lambda name, img: calls.append((name, img)))
    return calls


def test_imshowEx_shrinks_large_images(shown):
    cv.imshowEx("win", np.zeros((3000, 4000, 3), np.uint8), maxSize=(800, 600))
    name, img = shown[0]
    assert name == "win"
    assert img.shape == (600, 800, 3)


def test_imshowEx_keeps_small_images(shown):
    src = np.zeros((100, 200), np.uint8)
    cv.imshowEx("win", src)
    assert shown[0][1] is src


def test_imshowEx_validation(shown):
    with pytest.raises(ValueError, match="maxSize"):
        cv.imshowEx("win", np.zeros((10, 10), np.uint8), maxSize=(0, 10))


@pytest.fixture
def fake_ipython(monkeypatch):
    displayed = []

    class Image:
        def __init__(self, data):
            self.data = data

    module = types.ModuleType("IPython.display")
    module.Image = Image
    module.display = displayed.append
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", module)
    return displayed


def test_imshowInline_displays_true_colors(fake_ipython):
    bgr = np.zeros((20, 30, 3), np.uint8)
    bgr[:] = (255, 0, 0)  # pure blue in BGR
    cv.imshowInline(bgr)
    decoded = cv.imdecode(np.frombuffer(fake_ipython[0].data, np.uint8), cv.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, bgr)  # PNG keeps BGR order, so the notebook shows blue


def test_imshowInline_maxSize(fake_ipython):
    cv.imshowInline(np.zeros((1000, 2000, 3), np.uint8), maxSize=(200, 200))
    decoded = cv.imdecode(np.frombuffer(fake_ipython[0].data, np.uint8), cv.IMREAD_UNCHANGED)
    assert decoded.shape == (100, 200, 3)


def test_imshowInline_without_ipython(monkeypatch):
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    with pytest.raises(ImportError, match="IPython"):
        cv.imshowInline(np.zeros((2, 2), np.uint8))


def test_imshowInline_unencodable(fake_ipython):
    with pytest.raises(ValueError, match="encode"):
        cv.imshowInline(np.zeros((2, 2), np.uint8), ext=".nope")
