"""Shared inputs for the benchmarks.

Run with ``pytest benchmarks`` (needs ``pytest-benchmark``, from the ``bench`` dependency group). Without the plugin
the ``benchmark`` fixture below just calls the function once, so the files still run as smoke tests.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import cv2u as cv

try:
    import pytest_benchmark  # noqa: F401
except ImportError:  # pragma: no cover - only without the bench dependency group

    @pytest.fixture
    def benchmark() -> Callable[..., Any]:
        def run(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return run


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def bgr(rng: np.random.Generator) -> np.ndarray:
    """A 1080p BGR photo-like image: smooth gradients plus noise and a few shapes."""
    img = cv.makeGradient((1920, 1080), (40, 60, 90), (200, 180, 160))
    noise = rng.integers(-20, 20, img.shape, np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv.circle(img, (600, 500), 200, (255, 255, 255), -1)
    cv.rectangle(img, (1200, 300), (1600, 800), (0, 0, 0), -1)
    cv.line(img, (0, 1000), (1919, 100), (0, 255, 0), 5)
    return img


@pytest.fixture(scope="session")
def gray(bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)


@pytest.fixture(scope="session")
def mask(gray: np.ndarray) -> np.ndarray:
    return cv.binarize(gray, "otsu")


@pytest.fixture(scope="session")
def small(bgr: np.ndarray) -> np.ndarray:
    """A 640x360 version, for the per-call overhead benchmarks."""
    return cv.resize(bgr, (640, 360), interpolation=cv.INTER_AREA)


@pytest.fixture(scope="session")
def boxes(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """1000 detector boxes ``(x, y, w, h)`` with scores."""
    xy = rng.uniform(0, 1800, (1000, 2))
    wh = rng.uniform(20, 200, (1000, 2))
    return np.hstack([xy, wh]).astype(np.float32), rng.uniform(0, 1, 1000).astype(np.float32)
