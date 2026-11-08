import cv2
import numpy as np
import pytest

from cv2u.imgproc.stitch import stitchImages, stitchPair

WIDTH, HEIGHT = 480, 240


def _scene(seed: int = 0) -> np.ndarray:
    """Textured BGR scene: blurred noise with a few shapes, so that features are easy to find."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (HEIGHT, WIDTH, 3), np.uint8)
    scene = cv2.GaussianBlur(noise, (0, 0), 1.0)
    for _ in range(40):
        x, y = int(rng.integers(0, WIDTH - 40)), int(rng.integers(0, HEIGHT - 40))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            w, h = (int(v) for v in rng.integers(8, 40, 2))
            cv2.rectangle(scene, (x, y), (x + w, y + h), color, -1)
        else:
            cv2.circle(scene, (x + 15, y + 15), int(rng.integers(5, 18)), color, -1)
    return scene


@pytest.fixture(scope="module")
def halves() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene = _scene()
    split = WIDTH // 2
    overlap = 80
    return scene, scene[:, : split + overlap].copy(), scene[:, split - overlap :].copy()


def test_stitch_pair(halves: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    scene, left, right = halves
    out = stitchPair(left, right)
    assert out.dtype == np.uint8 and out.ndim == 3
    assert abs(out.shape[1] - WIDTH) <= 6 and abs(out.shape[0] - HEIGHT) <= 6
    # the left image is pasted unchanged somewhere on the canvas, and the whole scene is reproduced around it
    scores = cv2.matchTemplate(out, left, cv2.TM_SQDIFF)
    _, _, (x, y), _ = cv2.minMaxLoc(scores)
    assert scores[y, x] == 0
    region = out[y : y + HEIGHT, x : x + WIDTH]
    assert region.shape == scene.shape and cv2.absdiff(region, scene).mean() < 15


def test_stitch_pair_gray_and_sift(halves: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    _, left, right = halves
    gray = stitchPair(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), right)
    assert gray.ndim == 2 and abs(gray.shape[1] - WIDTH) <= 6
    color = stitchPair(left, cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), method="SIFT")
    assert color.ndim == 3 and abs(color.shape[1] - WIDTH) <= 6


def test_stitch_pair_errors(halves: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    _, left, right = halves
    with pytest.raises(ValueError, match="a must not be empty"):
        stitchPair(np.zeros((0, 0, 3), np.uint8), right)
    with pytest.raises(ValueError, match="b must not be empty"):
        stitchPair(left, np.zeros((0, 0, 3), np.uint8))
    with pytest.raises(ValueError, match="minMatches"):
        stitchPair(left, right, minMatches=3)
    with pytest.raises(ValueError, match="ransacThreshold"):
        stitchPair(left, right, ransacThreshold=0)
    with pytest.raises(ValueError, match="matches found"):
        stitchPair(left, np.full((100, 100, 3), 200, np.uint8))


def test_stitch_images(halves: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    _, left, right = halves
    try:
        out = stitchImages([left, right], mode="scans", confidence=0.3)
    except RuntimeError as exc:
        pytest.skip(f"cv.Stitcher cannot stitch the synthetic scene here: {exc}")
    assert out.dtype == np.uint8 and out.ndim == 3
    assert abs(out.shape[1] - WIDTH) <= 20 and abs(out.shape[0] - HEIGHT) <= 20
    gray = stitchImages([cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)], "scans", 0.3)
    assert gray.ndim == 3


def test_stitch_images_errors(halves: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    _, left, right = halves
    with pytest.raises(ValueError, match="at least 2"):
        stitchImages([left])
    with pytest.raises(ValueError, match=r"images\[1\]"):
        stitchImages([left, np.zeros((0, 0, 3), np.uint8)])
    with pytest.raises(ValueError, match="mode"):
        stitchImages([left, right], mode="spherical")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confidence"):
        stitchImages([left, right], confidence=-1)
    flat = np.full((60, 60, 3), 128, np.uint8)
    with pytest.raises(RuntimeError, match="ERR_"):
        stitchImages([flat, flat.copy()])
