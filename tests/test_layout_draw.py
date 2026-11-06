import cv2
import numpy as np
import pytest

import cv2u as cv


@pytest.fixture
def img():
    return np.random.default_rng(0).integers(0, 256, (50, 70, 3), dtype=np.uint8)


# --- translate -------------------------------------------------------------------------


def test_translate(img):
    out = cv.translate(img, 3, 2)
    assert out.shape == img.shape
    np.testing.assert_array_equal(out[2:, 3:], img[:-2, :-3])
    assert not out[:2].any()
    assert not out[:, :3].any()
    np.testing.assert_array_equal(cv.translate(img, -3, -2)[:-2, :-3], img[2:, 3:])


def test_translate_borders(img):
    filled = cv.translate(img, 5, 0, borderValue=(1, 2, 3))
    assert tuple(filled[0, 0]) == (1, 2, 3)
    replicated = cv.translate(img, 5, 0, borderMode=cv2.BORDER_REPLICATE)
    np.testing.assert_array_equal(replicated[:, 0], img[:, 0])


# --- padToMultiple / padToSize ------------------------------------------------------------


def test_padToMultiple(img):
    out = cv.padToMultiple(img, 32, color=(9, 9, 9))
    assert out.shape == (64, 96, 3)
    np.testing.assert_array_equal(out[:50, :70], img)  # original stays at the top-left
    assert tuple(out[63, 95]) == (9, 9, 9)
    same = cv.padToMultiple(img, 10)
    np.testing.assert_array_equal(same, img)
    assert same is not img
    with pytest.raises(ValueError):
        cv.padToMultiple(img, 0)


def test_padToSize(img):
    centered = cv.padToSize(img, (100, 60), color=255)
    assert centered.shape == (60, 100, 3)
    np.testing.assert_array_equal(centered[5:55, 15:85], img)
    corner = cv.padToSize(img, (100, 60), center=False)
    np.testing.assert_array_equal(corner[:50, :70], img)
    with pytest.raises(ValueError, match="letterbox"):
        cv.padToSize(img, (60, 60))


# --- splitTiles / mergeTiles ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "tile", "overlap"), [((70, 50), (32, 32), 0), ((70, 50), (32, 20), 8), ((64, 64), (32, 32), 0)]
)
def test_tiles_roundtrip(img, size, tile, overlap):
    src = cv.resize(img, size)
    tiles = cv.splitTiles(src, tile, overlap=overlap)
    assert all(t.shape[:2] == (tile[1], tile[0]) for t, _ in tiles)  # every tile has the full size
    covered = np.zeros(src.shape[:2], bool)
    for t, (x, y) in tiles:
        covered[y : y + t.shape[0], x : x + t.shape[1]] = True
        np.testing.assert_array_equal(t, src[y : y + t.shape[0], x : x + t.shape[1]])
    assert covered.all()
    np.testing.assert_array_equal(cv.mergeTiles(tiles, size), src)


def test_mergeTiles_averages_processed_overlaps():
    src = np.zeros((10, 30), np.uint16)
    tiles = cv.splitTiles(src, (16, 10), overlap=6)
    assert [xy for _, xy in tiles] == [(0, 0), (10, 0), (14, 0)]  # the last tile is moved inside the image
    processed = [(np.full_like(t, 100 * (i + 1)), xy) for i, (t, xy) in enumerate(tiles)]
    merged = cv.mergeTiles(processed, (30, 10))
    assert merged.dtype == np.uint16
    assert merged[0, 0] == 100  # only the first tile
    assert merged[0, 12] == 150  # first and second tile: their average
    assert merged[0, 15] == 200  # all three tiles
    assert merged[0, 29] == 300  # only the last tile


def test_splitTiles_small_image_and_validation(img):
    tiles = cv.splitTiles(img, (100, 100))
    assert len(tiles) == 1
    assert tiles[0][0].shape == img.shape
    with pytest.raises(ValueError):
        cv.splitTiles(img, (0, 10))
    with pytest.raises(ValueError):
        cv.splitTiles(img, (10, 10), overlap=10)
    with pytest.raises(ValueError):
        cv.mergeTiles([], (10, 10))


# --- mapTiles -----------------------------------------------------------------------------------


def test_mapTiles_identity_and_channel_change(img):
    src = cv.resize(img, (70, 50))
    np.testing.assert_array_equal(cv.mapTiles(src, (32, 32), lambda t: t, overlap=8), src)
    gray = cv.mapTiles(src, (32, 32), lambda t: cv.cvtColor(t, cv.COLOR_BGR2GRAY))
    assert gray.shape == (50, 70)
    np.testing.assert_array_equal(gray, cv.cvtColor(src, cv.COLOR_BGR2GRAY))
    heat = cv.mapTiles(src, (32, 32), lambda t: t[..., 0].astype(np.float32) / 255)
    assert heat.dtype == np.float32
    assert heat.shape == (50, 70)


def test_mapTiles_calls_func_per_tile_with_progress(img):
    src = cv.resize(img, (64, 64))
    seen = []
    progress = []
    out = cv.mapTiles(
        src, (32, 32), lambda t: (seen.append(t.shape), t)[1], progress=lambda d, n: progress.append((d, n))
    )
    assert seen == [(32, 32, 3)] * 4
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    np.testing.assert_array_equal(out, src)


def test_mapTiles_validation(img):
    with pytest.raises(ValueError, match="same size"):
        cv.mapTiles(img, (16, 16), lambda t: t[:8])
    with pytest.raises(ValueError, match="src"):
        cv.mapTiles(np.zeros((0, 10, 3), np.uint8), (16, 16), lambda t: t)
    with pytest.raises(ValueError):
        cv.mapTiles(img, (0, 16), lambda t: t)


# --- drawDashedLine / drawDashedRectangle / drawHistogram ------------------------------------


def _runs(row):
    """Number of separate ink segments in a 1-D array."""
    ink = np.asarray(row) > 0
    return int(np.count_nonzero(ink[1:] & ~ink[:-1]) + ink[0])


def test_drawDashedLine_has_gaps():
    canvas = np.zeros((10, 100), np.uint8)
    out = cv.drawDashedLine(canvas, (0, 5), (99, 5), 255, dashLength=10)
    assert out is canvas  # in place, like cv.line
    assert _runs(canvas[5]) == 5
    solid = cv.line(np.zeros((10, 100), np.uint8), (0, 5), (99, 5), 255)
    assert _runs(solid[5]) == 1


def test_drawDashedLine_options_and_validation():
    canvas = np.zeros((10, 100), np.uint8)
    cv.drawDashedLine(canvas, (0, 5), (99, 5), 255, dashLength=4, gapLength=16)
    assert _runs(canvas[5]) == 5
    assert canvas[5].sum() < 30 * 255
    untouched = np.zeros((5, 5), np.uint8)
    cv.drawDashedLine(untouched, (2, 2), (2, 2), 255)
    assert not untouched.any()
    with pytest.raises(ValueError):
        cv.drawDashedLine(canvas, (0, 0), (5, 5), 255, dashLength=0)


def test_drawDashedRectangle():
    canvas = np.zeros((60, 80, 3), np.uint8)
    cv.drawDashedRectangle(canvas, (10, 10, 60, 40), (0, 255, 0), dashLength=6)
    assert tuple(canvas[10, 10]) == (0, 255, 0)
    assert _runs(canvas[10, 10:70, 1]) > 3
    assert _runs(canvas[10:50, 69, 1]) > 2
    assert not canvas[30, 30].any()


def test_drawHistogram():
    gradient = np.tile(np.arange(256, dtype=np.uint8), (20, 1))
    plot = cv.drawHistogram(cv.cvtColor(gradient, cv.COLOR_GRAY2BGR), size=(300, 120))
    assert plot.shape == (120, 300, 3)
    assert (plot != 24).any()
    gray_plot = cv.drawHistogram(gradient, log=True)
    assert gray_plot.shape == (200, 512, 3)
    with pytest.raises(ValueError):
        cv.drawHistogram(gradient.astype(np.float32))


# --- largestComponent / skeletonize / cutout ------------------------------------------------


@pytest.fixture
def shapes():
    mask = np.zeros((60, 80), np.uint8)
    cv2.rectangle(mask, (5, 5), (20, 20), 255, cv2.FILLED)
    cv2.rectangle(mask, (30, 10), (75, 50), 255, cv2.FILLED)
    return mask


def test_largestComponent(shapes):
    out = cv.largestComponent(shapes)
    assert out[30, 50] == 255
    assert out[10, 10] == 0
    assert not cv.largestComponent(np.zeros((5, 5), np.uint8)).any()
    with pytest.raises(ValueError):
        cv.largestComponent(shapes, connectivity=5)


def test_skeletonize_bar_becomes_a_thin_line():
    bar = np.zeros((30, 80), np.uint8)
    bar[10:20, 5:75] = 255
    skeleton = cv.skeletonize(bar)
    assert set(np.unique(skeleton)) == {0, 255}
    middle = skeleton[:, 15:65] > 0
    assert (middle.sum(axis=0) == 1).all()  # exactly one pixel thick
    assert cv.connectedComponents(skeleton)[0] == 2  # one piece (plus background)


def test_skeletonize_keeps_a_ring_closed():
    ring = np.zeros((60, 60), np.uint8)
    cv2.circle(ring, (30, 30), 20, 255, 6)
    skeleton = cv.skeletonize(ring)
    assert cv.connectedComponents(skeleton)[0] == 2
    holes = cv.connectedComponents(255 - skeleton, connectivity=4)[0] - 1
    assert holes == 2  # outside and inside: the loop is not broken
    assert skeleton.sum() < ring.sum() / 3
    assert not cv.skeletonize(np.zeros((5, 5), np.uint8)).any()


def test_cutout(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[10:30, 20:40] = 255
    out = cv.cutout(img, mask)
    assert out.shape == (50, 70, 4)
    np.testing.assert_array_equal(out[..., :3], img)
    np.testing.assert_array_equal(out[..., 3], mask)
    soft = cv.cutout(img, mask > 0, feather=3)
    assert 0 < soft[10, 25, 3] < 255  # the edge is soft
    assert soft[20, 30, 3] == 255


def test_cutout_combines_existing_alpha_and_validates(img):
    bgra = np.dstack([img, np.full(img.shape[:2], 128, np.uint8)])
    out = cv.cutout(bgra, np.full(img.shape[:2], 255, np.uint8))
    assert (out[..., 3] == 128).all()
    with pytest.raises(ValueError):
        cv.cutout(img, np.zeros((5, 5), np.uint8))
