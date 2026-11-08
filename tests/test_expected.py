"""Things that look like they should work in OpenCV but do not: each test shows the OpenCV behavior first."""

import cv2
import numpy as np
import pytest

import cv2u as cv

# --- toGray / toBGR / toBGRA --------------------------------------------------------


@pytest.fixture
def images():
    bgr = np.zeros((4, 5, 3), np.uint8)
    bgr[:] = (10, 20, 30)
    return {
        "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
        "gray3d": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[..., None],
        "bgr": bgr,
        "bgra": cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA),
    }


def test_cvtColor_fails_on_images_that_are_already_converted(images):
    with pytest.raises(cv2.error):
        cv2.cvtColor(images["gray"], cv2.COLOR_BGR2GRAY)
    with pytest.raises(cv2.error):
        cv2.cvtColor(images["bgr"], cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("name", ["gray", "gray3d", "bgr", "bgra"])
def test_conversions_accept_any_input(images, name):
    src = images[name]
    gray, bgr, bgra = cv.toGray(src), cv.toBGR(src), cv.toBGRA(src)
    assert gray.shape == (4, 5)
    assert bgr.shape == (4, 5, 3)
    assert bgra.shape == (4, 5, 4)
    assert gray.dtype == bgr.dtype == bgra.dtype == np.uint8
    assert int(gray[0, 0]) == 22  # luminance of BGR (10, 20, 30), whatever the input format
    assert bgra[0, 0, 3] == 255


def test_conversions_return_copies(images):
    for func, name in ((cv.toGray, "gray"), (cv.toBGR, "bgr"), (cv.toBGRA, "bgra")):
        out = func(images[name])
        out[...] = 0
        assert images[name].any()


def test_toBGRA_alpha(images):
    assert (cv.toBGRA(images["bgr"], alpha=128)[..., 3] == 128).all()
    bgra = images["bgra"].copy()
    bgra[..., 3] = 7
    assert (cv.toBGRA(bgra)[..., 3] == 7).all()  # existing alpha is kept
    assert (cv.toBGRA(np.zeros((2, 2, 3), np.float32))[..., 3] == 1.0).all()


def test_conversions_keep_16_bit(images):
    src = images["bgr"].astype(np.uint16) * 257
    assert cv.toGray(src).dtype == np.uint16
    assert cv.toBGRA(src)[0, 0, 3] == 65535


def test_conversions_validation():
    with pytest.raises(ValueError):
        cv.toGray(np.zeros((2, 2, 2), np.uint8))


# --- imwriteEx(makeDirs=True) --------------------------------------------------------


def test_imwrite_into_missing_folder_fails_silently(tmp_path):
    img = np.zeros((2, 2, 3), np.uint8)
    assert cv2.imwrite(str(tmp_path / "missing" / "a.png"), img) is False  # no exception, no warning
    assert cv.imwriteEx(tmp_path / "missing" / "a.png", img) is False  # same default
    assert cv.imwriteEx(tmp_path / "새 폴더" / "sub" / "a.png", img, makeDirs=True) is True
    assert cv.imreadEx(tmp_path / "새 폴더" / "sub" / "a.png") is not None


# --- findContoursEx -------------------------------------------------------------------


@pytest.fixture
def blobs():
    img = np.zeros((60, 80), np.uint8)
    cv2.rectangle(img, (2, 2), (5, 5), 255, cv2.FILLED)  # area 9
    cv2.rectangle(img, (10, 10), (40, 40), 255, cv2.FILLED)  # area 900
    cv2.circle(img, (60, 30), 10, 255, cv2.FILLED)  # area ~ 300
    return img


def test_findContours_needs_single_channel(blobs):
    with pytest.raises(cv2.error):
        cv2.findContours(cv2.cvtColor(blobs, cv2.COLOR_GRAY2BGR), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def test_findContoursEx_sorted_by_area(blobs):
    areas = [cv2.contourArea(c) for c in cv.findContoursEx(blobs)]
    assert len(areas) == 3
    assert areas == sorted(areas, reverse=True)
    assert areas[0] == pytest.approx(900)


def test_findContoursEx_minArea_and_unsorted(blobs):
    assert len(cv.findContoursEx(blobs, minArea=50)) == 2
    unsorted = cv.findContoursEx(blobs, sort=False)
    opencv, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert [c.tolist() for c in unsorted] == [c.tolist() for c in opencv]


@pytest.mark.parametrize("kind", ["bgr", "bool", "dark-color"])
def test_findContoursEx_any_input(blobs, kind):
    if kind == "bgr":
        image = cv2.cvtColor(blobs, cv2.COLOR_GRAY2BGR)
    elif kind == "bool":
        image = blobs > 0
    else:  # a color that is 0 in grayscale must still count as foreground
        image = np.zeros((*blobs.shape, 3), np.uint8)
        image[blobs > 0] = (1, 0, 0)
        assert not cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).any()
    assert len(cv.findContoursEx(image)) == 3


def test_findContoursEx_output_works_with_drawContours(blobs):
    canvas = np.zeros_like(blobs)
    cv2.drawContours(canvas, cv.findContoursEx(blobs, minArea=50), cv.ALL_CONTOURS, 255, cv2.FILLED)
    assert canvas[25, 25] == 255
    assert canvas[3, 3] == 0


# --- fillHoles / removeSmallObjects -----------------------------------------------------


@pytest.fixture
def ring():
    mask = np.zeros((40, 40), np.uint8)
    cv2.circle(mask, (20, 20), 15, 255, cv2.FILLED)
    cv2.circle(mask, (20, 20), 6, 0, cv2.FILLED)  # a hole
    mask[0:3, 0:3] = 255  # a speck touching the border
    return mask


def test_fillHoles(ring):
    filled = cv.fillHoles(ring)
    assert filled.dtype == np.uint8
    assert filled[20, 20] == 255  # the hole is filled
    assert filled[39, 39] == 0  # background connected to the border stays
    assert filled[1, 1] == 255


def test_fillHoles_maxHoleArea(ring):
    assert cv.fillHoles(ring, maxHoleArea=10)[20, 20] == 0  # the hole (~113 px) is too large
    assert cv.fillHoles(ring, maxHoleArea=500)[20, 20] == 255


def test_fillHoles_does_not_fill_open_regions():
    cup = np.zeros((20, 20), np.uint8)
    cup[5:15, 5] = cup[5:15, 14] = cup[14, 5:15] = 255  # a U shape open at the top
    assert cv.fillHoles(cup)[10, 10] == 0


def test_removeSmallObjects(ring):
    cleaned = cv.removeSmallObjects(ring, minArea=20)
    assert cleaned[1, 1] == 0  # the 9-pixel speck is gone
    assert cleaned[20, 10] == 255  # the ring stays
    assert set(np.unique(cleaned)) <= {0, 255}


def test_removeSmallObjects_connectivity():
    diagonal = np.zeros((5, 5), bool)
    diagonal[0, 0] = diagonal[1, 1] = diagonal[2, 2] = True  # 3 pixels touching only at corners
    assert cv.removeSmallObjects(diagonal, minArea=3, connectivity=8).sum() == 3 * 255
    assert cv.removeSmallObjects(diagonal, minArea=3, connectivity=4).sum() == 0
    with pytest.raises(ValueError):
        cv.removeSmallObjects(diagonal, minArea=3, connectivity=6)


# --- findTemplate ----------------------------------------------------------------------


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 60, (120, 160), dtype=np.uint8)
    patch = rng.integers(100, 256, (12, 16), dtype=np.uint8)
    for x, y in ((10, 20), (90, 30), (50, 90)):
        img[y : y + 12, x : x + 16] = patch
    return img, patch


def test_matchTemplate_only_returns_a_score_map(scene):
    img, patch = scene
    assert cv2.matchTemplate(img, patch, cv2.TM_CCOEFF_NORMED).shape == (109, 145)


def test_findTemplate_finds_every_copy(scene):
    img, patch = scene
    matches = cv.findTemplate(img, patch)
    assert matches.shape == (3, 5) and matches.dtype == np.float64
    assert sorted((x, y) for x, y, *_ in matches.tolist()) == [(10, 20), (50, 90), (90, 30)]
    assert (matches[:, 2] == 16).all() and (matches[:, 3] == 12).all()
    scores = matches[:, 4].tolist()
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0, abs=1e-4)
    assert cv.findTemplate(img, patch, threshold=1.0, maxCount=0).shape == (0, 5)  # always an array


def test_findTemplate_maxCount_and_threshold(scene):
    img, patch = scene
    assert len(cv.findTemplate(img, patch, maxCount=2)) == 2
    assert cv.findTemplate(img[:, :5], patch[:, :5], threshold=1.0, maxCount=1) is not None


@pytest.mark.parametrize("method", [cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED])
def test_findTemplate_other_methods(scene, method):
    img, patch = scene
    matches = cv.findTemplate(img, patch, threshold=0.95, method=method)
    assert (10, 20) in [(x, y) for x, y, *_ in matches.tolist()]


def test_findTemplate_mixed_channels(scene):
    img, patch = scene
    matches = cv.findTemplate(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), patch)
    assert len(matches) == 3


@pytest.mark.parametrize(
    ("kwargs", "templ_shape"),
    [
        ({"method": cv2.TM_CCOEFF}, (4, 4)),
        ({"threshold": 1.5}, (4, 4)),
        ({"threshold": float("nan")}, (4, 4)),
        ({"maxCount": -1}, (4, 4)),
        ({}, (200, 10)),
        ({}, (0, 4)),
    ],
)
def test_findTemplate_validation(scene, kwargs, templ_shape):
    img, _ = scene
    with pytest.raises(ValueError):
        cv.findTemplate(img, np.zeros(templ_shape, np.uint8), **kwargs)


# --- cropToContent ---------------------------------------------------------------------


def test_cropToContent_white_margin():
    page = np.full((100, 80, 3), 255, np.uint8)
    page[20:50, 10:70] = (0, 0, 0)
    out = cv.cropToContent(page)
    assert out.shape == (30, 60, 3)
    assert cv.cropToContent(page, margin=5).shape == (40, 70, 3)


def test_cropToContent_tolerance_and_background():
    frame = np.zeros((50, 50), np.uint8)
    frame[:, :] = 3  # dark noise-level border
    frame[10:40, 5:45] = 200
    assert cv.cropToContent(frame).shape == (30, 40)
    noisy = frame.copy()
    noisy[0, 49] = 8  # one slightly brighter border pixel
    assert cv.cropToContent(noisy).shape == (40, 45)  # without tolerance the corner pixel counts as content
    assert cv.cropToContent(noisy, tolerance=10).shape == (30, 40)
    assert cv.cropToContent(frame, background=200).shape == (50, 50)  # everything differs from 200 except content


def test_cropToContent_all_background_returns_a_copy():
    flat = np.full((5, 5), 9, np.uint8)
    out = cv.cropToContent(flat)
    assert out.shape == flat.shape
    assert out is not flat


def test_cropToContent_validation():
    with pytest.raises(ValueError):
        cv.cropToContent(np.zeros((3, 3), np.uint8), tolerance=-1)
