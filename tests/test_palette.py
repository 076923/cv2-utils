import cv2
import numpy as np
import pytest

from cv2u.imgproc.palette import (
    CSS_COLORS,
    blendColors,
    colorDistance,
    colormapPalette,
    colorName,
    complementaryColor,
    darkenColor,
    gradientPalette,
    invertColor,
    lightenColor,
    nearestColor,
    randomColors,
    textColorFor,
)

RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)


def _is_color(c):
    return isinstance(c, tuple) and len(c) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in c)


def test_css_colors():
    assert len(CSS_COLORS) == 148
    assert CSS_COLORS["red"] == RED and CSS_COLORS["steelblue"] == (180, 130, 70)
    assert CSS_COLORS["rebeccapurple"] == (153, 51, 102)
    assert all(_is_color(c) for c in CSS_COLORS.values())


def test_lighten_darken():
    assert lightenColor(RED, 0) == RED and darkenColor(RED, 0) == RED
    assert lightenColor(RED, 1) == (255, 255, 255) and darkenColor(RED, 1) == (0, 0, 0)
    assert lightenColor(RED, 0.5) == (128, 128, 255)
    assert darkenColor(RED, 0.5) == (0, 0, 128)
    assert darkenColor(np.array([0, 0, 255.0]), 0.5) == (0, 0, 128)  # arrays and floats accepted
    assert _is_color(lightenColor((17, 99, 200), 0.3))


@pytest.mark.parametrize("fn", [lightenColor, darkenColor])
def test_lighten_darken_errors(fn):
    with pytest.raises(ValueError, match="amount"):
        fn(RED, 1.5)
    with pytest.raises(ValueError, match="amount"):
        fn(RED, float("nan"))
    with pytest.raises(ValueError, match="color"):
        fn((0, 0, 300), 0.5)
    with pytest.raises(ValueError, match="color"):
        fn((0, 0), 0.5)


def test_blend():
    assert blendColors(RED, BLUE, 0) == RED and blendColors(RED, BLUE, 1) == BLUE
    assert blendColors((0, 0, 0), (255, 255, 255)) == (128, 128, 128)
    with pytest.raises(ValueError, match="t must"):
        blendColors(RED, BLUE, 2)
    with pytest.raises(ValueError, match="b must"):
        blendColors(RED, (-1, 0, 0))


def test_complementary_and_invert():
    assert complementaryColor(RED) == (255, 255, 0)  # cyan
    assert complementaryColor((0, 0, 128)) == (128, 128, 0)  # dark red -> dark cyan
    assert complementaryColor((128, 128, 128)) == (128, 128, 128)
    assert invertColor(RED) == (255, 255, 0)
    assert invertColor((0, 0, 128)) == (255, 255, 127)
    with pytest.raises(ValueError, match="color"):
        invertColor((256, 0, 0))
    with pytest.raises(ValueError, match="color"):
        complementaryColor((0, 0, float("inf")))


def test_color_distance():
    assert colorDistance(RED, RED) == 0
    assert colorDistance((0, 0, 0), (255, 255, 255)) == pytest.approx(100, abs=0.1)
    assert colorDistance((0, 0, 0), (255, 255, 255), "bgr") == pytest.approx(255 * 3**0.5)
    assert colorDistance(RED, (0, 0, 250)) < colorDistance(RED, GREEN)
    with pytest.raises(ValueError, match="space"):
        colorDistance(RED, GREEN, "hsv")
    with pytest.raises(ValueError, match="a must"):
        colorDistance((0, 0, 999), GREEN)


def test_nearest_color():
    palette = [RED, GREEN, BLUE, (255, 255, 255)]
    assert nearestColor((0, 10, 240), palette) == 0
    assert nearestColor((250, 250, 250), palette) == 3
    assert nearestColor((10, 200, 10), np.array(palette), space="bgr") == 1
    with pytest.raises(ValueError, match="empty"):
        nearestColor(RED, [])
    with pytest.raises(ValueError, match="palette"):
        nearestColor(RED, [(0, 0, 300)])
    with pytest.raises(ValueError, match="space"):
        nearestColor(RED, palette, space="xyz")


def test_color_name():
    assert colorName(RED) == "red"
    assert colorName((0, 0, 250)) == "red"
    assert colorName((255, 255, 255)) == "white"
    assert colorName((128, 128, 128)) == "gray"  # alphabetically before "grey"
    assert colorName((255, 255, 0)) == "aqua"  # before "cyan"
    assert colorName((180, 130, 70)) == "steelblue"
    assert colorName((30, 20, 220)) == "crimson"
    with pytest.raises(ValueError, match="color"):
        colorName((0, 0))


def test_text_color_for():
    assert textColorFor((255, 255, 255)) == (0, 0, 0)
    assert textColorFor((0, 0, 0)) == (255, 255, 255)
    assert textColorFor((0, 255, 255)) == (0, 0, 0)  # yellow is light
    assert textColorFor((255, 0, 0)) == (255, 255, 255)  # blue is dark
    with pytest.raises(ValueError, match="background"):
        textColorFor((0, 0, -5))


def test_random_colors():
    colors = randomColors(5, seed=1)
    assert len(colors) == 5 and all(_is_color(c) for c in colors)
    assert colors == randomColors(5, seed=1)
    assert colors != randomColors(5, seed=2)
    assert all(max(c) == 255 for c in colors)  # value 1 -> one channel at 255
    assert randomColors(0) == []
    pastel = randomColors(4, seed=0, saturation=0.2, value=0.9)
    assert all(min(c) >= 180 and max(c) <= 230 for c in pastel)
    with pytest.raises(ValueError, match="count"):
        randomColors(-1)
    with pytest.raises(ValueError, match="saturation"):
        randomColors(2, saturation=2)
    with pytest.raises(ValueError, match="value"):
        randomColors(2, value=float("nan"))


def test_gradient_palette():
    assert gradientPalette((0, 0, 0), (255, 255, 255), 3) == [(0, 0, 0), (128, 128, 128), (255, 255, 255)]
    assert gradientPalette(RED, BLUE, 2) == [RED, BLUE]
    assert gradientPalette(RED, BLUE, 1) == [RED]
    assert gradientPalette(RED, BLUE, 0) == []
    with pytest.raises(ValueError, match="count"):
        gradientPalette(RED, BLUE, -2)
    with pytest.raises(ValueError, match="end"):
        gradientPalette(RED, (0, 0, 256), 2)


def test_colormap_palette():
    colors = colormapPalette(cv2.COLORMAP_JET, 5)
    assert len(colors) == 5 and all(_is_color(c) for c in colors)
    expected = cv2.applyColorMap(np.array([[0, 64, 128, 191, 255]], np.uint8), cv2.COLORMAP_JET)[0]
    assert colors == [tuple(int(v) for v in c) for c in expected]
    assert colormapPalette(cv2.COLORMAP_VIRIDIS, 1) == colormapPalette(cv2.COLORMAP_VIRIDIS, 2)[:1]
    assert colormapPalette(cv2.COLORMAP_HOT, 0) == []
    with pytest.raises(ValueError, match="count"):
        colormapPalette(cv2.COLORMAP_HOT, -1)
