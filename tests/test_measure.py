import math

import cv2
import numpy as np
import pytest

from cv2u.imgproc.measure import (
    Ellipse,
    Feret,
    PixelScale,
    ShapeDescriptors,
    contourCurvature,
    contourDistance,
    feretDiameter,
    fitEllipseEx,
    huMomentsEx,
    intensityProfile,
    matchShapesEx,
    measureArea,
    measureLength,
    radialProfile,
    shapeDescriptors,
)

SQUARE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.int32)
RECT = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], np.float64)


def circle(radius=50, center=(100, 100), delta=5):
    return cv2.ellipse2Poly(center, (radius, radius), 0, 0, 360, delta)


def rotated(points, angle):
    """Rotate points about their centroid by ``angle`` degrees (image coordinates, y down)."""
    center = points.mean(axis=0)
    a = math.radians(angle)
    m = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (points - center) @ m.T + center


# --- shapeDescriptors ---------------------------------------------------------------


def test_shapeDescriptors_square():
    s = shapeDescriptors(SQUARE)
    assert isinstance(s, ShapeDescriptors)
    assert s.area == 10000
    assert s.perimeter == 400
    assert s.circularity == pytest.approx(math.pi / 4)
    assert s.compactness == pytest.approx(4 / math.pi)
    assert s.solidity == pytest.approx(1)
    assert s.convexity == pytest.approx(1)
    assert s.aspectRatio == pytest.approx(1)
    assert s.eccentricity == pytest.approx(0, abs=1e-6)
    assert s.equivalentDiameter == pytest.approx(math.sqrt(40000 / math.pi))
    assert s.extent == pytest.approx(10000 / (101 * 101))


def test_shapeDescriptors_circle_is_round():
    s = shapeDescriptors(circle())
    assert s.circularity == pytest.approx(1, abs=0.01)
    assert s.eccentricity == pytest.approx(0, abs=0.05)
    assert s.aspectRatio == pytest.approx(1, abs=0.02)
    assert s.equivalentDiameter == pytest.approx(100, abs=0.5)


@pytest.mark.parametrize("angle", [0, 30, 120])
def test_shapeDescriptors_rectangle_rotation_invariant(angle):
    s = shapeDescriptors(rotated(RECT, angle))
    assert s.aspectRatio == pytest.approx(2, abs=1e-6)
    # Uniform rectangle: eigenvalues are (w^2, h^2) / 12 -> eccentricity sqrt(1 - 1/4).
    assert s.eccentricity == pytest.approx(math.sqrt(0.75), abs=1e-6)
    assert s.orientation == pytest.approx(angle, abs=1e-6)
    assert s.circularity == pytest.approx(4 * math.pi * 5000 / 300**2, abs=1e-6)


def test_shapeDescriptors_dented_shape_is_not_convex():
    dented = np.array([[0, 0], [100, 0], [100, 100], [50, 30], [0, 100]])
    s = shapeDescriptors(dented)
    assert s.solidity < 0.8
    assert s.convexity < 1
    assert s.circularity < math.pi / 4


def test_shapeDescriptors_degenerate():
    s = shapeDescriptors([[0, 0], [10, 0], [20, 0]])
    assert s.area == 0
    assert s.circularity == 0
    assert s.compactness == 0
    assert s.aspectRatio == 0


@pytest.mark.parametrize("contour", [[], [[0, 0], [1, 1]], [[0, 0, 0]] * 3, [[0, 0], [1, 1], [float("nan"), 2]]])
def test_shapeDescriptors_validation(contour):
    with pytest.raises(ValueError, match="contour"):
        shapeDescriptors(contour)


# --- feretDiameter ------------------------------------------------------------------


def test_feretDiameter_rotated_rectangle():
    f = feretDiameter(cv2.boxPoints(((100, 100), (80, 40), 30)))
    assert isinstance(f, Feret)
    assert f.max == pytest.approx(math.hypot(80, 40), abs=1e-4)
    assert f.min == pytest.approx(40, abs=1e-4)
    assert f.minAngle == pytest.approx(120, abs=1e-3)
    diagonal = math.degrees(math.atan2(40, 80))
    assert min(abs(f.maxAngle - (30 - diagonal)), abs(f.maxAngle - (30 + diagonal))) < 1e-3


def test_feretDiameter_square_min_is_side():
    f = feretDiameter(SQUARE)
    assert f.max == pytest.approx(100 * math.sqrt(2))
    assert f.min == pytest.approx(100)
    assert f.maxAngle in (pytest.approx(45), pytest.approx(135))
    assert f.minAngle in (pytest.approx(0), pytest.approx(90))


def test_feretDiameter_circle():
    f = feretDiameter(circle())
    assert f.max == pytest.approx(100, abs=1.5)
    assert f.min == pytest.approx(100, abs=1.5)


def test_feretDiameter_min_differs_from_minAreaRect_short_side():
    # An equilateral-ish triangle: minAreaRect's short side is its height, which is the min Feret here too; a
    # thin parallelogram shows the difference.
    para = np.array([[0, 0], [100, 0], [130, 10], [30, 10]], np.float64)
    f = feretDiameter(para)
    assert f.max == pytest.approx(math.hypot(130, 10))
    assert f.min == pytest.approx(10)
    assert f.minAngle == pytest.approx(90)


def test_feretDiameter_two_points_and_contour_layout():
    f = feretDiameter(np.array([[[0, 0]], [[30, 40]]], np.int32))
    assert f.max == 50
    assert f.min == 0
    assert f.maxAngle == pytest.approx(math.degrees(math.atan2(40, 30)))
    assert f.minAngle == pytest.approx(f.maxAngle + 90)


def test_feretDiameter_validation():
    with pytest.raises(ValueError, match="at least 2"):
        feretDiameter([[1, 1]])


# --- fitEllipseEx -------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0, 20, 110])
def test_fitEllipseEx_known_ellipse(angle):
    e = fitEllipseEx(cv2.ellipse2Poly((100, 80), (60, 30), angle, 0, 360, 5))
    assert isinstance(e, Ellipse)
    assert e.center == pytest.approx((100, 80), abs=0.2)
    assert e.axes[0] == pytest.approx(120, abs=1)
    assert e.axes[1] == pytest.approx(60, abs=1)
    assert e.axes[0] >= e.axes[1]
    assert e.angle == pytest.approx(angle, abs=1)


def test_fitEllipseEx_major_is_vertical():
    e = fitEllipseEx(cv2.ellipse2Poly((50, 50), (20, 40), 0, 0, 360, 5))
    assert e.axes == pytest.approx((80, 40), abs=1)
    assert e.angle == pytest.approx(90, abs=1)


def test_fitEllipseEx_validation():
    with pytest.raises(ValueError, match="at least 5"):
        fitEllipseEx([[0, 0], [1, 0], [1, 1], [0, 1]])


def test_fitEllipseEx_collinear_points():
    line = np.array([[i, 2 * i] for i in range(6)], np.float32)
    result = cv2.fitEllipse(line)
    values = (*result[0], *result[1], result[2])
    if all(math.isfinite(v) for v in values):
        pytest.skip("this OpenCV build returns a finite ellipse for collinear points")
    with pytest.raises(ValueError, match="degenerate"):
        fitEllipseEx(line)


# --- huMomentsEx --------------------------------------------------------------------


def test_huMomentsEx_contour_invariant_to_scale_and_rotation():
    triangle = np.array([[0, 0], [100, 0], [20, 60]], np.float64)
    base = huMomentsEx(triangle)
    assert base.shape == (7,)
    assert base.dtype == np.float64
    assert np.all(base[:6] != 0)
    np.testing.assert_allclose(huMomentsEx(triangle * 3.0)[:6], base[:6], atol=1e-6)
    np.testing.assert_allclose(huMomentsEx(rotated(triangle, 37))[:6], base[:6], atol=1e-6)
    np.testing.assert_allclose(huMomentsEx(np.flipud(triangle))[:6], base[:6], atol=1e-6)


def test_huMomentsEx_raw_values_and_log_scaling():
    raw = huMomentsEx(RECT, log=False)
    scaled = huMomentsEx(RECT)
    assert raw[0] > 0
    assert scaled[0] == pytest.approx(-math.log10(raw[0]))
    assert np.all(scaled[raw == 0] == 0)


def test_huMomentsEx_image_matches_contour():
    img = np.zeros((200, 200), np.uint8)
    cv2.rectangle(img, (40, 60), (160, 120), 255, cv2.FILLED)
    from_image = huMomentsEx(img)
    from_contour = huMomentsEx(np.array([[40, 60], [161, 60], [161, 121], [40, 121]]))
    np.testing.assert_allclose(from_image[:2], from_contour[:2], atol=0.01)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    np.testing.assert_allclose(huMomentsEx(color), from_image, atol=1e-9)
    # Only the support matters, not the intensity.
    np.testing.assert_allclose(huMomentsEx(img // 5), from_image, atol=1e-9)
    np.testing.assert_allclose(huMomentsEx(img.astype(np.float32) / 255), from_image, atol=1e-9)


def test_huMomentsEx_validation():
    with pytest.raises(ValueError, match="src"):
        huMomentsEx(np.zeros((0, 10), np.uint8))
    with pytest.raises(ValueError, match="src"):
        huMomentsEx(np.zeros((10, 10, 2), np.uint8))


# --- matchShapesEx ------------------------------------------------------------------


def test_matchShapesEx_identical_and_different():
    assert matchShapesEx(SQUARE, SQUARE * 2.0) == pytest.approx(0, abs=1e-9)
    assert matchShapesEx(SQUARE, RECT) > 0.05
    assert matchShapesEx(SQUARE, circle(), cv2.CONTOURS_MATCH_I2) > 0


def test_matchShapesEx_accepts_masks():
    mask = np.zeros((200, 200), np.uint8)
    cv2.rectangle(mask, (40, 40), (140, 140), 255, cv2.FILLED)
    assert matchShapesEx(mask, mask.copy()) == 0
    assert matchShapesEx(mask, SQUARE) < 0.05
    bgr16 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.uint16) * 257
    assert matchShapesEx(bgr16, mask) == pytest.approx(0, abs=1e-6)


def test_matchShapesEx_validation():
    with pytest.raises(ValueError, match="method"):
        matchShapesEx(SQUARE, SQUARE, 99)
    with pytest.raises(ValueError, match="a"):
        matchShapesEx(np.zeros((0, 0), np.uint8), SQUARE)
    with pytest.raises(ValueError, match="b"):
        matchShapesEx(SQUARE, np.zeros((0, 2)))


# --- contourDistance ----------------------------------------------------------------


def test_contourDistance_separated_squares():
    assert contourDistance(SQUARE, SQUARE + np.array([150, 0])) == pytest.approx(50)
    assert contourDistance(SQUARE, SQUARE + np.array([130, 140])) == pytest.approx(50)  # corner to corner: (30, 40)


@pytest.mark.parametrize(
    "other",
    [
        SQUARE + np.array([100, 100]),  # touching at a corner
        SQUARE + np.array([50, 50]),  # overlapping
        SQUARE // 2 + [25, 25],  # inside
        SQUARE * 2 - [50, 50],  # contains
    ],
)
def test_contourDistance_zero_when_touching_or_overlapping(other):
    assert contourDistance(SQUARE, other) == 0
    assert contourDistance(other, SQUARE) == 0


def test_contourDistance_crossing_without_vertices_inside():
    tall = np.array([[-10, -100], [10, -100], [10, 100], [-10, 100]])
    wide = np.array([[-100, -10], [100, -10], [100, 10], [-100, 10]])
    assert contourDistance(tall, wide) == 0


def test_contourDistance_point_and_segment():
    assert contourDistance([[50, 150]], SQUARE) == pytest.approx(50)
    assert contourDistance(SQUARE, [[150, 50], [250, 50]]) == pytest.approx(50)
    assert contourDistance([[0, 0]], [[3, 4]]) == pytest.approx(5)


def test_contourDistance_validation():
    with pytest.raises(ValueError, match="b"):
        contourDistance(SQUARE, [])


# --- contourCurvature ---------------------------------------------------------------


def test_contourCurvature_circle():
    contour = circle(50, delta=10)
    k = contourCurvature(contour, step=2)
    assert k.shape == (len(contour),)
    assert k.dtype == np.float32
    assert np.all(k > 0)
    np.testing.assert_allclose(k, 1 / 50, rtol=0.1)
    np.testing.assert_allclose(contourCurvature(contour[::-1], step=2), -k[::-1], rtol=1e-5)


def test_contourCurvature_findContours_outer_is_negative():
    img = np.zeros((200, 200), np.uint8)
    cv2.circle(img, (100, 100), 50, 255, cv2.FILLED)
    contour = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    k = contourCurvature(contour, step=8)
    assert np.all(k <= 0)
    assert np.median(k) == pytest.approx(-1 / 50, rel=0.3)


def test_contourCurvature_square_corners():
    k = contourCurvature(SQUARE, step=1)
    assert np.all(np.abs(k) > 0)
    straight = np.array([[0, 0], [50, 0], [100, 0], [100, 100], [0, 100]])
    assert contourCurvature(straight, step=1)[1] == 0


def test_contourCurvature_validation():
    with pytest.raises(ValueError, match="step"):
        contourCurvature(SQUARE, step=0)
    with pytest.raises(ValueError, match="contour"):
        contourCurvature([[0, 0], [1, 1]])


# --- intensityProfile ---------------------------------------------------------------


@pytest.fixture
def ramp():
    return np.tile(np.arange(100, dtype=np.uint8), (60, 1))


def test_intensityProfile_horizontal_ramp(ramp):
    profile = intensityProfile(ramp, (0, 10), (99, 10))
    assert profile.shape == (100,)
    assert profile.dtype == np.float32
    np.testing.assert_allclose(profile, np.arange(100))


def test_intensityProfile_subpixel_and_diagonal(ramp):
    profile = intensityProfile(ramp, (10.5, 5), (20.5, 5))
    np.testing.assert_allclose(profile, np.arange(10.5, 21, 1.0), atol=1e-4)
    diagonal = intensityProfile(ramp, (0, 0), (30, 40))
    assert diagonal.shape == (51,)
    np.testing.assert_allclose(diagonal, np.linspace(0, 30, 51), atol=1e-4)


def test_intensityProfile_width_averages_perpendicular():
    img = np.zeros((20, 50), np.float32)
    img[8] = 10
    img[12] = 30
    assert np.all(intensityProfile(img, (0, 10), (49, 10)) == 0)
    assert np.all(intensityProfile(img, (0, 10), (49, 10), width=9) == pytest.approx(40 / 9))


def test_intensityProfile_color_and_single_point():
    img = np.zeros((10, 10, 3), np.uint8)
    img[..., 2] = 200
    profile = intensityProfile(img, (2, 2), (6, 2))
    assert profile.shape == (5, 3)
    np.testing.assert_array_equal(profile[:, 2], 200)
    assert intensityProfile(img, (3, 3), (3, 3)).shape == (1, 3)


def test_intensityProfile_outside_replicates_border(ramp):
    np.testing.assert_array_equal(intensityProfile(ramp, (99, 0), (104, 0)), 99)


def test_intensityProfile_validation(ramp):
    with pytest.raises(ValueError, match="width"):
        intensityProfile(ramp, (0, 0), (5, 5), width=0)
    with pytest.raises(ValueError, match="p1"):
        intensityProfile(ramp, (0,), (5, 5))
    with pytest.raises(ValueError, match="p2"):
        intensityProfile(ramp, (0, 0), (float("inf"), 5))
    with pytest.raises(ValueError, match="src"):
        intensityProfile(np.zeros((0, 5), np.uint8), (0, 0), (1, 1))


# --- radialProfile ------------------------------------------------------------------


def test_radialProfile_distance_image():
    ys, xs = np.mgrid[0:101, 0:101]
    distance = np.hypot(xs - 50, ys - 50).astype(np.float32)
    profile = radialProfile(distance)
    assert profile.shape == (72,)  # farthest corner: round(50 * sqrt(2)) = 71
    np.testing.assert_allclose(profile[:50], np.arange(50), atol=0.5)
    assert profile[0] == 0


def test_radialProfile_center_and_maxRadius():
    ys, xs = np.mgrid[0:40, 0:60]
    img = np.hypot(xs, ys)
    profile = radialProfile(img, (0, 0), maxRadius=10)
    assert profile.shape == (11,)
    np.testing.assert_allclose(profile, np.arange(11), atol=0.5)


def test_radialProfile_color_and_empty_radii():
    img = np.full((5, 5, 3), (10, 20, 30), np.uint8)
    profile = radialProfile(img, maxRadius=10)
    assert profile.shape == (11, 3)
    np.testing.assert_array_equal(profile[0], (10, 20, 30))
    assert np.all(np.isnan(profile[5:]))


def test_radialProfile_validation():
    with pytest.raises(ValueError, match="maxRadius"):
        radialProfile(np.zeros((5, 5), np.uint8), maxRadius=-1)
    with pytest.raises(ValueError, match="center"):
        radialProfile(np.zeros((5, 5), np.uint8), (float("nan"), 0))
    with pytest.raises(ValueError, match="src"):
        radialProfile(np.zeros((5, 5, 2), np.uint8))


# --- PixelScale, measureLength, measureArea -----------------------------------------


def test_PixelScale_conversions():
    scale = PixelScale.fromReference(pixelLength=250, realLength=25, unit="mm")
    assert scale.pixelsPerUnit == 10
    assert scale.unit == "mm"
    assert scale.toReal(35) == 3.5
    assert scale.toPixels(2) == 20
    assert scale.area(400) == 4
    assert PixelScale(4).unit == "mm"
    assert PixelScale(4, "cm").area(16) == 1


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_PixelScale_validation(value):
    with pytest.raises(ValueError, match="pixelsPerUnit"):
        PixelScale(value)
    with pytest.raises(ValueError, match="pixelLength"):
        PixelScale.fromReference(value, 10)
    with pytest.raises(ValueError, match="realLength"):
        PixelScale.fromReference(10, value)


def test_measureLength_and_measureArea():
    scale = PixelScale(10, "mm")
    assert measureLength((0, 0), (30, 40), scale) == 5
    assert measureArea(SQUARE, scale) == 100
    assert measureArea(circle(50), scale) == pytest.approx(math.pi * 25, rel=0.01)
    with pytest.raises(ValueError, match="p2"):
        measureLength((0, 0), (1, float("nan")), scale)
    with pytest.raises(ValueError, match="contour"):
        measureArea([[0, 0], [1, 1]], scale)
