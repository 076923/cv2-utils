import cv2
import numpy as np
import pytest

from cv2u.core.points import (
    angleBetween,
    clipPoints,
    denormalizePoints,
    distancePoints,
    flipPoints,
    lineIntersection,
    nearestPoints,
    normalizePoints,
    orderPoints,
    pointLineDistance,
    pointsBoundingBox,
    polygonCentroid,
    rotatePoints,
    scalePoints,
    translatePoints,
)

SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)


def test_rotate_points_matches_warp_affine_convention():
    pts = np.array([[30.0, 10.0], [5.0, 5.0]])
    M = cv2.getRotationMatrix2D((20, 20), 37, 1.5)
    expected = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
    out = rotatePoints(pts, 37, center=(20, 20), scale=1.5)
    assert out.dtype == np.float64 and out.shape == (2, 2)
    assert np.allclose(out, expected)
    assert np.allclose(rotatePoints((1, 0), 90), (0, -1))  # counter-clockwise on screen (y down)
    assert rotatePoints(SQUARE.reshape(4, 1, 2), 10).shape == (4, 1, 2)
    assert np.allclose(rotatePoints(SQUARE, 0), SQUARE)


def test_rotate_points_errors():
    with pytest.raises(ValueError, match="angle"):
        rotatePoints(SQUARE, float("nan"))
    with pytest.raises(ValueError, match="scale"):
        rotatePoints(SQUARE, 10, scale=0)
    with pytest.raises(ValueError, match="center"):
        rotatePoints(SQUARE, 10, center=(float("inf"), 0))
    with pytest.raises(ValueError, match="points"):
        rotatePoints([[1, 2, 3]], 10)
    with pytest.raises(ValueError, match="finite"):
        rotatePoints([[1, float("nan")]], 10)


def test_scale_translate():
    assert np.array_equal(scalePoints(SQUARE, 2), SQUARE * 2)
    assert np.array_equal(scalePoints(SQUARE, (2, 0.5)), SQUARE * (2, 0.5))
    assert np.array_equal(scalePoints(SQUARE, 2, center=(5, 5)), SQUARE * 2 - 5)
    assert np.array_equal(translatePoints(SQUARE, 3, -1), SQUARE + np.array([3, -1]))
    assert translatePoints((1, 2), 1, 1).shape == (2,)
    with pytest.raises(ValueError, match="scale"):
        scalePoints(SQUARE, float("inf"))
    with pytest.raises(ValueError, match="dx"):
        translatePoints(SQUARE, float("nan"), 0)


def test_flip_points_matches_cv_flip():
    img = np.zeros((20, 30), np.uint8)
    pts = np.array([[3, 4], [29, 0], [10, 19]])
    for code in (0, 1, -1):
        flipped = cv2.flip(cv2.circle(img.copy(), (3, 4), 0, 255, -1), code)
        out = flipPoints(pts, (30, 20), code)
        x, y = out[0].astype(int)
        assert flipped[y, x] == 255
    assert np.array_equal(flipPoints(pts, (30, 20), 1), [[26, 4], [0, 0], [19, 19]])
    assert np.array_equal(flipPoints(pts, (30, 20), 0), [[3, 15], [29, 19], [10, 0]])
    assert np.array_equal(flipPoints(pts, (30, 20), -1), [[26, 15], [0, 19], [19, 0]])
    with pytest.raises(ValueError, match="size"):
        flipPoints(pts, (0, 20), 1)


def test_normalize_denormalize_clip():
    norm = normalizePoints(SQUARE, (20, 40))
    assert np.allclose(norm, SQUARE / (20, 40))
    assert np.allclose(denormalizePoints(norm, (20, 40)), SQUARE)
    assert np.array_equal(clipPoints([[-5, 3], [25, 50]], (20, 40)), [[0, 3], [19, 39]])
    assert clipPoints((-1, -1), (5, 5)).shape == (2,)
    for fn in (normalizePoints, denormalizePoints, clipPoints):
        with pytest.raises(ValueError, match="size"):
            fn(SQUARE, (10, -1))


def test_bounding_box():
    assert pointsBoundingBox(SQUARE) == (0, 0, 11, 11)
    assert pointsBoundingBox(SQUARE) == tuple(cv2.boundingRect(SQUARE.astype(np.int32)))
    assert pointsBoundingBox([[1.5, 2.2], [3.1, 2.9]]) == (1, 2, 4, 2)
    assert pointsBoundingBox((4, 5)) == (4, 5, 1, 1)
    with pytest.raises(ValueError, match="empty"):
        pointsBoundingBox(np.empty((0, 2)))


def test_distance_and_nearest():
    a = [[0, 0], [10, 0]]
    b = [[0, 3], [10, 4], [100, 100]]
    dist = distancePoints(a, b)
    assert dist.shape == (2, 3) and dist.dtype == np.float64
    assert dist[0, 0] == 3 and dist[1, 1] == 4
    assert distancePoints((0, 0), (3, 4))[0, 0] == 5
    indices, distances = nearestPoints(a, b)
    assert indices.tolist() == [0, 1] and distances.tolist() == [3, 4]
    assert distancePoints(np.empty((0, 2)), b).shape == (0, 3)
    with pytest.raises(ValueError, match="empty"):
        nearestPoints(a, np.empty((0, 2)))
    with pytest.raises(ValueError, match="b must"):
        distancePoints(a, [[1, 2, 3]])


def test_order_points():
    shuffled = np.array([[10, 10], [0, 10], [0, 0], [10, 0]])
    assert np.array_equal(orderPoints(shuffled), SQUARE)
    assert np.array_equal(orderPoints(shuffled.reshape(4, 1, 2)), SQUARE)
    assert orderPoints(SQUARE).dtype == np.float64
    with pytest.raises(ValueError, match="4 points"):
        orderPoints(SQUARE[:3])


def test_polygon_centroid():
    assert polygonCentroid(SQUARE) == (5.0, 5.0)
    assert polygonCentroid(SQUARE[::-1]) == (5.0, 5.0)  # orientation does not matter
    triangle = [[0, 0], [6, 0], [0, 3]]
    assert polygonCentroid(triangle) == pytest.approx((2.0, 1.0))
    m = cv2.moments(np.array(triangle, np.int32))
    assert polygonCentroid(triangle) == pytest.approx((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    assert polygonCentroid([[0, 0], [4, 0]]) == (2.0, 0.0)  # no area: mean
    assert polygonCentroid((3, 4)) == (3.0, 4.0)
    with pytest.raises(ValueError, match="empty"):
        polygonCentroid(np.empty((0, 2)))


def test_angle_between():
    assert angleBetween((0, 0), (1, 0)) == 0
    assert angleBetween((0, 0), (1, 1)) == 45
    assert angleBetween((0, 0), (0, -1)) == -90
    assert angleBetween((0, 0), (0, 1), degrees=False) == pytest.approx(np.pi / 2)
    with pytest.raises(ValueError, match="p2"):
        angleBetween((0, 0), (1, float("nan")))
    with pytest.raises(ValueError, match="p1"):
        angleBetween([[0, 0], [1, 1]], (1, 1))


def test_line_intersection():
    assert lineIntersection(((0, 0), (1, 1)), ((0, 1), (1, 0))) == (0.5, 0.5)
    assert lineIntersection(((0, 0), (1, 0)), ((5, -1), (5, 1))) == (5, 0)
    assert lineIntersection(((0, 0), (2, 0)), ((10, 5), (12, 5))) is None  # parallel
    assert lineIntersection(((0, 0), (1, 1)), ((2, 2), (3, 3))) is None  # coincident
    assert lineIntersection(((0, 0), (1, 0)), ((5, 1), (5, 2))) == (5, 0)  # beyond the segments
    with pytest.raises(ValueError, match="line2"):
        lineIntersection(((0, 0), (1, 0)), ((5, 1), (5, 1)))
    with pytest.raises(ValueError, match="line1"):
        lineIntersection(((0, 0), (1, 0), (2, 0)), ((5, 1), (5, 2)))


def test_point_line_distance():
    assert pointLineDistance((0, 1), ((0, 0), (1, 0))) == 1
    assert pointLineDistance((5, 3), ((0, 0), (10, 0))) == 3
    assert pointLineDistance((13, 4), ((0, 0), (10, 0))) == 4  # infinite line
    assert pointLineDistance((13, 4), ((0, 0), (10, 0)), segment=True) == 5  # to the end point
    assert pointLineDistance((-3, -4), ((0, 0), (10, 0)), segment=True) == 5
    assert pointLineDistance((1, 1), ((0, 0), (2, 2))) == pytest.approx(0)
    with pytest.raises(ValueError, match="line"):
        pointLineDistance((0, 0), ((1, 1), (1, 1)))
    with pytest.raises(ValueError, match="point"):
        pointLineDistance((0, float("inf")), ((0, 0), (1, 0)))
