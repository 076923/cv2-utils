import math
import cv2
import numpy as np


def gammaCorrection(src, gamma=1.0):
    table = np.array([math.pow((i / 255.0), 1.0 / gamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(src, table.astype(np.uint8))


def centerPoint(contour):
    moment = cv2.moments(contour)
    x = moment['m10'] / moment['m00']
    y = moment['m01'] / moment['m00']
    return x, y


def cartesianToPolar(x, y):
    return np.arctan2(y, x), np.hypot(x, y)


def polarToCartesian(theta, rho):
    return rho * np.cos(theta), rho * np.sin(theta)


def isInside(p, polygon):
    crosses = 0
    if len(polygon.shape) == 3:
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)

            if (polygon[i][0][1] > p[1]) != (polygon[j][0][1] > p[1]):
                atX = (polygon[j][0][0] - polygon[i][0][0]) * (p[1] - polygon[i][0][1]) / (polygon[j][0][1] - polygon[i][0][1]) + polygon[i][0][0]

                if p[0] < atX:
                    crosses = crosses + 1
    else:
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)

            if (polygon[i][1] > p[1]) != (polygon[j][1] > p[1]):
                atX = (polygon[j][0] - polygon[i][0]) * (p[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]

                if p[0] < atX:
                    crosses = crosses + 1

    return crosses % 2 > 0


def DHEdge(src):
    img = src.copy()
    view = np.lib.stride_tricks.as_strided(src, shape = (3, 3, src.shape[0] - 2, src.shape[1] - 2), strides = src.strides * 2)
    img[1:-1, 1:-1] = view.max(axis=(0, 1)) - view.min(axis=(0, 1))
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, 1, -1]])
    sharpen = cv2.filter2D(img, -1, kernel)
    dst = np.clip(math.pi * sharpen - 255, 0, 255).astype(np.uint8)
    return dst