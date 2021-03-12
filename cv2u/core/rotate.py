import math
import cv2


def rotateEx(src, angle, scale=1, flags=None, borderMode=None, borderValue=None):
    height, width = src.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_width = int((height * abs(sin)) + (width * abs(cos)))
    bound_height = int((height * abs(cos)) + (width * abs(sin)))

    matrix[0, 2] += (bound_width / 2) - center[0]
    matrix[1, 2] += (bound_height / 2) - center[1]

    dst = cv2.warpAffine(src, matrix, (bound_width, bound_height), flags=None, borderMode=None, borderValue=None)
    return dst
