import cv2
import numpy as np
from urllib.request import urlopen


def urlread(url, flags=cv2.IMREAD_UNCHANGED):
    response = urlopen(url)
    img = np.asarray(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(img, flags)
    return img