import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def putTextEx(src, text, org, fontPath, fontSize, color):
    if len(src.shape) == 2 or src.shape[2] == 1:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    font = ImageFont.truetype(fontPath, size=fontSize)
    dst = Image.fromarray(src)
    img = ImageDraw.Draw(dst)
    img.text(org, text, font=font, fill=color)
    return np.array(dst)