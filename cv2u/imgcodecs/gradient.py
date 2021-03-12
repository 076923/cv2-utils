import cv2
import numpy as np

def makeGradient(shape, start, end):
    
    def gradient(_start, _end):
        return np.linspace(_start, _end, num=shape[0]*shape[1], endpoint=True, retstep=False, dtype=np.uint8)

    if shape[2] == 1:
        start = start[0] if len(start) > 1 else start
        end = end[0] if len(end) > 1 else end
        img = gradient(start, end).reshape(*shape)
        
    elif shape[2] == 3:
        b = gradient(start[0], end[0]).reshape(*shape[:2], 1)
        g = gradient(start[1], end[1]).reshape(*shape[:2], 1)
        r = gradient(start[2], end[2]).reshape(*shape[:2], 1)
        img = cv2.merge((b, g, r))
    
    return img