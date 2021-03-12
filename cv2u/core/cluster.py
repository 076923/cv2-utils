import cv2
import numpy as np
from sklearn.cluster import DBSCAN as skDBSCAN


def DBSCAN(src, eps, min_samples):
    arr = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).reshape(-1, src.shape[2])
    clustering = skDBSCAN(eps=eps, min_samples=min_samples).fit(arr)
    labels = clustering.labels_ + 1
    maps = labels.reshape(src.shape[:2])
    return maps, labels
    
    
def drawDBSCAN(src, maps, labels):
    colors = []
    for lb in set(labels):
        mask = np.where(maps == lb, 255, 0).astype(np.uint8)
        color = list(map(int, list(cv2.mean(src, mask)[:src.shape[2]])))
        colors.append(np.array(color, dtype=np.uint8))

    colors = np.asarray(colors)    
    dst = colors[labels].astype(np.uint8).reshape(src.shape)
    return dst
    