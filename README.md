<p align="center"><img src="https://raw.githubusercontent.com/076923/cv2-utils/master/images/logo.png"></p>

Python OpenCV Utilities
-----------------------

Python OpenCV Utilities is a module that provides useful extensions.

It contains useful and convenient features.

## Dependencies

Python OpenCV Utilities requires:

- opencv-python
- pillow
- urllib3
- scikit-learn

## install

To use the Python OpenCV Utilities algorithm, proceed as follows:


    pip install python-opencv-utils


You use **rotateEx** function as follows:

```python3
import cv2u as cv2

src = cv2.imread("image.jpg")
dst = cv2.rotateEx(src, 45)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
```

| src | dst |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/076923/cv2-utils/master/images/rotateEx-src.jpg"> | <img src="https://raw.githubusercontent.com/076923/cv2-utils/master/images/rotateEx-dst.jpg"> |

<br>

You use **DBSCAN** function as follows:

```python3
import cv2u as cv2

src = cv2.imread("image.jpg")
maps, labels = cv2.DBSCAN(src, 1.7, 2)
dst = cv2.drawDBSCAN(src, maps, labels)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()

```

| src | dst |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/076923/cv2-utils/master/images/DBSCAN-src.jpg"> | <img src="https://raw.githubusercontent.com/076923/cv2-utils/master/images/DBSCAN-dst.jpg"> |

<br>

You use **extensions** flag as follows:

```python3

import cv2u as cv2

src = cv2.imread("image.jpg")
dst = cv2.flip(src, cv2.FLIP_MODE_X)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
```

## features

- core
  - DBSCAN
  - drawDBSCAN
  - gammaCorrection
  - centerPoint
  - cartesianToPolar
  - polarToCartesian
  - isInside
  - DHEdge
  - rotateEx
- imgcodecs
  - urlread
  - makeGradient
  - imcolor
  - imcolor_like
- imgproc
  - putTextEx
- flag
  - FLIP_MODE_X
  - FLIP_MODE_Y
  - FLIP_MODE_XY


## Authors

Daehee Yun(s076923@gmail.com)

## License

Apache License 2.0