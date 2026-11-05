# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **OpenCV 4.10 and newer are supported again, alongside OpenCV 5.** 4.10.0.84 is the first OpenCV build for NumPy 2.
- **The OpenCV distribution is chosen with an extra**: `[opencv]` (`opencv-python`), `[headless]`
  (`opencv-python-headless`), `[contrib]` or `[contrib-headless]`. `opencv-python` is no longer a hard dependency,
  so cv2u no longer installs a second, conflicting `cv2` next to a headless or contrib build. Without an extra, an
  already installed OpenCV is used; `import cv2u` explains how to install one if there is none.
  Upgrade with `pip install -U "python-opencv-utils[opencv]"` to keep the 0.1.0 behavior.

### Added

- `imgproc`
  - `transformPoints`: map points through a 2x3 affine or 3x3 perspective matrix, whatever their layout.
  - `cropRotatedRect`: cut out a rotated box (`cv.minAreaRect`) as an upright image.
  - `deskew`: straighten a tilted page; the angle is found from the text lines.
  - `flattenAlpha`: composite a transparent image onto a background color or image.
  - `matchHistograms`, `colorTransfer`: give an image the tones (histogram) or the color mood (Lab statistics)
    of another one.
  - `imageHash`, `hashDistance`: perceptual hashes (dHash, aHash, pHash) for near-duplicate detection.
  - `pixelate`: mosaic the whole image or one box.
  - `drawRoundedRectangle`: rectangles with rounded corners, outlined or filled.
- `core`
  - `boxIoU`: intersection over union of every pair of boxes.
- `imgcodecs`
  - `listImages`: the image files of a folder in natural order, optionally recursive.

## [0.1.0] - 2026-11-05

First official release.

### Highlights

- **Drop-in replacement for `cv2`**: `import cv2u as cv` gives you all of OpenCV plus the extensions below,
  following the official OpenCV `import cv2 as cv` convention.
- **Typed**: type checkers and IDEs see OpenCV's own signatures through cv2u (`cv.imread`, `cv.cvtColor`, ...),
  and every extension has type hints (`py.typed`, checked with `mypy --strict` and pyright).
- **[Design principles](https://076923.github.io/cv2-utils/principles/)** that decide what belongs in cv2u,
  partly enforced by tests (no extension shadows an OpenCV name; OpenCV-style naming).

### Requirements

- Python 3.11+ (tested on 3.11 through 3.15, and on the 3.16 pre-releases).
- OpenCV 5.0+ (`opencv-python`), NumPy 2.0+, Pillow 10.1+. `import cv2u` raises `ImportError` if an older `cv2`
  is installed.
- Optional `cluster` extra (`pip install "python-opencv-utils[cluster]"`) for scikit-learn, needed by
  `meanShiftColors`, `DBSCAN` and `clusterPoints`.

### Features

- `imgcodecs`
  - `imreadEx`, `imwriteEx`: `cv.imread`/`cv.imwrite` that also work with non-ASCII (e.g. Korean) paths on Windows;
    `imwriteEx(..., makeDirs=True)` creates missing folders.
  - `urlread`: read an image from a URL, with `timeout` and `headers`.
  - `toPillow`, `fromPillow`: convert to/from Pillow with the right channel order (also palette, CMYK and
    transparent images).
  - `imencodeBase64`, `imdecodeBase64`: images as base64 text or data URIs.
- `imgproc`
  - `toGray`, `toBGR`, `toBGRA`: convert from any channel count, without errors for images already in that format.
  - `hexToBGR`, `bgrToHex`: convert between hex colors and BGR tuples.
  - `adjustBrightnessContrast`, `adjustHSV`, `autoContrast`, `whiteBalance`: photo-style color adjustments.
  - `sharpnessScore`, `ssim`, `diffMask`: blur detection, structural similarity and change detection.
  - `translate`: shift an image keeping its size.
  - `drawDashedLine`, `drawDashedRectangle`, `drawHistogram`: dashed drawing and histogram plots.
  - `largestComponent`, `skeletonize`, `cutout`: keep the main object, thin to lines, cut out with alpha.
  - `findContoursEx`: `cv.findContours` for any image, sorted by area, with small contours dropped.
  - `fillHoles`, `removeSmallObjects`: binary mask clean-up.
  - `findTemplate`: `cv.matchTemplate` as a list of non-overlapping matches.
  - `resizeEx`: resize keeping the aspect ratio, choosing `INTER_AREA`/`INTER_LINEAR` automatically.
  - `letterbox`: fit into a fixed size keeping the aspect ratio and pad; returns the scale and offset.
  - `fourPointTransform`: warp a quadrilateral (corners in any order) to an upright rectangle.
  - `autoCanny`: `cv.Canny` with thresholds chosen from the median brightness.
  - `equalizeHistEx`: `cv.equalizeHist` for color images too, optionally with CLAHE.
  - `unsharpMask`: unsharp-mask sharpening.
  - `drawLabel`: detector-style box with a label tag (Unicode labels with a TrueType font), drawn in place.
  - `overlayMask`: tint masked pixels, e.g. to visualize a segmentation.
  - `putTextEx`: draw Unicode text (e.g. Korean) with a TrueType font.
- `videoio`
  - `readFrames`: iterate over video frames (with `step` and `start`); the capture is released automatically.
  - `writeVideo`: write frames with the size, color mode and codec detected automatically.
  - `videoInfo`: size, frame rate, frame count and duration in one call.
  - `readFrameAt`: read a single frame by number or time.
- `highgui`
  - `imshowInline`: show images in Jupyter notebooks with their real colors.
  - `imshowEx`: `cv.imshow` that shrinks images larger than the screen.
- `core`
  - `rotateEx`: rotate by any angle without cropping the corners.
  - `hconcatEx`, `vconcatEx`: concatenate images of different sizes and channel counts, with optional gaps.
  - `overlayImage`: alpha-blend a (transparent) image onto another, clipped at the borders.
  - `montage`: arrange images in a grid.
  - `cropEx`: crop safely when the box extends outside the image.
  - `cropToContent`: trim a uniform border.
  - `padToMultiple`, `padToSize`, `splitTiles`, `mergeTiles`: padding and tiling for large images and networks.
  - `inRangeEx`: `cv.inRange` for hue ranges that wrap around, such as red.
  - `imcolor`, `imcolorLike`, `makeGradient` (`"horizontal"`, `"vertical"` or `"raster"`).
  - `centerPoint`, `isInside`, `cartesianToPolar`, `polarToCartesian` (with `angleInDegrees`).
  - `gammaCorrection`, `DHEdge`.
  - `dominantColors`, `colorQuantize`, `kmeansColors`: color palettes, posterization and deterministic color
    k-means, built on `cv.kmeans` (no extra needed).
  - `peakColors`: cv2u's own color clustering. It follows the peaks of the Lab color histogram, finds the number
    of colors by itself, uses every pixel, is deterministic and needs no extra.
  - `meanShiftColors`: color clustering that finds the number of colors by itself.
  - `DBSCAN`, `drawDBSCAN`: color clustering in the Lab color space.
  - `clusterPoints`: group nearby points (keypoints, detections) with DBSCAN.
- `flag`: named constants for magic numbers that OpenCV takes without a constant of its own
  - `FLIP_MODE_X`, `FLIP_MODE_Y`, `FLIP_MODE_XY` (`cv.flip`), `DDEPTH_SAME` (`ddepth=-1`),
    `ALL_CONTOURS` (`cv.drawContours`), `REDUCE_TO_ROW`, `REDUCE_TO_COL` (`cv.reduce`),
    `WAIT_FOREVER` and `KEY_ESC`, `KEY_ENTER`, `KEY_SPACE`, `KEY_TAB`, `KEY_BACKSPACE` (`cv.waitKey`).
- Invalid input raises `ValueError` with a clear message.

### Project

- Documentation site built with [Zensical](https://zensical.org): <https://076923.github.io/cv2-utils/>.
- Test suite with golden-image regression tests and coverage; CI on Linux, Windows and macOS.
- Releases are published from GitHub Actions with PyPI trusted publishing and digital attestations.
- `SECURITY.md`, `CODE_OF_CONDUCT.md` and `CITATION.cff`.

[Unreleased]: https://github.com/076923/cv2-utils/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/076923/cv2-utils/releases/tag/v0.1.0
