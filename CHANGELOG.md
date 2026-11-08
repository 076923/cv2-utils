# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-08

The first stable release: 162 new APIs (135 -> 297), two new subpackages (`cv2u.calib3d`, `cv2u.dnn`), and a one-time
clean-up of the existing API (consistent names, signatures and validation). From 1.0 on, the public API follows
semantic versioning and renames keep a deprecated alias (see [Design principles](docs/principles.md)).

### Added

- `VideoReader`, `VideoWriterEx` and `CameraStream`: context-manager/iterator capture with seeking, a lazily-opened writer that infers codec, size and color mode from the first frame, and a threaded latest-frame camera reader.
- `FPSCounter`, `frameToTime`, `timeToFrame`, `formatTimestamp`: frame-rate measurement and frame/time conversions.
- `extractFrames`, `trimVideo`, `concatVideos`, `resizeVideo`, `videoToGif`, `imagesToVideo`: video file editing utilities with Unicode-safe paths and automatic directory creation.
- `detectScenes`, `sampleFrames`, `videoThumbnail`: scene-cut detection and evenly spaced frame sampling / contact sheets.
- Data augmentation (`imgproc`): seeded noise, motion blur, vignette, JPEG artifacts, random erase, solarize and posterize (`addNoise`, `motionBlur`, `vignette`, `jpegArtifacts`, `randomErase`, `solarize`, `posterize`).
- Seeded random transforms that keep detection boxes aligned: `randomCrop`, `randomFlip`, `randomRotate`, `randomScale`, `randomPerspective`; photometric `randomBrightnessContrast`, `randomHSV`, `randomBlur`.
- Multi-image augmentations `mosaic` (YOLO-style 4-image mosaic with boxes), `mixup` and `cutMix` (both return the mixing weight `lam` for labels).
- `maskToPolygons`, `polygonsToMask`, `boxesToPolygons`, `polygonsToBoxes`, `polygonArea`, `polygonIoU`, `maskIoU`: polygon annotations and overlap measures (`polygonIoU` rasterizes at a configurable resolution, so it works for concave polygons).
- `rleEncode`, `rleDecode`: COCO uncompressed RLE (column-major, pycocotools-compatible) without a pycocotools dependency.
- `boxesToYolo`, `yoloToBoxes`, `readYoloLabels`, `writeYoloLabels`: YOLO normalized rows and `.txt` label files on any path, with `makeDirs`.
- `labelMapToColor`, `colorToLabelMap`, `instanceMasksToLabelMap`, `labelMapToMasks`, `drawSegmentation`, `drawPolygons`: label maps, instance masks and their visualization with distinct class colors.
- `flipBoxes`, `flipPoints`, `flipPolygons`: flip annotations together with `cv.flip` (same `flipCode` semantics as `FLIP_MODE_*`).
- `cv2u.dnn` subpackage (mirrors `cv.dnn`; every `cv2.dnn` name passes through): `loadModel`, `readClassNames`, `preprocessImage`, `inferImage`, `postprocessDetections`, `postprocessClassification`, `softmax`, `sigmoid`, `drawDetections`, and the `PreprocessInfo` / `Detections` named tuples for running detectors and classifiers end to end.
- `heatmap`, `applyColorMapEx` and `colorBar`: colorize float maps of any size, overlay them on an image and draw a legend strip.
- `drawArrow` (pixel-sized heads, both ends), `drawRotatedRect`/`drawRotatedBoxes` (from `cv.minAreaRect`), `drawPolygon` (float points, translucent fill), `drawSkeleton` with `COCO_SKELETON`, `drawTrajectory` (fading tracks) and `drawCrosshair`.
- HUD helpers `drawTextBlock`, `drawFPS`, `drawProgressBar` and `drawLegend` with translucent panels.
- `compareImages` (side, vertical, split, blend, diff), `labeledMontage` (titled grid) and `blendImages` (weighted average of N images).
- `imageStats`/`ImageStats`, `isGrayscale`, `isBlank`, `isBlurry`, `brightnessScore`, `contrastScore`, `exposureScore`, `entropy`, `estimateNoise` (Immerkær) and `colorfulness` (Hasler & Süsstrunk): one-call image statistics and quality checks for any gray/BGR/BGRA image and dtype.
- `mse`, `mae`, `histogramEx` (per-channel `(channels, bins)` histograms), `histogramCompare` (normalized HSV/BGR/gray histograms through `cv.compareHist`), `dominantHue` and `meanColor`.
- Document and OCR preprocessing (`imgproc`): `removeShadow`, `enhanceDocument`, `sauvolaThreshold`, `niblackThreshold` (integral-image local thresholds, no contrib needed), `removeSpeckles`, `removeLines`.
- Page layout: `detectTextLines`, `detectTextBlocks`, `removeBorder`, `estimateSkew` (the angle behind `deskew`), `detectPage` and `scanDocument` (detect + `fourPointTransform` + enhance).
- `cv2u.calib3d` subpackage: `CameraCalibration`, `makeChessboard`, `objectPointsGrid`, `findChessboardEx`, `calibrateCameraEx`, `reprojectionErrors`, `undistortEx`, `saveCalibration`, `loadCalibration` for chessboard camera calibration, undistortion and a JSON calibration file.
- `stitchImages` (`cv.Stitcher` with exceptions and status names) and `stitchPair` (homography stitch on an enlarged canvas).
- `imageInfo`, `ImageInfo`, `exifOrientation`: image file header metadata (size, channels, mode, format, file size, EXIF orientation) without decoding.
- `imreadOriented`, `imreadResized`: read with the EXIF orientation applied for every flag and format, and read-and-resize with `IMREAD_REDUCED_*` fast paths for 1/2, 1/4, 1/8 scales.
- `imencodeEx`, `imdecodeEx`, `convertImageFormat`: encode to `bytes` with errors raised, decode from any bytes-like object, and convert files between formats (alpha flattened for JPEG, orientation applied, `quality`).
- Color arithmetic (`imgproc`): `CSS_COLORS`, `lightenColor`, `darkenColor`, `blendColors`, `complementaryColor`, `invertColor`, `colorDistance`, `nearestColor`, `colorName`, `textColorFor`, `randomColors`, `gradientPalette`, `colormapPalette`.
- Point sets (`core`): `rotatePoints`, `scalePoints`, `translatePoints`, `flipPoints`, `normalizePoints`, `denormalizePoints`, `clipPoints`, `pointsBoundingBox`, `distancePoints`, `nearestPoints`, `orderPoints`, `polygonCentroid`, `angleBetween`, `lineIntersection`, `pointLineDistance`.
- `Clusters` and `Letterbox` named tuples (see Changed).

### Changed
- `DBSCAN` is now `dbscanColors(src, eps, minSamples=4)` and returns `Clusters(maps, centers)` like the other clusterers; labels are `0 .. k-1` with noise `-1` (was `(maps, labels)` with noise `0` and clusters from `1`). `clusterPoints` follows the same convention (noise `-1`, `int32` labels).
- `drawDBSCAN(src, maps, labels)` is now `drawClusters(src, maps)`: returns a new image, accepts gray or BGR input; noise pixels keep their original color.
- `kmeansColors`, `meanShiftColors`, `peakColors` and `dbscanColors` return a `Clusters(maps, centers)` named tuple, and `letterbox` returns `Letterbox(image, scale, offset)`; tuple unpacking keeps working.
- Renamed: `DHEdge` -> `boldEdges`, `cropEx` -> `cropRect`, `centerCrop` -> `cropCenter`, `detectBlobs` -> `findBlobs`, `urlread` -> `imreadUrl` (now defaults to `IMREAD_COLOR` like `cv.imread`).
- Renamed parameters: `matchFeatures(src, reference, ..., mask, referenceMask)` (was `dst`, `dstMask`); `overlayImage(alpha=)` (was `opacity`); `cropContour`, `applyMask` and `flattenAlpha` take `fill=` (was `background`); `expandBoxes(margin=)` (was `pixels`).
- `imcolor(size, color, channels=3)` and `makeGradient(size, ...)` take `size=(width, height)` like `cv.resize` instead of a NumPy shape; `channels=1` gives a 2-D image.
- `putTextEx(img, ...)` and `drawContoursEx(img, contours, colors=None, labels=False, thickness=2, fontScale=0.5)` draw in place and return `img` like every other `draw*` function (they returned a BGR copy); grayscale images stay grayscale.
- `findTemplate` returns a float64 `(N, 5)` array `[x, y, w, h, score]` instead of a list of tuples, so the result feeds `nms` and `drawBoxes` directly.
- `cropRect` raises `ValueError` when the box lies entirely outside the image instead of returning an empty array.
- `cropRotatedRect` defaults to `borderMode=cv.BORDER_CONSTANT` like `rotateEx` and `translate`; `rotateEx` has explicit defaults (`INTER_LINEAR`, `BORDER_CONSTANT`, `0`).
- `drawRoundedRectangle(img, rect, color, thickness=1, radius=8, lineType=...)`: `radius` comes before `lineType`. `drawPoints` default `fontScale` is 0.5.
- Boolean switches are keyword-only: `padToSize(center=)`, `scaleBoxes(inverse=)`, `equalizeHistEx(clahe=)`, `binarize(invert=)`, `autoContrast(perChannel=)`, `findBlobs(dark=)`, `findContoursEx(sort=)`, `cropContour(transparent=)`, `drawHistogram(log=)`, `imencodeBase64(dataUri=)`, `imreadAll(strict=)`, `imwriteAll(strict=)`.
- `autoCanny` accepts 16-bit and float images (stretched to 8 bits) instead of raising; `findLines` does the same. `nms` rejects `iouThreshold > 1`. `minArea` is a `float` everywhere.
- Validation is now uniform: every function rejects empty images, and every numeric parameter rejects NaN/infinity and out-of-range values, with a `ValueError` naming the argument (about 50 functions gained checks). Every public function has complete `Args`/`Returns`/`Raises` docstrings.

### Removed
- `DBSCAN`, `drawDBSCAN`, `DHEdge`, `cropEx`, `centerCrop`, `detectBlobs` and `urlread` (renamed above). 1.0.0 is a one-time reset without deprecated aliases; from 1.0 on, renames keep a deprecated alias for at least one minor release.

## [0.4.0] - 2026-11-07

3 new APIs (132 -> 135), and the hot paths measured.

### Added

- `mapTiles` (`core`): run a function over a large image tile by tile and merge the results (`splitTiles`, the
  function, `mergeTiles`), with an optional progress callback.
- `imreadAll` and `imwriteAll` (`imgcodecs`): read or write many files; an unreadable file raises `OSError` with
  its name, or gives `None` / `False` with `strict=False`.
- `toPillow` accepts every single-channel dtype Pillow has a mode for: `uint16` (`"I;16"`), `int32` (`"I"`),
  `float32` (`"F"`), `bool` (`"1"`), and 2-channel gray + alpha (`"LA"`); all round-trip through `fromPillow`.
- Benchmarks (`benchmarks/`, `pytest benchmarks`): timings of the most used functions with pytest-benchmark, and
  regression guards that compare each wrapper with the plain OpenCV call it wraps (`test_overhead.py`). CI runs
  them on every pull request.

### Changed

- `nms` uses OpenCV's C++ non-maximum suppression (`cv.dnn.NMSBoxes`) for the overlap test: about 20x faster on
  1000 boxes. The result is unchanged (stable score order, `scoreThreshold` inclusive, empty array when nothing
  survives). The NumPy implementation remains as the fallback for builds without the `dnn` module.
- `overlayMask` blends with `cv.addWeighted` and `cv.copyTo` instead of NumPy fancy indexing: about 8x faster on a
  1080p image with a large mask.
- `mergeTiles` places the tiles directly when they do not overlap and accumulates in `float32` for 8- and 16-bit
  tiles (`float64` for wider dtypes): 2-4x faster.
- `findContoursEx` no longer measures every contour when `sort=False` and `minArea=0`.

### Fixed

- `fromPillow` returned 8-bit BGR for big-endian or native 16-bit grayscale (`"I;16B"`, `"I;16L"`, `"I;16N"`),
  silently losing precision; they now give native `uint16`. Premultiplied-alpha images (`"RGBa"`, `"La"`) keep
  their alpha channel as BGRA instead of raising or dropping it.
- `toPillow` raised an OpenCV error on empty images; it now raises `ValueError`.

## [0.3.0] - 2026-11-06

32 new APIs (100 -> 132).

### Added

- Boxes (`core`): `nms`, `xywhToXyxy`, `xyxyToXywh`, `clipBoxes`, `scaleBoxes` (undo a resize or letterbox) and
  `expandBoxes`: detector output to clean boxes on the original image.
- Binarization and morphology (`imgproc`): `binarize` (Otsu, adaptive, triangle or fixed, from any image) and
  `morphology` (`cv.morphologyEx` by operation name with the kernel built for you).
- Detection (`imgproc`): `findLines`, `findCircles` and `detectBlobs` with plain `(N, 4)` / `(N, 3)` array results.
- Features (`imgproc`): `matchFeatures` (detect, describe, match, ratio test) and `alignImages` (plus RANSAC and
  the warp) for ORB and SIFT.
- Masks and crops: `maskBoundingRect`, `applyMask`, `centerCrop`.
- Drawing: `drawBoxes`, `drawPoints`, `drawGrid` and `distinctColors`.
- Contours (`imgproc`), 11 new APIs: the steps after `cv.findContours` that every project rewrites.
  - `contourStats`: area, perimeter, rect, center, aspect ratio, extent, solidity, circularity, equivalent
    diameter and orientation as one `ContourStats` named tuple.
  - `sortContours`: reading order (left-to-right, top-to-bottom, ...) or by area / perimeter.
  - `filterContours`: keep the contours whose measurements fall within the given ranges.
  - `contoursToMask`: contours back to a filled or outlined mask (the inverse of `findContoursEx`).
  - `cropContour`: crop one object with the outside blanked, kept, or transparent (BGRA).
  - `approxContour`: `cv.approxPolyDP` with the tolerance as a fraction of the perimeter.
  - `mergeContours`: one convex hull around several pieces.
  - `drawContoursEx`: each contour in its own color, optionally numbered or labeled at its centroid.
  - `boundingRects`: every `cv.boundingRect` as an `(N, 4)` array, ready for `boxIoU`.
  - `asContour`: any points to the `(N, 1, 2)` `int32` layout OpenCV wants.

- `urlread(..., maxBytes=...)`: reject responses larger than a limit, from the `Content-Length` header when
  present and otherwise while reading, so a large or hostile URL cannot fill memory.

### Fixed

- `writeVideo` silently dropped frames whose channel count differed from the first frame (the writer skipped
  them but the returned count still included them), and accepted BGRA first frames that produced an empty video.
  Every frame must now be grayscale or BGR and match the first one; anything else raises `ValueError`.
- `rotateEx` with `scale` 0 or negative returned an unscaled image instead of failing; it now raises `ValueError`.
- `fourPointTransform` accepted collinear or duplicate points (a singular transform and a garbage image) although
  its documentation promised a `ValueError`; it now checks the area of the quadrilateral. A `size` smaller than
  1x1 is rejected too.
- `peakColors` crashed on an empty image or `sigma=nan` with an unrelated error; both raise `ValueError`.
- Consistent input validation across the extensions (see docs/principles.md): empty images, `NaN` / infinite
  parameters and colors outside the image's dtype range raise `ValueError` naming the argument, where they used
  to fail with `IndexError`, `ZeroDivisionError`, `OverflowError`, `TypeError` or an OpenCV C++ assertion
  (`adjustBrightnessContrast`, `adjustHSV`, `autoContrast`, `whiteBalance`, `cropToContent`, `fillHoles`,
  `removeSmallObjects`, `largestComponent`, `findContoursEx`, `skeletonize`, `cutout`, `letterbox`, `resizeEx`,
  `equalizeHistEx`, `unsharpMask`, `pixelate`, `ssim`, `diffMask`, `gammaCorrection`, `toGray`, `toBGR`,
  `toBGRA`, `hconcatEx`, `vconcatEx`, `montage`, `overlayImage`, `padToSize`, `padToMultiple`).

## [0.2.1] - 2026-11-06

### Added

- Tests that every extension in a submodule is exported from `cv2u`, and that every `cv.<name>` in the README
  resolves. The public API can no longer drift from the documentation unnoticed.

## [0.2.0] - 2026-11-06

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

[0.4.0]: https://github.com/076923/cv2-utils/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/076923/cv2-utils/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/076923/cv2-utils/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/076923/cv2-utils/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/076923/cv2-utils/releases/tag/v0.1.0
