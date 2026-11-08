---
description: "The design principles of cv2u: fill the gaps in OpenCV, look like OpenCV, never shadow OpenCV, stay light, typed and validated, and change carefully."
---

# Design principles

cv2u is **what you wish OpenCV had, in the place you would look for it**: `import cv2u as cv` gives you all of
OpenCV plus a small set of extensions. These principles decide what goes into the library and how it looks.
Several of them are enforced by the test suite.

## 1. Fill the gaps in OpenCV

Add a function only when OpenCV lacks it, or when doing it with OpenCV takes boilerplate that everyone rewrites.
Never re-implement something OpenCV already does well.

| cv2u | The gap in OpenCV |
|---|---|
| `imreadEx`, `imwriteEx` | `cv.imread` and `cv.imwrite` fail on non-ASCII paths (e.g. Korean) on Windows. |
| `rotateEx` | `cv.rotate` only rotates by multiples of 90°, and `cv.warpAffine` crops the corners. |
| `resizeEx`, `letterbox` | `cv.resize` needs the full target size; keeping the aspect ratio or padding is manual. |
| `hconcatEx`, `vconcatEx` | `cv.hconcat`/`cv.vconcat` require the same height/width, channels and type. |
| `overlayImage` | No function pastes a transparent (BGRA) image onto another, with clipping at the borders. |
| `fourPointTransform` | Warping a quadrilateral to a rectangle means ordering the corners and computing the size by hand. |
| `autoCanny` | `cv.Canny` needs two hand-tuned thresholds for every image. |
| `readFrames`, `writeVideo`, `videoInfo` | `cv.VideoCapture`/`cv.VideoWriter` need a read loop, `release()`, fourcc codes and many `get()` calls. |
| `imshowInline`, `imshowEx` | `cv.imshow` does not work in Jupyter, and shows large images cropped by the screen. |
| `toPillow`, `fromPillow` | Converting to/from Pillow needs channel reordering, and `np.asarray` gets palette, CMYK, premultiplied-alpha and big-endian 16-bit images wrong. |
| `imencodeBase64`, `imdecodeBase64` | Web APIs exchange images as base64 or data URIs; OpenCV only encodes to bytes. |
| `drawLabel`, `overlayMask` | A detector-style labeled box or a translucent mask takes `getTextSize` math and blending code. |
| `cropRect` | Slicing with boxes that extend outside the image silently returns wrong or empty regions. |
| `montage` | No function arranges many images in a grid. |
| `equalizeHistEx` | `cv.equalizeHist` rejects color images. |
| `inRangeEx` | `cv.inRange` silently selects nothing for hue ranges that wrap around, such as red. |
| `unsharpMask` | The standard sharpening filter has no OpenCV function. |
| `toGray`, `toBGR`, `toBGRA` | `cv.cvtColor` raises an error when the image already has the target format. |
| `findContoursEx` | `cv.findContours` rejects color images and returns the contours unsorted. |
| `fillHoles`, `removeSmallObjects` | Basic mask clean-up has no OpenCV function. |
| `contourStats`, `filterContours`, `sortContours` | OpenCV measures one property per call (`contourArea`, `arcLength`, `boundingRect`, `convexHull`...); "sort left to right" and "keep the round ones" are rewritten in every project. |
| `contoursToMask`, `cropContour`, `drawContoursEx` | `cv.drawContours` needs an image, `-1`, a color and `cv.FILLED` to make a mask; cropping an object with its outside blanked, or drawing contours in distinct colors with labels, is bookkeeping. |
| `approxContour`, `mergeContours`, `boundingRects`, `asContour` | `cv.approxPolyDP` takes an absolute tolerance; merging pieces, boxing every contour and the `(N, 1, 2)` `int32` layout are boilerplate. |
| `findTemplate` | `cv.matchTemplate` returns a score map, not the matches. |
| `cropToContent` | Trimming a uniform border takes thresholding and bounding-box code. |
| `imwriteEx(..., makeDirs=True)` | `cv.imwrite` silently returns `False` when the folder does not exist. |
| `ssim`, `skeletonize`, `whiteBalance` | They exist only in the `quality`, `ximgproc` and `xphoto` contrib modules, which `opencv-python` does not include. |
| `hexToBGR`, `bgrToHex` | Colors from the web are `#rrggbb`; OpenCV wants BGR tuples. |
| `adjustBrightnessContrast`, `adjustHSV`, `autoContrast` | `cv.convertScaleAbs` scales around 0 and takes absolute values; HSV edits need the hue/2 convention and wrap-around; `cv.normalize` is defeated by a single outlier. |
| `sharpnessScore`, `diffMask` | Blur detection and change detection are a few lines everyone rewrites. |
| `translate`, `padToMultiple`, `padToSize`, `splitTiles`, `mergeTiles` | Shifting, padding and tiling all need `warpAffine`/`copyMakeBorder` arithmetic. |
| `drawDashedLine`, `drawDashedRectangle`, `drawHistogram` | OpenCV draws only solid lines, and `cv.calcHist` only returns numbers. |
| `largestComponent`, `cutout` | Keeping the main object and cutting it out with transparency take component and alpha bookkeeping. |
| `transformPoints` | `cv.transform` and `cv.perspectiveTransform` need `(N, 1, 2)` float arrays and a different function per matrix type. |
| `cropRotatedRect`, `deskew` | OpenCV describes rotated boxes (`cv.minAreaRect`) but cannot extract one, and has no skew estimation for pages. |
| `flattenAlpha` | `cv.COLOR_BGRA2BGR` drops the alpha channel, so transparent areas turn into hidden colors (often black). |
| `matchHistograms`, `colorTransfer` | `cv.equalizeHist` only targets a flat histogram; matching another image's tones or colors is manual. |
| `imageHash`, `hashDistance` | The `img_hash` module is in contrib, which `opencv-python` does not include. |
| `boxIoU` | Box overlap is computed only inside `cv.dnn.NMSBoxes`; matching and tracking code rewrites it. |
| `nms`, `xywhToXyxy`, `xyxyToXywh`, `clipBoxes`, `scaleBoxes`, `expandBoxes` | `cv.dnn.NMSBoxes` wants lists and returns a tuple when empty; mapping boxes back through a letterbox, clipping and padding them are the same arithmetic in every detector script. |
| `binarize`, `morphology` | `cv.threshold` wants 8-bit grayscale and returns a tuple, adaptive thresholding is a separate function; every `morphologyEx` call starts with `getStructuringElement`. |
| `findLines`, `findCircles`, `findBlobs` | `cv.HoughLinesP` / `cv.HoughCircles` return `None` or `(N, 1, 4)` / `(1, N, 3)` and need an edge or blurred image first; `SimpleBlobDetector` takes a `Params` object with a dozen fields. |
| `matchFeatures`, `alignImages` | Detector, matcher norm, `knnMatch`, ratio test, keypoint unpacking and `findHomography` with RANSAC are the same twelve lines before every registration. |
| `maskBoundingRect`, `applyMask`, `cropCenter` | `cv.boundingRect(cv.findNonZero(mask))` fails on empty or color masks; `cv.bitwise_and` with a mask only gives black; a center crop is index arithmetic. |
| `drawBoxes`, `drawPoints`, `drawGrid`, `distinctColors` | Drawing many detections or keypoints in distinct colors, with labels, is a loop everyone writes; OpenCV has no palette. |
| `pixelate`, `drawRoundedRectangle` | Mosaic anonymization and rounded boxes take resize and arc arithmetic. |
| `listImages`, `imreadAll`, `imwriteAll` | Collecting the image files of a folder (in natural order) is glob and sort boilerplate; `cv.imread` returns `None` silently, so a batch loop needs its own error handling. |
| `mapTiles` | Running a fixed-input model over a large image is split, loop, merge and seam averaging every time. |
| `readFrameAt` | Reading one frame means open, seek, read and release. |
| `putTextEx` | `cv.putText` only has the Hershey fonts, which cover ASCII only. |
| `imreadUrl` | `cv.imread` cannot read from a URL. |
| `imcolor`, `imcolorLike`, `makeGradient` | There is no function to create a filled or gradient image. |
| `gammaCorrection`, `centerPoint`, `boldEdges` | Lookup-table, image-moment and morphological-gradient boilerplate. |
| `dominantColors`, `colorQuantize`, `kmeansColors` | `cv.kmeans` needs reshaping, float conversion and criteria, and its results change from run to run (global RNG). |
| `peakColors` | Automatic color clustering usually needs scikit-learn, samples the pixels and is slow; cv2u's own histogram-peak method needs neither. |
| `meanShiftColors`, `dbscanColors`, `drawClusters`, `clusterPoints` | OpenCV has no clustering that finds the number of groups by itself, for colors or for points. |
| `VideoReader`, `VideoWriterEx`, `CameraStream` | `cv.VideoCapture`/`cv.VideoWriter` need `isOpened()` checks, `CAP_PROP_*` getters, fourcc/size/`isColor` up front, manual `release()`, and buffer live streams so a slow consumer reads stale frames. |
| `extractFrames`, `trimVideo`, `concatVideos`, `resizeVideo`, `videoToGif`, `imagesToVideo` | Every "ffmpeg-lite" task (dump frames, cut, join, resize, GIF, time-lapse) is the same read/write loop; OpenCV cannot write GIFs at all. |
| `detectScenes`, `sampleFrames`, `videoThumbnail`, `FPSCounter`, `formatTimestamp` | Scene cuts, preview sheets, FPS overlays and `HH:MM:SS.mmm` stamps are rewritten in every video script. |
| `addNoise`, `motionBlur`, `vignette`, `jpegArtifacts`, `solarize`, `posterize` | Classic photo degradations for augmentation have no OpenCV function (`cv.randn` needs a preallocated array; directional blur, vignetting, solarize and posterize are rewritten in every project). |
| `randomCrop`, `randomFlip`, `randomRotate`, `randomScale`, `randomPerspective` | `cv.flip`/`cv.warpAffine` never touch the detection boxes; keeping labels aligned with a random geometric transform is bookkeeping everyone rewrites. |
| `mosaic`, `mixup`, `cutMix`, `randomErase` | The standard detector/classifier mixing augmentations otherwise require albumentations or a framework-specific loader. |
| `maskToPolygons`, `polygonsToMask`, `rleEncode`, `rleDecode` | Datasets store masks as COCO polygons or RLE; `cv.findContours`/`cv.fillPoly` need `int32 (N, 1, 2)` arrays, keep one-pixel specks, and OpenCV has no run-length encoding at all. |
| `boxesToYolo`, `yoloToBoxes`, `readYoloLabels`, `writeYoloLabels`, `flipBoxes`, `flipPolygons` | Normalized `class cx cy w h` label files and keeping annotations aligned with `cv.flip` are the same arithmetic in every training script. |
| `polygonIoU`, `polygonArea`, `maskIoU`, `labelMapToColor`, `colorToLabelMap`, `drawSegmentation` | `cv.intersectConvexConvex` only handles convex shapes, `cv.contourArea` rounds to integers, and coloring / recovering a class-id map has no OpenCV function. |
| `preprocessImage`, `postprocessDetections` | `cv.dnn.blobFromImage` stretches the image and forgets the geometry, and nothing in OpenCV decodes YOLO-style outputs; cv2u letterboxes, keeps the mapping and returns boxes on the original image after per-class NMS. |
| `loadModel`, `readClassNames` | `cv.dnn.readNet` cannot open non-ASCII paths on Windows and reports a missing file as a C++ assertion; cv2u reads ONNX/TFLite/TF models through Python and raises `FileNotFoundError`. |
| `inferImage`, `drawDetections` | `Net.forward()` returns only the first output and OpenCV has no labeled-detection drawing; cv2u returns every output layer and draws boxes with class names and scores. |
| `heatmap`, `applyColorMapEx`, `colorBar` | `cv.applyColorMap` only takes an 8-bit map of the image size; overlaying a float attention/saliency map means normalize, convert, resize and `addWeighted` every time, and there is no legend strip. |
| `drawArrow`, `drawRotatedRect`, `drawPolygon`, `drawSkeleton`, `drawTrajectory`, `drawCrosshair` | `cv.arrowedLine` sizes the head relative to the arrow length, OpenCV cannot draw its own `RotatedRect`, `polylines`/`fillPoly` need `int32 (N, 1, 2)` and draw opaque only, and poses, tracks and crosshairs have no drawing function. |
| `drawTextBlock`, `drawFPS`, `drawProgressBar`, `drawLegend`, `compareImages`, `labeledMontage`, `blendImages` | `cv.putText` writes one line with no background; HUD panels, before/after views (split, blend, diff) and titled grids are the same boilerplate in every demo; `cv.addWeighted` blends exactly two images. |
| `imageStats`, `isBlank`, `isBlurry`, `estimateNoise` | Dataset triage (flat frames, out-of-focus shots, noise level, entropy) needs a handful of numpy formulas per image; `cv.meanStdDev`/`cv.minMaxLoc` only cover parts and `cv.minMaxLoc` rejects color images. |
| `histogramEx`, `histogramCompare` | `cv.calcHist` takes lists of images, channels and ranges per dtype and returns a `(bins, 1)` column per call; comparing two images needs the same setup twice plus normalization. |
| `mse`, `mae`, `colorfulness`, `dominantHue` | Only `cv.PSNR` exists for error metrics (log scale, overflow-prone on `uint8` by hand), and there is no colorfulness or dominant-hue measure. |
| `sauvolaThreshold`, `niblackThreshold` | Local thresholds for OCR only exist in the `ximgproc` contrib module (`niBlackThreshold`), which `opencv-python` does not ship; here they are built from `cv.integral2`. |
| `removeShadow`, `removeLines`, `removeSpeckles`, `enhanceDocument` | Cleaning a photographed page (shadows, table rules, dust) is a well-known chain of dilate/medianBlur/morphologyEx/inpaint calls that every OCR project rewrites. |
| `detectPage`, `scanDocument`, `detectTextLines`, `detectTextBlocks`, `estimateSkew`, `removeBorder` | There is no function from "photo of a document" to an upright page, its text lines or its skew angle; only the primitives (`Canny`, `approxPolyDP`, `connectedComponentsWithStats`). |
| `findChessboardEx`, `calibrateCameraEx`, `undistortEx` | The calibration tutorial is a page of boilerplate: object-point grids, `(found, corners)` tuples with `(N,1,2)` shapes, a separate `cornerSubPix` on a gray copy, and `undistort` that leaves black borders unless you also call `getOptimalNewCameraMatrix` and crop the ROI. |
| `saveCalibration`, `loadCalibration` | `cv.FileStorage` writes YAML/XML and cannot open non-ASCII paths on Windows; a JSON file works everywhere. |
| `stitchImages`, `stitchPair` | `cv.Stitcher.stitch` returns a bare integer status and `None` instead of raising, and the bare `warpPerspective` stitch cuts off whatever lands outside the first image. |
| `imageInfo`, `imreadOriented`, `imreadResized` | `cv.imread` must decode the whole file to tell its size, applies the EXIF orientation only for `IMREAD_COLOR`/`IMREAD_GRAYSCALE` (not for `IMREAD_UNCHANGED` or `IMREAD_REDUCED_*`), and reduced-size decoding is a flag few know. |
| `imencodeEx`, `imdecodeEx`, `convertImageFormat` | `cv.imencode` returns `(ok, array)` instead of `bytes`, `cv.imdecode` needs `np.frombuffer` first, and converting a file to JPEG silently drops the alpha channel. |
| `lightenColor`, `colorName`, `textColorFor`, `rotatePoints`, `distancePoints`, `lineIntersection` | OpenCV converts whole images between color spaces but has nothing for one color, and transforms points only through matrices: no distance matrix, corner ordering, flip or line intersection. |
| `FLIP_MODE_*`, `DDEPTH_SAME`, `ALL_CONTOURS`, `REDUCE_TO_*`, `WAIT_FOREVER`, `KEY_*` | Parameters such as `flipCode`, `ddepth`, `contourIdx`, `dim` and `cv.waitKey` take magic numbers with no OpenCV constant (OpenCV's own constants like `cv.FILLED` are used as they are). |

## 2. Look like OpenCV

An extension should feel like it was always part of OpenCV.

- **Modules** mirror OpenCV's: `core`, `imgcodecs` (reading and writing images), `imgproc` (image processing),
  `videoio` (video files and cameras), `highgui` (showing images), `calib3d` (camera calibration) and `dnn`
  (inference; `cv.dnn` is the `cv2u.dnn` package, which passes every OpenCV name through).
  An extension goes where the OpenCV function it extends lives (`resizeEx` in `imgproc` like `cv.resize`).
- **Names** use camelCase like OpenCV. An extended version of an existing OpenCV function gets the `Ex` suffix
  (`rotateEx` for `cv.rotate`, `putTextEx` for `cv.putText`). Constants use `UPPER_CASE`.
- **Parameters** reuse OpenCV's names and conventions: the source image comes first and is called `src`,
  images are 8-bit BGR by default, and OpenCV parameter names are kept (`flags`, `borderMode`, `borderValue`,
  `angleInDegrees`, `interpolation`); sizes are `(width, height)` like `cv.resize`.
- **Results** behave like OpenCV's: image-processing functions return new images instead of modifying their
  input, while drawing functions (`drawLabel`) draw in place and return the image, like `cv.rectangle`.
  Readers return `None` when the data cannot be decoded, like `cv.imread`.

Where an extension intentionally differs from a similar OpenCV function, the docstring says so
(for example, `cartesianToPolar` returns `(theta, rho)` while `cv.cartToPolar` returns `(magnitude, angle)`).

## 3. Never shadow OpenCV

cv2u adds names, it never replaces them. No extension may have the same name as anything in `cv2`, so
`cv.<anything from OpenCV>` always means exactly what it means in OpenCV. *(Enforced by a test.)*

## 4. Stay light

The core depends only on OpenCV (any distribution, chosen with an extra: `[opencv]`, `[headless]`, `[contrib]` or
`[contrib-headless]`), `numpy` and `pillow`. Heavier dependencies are optional extras that are
imported lazily, with an `ImportError` that says how to install them (`pip install "python-opencv-utils[cluster]"`).
Environment-specific packages are never required: `imshowInline` uses IPython only when it is called in a notebook.
Where OpenCV can do the job, it is used instead of an extra: `kmeansColors` is built on `cv.kmeans`, so palettes and
color quantization work without scikit-learn.

## 5. Typed and validated

- Every public function has type hints and a docstring. Type checkers and IDEs also see OpenCV's own signatures
  through cv2u. *(Enforced by mypy, pyright and a test.)*
- Invalid input raises `ValueError` with a message that names the problem, instead of an internal NumPy or OpenCV
  error. Concretely: an empty image, a `NaN` or infinite number, a negative size or a color outside the image's dtype
  range is rejected up front (helpers in `cv2u/_validate.py`); OpenCV flag values are passed through, as OpenCV
  reports those itself. *(Enforced by `tests/test_validation.py`.)*

## 6. Change carefully

Once a function is released, code that uses it should keep working. 1.0.0 reset the API once (renames and
signature changes without aliases; see the changelog); from 1.0 on the deprecation rule applies:

- A renamed function keeps its old name as a deprecated alias that emits a `DeprecationWarning`
  for at least one minor release.
- Never change what an existing call returns silently. New behavior goes behind a new parameter whose default keeps
  the current behavior.
- Output changes are caught by golden-image regression tests.
