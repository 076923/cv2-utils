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
| `toPillow`, `fromPillow` | Converting to/from Pillow needs channel reordering, and `np.asarray` gets palette and CMYK images wrong. |
| `imencodeBase64`, `imdecodeBase64` | Web APIs exchange images as base64 or data URIs; OpenCV only encodes to bytes. |
| `drawLabel`, `overlayMask` | A detector-style labeled box or a translucent mask takes `getTextSize` math and blending code. |
| `cropEx` | Slicing with boxes that extend outside the image silently returns wrong or empty regions. |
| `montage` | No function arranges many images in a grid. |
| `equalizeHistEx` | `cv.equalizeHist` rejects color images. |
| `inRangeEx` | `cv.inRange` silently selects nothing for hue ranges that wrap around, such as red. |
| `unsharpMask` | The standard sharpening filter has no OpenCV function. |
| `toGray`, `toBGR`, `toBGRA` | `cv.cvtColor` raises an error when the image already has the target format. |
| `findContoursEx` | `cv.findContours` rejects color images and returns the contours unsorted. |
| `fillHoles`, `removeSmallObjects` | Basic mask clean-up has no OpenCV function. |
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
| `pixelate`, `drawRoundedRectangle` | Mosaic anonymization and rounded boxes take resize and arc arithmetic. |
| `listImages` | Collecting the image files of a folder (in natural order) is glob and sort boilerplate. |
| `readFrameAt` | Reading one frame means open, seek, read and release. |
| `putTextEx` | `cv.putText` only has the Hershey fonts, which cover ASCII only. |
| `urlread` | `cv.imread` cannot read from a URL. |
| `imcolor`, `imcolorLike`, `makeGradient` | There is no function to create a filled or gradient image. |
| `gammaCorrection`, `centerPoint` | Lookup-table and image-moment boilerplate. |
| `dominantColors`, `colorQuantize`, `kmeansColors` | `cv.kmeans` needs reshaping, float conversion and criteria, and its results change from run to run (global RNG). |
| `peakColors` | Automatic color clustering usually needs scikit-learn, samples the pixels and is slow; cv2u's own histogram-peak method needs neither. |
| `meanShiftColors`, `DBSCAN`, `drawDBSCAN`, `clusterPoints` | OpenCV has no clustering that finds the number of groups by itself, for colors or for points. |
| `FLIP_MODE_*`, `DDEPTH_SAME`, `ALL_CONTOURS`, `REDUCE_TO_*`, `WAIT_FOREVER`, `KEY_*` | Parameters such as `flipCode`, `ddepth`, `contourIdx`, `dim` and `cv.waitKey` take magic numbers with no OpenCV constant (OpenCV's own constants like `cv.FILLED` are used as they are). |

## 2. Look like OpenCV

An extension should feel like it was always part of OpenCV.

- **Modules** mirror OpenCV's: `core`, `imgcodecs` (reading and writing images), `imgproc` (image processing),
  `videoio` (video files and cameras) and `highgui` (showing images).
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

Once a function is released, code that uses it should keep working:

- A renamed function keeps its old name as a deprecated alias that emits a `DeprecationWarning`
  for at least one minor release.
- Never change what an existing call returns silently. New behavior goes behind a new parameter whose default keeps
  the current behavior.
- Output changes are caught by golden-image regression tests.
