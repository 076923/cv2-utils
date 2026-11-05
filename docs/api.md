---
description: "API reference for cv2u: imreadEx, imwriteEx, toPillow, imencodeBase64, readFrames, writeVideo, imshowInline, resizeEx, letterbox, drawLabel, overlayMask, cropEx, montage, equalizeHistEx and more."
---

# API reference

All functions and constants below are available directly on the package, e.g. `cv.rotateEx`.
The modules mirror OpenCV's modules (see [Design principles](principles.md)).

## core

### Geometry

::: cv2u.core.rotate.rotateEx

::: cv2u.core.extension.centerPoint

::: cv2u.core.extension.isInside

::: cv2u.core.extension.cartesianToPolar

::: cv2u.core.extension.polarToCartesian

### Creating images

::: cv2u.core.images.imcolor

::: cv2u.core.images.imcolorLike

::: cv2u.core.images.makeGradient

### Layout

::: cv2u.core.layout.padToMultiple

::: cv2u.core.layout.padToSize

::: cv2u.core.layout.splitTiles

::: cv2u.core.layout.mergeTiles

### Regions

::: cv2u.core.roi.cropEx

::: cv2u.core.roi.cropToContent

::: cv2u.core.roi.inRangeEx

::: cv2u.core.boxes.boxIoU

### Combining images

::: cv2u.core.compose.hconcatEx

::: cv2u.core.compose.vconcatEx

::: cv2u.core.compose.montage

::: cv2u.core.compose.overlayImage

### Pixel operations

::: cv2u.core.extension.gammaCorrection

::: cv2u.core.extension.DHEdge

### Clustering

`peakColors` (cv2u's own method), `kmeansColors`, `dominantColors` and `colorQuantize` need nothing else.
`meanShiftColors`, `DBSCAN` and `clusterPoints` require the `cluster` extra:
`pip install "python-opencv-utils[cluster]"`.

::: cv2u.core.cluster.dominantColors

::: cv2u.core.cluster.colorQuantize

::: cv2u.core.peaks.peakColors

::: cv2u.core.cluster.kmeansColors

::: cv2u.core.cluster.meanShiftColors

::: cv2u.core.cluster.DBSCAN

::: cv2u.core.cluster.drawDBSCAN

::: cv2u.core.cluster.clusterPoints

## imgcodecs

::: cv2u.imgcodecs.files.imreadEx

::: cv2u.imgcodecs.files.imwriteEx

::: cv2u.imgcodecs.files.listImages

::: cv2u.imgcodecs.urlread.urlread

::: cv2u.imgcodecs.convert.toPillow

::: cv2u.imgcodecs.convert.fromPillow

::: cv2u.imgcodecs.convert.imencodeBase64

::: cv2u.imgcodecs.convert.imdecodeBase64

## imgproc

### Color conversion

::: cv2u.imgproc.color.toGray

::: cv2u.imgproc.color.toBGR

::: cv2u.imgproc.color.toBGRA

::: cv2u.imgproc.color.flattenAlpha

::: cv2u.imgproc.color.hexToBGR

::: cv2u.imgproc.color.bgrToHex

### Color adjustment

::: cv2u.imgproc.adjust.adjustBrightnessContrast

::: cv2u.imgproc.adjust.adjustHSV

::: cv2u.imgproc.adjust.autoContrast

::: cv2u.imgproc.adjust.whiteBalance

::: cv2u.imgproc.adjust.matchHistograms

::: cv2u.imgproc.adjust.colorTransfer

### Quality and comparison

::: cv2u.imgproc.quality.sharpnessScore

::: cv2u.imgproc.quality.ssim

::: cv2u.imgproc.quality.diffMask

::: cv2u.imgproc.quality.imageHash

::: cv2u.imgproc.quality.hashDistance

### Masks and contours

::: cv2u.imgproc.mask.findContoursEx

::: cv2u.imgproc.mask.fillHoles

::: cv2u.imgproc.mask.removeSmallObjects

::: cv2u.imgproc.mask.largestComponent

::: cv2u.imgproc.mask.skeletonize

::: cv2u.imgproc.mask.cutout

### Matching

::: cv2u.imgproc.template.findTemplate

### Geometry and filters

::: cv2u.imgproc.resize.resizeEx

::: cv2u.imgproc.resize.letterbox

::: cv2u.imgproc.geometry.fourPointTransform

::: cv2u.imgproc.geometry.translate

::: cv2u.imgproc.geometry.transformPoints

::: cv2u.imgproc.geometry.cropRotatedRect

::: cv2u.imgproc.geometry.deskew

::: cv2u.imgproc.edges.autoCanny

::: cv2u.imgproc.enhance.equalizeHistEx

::: cv2u.imgproc.enhance.unsharpMask

::: cv2u.imgproc.enhance.pixelate

### Drawing

::: cv2u.imgproc.draw.drawLabel

::: cv2u.imgproc.draw.overlayMask

::: cv2u.imgproc.draw.drawDashedLine

::: cv2u.imgproc.draw.drawDashedRectangle

::: cv2u.imgproc.draw.drawRoundedRectangle

::: cv2u.imgproc.draw.drawHistogram

::: cv2u.imgproc.puttext.putTextEx

## videoio

::: cv2u.videoio.video.readFrames

::: cv2u.videoio.video.writeVideo

::: cv2u.videoio.video.readFrameAt

::: cv2u.videoio.video.videoInfo

::: cv2u.videoio.video.VideoInfo
    options:
      members: [width, height, fps, frameCount, duration]

## highgui

::: cv2u.highgui.display.imshowInline

::: cv2u.highgui.display.imshowEx

## flag

Named constants for magic numbers that OpenCV functions take without a constant of their own.
OpenCV's own constants (`cv.FILLED`, `cv.ROTATE_90_CLOCKWISE`, `cv.FILTER_SCHARR`, ...) are used as they are.

| Constant | Value | Use |
|---|---|---|
| `FLIP_MODE_X`, `FLIP_MODE_Y`, `FLIP_MODE_XY` | `0`, `1`, `-1` | `cv.flip(src, flipCode)` |
| `DDEPTH_SAME` | `-1` | `ddepth` of `cv.filter2D`, `cv.Sobel`, `cv.Laplacian`, ...: same depth as the source |
| `ALL_CONTOURS` | `-1` | `contourIdx` of `cv.drawContours`: draw every contour |
| `REDUCE_TO_ROW`, `REDUCE_TO_COL` | `0`, `1` | `dim` of `cv.reduce` |
| `WAIT_FOREVER` | `0` | `delay` of `cv.waitKey`: wait for a key press |
| `KEY_ESC`, `KEY_ENTER`, `KEY_SPACE`, `KEY_TAB`, `KEY_BACKSPACE` | `27`, `13`, `32`, `9`, `8` | Return value of `cv.waitKey` |

Arrow and function keys are not included: the codes returned by `cv.waitKeyEx` differ between platforms and GUI
backends.

::: cv2u.flag
    options:
      show_root_heading: false
      heading_level: 4

