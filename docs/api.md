---
description: "API reference for cv2u: VideoReader, augmentation, YOLO and COCO annotations, dnn inference, calibration, document scanning, imreadEx, imwriteEx, toPillow, imencodeBase64, readFrames, writeVideo, imshowInline, resizeEx, letterbox, drawLabel, overlayMask, cropRect, montage, equalizeHistEx and more."
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

::: cv2u.core.layout.mapTiles

### Regions

::: cv2u.core.roi.cropRect

::: cv2u.core.roi.cropCenter

::: cv2u.core.roi.cropToContent

::: cv2u.core.roi.inRangeEx

::: cv2u.core.boxes.boxIoU

::: cv2u.core.boxes.nms

::: cv2u.core.boxes.xywhToXyxy

::: cv2u.core.boxes.xyxyToXywh

::: cv2u.core.boxes.clipBoxes

::: cv2u.core.boxes.scaleBoxes

::: cv2u.core.boxes.expandBoxes

### Combining images

::: cv2u.core.compose.hconcatEx

::: cv2u.core.compose.vconcatEx

::: cv2u.core.compose.montage

::: cv2u.core.compose.overlayImage

### Pixel operations

::: cv2u.core.extension.gammaCorrection

::: cv2u.core.extension.boldEdges

### Clustering

`peakColors` (cv2u's own method), `kmeansColors`, `dominantColors` and `colorQuantize` need nothing else.
`meanShiftColors`, `dbscanColors` and `clusterPoints` require the `cluster` extra:
`pip install "python-opencv-utils[cluster]"`.

::: cv2u.core.cluster.Clusters

::: cv2u.core.cluster.dominantColors

::: cv2u.core.cluster.colorQuantize

::: cv2u.core.peaks.peakColors

::: cv2u.core.cluster.kmeansColors

::: cv2u.core.cluster.meanShiftColors

::: cv2u.core.cluster.dbscanColors

::: cv2u.core.cluster.drawClusters

::: cv2u.core.cluster.clusterPoints

### Point sets

::: cv2u.core.points.rotatePoints

::: cv2u.core.points.scalePoints

::: cv2u.core.points.translatePoints

::: cv2u.core.points.flipPoints

::: cv2u.core.points.normalizePoints

::: cv2u.core.points.denormalizePoints

::: cv2u.core.points.clipPoints

::: cv2u.core.points.pointsBoundingBox

::: cv2u.core.points.distancePoints

::: cv2u.core.points.nearestPoints

::: cv2u.core.points.orderPoints

::: cv2u.core.points.polygonCentroid

::: cv2u.core.points.angleBetween

::: cv2u.core.points.lineIntersection

::: cv2u.core.points.pointLineDistance

### Annotation formats

::: cv2u.core.annotations.maskToPolygons

::: cv2u.core.annotations.polygonsToMask

::: cv2u.core.annotations.boxesToPolygons

::: cv2u.core.annotations.polygonsToBoxes

::: cv2u.core.annotations.polygonArea

::: cv2u.core.annotations.polygonIoU

::: cv2u.core.annotations.maskIoU

::: cv2u.core.annotations.rleEncode

::: cv2u.core.annotations.rleDecode

::: cv2u.core.annotations.boxesToYolo

::: cv2u.core.annotations.yoloToBoxes

::: cv2u.core.annotations.readYoloLabels

::: cv2u.core.annotations.writeYoloLabels

::: cv2u.core.annotations.labelMapToColor

::: cv2u.core.annotations.colorToLabelMap

::: cv2u.core.annotations.instanceMasksToLabelMap

::: cv2u.core.annotations.labelMapToMasks

::: cv2u.core.annotations.drawPolygons

::: cv2u.core.annotations.drawSegmentation

::: cv2u.core.annotations.flipBoxes

::: cv2u.core.annotations.flipPolygons

## imgcodecs

::: cv2u.imgcodecs.files.imreadEx

::: cv2u.imgcodecs.files.imwriteEx

::: cv2u.imgcodecs.files.listImages

::: cv2u.imgcodecs.files.imreadAll

::: cv2u.imgcodecs.files.imwriteAll

::: cv2u.imgcodecs.url.imreadUrl

::: cv2u.imgcodecs.convert.toPillow

::: cv2u.imgcodecs.convert.fromPillow

::: cv2u.imgcodecs.convert.imencodeBase64

::: cv2u.imgcodecs.convert.imdecodeBase64

### Image file metadata

::: cv2u.imgcodecs.meta.ImageInfo

::: cv2u.imgcodecs.meta.imageInfo

::: cv2u.imgcodecs.meta.exifOrientation

::: cv2u.imgcodecs.meta.imreadOriented

::: cv2u.imgcodecs.meta.imreadResized

::: cv2u.imgcodecs.meta.imencodeEx

::: cv2u.imgcodecs.meta.imdecodeEx

::: cv2u.imgcodecs.meta.convertImageFormat

## imgproc

### Color conversion

::: cv2u.imgproc.color.toGray

::: cv2u.imgproc.color.toBGR

::: cv2u.imgproc.color.toBGRA

::: cv2u.imgproc.color.flattenAlpha

::: cv2u.imgproc.color.hexToBGR

::: cv2u.imgproc.color.bgrToHex

::: cv2u.imgproc.color.distinctColors

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

::: cv2u.imgproc.mask.maskBoundingRect

::: cv2u.imgproc.mask.applyMask

::: cv2u.imgproc.binary.binarize

::: cv2u.imgproc.binary.morphology

::: cv2u.imgproc.contour.contourStats

::: cv2u.imgproc.contour.ContourStats

::: cv2u.imgproc.contour.sortContours

::: cv2u.imgproc.contour.filterContours

::: cv2u.imgproc.contour.contoursToMask

::: cv2u.imgproc.contour.cropContour

::: cv2u.imgproc.contour.approxContour

::: cv2u.imgproc.contour.mergeContours

::: cv2u.imgproc.contour.drawContoursEx

::: cv2u.imgproc.contour.boundingRects

::: cv2u.imgproc.contour.asContour

### Matching and detection

::: cv2u.imgproc.template.findTemplate

::: cv2u.imgproc.features.matchFeatures

::: cv2u.imgproc.features.FeatureMatches

::: cv2u.imgproc.features.alignImages

::: cv2u.imgproc.detect.findLines

::: cv2u.imgproc.detect.findCircles

::: cv2u.imgproc.detect.findBlobs

### Geometry and filters

::: cv2u.imgproc.resize.resizeEx

::: cv2u.imgproc.resize.letterbox

::: cv2u.imgproc.resize.Letterbox

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

::: cv2u.imgproc.draw.drawBoxes

::: cv2u.imgproc.draw.drawPoints

::: cv2u.imgproc.draw.drawGrid

::: cv2u.imgproc.draw.drawDashedLine

::: cv2u.imgproc.draw.drawDashedRectangle

::: cv2u.imgproc.draw.drawRoundedRectangle

::: cv2u.imgproc.draw.drawHistogram

::: cv2u.imgproc.puttext.putTextEx

### Image statistics and quality checks

::: cv2u.imgproc.stats.ImageStats

::: cv2u.imgproc.stats.imageStats

::: cv2u.imgproc.stats.isGrayscale

::: cv2u.imgproc.stats.isBlank

::: cv2u.imgproc.stats.isBlurry

::: cv2u.imgproc.stats.brightnessScore

::: cv2u.imgproc.stats.contrastScore

::: cv2u.imgproc.stats.colorfulness

::: cv2u.imgproc.stats.exposureScore

::: cv2u.imgproc.stats.entropy

::: cv2u.imgproc.stats.estimateNoise

::: cv2u.imgproc.stats.mse

::: cv2u.imgproc.stats.mae

::: cv2u.imgproc.stats.histogramEx

::: cv2u.imgproc.stats.histogramCompare

::: cv2u.imgproc.stats.dominantHue

::: cv2u.imgproc.stats.meanColor

### Color arithmetic

::: cv2u.imgproc.palette.CSS_COLORS

::: cv2u.imgproc.palette.lightenColor

::: cv2u.imgproc.palette.darkenColor

::: cv2u.imgproc.palette.blendColors

::: cv2u.imgproc.palette.complementaryColor

::: cv2u.imgproc.palette.invertColor

::: cv2u.imgproc.palette.colorDistance

::: cv2u.imgproc.palette.nearestColor

::: cv2u.imgproc.palette.colorName

::: cv2u.imgproc.palette.textColorFor

::: cv2u.imgproc.palette.randomColors

::: cv2u.imgproc.palette.gradientPalette

::: cv2u.imgproc.palette.colormapPalette

### Data augmentation

::: cv2u.imgproc.augment.addNoise

::: cv2u.imgproc.augment.motionBlur

::: cv2u.imgproc.augment.vignette

::: cv2u.imgproc.augment.jpegArtifacts

::: cv2u.imgproc.augment.randomErase

::: cv2u.imgproc.augment.solarize

::: cv2u.imgproc.augment.posterize

::: cv2u.imgproc.augment.randomCrop

::: cv2u.imgproc.augment.randomFlip

::: cv2u.imgproc.augment.randomRotate

::: cv2u.imgproc.augment.randomScale

::: cv2u.imgproc.augment.randomPerspective

::: cv2u.imgproc.augment.randomBrightnessContrast

::: cv2u.imgproc.augment.randomHSV

::: cv2u.imgproc.augment.randomBlur

::: cv2u.imgproc.augment.mosaic

::: cv2u.imgproc.augment.mixup

::: cv2u.imgproc.augment.cutMix

### Documents and OCR

::: cv2u.imgproc.document.removeShadow

::: cv2u.imgproc.document.enhanceDocument

::: cv2u.imgproc.document.sauvolaThreshold

::: cv2u.imgproc.document.niblackThreshold

::: cv2u.imgproc.document.removeSpeckles

::: cv2u.imgproc.document.removeLines

::: cv2u.imgproc.document.detectTextLines

::: cv2u.imgproc.document.detectTextBlocks

::: cv2u.imgproc.document.removeBorder

::: cv2u.imgproc.document.estimateSkew

::: cv2u.imgproc.document.detectPage

::: cv2u.imgproc.document.scanDocument

### Stitching

::: cv2u.imgproc.stitch.stitchImages

::: cv2u.imgproc.stitch.stitchPair

### Visualization

::: cv2u.imgproc.visualize.applyColorMapEx

::: cv2u.imgproc.visualize.heatmap

::: cv2u.imgproc.visualize.colorBar

::: cv2u.imgproc.visualize.drawArrow

::: cv2u.imgproc.visualize.drawRotatedRect

::: cv2u.imgproc.visualize.drawRotatedBoxes

::: cv2u.imgproc.visualize.drawPolygon

::: cv2u.imgproc.visualize.drawSkeleton

::: cv2u.imgproc.visualize.COCO_SKELETON

::: cv2u.imgproc.visualize.drawTrajectory

::: cv2u.imgproc.visualize.drawCrosshair

::: cv2u.imgproc.visualize.drawTextBlock

::: cv2u.imgproc.visualize.drawFPS

::: cv2u.imgproc.visualize.drawProgressBar

::: cv2u.imgproc.visualize.drawLegend

::: cv2u.imgproc.visualize.compareImages

::: cv2u.imgproc.visualize.labeledMontage

::: cv2u.imgproc.visualize.blendImages

## videoio

::: cv2u.videoio.video.readFrames

::: cv2u.videoio.video.writeVideo

::: cv2u.videoio.video.readFrameAt

::: cv2u.videoio.video.videoInfo

::: cv2u.videoio.video.VideoInfo
    options:
      members: [width, height, fps, frameCount, duration]

### Readers and writers

::: cv2u.videoio.reader.VideoReader

::: cv2u.videoio.reader.VideoWriterEx

::: cv2u.videoio.reader.CameraStream

::: cv2u.videoio.reader.FPSCounter

::: cv2u.videoio.reader.frameToTime

::: cv2u.videoio.reader.timeToFrame

::: cv2u.videoio.reader.formatTimestamp

### Video files

::: cv2u.videoio.edit.extractFrames

::: cv2u.videoio.edit.trimVideo

::: cv2u.videoio.edit.concatVideos

::: cv2u.videoio.edit.resizeVideo

::: cv2u.videoio.edit.videoToGif

::: cv2u.videoio.edit.imagesToVideo

::: cv2u.videoio.edit.detectScenes

::: cv2u.videoio.edit.sampleFrames

::: cv2u.videoio.edit.videoThumbnail

## highgui

::: cv2u.highgui.display.imshowInline

::: cv2u.highgui.display.imshowEx

## calib3d

Camera calibration helpers; the OpenCV calib3d functions (`cv.calibrateCamera`, `cv.undistort`, ...) stay available as they are.

### Camera calibration

::: cv2u.calib3d.calibrate.CameraCalibration

::: cv2u.calib3d.calibrate.makeChessboard

::: cv2u.calib3d.calibrate.objectPointsGrid

::: cv2u.calib3d.calibrate.findChessboardEx

::: cv2u.calib3d.calibrate.calibrateCameraEx

::: cv2u.calib3d.calibrate.reprojectionErrors

::: cv2u.calib3d.calibrate.undistortEx

::: cv2u.calib3d.calibrate.saveCalibration

::: cv2u.calib3d.calibrate.loadCalibration

## dnn

`cv.dnn` is the `cv2u.dnn` package: it adds the helpers below and passes every `cv2.dnn` name (`cv.dnn.readNet`, `cv.dnn.NMSBoxes`, ...) through.

### Inference

::: cv2u.dnn.inference.PreprocessInfo

::: cv2u.dnn.inference.Detections

::: cv2u.dnn.inference.loadModel

::: cv2u.dnn.inference.readClassNames

::: cv2u.dnn.inference.preprocessImage

::: cv2u.dnn.inference.inferImage

::: cv2u.dnn.inference.postprocessDetections

::: cv2u.dnn.inference.postprocessClassification

::: cv2u.dnn.inference.softmax

::: cv2u.dnn.inference.sigmoid

::: cv2u.dnn.inference.drawDetections

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

