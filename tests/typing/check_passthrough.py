"""Static typing checks, run by mypy (not by pytest).

``warn_unused_ignores`` turns every ``# type: ignore[...]`` below into an assertion:
if ``cv.imread`` degraded to ``Any``, the call would no longer be an error, the ignore
would be unused, and mypy would fail.
"""

import cv2
import numpy as np

import cv2u as cv

# OpenCV functions keep their real signatures through cv2u.
img = cv.imread("image.png")
cv.imread(123)  # type: ignore[call-overload]
cv.cvtColor("not an image", cv.COLOR_BGR2GRAY)  # type: ignore[call-overload]
gray: cv2.typing.MatLike = cv.cvtColor(np.zeros((2, 2, 3), np.uint8), cv.COLOR_BGR2GRAY)
code: int = cv.COLOR_BGR2GRAY
net = cv.dnn.readNet("model.onnx")

# cv2u extensions are typed too.
rotated: np.ndarray = cv.rotateEx(gray, 45)
cv.rotateEx(gray, "45")  # type: ignore[arg-type]
flip_code: int = cv.FLIP_MODE_X
gradient: np.ndarray = cv.makeGradient((2, 2), 0, 255, direction="horizontal")
gray_gradient: np.ndarray = cv.makeGradient((2, 2), 0, 255, channels=1)
canvas: np.ndarray = cv.imcolor((4, 4), (1, 2, 3))
cv.makeGradient((2, 2), 0, 255, direction="diagonal")  # type: ignore[arg-type]
filled: np.ndarray = cv.imcolorLike(gradient, 0)
maybe_img: np.ndarray | None = cv.imreadEx("사진.png")
written: bool = cv.imwriteEx("out.png", filled)
boxed, scale, (left, top) = cv.letterbox(filled, (640, 640))
fitted: cv.Letterbox = cv.letterbox(filled, (640, 640))
fitted_image: np.ndarray = fitted.image
cv.resizeEx(filled, width="640")  # type: ignore[arg-type]
side_by_side: np.ndarray = cv.hconcatEx([filled, boxed], gap=4)
stamped: np.ndarray = cv.overlayImage(filled, boxed, org=(10, 10), alpha=0.5)
info: cv.VideoInfo = cv.videoInfo("clip.mp4")
seconds: float = info.duration
for frame in cv.readFrames("clip.mp4", step=2):
    frame_shape = frame.shape
written_frames: int = cv.writeVideo("out.avi", [filled], fps=30)
cv.writeVideo("out.avi", [filled], fps="30")  # type: ignore[arg-type]
edges: np.ndarray = cv.autoCanny(filled)
flat: np.ndarray = cv.fourPointTransform(filled, [[0, 0], [9, 0], [9, 9], [0, 9]])
cv.imshowEx("window", filled, maxSize=(800, 600))
pil_image = cv.toPillow(filled)
from_pil: np.ndarray = cv.fromPillow(pil_image)
encoded: str = cv.imencodeBase64(filled, ".jpg", dataUri=True)
decoded_img: np.ndarray | None = cv.imdecodeBase64(encoded)
labeled: np.ndarray = cv.drawLabel(filled, (1, 1, 5, 5), "cat", (0, 255, 0))
cv.drawLabel(filled, (1, 1, 5), "cat", (0, 255, 0))  # type: ignore[arg-type]
tinted: np.ndarray = cv.overlayMask(filled, filled > 0)
cropped: np.ndarray = cv.cropRect(filled, (-5, -5, 20, 20), fill=0)
red_mask: np.ndarray = cv.inRangeEx(filled, (170, 100, 100), (10, 255, 255))
grid: np.ndarray = cv.montage([filled, cropped], cols=2, cellSize=(64, 64))
equalized: np.ndarray = cv.equalizeHistEx(filled, clahe=True)
sharpened: np.ndarray = cv.unsharpMask(filled, sigma=2.0)
label_map, palette_centers = cv.kmeansColors(filled, 4)
clusters: cv.Clusters = cv.kmeansColors(filled, 4)
cluster_maps: np.ndarray = clusters.maps
palette: list[tuple[tuple[int, ...], float]] = cv.dominantColors(filled, k=3)
posterized: np.ndarray = cv.colorQuantize(filled, 8)
auto_map, auto_centers = cv.meanShiftColors(filled)
peak_map, peak_centers = cv.peakColors(filled, minRegion=20)
gray_img: np.ndarray = cv.toGray(filled)
bgr_img: np.ndarray = cv.toBGR(filled)
bgra_img: np.ndarray = cv.toBGRA(filled, alpha=128)
contours: list[np.ndarray] = cv.findContoursEx(filled, minArea=10)
solid: np.ndarray = cv.fillHoles(filled, maxHoleArea=100)
clean: np.ndarray = cv.removeSmallObjects(filled, minArea=20)
found: np.ndarray = cv.findTemplate(filled, filled[:4, :4], threshold=0.9)
trimmed: np.ndarray = cv.cropToContent(filled, tolerance=8)
saved: bool = cv.imwriteEx("out/a.png", filled, makeDirs=True)
orange: tuple[int, ...] = cv.hexToBGR("#ff8800")
hex_text: str = cv.bgrToHex(orange)
toned: np.ndarray = cv.adjustBrightnessContrast(filled, brightness=10, contrast=1.2)
shifted_hue: np.ndarray = cv.adjustHSV(filled, hue=30, saturation=1.5)
leveled: np.ndarray = cv.autoContrast(filled, clip=0.02)
balanced: np.ndarray = cv.whiteBalance(filled, method="grayWorld")
focus: float = cv.sharpnessScore(filled)
similarity: float = cv.ssim(filled, filled)
changes: np.ndarray = cv.diffMask(filled, filled, threshold=30, minArea=5)
moved: np.ndarray = cv.translate(filled, 5, -3)
padded: np.ndarray = cv.padToMultiple(filled, 32)
framed: np.ndarray = cv.padToSize(filled, (640, 640))
tiles = cv.splitTiles(filled, (256, 256), overlap=32)
merged: np.ndarray = cv.mergeTiles(tiles, (640, 480))
cv.drawDashedLine(filled, (0, 0), (10, 10), (0, 255, 0), dashLength=4)
cv.drawDashedRectangle(filled, (1, 1, 8, 8), (0, 255, 0))
histogram: np.ndarray = cv.drawHistogram(filled, log=True)
main_object: np.ndarray = cv.largestComponent(filled)
lines: np.ndarray = cv.skeletonize(filled)
sticker: np.ndarray = cv.cutout(filled, filled[..., 0], feather=2)
thumbnail: np.ndarray | None = cv.readFrameAt("clip.mp4", seconds=1.5)
groups: np.ndarray = cv.clusterPoints([[0, 0], [1, 1]], eps=2.0)
db_maps, db_centers = cv.dbscanColors(filled, 1.7, minSamples=2)
cv.dbscanColors(filled, 1.7, min_samples=2)  # type: ignore[call-arg]
painted: np.ndarray = cv.drawClusters(filled, db_maps)

# Names that are not in the generated stub fall back to Any instead of an error.
cv.some_function_added_in_a_later_opencv_release()
