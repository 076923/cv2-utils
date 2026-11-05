"""Python OpenCV Utilities.

``cv2u`` is a drop-in replacement for ``cv2``. Import it the same way the
official OpenCV documentation imports ``cv2`` (``import cv2 as cv``)::

    import cv2u as cv

Every attribute that ``cv2u`` does not define itself is looked up on the
``cv2`` module, so ``cv.imread``, ``cv.COLOR_BGR2GRAY``, ``cv.dnn`` and so on
all work as usual, alongside the extensions provided by this package.

Requires OpenCV 4.10 or newer (including OpenCV 5), from any of the OpenCV
distributions: ``opencv-python``, ``opencv-python-headless``,
``opencv-contrib-python`` or ``opencv-contrib-python-headless``.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

# Imported first: fails with install instructions when OpenCV is missing or too old.
from ._opencv import cv2 as _cv2

if TYPE_CHECKING:
    # Lets type checkers and IDEs see OpenCV's own signatures (``cv.imread`` etc.).
    # At runtime these names come from ``__getattr__`` below.
    from . import _cv2_exports
    from ._cv2_exports import *  # noqa: F403

from .core import (
    DBSCAN,
    DHEdge,
    boxIoU,
    cartesianToPolar,
    centerPoint,
    clusterPoints,
    colorQuantize,
    cropEx,
    cropToContent,
    dominantColors,
    drawDBSCAN,
    gammaCorrection,
    hconcatEx,
    imcolor,
    imcolorLike,
    inRangeEx,
    isInside,
    kmeansColors,
    makeGradient,
    meanShiftColors,
    mergeTiles,
    montage,
    overlayImage,
    padToMultiple,
    padToSize,
    peakColors,
    polarToCartesian,
    rotateEx,
    splitTiles,
    vconcatEx,
)
from .flag import (
    ALL_CONTOURS,
    DDEPTH_SAME,
    FLIP_MODE_X,
    FLIP_MODE_XY,
    FLIP_MODE_Y,
    KEY_BACKSPACE,
    KEY_ENTER,
    KEY_ESC,
    KEY_SPACE,
    KEY_TAB,
    REDUCE_TO_COL,
    REDUCE_TO_ROW,
    WAIT_FOREVER,
)
from .highgui import (
    imshowEx,
    imshowInline,
)
from .imgcodecs import (
    fromPillow,
    imdecodeBase64,
    imencodeBase64,
    imreadEx,
    imwriteEx,
    listImages,
    toPillow,
    urlread,
)
from .imgproc import (
    adjustBrightnessContrast,
    adjustHSV,
    autoCanny,
    autoContrast,
    bgrToHex,
    colorTransfer,
    cropRotatedRect,
    cutout,
    deskew,
    diffMask,
    drawDashedLine,
    drawDashedRectangle,
    drawHistogram,
    drawLabel,
    drawRoundedRectangle,
    equalizeHistEx,
    fillHoles,
    findContoursEx,
    findTemplate,
    flattenAlpha,
    fourPointTransform,
    hashDistance,
    hexToBGR,
    imageHash,
    largestComponent,
    letterbox,
    matchHistograms,
    overlayMask,
    pixelate,
    putTextEx,
    removeSmallObjects,
    resizeEx,
    sharpnessScore,
    skeletonize,
    ssim,
    toBGR,
    toBGRA,
    toGray,
    transformPoints,
    translate,
    unsharpMask,
    whiteBalance,
)
from .videoio import (
    VideoInfo,
    readFrameAt,
    readFrames,
    videoInfo,
    writeVideo,
)

try:
    __version__ = version("python-opencv-utils")
except PackageNotFoundError:  # pragma: no cover - running from a source tree that is not installed
    __version__ = "0.0.0"

# Only the cv2u extensions. OpenCV names are reachable as attributes, not through ``import *``.
__all__ = [
    "ALL_CONTOURS",
    "DBSCAN",
    "DDEPTH_SAME",
    "FLIP_MODE_X",
    "FLIP_MODE_XY",
    "FLIP_MODE_Y",
    "KEY_BACKSPACE",
    "KEY_ENTER",
    "KEY_ESC",
    "KEY_SPACE",
    "KEY_TAB",
    "REDUCE_TO_COL",
    "REDUCE_TO_ROW",
    "WAIT_FOREVER",
    "DHEdge",
    "VideoInfo",
    "adjustBrightnessContrast",
    "adjustHSV",
    "autoCanny",
    "autoContrast",
    "bgrToHex",
    "boxIoU",
    "cartesianToPolar",
    "centerPoint",
    "clusterPoints",
    "colorQuantize",
    "colorTransfer",
    "cropEx",
    "cropRotatedRect",
    "cropToContent",
    "cutout",
    "deskew",
    "diffMask",
    "dominantColors",
    "drawDBSCAN",
    "drawDashedLine",
    "drawDashedRectangle",
    "drawHistogram",
    "drawLabel",
    "drawRoundedRectangle",
    "equalizeHistEx",
    "fillHoles",
    "findContoursEx",
    "findTemplate",
    "flattenAlpha",
    "fourPointTransform",
    "fromPillow",
    "gammaCorrection",
    "hashDistance",
    "hconcatEx",
    "hexToBGR",
    "imageHash",
    "imcolor",
    "imcolorLike",
    "imdecodeBase64",
    "imencodeBase64",
    "imreadEx",
    "imshowEx",
    "imshowInline",
    "imwriteEx",
    "inRangeEx",
    "isInside",
    "kmeansColors",
    "largestComponent",
    "letterbox",
    "listImages",
    "makeGradient",
    "matchHistograms",
    "meanShiftColors",
    "mergeTiles",
    "montage",
    "overlayImage",
    "overlayMask",
    "padToMultiple",
    "padToSize",
    "peakColors",
    "pixelate",
    "polarToCartesian",
    "putTextEx",
    "readFrameAt",
    "readFrames",
    "removeSmallObjects",
    "resizeEx",
    "rotateEx",
    "sharpnessScore",
    "skeletonize",
    "splitTiles",
    "ssim",
    "toBGR",
    "toBGRA",
    "toGray",
    "toPillow",
    "transformPoints",
    "translate",
    "unsharpMask",
    "urlread",
    "vconcatEx",
    "videoInfo",
    "whiteBalance",
    "writeVideo",
]

if TYPE_CHECKING:
    # Mark the OpenCV names as public API of this typed package (pyright's reportPrivateImportUsage,
    # mypy's --no-implicit-reexport). Runtime ``from cv2u import *`` still exports only the extensions.
    __all__ += _cv2_exports.__all__


def __getattr__(name: str) -> Any:
    # OpenCV's ``cv2/__init__.py`` defines ``__all__ = []``, so
    # ``from cv2 import *`` re-exports nothing. Delegate lookups instead.
    try:
        return getattr(_cv2, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_cv2)))
