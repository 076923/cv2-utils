"""``cv2u.dnn``: inference helpers on top of ``cv.dnn``, plus everything from ``cv2.dnn``.

Because ``cv2u`` replaces ``cv2``, this package stands in for ``cv2.dnn`` too: ``cv.dnn.readNet``,
``cv.dnn.NMSBoxes``, ``cv.dnn.DNN_BACKEND_OPENCV`` and the rest are looked up on ``cv2.dnn`` when they are not
defined here.
"""

from typing import Any

import cv2

from .inference import (
    Detections,
    PreprocessInfo,
    drawDetections,
    inferImage,
    loadModel,
    postprocessClassification,
    postprocessDetections,
    preprocessImage,
    readClassNames,
    sigmoid,
    softmax,
)

__all__ = [
    "Detections",
    "PreprocessInfo",
    "drawDetections",
    "inferImage",
    "loadModel",
    "postprocessClassification",
    "postprocessDetections",
    "preprocessImage",
    "readClassNames",
    "sigmoid",
    "softmax",
]

_cv2_dnn: Any = getattr(cv2, "dnn", None)


def __getattr__(name: str) -> Any:
    if _cv2_dnn is not None:
        try:
            return getattr(_cv2_dnn, name)
        except AttributeError:
            pass
    raise AttributeError(f"module 'cv2u.dnn' has no attribute {name!r} (not in cv2u and not in cv2.dnn)")


def __dir__() -> list[str]:
    names = set(globals())
    if _cv2_dnn is not None:
        names |= set(dir(_cv2_dnn))
    return sorted(names)
