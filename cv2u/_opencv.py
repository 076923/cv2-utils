"""Import ``cv2`` and check that it is a supported OpenCV, with install instructions if not."""

import re

INSTALL_HINT = (
    'pip install "python-opencv-utils[opencv]" (or [headless], [contrib], [contrib-headless] '
    "for the other OpenCV distributions; install only one of them)"
)

MIN_VERSION = (4, 10)

try:
    import cv2
except ImportError as error:
    # The OpenCV distribution is an extra, so a plain `pip install python-opencv-utils` does not bring one.
    raise ImportError(
        f"cv2u needs OpenCV, but cv2 cannot be imported ({error}). Install it with: {INSTALL_HINT}"
    ) from error

if tuple(int(part) for part in re.findall(r"\d+", cv2.__version__)[:2]) < MIN_VERSION:
    raise ImportError(
        f"cv2u requires OpenCV 4.10 or newer, but cv2 {cv2.__version__} is installed ({cv2.__file__}). "
        f"Upgrade it with: {INSTALL_HINT}"
    )

__all__ = ["INSTALL_HINT", "MIN_VERSION", "cv2"]
