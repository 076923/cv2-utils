"""Named constants for the magic numbers that OpenCV functions take without a constant of their own.

OpenCV already has constants for most flags (``cv.ROTATE_90_CLOCKWISE``, ``cv.FILLED``, ``cv.FILTER_SCHARR``, ...).
These cover the remaining cases, so code reads ``cv.flip(img, cv.FLIP_MODE_Y)`` instead of ``cv.flip(img, 1)``.
"""

__all__ = [
    "ALL_CONTOURS",
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
]

# --- cv.flip(src, flipCode) -----------------------------------------------------------

FLIP_MODE_X = 0
"""``cv.flip`` flipCode: flip around the x-axis (upside down)."""
FLIP_MODE_Y = 1
"""``cv.flip`` flipCode: flip around the y-axis (mirror left-right)."""
FLIP_MODE_XY = -1
"""``cv.flip`` flipCode: flip around both axes (same as rotating by 180°)."""

# --- depth ------------------------------------------------------------------------------

DDEPTH_SAME = -1
"""``ddepth``/``dtype`` meaning "same depth as the source", e.g. ``cv.filter2D(src, cv.DDEPTH_SAME, kernel)``.

Accepted by ``cv.filter2D``, ``cv.sepFilter2D``, ``cv.boxFilter``, ``cv.Sobel``, ``cv.Scharr``, ``cv.Laplacian``,
and as ``dtype`` by ``cv.add``, ``cv.subtract``, ``cv.multiply`` and ``cv.divide``.
"""

# --- cv.drawContours(image, contours, contourIdx, ...) ------------------------------------

ALL_CONTOURS = -1
"""``cv.drawContours`` contourIdx: draw every contour.

Example: ``cv.drawContours(img, contours, cv.ALL_CONTOURS, color)``.
"""

# --- cv.reduce(src, dim, rtype) -----------------------------------------------------------

REDUCE_TO_ROW = 0
"""``cv.reduce`` dim: reduce every column, giving a single row (shape ``(1, width)``)."""
REDUCE_TO_COL = 1
"""``cv.reduce`` dim: reduce every row, giving a single column (shape ``(height, 1)``)."""

# --- cv.waitKey -----------------------------------------------------------------------------

WAIT_FOREVER = 0
"""``cv.waitKey`` delay: wait until a key is pressed, e.g. ``cv.waitKey(cv.WAIT_FOREVER)``."""

KEY_BACKSPACE = 8
"""Key code returned by ``cv.waitKey`` for Backspace."""
KEY_TAB = 9
"""Key code returned by ``cv.waitKey`` for Tab."""
KEY_ENTER = 13
"""Key code returned by ``cv.waitKey`` for Enter. Some Linux backends return 10 (``"\\n"``) instead."""
KEY_ESC = 27
"""Key code returned by ``cv.waitKey`` for Escape, e.g. ``if cv.waitKey(1) == cv.KEY_ESC: break``."""
KEY_SPACE = 32
"""Key code returned by ``cv.waitKey`` for the space bar."""
