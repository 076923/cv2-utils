import cv2
import numpy as np

from .color import toGray

__all__ = ["findTemplate"]

_NORMED_METHODS = {cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED}


def findTemplate(
    src: np.ndarray,
    templ: np.ndarray,
    threshold: float = 0.8,
    maxCount: int | None = None,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> list[tuple[int, int, int, int, float]]:
    """Find where ``templ`` appears in ``src``: the matches as boxes, not a score map.

    ``cv.matchTemplate`` returns a map of scores; turning it into matches means thresholding, finding maxima with
    ``cv.minMaxLoc`` and suppressing the overlapping neighbors of each match. This does all of that.

    Args:
        src: Image to search in.
        templ: Template image, not larger than ``src``. If one of the two is grayscale, both are compared in
            grayscale.
        threshold: Minimum score in ``[0, 1]``; 1 is a perfect match.
        maxCount: Return at most this many matches.
        method: ``cv.TM_CCOEFF_NORMED`` (default), ``cv.TM_CCORR_NORMED`` or ``cv.TM_SQDIFF_NORMED``.
            For ``TM_SQDIFF_NORMED`` the score is ``1 - difference``, so higher is always better.

    Returns:
        ``(x, y, width, height, score)`` for each match, best first; the boxes do not overlap by more than half
        the template in either direction.

    Raises:
        ValueError: If ``method`` is not normalized, ``threshold`` is outside ``[0, 1]`` or ``templ`` is larger than
            ``src``.
    """
    if method not in _NORMED_METHODS:
        raise ValueError("method must be cv.TM_CCOEFF_NORMED, cv.TM_CCORR_NORMED or cv.TM_SQDIFF_NORMED")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    h, w = templ.shape[:2]
    if h > src.shape[0] or w > src.shape[1]:
        raise ValueError(f"templ {templ.shape[:2]} is larger than src {src.shape[:2]}")
    if (src.ndim == 2) != (templ.ndim == 2) or (src.ndim == 3 and src.shape[2] != templ.shape[2]):
        src, templ = toGray(src), toGray(templ)

    scores = cv2.matchTemplate(src, templ, method)
    if method == cv2.TM_SQDIFF_NORMED:
        scores = 1.0 - scores
    scores = np.nan_to_num(scores, nan=0.0)  # flat regions give NaN with TM_CCOEFF_NORMED

    matches: list[tuple[int, int, int, int, float]] = []
    half_w, half_h = max(1, w // 2), max(1, h // 2)
    while maxCount is None or len(matches) < maxCount:
        _, best, _, (x, y) = cv2.minMaxLoc(scores)
        if best < threshold:
            break
        matches.append((int(x), int(y), int(w), int(h), float(best)))
        # Suppress neighbors closer than half the template, so a match is not reported twice.
        scores[max(0, y - half_h) : y + half_h + 1, max(0, x - half_w) : x + half_w + 1] = -1.0
    return matches
