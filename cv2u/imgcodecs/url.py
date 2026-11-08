from collections.abc import Mapping
from urllib.request import Request, urlopen

import cv2
import numpy as np

__all__ = ["imreadUrl"]

# Many image hosts reject urllib's default "Python-urllib/3.x" user agent with HTTP 403.
_USER_AGENT = "cv2u (+https://github.com/076923/cv2-utils)"


def imreadUrl(
    url: str,
    flags: int = cv2.IMREAD_COLOR,
    timeout: float | None = None,
    headers: Mapping[str, str] | None = None,
    maxBytes: int | None = None,
) -> np.ndarray | None:
    """Download an image from ``url`` and decode it like ``cv.imread`` (which cannot read from a URL).

    Args:
        url: ``http(s)://`` or ``file://`` URL of the image.
        flags: ``cv.IMREAD_*`` flags, as for ``cv.imread`` (``cv.IMREAD_COLOR`` by default, like ``cv.imread``;
            pass ``cv.IMREAD_UNCHANGED`` to keep an alpha channel).
        timeout: Timeout in seconds for the request. ``None`` uses the global default.
        headers: Extra HTTP headers, e.g. ``{"Authorization": "Bearer ..."}``. A ``User-Agent`` given here
            replaces the default one.
        maxBytes: Largest response accepted, in bytes. Larger downloads stop early with a ``ValueError`` instead
            of being held in memory. ``None`` accepts any size.

    Returns:
        The decoded image, or ``None`` if the data cannot be decoded (like ``cv.imread``).

    Raises:
        urllib.error.URLError: If the request fails (including HTTP errors).
        ValueError: If ``maxBytes`` is below 1, or the response is larger than ``maxBytes``.
    """
    if maxBytes is not None and maxBytes < 1:
        raise ValueError(f"maxBytes must be at least 1, got {maxBytes}")
    request = Request(url, headers={"User-Agent": _USER_AGENT, **(headers or {})})
    with urlopen(request, timeout=timeout) as response:
        if maxBytes is None:
            raw = response.read()
        else:
            declared = response.headers.get("Content-Length")
            if declared is not None and declared.isdigit() and int(declared) > maxBytes:
                raise ValueError(f"response is {declared} bytes, more than maxBytes={maxBytes}")
            raw = response.read(maxBytes + 1)  # one extra byte tells a too-large body from an exact fit
            if len(raw) > maxBytes:
                raise ValueError(f"response is larger than maxBytes={maxBytes}")
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)
