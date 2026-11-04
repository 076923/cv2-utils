from collections.abc import Mapping
from urllib.request import Request, urlopen

import cv2
import numpy as np

__all__ = ["urlread"]

# Many image hosts reject urllib's default "Python-urllib/3.x" user agent with HTTP 403.
_USER_AGENT = "cv2u (+https://github.com/076923/cv2-utils)"


def urlread(
    url: str,
    flags: int = cv2.IMREAD_UNCHANGED,
    timeout: float | None = None,
    headers: Mapping[str, str] | None = None,
) -> np.ndarray | None:
    """Download an image from ``url`` and decode it like ``cv.imread``.

    Args:
        url: ``http(s)://`` or ``file://`` URL of the image.
        flags: ``cv.IMREAD_*`` flags, as for ``cv.imread``.
        timeout: Timeout in seconds for the request. ``None`` uses the global default.
        headers: Extra HTTP headers, e.g. ``{"Authorization": "Bearer ..."}``. A ``User-Agent`` given here
            replaces the default one.

    Returns:
        The decoded image, or ``None`` if the data cannot be decoded (like ``cv.imread``).

    Raises:
        urllib.error.URLError: If the request fails (including HTTP errors).
    """
    request = Request(url, headers={"User-Agent": _USER_AGENT, **(headers or {})})
    with urlopen(request, timeout=timeout) as response:
        data = np.frombuffer(response.read(), dtype=np.uint8)
    return cv2.imdecode(data, flags)
