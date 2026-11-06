"""Exchange images with Pillow and with text-based APIs (base64, data URIs)."""

import base64
import binascii
from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image

from .._validate import check_image_not_empty

__all__ = ["fromPillow", "imdecodeBase64", "imencodeBase64", "toPillow"]

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def toPillow(src: np.ndarray) -> Image.Image:
    """Convert an OpenCV image to a Pillow image, fixing the channel order (BGR to RGB, BGRA to RGBA).

    Every single-channel dtype Pillow has a mode for is accepted, so masks, depth maps and float images round-trip
    through :func:`fromPillow` unchanged:

    | ``src``                        | Pillow mode |
    |--------------------------------|-------------|
    | ``uint8`` gray                 | ``"L"``     |
    | ``uint8`` gray + alpha (2 ch.) | ``"LA"``    |
    | ``uint8`` BGR / BGRA           | ``"RGB"`` / ``"RGBA"`` |
    | ``uint16`` gray                | ``"I;16"``  |
    | ``int32`` gray                 | ``"I"``     |
    | ``float32`` gray               | ``"F"``     |
    | ``bool`` gray                  | ``"1"``     |

    Args:
        src: Image to convert. Color images must be 8-bit; Pillow has no 16-bit or float color modes.

    Raises:
        ValueError: If the image is empty, or its dtype or channel count has no Pillow mode.
    """
    check_image_not_empty("src", src)
    if src.ndim == 2 or (src.ndim == 3 and src.shape[2] == 1):
        gray = np.ascontiguousarray(src.reshape(src.shape[:2]))
        if gray.dtype in (np.uint8, np.uint16, np.int32, np.float32, np.bool_):
            return Image.fromarray(gray)  # "L", "I;16", "I", "F", "1"
        raise ValueError(f"grayscale images must be uint8, uint16, int32, float32 or bool, got {src.dtype}")
    if src.ndim != 3:
        raise ValueError(f"src must be a 2-D or 3-D image, got shape {src.shape}")
    if src.dtype != np.uint8:
        raise ValueError(f"color images must be 8-bit (uint8), got {src.dtype}")
    if src.shape[2] == 2:
        return Image.fromarray(np.ascontiguousarray(src), mode="LA")
    if src.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    if src.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA))
    raise ValueError(f"src must have 1, 2, 3 or 4 channels, got shape {src.shape}")


def fromPillow(image: Image.Image) -> np.ndarray:
    """Convert a Pillow image to an OpenCV image: BGR, BGRA (with transparency) or grayscale.

    Handles the cases where ``np.asarray(image)`` silently gives the wrong result: palette images (``"P"``, which
    would give palette indices instead of colors), ``"CMYK"`` (4 channels that are not BGRA), ``"YCbCr"``, ``"LAB"``,
    ``"HSV"``, ``"LA"``, premultiplied-alpha (``"RGBa"``, ``"La"``), 1-bit and big-endian 16-bit images. Every Pillow
    mode is accepted.

    Returns:
        ``uint8`` BGR for color images, BGRA for images with transparency (a palette with a transparent entry,
        ``"RGBA"``, ``"LA"``, ``"PA"``, ``"RGBa"``, ``"La"``), a 2-D array for grayscale: ``uint8`` for ``"L"`` and
        ``"1"``, ``uint16`` for 16-bit (``"I;16"`` in any byte order), ``int32`` for ``"I"``, ``float32`` for ``"F"``.
    """
    mode = image.mode
    if mode in ("L", "I", "F"):
        return np.array(image)  # 8-bit, 32-bit integer or float grayscale, as is
    if mode.startswith("I;16"):
        return np.array(image, dtype=np.uint16)  # native byte order, whatever the file used
    if mode == "1":
        return np.array(image.convert("L"))
    if mode == "La":
        image = image.convert("LA")  # Pillow cannot convert premultiplied "La" to RGB(A) directly
        mode = "LA"
    has_alpha = mode in ("RGBA", "RGBa", "LA", "PA") or (mode == "P" and "transparency" in image.info)
    if has_alpha:
        return cv2.cvtColor(np.array(image.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def imencodeBase64(img: np.ndarray, ext: str = ".png", params: Sequence[int] = (), dataUri: bool = False) -> str:
    """Encode an image as a base64 string, e.g. for JSON APIs or HTML ``<img src=...>``.

    Args:
        img: Image to encode.
        ext: Image format, as for ``cv.imencode`` (``".png"``, ``".jpg"``, ``".webp"``, ...).
        params: Format-specific ``cv.IMWRITE_*`` parameter pairs.
        dataUri: Return a data URI (``"data:image/png;base64,..."``) instead of the bare base64 text.

    Raises:
        ValueError: If the image cannot be encoded in this format.
    """
    try:
        ok, buffer = cv2.imencode(ext, img, list(params))
    except cv2.error as e:
        raise ValueError(f"cannot encode the image as {ext!r}") from e
    if not ok:
        raise ValueError(f"cannot encode the image as {ext!r}")
    text = base64.b64encode(buffer.tobytes()).decode("ascii")
    if not dataUri:
        return text
    mime = _MIME_TYPES.get(ext.lower() if ext.startswith(".") else f".{ext.lower()}", "application/octet-stream")
    return f"data:{mime};base64,{text}"


def imdecodeBase64(data: str | bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode a base64 string or a data URI (``"data:image/png;base64,..."``) into an image.

    Args:
        data: Base64 text (``str`` or ``bytes``), with or without a ``data:`` prefix. Whitespace is ignored.
        flags: ``cv.IMREAD_*`` flags, as for ``cv.imdecode``.

    Returns:
        The image, or ``None`` if the bytes are not a decodable image (like ``cv.imdecode``).

    Raises:
        ValueError: If ``data`` is not valid base64.
    """
    text = data.decode("ascii", errors="replace") if isinstance(data, bytes) else data
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    try:
        raw = base64.b64decode("".join(text.split()), validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("data is not valid base64") from e
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)
