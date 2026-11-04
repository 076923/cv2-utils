"""Exchange images with Pillow and with text-based APIs (base64, data URIs)."""

import base64
import binascii
from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image

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

    Args:
        src: 8-bit grayscale, BGR or BGRA image, or a 16-bit grayscale image.

    Raises:
        ValueError: If the dtype or channel count is not supported.
    """
    if src.ndim == 2 or (src.ndim == 3 and src.shape[2] == 1):
        gray = src.reshape(src.shape[:2])
        if gray.dtype == np.uint8:
            return Image.fromarray(gray, mode="L")
        if gray.dtype == np.uint16:
            return Image.fromarray(gray)  # mode "I;16"
        raise ValueError(f"grayscale images must be uint8 or uint16, got {src.dtype}")
    if src.dtype != np.uint8:
        raise ValueError(f"color images must be 8-bit (uint8), got {src.dtype}")
    if src.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    if src.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA))
    raise ValueError(f"src must have 1, 3 or 4 channels, got shape {src.shape}")


def fromPillow(image: Image.Image) -> np.ndarray:
    """Convert a Pillow image to an OpenCV image: BGR, BGRA (with transparency) or grayscale.

    Handles the cases where ``np.asarray(image)`` silently gives the wrong result: palette images (``"P"``, which
    would give palette indices instead of colors), ``"CMYK"`` (4 channels that are not BGRA), ``"YCbCr"``, ``"LA"`` and
    1-bit images.

    Returns:
        ``uint8`` BGR for color images, BGRA for images with transparency, a 2-D array for grayscale
        (``uint16`` for 16-bit grayscale).
    """
    mode = image.mode
    if mode in ("L", "I;16"):
        return np.array(image)
    if mode == "1":
        return np.array(image.convert("L"))
    if mode in ("I", "F"):
        return np.array(image)  # 32-bit integer or float grayscale, as is
    has_alpha = mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in image.info)
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
