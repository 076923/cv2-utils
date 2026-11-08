"""Image file metadata, EXIF orientation, and reading / encoding helpers with clear failure modes."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from ..imgproc.color import flattenAlpha
from ..imgproc.resize import resizeEx
from .files import imreadEx, imwriteEx

__all__ = [
    "ImageInfo",
    "convertImageFormat",
    "exifOrientation",
    "imageInfo",
    "imdecodeEx",
    "imencodeEx",
    "imreadOriented",
    "imreadResized",
]


class ImageInfo(NamedTuple):
    """Header information of an image file, from :func:`imageInfo`."""

    width: int
    """Stored width in pixels (before the EXIF orientation is applied)."""
    height: int
    """Stored height in pixels (before the EXIF orientation is applied)."""
    channels: int
    """Number of channels of the stored image (1 for grayscale and palette images, 3 for RGB, 4 for RGBA / CMYK)."""
    mode: str
    """Pillow mode, e.g. ``"RGB"``, ``"RGBA"``, ``"L"``, ``"P"``, ``"I;16"``."""
    format: str
    """File format, e.g. ``"JPEG"``, ``"PNG"``, ``"WEBP"``, ``"TIFF"``."""
    fileSize: int
    """Size of the file in bytes."""
    exifOrientation: int
    """EXIF orientation tag (1-8); 1 when the file has none."""


def _open(filename: str | os.PathLike[str]) -> Image.Image:
    """Open an image lazily with Pillow; only the header is parsed."""
    try:
        return Image.open(os.fspath(filename))
    except UnidentifiedImageError as e:
        raise OSError(f"cannot read image: {os.fspath(filename)!r}") from e


def _orientation(image: Image.Image) -> int:
    value = image.getexif().get(0x0112, 1)
    return int(value) if isinstance(value, int) and 1 <= value <= 8 else 1


def imageInfo(filename: str | os.PathLike[str]) -> ImageInfo:
    """Read the size, channels, format, file size and EXIF orientation of an image file without decoding it.

    ``cv.imread`` has to decode the whole file to tell you its size; here Pillow parses only the header, so
    listing the dimensions of a folder of large photos is fast. Works with any path, including non-ASCII paths.

    Args:
        filename: Path to the image file.

    Returns:
        An :class:`ImageInfo` named tuple ``(width, height, channels, mode, format, fileSize, exifOrientation)``.
        ``width`` and ``height`` are as stored; for ``exifOrientation`` 5-8 the displayed image is ``height`` wide.

    Raises:
        OSError: If the file does not exist or is not an image Pillow recognizes.
    """
    with _open(filename) as image:
        width, height = image.size
        return ImageInfo(
            width=int(width),
            height=int(height),
            channels=len(image.getbands()),
            mode=image.mode,
            format=image.format or "",
            fileSize=os.path.getsize(os.fspath(filename)),
            exifOrientation=_orientation(image),
        )


def exifOrientation(filename: str | os.PathLike[str]) -> int:
    """Return the EXIF orientation tag (1-8) of an image file, or 1 when it has none.

    Only the file header is read. Values: 1 upright, 2 mirrored, 3 rotated 180°, 4 flipped vertically,
    5 transposed, 6 rotated 90° clockwise, 7 transversed, 8 rotated 90° counter-clockwise (the image must be
    rotated / flipped that way to display upright, see :func:`imreadOriented`).

    Raises:
        OSError: If the file does not exist or is not an image Pillow recognizes.
    """
    with _open(filename) as image:
        return _orientation(image)


def _apply_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate / flip ``img`` so that an image stored with this EXIF orientation displays upright."""
    if orientation == 2:
        return cv2.flip(img, 1)
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(img, 0)
    if orientation == 5:
        return cv2.transpose(img)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(img), -1)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _read_header_orientation(filename: str | os.PathLike[str]) -> int:
    try:
        with Image.open(os.fspath(filename)) as image:
            return _orientation(image)
    except OSError:
        return 1  # formats Pillow cannot open (e.g. .exr, .hdr) have no EXIF orientation anyway


def imreadOriented(filename: str | os.PathLike[str], flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image and apply its EXIF orientation, whatever the ``flags`` and the file format.

    OpenCV honors the orientation tag only in some cases: with ``cv.IMREAD_COLOR`` / ``cv.IMREAD_GRAYSCALE`` it
    rotates JPEG, PNG, WebP and TIFF files, but with ``cv.IMREAD_UNCHANGED`` (needed to keep the alpha channel or
    16-bit depth) and with every ``cv.IMREAD_REDUCED_*`` flag the image comes out as stored, i.e. sideways for
    phone photos. This reads the tag with Pillow and applies the rotation / flip itself, so the result is upright
    for every flag; like :func:`imreadEx` it works with non-ASCII paths.

    Args:
        filename: Path to the image file.
        flags: ``cv.IMREAD_*`` flags, as for ``cv.imread``. ``cv.IMREAD_IGNORE_ORIENTATION`` is added internally so
            the orientation is never applied twice.

    Returns:
        The upright image, or ``None`` if the file cannot be read or decoded (like ``cv.imread``).
    """
    img = imreadEx(filename, flags | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return None
    return _apply_orientation(img, _read_header_orientation(filename))


_REDUCED_FLAGS = {
    (cv2.IMREAD_COLOR, 2): cv2.IMREAD_REDUCED_COLOR_2,
    (cv2.IMREAD_COLOR, 4): cv2.IMREAD_REDUCED_COLOR_4,
    (cv2.IMREAD_COLOR, 8): cv2.IMREAD_REDUCED_COLOR_8,
    (cv2.IMREAD_GRAYSCALE, 2): cv2.IMREAD_REDUCED_GRAYSCALE_2,
    (cv2.IMREAD_GRAYSCALE, 4): cv2.IMREAD_REDUCED_GRAYSCALE_4,
    (cv2.IMREAD_GRAYSCALE, 8): cv2.IMREAD_REDUCED_GRAYSCALE_8,
}


def imreadResized(
    filename: str | os.PathLike[str],
    size: tuple[int, int] | None = None,
    scale: float | None = None,
    width: int | None = None,
    height: int | None = None,
    flags: int = cv2.IMREAD_COLOR,
    interpolation: int | None = None,
) -> np.ndarray | None:
    """Read an image and resize it in one call, decoding JPEG files at reduced size when possible.

    Give exactly one of ``size`` (exact ``(width, height)``), ``scale`` (a factor), ``width`` or ``height`` (the
    other side keeps the aspect ratio, see :func:`resizeEx`). For ``scale`` 1/2, 1/4 or 1/8 with
    ``cv.IMREAD_COLOR`` / ``cv.IMREAD_GRAYSCALE`` the file is decoded with the matching ``cv.IMREAD_REDUCED_*`` flag,
    which for JPEG skips most of the decoding work (other formats are decoded in full and then shrunk). The EXIF
    orientation is applied as in :func:`imreadOriented` (OpenCV ignores it for reduced decoding).

    Args:
        filename: Path to the image file.
        size: Target size as ``(width, height)``.
        scale: Scale factor, e.g. ``0.5`` for half size.
        width: Target width; the height follows the aspect ratio.
        height: Target height; the width follows the aspect ratio.
        flags: ``cv.IMREAD_*`` flags, as for ``cv.imread``.
        interpolation: ``cv.INTER_*`` flag, as for :func:`resizeEx`.

    Returns:
        The resized image, or ``None`` if the file cannot be read or decoded (like ``cv.imread``).

    Raises:
        ValueError: If not exactly one of ``size``, ``scale``, ``width`` and ``height`` is given, or it is not
            positive.
    """
    given = [
        name
        for name, value in (("size", size), ("scale", scale), ("width", width), ("height", height))
        if value is not None
    ]
    if len(given) != 1:
        raise ValueError(f"give exactly one of size, scale, width or height, got {given or 'none'}")
    if size is not None and (size[0] <= 0 or size[1] <= 0):
        raise ValueError(f"size must be positive, got {size[0]}x{size[1]}")
    if scale is not None and not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be a positive finite number, got {scale}")

    reduced = _REDUCED_FLAGS.get((flags, round(1 / scale))) if scale is not None and 1 / scale in (2, 4, 8) else None
    img = imreadEx(filename, (reduced if reduced is not None else flags) | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return None
    orientation = _read_header_orientation(filename)
    if scale is not None:
        if reduced is not None:
            full_w, full_h = _stored_size(filename, img, round(1 / scale))
        else:
            full_h, full_w = img.shape[:2]
        target: tuple[int, int] = (max(1, int(full_w * scale + 0.5)), max(1, int(full_h * scale + 0.5)))
    elif size is not None:
        target = (int(size[0]), int(size[1]))
        if orientation >= 5:
            target = (target[1], target[0])  # the final rotation swaps the sides back
    else:
        if orientation >= 5:
            width, height = height, width
        img = resizeEx(img, width=width, height=height, interpolation=interpolation)
        return _apply_orientation(img, orientation)
    if (img.shape[1], img.shape[0]) != target:
        img = resizeEx(img, width=target[0], height=target[1], interpolation=interpolation)
    return _apply_orientation(img, orientation)


def _stored_size(filename: str | os.PathLike[str], reduced: np.ndarray, factor: int) -> tuple[int, int]:
    """Full stored ``(width, height)``, from the header when Pillow can read it, else from the reduced image."""
    try:
        with Image.open(os.fspath(filename)) as image:
            return int(image.size[0]), int(image.size[1])
    except OSError:
        return reduced.shape[1] * factor, reduced.shape[0] * factor


def imencodeEx(ext: str, img: np.ndarray, params: Sequence[int] = ()) -> bytes:
    """Encode an image to ``bytes`` like ``cv.imencode``, raising instead of returning ``(False, None)``.

    ``cv.imencode`` returns an ``(ok, buffer)`` tuple and a NumPy array, so sending the result over HTTP or
    writing it to a stream takes an unpacking, a check and ``.tobytes()`` every time. Here encoding failures
    (unknown extension, unsupported dtype or channel count) raise a ``ValueError`` naming the format.

    Args:
        ext: File extension that selects the format, e.g. ``".png"``, ``".jpg"``, ``".webp"`` (``"png"`` works too).
        img: Image to encode.
        params: Format-specific ``cv.IMWRITE_*`` parameter pairs, e.g. ``(cv.IMWRITE_JPEG_QUALITY, 90)``.

    Returns:
        The encoded file content.

    Raises:
        ValueError: If the image is empty or cannot be encoded in this format.
    """
    if img.size == 0:
        raise ValueError(f"img must not be empty, got shape {img.shape}")
    if not ext.startswith("."):
        ext = f".{ext}"
    try:
        ok, buffer = cv2.imencode(ext, img, list(params))
    except cv2.error as e:
        raise ValueError(f"cannot encode the image as {ext!r}: no encoder or unsupported image type") from e
    if not ok:
        raise ValueError(f"cannot encode the image as {ext!r}")
    return bytes(buffer.tobytes())


def imdecodeEx(data: bytes | bytearray | memoryview | np.ndarray, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode an image from ``bytes`` (an HTTP body, a file object's ``read()``) like ``cv.imdecode``.

    ``cv.imdecode`` only accepts a NumPy array, so ``np.frombuffer(data, np.uint8)`` is repeated before every
    call and fails on empty input. Any bytes-like object or ``uint8`` array is accepted here.

    Args:
        data: The encoded file content.
        flags: ``cv.IMREAD_*`` flags, as for ``cv.imdecode``.

    Returns:
        The image, or ``None`` if the data is empty or not a decodable image (like ``cv.imdecode``).
    """
    buffer = data if isinstance(data, np.ndarray) else np.frombuffer(data, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(np.ascontiguousarray(buffer).reshape(-1), flags)


_ALPHA_FORMATS = frozenset({".png", ".webp", ".tif", ".tiff", ".avif", ".bmp", ".jp2"})
_QUALITY_FLAGS = {
    ".jpg": cv2.IMWRITE_JPEG_QUALITY,
    ".jpeg": cv2.IMWRITE_JPEG_QUALITY,
    ".jpe": cv2.IMWRITE_JPEG_QUALITY,
    ".webp": cv2.IMWRITE_WEBP_QUALITY,
    ".avif": cv2.IMWRITE_AVIF_QUALITY,
}


def convertImageFormat(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    quality: int | None = None,
    background: float | Sequence[float] = (255, 255, 255),
    makeDirs: bool = False,
) -> bool:
    """Convert an image file to another format, chosen from the extension of ``dst``.

    The file is read as stored (``cv.IMREAD_UNCHANGED``: alpha channel and 16-bit depth are kept) with the EXIF
    orientation applied, since the tag is not carried over to the new file. Transparent images are composited
    onto ``background`` with :func:`flattenAlpha` when the target format has no alpha (JPEG), where
    ``cv.imwrite`` would silently drop the alpha channel and expose the hidden pixels.

    Args:
        src: Path of the image to convert.
        dst: Path to write to; the format comes from its extension.
        quality: JPEG / WebP / AVIF quality (0-100). Ignored for other formats.
        background: Background color for flattening transparency, white by default.
        makeDirs: Create missing parent directories of ``dst``.

    Returns:
        ``True`` on success, ``False`` if ``src`` cannot be read or ``dst`` cannot be written (like ``cv.imwrite``).

    Raises:
        ValueError: If ``dst`` has no file extension.
        cv2.error: If no encoder exists for the extension of ``dst``, like ``cv.imwrite``.
    """
    ext = Path(os.fspath(dst)).suffix.lower()
    if not ext:
        raise ValueError(f"cannot determine the image format: {os.fspath(dst)!r} has no file extension")
    img = imreadOriented(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if img.ndim == 3 and img.shape[2] == 4 and ext not in _ALPHA_FORMATS:
        img = flattenAlpha(img, background)
    params: list[int] = []
    if quality is not None and ext in _QUALITY_FLAGS:
        params = [_QUALITY_FLAGS[ext], int(quality)]
    return imwriteEx(dst, img, params, makeDirs)
