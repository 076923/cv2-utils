import cv2
import numpy as np

__all__ = ["equalizeHistEx", "pixelate", "unsharpMask"]


def equalizeHistEx(
    src: np.ndarray,
    clahe: bool = False,
    clipLimit: float = 2.0,
    tileGridSize: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """``cv.equalizeHist`` for color images too, optionally with CLAHE (adaptive equalization).

    ``cv.equalizeHist`` only accepts grayscale images. Equalizing B, G and R separately shifts the colors, so color
    images are equalized on the lightness channel of the Lab color space instead. The alpha channel of BGRA images is
    kept.

    Args:
        src: 8-bit grayscale, BGR or BGRA image.
        clahe: Use ``cv.createCLAHE`` (local, contrast-limited) instead of global equalization.
        clipLimit: CLAHE contrast limit.
        tileGridSize: CLAHE grid size, as ``(columns, rows)``.

    Raises:
        ValueError: If ``src`` is not 8-bit or has an unsupported channel count.
    """
    if src.dtype != np.uint8:
        raise ValueError(f"src must be an 8-bit (uint8) image, got dtype {src.dtype}")
    channels = 1 if src.ndim == 2 else src.shape[2]
    if channels not in (1, 3, 4):
        raise ValueError(f"src must have 1, 3 or 4 channels, got shape {src.shape}")

    def equalize(gray: np.ndarray) -> np.ndarray:
        if clahe:
            out: np.ndarray = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize).apply(gray)
            return out
        result: np.ndarray = cv2.equalizeHist(gray)
        return result

    if channels == 1:
        return equalize(src.reshape(src.shape[:2]))
    lab = cv2.cvtColor(src[..., :3], cv2.COLOR_BGR2LAB)
    lab[..., 0] = equalize(np.ascontiguousarray(lab[..., 0]))
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if channels == 4:
        return np.dstack([bgr, src[..., 3]])
    return bgr


def unsharpMask(src: np.ndarray, sigma: float = 1.0, amount: float = 1.0, threshold: int = 0) -> np.ndarray:
    """Sharpen with an unsharp mask: ``src + amount * (src - GaussianBlur(src, sigma))``.

    The standard sharpening filter of photo editors; OpenCV has no function for it.

    Args:
        src: Image to sharpen (any number of channels).
        sigma: Gaussian blur radius in pixels; larger values sharpen coarser details.
        amount: Strength; ``0`` returns the image unchanged, ``1`` doubles the local contrast.
        threshold: Only sharpen where the difference to the blurred image is at least this large,
            which avoids amplifying noise in flat areas.

    Raises:
        ValueError: If ``sigma`` is not positive, or ``amount`` or ``threshold`` is negative.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if amount < 0:
        raise ValueError(f"amount must not be negative, got {amount}")
    if threshold < 0:
        raise ValueError(f"threshold must not be negative, got {threshold}")
    blurred = cv2.GaussianBlur(src, (0, 0), sigma)
    sharpened: np.ndarray = cv2.addWeighted(src, 1.0 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast = cv2.absdiff(src, blurred) < threshold
        sharpened[low_contrast] = src[low_contrast]
    return sharpened


def pixelate(src: np.ndarray, blockSize: int = 16, rect: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Mosaic the image, or only the box ``rect``, e.g. to anonymize faces or license plates.

    Each ``blockSize`` x ``blockSize`` block is replaced by its average color.

    Args:
        src: Image to pixelate.
        blockSize: Size of the blocks in pixels.
        rect: Box ``(x, y, width, height)`` to pixelate, e.g. from a detector; it may extend outside the image.
            By default the whole image.

    Returns:
        A new image; the area outside ``rect`` is unchanged.

    Raises:
        ValueError: If ``blockSize`` is not positive.
    """
    if blockSize < 1:
        raise ValueError(f"blockSize must be positive, got {blockSize}")
    dst = src.copy()
    height, width = src.shape[:2]
    x, y, w, h = (0, 0, width, height) if rect is None else (int(v) for v in rect)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return dst
    roi = dst[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    small = cv2.resize(roi, (-(-rw // blockSize), -(-rh // blockSize)), interpolation=cv2.INTER_AREA)
    blocks = np.repeat(np.repeat(small, blockSize, axis=0), blockSize, axis=1)[:rh, :rw]
    roi[...] = blocks.reshape(roi.shape)
    return dst
