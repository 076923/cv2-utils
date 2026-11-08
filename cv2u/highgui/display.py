import cv2
import numpy as np

from .._validate import check_image_not_empty

__all__ = ["imshowEx", "imshowInline"]


def _fit(img: np.ndarray, maxSize: tuple[int, int]) -> np.ndarray:
    max_w, max_h = maxSize
    if max_w <= 0 or max_h <= 0:
        raise ValueError(f"maxSize must be positive, got {maxSize}")
    h, w = img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale == 1.0:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized: np.ndarray = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return resized


def imshowEx(winname: str, img: np.ndarray, maxSize: tuple[int, int] = (1280, 720)) -> None:
    """``cv.imshow`` that shrinks images larger than ``maxSize`` so the whole image fits on the screen.

    ``cv.imshow`` shows images at 100%, so a 4000x3000 photo is cropped by the screen. The aspect ratio is kept, and
    images that already fit are shown unchanged.

    Args:
        winname: Window name.
        img: Image to show.
        maxSize: Largest displayed size, as ``(width, height)``.

    Raises:
        ValueError: If ``img`` is empty or ``maxSize`` is not positive.
    """
    check_image_not_empty("img", img)
    cv2.imshow(winname, _fit(img, maxSize))


def imshowInline(img: np.ndarray, maxSize: tuple[int, int] | None = None, ext: str = ".png") -> None:
    """Show an image inline in a Jupyter notebook (``cv.imshow`` does not work there).

    The image is encoded with ``cv.imencode``, so BGR images show their real colors without a BGR-to-RGB conversion,
    unlike ``matplotlib.pyplot.imshow``.

    Args:
        img: 8-bit grayscale, BGR or BGRA image.
        maxSize: If given, shrink larger images to fit into ``(width, height)``.
        ext: Encoding used for the notebook output, ``".png"`` (lossless) or ``".jpg"`` (smaller).

    Raises:
        ImportError: If IPython is not installed (Jupyter always provides it).
        ValueError: If ``img`` is empty or cannot be encoded.
    """
    check_image_not_empty("img", img)
    try:
        from IPython.display import Image, display
    except ImportError as e:
        raise ImportError("imshowInline requires IPython, which Jupyter provides (pip install ipython)") from e

    shown = img if maxSize is None else _fit(img, maxSize)
    try:
        ok, buffer = cv2.imencode(ext, shown)
    except cv2.error as e:  # unknown extension
        raise ValueError(f"cannot encode the image as {ext!r}") from e
    if not ok:
        raise ValueError(f"cannot encode the image as {ext!r}")
    display(Image(data=buffer.tobytes()))
