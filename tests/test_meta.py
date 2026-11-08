import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from cv2u.imgcodecs.files import imreadEx
from cv2u.imgcodecs.meta import (
    ImageInfo,
    convertImageFormat,
    exifOrientation,
    imageInfo,
    imdecodeEx,
    imencodeEx,
    imreadOriented,
    imreadResized,
)


def _image() -> np.ndarray:
    img = np.zeros((20, 40, 3), np.uint8)
    img[:, :20] = (0, 0, 255)  # left half red
    img[:5, :] = (255, 0, 0)  # top stripe blue
    return img


def _save(path, img, orientation=None, **kwargs):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif.tobytes()
    pil.save(path, **kwargs)
    return path


# imageInfo / exifOrientation


def test_image_info_png(tmp_path):
    path = _save(tmp_path / "사진.png", _image())
    info = imageInfo(path)
    assert isinstance(info, ImageInfo)
    assert (info.width, info.height, info.channels, info.mode, info.format) == (40, 20, 3, "RGB", "PNG")
    assert info.fileSize == path.stat().st_size > 0
    assert info.exifOrientation == 1


def test_image_info_gray_and_alpha(tmp_path):
    gray = tmp_path / "g.png"
    Image.fromarray(np.zeros((8, 6), np.uint8)).save(gray)
    assert imageInfo(str(gray))[:4] == (6, 8, 1, "L")
    rgba = tmp_path / "a.png"
    Image.fromarray(np.zeros((8, 6, 4), np.uint8), mode="RGBA").save(rgba)
    assert imageInfo(rgba).channels == 4


def test_image_info_jpeg_orientation(tmp_path):
    path = _save(tmp_path / "o.jpg", _image(), orientation=6)
    info = imageInfo(path)
    assert (info.width, info.height, info.format, info.exifOrientation) == (40, 20, "JPEG", 6)
    assert exifOrientation(path) == 6
    assert exifOrientation(_save(tmp_path / "p.png", _image())) == 1


def test_image_info_errors(tmp_path):
    with pytest.raises(OSError):
        imageInfo(tmp_path / "missing.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(OSError):
        imageInfo(bad)
    with pytest.raises(OSError):
        exifOrientation(bad)


# imreadOriented


@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("ext", [".jpg", ".png", ".webp"])
def test_imread_oriented_matches_pillow(tmp_path, orientation, ext):
    img = _image()
    path = _save(tmp_path / f"o{ext}", img, orientation=orientation, quality=100)
    expected = ImageOps.exif_transpose(Image.open(path))
    expected_bgr = cv2.cvtColor(np.array(expected.convert("RGB")), cv2.COLOR_RGB2BGR)
    out = imreadOriented(path)
    assert out is not None
    assert out.shape == expected_bgr.shape
    assert np.abs(out.astype(int) - expected_bgr.astype(int)).max() <= 12  # JPEG loss


def test_imread_oriented_unchanged_and_gray(tmp_path):
    """OpenCV skips the orientation for IMREAD_UNCHANGED and REDUCED flags; imreadOriented applies it anyway."""
    path = _save(tmp_path / "o.png", _image(), orientation=6)
    assert cv2.imread(str(path), cv2.IMREAD_UNCHANGED).shape == (20, 40, 3)  # OpenCV: as stored
    assert cv2.imread(str(path)).shape == (40, 20, 3)  # OpenCV: rotated
    out = imreadOriented(path, cv2.IMREAD_UNCHANGED)
    assert out is not None and out.shape == (40, 20, 3)
    gray = imreadOriented(path, cv2.IMREAD_GRAYSCALE)
    assert gray is not None and gray.shape == (40, 20)


def test_imread_oriented_missing(tmp_path):
    assert imreadOriented(tmp_path / "missing.jpg") is None
    raw = tmp_path / "raw.pfm"
    assert cv2.imwrite(str(raw), np.ones((4, 6), np.float32))
    out = imreadOriented(raw, cv2.IMREAD_UNCHANGED)  # Pillow cannot open .pfm: orientation defaults to 1
    assert out is not None and out.shape == (4, 6)


# imreadResized


def test_imread_resized_modes(tmp_path):
    path = _save(tmp_path / "r.jpg", _image(), quality=100)
    assert imreadResized(path, size=(10, 10)).shape == (10, 10, 3)
    assert imreadResized(path, scale=0.5).shape == (10, 20, 3)
    assert imreadResized(path, scale=0.25, flags=cv2.IMREAD_GRAYSCALE).shape == (5, 10)
    assert imreadResized(path, scale=0.125).shape == (3, 5, 3)
    assert imreadResized(path, scale=0.3).shape == (6, 12, 3)
    assert imreadResized(path, width=20).shape == (10, 20, 3)
    assert imreadResized(path, height=40, interpolation=cv2.INTER_NEAREST).shape == (40, 80, 3)
    assert imreadResized(path, scale=0.5, flags=cv2.IMREAD_UNCHANGED).shape == (10, 20, 3)


def test_imread_resized_png_reduced_and_content(tmp_path):
    path = _save(tmp_path / "r.png", _image())
    half = imreadResized(path, scale=0.5)
    assert half.shape == (10, 20, 3)
    assert tuple(half[8, 2]) == (0, 0, 255) and tuple(half[8, 18]) == (0, 0, 0)


def test_imread_resized_orientation(tmp_path):
    path = _save(tmp_path / "o.jpg", _image(), orientation=6, quality=100)
    assert imreadResized(path, scale=0.5).shape == (20, 10, 3)
    assert imreadResized(path, size=(10, 30)).shape == (30, 10, 3)
    assert imreadResized(path, width=10).shape == (20, 10, 3)
    assert imreadResized(path, height=10).shape == (10, 5, 3)


def test_imread_resized_errors(tmp_path):
    path = _save(tmp_path / "r.png", _image())
    assert imreadResized(tmp_path / "missing.png", scale=0.5) is None
    with pytest.raises(ValueError, match="exactly one"):
        imreadResized(path)
    with pytest.raises(ValueError, match="exactly one"):
        imreadResized(path, scale=0.5, width=3)
    with pytest.raises(ValueError, match="size"):
        imreadResized(path, size=(0, 5))
    with pytest.raises(ValueError, match="scale"):
        imreadResized(path, scale=float("nan"))
    with pytest.raises(ValueError, match="scale"):
        imreadResized(path, scale=-1)


# imencodeEx / imdecodeEx


def test_imencode_imdecode_roundtrip():
    img = _image()
    data = imencodeEx(".png", img)
    assert isinstance(data, bytes) and data[:4] == b"\x89PNG"
    assert np.array_equal(imdecodeEx(data), img)
    assert np.array_equal(imdecodeEx(bytearray(data)), img)
    assert np.array_equal(imdecodeEx(memoryview(data)), img)
    assert np.array_equal(imdecodeEx(np.frombuffer(data, np.uint8)), img)
    assert imdecodeEx(data, cv2.IMREAD_GRAYSCALE).shape == (20, 40)
    small = imencodeEx("jpg", img, (cv2.IMWRITE_JPEG_QUALITY, 10))
    assert len(small) < len(imencodeEx("jpg", img, (cv2.IMWRITE_JPEG_QUALITY, 100)))


def test_imencode_errors():
    with pytest.raises(ValueError, match="encode"):
        imencodeEx(".xyz", _image())
    with pytest.raises(ValueError, match="empty"):
        imencodeEx(".png", np.zeros((0, 0, 3), np.uint8))
    with pytest.raises(ValueError, match="encode"):
        imencodeEx(".png", np.zeros((2, 2, 5), np.uint8))


def test_imdecode_bad_data():
    assert imdecodeEx(b"") is None
    assert imdecodeEx(b"not an image") is None


# convertImageFormat


def test_convert_image_format(tmp_path):
    src = _save(tmp_path / "원본.png", _image())
    dst = tmp_path / "out" / "변환.jpg"
    assert convertImageFormat(src, dst, quality=95, makeDirs=True)
    out = imreadEx(dst)
    assert out is not None and out.shape == (20, 40, 3)
    assert np.abs(out[8:16, 4:16].astype(int) - (0, 0, 255)).max() <= 6  # JPEG edges blur, so compare inside
    assert not convertImageFormat(src, tmp_path / "nodir" / "x.png")


def test_convert_flattens_alpha_for_jpeg(tmp_path):
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[..., :3] = (0, 0, 255)  # red, but fully transparent
    src = tmp_path / "a.png"
    assert cv2.imwrite(str(src), rgba)
    assert convertImageFormat(src, tmp_path / "a.jpg", background=(0, 255, 0))
    out = cv2.imread(str(tmp_path / "a.jpg"))
    assert np.abs(out[5, 5].astype(int) - np.array([0, 255, 0])).max() <= 4
    assert convertImageFormat(src, tmp_path / "a.webp")  # alpha kept
    assert cv2.imread(str(tmp_path / "a.webp"), cv2.IMREAD_UNCHANGED).shape[2] == 4


def test_convert_applies_orientation(tmp_path):
    src = _save(tmp_path / "o.jpg", _image(), orientation=6, quality=100)
    assert convertImageFormat(src, tmp_path / "o.png")
    assert cv2.imread(str(tmp_path / "o.png")).shape == (40, 20, 3)


def test_convert_errors(tmp_path):
    assert not convertImageFormat(tmp_path / "missing.png", tmp_path / "x.png")
    with pytest.raises(ValueError, match="extension"):
        convertImageFormat(_save(tmp_path / "s.png", _image()), tmp_path / "noext")
