import numpy as np
import pytest

import cv2u as cv


@pytest.fixture
def bgr():
    return np.random.default_rng(1).integers(0, 256, (20, 30, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", ["image.png", "이미지.png", "画像 ü é.png"])
def test_imwriteEx_imreadEx_roundtrip_unicode_paths(tmp_path, bgr, name):
    folder = tmp_path / "한글 폴더"
    folder.mkdir()
    path = folder / name
    assert cv.imwriteEx(path, bgr)  # pathlib.Path is accepted
    np.testing.assert_array_equal(cv.imreadEx(str(path)), bgr)


def test_imreadEx_flags(tmp_path, bgr):
    path = tmp_path / "사진.png"
    assert cv.imwriteEx(path, bgr)
    gray = cv.imreadEx(path, cv.IMREAD_GRAYSCALE)
    assert gray is not None
    assert gray.shape == bgr.shape[:2]


def test_imreadEx_matches_imread_on_ascii_paths(tmp_path, bgr):
    path = tmp_path / "plain.png"
    assert cv.imwrite(str(path), bgr)
    np.testing.assert_array_equal(cv.imreadEx(path), cv.imread(str(path)))


@pytest.mark.parametrize("content", [None, b"", b"not an image"])
def test_imreadEx_returns_none_like_imread(tmp_path, content):
    path = tmp_path / "없음.png"
    if content is not None:
        path.write_bytes(content)
    assert cv.imreadEx(path) is None


def test_imwriteEx_params(tmp_path, bgr):
    low, high = tmp_path / "low.jpg", tmp_path / "high.jpg"
    assert cv.imwriteEx(low, bgr, (cv.IMWRITE_JPEG_QUALITY, 10))
    assert cv.imwriteEx(high, bgr, (cv.IMWRITE_JPEG_QUALITY, 100))
    assert low.stat().st_size < high.stat().st_size


def test_imwriteEx_missing_directory_returns_false(tmp_path, bgr):
    assert cv.imwriteEx(tmp_path / "없는 폴더" / "a.png", bgr) is False


def test_imwriteEx_requires_extension(tmp_path, bgr):
    with pytest.raises(ValueError, match="extension"):
        cv.imwriteEx(tmp_path / "noext", bgr)


def test_imwriteEx_unknown_extension_raises_like_imwrite(tmp_path, bgr):
    import cv2

    with pytest.raises(cv2.error):
        cv.imwrite(str(tmp_path / "a.xyz"), bgr)
    with pytest.raises(cv2.error):
        cv.imwriteEx(tmp_path / "a.xyz", bgr)
