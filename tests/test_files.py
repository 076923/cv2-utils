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


# --- imreadAll / imwriteAll -------------------------------------------------------------------


def test_imwriteAll_imreadAll_roundtrip(tmp_path, bgr):
    images = [bgr, bgr[::-1], bgr[:, ::-1]]
    paths = [tmp_path / "사진" / f"img{i}.png" for i in range(3)]
    assert cv.imwriteAll(paths, images, makeDirs=True) == [True, True, True]
    back = cv.imreadAll(paths)
    assert len(back) == 3
    for a, b in zip(back, images, strict=True):
        np.testing.assert_array_equal(a, b)
    back_gray = cv.imreadAll(cv.listImages(tmp_path / "사진"), cv.IMREAD_GRAYSCALE)
    assert all(g.ndim == 2 for g in back_gray)


def test_imreadAll_strict_names_the_file(tmp_path, bgr):
    good = tmp_path / "a.png"
    assert cv.imwriteEx(good, bgr)
    missing = tmp_path / "없음.png"
    with pytest.raises(OSError, match=r"없음.png"):
        cv.imreadAll([good, missing])
    lenient = cv.imreadAll([good, missing], strict=False)
    assert lenient[1] is None
    np.testing.assert_array_equal(lenient[0], bgr)


def test_imwriteAll_validation_and_lenient_mode(tmp_path, bgr):
    with pytest.raises(ValueError, match="same length"):
        cv.imwriteAll([tmp_path / "a.png"], [bgr, bgr])
    bad = tmp_path / "없는 폴더" / "a.png"
    with pytest.raises(OSError, match=r"a.png"):
        cv.imwriteAll([bad], [bgr])
    assert cv.imwriteAll([bad, tmp_path / "b.png"], [bgr, bgr], strict=False) == [False, True]
    assert cv.imwriteAll([], []) == []
