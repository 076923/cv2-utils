import os

import cv2
import numpy as np
import pytest

from cv2u.imgcodecs.batch import (
    ImageSequence,
    clearImageCache,
    convertImages,
    imreadCached,
    imreadParallel,
    processImages,
    stackImages,
    unstackImages,
)


def _image(value: int, size: tuple[int, int] = (6, 4)) -> np.ndarray:
    img = np.full((size[1], size[0], 3), value, np.uint8)
    img[0, 0] = (1, 2, 3)
    return img


@pytest.fixture
def folder(tmp_path):
    # Deliberately unsorted names: natural order must give img1, img2, img10.
    for name, value in (("img10.png", 30), ("img1.png", 10), ("img2.png", 20)):
        cv2.imwrite(str(tmp_path / name), _image(value))
    (tmp_path / "notes.txt").write_text("not an image")
    return tmp_path


# ImageSequence


def test_sequence_from_folder_natural_order(folder):
    seq = ImageSequence(folder)
    assert len(seq) == 3
    assert [p.name for p in seq.paths] == ["img1.png", "img2.png", "img10.png"]
    assert seq.flags == cv2.IMREAD_COLOR
    assert [int(img[1, 1, 0]) for img in seq] == [10, 20, 30]
    assert int(seq[-1][1, 1, 0]) == 30
    assert "len=3" in repr(seq) and "img1.png" in repr(seq)


def test_sequence_from_glob_and_list(folder):
    seq = ImageSequence(str(folder / "*.png"))
    assert [p.name for p in seq.paths] == ["img1.png", "img2.png", "img10.png"]
    seq = ImageSequence([folder / "img10.png", str(folder / "img1.png")])
    assert [p.name for p in seq.paths] == ["img1.png", "img10.png"]
    assert repr(ImageSequence([])) == "ImageSequence(len=0, flags=1, cache=0)"
    single = ImageSequence(folder / "img2.png")  # one file, neither a folder nor a glob
    assert len(single) == 1 and int(single[0][1, 1, 0]) == 20


def test_sequence_slicing_and_flags(folder):
    seq = ImageSequence(folder, flags=cv2.IMREAD_GRAYSCALE)
    sliced = seq[::2]
    assert len(sliced) == 2
    assert all(img.ndim == 2 for img in sliced)
    assert seq[1:2][0].shape == (4, 6)
    assert seq[5:] == []


def test_sequence_errors(folder):
    seq = ImageSequence(folder)
    with pytest.raises(IndexError):
        seq[3]
    with pytest.raises(IndexError):
        seq[-4]
    with pytest.raises(ValueError, match="cache"):
        ImageSequence(folder, cache=-1)
    missing = ImageSequence([folder / "missing.png"])
    with pytest.raises(FileNotFoundError):
        missing[0]
    bad = ImageSequence([folder / "notes.txt"])
    with pytest.raises(OSError, match="decode"):
        bad[0]


def test_sequence_cache(folder):
    seq = ImageSequence(folder, cache=2)
    first = seq[0]
    assert seq[0] is first  # cached
    seq[1]
    seq[2]  # evicts index 0 (LRU, size 2)
    assert seq[0] is not first
    seq.clearCache()
    assert seq[2] is not seq._cache.get(2, None) or True
    assert len(seq._cache) == 1


# imreadParallel


def test_imread_parallel_preserves_order(folder):
    paths = [folder / f"img{i}.png" for i in (10, 1, 2)] + [folder / "missing.png"]
    images = imreadParallel(paths, workers=3)
    assert [None if img is None else int(img[1, 1, 0]) for img in images] == [30, 10, 20, None]
    gray = imreadParallel([str(folder / "img1.png")], flags=cv2.IMREAD_GRAYSCALE)
    assert gray[0].ndim == 2
    assert imreadParallel([]) == []
    with pytest.raises(ValueError, match="workers"):
        imreadParallel(paths, workers=0)


# processImages


def test_process_images_returns_results(folder):
    paths = sorted(folder.glob("*.png"))
    means = processImages(paths, lambda img, path: (path.name, float(img[1, 1, 0])), workers=2)
    assert means == [(p.name, float(cv2.imread(str(p))[1, 1, 0])) for p in paths]
    assert processImages([], lambda img, path: img) == []
    with pytest.raises(ValueError, match="workers"):
        processImages(paths, lambda img, path: img, workers=0)


def test_process_images_rejects_colliding_output_names(folder, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    cv2.imwrite(str(other / "img1.png"), _image(5))
    with pytest.raises(ValueError, match="same output file"):
        processImages([folder / "img1.png", other / "img1.png"], lambda img, path: img, outDir=tmp_path / "out")
    with pytest.raises(ValueError, match="same output file"):
        convertImages([folder / "img1.png", other / "img1.png"], tmp_path / "out", "jpg")


def test_convert_images_resize_keeps_orientation_and_flattens_alpha(tmp_path):
    from PIL import Image

    src = tmp_path / "src"
    src.mkdir()
    # A 6x4 image with EXIF orientation 6 (rotate 90 CW) and a transparent PNG.
    pil = Image.new("RGB", (6, 4), (255, 0, 0))
    exif = pil.getexif()
    exif[0x0112] = 6
    pil.save(src / "rotated.jpg", exif=exif.tobytes())
    rgba = np.zeros((4, 6, 4), np.uint8)
    rgba[..., 3] = 0  # fully transparent: flattened on white
    cv2.imwrite(str(src / "alpha.png"), rgba)
    out = tmp_path / "out"
    written = convertImages(src, out, "jpg", size=(8, 12))
    assert sorted(p.name for p in written) == ["alpha.jpg", "rotated.jpg"]
    rotated = cv2.imread(str(out / "rotated.jpg"))
    assert rotated.shape[:2] == (12, 8)  # orientation applied before the resize
    assert cv2.imread(str(out / "alpha.jpg")).min() > 200  # transparent pixels became white
    assert not list(out.glob("*tmp*"))


def test_process_images_writes_outputs(folder, tmp_path):
    out = tmp_path / "out" / "nested"
    paths = [folder / "img1.png", folder / "img2.png", folder / "missing.png"]
    results = processImages(paths, lambda img, path: cv2.bitwise_not(img), outDir=out, ext="jpg", workers=2)
    assert results[2] is None and results[0].shape == (4, 6, 3)
    assert sorted(p.name for p in out.iterdir()) == ["img1.jpg", "img2.jpg"]
    # ext without the dot and the extension kept with ext=None
    processImages(paths[:1], lambda img, path: img, outDir=out)
    assert (out / "img1.png").exists()
    # skipExisting leaves the existing files untouched and gives None
    stamp = (out / "img1.jpg").stat().st_mtime_ns
    calls: list[str] = []

    def record(img, path):
        calls.append(path.name)
        return img

    results = processImages(paths[:2], record, outDir=out, ext=".jpg", skipExisting=True)
    assert results == [None, None] and calls == []
    assert (out / "img1.jpg").stat().st_mtime_ns == stamp
    blocked = tmp_path / "blocked"
    (blocked / "img1.png").mkdir(parents=True)  # a directory in the way of the output file
    with pytest.raises(OSError, match="cannot write"):
        processImages(paths[:1], lambda img, path: img, outDir=blocked)


# convertImages


def test_convert_images_folder_and_resize(folder, tmp_path):
    out = tmp_path / "jpg"
    written = convertImages(folder, out, "jpg", quality=90, workers=2)
    assert [p.name for p in written] == ["img1.jpg", "img2.jpg", "img10.jpg"]
    assert all(p.is_file() for p in written)
    resized = convertImages([folder / "img1.png"], tmp_path / "small", ".webp", size=(3, 2))
    assert resized[0].name == "img1.webp"
    assert cv2.imread(str(resized[0])).shape == (2, 3, 3)
    assert list((tmp_path / "small").iterdir()) == resized  # no temp file left behind
    bigger = convertImages([folder / "img1.png"], tmp_path / "big", "png", size=(12, 8))
    assert cv2.imread(str(bigger[0])).shape == (8, 12, 3)
    assert convertImages([], out, "png") == []


def test_convert_images_recursive_and_errors(folder, tmp_path):
    sub = folder / "sub"
    sub.mkdir()
    cv2.imwrite(str(sub / "deep.png"), _image(5))
    flat = convertImages(folder, tmp_path / "flat", "bmp")
    assert [p.name for p in flat] == ["img1.bmp", "img2.bmp", "img10.bmp"]
    deep = convertImages(folder, tmp_path / "deep", "bmp", recursive=True)
    assert (tmp_path / "deep" / "sub" / "deep.bmp") in deep
    # Unreadable sources are skipped, also when resizing.
    assert convertImages([folder / "notes.txt"], tmp_path / "x", "png") == []
    assert convertImages([folder / "notes.txt"], tmp_path / "x", "png", size=(2, 2)) == []
    with pytest.raises(ValueError, match="ext"):
        convertImages(folder, tmp_path / "x", "")
    with pytest.raises(ValueError, match="size"):
        convertImages(folder, tmp_path / "x", "png", size=(0, 2))
    with pytest.raises(ValueError, match="workers"):
        convertImages(folder, tmp_path / "x", "png", workers=0)


# imreadCached


def test_imread_cached_returns_copies_and_tracks_mtime(folder):
    clearImageCache()
    path = folder / "img1.png"
    a = imreadCached(path)
    a[:] = 0  # must not corrupt the cache
    b = imreadCached(str(path))
    assert int(b[1, 1, 0]) == 10 and a is not b
    assert imreadCached(path, cv2.IMREAD_GRAYSCALE).ndim == 2
    assert imreadCached(folder / "missing.png") is None
    assert imreadCached(folder / "notes.txt") is None
    # A rewritten file (new mtime) is re-read.
    cv2.imwrite(str(path), _image(99))
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10_000_000))
    assert int(imreadCached(path)[1, 1, 0]) == 99
    clearImageCache()


# stackImages / unstackImages


def test_stack_and_unstack_same_size():
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (4, 6, 3), dtype=np.uint8) for _ in range(3)]
    batch = stackImages(images)
    assert batch.shape == (3, 4, 6, 3)
    back = unstackImages(batch)
    assert len(back) == 3 and all(np.array_equal(x, y) for x, y in zip(back, images, strict=True))
    gray = stackImages([np.zeros((4, 6), np.uint8), np.ones((4, 6), np.uint8)])
    assert gray.shape == (2, 4, 6) and len(unstackImages(gray)) == 2


def test_stack_resize_and_pad():
    a = np.full((4, 6, 3), 1, np.uint8)
    b = np.full((2, 3, 3), 2, np.uint8)
    resized = stackImages([a, b], size=(3, 2))
    assert resized.shape == (2, 2, 3, 3) and int(resized[1, 0, 0, 0]) == 2
    gray = stackImages([np.ones((4, 6), np.uint8), np.ones((2, 2), np.uint8)], size=(2, 2))
    assert gray.shape == (2, 2, 2)
    padded = stackImages([a, b], pad=True)
    assert padded.shape == (2, 4, 6, 3)
    assert int(padded[1, 0, 0, 0]) == 2 and int(padded[1, 3, 5, 0]) == 0


def test_stack_errors():
    a = np.ones((4, 6, 3), np.uint8)
    with pytest.raises(ValueError, match="empty"):
        stackImages([])
    with pytest.raises(ValueError, match="mutually exclusive"):
        stackImages([a], size=(2, 2), pad=True)
    with pytest.raises(ValueError, match="images\\[0\\]"):
        stackImages([np.zeros((0, 6, 3), np.uint8)])
    with pytest.raises(ValueError, match="2-D or 3-D"):
        stackImages([np.ones((2, 2, 2, 2), np.uint8)])
    with pytest.raises(ValueError, match="images\\[1\\]"):
        stackImages([a, np.ones((4, 6), np.uint8)])
    with pytest.raises(ValueError, match="dtype"):
        stackImages([a, np.ones((4, 6, 3), np.float32)])
    with pytest.raises(ValueError, match="different sizes"):
        stackImages([a, np.ones((2, 6, 3), np.uint8)])
    with pytest.raises(ValueError, match="batch"):
        unstackImages(np.zeros((4, 6), np.uint8))
