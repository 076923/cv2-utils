"""Regenerate the golden images used by ``tests/test_golden.py``.

Run this only when an output change is intended, and review the new images before committing::

    python tools/regenerate_goldens.py
"""

from pathlib import Path

import cv2

import cv2u as cv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"

# Small lossless inputs derived from the README sample images (keeps the tests fast and avoids JPEG decoder drift).
# tests/data/dbscan_src.png (a sneaker photo, 80x80) is a committed input that is not regenerated.
INPUTS = {
    "rotate_src.png": ("images/examples/rotateEx-src.jpg", (160, 160)),
}


def _write(name: str, img: cv2.typing.MatLike) -> None:
    path = DATA / name
    if not cv2.imwrite(str(path), img):
        raise SystemExit(f"failed to write {path}")
    print(f"wrote {path.relative_to(ROOT)} {img.shape}")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, (source, size) in INPUTS.items():
        src = cv2.imread(str(ROOT / source))
        if src is None:
            raise SystemExit(f"cannot read {source}")
        _write(name, cv2.resize(src, size, interpolation=cv2.INTER_AREA))

    rot = cv2.imread(str(DATA / "rotate_src.png"))
    gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
    _write("rotateEx_45.png", cv.rotateEx(rot, 45))
    _write("rotateEx_30_scale0.5.png", cv.rotateEx(rot, 30, scale=0.5))
    _write("gammaCorrection_2.2.png", cv.gammaCorrection(rot, 2.2))
    _write("boldEdges.png", cv.boldEdges(gray))

    dbs = cv2.imread(str(DATA / "dbscan_src.png"))
    maps, _ = cv.dbscanColors(dbs, 1.7, 2)
    _write("drawClusters.png", cv.drawClusters(dbs, maps))


if __name__ == "__main__":
    main()
