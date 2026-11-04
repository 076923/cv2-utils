import sys
from functools import cache
from pathlib import Path

import pytest
from PIL import ImageFont, features

# Common scalable fonts, preferred over an arbitrary file found on the system: some fonts (e.g. color emoji) only
# support fixed sizes and cannot be used for ordinary text.
_PREFERRED = [
    "C:/Windows/Fonts/malgun.ttf",  # Korean glyphs
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _usable(path: Path) -> bool:
    try:
        ImageFont.truetype(str(path), 16).getbbox("Ag")
    except OSError:
        return False
    return True


@cache
def _find_font() -> Path | None:
    candidates = [Path(p) for p in _PREFERRED]
    system = Path("/usr/share/fonts")
    if system.is_dir():
        candidates += sorted(system.rglob("*.ttf"))  # sorted: the same font on every run
    return next((p for p in candidates if p.is_file() and _usable(p)), None)


@pytest.fixture
def font_path() -> Path:
    """A scalable TrueType font available on this machine; skips the test if there is none."""
    if not features.check("freetype2"):
        pytest.skip("Pillow was built without FreeType support")
    font = _find_font()
    if font is None:
        pytest.skip(f"no usable TrueType font found on {sys.platform}")
    return font
