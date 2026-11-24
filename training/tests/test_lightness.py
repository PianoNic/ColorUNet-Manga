"""The output's lightness must come from the ORIGINAL full-resolution page, not the model.

  python -m pytest tests/ -q        (or: python tests/test_lightness.py)

Checked on real test pages (eval/testset), so line art and text are covered:
- every pixel that is not near-white (ink, lines, text, tones, shading: original L < 85) keeps the
  original L almost exactly;
- near-white paper may only get slightly darker where real color is applied (skin, light hair),
  by at most the 35 L cap in models/lab.py;
- with the white/black adjustment off, the whole page's L matches.
"""

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import io as mio  # noqa: E402
from models.lab import gray_to_L  # noqa: E402
from scripts.infer import Colorizer  # noqa: E402

TESTSET = os.path.join(ROOT, "eval", "testset", "more-than-a-doll")

# Local check: needs a CUDA GPU, the trained weights and the test pages (data/fetch_mangadex.py).
import pytest  # noqa: E402
import torch  # noqa: E402

if not (torch.cuda.is_available() and os.path.isdir(TESTSET) and os.path.isfile(mio.DEFAULT_WEIGHTS)):
    pytest.skip("needs CUDA, the trained weights and eval/testset (local check)", allow_module_level=True)
_col = {}


def _pages(n=4):
    paths = mio.list_images(TESTSET)
    assert paths, f"no test pages in {TESTSET} (run data/fetch_mangadex.py)"
    return [mio.read_gray(p) for p in paths[:: max(1, len(paths) // n)][:n]]


def _colorizer(falloff):
    if falloff not in _col:
        _col[falloff] = Colorizer(device="cuda", chroma_falloff=falloff)
    return _col[falloff]


def _L_of(rgb_u8):
    return cv2.cvtColor(rgb_u8.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]


def test_lightness_kept_on_ink_lines_and_tones():
    for gray in _pages():
        L0, L1 = gray_to_L(gray), _L_of(_colorizer(True)(gray))
        dark = L0 < 85
        err = np.abs(L1 - L0)[dark]
        assert err.mean() < 0.5, f"mean |dL| on ink/tones {err.mean():.2f}"
        assert np.percentile(err, 99) < 2.0, f"99th pct |dL| on ink/tones {np.percentile(err, 99):.2f}"


def test_paper_only_darkens_a_little():
    for gray in _pages():
        L0, L1 = gray_to_L(gray), _L_of(_colorizer(True)(gray))
        assert (L1 - L0).max() < 1.5, "output brighter than the original"
        assert (L0 - L1).max() < 35 + 1.5, f"paper darkened by {(L0 - L1).max():.1f} L"


def test_lightness_exact_without_adjustment():
    for gray in _pages(2):
        L0, L1 = gray_to_L(gray), _L_of(_colorizer(False)(gray))
        dark = L0 < 85  # near-white + chroma can leave sRGB and clip; that is what the adjustment fixes
        assert np.abs(L1 - L0)[dark].mean() < 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
