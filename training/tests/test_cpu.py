"""Small CPU-only checks for CI: no trained weights, no test pages, no GPU.

The full check on real pages with the trained model is tests/test_lightness.py (local, CUDA).
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.augment import white_paper  # noqa: E402
from models.hints import hex_to_rgb01, sample_hints  # noqa: E402
from models.lab import gray_to_L, lab_to_rgb_torch, merge_lab, rgb_to_lab_torch  # noqa: E402
from models.unet import ColorUNet  # noqa: E402


def _page(h=96, w=64, seed=0):
    """A synthetic page: paper white with dark strokes and a gray tone block."""
    rng = np.random.default_rng(seed)
    page = np.full((h, w), 250, np.uint8)
    page[20:70, 10:40] = 150
    for _ in range(12):
        y, x = rng.integers(0, h), rng.integers(0, w - 20)
        page[y, x : x + 20] = 10
    return page


def test_lab_round_trip():
    rgb = torch.rand(1, 3, 16, 16)
    back = lab_to_rgb_torch(rgb_to_lab_torch(rgb))
    assert torch.allclose(back, rgb, atol=2e-3)


def test_merge_keeps_the_pages_lightness_on_ink():
    page = _page()
    rng = np.random.default_rng(1)
    color_low = rng.random((48, 32, 3), dtype=np.float32)
    out = merge_lab(page, color_low, device="cpu")
    assert out.shape == (*page.shape, 3) and out.dtype == np.uint8

    out_L = rgb_to_lab_torch(torch.from_numpy(out).permute(2, 0, 1)[None].float() / 255)[0, 0].numpy()
    L = gray_to_L(page)
    ink = L < 85
    assert ink.any()
    assert np.abs(out_L[ink] - L[ink]).max() < 1.5


def test_merge_without_adjustments_keeps_the_whole_pages_lightness():
    page = _page(seed=2)
    color_low = np.full((48, 32, 3), 0.5, np.float32)
    out = merge_lab(page, color_low, chroma_falloff=False, device="cpu")
    out_L = rgb_to_lab_torch(torch.from_numpy(out).permute(2, 0, 1)[None].float() / 255)[0, 0].numpy()
    assert np.abs(out_L - gray_to_L(page)).max() < 1.5


def test_hints_are_points_with_a_mask():
    torch.manual_seed(0)
    hint = sample_hints(torch.rand(3, 64, 64), max_points=5)
    assert hint.shape == (4, 64, 64)
    assert hint[3].sum() > 0
    assert torch.allclose(hex_to_rgb01("#ff8000"), torch.tensor([1.0, 128 / 255, 0.0]))


def test_white_paper_lifts_light_grays_only():
    gray = np.linspace(0, 1, 101, dtype=np.float32)
    lifted = white_paper(gray, np.random.default_rng(0))
    assert np.all(lifted >= gray - 1e-6)
    assert np.allclose(lifted[:60], gray[:60])
    assert lifted[-5:].min() > 0.99


def test_unet_shapes_on_cpu():
    net = ColorUNet(pretrained=False).eval()
    x = torch.zeros(1, 5, 64, 64)
    with torch.no_grad():
        rgb, _ = net(x)
    assert rgb.shape == (1, 3, 64, 64)
    assert rgb.min() >= -1 and rgb.max() <= 1
