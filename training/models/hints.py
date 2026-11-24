"""Sparse color hints for manga-colorization-v2.

The generator takes 4 hint channels next to the gray input: RGB * mask in [-1, 1], and the mask.
Training samples random points from the target (sample_hints); inference reads a per-series palette
file (load_palette / palette_hints). Both draw the same small square per point, so what the model
sees at inference matches training.

Palette file (JSON), coordinates as fractions of page width/height:
{
  "series": "Sono Bisque Doll wa Koi o Suru",
  "all": [ {"x": 0.5, "y": 0.02, "color": "#ffffff", "label": "optional, applied to every page"} ],
  "pages": {
    "ch0115.5_p012": [ {"x": 0.12, "y": 0.80, "color": "#f3d27a", "label": "Marin hair"} ]
  }
}
Page keys are file names without extension.
"""

import json
import os

import torch

PATCH = 3  # hint square size in model-input pixels (training and inference)


def _draw(hint: torch.Tensor, y: int, x: int, rgb01: torch.Tensor, patch: int = PATCH):
    H, W = hint.shape[1:]
    y0, x0 = min(max(0, y - patch // 2), H - patch), min(max(0, x - patch // 2), W - patch)
    hint[:3, y0 : y0 + patch, x0 : x0 + patch] = (rgb01 * 2 - 1).view(3, 1, 1)
    hint[3, y0 : y0 + patch, x0 : x0 + patch] = 1


def sample_hints(color: torch.Tensor, max_points: int = 16, min_points: int = 1) -> torch.Tensor:
    """color (3,H,W) in [0,1] -> hint (4,H,W) with 1..max_points random points taken from color."""
    _, H, W = color.shape
    hint = torch.zeros(4, H, W)
    for _ in range(torch.randint(min_points, max_points + 1, ()).item()):
        y, x = torch.randint(0, H - PATCH + 1, ()).item(), torch.randint(0, W - PATCH + 1, ()).item()
        rgb = color[:, y : y + PATCH, x : x + PATCH].mean((1, 2))
        _draw(hint, y + PATCH // 2, x + PATCH // 2, rgb)
    return hint


def hex_to_rgb01(c: str) -> torch.Tensor:
    c = c.lstrip("#")
    return torch.tensor([int(c[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def load_palette(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        pal = json.load(f)
    for p in pal.get("all", []) + [p for pts in pal.get("pages", {}).values() for p in pts]:
        if not (0 <= p["x"] <= 1 and 0 <= p["y"] <= 1):
            raise ValueError(f"palette point outside page (x, y must be 0-1): {p}")
        hex_to_rgb01(p["color"])
    return pal


def points_for(pal: dict, page_path: str) -> list:
    if pal is None:
        return []
    name = os.path.splitext(os.path.basename(page_path))[0]
    return pal.get("all", []) + pal.get("pages", {}).get(name, [])


def palette_hints(points: list, padded_hw, content_hw) -> torch.Tensor:
    """Points (fractions of the page) -> hint (1,4,H,W) on the padded model-input grid."""
    hint = torch.zeros(4, *padded_hw)
    h, w = content_hw
    for p in points:
        _draw(hint, int(round(p["y"] * (h - 1))), int(round(p["x"] * (w - 1))), hex_to_rgb01(p["color"]))
    return hint[None]
