"""Lab color helpers.

The core trick: the model only decides color (Lab a/b). Lightness (L) always comes
from the original full-resolution grayscale page, so line art and text stay exact.

The merge runs in PyTorch (GPU when available, multithreaded CPU otherwise): a 2000x2800 page
takes a few tens of ms on the GPU versus ~1 s for the same math in NumPy.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_M_RGB2XYZ = torch.tensor(
    [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]
)
_M_XYZ2RGB = torch.tensor(
    [[3.240479, -1.537150, -0.498535], [-0.969256, 1.875992, 0.041556], [0.055648, -0.204043, 1.057311]]
)
_WHITE = torch.tensor([0.950456, 1.0, 1.088754])

# L* of each 8-bit gray level (gray treated as sRGB gray)
_L_LUT = cv2.cvtColor(np.repeat(np.arange(256, dtype=np.float32)[:, None, None] / 255, 3, 2), cv2.COLOR_RGB2LAB)[
    :, 0, 0
].copy()


CHROMA_BOOST = 1.4  # model colors are a bit timid: 1.4x chroma (gamut fit keeps them valid)
CAST_REMOVAL = 0.7  # share of the page-wide color cast (e.g. yellow rooms) pulled back to neutral


def remove_cast(L_low, ab_low, k=CAST_REMOVAL):
    """Pull the page's dominant color cast (mean a/b of weak-to-moderate colored mid/light areas: walls,
    floors, paper) toward neutral, only on pixels whose color is close to it; distinct colors (skin,
    blush, hair, sky) stay. L_low (N,1,h,w), ab_low (N,2,h,w)."""
    c = torch.sqrt((ab_low * ab_low).sum(1, keepdim=True) + 1e-6)
    w = (c > 3).float() * (c < 30).float() * (L_low > 40).float()
    cast = (ab_low * w).sum((2, 3), keepdim=True) / w.sum((2, 3), keepdim=True).clamp_min(1.0)
    d = torch.sqrt(((ab_low - cast) ** 2).sum(1, keepdim=True) + 1e-6)
    near = 1 - _smoothstep(6.0, 18.0, d)
    return ab_low - k * near * cast


def gray_to_L(gray_u8: np.ndarray) -> np.ndarray:
    """uint8 HxW grayscale -> float32 HxW L* in [0, 100]."""
    return _L_LUT[gray_u8]


# ---------------------------------------------------------------- color conversions (torch)


def rgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable sRGB (N,3,H,W in [0,1]) -> Lab (N,3,H,W), same convention as OpenCV float."""
    rgb = rgb.float().clamp(0, 1)
    lin = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).clamp_min(1e-8) ** 2.4, rgb / 12.92)
    xyz = torch.einsum("ij,njhw->nihw", _M_RGB2XYZ.to(rgb.device), lin)
    xyz = xyz / _WHITE.to(rgb.device).view(1, 3, 1, 1)
    d = 6 / 29
    f = torch.where(xyz > d**3, xyz.clamp_min(1e-8) ** (1 / 3), xyz / (3 * d * d) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], 1)


def lab_to_rgb_torch(lab: torch.Tensor) -> torch.Tensor:
    """Lab (N,3,H,W) -> sRGB (N,3,H,W), NOT clipped (out-of-gamut values stay visible)."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16) / 116
    f = torch.stack([fy + a / 500, fy, fy - b / 200], 1)
    d = 6 / 29
    xyz = torch.where(f > d, f**3, 3 * d * d * (f - 4 / 29)) * _WHITE.to(lab.device).view(1, 3, 1, 1)
    lin = torch.einsum("ij,njhw->nihw", _M_XYZ2RGB.to(lab.device), xyz)
    return torch.where(lin > 0.0031308, 1.055 * lin.abs() ** (1 / 2.4) - 0.055, 12.92 * lin)


# ---------------------------------------------------------------- merge steps (torch, N=1)


def _box(x, r):
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), 2 * r + 1, stride=1)


def guided_upsample(L_full: torch.Tensor, ab_low: torch.Tensor, radius: int = 2, eps: float = 1e-3):
    """Edge-aware upsampling (fast guided filter, He & Sun 2015). L_full (1,1,H,W), ab_low (1,2,h,w).

    Fits ab ~ A * L + B in small windows at the model's resolution, upsamples A and B smoothly and
    applies them to the full-resolution L. Color edges therefore follow the original line art
    instead of the blurry low-res boundary, which removes halos and color past outlines.
    """
    H, W = L_full.shape[2:]
    g_full = L_full / 100
    g_low = F.interpolate(g_full, size=ab_low.shape[2:], mode="area")
    mean_g = _box(g_low, radius)
    var_g = _box(g_low * g_low, radius) - mean_g**2
    mean_p = _box(ab_low, radius)
    A = (_box(g_low * ab_low, radius) - mean_g * mean_p) / (var_g + eps)
    B = mean_p - A * mean_g
    up = lambda x: F.interpolate(_box(x, radius), size=(H, W), mode="bilinear", align_corners=False)
    return up(A) * g_full + up(B)


def _smoothstep(e0, e1, x):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def white_black_adjust(L, ab, L_model, white=(86.0, 97.0), black=(6.0, 22.0), tint=(1.5, 6.0), max_drop=35.0):
    """Clean up colors on paper-white and ink, with smooth falloffs (no hard edges).

    - Ink (L near 0): chroma fades to 0, black stays black.
    - Near-white (paper, bubbles, highlights): weak tints (|ab| < ~tint) fade to 0, so whites stay
      white. Strong colors are kept, because in manga skin and light hair are bare paper too; there
      L is lowered toward the model's own lightness (by at most max_drop) so the color can show.
    Mid tones and everything darker keep the original L exactly.
    """
    c = ab.norm(dim=1, keepdim=True)
    w_white = _smoothstep(white[0], white[1], L)
    keep = _smoothstep(tint[0], tint[1], c)  # 0 = weak tint, 1 = real color
    ab = ab * (1 - w_white * (1 - keep)) * _smoothstep(black[0], black[1], L)
    L = L - w_white * keep * (L - L_model).clamp(0, max_drop)
    return L, ab


def fit_gamut(lab: torch.Tensor, steps: int = 6) -> torch.Tensor:
    """Lab -> RGB, shrinking chroma (never L) where the color falls outside sRGB.

    Plain clipping would change lightness (pale yellow casts on paper, darker reds); this keeps L.
    """

    def ok(scale):
        t = torch.cat([lab[:, :1], lab[:, 1:] * scale], 1)
        r = lab_to_rgb_torch(t)
        return ((r >= -0.003) & (r <= 1.003)).all(1, keepdim=True)

    hi = torch.ones_like(lab[:, :1])
    lo = ok(hi).float()  # in-gamut pixels: scale 1, done (lo == hi)
    for _ in range(steps):  # bisection on the chroma scale for the rest
        mid = (lo + hi) / 2
        good = ok(mid)
        lo = torch.where(good, mid, lo)
        hi = torch.where(good, hi, mid)
    return lab_to_rgb_torch(torch.cat([lab[:, :1], lab[:, 1:] * lo], 1)).clamp(0, 1)


def merge_lab(
    gray_u8: np.ndarray,
    rgb_low: np.ndarray,
    upsample: str = "guided",
    chroma_falloff: bool = True,
    L_source: np.ndarray = None,
    device: str = None,
) -> np.ndarray:
    """Combine the full-res grayscale page with low-res predicted color.

    gray_u8: HxW uint8 original page; its L is the output's lightness (or L_source if given, e.g. a
    descreened page). rgb_low: hxwx3 float32 in [0, 1], same aspect. Returns HxWx3 uint8 RGB.
    upsample: 'guided' (edge-aware, default) or 'bilinear'.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    H, W = gray_u8.shape
    src = L_source if L_source is not None else gray_u8
    with torch.no_grad():
        L = torch.from_numpy(gray_to_L(src)).to(device)[None, None]
        lab_low = rgb_to_lab_torch(torch.from_numpy(np.ascontiguousarray(rgb_low)).to(device).permute(2, 0, 1)[None])
        ab_low = remove_cast(lab_low[:, :1], lab_low[:, 1:]) if chroma_falloff else lab_low[:, 1:]
        if upsample == "guided":
            ab = guided_upsample(L, ab_low) * CHROMA_BOOST
        else:
            ab = F.interpolate(ab_low, size=(H, W), mode="bilinear", align_corners=False)
        if chroma_falloff:
            # model lightness without its (blurry) line art: a max filter removes thin dark lines,
            # so lowering L on colored paper never draws soft halos next to the original lines
            L_model = F.max_pool2d(lab_low[:, :1], 3, stride=1, padding=1)
            L_model = F.interpolate(L_model, size=(H, W), mode="bilinear", align_corners=False)
            L, ab = white_black_adjust(L, ab, L_model)
        rgb = fit_gamut(torch.cat([L, ab], 1))
        return (rgb[0].permute(1, 2, 0) * 255 + 0.5).to(torch.uint8).cpu().numpy()
