"""License-clean colorization U-Net (the "Student model spec" in the project brief).

- Encoder: timm ConvNeXt-Tiny, pretrained `convnext_tiny.fb_in1k` (weights: Apache-2.0 per the timm
  model card; ConvNeXt code: MIT). 1-channel input (timm folds the RGB stem weights).
  No manga-colorization-v2 weights anywhere.
- Hints: the same 4 channels as mcv2 (RGB*mask in [-1,1], mask). They bypass the pretrained encoder:
  a small full-resolution branch, plus mask-weighted average colors pooled to every decoder scale.
- Output: Lab at the input resolution. a/b is the color; L is only used by the merge's white
  handling (model lightness), the final image keeps the page's own L.

Stage 3 options (both off for stage 1/2 checkpoints, which load unchanged):
- ref=True: a color reference of the series (cover or a colored page) steers the colors. A second
  ConvNeXt-Tiny encodes it once (ref_tokens(), 256x256 -> 64 tokens); cross-attention injects the
  tokens at strides 32 and 16. Without a reference, learned "no reference" tokens are used.
- cls=True: a/b is predicted as a choice among 22x22 color bins (stride 4, annealed mean with
  temperature T), instead of a regression that drifts to gray when unsure ("Colorful Image
  Colorization", Zhang et al. 2016). L stays a regression (only used by the white handling).

Calling convention (infer.py, export and benchmark):
  forward(x (N,5,H,W), tokens=None) -> (rgb in [-1,1], None);  forward_lab(x, tokens=None) -> Lab
"""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lab import lab_to_rgb_torch

ENCODER = "convnext_tiny.fb_in1k"


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


def pool_hints(hint, stride):
    """Mask-weighted average hint colors at 1/stride resolution: (N,4,H,W) -> (N,4,H/s,W/s).

    Fixed-stride avg_pool (input sizes are multiples of 32), so it exports to ONNX with dynamic H, W.
    """
    rgb, mask = hint[:, :3], hint[:, 3:]
    m = F.avg_pool2d(mask, stride)
    c = F.avg_pool2d(rgb, stride) / m.clamp_min(1e-4)
    return torch.cat([c * (m > 0), (m > 0).to(m.dtype)], 1)


AB_STEP, AB_BINS = 10.0, 22  # bin centers -105..105 on a and b
REF_SIDE, REF_TOKENS, REF_DIM = 256, 64, 256
COOL_GUARD = False
PURPLE_TO = (12.0, 26.0)  # warm brown (Lab a, b) that wrongly guessed purple turns into on light areas


def golden_shift(net, strength=0.5, hue_range=(20.0, 60.0), min_chroma=18.0, target_hue=82.0):
    """Shift the color bins of saturated salmon/peach/orange-pink (Lab hue 20-60 deg) toward golden
    yellow (and a little more chroma), so light blonde hair comes out golden instead of peach.
    Blush/red (hue < 20) and skin (lower chroma) are not touched. Changes net.centers in place."""
    c = net.centers
    hue = torch.rad2deg(torch.atan2(c[:, 1], c[:, 0]))
    chroma = c.norm(dim=1)
    sel = (hue > hue_range[0]) & (hue < hue_range[1]) & (chroma > min_chroma)
    h = torch.deg2rad(hue[sel] + strength * (target_hue - hue[sel]))
    r = chroma[sel] * (1 + 0.25 * strength)
    c[sel] = torch.stack([r * torch.cos(h), r * torch.sin(h)], 1)
    return int(sel.sum())


def gray01_to_L(g):
    """Grayscale in [0,1] -> CIE L (torch, ONNX-friendly)."""
    lin = torch.where(g > 0.04045, ((g + 0.055) / 1.055).clamp_min(1e-6) ** 2.4, g / 12.92)
    f = torch.where(lin > (6 / 29) ** 3, lin.clamp_min(1e-6) ** (1 / 3), lin / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f - 16


def purple_share(rgb01):
    """Share of clearly colored pixels of an image (N,3,H,W in [0,1]) that are violet/purple: (N,)."""
    from models.lab import rgb_to_lab_torch

    lab = rgb_to_lab_torch(rgb01)
    a, b = lab[:, 1], lab[:, 2]
    colored = (torch.sqrt(a * a + b * b + 1e-6) > 12).float()
    purple = colored * (a > 0).float() * (b < -4).float()
    return purple.sum((1, 2)) / colored.sum((1, 2)).clamp_min(1.0)


def cover_brake(rgb01):
    """Purple fix strength from the series cover: 1 for a purple-free cover, 0 if >= 6% of its colored
    pixels are purple (then purple hair or clothes are probably real)."""
    return (1 - purple_share(rgb01) / 0.06).clamp(0, 1)


def ab_centers():
    c = torch.arange(AB_BINS, dtype=torch.float32) * AB_STEP - (AB_BINS - 1) * AB_STEP / 2
    a, b = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([a.flatten(), b.flatten()], 1)  # (Q, 2), Q = 484


class CrossAttention(nn.Module):
    """Image features attend to reference tokens; zero-initialised output, so a stage-2 network
    starts out unchanged."""

    def __init__(self, ch, dim=REF_DIM, heads=8):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv2d(ch, dim, 1)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Conv2d(dim, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, tokens):
        N, _, H, W = x.shape
        h, d = self.heads, self.dim // self.heads
        q = self.q(self.norm(x)).reshape(N, h, d, H * W).transpose(2, 3)  # N,h,HW,d
        k, v = self.kv(tokens).reshape(N, -1, 2, h, d).permute(2, 0, 3, 1, 4)  # 2,N,h,T,d
        att = torch.softmax(q @ k.transpose(2, 3) / math.sqrt(d), -1)
        y = (att @ v).transpose(2, 3).reshape(N, self.dim, H, W)
        return x + self.proj(y)


class ColorUNet(nn.Module):
    def __init__(
        self, pretrained=True, encoder=ENCODER, dec=(256, 128, 64, 32), ref=False, cls=False, temperature=0.38
    ):
        super().__init__()
        self.use_ref, self.use_cls, self.temperature = ref, cls, temperature
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, in_chans=1)
        ch = self.encoder.feature_info.channels()  # convnext_tiny: 96, 192, 384, 768 (strides 4-32)
        self.full = conv_block(1 + 4, 32)  # full-res detail + hints
        self.stem2 = nn.Sequential(nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GELU(), conv_block(48, 48))
        self.up3 = conv_block(ch[3] + ch[2] + 4, dec[0])  # -> stride 16
        self.up2 = conv_block(dec[0] + ch[1] + 4, dec[1])  # -> 8
        self.up1 = conv_block(dec[1] + ch[0] + 4, dec[2])  # -> 4
        self.up0 = conv_block(dec[2] + 48 + 4, dec[3])  # -> 2
        self.out = nn.Sequential(conv_block(dec[3] + 32, 32), nn.Conv2d(32, 3, 1))
        self.register_buffer("mean", torch.tensor(0.45))
        self.register_buffer("std", torch.tensor(0.226))
        if ref:
            self.ref_encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, out_indices=(3,))
            self.ref_proj = nn.Linear(ch[3], REF_DIM)
            self.null_tokens = nn.Parameter(torch.randn(1, REF_TOKENS, REF_DIM) * 0.02)
            self.ref_pos = nn.Parameter(torch.randn(1, REF_TOKENS, REF_DIM) * 0.02)
            self.att32 = CrossAttention(ch[3])
            self.att16 = CrossAttention(dec[0])
            self.register_buffer("ref_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("ref_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if cls:
            self.cls_head = nn.Sequential(conv_block(dec[2], dec[2]), nn.Conv2d(dec[2], AB_BINS * AB_BINS, 1))
            self.register_buffer("centers", ab_centers())
            c = ab_centers()
            mask = (c[:, 0] > 0) & (c[:, 1] < -4) & (c.norm(dim=1) > 15)
            self.register_buffer("purple_mask", mask.float(), persistent=False)
            hue = torch.rad2deg(torch.atan2(c[:, 1], c[:, 0])) % 360
            # cool-tint guard: off (page cast removal in models/lab.py handles blue casts; the guard turned
            # real blue-gray clothes like sailor uniforms gray). Set COOL_GUARD = True to re-enable.
            cool = (hue > 190) & (hue < 330) & (c.norm(dim=1) < 22) & torch.tensor(COOL_GUARD)
            self.register_buffer("cool_mask", cool.float(), persistent=False)

    def ref_tokens(self, ref_rgb):
        """Reference image(s) (N,3,256,256) in [0,1] -> tokens (N,64,256). Computed once per series."""
        f = self.ref_encoder((ref_rgb - self.ref_mean) / self.ref_std)[0]  # N,768,8,8
        return self.ref_proj(f.flatten(2).transpose(1, 2)) + self.ref_pos

    def tokens_for(self, n, tokens=None, has_ref=None):
        """tokens, with the learned no-reference tokens where there is none (has_ref (N,) 0/1)."""
        null = self.null_tokens.expand(n, -1, -1)
        if tokens is None:
            return null
        if has_ref is None:
            return tokens
        m = has_ref.view(-1, 1, 1).to(tokens.dtype)
        return tokens * m + null * (1 - m)

    def ab_from_logits(self, logits, light=None, brake=None):
        """Annealed mean of the bin distribution: (N,Q,h,w) -> (N,2,h,w).

        brake (N,) in [0,1], light (N,1,h,w) = page L: the purple fix (plus the cool-tint guard below). Probability on saturated purple
        bins moves to warm brown where the page is light (blonde/light hair, L > ~70) and to neutral
        where it is darker (clothes, shadows), so wrongly guessed purple disappears.
        """
        p = torch.softmax(logits.float() / self.temperature, 1)
        w = self.centers.float().T[:, :, None, None]
        ab = F.conv2d(p, w)  # 1x1 conv = weighted sum of bin centers
        if brake is None or light is None:
            return ab
        m = self.purple_mask[None, :, None, None]
        pp = p * m
        ab_purple = F.conv2d(pp, w)
        mass = pp.sum(1, keepdim=True)
        t = ((light - 55) / 20).clamp(0, 1)
        wl = t * t * (3 - 2 * t)
        brown = torch.tensor(PURPLE_TO, device=ab.device, dtype=ab.dtype).view(1, 2, 1, 1)
        k = brake.view(-1, 1, 1, 1).to(ab.dtype)
        ab = ab - k * ab_purple + k * mass * wl * brown
        # cool-tint guard (always on with the fix): a weak blue/lavender cast on mid and dark areas
        # (gray clothes, dark fabric) becomes neutral; light areas (sky) and strong blues stay
        ab_cool = F.conv2d(p * self.cool_mask[None, :, None, None], w)
        t2 = ((light - 62) / 18).clamp(0, 1)
        return ab - (1 - t2 * t2 * (3 - 2 * t2)) * ab_cool

    def forward_lab(self, x, tokens=None, return_logits=False, brake=None):
        gray, hint = x[:, :1], x[:, 1:5]
        f = self.encoder((gray - self.mean) / self.std)
        full = self.full(torch.cat([gray, hint], 1))
        half = self.stem2(full)

        def up(y, skip, stride, block):
            y = F.interpolate(y, size=skip.shape[2:], mode="bilinear", align_corners=False)
            return block(torch.cat([y, skip, pool_hints(hint, stride)], 1))

        f3 = f[3]
        if self.use_ref:
            tok = self.tokens_for(x.shape[0], tokens)
            f3 = self.att32(f3, tok)
        y = up(f3, f[2], 16, self.up3)
        if self.use_ref:
            y = self.att16(y, tok)
        y = up(y, f[1], 8, self.up2)
        y = up(y, f[0], 4, self.up1)
        logits = self.cls_head(y) if self.use_cls else None
        y = up(y, half, 2, self.up0)
        y = F.interpolate(y, size=full.shape[2:], mode="bilinear", align_corners=False)
        y = self.out(torch.cat([y, full], 1)).float()
        L = torch.sigmoid(y[:, :1]) * 100
        if self.use_cls:
            light = None
            if brake is not None:
                light = gray01_to_L(F.avg_pool2d(gray.float(), 4))  # logits are at stride 4
            ab = F.interpolate(
                self.ab_from_logits(logits, light, brake), size=full.shape[2:], mode="bilinear", align_corners=False
            )
        else:
            ab = torch.tanh(y[:, 1:]) * 110
        lab = torch.cat([L, ab], 1)
        return (lab, logits) if return_logits else lab

    def forward(self, x, tokens=None, brake=None):
        rgb = lab_to_rgb_torch(self.forward_lab(x, tokens, brake=brake)).clamp(0, 1)
        return rgb * 2 - 1, None
