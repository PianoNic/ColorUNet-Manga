"""Export the ColorUNet + the whole merge as ONE ONNX graph for the browser (ONNX Runtime Web, WebGPU).

  python export/to_web.py --weights checkpoints/stage1_commons/best.pt --out export/web
    -> export/web/colorizer-fp32.onnx, colorizer-fp16.onnx (model in fp16, merge in fp32), meta.json

Graph contract (what the browser does is only resize + pack pixels):
  inputs  gray  (1,1,H,W)  float32 in [0,1]   the full-resolution page (its lightness is kept exactly)
          small (1,1,h,w)  float32 in [0,1]   the same page resized to h,w = multiples of 32 near
                                              --long-side (stretch, no padding)
  output  rgb   (1,3,H,W)  float32 in [0,1]
Inside: model (no hints) -> Lab a/b at h,w -> guided upsampling on the page's L -> white/black
cleanup -> gamut fit. Same math as models/lab.py, written with ops ONNX Runtime Web's WebGPU backend
supports (1x1 conv instead of einsum, plain pooling/resize).
Stage-3 checkpoints (--ref, --cls) export without a reference input: the graph uses the model's
learned no-reference tokens. Only license-clean ColorUNet checkpoints are exported by default; enhanced-data checkpoints need
--allow-enhanced (for a site only the owner uses) and are marked enhanced_data in meta.json.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet import ColorUNet  # noqa: E402

D = 6 / 29
WHITE = (0.950456, 1.0, 1.088754)
XYZ2RGB = [[3.240479, -1.537150, -0.498535], [-0.969256, 1.875992, 0.041556], [0.055648, -0.204043, 1.057311]]


def gray_to_L(g):
    lin = torch.where(g > 0.04045, ((g + 0.055) / 1.055).clamp_min(1e-6) ** 2.4, g / 12.92)
    f = torch.where(lin > D**3, lin.clamp_min(1e-6) ** (1 / 3), lin / (3 * D * D) + 4 / 29)
    return 116 * f - 16


def box(x, r):
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), 2 * r + 1, stride=1)


def smoothstep(e0, e1, x):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


class WebColorizer(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        m = torch.tensor(XYZ2RGB) * torch.tensor(WHITE)[None, :]  # white point folded in
        self.register_buffer("xyz2rgb", m[:, :, None, None])

    def lab_to_rgb(self, L, ab):
        fy = (L + 16) / 116
        f = torch.cat([fy + ab[:, :1] / 500, fy, fy - ab[:, 1:] / 200], 1)
        xyz = torch.where(f > D, f**3, 3 * D * D * (f - 4 / 29))
        lin = F.conv2d(xyz, self.xyz2rgb)
        return torch.where(lin > 0.0031308, 1.055 * lin.abs().clamp_min(1e-6) ** (1 / 2.4) - 0.055, 12.92 * lin)

    def forward(self, gray, small, tokens=None, brake=None):
        x = torch.cat([small, torch.zeros_like(small).repeat(1, 4, 1, 1)], 1)
        lab_low = self.net.forward_lab(x, tokens, brake=brake)
        L = gray_to_L(gray)
        H, W = gray.shape[2], gray.shape[3]
        up = lambda t: F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)

        # guided upsampling (fast guided filter, radius 2, eps 1e-3) of a/b along the page's L
        g_low = F.interpolate(L / 100, size=small.shape[2:], mode="bilinear", align_corners=False)
        ab_low = lab_low[:, 1:]
        # page cast removal (same as models/lab.py remove_cast): yellow/orange rooms back toward neutral
        c0 = torch.sqrt(ab_low[:, :1] ** 2 + ab_low[:, 1:] ** 2 + 1e-6)
        wc = (c0 > 3).float() * (c0 < 30).float() * (lab_low[:, :1] > 40).float()
        cast = (ab_low * wc).sum((2, 3), keepdim=True) / wc.sum((2, 3), keepdim=True).clamp_min(1.0)
        dc = torch.sqrt(((ab_low - cast) ** 2).sum(1, keepdim=True) + 1e-6)
        ab_low = ab_low - 0.7 * (1 - smoothstep(6.0, 18.0, dc)) * cast
        r, eps = 2, 1e-3
        mean_g = box(g_low, r)
        var_g = box(g_low * g_low, r) - mean_g * mean_g
        mean_p = box(ab_low, r)
        A = (box(g_low * ab_low, r) - mean_g * mean_p) / (var_g + eps)
        B = mean_p - A * mean_g
        ab = (up(box(A, r)) * (L / 100) + up(box(B, r))) * 1.4  # same chroma boost as models/lab.py

        # white/black cleanup: tints fade on paper, ink stays black, real colors on paper get room
        L_model = up(F.max_pool2d(lab_low[:, :1], 3, stride=1, padding=1))
        c = torch.sqrt(ab[:, :1] ** 2 + ab[:, 1:] ** 2 + 1e-6)
        w_white = smoothstep(86.0, 97.0, L)
        keep = smoothstep(1.5, 6.0, c)  # weak tints on paper fade out (bubbles stay white), light skin stays
        ab = ab * (1 - w_white * (1 - keep)) * smoothstep(6.0, 22.0, L)
        L = L - w_white * keep * (L - L_model).clamp(0, 35.0)

        # gamut fit: shrink chroma (never L) until the color is inside sRGB (6 bisection steps)
        def ok(scale):
            rgb = self.lab_to_rgb(L, ab * scale)
            return ((rgb >= -0.003) & (rgb <= 1.003)).all(1, keepdim=True).float()

        hi = torch.ones_like(L)
        lo = ok(hi)
        for _ in range(6):
            mid = (lo + hi) / 2
            good = ok(mid)
            lo = good * mid + (1 - good) * lo
            hi = good * hi + (1 - good) * mid
        return self.lab_to_rgb(L, ab * lo).clamp(0, 1)


class HalfNet(nn.Module):
    """The network in fp16, its inputs/outputs in fp32 (the merge stays fp32 for exact lightness)."""

    def __init__(self, net):
        super().__init__()
        self.net = net.half()

    def forward_lab(self, x, tokens=None, brake=None):
        return self.net.forward_lab(x.half(), None if tokens is None else tokens.half(), brake=brake).float()

    def ref_tokens(self, ref):
        return self.net.ref_tokens(ref.half()).float()


class RefEncoder(nn.Module):
    """Cover / colored reference page (1,3,256,256) in [0,1] -> color tokens (1,64,256), once per series."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, ref):
        from models.unet import cover_brake

        return self.net.ref_tokens(ref), cover_brake(ref)


def load_unet(path, allow_enhanced=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    if a.get("model") != "unet":
        raise SystemExit("only ColorUNet checkpoints can be exported for the web")
    if a.get("enhanced_data") and not allow_enhanced:
        raise SystemExit(
            f"{path} was trained on enhanced data; it must not be served to other people "
            "browsers. --allow-enhanced exports it for a site only you use."
        )
    net = ColorUNet(pretrained=False, ref=bool(a.get("ref")), cls=bool(a.get("cls")))
    net.load_state_dict(ck["generator"])
    return net.eval(), a


def export(model, path, opset=17, device="cpu", with_tokens=False):
    model = model.to(device)
    gray = torch.rand(1, 1, 1100, 780, device=device)
    small = torch.rand(1, 1, 768, 544, device=device)
    inputs, names = (gray, small), ["gray", "small"]
    if with_tokens:
        from models.unet import REF_DIM, REF_TOKENS

        inputs = (gray, small, torch.randn(1, REF_TOKENS, REF_DIM, device=device), torch.ones(1, device=device))
        names = names + ["tokens", "brake"]
    torch.onnx.export(
        model,
        inputs,
        path,
        input_names=names,
        output_names=["rgb"],
        opset_version=opset,
        dynamo=False,
        dynamic_axes={"gray": {2: "H", 3: "W"}, "small": {2: "h", 3: "w"}, "rgb": {2: "H", 3: "W"}},
    )


def export_ref_encoder(model, path, opset=17, device="cpu"):
    from models.unet import REF_SIDE

    model = model.to(device)
    torch.onnx.export(
        model,
        (torch.rand(1, 3, REF_SIDE, REF_SIDE, device=device),),
        path,
        input_names=["ref"],
        output_names=["tokens", "brake"],
        opset_version=opset,
        dynamo=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="export/web")
    ap.add_argument("--long-side", type=int, default=768)
    ap.add_argument(
        "--golden",
        type=float,
        default=0.0,
        help="color-bin model: shift salmon/peach toward golden blonde (0.5 = recommended)",
    )
    ap.add_argument(
        "--allow-enhanced",
        action="store_true",
        help="export a checkpoint trained on enhanced data (only for a site only you use)",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    net, train_args = load_unet(args.weights, args.allow_enhanced)
    if args.golden and net.use_cls:
        from models.unet import golden_shift

        print(f"golden shift {args.golden}: {golden_shift(net, args.golden)} color bins moved")

    import copy

    files = ["colorizer-fp16.onnx", "colorizer-fp32.onnx"]
    with torch.no_grad():
        export(WebColorizer(copy.deepcopy(net)).eval(), os.path.join(args.out, "colorizer-fp32.onnx"))
        # fp16 convolutions only trace on the GPU
        export(
            WebColorizer(HalfNet(copy.deepcopy(net))).eval(),
            os.path.join(args.out, "colorizer-fp16.onnx"),
            device="cuda",
        )
        if net.use_ref:
            # cover reference: encoder (once per series) + colorizer with a tokens input; no cover ->
            # null-tokens.bin (the model's learned no-reference tokens, float32 little-endian 64x256)
            export(
                WebColorizer(HalfNet(copy.deepcopy(net))).eval(),
                os.path.join(args.out, "colorizer-ref-fp16.onnx"),
                device="cuda",
                with_tokens=True,
            )
            export_ref_encoder(
                RefEncoder(HalfNet(copy.deepcopy(net))).eval(),
                os.path.join(args.out, "ref-encoder-fp16.onnx"),
                device="cuda",
            )
            net.null_tokens.detach().float().cpu().numpy().astype("<f4").tofile(
                os.path.join(args.out, "null-tokens.bin")
            )
            files += ["colorizer-ref-fp16.onnx", "ref-encoder-fp16.onnx", "null-tokens.bin"]
    meta = {
        "longSide": args.long_side,
        "multiple": 32,
        "weights": os.path.basename(args.weights),
        "step": torch.load(args.weights, map_location="cpu", weights_only=False).get("step"),
        "model": "ColorUNet (enhanced data: enhanced use only)"
        if train_args.get("enhanced_data")
        else "ColorUNet (license-clean)",
        "enhanced_data": bool(train_args.get("enhanced_data")),
        "inputs": ["gray", "small"],
        "output": "rgb",
        "golden": args.golden,
    }
    if net.use_ref:
        from models.unet import REF_DIM, REF_SIDE, REF_TOKENS

        meta["reference"] = {
            "encoder": "ref-encoder-fp16.onnx",
            "colorizer": "colorizer-ref-fp16.onnx",
            "null": "null-tokens.bin",
            "refInput": ["ref", [1, 3, REF_SIDE, REF_SIDE], "float32 RGB in [0,1], cover stretched to a square"],
            "tokens": ["tokens", [1, REF_TOKENS, REF_DIM], "float32"],
            "brake": [
                "brake",
                [1],
                "float32 in [0,1]: purple fix strength; the encoder outputs it per cover, use 1.0 with null tokens",
            ],
            "colorizerInputs": ["gray", "small", "tokens", "brake"],
        }
    # version: changes with every export (hash of the model files), so a browser cache keyed on
    # meta.json picks up a new model
    import hashlib
    import time

    h = hashlib.sha256()
    for n in files:
        with open(os.path.join(args.out, n), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
    meta["version"] = h.hexdigest()[:16]
    meta["exported"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for n in files:
        print(f"{n}: {os.path.getsize(os.path.join(args.out, n)) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
