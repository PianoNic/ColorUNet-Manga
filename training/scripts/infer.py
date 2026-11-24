"""Colorize pages: the model picks color at low resolution, the original page keeps the lightness.

1. read page at full resolution (grayscale = L)
2. optional descreen: smooth screentone regions of L (lines and text untouched)
3. model input: page downscaled to --long-side; optional palette hints
4. take only Lab a/b of the model output, upsample edge-aware (guided by the full-res page)
5. fade chroma to 0 on paper/highlights and ink, merge with L -> RGB

python scripts/infer.py eval/testset/more-than-a-doll --out eval/out/lab --sbs
python scripts/infer.py pages/ --weights checkpoints/stage2/best.pt --palette palettes/bisque_doll.json
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import io as mio  # noqa: E402
from models.descreen import descreen as descreen_page  # noqa: E402
from models.hints import load_palette, palette_hints, points_for  # noqa: E402
from models.lab import merge_lab  # noqa: E402
from models.unet import cover_brake  # noqa: E402


class Colorizer:
    def __init__(
        self,
        weights=mio.DEFAULT_WEIGHTS,
        device="cuda",
        long_side=768,
        fp16=True,
        upsample="guided",
        chroma_falloff=True,
        descreen=False,
    ):
        self.device, self.long_side = device, long_side
        self.upsample, self.chroma_falloff, self.descreen = upsample, chroma_falloff, descreen
        self.fp16 = fp16 and device == "cuda"
        self.gen = mio.load_model(weights, device)
        self.tokens = self.brake = None

    @torch.no_grad()
    def set_reference(self, path):
        """Use a colored reference of the series (its cover) for every following page; None to clear."""
        if path is None or not getattr(self.gen, "use_ref", False):
            self.tokens = self.brake = None
            return
        ref = mio.load_ref(path, self.device)
        self.tokens = self.gen.ref_tokens(ref)
        self.brake = cover_brake(ref) if getattr(self.gen, "use_cls", False) else None

    @torch.no_grad()
    def __call__(self, gray_u8: np.ndarray, points=None) -> np.ndarray:
        page = descreen_page(gray_u8) if self.descreen else gray_u8
        g, hw = mio.resize_for_model(page, self.long_side)
        hint = palette_hints(points, g.shape, hw).to(self.device) if points else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            out, _ = self.gen(mio.make_input(g, self.device, hint), self.tokens, brake=self.brake)
        return merge_lab(
            gray_u8, mio.to_rgb01(out, hw), self.upsample, self.chroma_falloff, L_source=page if self.descreen else None
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("--out", default="eval/out/lab")
    ap.add_argument("--weights", default=mio.DEFAULT_WEIGHTS)
    ap.add_argument("--long-side", type=int, default=768, help="model input size (long side, px)")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--upsample", choices=["guided", "bilinear"], default="guided")
    ap.add_argument(
        "--chroma-falloff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fade color to 0 on near-white paper/highlights and near-black ink",
    )
    ap.add_argument("--descreen", action="store_true", help="smooth screentone regions of L")
    ap.add_argument("--palette", help="per-series palette JSON (see models/hints.py)")
    ap.add_argument(
        "--ref", help="the series cover (or a colored page) as color reference, for models trained with one"
    )
    ap.add_argument("--sbs", action="store_true", help="also save [original | color] side by side")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    col = Colorizer(
        args.weights, device, args.long_side, not args.fp32, args.upsample, args.chroma_falloff, args.descreen
    )
    col.set_reference(args.ref)
    palette = load_palette(args.palette) if args.palette else None
    paths = [args.input] if os.path.isfile(args.input) else mio.list_images(args.input)
    os.makedirs(args.out, exist_ok=True)

    times = []
    for i, p in enumerate(paths):
        gray = mio.read_gray(p)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        rgb = col(gray, points_for(palette, p))
        dt = time.perf_counter() - t0
        if i > 0:
            times.append(dt)
        name = os.path.splitext(os.path.basename(p))[0]
        mio.write_rgb(os.path.join(args.out, name + ".png"), rgb)
        if args.sbs:
            mio.write_rgb(os.path.join(args.out, name + "_sbs.jpg"), mio.side_by_side(gray, rgb))
        print(f"{name}: {dt:.3f} s")
    if times:
        print(
            f"\n{len(paths)} pages on {device}: {np.mean(times):.3f} s/page mean "
            f"(full pipeline; first page excluded as warm-up)"
        )


if __name__ == "__main__":
    main()
