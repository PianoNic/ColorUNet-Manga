"""Compare model snapshots on the probe pages: purple share, cover effect, colorfulness.

  python eval/snapshot_scores.py checkpoints/stage3c/step*.pt --ref data/enhanced/teacher_refs/bisque_vol12.jpg

purple = share of clearly colored pixels (chroma > 12) whose Lab hue is violet/purple/magenta
(a > 0, b < -4: hue roughly 280-350 deg); in this series that is almost always a wrong hair color.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import io as mio  # noqa: E402
from models.lab import merge_lab  # noqa: E402


def stats(rgb):
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    a, b = lab[..., 1], lab[..., 2]
    c = np.hypot(a, b)
    colored = c > 12
    purple = colored & (a > 0) & (b < -4)
    return colored.mean(), purple.sum() / max(1, colored.sum()), c.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--pages", default="eval/testset/more-than-a-doll")
    ap.add_argument("--ref", default="data/enhanced/teacher_refs/bisque_vol12.jpg")
    args = ap.parse_args()
    paths = mio.list_images(args.pages)
    paths = paths[:: max(1, len(paths) // 8)][:8]
    grays = [mio.read_gray(p) for p in paths]
    print(
        f"{'snapshot':28s} {'colored%':>9s} {'purple% (no cover)':>19s} {'purple% (cover)':>16s} {'chroma':>7s} {'cover effect':>13s}"
    )
    for w in args.weights:
        gen = mio.load_model(w, "cuda")
        tok = gen.ref_tokens(mio.load_ref(args.ref, "cuda")) if getattr(gen, "use_ref", False) else None
        res = {None: [], "ref": []}
        diff = []
        for g in grays:
            x, hw = mio.resize_for_model(g, 768)
            outs = {}
            for key, t in ((None, None), ("ref", tok)):
                if key == "ref" and t is None:
                    continue
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    out, _ = gen(mio.make_input(x, "cuda"), t)
                outs[key] = merge_lab(g, mio.to_rgb01(out, hw))
                res[key].append(stats(outs[key]))
            if "ref" in outs:
                diff.append(np.abs(outs["ref"].astype(np.float32) - outs[None]).mean())
        n = np.mean(res[None], 0)
        r = np.mean(res["ref"], 0) if res["ref"] else [np.nan] * 3
        print(
            f"{os.path.basename(w):28s} {n[0] * 100:8.1f}% {n[1] * 100:18.1f}% {r[1] * 100:15.1f}% {n[2]:7.2f} "
            f"{np.mean(diff) if diff else float('nan'):10.2f}/255"
        )
        del gen
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
