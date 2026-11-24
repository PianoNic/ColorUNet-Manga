"""Do color hints change the output? Colorize one page with no hint and with blonde / pink / blue hints
on the same points, save a strip, and report how far the color near the points moved toward each hint.

  python eval/hint_test.py --weights checkpoints/unet_pc1/last.pt --out eval/compare/hints_unet.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import io as mio  # noqa: E402
from scripts.infer import Colorizer  # noqa: E402

# Marin's hair on ch0115.5_p012 (fractions of page width/height)
PAGE = "eval/testset/more-than-a-doll/ch0115.5_p012.png"
POINTS = [(0.244, 0.857), (0.289, 0.747), (0.222, 0.92)]
COLORS = {"blonde": "#f3d27a", "pink": "#f08cb4", "blue": "#5a8fe0"}


def ab_at(rgb, pts, r=12):
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    H, W = rgb.shape[:2]
    return np.mean(
        [
            lab[int(y * H) - r : int(y * H) + r, int(x * W) - r : int(x * W) + r, 1:].reshape(-1, 2).mean(0)
            for x, y in pts
        ],
        axis=0,
    )


def hex_ab(c):
    rgb = np.array([[[int(c[i : i + 2], 16) for i in (1, 3, 5)]]], np.float32) / 255
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=mio.DEFAULT_WEIGHTS)
    ap.add_argument("--page", default=PAGE)
    ap.add_argument("--out", default="eval/compare/hints_test.jpg")
    args = ap.parse_args()

    col = Colorizer(args.weights)
    gray = mio.read_gray(args.page)
    base = col(gray)
    tiles, base_ab = [base], ab_at(base, POINTS)
    print(f"no hint: hair a/b = {base_ab.round(1)}")
    for name, c in COLORS.items():
        out = col(gray, [{"x": x, "y": y, "color": c} for x, y in POINTS])
        got, want = ab_at(out, POINTS), hex_ab(c)
        moved = 1 - np.linalg.norm(got - want) / max(np.linalg.norm(base_ab - want), 1e-6)
        print(f"{name:<7} hint a/b {want.round(1)}  got {got.round(1)}  moved {moved * 100:5.1f}% of the way")
        tiles.append(out)
    H, W = gray.shape
    y0, y1, x1 = int(0.7 * H), int(0.97 * H), int(0.55 * W)
    crops = []
    for i, t in enumerate(tiles):
        c = np.ascontiguousarray(t[y0:y1, :x1])
        if i:
            for x, y in POINTS:
                cv2.circle(c, (int(x * W), int(y * H) - y0), 14, (255, 0, 0), 3)
        cv2.putText(c, ["no hint", *COLORS][i], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        crops.append(cv2.resize(c, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA))
    mio.write_rgb(args.out, np.concatenate(crops, 1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
