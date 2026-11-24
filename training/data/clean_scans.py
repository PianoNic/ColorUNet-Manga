"""Clean scanned print comics (Golden Age scans) into usable color targets.

  python data/clean_scans.py data/raw_color/commons/pages --out data/raw_color/commons/clean
  python data/clean_scans.py page.jpg --out /tmp/x --compare       # also save before/after

1. Yellowed paper: estimate the paper color from bright, low-saturation pixels and scale each channel
   so paper becomes white (a white balance, so colors shift consistently with it); then stretch the
   black point so ink is black again.
2. Halftone dots: four-color print is made of CMYK dot patterns. If the scan resolves them (sharp
   peaks in the page's spectrum, searched only at periods a print screen can have), the page is
   area-downscaled until the dot period is ~1 px: the dots average into flat color while lines stay
   as sharp as that size allows. (Masking line art and blurring around it was tried; dark-tone dot
   networks and thin lines could not be separated reliably.) Training uses ~770-960 px pages anyway.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.io import list_images  # noqa: E402


def white_balance(bgr: np.ndarray) -> np.ndarray:
    img = bgr.astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    bright = v >= np.percentile(v, 70)  # >= : pages with lots of pure white
    paper = bright & (s <= np.percentile(s[bright], 60))
    if paper.sum() < 100:
        return bgr
    white = np.percentile(img[paper], 75, axis=0)  # paper color per channel
    img = img * (255.0 / np.maximum(white, 1))[None, None]
    gray = img.mean(2)
    black = np.percentile(gray, 0.5)  # stretch ink back to black
    img = (img - black) * 255.0 / max(255.0 - black, 1)
    return np.clip(img, 0, 255).astype(np.uint8)


def halftone_period(bgr: np.ndarray, n: int = 256):
    """Dot period in px if a print screen is visible (sharp FFT peaks), else None.

    Only periods a print screen can have at this scan size are searched: a comic page is ~7 in wide
    and printed at ~85-133 lpi, so the dot period is at most ~1/600 of the long side (~5 px on a
    3000 px scan); longer periodic structure is lettering, tiles or hatching. Each color channel is
    checked, since every ink (C, M, Y, K) has its own screen.
    """
    H, W = bgr.shape[:2]
    max_period = max(H, W) / 600
    if min(H, W) < n or max_period < 2.2:
        return None
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    yy, xx = np.mgrid[-n // 2 : n // 2, -n // 2 : n // 2]
    r = np.hypot(yy, xx)
    band = (r >= n / max_period) & (r <= n / 2.2)
    ref = (r > n / 14) & (r < n / 2.2)  # background level for the score
    best = (0.0, None)
    for c in range(3):
        for cy, cx in [(H // 3, W // 3), (H // 2, W // 2), (2 * H // 3, 2 * W // 3), (H // 3, 2 * W // 3)]:
            patch = bgr[cy - n // 2 : cy + n // 2, cx - n // 2 : cx + n // 2, c].astype(np.float32)
            f = np.abs(np.fft.fftshift(np.fft.fft2((patch - patch.mean()) * win)))
            i = np.argmax(f[band])
            score = f[band][i] / (np.median(f[ref]) + 1e-6)
            if score > best[0]:
                best = (score, n / r[band][i])
    return best[1] if best[0] > 20 else None


def clean_scan(bgr: np.ndarray, verbose: bool = False, min_side: int = 900) -> np.ndarray:
    """White-balance the paper, then remove a visible print screen by area-downscaling until the dot
    period is ~1 px (dots average into flat color; lines stay as sharp as that resolution allows).
    Training sees pages at ~770-960 px anyway; the long side is never taken below min_side."""
    bgr = white_balance(bgr)
    period = halftone_period(bgr)
    if not period:
        if verbose:
            print("  no resolvable halftone: white balance only")
        return bgr
    scale = max(1.0 / period, min_side / max(bgr.shape[:2]))
    if verbose:
        print(f"  halftone period {period:.1f} px -> downscale x{scale:.2f}")
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="image or folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", action="store_true", help="also save before|after crops")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths = [args.input] if os.path.isfile(args.input) else list_images(args.input)
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(args.out, name + ".png")
        if os.path.exists(dst) and not args.compare:
            continue
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        print(os.path.basename(p))
        out = clean_scan(img, verbose=True)
        cv2.imencode(".png", out)[1].tofile(dst)
        if args.compare:
            h, w = img.shape[:2]
            y, x, s = h // 3, w // 3, min(500, h // 3, w // 3)
            cmp = np.concatenate([img[y : y + s, x : x + s], out[y : y + s, x : x + s]], 1)
            cv2.imencode(".jpg", cmp)[1].tofile(os.path.join(args.out, name + "_compare.jpg"))
    print(f"{len(paths)} pages cleaned into {args.out}")


if __name__ == "__main__":
    main()
