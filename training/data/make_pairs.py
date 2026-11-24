"""Turn colored pages / illustrations into (grayscale, color) training pairs.

  python data/make_pairs.py data/raw_color --out data/pairs --screentone 0.4 --jpeg 0.5
  python data/make_pairs.py data/raw_color/commons/pages --out data/pairs --variants clean,tone --clean-scans

Writes data/pairs/color/<name>.png and data/pairs/gray/<name>.png (same size, long side <= --max-side).
Part of the grays get synthetic screentone (halftone dots in mid-tones, line art kept) so the model
also learns printed-style pages; part get JPEG artifacts like real uploads.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.io import list_images  # noqa: E402


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def screentone(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace mid-tones with a rotated halftone dot screen; keep near-black lines and white paper."""
    H, W = gray.shape
    g = gray.astype(np.float32) / 255
    period = rng.uniform(2.2, 3.6) * max(H, W) / 1600  # fine print-like dots; scale with resolution
    theta = np.deg2rad(rng.choice([45, 45, 30, 15, 0]) + rng.uniform(-3, 3))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = (xx * np.cos(theta) + yy * np.sin(theta)) * 2 * np.pi / period
    v = (-xx * np.sin(theta) + yy * np.cos(theta)) * 2 * np.pi / period
    thresh = 0.5 + 0.25 * (np.cos(u) + np.cos(v))  # in [0, 1]
    dots = (g > thresh).astype(np.float32)
    # soften dot edges a bit, like a scanned print
    dots = cv2.GaussianBlur(dots, (0, 0), max(0.3, period / 10))

    lo, hi = rng.uniform(0.2, 0.35), rng.uniform(0.72, 0.85)  # light skin/paint stays clean
    mid = ((g > lo) & (g < hi)).astype(np.float32)
    # protect anti-aliased line edges: anything within a few px of dark ink stays as drawn
    near_ink = cv2.erode(g, np.ones((7, 7), np.uint8)) < 0.35
    mid[near_ink] = 0
    mid = cv2.GaussianBlur(mid, (0, 0), 1.0)
    # only tone some regions, like hand-applied tone sheets (painted sources have no ink lines, so
    # without this the whole page, faces included, would turn into dots)
    blobs = cv2.resize(rng.random((max(2, H // 300), max(2, W // 300))).astype(np.float32), (W, H))
    mid *= (blobs > rng.uniform(0.4, 0.65)).astype(np.float32)
    out = g * (1 - mid) + dots * mid
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def jpeg(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q = int(rng.integers(55, 95))
    ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def is_mostly_gray(bgr: np.ndarray) -> bool:
    """Skip inputs that are already black and white (no color to learn from)."""
    small = cv2.resize(bgr, (128, 128), interpolation=cv2.INTER_AREA)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1]
    return float(np.mean(sat > 40)) < 0.03


def process(job):
    try:
        return _process(job)
    except Exception as e:  # one bad file must not stop a run over thousands
        return f"error: {type(e).__name__}"


def _process(job):
    path, out, max_side, p_tone, p_jpeg, variants, clean, seed = job
    rng = np.random.default_rng(seed)
    name = os.path.splitext(os.path.basename(path))[0]
    dst_c = os.path.join(out, "color", name + ".png")
    # one gray per page (random tone), or one per requested variant: gray/<name>__clean.png, __tone.png
    dst_g = (
        {v: os.path.join(out, "gray", f"{name}__{v}.png") for v in variants}
        if variants
        else {None: os.path.join(out, "gray", name + ".png")}
    )
    if os.path.exists(dst_c) and all(os.path.exists(d) for d in dst_g.values()):
        return "skip"
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "unreadable"
    if is_mostly_gray(img):
        return "gray"
    if clean:  # scanned print: white paper, halftone dots averaged out (data/clean_scans.py)
        from data.clean_scans import clean_scan

        img = clean_scan(img)
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imencode(".png", img)[1].tofile(dst_c)
    for v, dst in dst_g.items():
        gray = to_gray(img)
        if v == "tone" or (v is None and rng.random() < p_tone):
            gray = screentone(gray, rng)
        if rng.random() < p_jpeg:
            gray = jpeg(gray, rng)
        cv2.imencode(".png", gray)[1].tofile(dst)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="folder with colored pages/illustrations (searched recursively)")
    ap.add_argument("--out", default="data/pairs")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--screentone", type=float, default=0.4, help="share of pages that get synthetic tone")
    ap.add_argument("--jpeg", type=float, default=0.5, help="share of pages that get JPEG artifacts")
    ap.add_argument(
        "--variants",
        help="comma list of gray versions per page, e.g. clean,tone "
        "(overrides --screentone; training maps every variant to the same color page)",
    )
    ap.add_argument(
        "--clean-scans",
        action="store_true",
        help="scanned print comics: white-balance yellowed paper, remove halftone (data/clean_scans.py)",
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = []
    for root, _, _ in os.walk(args.input):
        paths += list_images(root)
    os.makedirs(os.path.join(args.out, "color"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "gray"), exist_ok=True)
    variants = args.variants.split(",") if args.variants else None
    if variants and not set(variants) <= {"clean", "tone"}:
        raise SystemExit("--variants takes clean and/or tone")
    jobs = [
        (p, args.out, args.max_side, args.screentone, args.jpeg, variants, args.clean_scans, args.seed + i)
        for i, p in enumerate(paths)
    ]
    counts = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(process, jobs, chunksize=8), 1):
            counts[r] = counts.get(r, 0) + 1
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} {counts}")
    print(f"done: {counts}")


if __name__ == "__main__":
    main()
