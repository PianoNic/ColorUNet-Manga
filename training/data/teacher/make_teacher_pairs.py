"""Training pairs from teacher colorizations: input = the REAL black-and-white page, target = the
teacher's colors on the page's own line art.

  python data/teacher/make_teacher_pairs.py --pages data/enhanced/mangadex_bw/pages \
      --teacher data/enhanced/teacher_klein4b --out data/pairs_teacher

For each page:
1. the teacher image (generated at ~0.6 MP, slightly redrawn) is resized to the page and aligned to
   it (ECC affine on the lightness, at low resolution); pages that do not align are dropped;
2. target lightness = the page's own L (ink, lines, text, tones exact), except on paper-white where
   the teacher put color (skin, light hair): there L may drop toward the teacher's L by at most 14,
   the same rule as models/lab.py uses at inference;
3. target a/b = the teacher's a/b; where the teacher redrew the page (its L differs a lot from the
   page's), a/b is filled in from the surroundings instead of copying misplaced color.
Writes <out>/color/<name>.png and <out>/gray/<name>__bw.png (long side <= --max-side), plus enhanced.txt.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2

cv2.setNumThreads(1)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.io import list_images  # noqa: E402

MAX_DROP = 14.0


def lab_of(rgb_u8):
    return cv2.cvtColor(rgb_u8.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def align(L_page, L_teach, side=512):
    """Affine warp (2x3, in full-res pixels) mapping the teacher onto the page, and its ECC score."""
    s = side / max(L_page.shape)
    a = cv2.resize(L_page, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b = cv2.resize(L_teach, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    a, b = cv2.GaussianBlur(a, (0, 0), 1.5), cv2.GaussianBlur(b, (0, 0), 1.5)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(
            a, b, warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5
        )
    except cv2.error:
        return None, 0.0
    warp[:, 2] /= s
    return warp, float(cc)


def fill_from_surroundings(ab, ok, sigma):
    """Normalized convolution: replace a/b where ok == 0 by the weighted mean of nearby ok pixels."""
    w = cv2.GaussianBlur(ok, (0, 0), sigma)
    out = ab.copy()
    for c in range(2):
        num = cv2.GaussianBlur(ab[..., c] * ok, (0, 0), sigma)
        out[..., c] = np.where(ok > 0.5, ab[..., c], num / np.maximum(w, 1e-3))
    return out


def process(job):
    page_path, teach_path, out, max_side, min_cc = job
    name = os.path.splitext(os.path.basename(page_path))[0]
    try:
        gray = cv2.imdecode(np.fromfile(page_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        teach = cv2.imdecode(np.fromfile(teach_path, np.uint8), cv2.IMREAD_COLOR)
        if gray is None or teach is None:
            return name, "unreadable"
        s = min(1.0, max_side / max(gray.shape))
        if s < 1:
            gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        H, W = gray.shape
        teach = cv2.cvtColor(cv2.resize(teach, (W, H), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)

        L_page = lab_of(np.repeat(gray[..., None], 3, 2))[..., 0]
        lab_t = lab_of(teach)
        warp, cc = align(L_page / 100, lab_t[..., 0] / 100)
        if warp is None or cc < min_cc:
            return name, f"not aligned (ecc {cc:.2f})"
        lab_t = cv2.warpAffine(
            lab_t, warp, (W, H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
        )
        L_t, ab = lab_t[..., 0], lab_t[..., 1:]

        # where the teacher redrew the page (moved a line, painted over white, changed a tone),
        # its color is not trustworthy: fill a/b from the surroundings
        dL = np.abs(cv2.GaussianBlur(L_t, (0, 0), 2) - cv2.GaussianBlur(L_page, (0, 0), 2))
        paper = smoothstep(86, 97, L_page)
        mismatch = np.where(paper > 0.5, dL > 30, dL > 18).astype(np.float32)  # paper may darken (skin)
        ok = 1 - cv2.dilate(mismatch, np.ones((5, 5), np.uint8))
        if ok.mean() < 0.6:
            return name, f"teacher redrew too much ({1 - ok.mean():.0%})"
        ab = fill_from_surroundings(ab, ok, sigma=max(H, W) / 150)

        # target lightness: the page's own, except colored paper may drop toward the teacher's L
        chroma = np.hypot(ab[..., 0], ab[..., 1])
        keep = smoothstep(4, 14, chroma)
        L = L_page - paper * keep * np.clip(L_page - L_t, 0, MAX_DROP)
        ab = ab * smoothstep(6, 22, L_page)[..., None]  # ink stays neutral
        rgb = cv2.cvtColor(np.dstack([L, ab]).astype(np.float32), cv2.COLOR_LAB2RGB)
        color = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)

        cv2.imencode(".png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))[1].tofile(
            os.path.join(out, "color", name + ".png")
        )
        cv2.imencode(".png", gray)[1].tofile(os.path.join(out, "gray", name + "__bw.png"))
        return name, None
    except Exception as e:  # noqa: BLE001
        return name, f"error {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True, help="original black-and-white pages")
    ap.add_argument("--teacher", required=True, help="teacher colorizations (<name>.png)")
    ap.add_argument("--out", default="data/pairs_teacher")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--min-ecc", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for d in ("color", "gray"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    with open(os.path.join(args.out, "enhanced.txt"), "w", encoding="utf-8") as f:
        f.write(
            "enhanced: teacher colorizations of copyrighted manga pages (MangaDex). enhanced training only, "
            'never for the platform model. See README "enhanced data".\n'
        )
    teach = {os.path.splitext(f)[0]: os.path.join(args.teacher, f) for f in os.listdir(args.teacher)}
    jobs = [
        (p, teach[n], args.out, args.max_side, args.min_ecc)
        for p in list_images(args.pages)
        if (n := os.path.splitext(os.path.basename(p))[0]) in teach
        and not os.path.exists(os.path.join(args.out, "color", n + ".png"))
    ]
    print(f"{len(jobs)} pages to pair", flush=True)
    bad = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (_name, why) in enumerate(ex.map(process, jobs, chunksize=4)):
            if why:
                k = why.split(" (")[0]
                bad[k] = bad.get(k, 0) + 1
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} | dropped {bad}", flush=True)
    print(f"done: {len(jobs) - sum(bad.values())} pairs, dropped {bad}")


if __name__ == "__main__":
    main()
