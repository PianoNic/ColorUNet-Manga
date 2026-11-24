"""Train the ColorUNet (models/unet.py) on (gray, color) pairs. Built for the RTX 5080 (16 GB).

  # stage 1: from the ImageNet encoder on the license-clean pairs
  python scripts/finetune.py --pairs data/pairs --out checkpoints/stage1_commons
  # stage 2: continue from stage 1 on manga pairs, with some stage-1 data mixed in
  python scripts/finetune.py --pairs data/pairs_enhanced --mix data/pairs:0.15 \
      --weights checkpoints/stage1_commons/best.pt --lr 1e-4 --out checkpoints/stage2
  python scripts/finetune.py ... --resume checkpoints/stage2/last.pt

Color hints: every sample gets 1..--max-hints random color points from its target; whole batches
(--hint-drop, default half) are trained with no hints, so the model works with and without them.

White paper (--white-paper, share of samples): light grays of the INPUT are lifted to pure white
(models/augment.py), like skin and light hair left as bare paper in drawn manga; the target keeps
its colors, so white areas inside a face or hair outline still get color.

Loss: L1 on Lab a/b (the only part we keep at inference, see models/lab.py) + a little on L.
bf16 autocast, gradient accumulation, checkpoints and colorized sample pages every --sample-every
steps (with --target: the reference colorization as an extra row). A held-out set (--holdout-pct of
the PAGES of --pairs, all their variants together; the mixed-in set keeps its own held-out pages
out of training too) gives a validation loss every --eval-every steps, without and with hints, plus
the error near hint points; best.pt keeps the lowest held-out loss, val_log.csv the history.
Pairs folders with enhanced.txt mark every checkpoint trained on them as enhanced_data.

Stage 3 (--ref, --cls; see models/unet.py):
  python scripts/finetune.py --pairs data/pairs_teacher --mix data/pairs_enhanced:0.35 --mix data/pairs:0.1 \
      --weights checkpoints/stage2_manga/best.pt --ref --cls --out checkpoints/stage3
- --ref: every sample gets a color reference of its series: teacher pairs their series cover
  (data/enhanced/mangadex_bw manifest), MangaDex color pairs another colored page of the same title;
  --ref-drop of the samples get none (learned no-reference tokens), so the model works without one.
- --cls: a/b as a choice among color bins: cross-entropy on soft-encoded target bins, weighted so
  rare (saturated) colors count more (--rebalance), plus --ab-l1 x the old L1 on the decoded a/b.
"""

import argparse
import hashlib
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import io as mio  # noqa: E402
from models.augment import white_paper  # noqa: E402
from models.hints import sample_hints  # noqa: E402
from models.lab import merge_lab, rgb_to_lab_torch  # noqa: E402

VARIANTS = {"clean", "tone", "lineart", "bw"}
BW_MANIFESTS = ["data/enhanced/mangadex_bw/manifest.jsonl", "data/enhanced/mangadex_bw_pod/manifest.jsonl"]


class RefFinder:
    """name -> reference image path for that pair, or None (picklable for DataLoader workers).

    Teacher pairs (pages of mangadex_bw): the series cover, sometimes another colored page of the
    same series. MangaDex color pairs (<manga8>_<chapter8>_<i>): another colored page of the same
    title. Other sets (Commons, Openclipart): none.
    """

    def __init__(self, root, manifests=None):
        import json

        self.color_dir = os.path.join(root, "color")
        self.by_series = {}
        for f in os.listdir(self.color_dir):
            self.by_series.setdefault(f[:8], []).append(f)
        self.covers = {}
        for man in manifests or BW_MANIFESTS:
            if os.path.exists(man):
                rdir = os.path.join(os.path.dirname(man), "refs")
                for line in open(man, encoding="utf-8"):
                    d = json.loads(line)
                    self.covers[os.path.splitext(d["file"])[0]] = os.path.join(rdir, d["ref"])

    def __call__(self, name, rnd):
        page = os.path.splitext(page_of(name))[0]
        cover = self.covers.get(page)
        same = self.by_series.get(page[:8], []) if len(page) > 17 and page[8] == "_" else []
        others = [f for f in same if os.path.splitext(f)[0] != page] if len(same) > 1 else []
        if cover and os.path.exists(cover) and (not others or rnd.random() < 0.8):
            return cover
        if others:
            return os.path.join(self.color_dir, others[rnd.randrange(len(others))])
        return None


def ref_image(path, rnd, eval=False, side=256):
    """Reference image -> (3,side,side) float tensor in [0,1]; random square crop + flip in training."""
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    if not eval:
        c = int(min(H, W) * rnd.uniform(0.7, 1.0))
        y, x = rnd.randint(0, H - c), rnd.randint(0, W - c)
        img = img[y : y + c, x : x + c]
        if rnd.random() < 0.5:
            img = img[:, ::-1]
    img = cv2.resize(np.ascontiguousarray(img), (side, side), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)


def page_of(gray_name):
    """gray/<page>__<variant>.png (e.g. __clean, __tone, __lineart) all belong to color/<page>.png."""
    stem, ext = os.path.splitext(gray_name)
    base, _, variant = stem.rpartition("__")  # page names may contain '__' themselves
    return (base if base and variant in VARIANTS else stem) + ext


def split_pages(names, val_pct, seed=0):
    """Deterministic held-out split by PAGE (all variants of a page on the same side)."""
    is_val = lambda n: int(hashlib.md5(f"{seed}:{page_of(n)}".encode()).hexdigest(), 16) % 1000 < val_pct * 10
    return [n for n in names if not is_val(n)], [n for n in names if is_val(n)]


class Pairs(Dataset):
    """Random crops at roughly the scale the model sees at inference (long side ~ page_side).

    eval=True: deterministic (center crop at exactly page_side, no flips/augmentation, hints from a
    fixed per-item seed), for the held-out validation loss.
    """

    def __init__(
        self,
        root,
        crop=512,
        page_side=768,
        max_hints=16,
        names=None,
        eval=False,
        white_paper_p=0.0,
        refs=False,
        ref_self=0.0,
    ):
        self.gray_dir, self.color_dir = os.path.join(root, "gray"), os.path.join(root, "color")
        self.ref_finder = RefFinder(root) if refs else None
        self.ref_self = ref_self
        self.names = sorted(names if names is not None else os.listdir(self.gray_dir))
        self.crop, self.page_side, self.max_hints, self.eval = crop, page_side, max_hints, eval
        self.white_paper_p = white_paper_p

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        rnd = random.Random(i) if self.eval else random
        color = cv2.imdecode(np.fromfile(os.path.join(self.color_dir, page_of(n)), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.imdecode(np.fromfile(os.path.join(self.gray_dir, n), np.uint8), cv2.IMREAD_GRAYSCALE)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        # rescale the page so its long side is ~page_side (x0.8-1.25), then crop
        s = self.page_side * (1.0 if self.eval else rnd.uniform(0.8, 1.25)) / max(gray.shape)
        H, W = gray.shape
        h, w = max(self.crop, int(H * s)), max(self.crop, int(W * s))
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_AREA)
        if self.eval:
            y, x = (h - self.crop) // 2, (w - self.crop) // 2
        else:
            y, x = rnd.randint(0, h - self.crop), rnd.randint(0, w - self.crop)
        gray = gray[y : y + self.crop, x : x + self.crop].astype(np.float32) / 255
        color = color[y : y + self.crop, x : x + self.crop].astype(np.float32) / 255
        if not self.eval:
            if rnd.random() < 0.5:
                gray, color = gray[:, ::-1], color[:, ::-1]
            if rnd.random() < self.white_paper_p:
                gray = white_paper(gray, np.random.default_rng(rnd.randrange(2**31)))
            # input-only photometric augmentation: contrast + gamma
            if rnd.random() < 0.5:
                c = rnd.uniform(0.85, 1.15)
                gray = np.clip((gray - 0.5) * c + 0.5, 0, 1)
            if rnd.random() < 0.5:
                gray = gray ** rnd.uniform(0.8, 1.25)

        gray_t = torch.from_numpy(np.ascontiguousarray(gray))[None]
        color_t = torch.from_numpy(np.ascontiguousarray(color)).permute(2, 0, 1)
        # hints are always sampled here; the training loop drops them for whole batches
        if self.eval:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(i)
                hint = sample_hints(color_t, self.max_hints)
        else:
            hint = sample_hints(color_t, self.max_hints)
        if self.ref_finder is None:
            return torch.cat([gray_t, hint], 0), color_t
        if not self.eval and rnd.random() < self.ref_self:
            # the target page itself as reference (another random crop, maybe flipped): teaches the
            # model to take colors FROM the reference; at inference the series cover plays that role
            path = os.path.join(self.color_dir, page_of(n))
        else:
            path = self.ref_finder(n, rnd)
        if path is None:
            return torch.cat([gray_t, hint], 0), color_t, torch.zeros(3, 256, 256), torch.tensor(0.0)
        return torch.cat([gray_t, hint], 0), color_t, ref_image(path, rnd, self.eval), torch.tensor(1.0)


@torch.no_grad()
def save_samples(gen, val_paths, out_dir, step, device, target=None, ref_path=None):
    gen.eval()
    variants = [None]
    if getattr(gen, "use_ref", False) and ref_path:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            variants.append(gen.ref_tokens(mio.load_ref(ref_path, device)))
    rows = []
    for tok in variants:
        tiles = []
        for p in val_paths:
            gray = mio.read_gray(p)
            g, hw = mio.resize_for_model(gray, 768)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out, _ = gen(mio.make_input(g, device), tok)
            rgb = merge_lab(gray, mio.to_rgb01(out, hw))
            s = 700 / rgb.shape[0]
            tiles.append(cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        h = min(t.shape[0] for t in tiles)
        rows.append(np.concatenate([t[:h] for t in tiles], 1))
    sep = np.full((12, rows[0].shape[1], 3), 255, np.uint8)
    grid = rows[0]
    for r in rows[1:]:  # second row: with the series cover as color reference
        grid = np.concatenate([grid, sep, r], 0)
    if target is not None:  # the reference colorization of the same pages, as a second row
        t = cv2.resize(
            target,
            (grid.shape[1], int(target.shape[0] * grid.shape[1] / target.shape[1])),
            interpolation=cv2.INTER_AREA,
        )
        grid = np.concatenate([grid, np.full((12, grid.shape[1], 3), 255, np.uint8), t], 0)
    mio.write_rgb(os.path.join(out_dir, "samples", f"step{step:07d}.jpg"), grid)
    gen.train()


def soft_bins(ab, centers, sigma=5.0):
    """Target a/b (N,2,h,w) -> soft one-hot over the color bins (N,Q,h,w), Gaussian around the target."""
    d2 = ((ab[:, None] - centers[None, :, :, None, None]) ** 2).sum(2)  # N,Q,h,w
    w = torch.exp(-d2 / (2 * sigma**2))
    return w / w.sum(1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def bin_weights(ds, centers, n=400, lam=0.5, seed=0):
    """Rebalancing weights per bin, w ~ 1 / ((1-lam) p + lam / Q), normalised so E_p[w] = 1, from the
    a/b distribution of n random training targets."""
    rnd = random.Random(seed)
    hist = torch.zeros(len(centers))
    for i in rnd.sample(range(len(ds)), min(n, len(ds))):
        color = ds[i][1]
        ab = rgb_to_lab_torch(F.avg_pool2d(color[None], 4))[:, 1:]
        idx = ((ab[:, :, None] - centers.T[None, :, :, None, None]) ** 2).sum(1).argmin(1)
        hist += torch.bincount(idx.flatten(), minlength=len(centers)).float()
    p = hist / hist.sum()
    # smooth the histogram over neighbouring bins (5x5 grid neighbourhood)
    q = int(len(centers) ** 0.5)
    p = F.avg_pool2d(p.view(1, 1, q, q), 3, stride=1, padding=1, count_include_pad=False).flatten()
    p = p / p.sum()
    w = 1 / ((1 - lam) * p + lam / len(p))
    return w / (p * w).sum()


@torch.no_grad()
def validate(gen, dl, device):
    """Held-out a/b L1 (Lab units) without and with hints, plus the same within ~15 px of hint points."""
    gen.eval()
    sums = {
        "ab": 0.0,
        "ab_hint": 0.0,
        "near": 0.0,
        "near_hint": 0.0,
        "ab_noref": 0.0,
        "chroma_pred": 0.0,
        "chroma_tgt": 0.0,
    }
    n_px, n_near = 0, 0
    for batch in dl:
        x, target = batch[0].to(device), batch[1].to(device)
        tokens = None
        if getattr(gen, "use_ref", False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = gen.tokens_for(x.shape[0], gen.ref_tokens(batch[2].to(device)), batch[3].to(device))
        tgt = rgb_to_lab_torch(target)[:, 1:]
        near = F.max_pool2d(x[:, 4:5], 31, stride=1, padding=15) > 0
        no_hint = torch.cat([x[:, :1], torch.zeros_like(x[:, 1:])], 1)
        runs = [("", no_hint, tokens), ("_hint", x, tokens)]
        if tokens is not None:
            runs.append(("_noref", no_hint, None))
        for key, inp, tok in runs:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ab = gen.forward_lab(inp, tok).float()[:, 1:]
            err = (ab - tgt).abs().mean(1, keepdim=True)
            sums["ab" + key] += err.sum().item()
            if key in ("", "_hint"):
                sums["near" + key] += err[near].sum().item()
            if key == "":
                sums["chroma_pred"] += ab.norm(dim=1).sum().item()
                sums["chroma_tgt"] += tgt.norm(dim=1).sum().item()
        n_px += err.numel()
        n_near += near.sum().item()
    gen.train()
    return {
        "ab": sums["ab"] / n_px,
        "ab_hint": sums["ab_hint"] / n_px,
        "near": sums["near"] / max(n_near, 1),
        "near_hint": sums["near_hint"] / max(n_near, 1),
        "ab_noref": sums["ab_noref"] / n_px if sums["ab_noref"] else float("nan"),
        "chroma_ratio": sums["chroma_pred"] / max(sums["chroma_tgt"], 1e-6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs")
    ap.add_argument(
        "--mix",
        action="append",
        default=[],
        help="DIR:FRACTION, e.g. data/pairs:0.15 = 15%% of training samples from DIR (repeatable)",
    )
    ap.add_argument("--val", default="eval/testset/more-than-a-doll", help="real manga pages for sample grids")
    ap.add_argument(
        "--target",
        default="eval/reference/probe_pages_target.jpeg",
        help='reference colorization of the sample pages, shown under them ("" to skip)',
    )
    ap.add_argument(
        "--holdout-pct",
        type=float,
        default=5.0,
        help="%% of pages (all their variants) held out for the validation loss",
    )
    ap.add_argument("--holdout-max", type=int, default=300, help="max held-out samples per evaluation")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/run")
    ap.add_argument("--model", choices=["unet"], default="unet", help=argparse.SUPPRESS)
    ap.add_argument("--weights", help="start from this checkpoint (default: ImageNet encoder only)")
    ap.add_argument("--resume")
    ap.add_argument("--steps", type=int, default=20000, help="optimizer steps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation (effective batch = batch*accum)")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr-mult", type=float, default=0.1, help="encoder lr = lr * this")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--l-weight", type=float, default=0.1, help="weight of the L1 loss on predicted L")
    ap.add_argument("--white-paper", type=float, default=0.0, help="share of samples with white-paper input")
    ap.add_argument(
        "--hint-drop",
        type=float,
        default=0.5,
        help="share of batches trained with no hints at all, so the model works without them",
    )
    ap.add_argument("--max-hints", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument(
        "--refresh-every",
        type=int,
        default=0,
        help="rebuild the training list every N steps (pairs still being generated)",
    )
    ap.add_argument("--ref", action="store_true", help="color reference input (series cover)")
    ap.add_argument("--ref-drop", type=float, default=0.3, help="share of samples trained without a reference")
    ap.add_argument(
        "--ref-self",
        type=float,
        default=0.0,
        help="share of samples whose reference is a crop of their own target page",
    )
    ap.add_argument(
        "--sample-ref",
        default="data/enhanced/teacher_refs/bisque_vol12.jpg",
        help="reference for the sample pages (second sample row)",
    )
    ap.add_argument("--cls", action="store_true", help="a/b as color-bin classification")
    ap.add_argument("--rebalance", type=float, default=0.5, help="lambda of the rare-color weighting (1 = none)")
    ap.add_argument("--ab-l1", type=float, default=0.1, help="weight of the L1 on decoded a/b with --cls")
    ap.add_argument(
        "--white-paper-pairs",
        default="",
        help="comma list of pairs dirs that get --white-paper (default: all except teacher pairs)",
    )
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cudnn.benchmark = True
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    mixes = [(m.rsplit(":", 1)[0], float(m.rsplit(":", 1)[1])) for m in args.mix]

    # pairs folders containing enhanced.txt hold copyrighted pages for enhanced use only: mark the run
    args.enhanced_data = any(
        os.path.exists(os.path.join(d, "enhanced.txt")) for d in [args.pairs] + [m[0] for m in mixes]
    )
    if args.weights:
        prev = torch.load(args.weights, map_location="cpu", weights_only=False)
        args.enhanced_data |= bool(prev.get("args", {}).get("enhanced_data", False))
    if args.enhanced_data:
        print("NOTE: trained on enhanced data -> this checkpoint is for enhanced use only", flush=True)

    from models.unet import ColorUNet

    gen = ColorUNet(
        pretrained=True if (args.ref and args.weights) else not args.weights, ref=args.ref, cls=args.cls
    ).to(device)
    if args.weights:
        r = gen.load_state_dict(prev["generator"], strict=not (args.ref or args.cls))
        if not (args.ref or args.cls) or r.unexpected_keys:
            assert not r.unexpected_keys, r.unexpected_keys
        else:
            print(
                f"new parts (not in --weights): {len({k.split('.')[0] for k in r.missing_keys})} modules, "
                f"{len(r.missing_keys)} tensors",
                flush=True,
            )
    gen.train()
    if args.freeze_encoder:
        for p in gen.encoder.parameters():
            p.requires_grad_(False)
    params = [p for p in gen.parameters() if p.requires_grad]
    enc = {id(p) for p in gen.encoder.parameters()}
    if args.ref:
        enc |= {id(p) for p in gen.ref_encoder.parameters()}
    groups = [
        {"params": [p for p in params if id(p) not in enc], "lr": args.lr},
        {"params": [p for p in params if id(p) in enc], "lr": args.lr * args.encoder_lr_mult},
    ]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=args.steps, pct_start=0.03, anneal_strategy="cos"
    )
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck["args"]["steps"] != args.steps:
            raise SystemExit(
                f"--steps must match the checkpoint ({ck['args']['steps']}) to resume the LR "
                f"schedule; to train further, start a new run with --weights {args.resume}"
            )
        gen.load_state_dict(ck["generator"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"resumed at step {step}")

    wp_dirs = set(filter(None, args.white_paper_pairs.split(",")))
    wp = lambda d: args.white_paper if (d in wp_dirs if wp_dirs else "teacher" not in d) else 0.0

    def build_train():
        """Training loader over the pairs on disk now (--refresh-every rebuilds it while a teacher run
        keeps adding pairs). The page-hash split keeps held-out pages out of training every time."""
        train_names, val_names = split_pages(os.listdir(os.path.join(args.pairs, "gray")), args.holdout_pct)
        ds = Pairs(
            args.pairs,
            args.crop,
            max_hints=args.max_hints,
            names=train_names,
            white_paper_p=wp(args.pairs),
            refs=args.ref,
            ref_self=args.ref_self,
        )
        main_ds = ds
        sampler, shuffle = None, True
        if mixes:
            parts = [ds]
            main_frac = 1 - sum(f for _, f in mixes)
            weights = [main_frac / len(ds)] * len(ds)
            for mix_dir, mix_frac in mixes:
                mix_train, _ = split_pages(os.listdir(os.path.join(mix_dir, "gray")), args.holdout_pct)
                mix_ds = Pairs(
                    mix_dir,
                    args.crop,
                    max_hints=args.max_hints,
                    names=mix_train,
                    white_paper_p=wp(mix_dir),
                    refs=args.ref,
                    ref_self=args.ref_self,
                )
                parts.append(mix_ds)
                weights += [mix_frac / len(mix_ds)] * len(mix_ds)
                print(f"mixing in {len(mix_ds)} pairs from {mix_dir} at {mix_frac:.0%} of samples", flush=True)
            ds = ConcatDataset(parts)
            sampler = WeightedRandomSampler(weights, num_samples=max(len(train_names), 2000), replacement=True)
            shuffle = False
        dl = DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
        )
        return dl, main_ds, train_names, val_names

    dl, main_ds, train_names, val_names = build_train()
    with open(os.path.join(args.out, "holdout.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_names))
    val_ds = Pairs(
        args.pairs,
        args.crop,
        max_hints=args.max_hints,
        names=val_names[:: max(1, len(val_names) // args.holdout_max)][: args.holdout_max],
        eval=True,
        refs=args.ref,
    )
    cls_w = None
    if args.cls:
        cls_w = (
            bin_weights(main_ds, gen.centers.cpu(), lam=args.rebalance)
            if args.rebalance < 1
            else torch.ones(len(gen.centers))
        ).to(device)
        args.cls_weights = cls_w.tolist()
        print(f"color-bin weights: min {cls_w.min():.2f} max {cls_w.max():.2f}", flush=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=2)
    best = float("inf")
    val_paths = mio.list_images(args.val)
    val_paths = val_paths[:: max(1, len(val_paths) // args.n_val)][: args.n_val]
    target = None
    if args.target and os.path.exists(args.target):
        target = cv2.cvtColor(cv2.imdecode(np.fromfile(args.target, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    n_trainable = sum(p.numel() for p in params)
    print(
        f"{len(train_names)} training pairs ({len({page_of(n) for n in train_names})} pages), "
        f"{len(val_ds)} held-out ({len({page_of(n) for n in val_names})} pages), "
        f"{n_trainable / 1e6:.1f}M trainable params, effective batch {args.batch * args.accum}",
        flush=True,
    )

    def checkpoint(name):
        torch.save(
            {
                "generator": gen.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "step": step,
                "args": vars(args),
            },
            os.path.join(args.out, name),
        )

    t0, running, micro = time.time(), 0.0, 0
    last_refresh = step
    while step < args.steps:
        if args.refresh_every and step - last_refresh >= args.refresh_every:
            dl, _, train_names, _ = build_train()
            last_refresh = step
            print(f"refreshed: {len(train_names)} training pairs", flush=True)
        for batch in dl:
            x = batch[0].to(device, non_blocking=True)
            if random.random() < args.hint_drop:
                x[:, 1:] = 0
            target_rgb = batch[1].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = None
                if args.ref:
                    has = batch[3].to(device) * (torch.rand(x.shape[0], device=device) >= args.ref_drop)
                    tokens = gen.tokens_for(x.shape[0], gen.ref_tokens(batch[2].to(device, non_blocking=True)), has)
                pred, logits = gen.forward_lab(x, tokens, return_logits=True)
            tgt = rgb_to_lab_torch(target_rgb)
            if args.cls:
                # color-bin cross-entropy at the logits' resolution (stride 4), rare colors weighted up
                tgt_ab = rgb_to_lab_torch(F.avg_pool2d(target_rgb, 4))[:, 1:]
                soft = soft_bins(tgt_ab, gen.centers)
                w = cls_w[soft.argmax(1)]  # N,h,w
                ce = -(soft * F.log_softmax(logits.float(), 1)).sum(1)
                loss = (
                    (w * ce).mean()
                    + args.ab_l1 * F.l1_loss(pred[:, 1:], tgt[:, 1:]) / 110
                    + args.l_weight * F.l1_loss(pred[:, :1], tgt[:, :1]) / 100
                )
            else:
                # loss on a/b (the color we keep) + a little on L (only used by the white handling)
                loss = (
                    F.l1_loss(pred[:, 1:], tgt[:, 1:]) / 110 + args.l_weight * F.l1_loss(pred[:, :1], tgt[:, :1]) / 100
                )
            (loss / args.accum).backward()
            running += loss.item()
            micro += 1
            if micro % args.accum:
                continue

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                print(
                    f"step {step}/{args.steps}  loss {running / (args.log_every * args.accum):.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  {args.log_every / dt:.2f} it/s  "
                    f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                    flush=True,
                )
                running, t0 = 0.0, time.time()
            if len(val_ds) and (step % args.eval_every == 0 or step == args.steps):
                v = validate(gen, val_dl, device)
                print(
                    f"  held-out a/b L1: no hints {v['ab']:.2f}  with hints {v['ab_hint']:.2f}  |  "
                    f"near hint points: no hints {v['near']:.2f}  with hints {v['near_hint']:.2f}  |  "
                    f"no reference {v['ab_noref']:.2f}  |  colorfulness vs target {v['chroma_ratio']:.2f}",
                    flush=True,
                )
                with open(os.path.join(args.out, "val_log.csv"), "a", encoding="utf-8") as f:
                    f.write(
                        f"{step},{v['ab']:.4f},{v['ab_hint']:.4f},{v['near']:.4f},{v['near_hint']:.4f},"
                        f"{v['ab_noref']:.4f},{v['chroma_ratio']:.4f}\n"
                    )
                if v["ab"] < best:
                    best = v["ab"]
                    checkpoint("best.pt")
            if step % args.sample_every == 0 or step == args.steps:
                checkpoint("last.pt")
                save_samples(gen, val_paths, args.out, step, device, target, args.sample_ref)
            if step >= args.steps or (args.refresh_every and step - last_refresh >= args.refresh_every):
                break
    checkpoint("last.pt")
    print("done")


if __name__ == "__main__":
    main()
