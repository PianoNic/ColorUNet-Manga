"""Teacher colorization with FLUX.2 [klein] (local, image editing with a color reference).

  python data/teacher/klein_colorize.py PAGES... --ref data/enhanced/teacher_refs/bisque_vol12.jpg --out DIR
  python data/teacher/klein_colorize.py data/enhanced/mangadex_bw/pages       --manifest data/enhanced/mangadex_bw/manifest.jsonl --out data/enhanced/teacher_klein4b

Each black-and-white page is passed as image 1, a colored reference of the same series (cover or
colored page) as image 2, so every page gets the same character colors. With --manifest (from
data/enhanced_sources/mangadex_bw.py) every page uses its own series' cover. Pages already done are
skipped, so an interrupted run can be restarted. Output: DIR/<name>.png at the
generation size (resize/align to the original afterwards; only its colors are used for training).
Outputs of commercial manga pages are enhanced data (see README, enhanced data rules).
"""

import argparse
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models import io as mio  # noqa: E402

PROMPT = (
    "Colorize image 1, a black-and-white manga page, exactly as it is drawn, like an official "
    "full-color manga edition. Soft, natural, slightly muted pastel colors and gentle cel shading. "
    "Natural skin tone on every face, neck and hand. Hair, eye and clothing colors of the characters "
    "as in image 2. Do not add anything new: no sky, gradients, sparkles or scenery that is not drawn "
    "in image 1; empty white areas, white backgrounds and speech bubbles stay white; gray clothes stay "
    "gray, black stays black. Keep line art, faces and text unchanged."
)


def gen_size(w, h, area=1024 * 1024, mult=16):
    s = (area / (w * h)) ** 0.5
    return max(mult, round(w * s / mult) * mult), max(mult, round(h * s / mult) * mult)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", help="page files or folders")
    ap.add_argument("--ref", help="colored reference image (all pages)")
    ap.add_argument("--manifest", help="mangadex_bw manifest.jsonl: per-page series cover as reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--area", type=int, default=640 * 912, help="generation pixel count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--offload", action="store_true", help="model CPU offload (slow)")
    ap.add_argument("--shard", default="0/1", help="K/N: this process takes every N-th page from K")
    ap.add_argument("--reverse", action="store_true", help="work from the end of the page list")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp8", action="store_true", help="keep transformer weights in fp8 (for 9B on 16 GB)")
    args = ap.parse_args()

    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    # the prompt is the same for every page: encode it once, then free the text encoder's VRAM
    # (transformer + text encoder together do not fit in 16 GB and spill into slow shared memory)
    te_gb = sum(p.numel() * p.element_size() for p in pipe.text_encoder.parameters()) / 2**30
    te_dev = "cuda" if te_gb < 10 else "cpu"  # 9B's 16 GB text encoder: run it once on the CPU
    pipe.text_encoder.to(te_dev)
    with torch.no_grad():
        enc = pipe.encode_prompt(args.prompt, device=te_dev)
    prompt_embeds = (enc[0] if isinstance(enc, tuple) else enc).to("cuda")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    torch.cuda.empty_cache()
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        if args.fp8:  # store weights in fp8, compute in bf16 (9B: ~17 GB -> ~9 GB VRAM)
            pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
    refs = {}

    def load_ref(path):
        if path not in refs:
            im = Image.open(path).convert("RGB")
            im.thumbnail((1024, 1024))
            refs.clear()  # pages are grouped by series: keep one cover in memory
            refs[path] = im
        return refs[path]

    ref_of = {}
    if args.manifest:
        import json

        rdir = os.path.join(os.path.dirname(args.manifest), "refs")
        for line in open(args.manifest, encoding="utf-8"):
            d = json.loads(line)
            ref_of[d["file"]] = os.path.join(rdir, d["ref"])
    elif not args.ref:
        raise SystemExit("--ref or --manifest is required")
    paths = []
    for p in args.pages:
        paths += [p] if os.path.isfile(p) else mio.list_images(p)
    if args.reverse:
        paths = paths[::-1]
    k, n = map(int, args.shard.split("/"))
    paths = paths[k::n]
    os.makedirs(args.out, exist_ok=True)

    times = []
    for i, p in enumerate(paths):
        name = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(args.out, name + ".png")
        if os.path.exists(dst):
            continue
        ref_path = ref_of.get(os.path.basename(p), args.ref)
        if not ref_path:
            print(f"{name}: no reference, skipped", flush=True)
            continue
        ref = load_ref(ref_path)
        page = Image.fromarray(mio.read_gray(p)).convert("RGB")
        w, h = gen_size(*page.size, area=args.area)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = pipe(
            image=[page.resize((w, h), Image.LANCZOS), ref],
            prompt_embeds=prompt_embeds,
            height=h,
            width=w,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator("cuda").manual_seed(args.seed + i),
        ).images[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i > 0:
            times.append(dt)
        img.save(dst + ".tmp.png")
        os.replace(dst + ".tmp.png", dst)  # never leave a half-written page
        print(f"{name}: {w}x{h} {dt:.2f} s", flush=True)
    if times:
        print(f"mean {sum(times) / len(times):.2f} s/page (first excluded)")


if __name__ == "__main__":
    main()
