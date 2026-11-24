# ColorUNet-Manga: training

Training, export and evaluation for ColorUNet-Manga, a fast single-pass manga colorizer, with the Lab
trick: the model picks color at ~768 px, the original full-resolution page keeps its lightness, so lines
and text stay exact. Released weights: [PianoNic/ColorUNet-Manga-61M](https://huggingface.co/PianoNic/ColorUNet-Manga-61M).
Training data and licences: [DATASETS.md](DATASETS.md). How the stages came about: [../docs/training-history.md](../docs/training-history.md).

Data, checkpoints and exports live next to the code in this folder but are not in git (see `.gitignore`):
`data/pairs*/`, `data/raw*/`, `data/enhanced/`, `checkpoints/`, `eval/testset*/`, `export/web*/`.

## Model

`models/unet.py`, ColorUNet: ConvNeXt-Tiny encoder, U-Net decoder, 4 color-hint channels, Lab output
(32M parameters). Its only pretrained part is the timm encoder
[`convnext_tiny.fb_in1k`](https://huggingface.co/timm/convnext_tiny.fb_in1k): weights released
under **Apache-2.0** (timm model card; ConvNeXt code is MIT, https://github.com/facebookresearch/ConvNeXt).
Caveat: those weights were trained on ImageNet-1k, whose image terms are non-commercial research;
weights trained on it are widely used commercially, but that is not settled law. If this matters
for the platform, the encoder can be swapped for one trained on licensed data, or the U-Net trained
with `pretrained=False` (more data needed).

**manga-colorization-v2 is no longer used** (removed 2026-10-13): its code, weights, the FFDNet
denoiser that came with it, its baseline script and all outputs made with it are deleted. Nothing in
this project derives from it, so no permission from its author is needed.

| Checkpoint | Data | Status |
| --- | --- | --- |
| `checkpoints/stage1_commons/best.pt` (default) | public-domain comics (Commons), 20k steps | license-clean |
| `checkpoints/stage2_manga/best.pt` | stage 1 + MangaDex manga pages (+15% Commons), white-paper augmentation, 55k steps (best at 42.5k) | enhanced data (see DATASETS.md) |
| `checkpoints/stage3c/step10000.pt` (**released**) | stage 3b snapshot + FLUX.2 [klein] 4B teacher colorings of MangaDex pages (+25% stage-2 pages), series-cover reference encoder and color-class head, 10k of 18k steps | enhanced data; [ColorUNet-Manga-61M](https://huggingface.co/PianoNic/ColorUNet-Manga-61M) |

## Setup

```bash
python -m venv --system-site-packages .venv      # reuses the global torch 2.10 + CUDA 12.8
.venv/Scripts/python -m pip install -r requirements.txt
```

All packages go into `.venv`, never the global Python (`onnx` needs protobuf >= 4.25; the global
install keeps protobuf 3.20.1 for other tools).

## Pipeline (scripts/infer.py)

1. Read the page at full resolution; its L (lightness) is the output's L.
2. `--descreen` (optional): smooth screentone regions of L; lines, text, punctuation untouched.
3. Model input: page at `--long-side` 768. `--palette` adds color hints.
4. Keep only the model's Lab a/b; edge-aware (guided) upsampling to full resolution.
5. White/black cleanup: chroma fades out on ink; on paper-white, weak tints fade out (no yellow
   cast) while real colors stay (skin and light hair are bare paper in manga), with L lowered there
   by at most 14 so the color can show. Out-of-gamut colors lose chroma, never lightness.

Lines, text, tones and shading keep the original L exactly: `tests/test_lightness.py`.
The browser version (`export/to_web.py`) puts steps 3-5 into one ONNX graph for WebGPU.

## Scripts

| Step | Command |
| --- | --- |
| Test pages (MangaDex, as a reader app shows them) | `python data/fetch_mangadex.py --manga-id aa6c76f7-5f5f-46b6-a800-911145f81b9b --lang es,pt-br,es-la,ru --chapters 3` |
| Colorize | `python scripts/infer.py eval/testset/more-than-a-doll --sbs [--descreen] [--palette p.json] [--weights ckpt.pt]` |
| Commons public-domain comics | `python data/fetch_commons.py --size-only`, then `--download`; pairs: `python data/make_pairs.py data/raw_color/commons/pages --out data/pairs --variants clean,tone --clean-scans` |
| Openclipart (CC0) | `python data/openclipart.py fetch --ids-file ids.txt`, then `python data/openclipart.py pairs --out data/pairs` |
| Clean scans only (preview) | `python data/clean_scans.py <scan or folder> --out <dir> --compare` |
| Training pairs (other sources) | `python data/make_pairs.py data/raw_color --out data/pairs` |
| Stage 1 (5080) | `python scripts/finetune.py --pairs data/pairs --out checkpoints/stage1_commons` |
| Stage 2 (5080) | `python scripts/finetune.py --pairs data/pairs_enhanced --mix data/pairs:0.15 --weights checkpoints/stage1_commons/best.pt --lr 1e-4 --white-paper 0.5 --out checkpoints/stage2_manga` |
| Hint test (blonde/pink/blue) | `python eval/hint_test.py --weights <checkpoint> --out eval/compare/hints.jpg` |
| Browser model (WebGPU) | `python export/to_web.py --weights <checkpoint> --out export/web` (enhanced checkpoints need `--allow-enhanced`); the files in `export/web/` are what the Hugging Face repo ships |
| ONNX / TensorRT | `python export/to_onnx.py --weights <ckpt> --out export/colorunet.onnx`; on the 4070: `--onnx export/colorunet.onnx --trt export/colorunet_4070.engine` |
| Benchmark | `python scripts/benchmark.py eval/testset/more-than-a-doll [--trt ...]` |
| Tests | `python -m pytest tests/ -q` |
| Crop comparisons | `python eval/compare_crops.py <page> --cols label:folder ... --boxes x,y ... --out eval/compare/x.jpg` |

## Training

`scripts/finetune.py`: L1 on Lab a/b (+ a little on L), bf16, batch 8 x accum 2 at 512 px crops
(9.2 GB VRAM, ~3.5-4 steps/s on the 5080). Every sample gets 1-16 random color hints from its target;
half the batches train without hints. `--white-paper`: light grays of the input are lifted to pure
white (`models/augment.py`) so white skin/hair areas still get color. `--mix DIR:FRACTION` mixes
another pairs set in (stage 2 keeps 15% Commons so general skills are not forgotten).
Held-out validation: `--holdout-pct` (5%) of the pages, all variants of a page on the same side,
logged to `val_log.csv`; `best.pt` = lowest held-out a/b error. Sample grids show the four probe
pages with the reference colorization (`eval/reference/probe_pages_target.jpeg`) underneath.

## Color hints

`models/hints.py`: points `{x, y, color}` (x, y as fractions of the page) in a palette JSON, per
page name or for all pages. Points are tied to page positions; applying one series palette
automatically to new pages would need a character/region detector to place them.

## Training data

Stage 1 uses only sources whose license allows training and commercial use (below); every file's source
and license is recorded in a manifest next to it. Stages 2 and 3 add MangaDex manga pages and teacher
colorizations of them, which are not redistributable: see [DATASETS.md](DATASETS.md).

| Source | License | How it becomes pairs |
| --- | --- | --- |
| Wikimedia Commons, Category:Public domain comics (Golden Age US comics), 3,357 files, 2,459 in color | Public domain; only files whose Commons license metadata says so (`manifest.jsonl`: page, license, author, category); trademarked characters (Mickey, Popeye, Garfield, Tarzan, Captain Marvel, ...) excluded | `--clean-scans`: paper white-balanced, halftone dots averaged out by downscaling to the dot period (detected from the page's spectrum); then gray / screentone variants |
| Openclipart | CC0 1.0 (checked per item) | SVG rendered twice: color, and the same SVG rewritten into exact black line art (`__lineart`, kept only if it covers the drawing's edges), plus gray / screentone variants. Titles/tags naming copyrighted characters, franchises, brands, logos, or AI generation are dropped (`BLOCKLIST` in `data/openclipart.py`) |
| Creators of the platform | Signed permission (`data/creator_intake.md`) | as delivered |

Access notes: Commons limits clients without contact info in the User-Agent to 10 API requests/min
(set `COMMONS_CONTACT` to an email or URL for 200/min). Openclipart bulk listing needs an API account
and app token (https://openclipart.org/api/tutorial); single items are fetched from public pages.

## Removed data: Pepper&Carrot

Pepper&Carrot (David Revoy, CC BY 4.0) was our first training source. On 2026-10-08 the author added
"AI derivation prohibited" to all his art and comics, applied retroactively
([post](https://www.davidrevoy.com/article1178/license-update-ai-derivation-prohibited-on-all-my-art-lore-stories-and-comics),
[license page](https://www.peppercarrot.com/en/license/index.html)). On 2026-10-12 we stopped using it
and deleted the downloaded pages, the 564 training pairs, the ColorUNet checkpoint trained on them
(stopped at step 1,000) and its outputs. `data/fetch_peppercarrot.py` is kept for reference and
refuses to run.

## Measured (RTX 5080)

| | |
| --- | --- |
| Full pipeline (PyTorch, 768 px model input) | ~0.1 s per page |
| Browser (ONNX Runtime Web, WebGPU, fp16, while training shared the GPU) | ~1.5-1.9 s per page |
| Stage 1 held-out a/b L1 | 10.30 (step 1k) -> 8.80 (step 18k, best) |
| Stage 2 held-out a/b L1 (enhanced manga pages) | 7.84 (step 2.5k) -> 7.31 (step 42.5k, best) |
| Hint test, stage 1 -> stage 2 (blonde / pink / blue) | 14% / 2% / 3% -> 49% / 26% / 22% of the way |

## Known issues

- Skin and light hair that are bare paper in the original stay mostly white with stage 1 (training
  pairs are converted color pages, where skin is light gray). Stage 2 (white-paper augmentation,
  real manga) adds blush and light tints and removes stage 1's smears, but skin/blonde hair are still
  much paler than the reference (`eval/compare/probe_stage1_stage2_target.jpg`).
- Backgrounds mostly get weak or patchy color; colors can drift between pages until hints are used.
- Hints work only weakly on white-paper areas (stage 1: 2-14% of the way in `eval/hint_test.py`).
- Descreen leaves a faint dot grid in the darkest tone areas, and toned regions get muted color.
