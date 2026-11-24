# ColorUNet-Manga

Fast single-pass colorization of black-and-white manga pages, small enough to run in the browser
(WebGPU). The network only picks the colors; the page's own lightness is kept at full resolution, so
line art, screentones and text stay exactly as drawn. Given the series cover, it keeps characters'
hair, eye and clothing colors consistent across pages.

- **Weights:** [PianoNic/ColorUNet-Manga-61M](https://huggingface.co/PianoNic/ColorUNet-Manga-61M)
  (PyTorch safetensors, ONNX fp16/fp32 for WebGPU)

## How it works

1. The page goes into the network at a long side of 768 px, optionally with color hints
   (`{x, y, color}` points) and the series cover as a reference.
2. The network (ConvNeXt-Tiny encoder, U-Net decoder, 33M parameters, plus a 28M cover encoder)
   predicts Lab a/b color.
3. The color is upsampled edge-aware to the full page; the page's own L channel is kept, chroma fades
   out on ink and weak tints fade out on paper white, out-of-gamut colors lose chroma, never lightness.

For the browser, steps 2 and 3 are one ONNX graph (`training/export/to_web.py`).

## Repository

| Path | What |
| --- | --- |
| [`training/`](training/README.md) | Model, data pipeline, training, export, evaluation, tests |
| [`training/DATASETS.md`](training/DATASETS.md) | Every training source, its licence, what was rejected and why |
| [`docs/training-history.md`](docs/training-history.md) | The training stages and their numbers |
| [`models/NOTICE.md`](models/NOTICE.md) | Attribution for the released weights |

## Training data

The released weights were trained on public-domain comics, CC0 clipart and **copyrighted manga pages
from MangaDex** (and teacher colorings of them). The weights contain no page images and the manga data
is not redistributed. Whether such training is permitted depends on your jurisdiction; see
[`training/DATASETS.md`](training/DATASETS.md). A license-clean model can be rebuilt from stage 1.

## Licence

Code and weights: MIT, see [LICENSE](LICENSE). Third-party components and data: [`models/NOTICE.md`](models/NOTICE.md).
