<div align="center">

# ColorUNet-Manga

**Fast single-pass manga colorization. Runs in the browser (WebGPU).**

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-ColorUNet--Manga--61M-yellow)](https://huggingface.co/PianoNic/ColorUNet-Manga-61M)
[![CI](https://github.com/PianoNic/ColorUNet-Manga/actions/workflows/training-ci.yml/badge.svg)](https://github.com/PianoNic/ColorUNet-Manga/actions/workflows/training-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

![Black-and-white input and ColorUNet-Manga output](assets/demo.jpg)
<sub>Public-domain pages never seen in training (Wikimedia Commons): Wally Wood, <i>Eerie</i> No. 2 (1951); Sheldon Moldoff, <i>The Dead Lover Returns</i>.</sub>

</div>

Line art, screentones and text stay pixel-exact: the network only predicts color, the page keeps its own
lightness. Optional color hints and a series cover keep characters' colors consistent across pages.

## Quick start

```bash
git clone https://github.com/PianoNic/ColorUNet-Manga && cd ColorUNet-Manga/training
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or /cpu
pip install -r requirements.txt
hf download PianoNic/ColorUNet-Manga-61M --local-dir weights

python scripts/infer.py page.png --weights weights/model.safetensors --sbs
python scripts/infer.py chapter/ --weights weights/model.safetensors --ref cover.jpg   # with the series cover
```

## Models

| Model | Params | Input | Download |
| --- | --- | --- | --- |
| ColorUNet-Manga-61M | 33M colorizer + 28M cover encoder | page (long side 768), optional hints and cover | [🤗 Hugging Face](https://huggingface.co/PianoNic/ColorUNet-Manga-61M) (safetensors, ONNX fp16/fp32) |

## Training

Data pipeline, staged training, ONNX export and evaluation: [`training/`](training/README.md).
Sources and licences: [`training/DATASETS.md`](training/DATASETS.md). The released weights were trained in
part on copyrighted manga pages, which are not redistributed.

## License

MIT. Third-party components and training data: [`models/NOTICE.md`](models/NOTICE.md).
