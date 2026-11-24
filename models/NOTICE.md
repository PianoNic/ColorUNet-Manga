# Model weights

The weights are not in this repository: they are on Hugging Face,
[PianoNic/ColorUNet-Manga-61M](https://huggingface.co/PianoNic/ColorUNet-Manga-61M) (PyTorch
safetensors and ONNX for WebGPU). They are released under the MIT licence of this repository.

What went into them, and their terms:

- **Encoder initialisation:** timm `convnext_tiny.fb_in1k`, weights Apache-2.0, ConvNeXt code MIT
  (https://github.com/facebookresearch/ConvNeXt). Pretrained on ImageNet-1k.
- **Wikimedia Commons, Category:Public domain comics:** public domain, per each file's licence metadata.
- **Openclipart:** CC0 1.0.
- **MangaDex manga pages** (stages 2 and 3): copyrighted by their publishers and authors; used for
  training only, not redistributed.
- **FLUX.2 [klein] 4B** (Black Forest Labs, Apache-2.0): used as a teacher to color the stage-3 pages.

Details per source: [`training/DATASETS.md`](../training/DATASETS.md).
