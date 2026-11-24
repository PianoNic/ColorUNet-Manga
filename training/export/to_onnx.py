"""Export the ColorUNet to ONNX (model only), and (on the deployment GPU) build a TensorRT fp16 engine.
For the browser (model + merge in one graph) use export/to_web.py instead.

  # on any machine (e.g. the 5080 after training):
  python export/to_onnx.py --weights checkpoints/stage1_commons/best.pt --out export/colorunet.onnx
  # on the RTX 4070 itself (engines are tied to the GPU architecture; only the .onnx moves):
  python export/to_onnx.py --onnx export/colorunet.onnx --trt export/colorunet_4070.engine

Graph: input "x" (1, 5, H, W) float32 = gray in [0,1] + 4 hint channels; H, W multiples of 32.
       output "rgb" (1, 3, H, W) in [-1, 1]. Lab merge stays outside the graph (scripts/infer.py).
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import io as mio  # noqa: E402


class MainOutput(torch.nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, x):
        return self.gen(x)[0]


def export_onnx(weights, path, h=1088, w=768, opset=17):
    gen = mio.load_model(weights, "cpu")
    model = MainOutput(gen).eval()
    x = torch.rand(1, 5, h, w)
    x[:, 1:] = 0
    torch.onnx.export(
        model,
        (x,),
        path,
        input_names=["x"],
        output_names=["rgb"],
        opset_version=opset,
        dynamic_axes={"x": {2: "H", 3: "W"}, "rgb": {2: "H", 3: "W"}},
        dynamo=False,
    )
    # verify against PyTorch on a different size than the trace
    import onnxruntime as ort

    xt = torch.rand(1, 5, 768, 544)
    xt[:, 1:] = 0
    with torch.no_grad():
        ref = model(xt).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"x": xt.numpy()})[0]
    print(f"wrote {path} ({os.path.getsize(path) / 2**20:.0f} MB); max |onnx - torch| = {np.abs(got - ref).max():.2e}")


def build_trt(onnx_path, engine_path, min_hw=(256, 256), opt_hw=(1088, 768), max_hw=(1600, 1600)):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1, 5, *min_hw), (1, 5, *opt_hw), (1, 5, *max_hw))
    config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit("TensorRT build failed")
    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"wrote {engine_path} for {torch.cuda.get_device_name(0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=mio.DEFAULT_WEIGHTS)
    ap.add_argument("--out", default="export/colorunet.onnx")
    ap.add_argument("--onnx", help="skip export, use this existing .onnx")
    ap.add_argument("--trt", help="also build a TensorRT fp16 engine here (run on the deploy GPU)")
    ap.add_argument("--allow-enhanced", action="store_true", help="export a checkpoint trained on enhanced data")
    args = ap.parse_args()

    onnx_path = args.onnx or args.out
    if not args.onnx and os.path.exists(args.weights):
        ck = torch.load(args.weights, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and ck.get("args", {}).get("enhanced_data") and not args.allow_enhanced:
            raise SystemExit(
                f"{args.weights} was trained on enhanced data (copyrighted pages, enhanced use only); "
                "refusing to export it for deployment. --allow-enhanced exports for enhanced use."
            )
    if not args.onnx:
        export_onnx(args.weights, onnx_path)
    if args.trt:
        build_trt(onnx_path, args.trt)


if __name__ == "__main__":
    main()
