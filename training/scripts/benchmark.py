"""Benchmark ms per page for each backend: model time and full pipeline (resize + model + Lab merge).

  python scripts/benchmark.py eval/testset/more-than-a-doll --onnx export/colorunet.onnx
  python scripts/benchmark.py pages/ --onnx export/colorunet.onnx --trt export/colorunet_4070.engine   # on the 4070

Backends: torch-cuda fp32/fp16, onnxruntime CPU (and CUDA if onnxruntime-gpu is installed), TensorRT.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import io as mio  # noqa: E402
from models.lab import merge_lab  # noqa: E402


def torch_runner(weights, fp16):
    gen = mio.load_model(weights, "cuda")

    @torch.no_grad()
    def run(x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            out = gen(torch.from_numpy(x).cuda())[0]
        return out.float().cpu().numpy()

    return run


def ort_runner(path, provider, threads=None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, so, providers=[provider])
    return lambda x: sess.run(None, {"x": x})[0]


def trt_runner(engine_path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()

    def run(x):
        xin = torch.from_numpy(x).cuda()
        ctx.set_input_shape("x", tuple(xin.shape))
        out = torch.empty((1, 3, *xin.shape[2:]), device="cuda", dtype=torch.float32)
        ctx.set_tensor_address("x", xin.data_ptr())
        ctx.set_tensor_address("rgb", out.data_ptr())
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return out.cpu().numpy()

    return run


def bench(name, run, pages, long_side, repeats):
    model_ms, total_ms = [], []
    for k in range(repeats + 1):
        for gray in pages:
            t0 = time.perf_counter()
            g, hw = mio.resize_for_model(gray, long_side)
            x = np.concatenate([g[None, None], np.zeros((1, 4, *g.shape), np.float32)], 1)
            t1 = time.perf_counter()
            out = run(x)
            t2 = time.perf_counter()
            rgb = np.clip(out[0, :, : hw[0], : hw[1]].transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)
            merge_lab(gray, rgb.astype(np.float32))
            t3 = time.perf_counter()
            if k > 0:  # first pass = warm-up
                model_ms.append((t2 - t1) * 1000)
                total_ms.append((t3 - t0) * 1000)
    print(
        f"{name:<16} model {np.median(model_ms):7.1f} ms   full page {np.median(total_ms):7.1f} ms "
        f"(median of {len(total_ms)})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--weights", default=mio.DEFAULT_WEIGHTS)
    ap.add_argument("--onnx", default="export/mio.onnx")
    ap.add_argument("--trt", help="TensorRT engine built on this GPU")
    ap.add_argument("--long-side", type=int, default=768)
    ap.add_argument("--pages", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cpu-threads", type=int)
    ap.add_argument("--skip-cpu", action="store_true")
    args = ap.parse_args()

    pages = [mio.read_gray(p) for p in mio.list_images(args.input)[: args.pages]]
    print(f"{len(pages)} pages, ~{pages[0].shape[1]}x{pages[0].shape[0]}, model input long side {args.long_side}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        bench("torch fp32", torch_runner(args.weights, False), pages, args.long_side, args.repeats)
        bench("torch fp16", torch_runner(args.weights, True), pages, args.long_side, args.repeats)
    import onnxruntime as ort

    if os.path.exists(args.onnx):
        if "CUDAExecutionProvider" in ort.get_available_providers():
            bench("onnxrt cuda", ort_runner(args.onnx, "CUDAExecutionProvider"), pages, args.long_side, args.repeats)
        if not args.skip_cpu:
            bench(
                "onnxrt cpu",
                ort_runner(args.onnx, "CPUExecutionProvider", args.cpu_threads),
                pages,
                args.long_side,
                max(1, args.repeats - 1),
            )
    if args.trt:
        bench("tensorrt fp16", trt_runner(args.trt), pages, args.long_side, args.repeats)


if __name__ == "__main__":
    main()
