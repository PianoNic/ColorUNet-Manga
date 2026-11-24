"""Page I/O, model-input preparation and the model loader (ColorUNet only, no third-party code)."""

import os

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(ROOT, "checkpoints", "stage1_commons", "best.pt")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def load_model(weights: str = DEFAULT_WEIGHTS, device: str = "cuda"):
    """Load a ColorUNet checkpoint written by scripts/finetune.py."""
    import sys

    from models.unet import ColorUNet

    ck = torch.load(weights, map_location="cpu", weights_only=False)
    if ck.get("args", {}).get("enhanced_data"):
        print(
            f"WARNING: {weights} was trained on enhanced data: enhanced use only, not for the platform", file=sys.stderr
        )
    a = ck.get("args", {})
    net = ColorUNet(pretrained=False, ref=bool(a.get("ref")), cls=bool(a.get("cls")))
    net.load_state_dict(ck["generator"])
    return net.to(device).eval()


def load_ref(path_or_rgb, device: str):
    """Color reference (cover / colored page) -> (1,3,256,256) float in [0,1] for ColorUNet.ref_tokens."""
    from models.unet import REF_SIDE

    if isinstance(path_or_rgb, str):
        img = cv2.imdecode(np.fromfile(path_or_rgb, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = path_or_rgb
    img = cv2.resize(img, (REF_SIDE, REF_SIDE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)[None].to(device)


def resize_for_model(gray: np.ndarray, long_side: int):
    """Scale so the long side is ~long_side, pad (white) to multiples of 32.

    Returns (gray_resized_padded float32 HxW in [0,1], (h, w) of the unpadded content).
    """
    H, W = gray.shape
    s = long_side / max(H, W)
    h, w = max(32, int(round(H * s))), max(32, int(round(W * s)))
    g = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    g = g.astype(np.float32) / 255.0 if g.dtype == np.uint8 else g.astype(np.float32)
    ph, pw = (-h) % 32, (-w) % 32
    g = np.pad(g, ((0, ph), (0, pw)), mode="constant", constant_values=1.0)
    return g, (h, w)


def make_input(gray01: np.ndarray, device: str, hint: torch.Tensor = None) -> torch.Tensor:
    """Model input (1,5,H,W): gray in [0,1] + 4 hint channels (RGB*mask in [-1,1], mask)."""
    x = torch.from_numpy(gray01)[None, None].to(device)
    if hint is None:
        hint = torch.zeros(1, 4, *gray01.shape, device=device)
    return torch.cat([x, hint], 1)


def to_rgb01(out: torch.Tensor, hw) -> np.ndarray:
    """Model RGB output (1,3,H,W in [-1,1]) -> cropped HxWx3 float32 RGB in [0,1]."""
    h, w = hw
    rgb = (out[0, :, :h, :w].float() * 0.5 + 0.5).clamp(0, 1)
    return rgb.permute(1, 2, 0).cpu().numpy()


def read_gray(path: str) -> np.ndarray:
    """Read any image as uint8 grayscale (handles unicode paths on Windows)."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read {path}")
    if img.dtype != np.uint8:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 3:
        if img.shape[2] == 4:  # alpha over white
            a = img[..., 3:4].astype(np.float32) / 255
            img = (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    return img


def write_rgb(path: str, rgb_u8: np.ndarray):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"cannot encode {path}")
    buf.tofile(path)


def list_images(folder: str):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(folder, f))
    )


def side_by_side(gray_u8, rgb_u8, max_h=1600):
    left = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
    pair = np.concatenate([left, rgb_u8], axis=1)
    if pair.shape[0] > max_h:
        s = max_h / pair.shape[0]
        pair = cv2.resize(pair, (int(pair.shape[1] * s), max_h), interpolation=cv2.INTER_AREA)
    return pair
