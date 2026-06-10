#!/usr/bin/env python3
"""
DA3 PyTorch reference inference using the official depth_anything_3 package.

This script intentionally avoids the old handwritten DA3 model clone. The CPU
and CUDA runners in this repository track the official DA3 modules, so the
reference should do the same.

Usage:
    uv run python run_reference.py \
        --model-dir /mnt/disk01/models/da3-small \
        --image input.ppm \
        --output-dir output/ \
        --device cuda
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from depth_anything_3.api import DepthAnything3


def infer_model_name(model_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = model_dir.name.lower()
    if "small" in name:
        return "da3-small"
    if "base" in name:
        return "da3-base"
    if "giant" in name:
        return "da3-giant"
    if "large" in name:
        return "da3-large"
    return "da3-small"


def load_local_weights(model: DepthAnything3, model_dir: Path) -> None:
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        candidates = sorted(model_dir.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No safetensors file found in {model_dir}")
        st_path = candidates[0]

    state = load_file(str(st_path), device="cpu")
    state = {key.removeprefix("model."): value for key, value in state.items()}
    result = model.model.load_state_dict(state, strict=False)

    allowed_missing = {
        "head.scratch.output_conv2_aux.1.2.weight",
        "head.scratch.output_conv2_aux.1.2.bias",
        "head.scratch.output_conv2_aux.2.2.weight",
        "head.scratch.output_conv2_aux.2.2.bias",
        "head.scratch.output_conv2_aux.3.2.weight",
        "head.scratch.output_conv2_aux.3.2.bias",
    }
    unexpected = list(result.unexpected_keys)
    missing = [key for key in result.missing_keys if key not in allowed_missing]

    print(f"Loaded weights: {st_path}")
    print(f"  missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    if missing:
        print("WARNING: unexpected missing keys:")
        for key in missing[:20]:
            print(f"  {key}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    if unexpected:
        print("WARNING: unexpected extra keys:")
        for key in unexpected[:20]:
            print(f"  {key}")
        if len(unexpected) > 20:
            print(f"  ... and {len(unexpected) - 20} more")


def preprocess_square(image_path: Path, img_size: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
    img = np.array(Image.open(image_path).convert("RGB"))
    orig_h, orig_w = img.shape[:2]

    x = torch.from_numpy(img).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=True)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1)
    x = (x - mean) / std

    # Official DA3 expects [B, N, 3, H, W].
    return x.unsqueeze(0).to(device), orig_h, orig_w


def upsample_to_original(x: torch.Tensor, orig_h: int, orig_w: int) -> np.ndarray:
    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        x = x[0]
    y = F.interpolate(
        x.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=True,
    ).squeeze()
    return y.detach().cpu().numpy().astype(np.float32)


def write_pgm16(path: Path, depth: np.ndarray) -> None:
    d = depth.astype(np.float32)
    mn = float(np.nanmin(d))
    mx = float(np.nanmax(d))
    if mx <= mn:
        scaled = np.zeros(d.shape, dtype=np.uint16)
    else:
        scaled = np.clip((d - mn) / (mx - mn) * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    with path.open("wb") as f:
        f.write(f"P5\n{d.shape[1]} {d.shape[0]}\n65535\n".encode("ascii"))
        f.write(scaled.byteswap().tobytes())


def resolve_precision(precision: str, device: torch.device) -> tuple[str, torch.dtype | None]:
    if precision == "auto":
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return f"auto:{str(dtype).removeprefix('torch.')}", dtype
        if device.type == "cpu":
            return "auto:bf16", torch.bfloat16
        return "fp32", None
    if precision == "fp32":
        return "fp32", None
    if precision == "fp16":
        if device.type == "cpu":
            raise ValueError("fp16 reference precision requires --device cuda")
        return "fp16", torch.float16
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 reference precision requested, but CUDA BF16 is not supported")
        return "bf16", torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


def run_model(
    model: DepthAnything3,
    x: torch.Tensor,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    context = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    with torch.no_grad():
        with context:
            return model.model(x, None, None, [], False, False, "saddle_balanced")


def main() -> int:
    parser = argparse.ArgumentParser(description="Official DA3 PyTorch reference inference")
    parser.add_argument("--model-dir", required=True, help="Directory containing model.safetensors")
    parser.add_argument("--model-name", default=None, help="Official model name, e.g. da3-small")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--img-size", type=int, default=518, help="Square model input size")
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16", "bf16", "auto"],
        help="Reference math precision. auto matches official DepthAnything3.forward autocast.",
    )
    parser.add_argument(
        "--no-cam-token",
        action="store_true",
        help="Deprecated compatibility flag. Ignored: current CPU/CUDA runners use camera_token.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_cam_token:
        print("WARNING: --no-cam-token is ignored; current C/CUDA DA3 injects camera_token.")

    # Keep config existence check for clearer user errors and compatibility with old workflow.
    config_path = model_dir / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            json.load(f)

    model_name = infer_model_name(model_dir, args.model_name)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA requested but torch.cuda.is_available() is false", file=sys.stderr)
        return 1
    try:
        precision_label, autocast_dtype = resolve_precision(args.precision, device)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Model: {model_name}")
    print(f"Image: {args.image}")
    print(f"Device: {device}")
    print(f"Precision: {precision_label}")

    model = DepthAnything3(model_name)
    load_local_weights(model, model_dir)
    model = model.to(device).eval()

    x, orig_h, orig_w = preprocess_square(Path(args.image), args.img_size, device)
    print(f"Preprocessed input: {tuple(x.shape)}; original={orig_w}x{orig_h}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = run_model(model, x, device, autocast_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    depth = upsample_to_original(out["depth"], orig_h, orig_w)
    confidence = upsample_to_original(out["depth_conf"], orig_h, orig_w)

    np.save(output_dir / "depth_ref.npy", depth)
    np.save(output_dir / "confidence_ref.npy", confidence)
    write_pgm16(output_dir / "depth_ref.pgm", depth)
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(
            {
                "model_name": model_name,
                "model_dir": str(model_dir),
                "image": str(args.image),
                "device": str(device),
                "precision": precision_label,
                "img_size": args.img_size,
                "orig_width": orig_w,
                "orig_height": orig_h,
            },
            f,
            indent=2,
        )

    print(f"Inference: {elapsed_ms:.2f} ms")
    print(
        "Depth: "
        f"shape={depth.shape} min={depth.min():.6f} max={depth.max():.6f} "
        f"mean={depth.mean():.6f}"
    )
    print(
        "Confidence: "
        f"shape={confidence.shape} min={confidence.min():.6f} "
        f"max={confidence.max():.6f} mean={confidence.mean():.6f}"
    )
    print(f"Wrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
