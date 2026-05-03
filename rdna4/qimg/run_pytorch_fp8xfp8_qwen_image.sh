#!/usr/bin/env bash
set -euo pipefail

# Reproduce Qwen-Image PyTorch native FP8xFP8 generation outside Codex sandbox.
#
# This intentionally uses rdna4/qimg/.venv and keeps the whole repro in one
# script so it can be launched directly from the repo root:
#
#   bash rdna4/qimg/run_pytorch_fp8xfp8_qwen_image.sh
#
# Useful overrides:
#   SIZE=256 STEPS=20 CFG=4.0 bash rdna4/qimg/run_pytorch_fp8xfp8_qwen_image.sh
#   SCALE_MODE=comfy bash rdna4/qimg/run_pytorch_fp8xfp8_qwen_image.sh
#   SCALE_MODE=perrow SCALE_DIV=512 bash rdna4/qimg/run_pytorch_fp8xfp8_qwen_image.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QIMG_DIR="$ROOT/rdna4/qimg"
PY="$QIMG_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "missing qimg venv python: $PY" >&2
  echo "create it first with: uv venv --python 3.12 rdna4/qimg/.venv" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-/mnt/disk1/hf-cache}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-Qwen/Qwen-Image}"
FP8_DIT="${FP8_DIT:-/mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors}"
OUT="${OUT:-$QIMG_DIR/pytorch_fp8xfp8_comfy_256.png}"
LOG="${LOG:-${OUT%.png}.log}"
PROMPT="${PROMPT:-a red apple on a white table}"
NEGATIVE="${NEGATIVE:- }"
SIZE="${SIZE:-256}"
STEPS="${STEPS:-20}"
SEED="${SEED:-42}"
CFG="${CFG:-4.0}"
DTYPE="${DTYPE:-bf16}"

# SCALE_MODE=comfy matches current ComfyUI raw FP8 fast path: scalar scale=1,
# clamp activations to [-448, 448], then torch._scaled_mm(FP8, FP8).
# SCALE_MODE=perrow is the qimg-style diagnostic: scale_a=max(abs(row))/SCALE_DIV.
SCALE_MODE="${SCALE_MODE:-comfy}"
SCALE_DIV="${SCALE_DIV:-512}"

export ROOT QIMG_DIR MODEL FP8_DIT OUT PROMPT NEGATIVE SIZE STEPS SEED CFG DTYPE SCALE_MODE SCALE_DIV

if ! "$PY" - <<'PY' >/dev/null 2>&1; then
import diffusers, transformers, accelerate, torch
PY
  echo "Installing Python reference deps into $QIMG_DIR/.venv" >&2
  uv pip install --python "$PY" \
    "diffusers @ git+https://github.com/huggingface/diffusers" \
    transformers accelerate qwen-vl-utils
fi

mkdir -p "$(dirname "$OUT")"

{
"$PY" - <<'PY'
import gc
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(os.environ["ROOT"]) if "ROOT" in os.environ else Path.cwd()


def env(name, default):
    return os.environ.get(name, default)


MODEL = env("MODEL", "Qwen/Qwen-Image")
FP8_DIT = env("FP8_DIT", "/mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors")
OUT = env("OUT", str(ROOT / "rdna4/qimg/pytorch_fp8xfp8_comfy_256.png"))
PROMPT = env("PROMPT", "a red apple on a white table")
NEGATIVE = env("NEGATIVE", " ")
SIZE = int(env("SIZE", "256"))
STEPS = int(env("STEPS", "20"))
SEED = int(env("SEED", "42"))
CFG = float(env("CFG", "4.0"))
DTYPE = torch.bfloat16 if env("DTYPE", "bf16") == "bf16" else torch.float16
SCALE_MODE = env("SCALE_MODE", "comfy")
SCALE_DIV = float(env("SCALE_DIV", "512"))


class NativeFp8Linear(nn.Module):
    def __init__(self, base: nn.Linear, name: str):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.name = name

    def forward(self, x):
        w = self.weight
        if x.is_cuda and w.is_cuda and w.dtype == torch.float8_e4m3fn and hasattr(torch, "_scaled_mm"):
            shape = x.shape
            x2 = x.reshape(-1, shape[-1])
            if SCALE_MODE == "perrow":
                scale_a = torch.amax(x2.float().abs(), dim=1, keepdim=True).clamp_min(1.0e-12) / SCALE_DIV
                x8 = torch.clamp(x2.float() / scale_a, -448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
                scale_b = torch.ones((1, w.shape[0]), device=x.device, dtype=torch.float32)
                out_dtype = DTYPE
            else:
                scale_a = torch.ones((), device=x.device, dtype=torch.float32)
                scale_b = torch.ones((), device=x.device, dtype=torch.float32)
                x8 = torch.clamp(x2.float(), -448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
                out_dtype = DTYPE

            try:
                y = torch._scaled_mm(
                    x8,
                    w.t().contiguous(),
                    scale_a=scale_a,
                    scale_b=scale_b,
                    out_dtype=out_dtype,
                    bias=self.bias,
                )
            except TypeError:
                y = torch._scaled_mm(
                    x8,
                    w.t().contiguous(),
                    scale_a=scale_a,
                    scale_b=scale_b,
                    out_dtype=out_dtype,
                )
                if self.bias is not None:
                    y = y + self.bias.to(dtype=y.dtype)
            if isinstance(y, tuple):
                y = y[0]
            return y.reshape(*shape[:-1], y.shape[-1])
        return F.linear(x, w, self.bias)


def patch_linears(module: nn.Module, prefix: str = "") -> int:
    patched = 0
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            setattr(module, name, NativeFp8Linear(child, full))
            patched += 1
        else:
            patched += patch_linears(child, full)
    return patched


def count_dtypes(module: nn.Module):
    counts = {}
    for p in module.parameters():
        counts[p.dtype] = counts.get(p.dtype, 0) + p.numel()
    return {str(k): v for k, v in counts.items()}


print(f"torch={torch.__version__} hip={torch.version.hip} cuda_available={torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("GPU is not visible to PyTorch")
print(f"gpu={torch.cuda.get_device_name(0)} mem={torch.cuda.mem_get_info()}", flush=True)
print(f"model={MODEL}", flush=True)
print(f"fp8_dit={FP8_DIT}", flush=True)
print(f"out={OUT}", flush=True)
print(f"size={SIZE} steps={STEPS} seed={SEED} cfg={CFG} dtype={DTYPE} scale_mode={SCALE_MODE} scale_div={SCALE_DIV}", flush=True)

from diffusers import QwenImagePipeline, QwenImageTransformer2DModel

t0 = time.perf_counter()
transformer = QwenImageTransformer2DModel.from_single_file(
    FP8_DIT,
    config=MODEL,
    subfolder="transformer",
)
print(f"loaded transformer in {time.perf_counter() - t0:.1f}s dtypes={count_dtypes(transformer)}", flush=True)
patched = patch_linears(transformer)
print(f"patched {patched} nn.Linear modules with native FP8xFP8 wrapper", flush=True)
if "torch.float8_e4m3fn" not in count_dtypes(transformer):
    print("WARNING: transformer has no float8_e4m3fn parameters before pipeline attach", flush=True)

t0 = time.perf_counter()
pipe = QwenImagePipeline.from_pretrained(
    MODEL,
    transformer=transformer,
    torch_dtype=DTYPE,
)
print(f"loaded pipeline rest in {time.perf_counter() - t0:.1f}s", flush=True)
print(f"transformer dtypes after pipeline attach: {count_dtypes(pipe.transformer)}", flush=True)

pipe.enable_sequential_cpu_offload()
gc.collect()
torch.cuda.empty_cache()
print(f"after offload mem={torch.cuda.mem_get_info()}", flush=True)

gen = torch.Generator(device="cuda").manual_seed(SEED)
torch.cuda.synchronize()
t0 = time.perf_counter()
result = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    height=SIZE,
    width=SIZE,
    num_inference_steps=STEPS,
    true_cfg_scale=CFG,
    generator=gen,
)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
img = result.images[0]
img.save(OUT)
try:
    import numpy as np

    arr = np.asarray(img).astype(np.float32)
    print(
        f"image stats: min={arr.min():.1f} max={arr.max():.1f} mean={arr.mean():.2f} std={arr.std():.2f}",
        flush=True,
    )
except Exception as exc:
    print(f"image stats skipped: {type(exc).__name__}: {exc}", flush=True)
print(f"saved {OUT} ({img.size[0]}x{img.size[1]}) total={dt:.2f}s mem={torch.cuda.mem_get_info()}", flush=True)
PY
} 2>&1 | tee "$LOG"

echo "log: $LOG"
echo "png: $OUT"
