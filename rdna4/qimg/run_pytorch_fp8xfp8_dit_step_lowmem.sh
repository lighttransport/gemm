#!/usr/bin/env bash
set -euo pipefail

# Low-memory PyTorch native FP8xFP8 DiT-step repro for Qwen-Image.
#
# This avoids the full diffusers pipeline, text encoder, and VAE. It loads the
# FP8 DiT on CPU, reads the pinned 256x256 packed latent/text tensors, then
# manually runs one DiT forward while moving only one module/block at a time to
# the GPU.
#
# Run from repo root:
#   bash rdna4/qimg/run_pytorch_fp8xfp8_dit_step_lowmem.sh
#
# Useful overrides:
#   SCALE_MODE=comfy bash rdna4/qimg/run_pytorch_fp8xfp8_dit_step_lowmem.sh
#   SCALE_MODE=perrow SCALE_DIV=512 bash rdna4/qimg/run_pytorch_fp8xfp8_dit_step_lowmem.sh
#   MAX_BLOCKS=1 bash rdna4/qimg/run_pytorch_fp8xfp8_dit_step_lowmem.sh

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
INIT_BIN="${INIT_BIN:-$ROOT/ref/qwen_image/init_latent_256.bin}"
TEXT_BIN="${TEXT_BIN:-$ROOT/ref/qwen_image/apple_text_256.bin}"
REF_BIN="${REF_BIN:-$ROOT/ref/qwen_image/final_latent_packed_256.bin}"
OUT_BIN="${OUT_BIN:-$QIMG_DIR/pytorch_fp8xfp8_dit_step_256.bin}"
LOG="${LOG:-${OUT_BIN%.bin}.log}"
DTYPE="${DTYPE:-bf16}"

# SCALE_MODE=comfy mirrors ComfyUI raw FP8: scale=1, clamp activation, FP8xFP8.
# SCALE_MODE=perrow uses qimg-style row scale for comparison.
SCALE_MODE="${SCALE_MODE:-comfy}"
SCALE_DIV="${SCALE_DIV:-512}"

# First denoise step in QwenImagePipeline passes t/1000 into the transformer.
TIMESTEP="${TIMESTEP:-1.0}"
SIZE="${SIZE:-256}"
MAX_BLOCKS="${MAX_BLOCKS:-60}"

export ROOT QIMG_DIR MODEL FP8_DIT INIT_BIN TEXT_BIN REF_BIN OUT_BIN DTYPE SCALE_MODE SCALE_DIV TIMESTEP SIZE MAX_BLOCKS

if ! "$PY" - <<'PY' >/dev/null 2>&1; then
import diffusers, transformers, accelerate, torch
PY
  echo "Installing Python reference deps into $QIMG_DIR/.venv" >&2
  uv pip install --python "$PY" \
    "diffusers @ git+https://github.com/huggingface/diffusers" \
    transformers accelerate qwen-vl-utils
fi

mkdir -p "$(dirname "$OUT_BIN")"

{
"$PY" - <<'PY'
import gc
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(os.environ["ROOT"])
MODEL = os.environ["MODEL"]
FP8_DIT = os.environ["FP8_DIT"]
INIT_BIN = Path(os.environ["INIT_BIN"])
TEXT_BIN = Path(os.environ["TEXT_BIN"])
REF_BIN = Path(os.environ["REF_BIN"])
OUT_BIN = Path(os.environ["OUT_BIN"])
DTYPE = torch.bfloat16 if os.environ.get("DTYPE", "bf16") == "bf16" else torch.float16
SCALE_MODE = os.environ.get("SCALE_MODE", "comfy")
SCALE_DIV = float(os.environ.get("SCALE_DIV", "512"))
TIMESTEP = float(os.environ.get("TIMESTEP", "1.0"))
SIZE = int(os.environ.get("SIZE", "256"))
MAX_BLOCKS = int(os.environ.get("MAX_BLOCKS", "60"))


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
            else:
                scale_a = torch.ones((), device=x.device, dtype=torch.float32)
                scale_b = torch.ones((), device=x.device, dtype=torch.float32)
                x8 = torch.clamp(x2.float(), -448.0, 448.0).to(torch.float8_e4m3fn).contiguous()

            try:
                y = torch._scaled_mm(
                    x8,
                    w.t().contiguous(),
                    scale_a=scale_a,
                    scale_b=scale_b,
                    out_dtype=DTYPE,
                    bias=self.bias,
                )
            except TypeError:
                y = torch._scaled_mm(
                    x8,
                    w.t().contiguous(),
                    scale_a=scale_a,
                    scale_b=scale_b,
                    out_dtype=DTYPE,
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
        counts[str(p.dtype)] = counts.get(str(p.dtype), 0) + p.numel()
    return counts


def mem(label: str):
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    print(f"{label}: free={free/2**30:.2f}GiB alloc={alloc/2**30:.2f}GiB total={total/2**30:.2f}GiB", flush=True)


def to_gpu(module: nn.Module):
    module.to(device="cuda")
    return module


def to_cpu(module: nn.Module):
    module.to(device="cpu")
    gc.collect()
    torch.cuda.empty_cache()


def metrics(ref: np.ndarray, got: np.ndarray):
    n = min(ref.size, got.size)
    r = ref.reshape(-1)[:n].astype(np.float64)
    g = got.reshape(-1)[:n].astype(np.float64)
    d = g - r
    mse = float(np.mean(d * d))
    cos = float(np.dot(r, g) / (np.linalg.norm(r) * np.linalg.norm(g) + 1e-30))
    peak = float(np.max(np.abs(r)))
    psnr = 20.0 * math.log10(max(peak, 1e-20)) - 10.0 * math.log10(max(mse, 1e-30))
    return cos, psnr, float(np.mean(np.abs(d))), float(np.max(np.abs(d)))


print(f"torch={torch.__version__} hip={torch.version.hip} cuda_available={torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("GPU is not visible to PyTorch")
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
print(f"fp8_dit={FP8_DIT}", flush=True)
print(f"init={INIT_BIN}", flush=True)
print(f"text={TEXT_BIN}", flush=True)
print(f"out={OUT_BIN}", flush=True)
print(f"dtype={DTYPE} scale_mode={SCALE_MODE} scale_div={SCALE_DIV} timestep={TIMESTEP} max_blocks={MAX_BLOCKS}", flush=True)
mem("start")

from diffusers import QwenImageTransformer2DModel

t0 = time.perf_counter()
model = QwenImageTransformer2DModel.from_single_file(FP8_DIT, config=MODEL, subfolder="transformer")
model.eval()
print(f"loaded transformer on CPU in {time.perf_counter() - t0:.1f}s dtypes={count_dtypes(model)}", flush=True)
patched = patch_linears(model)
print(f"patched {patched} linears", flush=True)

init = np.fromfile(INIT_BIN, dtype=np.float32)
if init.size % 64 != 0:
    raise ValueError(f"{INIT_BIN}: expected packed latent with K=64, got {init.size} floats")
n_img = init.size // 64
text = np.fromfile(TEXT_BIN, dtype=np.float32)
if text.size % 3584 != 0:
    raise ValueError(f"{TEXT_BIN}: expected text dim 3584, got {text.size} floats")
n_txt = text.size // 3584

hidden_states = torch.from_numpy(init.reshape(1, n_img, 64)).to(device="cuda", dtype=DTYPE)
encoder_hidden_states = torch.from_numpy(text.reshape(1, n_txt, 3584)).to(device="cuda", dtype=DTYPE)
encoder_hidden_states_mask = torch.ones((1, n_txt), device="cuda", dtype=torch.bool)
timestep = torch.full((1,), TIMESTEP, device="cuda", dtype=DTYPE)
img_shapes = [[(1, SIZE // 16, SIZE // 16)]]
modulate_index = None
guidance = None
mem("inputs")

with torch.inference_mode():
    to_gpu(model.img_in)
    hidden_states = model.img_in(hidden_states)
    to_cpu(model.img_in)
    mem("after img_in")

    to_gpu(model.txt_norm)
    encoder_hidden_states = model.txt_norm(encoder_hidden_states)
    to_cpu(model.txt_norm)
    to_gpu(model.txt_in)
    encoder_hidden_states = model.txt_in(encoder_hidden_states)
    to_cpu(model.txt_in)
    mem("after txt path")

    to_gpu(model.time_text_embed)
    temb = model.time_text_embed(timestep, hidden_states)
    to_cpu(model.time_text_embed)
    mem("after temb")

    image_rotary_emb = model.pos_embed(
        img_shapes,
        max_txt_seq_len=encoder_hidden_states.shape[1],
        device=hidden_states.device,
    )

    block_attention_kwargs = {}
    image_mask = torch.ones((1, hidden_states.shape[1]), dtype=torch.bool, device="cuda")
    joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
    block_attention_kwargs["attention_mask"] = joint_attention_mask[:, None, None, :]

    n_blocks = min(MAX_BLOCKS, len(model.transformer_blocks))
    for i in range(n_blocks):
        block = model.transformer_blocks[i]
        to_gpu(block)
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=block_attention_kwargs,
            modulate_index=modulate_index,
        )
        torch.cuda.synchronize()
        to_cpu(block)
        if i == 0 or (i + 1) % 5 == 0 or i + 1 == n_blocks:
            hs = hidden_states.float()
            print(
                f"block {i+1:02d}/{n_blocks}: hidden mean={hs.mean().item():.6g} std={hs.std().item():.6g}",
                flush=True,
            )
            mem(f"after block {i+1:02d}")

    if n_blocks == len(model.transformer_blocks):
        to_gpu(model.norm_out)
        hidden_states = model.norm_out(hidden_states, temb)
        to_cpu(model.norm_out)
        to_gpu(model.proj_out)
        output = model.proj_out(hidden_states)
        to_cpu(model.proj_out)
        label = "full_dit_step"
    else:
        output = hidden_states
        label = f"partial_after_{n_blocks}_blocks"

    torch.cuda.synchronize()
    out = output.float().detach().cpu().numpy()
    out.astype(np.float32).tofile(OUT_BIN)
    print(
        f"{label}: shape={out.shape} min={out.min():.6g} max={out.max():.6g} "
        f"mean={out.mean():.6g} std={out.std():.6g}",
        flush=True,
    )
    print(f"saved {OUT_BIN}", flush=True)

if REF_BIN.exists() and out.size == np.fromfile(REF_BIN, dtype=np.float32).size:
    ref = np.fromfile(REF_BIN, dtype=np.float32)
    cos, psnr, mae, maxd = metrics(ref, out)
    print(f"vs {REF_BIN.name}: cos={cos:.9f} psnr={psnr:.2f}dB mae={mae:.7g} max={maxd:.7g}", flush=True)
mem("done")
PY
} 2>&1 | tee "$LOG"

echo "log: $LOG"
echo "bin: $OUT_BIN"
