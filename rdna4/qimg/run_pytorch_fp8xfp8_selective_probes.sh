#!/usr/bin/env bash
set -euo pipefail

# Selective Qwen-Image PyTorch native FP8xFP8 probes.
#
# This never calls diffusers `from_single_file()` and never materializes the
# full DiT. It opens the safetensors file lazily, loads one tensor at a time,
# runs torch._scaled_mm(FP8, FP8), prints quality/stats, then releases it.
#
# Run from repo root:
#   bash rdna4/qimg/run_pytorch_fp8xfp8_selective_probes.sh
#
# Useful overrides:
#   SCALE_MODE=comfy bash rdna4/qimg/run_pytorch_fp8xfp8_selective_probes.sh
#   SCALE_MODE=perrow SCALE_DIV=512 bash rdna4/qimg/run_pytorch_fp8xfp8_selective_probes.sh
#   PROBES=img_in,txt_in,blk0_img_q bash rdna4/qimg/run_pytorch_fp8xfp8_selective_probes.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QIMG_DIR="$ROOT/rdna4/qimg"
PY="$QIMG_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "missing qimg venv python: $PY" >&2
  exit 1
fi

export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

FP8_DIT="${FP8_DIT:-/mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors}"
INIT_BIN="${INIT_BIN:-$ROOT/ref/qwen_image/init_latent_256.bin}"
TEXT_BIN="${TEXT_BIN:-$ROOT/ref/qwen_image/apple_text_256.bin}"
OUT_DIR="${OUT_DIR:-$QIMG_DIR/pytorch_fp8xfp8_probes}"
DTYPE="${DTYPE:-bf16}"
SCALE_MODE="${SCALE_MODE:-comfy}"
SCALE_DIV="${SCALE_DIV:-512}"
TIMESTEP="${TIMESTEP:-1.0}"
PROBES="${PROBES:-img_in,txt_in,temb_fc1,temb_fc2,blk0_img_mod,blk0_txt_mod,blk0_img_q,blk0_img_k,blk0_img_v,blk0_txt_q,blk0_txt_k,blk0_txt_v,blk0_img_mlp_fc1,blk0_img_mlp_fc2,blk0_txt_mlp_fc1,blk0_txt_mlp_fc2}"
LOG="${LOG:-$OUT_DIR/probes_${SCALE_MODE}_div${SCALE_DIV}.log}"

export ROOT FP8_DIT INIT_BIN TEXT_BIN OUT_DIR DTYPE SCALE_MODE SCALE_DIV TIMESTEP PROBES

mkdir -p "$OUT_DIR"

{
"$PY" - <<'PY'
import gc
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


FP8_DIT = os.environ["FP8_DIT"]
INIT_BIN = Path(os.environ["INIT_BIN"])
TEXT_BIN = Path(os.environ["TEXT_BIN"])
OUT_DIR = Path(os.environ["OUT_DIR"])
DTYPE = torch.bfloat16 if os.environ.get("DTYPE", "bf16") == "bf16" else torch.float16
SCALE_MODE = os.environ.get("SCALE_MODE", "comfy")
SCALE_DIV = float(os.environ.get("SCALE_DIV", "512"))
TIMESTEP = float(os.environ.get("TIMESTEP", "1.0"))
PROBES = [p for p in os.environ.get("PROBES", "").split(",") if p]


def mem(label):
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    print(f"{label}: free={free/2**30:.2f}GiB alloc={alloc/2**30:.2f}GiB", flush=True)


def f32_metrics(ref, got):
    r = ref.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
    g = got.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
    n = min(r.size, g.size)
    r = r[:n]
    g = g[:n]
    d = g - r
    mse = float(np.mean(d * d))
    cos = float(np.dot(r, g) / (np.linalg.norm(r) * np.linalg.norm(g) + 1e-30))
    peak = float(np.max(np.abs(r)))
    psnr = 20.0 * math.log10(max(peak, 1e-20)) - 10.0 * math.log10(max(mse, 1e-30))
    return cos, psnr, float(np.mean(np.abs(d))), float(np.max(np.abs(d)))


def stat(x):
    xf = x.detach().float()
    return (
        float(xf.min().cpu()),
        float(xf.max().cpu()),
        float(xf.mean().cpu()),
        float(xf.std().cpu()),
    )


def load_tensor(st, name, device="cuda"):
    t = st.get_tensor(name)
    return t.to(device=device)


def native_fp8_linear(x, w_fp8, bias_fp8=None):
    x2 = x.reshape(-1, x.shape[-1])
    if SCALE_MODE == "perrow":
        scale_a = torch.amax(x2.float().abs(), dim=1, keepdim=True).clamp_min(1.0e-12) / SCALE_DIV
        x8 = torch.clamp(x2.float() / scale_a, -448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
        scale_b = torch.ones((1, w_fp8.shape[0]), device=x.device, dtype=torch.float32)
    else:
        scale_a = torch.ones((), device=x.device, dtype=torch.float32)
        scale_b = torch.ones((), device=x.device, dtype=torch.float32)
        x8 = torch.clamp(x2.float(), -448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    y = torch._scaled_mm(
        x8,
        w_fp8.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=DTYPE,
    )
    if isinstance(y, tuple):
        y = y[0]
    if bias_fp8 is not None:
        y = y + bias_fp8.to(dtype=y.dtype)
    return y.reshape(*x.shape[:-1], y.shape[-1])


def ref_linear(x, w_fp8, bias_fp8=None):
    y = x.float() @ w_fp8.float().t()
    if bias_fp8 is not None:
        y = y + bias_fp8.float()
    return y


def run_probe(st, name, x, weight_name, bias_name=None, save=True):
    print(f"\nprobe {name}: x={tuple(x.shape)} weight={weight_name}", flush=True)
    w = load_tensor(st, weight_name)
    b = load_tensor(st, bias_name) if bias_name else None
    y_ref = ref_linear(x, w, b)
    y_native = native_fp8_linear(x, w, b)
    cos, psnr, mae, maxd = f32_metrics(y_ref, y_native)
    mn, mx, mean, sd = stat(y_native)
    print(
        f"{name}: native shape={tuple(y_native.shape)} min={mn:.6g} max={mx:.6g} "
        f"mean={mean:.6g} std={sd:.6g} cos_vs_f32={cos:.9f} psnr={psnr:.2f}dB "
        f"mae={mae:.7g} max={maxd:.7g}",
        flush=True,
    )
    if save:
        out = OUT_DIR / f"{name}.bin"
        y_native.detach().float().cpu().numpy().astype(np.float32).tofile(out)
        print(f"saved {out}", flush=True)
    del w, b, y_ref
    gc.collect()
    torch.cuda.empty_cache()
    return y_native


def run_mlp_pair(st, prefix, x, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    fc1_name = f"{prefix}_fc1"
    fc2_name = f"{prefix}_fc2"
    if fc1_name not in PROBES and fc2_name not in PROBES:
        return
    if x is None:
        print(f"skip {prefix}: missing prepared input", flush=True)
        return
    fc1 = run_probe(st, fc1_name, x, fc1_weight, fc1_bias, save=fc1_name in PROBES)
    if fc2_name in PROBES:
        # Qwen-Image FeedForward uses activation_fn="gelu-approximate".
        fc2_in = F.gelu(fc1.float(), approximate="tanh").to(DTYPE)
        run_probe(st, fc2_name, fc2_in, fc2_weight, fc2_bias, save=True)
        del fc2_in
    del fc1
    gc.collect()
    torch.cuda.empty_cache()


def timestep_embedding(timestep, dim=256):
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device="cuda")
    exponent = exponent / half_dim
    emb = torch.exp(exponent)
    emb = (torch.tensor([timestep], device="cuda", dtype=torch.float32) * 1000.0)[:, None] * emb[None, :]
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    return emb


def rms_norm(x, weight, eps=1e-6):
    return x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps) * weight.float()


print(f"torch={torch.__version__} hip={torch.version.hip} cuda_available={torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("GPU is not visible")
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
print(f"fp8_dit={FP8_DIT}", flush=True)
print(f"scale_mode={SCALE_MODE} scale_div={SCALE_DIV} dtype={DTYPE} probes={PROBES}", flush=True)
mem("start")

init = np.fromfile(INIT_BIN, dtype=np.float32)
text = np.fromfile(TEXT_BIN, dtype=np.float32)
img_tokens = torch.from_numpy(init.reshape(1, init.size // 64, 64)).to("cuda", dtype=DTYPE)
txt_tokens = torch.from_numpy(text.reshape(1, text.size // 3584, 3584)).to("cuda", dtype=DTYPE)
print(f"inputs: img={tuple(img_tokens.shape)} txt={tuple(txt_tokens.shape)}", flush=True)

with safe_open(FP8_DIT, framework="pt", device="cpu") as st, torch.inference_mode():
    img_h = None
    txt_h = None
    temb = None
    img_modulated = None
    txt_modulated = None

    if "img_in" in PROBES or any(p.startswith("blk0_img") for p in PROBES):
        img_h = run_probe(st, "img_in", img_tokens, "img_in.weight", "img_in.bias", save="img_in" in PROBES)

    if "txt_in" in PROBES or any(p.startswith("blk0_txt") for p in PROBES):
        tw = load_tensor(st, "txt_norm.weight")
        txt_normed = rms_norm(txt_tokens, tw)
        del tw
        txt_h = run_probe(st, "txt_in", txt_normed.to(DTYPE), "txt_in.weight", "txt_in.bias", save="txt_in" in PROBES)

    if any(p in PROBES for p in ["temb_fc1", "temb_fc2", "blk0_img_mod", "blk0_txt_mod"]):
        t_in = timestep_embedding(TIMESTEP)
        t1 = run_probe(st, "temb_fc1", t_in.to(DTYPE), "time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_1.bias", save="temb_fc1" in PROBES)
        t1 = F.silu(t1.float()).to(DTYPE)
        temb = run_probe(st, "temb_fc2", t1, "time_text_embed.timestep_embedder.linear_2.weight", "time_text_embed.timestep_embedder.linear_2.bias", save="temb_fc2" in PROBES)

    if any(p.startswith("blk0_img") for p in PROBES):
        if temb is None:
            t_in = timestep_embedding(TIMESTEP)
            t1 = run_probe(st, "_temb_fc1_tmp", t_in.to(DTYPE), "time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_1.bias", save=False)
            temb = run_probe(st, "_temb_fc2_tmp", F.silu(t1.float()).to(DTYPE), "time_text_embed.timestep_embedder.linear_2.weight", "time_text_embed.timestep_embedder.linear_2.bias", save=False)
        img_mod = run_probe(st, "blk0_img_mod", F.silu(temb.float()).to(DTYPE), "transformer_blocks.0.img_mod.1.weight", "transformer_blocks.0.img_mod.1.bias", save="blk0_img_mod" in PROBES)
        img_mod1, img_mod2 = img_mod.chunk(2, dim=-1)
        shift, scale, gate = img_mod1.chunk(3, dim=-1)
        img_normed = F.layer_norm(img_h.float(), (img_h.shape[-1],), weight=None, bias=None, eps=1e-6)
        img_modulated = (img_normed * (1.0 + scale.float().unsqueeze(1)) + shift.float().unsqueeze(1)).to(DTYPE)

    if any(p.startswith("blk0_txt") for p in PROBES):
        if temb is None:
            t_in = timestep_embedding(TIMESTEP)
            t1 = run_probe(st, "_temb_fc1_tmp", t_in.to(DTYPE), "time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_1.bias", save=False)
            temb = run_probe(st, "_temb_fc2_tmp", F.silu(t1.float()).to(DTYPE), "time_text_embed.timestep_embedder.linear_2.weight", "time_text_embed.timestep_embedder.linear_2.bias", save=False)
        txt_mod = run_probe(st, "blk0_txt_mod", F.silu(temb.float()).to(DTYPE), "transformer_blocks.0.txt_mod.1.weight", "transformer_blocks.0.txt_mod.1.bias", save="blk0_txt_mod" in PROBES)
        txt_mod1, txt_mod2 = txt_mod.chunk(2, dim=-1)
        shift, scale, gate = txt_mod1.chunk(3, dim=-1)
        txt_normed = F.layer_norm(txt_h.float(), (txt_h.shape[-1],), weight=None, bias=None, eps=1e-6)
        txt_modulated = (txt_normed * (1.0 + scale.float().unsqueeze(1)) + shift.float().unsqueeze(1)).to(DTYPE)

    probe_specs = {
        "blk0_img_q": (img_modulated, "transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.attn.to_q.bias"),
        "blk0_img_k": (img_modulated, "transformer_blocks.0.attn.to_k.weight", "transformer_blocks.0.attn.to_k.bias"),
        "blk0_img_v": (img_modulated, "transformer_blocks.0.attn.to_v.weight", "transformer_blocks.0.attn.to_v.bias"),
        "blk0_txt_q": (txt_modulated, "transformer_blocks.0.attn.add_q_proj.weight", "transformer_blocks.0.attn.add_q_proj.bias"),
        "blk0_txt_k": (txt_modulated, "transformer_blocks.0.attn.add_k_proj.weight", "transformer_blocks.0.attn.add_k_proj.bias"),
        "blk0_txt_v": (txt_modulated, "transformer_blocks.0.attn.add_v_proj.weight", "transformer_blocks.0.attn.add_v_proj.bias"),
    }
    for name, (x, w, b) in probe_specs.items():
        if name in PROBES:
            if x is None:
                print(f"skip {name}: missing prepared input", flush=True)
                continue
            run_probe(st, name, x, w, b)

    run_mlp_pair(
        st,
        "blk0_img_mlp",
        img_modulated,
        "transformer_blocks.0.img_mlp.net.0.proj.weight",
        "transformer_blocks.0.img_mlp.net.0.proj.bias",
        "transformer_blocks.0.img_mlp.net.2.weight",
        "transformer_blocks.0.img_mlp.net.2.bias",
    )
    run_mlp_pair(
        st,
        "blk0_txt_mlp",
        txt_modulated,
        "transformer_blocks.0.txt_mlp.net.0.proj.weight",
        "transformer_blocks.0.txt_mlp.net.0.proj.bias",
        "transformer_blocks.0.txt_mlp.net.2.weight",
        "transformer_blocks.0.txt_mlp.net.2.bias",
    )

mem("done")
PY
} 2>&1 | tee "$LOG"

echo "log: $LOG"
echo "out_dir: $OUT_DIR"
