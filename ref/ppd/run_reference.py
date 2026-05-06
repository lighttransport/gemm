#!/usr/bin/env python3
"""PyTorch reference runner for Pixel-Perfect Depth.

This mirrors the C/CUDA PPD pipeline closely enough for performance and drift
tracking: DA2 ViT-L semantic tokens, 4-step DiT denoising, fixed seed, and
align-corners bilinear resizing.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def read_ppm(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"P6":
            raise ValueError(f"{path} is not a binary P6 PPM")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline())
        if maxval != 255:
            raise ValueError(f"unsupported PPM maxval {maxval}")
        data = np.frombuffer(f.read(w * h * 3), dtype=np.uint8).reshape(h, w, 3).copy()
    return data


def c_randn(n: int) -> np.ndarray:
    libc = ctypes.CDLL(None)
    libc.srand(ctypes.c_uint(42))
    rand_max = float((1 << 31) - 1)
    out = np.empty(n, dtype=np.float32)
    i = 0
    while i + 1 < n:
        u1 = (float(libc.rand()) + 1.0) / (rand_max + 1.0)
        u2 = float(libc.rand()) / rand_max
        r = math.sqrt(-2.0 * math.log(u1))
        out[i] = r * math.cos(2.0 * math.pi * u2)
        out[i + 1] = r * math.sin(2.0 * math.pi * u2)
        i += 2
    if i < n:
        u1 = (float(libc.rand()) + 1.0) / (rand_max + 1.0)
        u2 = float(libc.rand()) / rand_max
        out[i] = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return out


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {k: v.detach() for k, v in state.items() if torch.is_tensor(v)}


def to_device_state(state: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype,
                    prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not k.startswith(prefixes):
            continue
        target_dtype = dtype if v.is_floating_point() else v.dtype
        out[k] = v.to(device=device, dtype=target_dtype, non_blocking=True)
    return out


def resize_chw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return F.interpolate(x.unsqueeze(0), size=(h, w), mode="bilinear",
                         align_corners=True).squeeze(0)


def sinusoidal_embed(t: float, dim: int, device: torch.device) -> torch.Tensor:
    half = dim // 2
    i = torch.arange(half, device=device, dtype=torch.float32)
    freq = torch.exp(-math.log(10000.0) * i / half)
    arg = t * freq
    return torch.cat([torch.cos(arg), torch.sin(arg)], dim=0)


def apply_rope_2d(x: torch.Tensor, g_h: int, g_w: int, freq_base: float) -> torch.Tensor:
    nt, heads, head_dim = x.shape
    half = head_dim // 2
    quarter = half // 2
    y = torch.arange(g_h, device=x.device, dtype=torch.float32).repeat_interleave(g_w)
    xpos = torch.arange(g_w, device=x.device, dtype=torch.float32).repeat(g_h)
    j = torch.arange(quarter, device=x.device, dtype=torch.float32)
    freq = torch.pow(torch.tensor(freq_base, device=x.device), -(2.0 * j) / half)
    cy, sy = torch.cos(y[:, None] * freq[None, :]), torch.sin(y[:, None] * freq[None, :])
    cx, sx = torch.cos(xpos[:, None] * freq[None, :]), torch.sin(xpos[:, None] * freq[None, :])
    out = x.clone()
    a, b = x[:, :, :quarter], x[:, :, quarter:half]
    out[:, :, :quarter] = a * cy[:, None, :] - b * sy[:, None, :]
    out[:, :, quarter:half] = a * sy[:, None, :] + b * cy[:, None, :]
    a, b = x[:, :, half:half + quarter], x[:, :, half + quarter:half + 2 * quarter]
    out[:, :, half:half + quarter] = a * cx[:, None, :] - b * sx[:, None, :]
    out[:, :, half + quarter:half + 2 * quarter] = a * sx[:, None, :] + b * cx[:, None, :]
    return out


def attention(qkv: torch.Tensor, heads: int, rope_grid: tuple[int, int] | None = None) -> torch.Tensor:
    nt, dim3 = qkv.shape
    dim = dim3 // 3
    head_dim = dim // heads
    q = qkv[:, :dim].reshape(nt, heads, head_dim)
    k = qkv[:, dim:2 * dim].reshape(nt, heads, head_dim)
    v = qkv[:, 2 * dim:].reshape(nt, heads, head_dim)
    if rope_grid is not None:
        q = apply_rope_2d(q, rope_grid[0], rope_grid[1], 100.0)
        k = apply_rope_2d(k, rope_grid[0], rope_grid[1], 100.0)
    q = q.permute(1, 0, 2).unsqueeze(0)
    k = k.permute(1, 0, 2).unsqueeze(0)
    v = v.permute(1, 0, 2).unsqueeze(0)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.squeeze(0).permute(1, 0, 2).reshape(nt, dim)


def qk_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps=1e-6)


def sem_forward(sem: dict[str, torch.Tensor], rgb: np.ndarray, proc_h: int, proc_w: int,
                dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    dim, heads, ps = 1024, 16, 14
    sem_g_h, sem_g_w = proc_h // 16, proc_w // 16
    sem_h, sem_w = sem_g_h * ps, sem_g_w * ps
    img = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    img = resize_chw((img - mean) / std, sem_h, sem_w).to(dtype)
    x = F.conv2d(img.unsqueeze(0), sem["pretrained.patch_embed.proj.weight"],
                 sem["pretrained.patch_embed.proj.bias"], stride=ps)
    x = x.flatten(2).transpose(1, 2).squeeze(0)
    cls = sem["pretrained.cls_token"].reshape(1, dim)
    pos = sem["pretrained.pos_embed"].reshape(1370, dim)
    if (sem_g_h, sem_g_w) != (37, 37):
        patch = pos[1:].reshape(37, 37, dim).permute(2, 0, 1)
        patch = resize_chw(patch, sem_g_h, sem_g_w).permute(1, 2, 0).reshape(-1, dim)
        pos = torch.cat([pos[:1], patch], dim=0)
    hidden = torch.cat([cls, x], dim=0) + pos
    for l in range(24):
        p = f"pretrained.blocks.{l}."
        y = F.layer_norm(hidden, (dim,), sem[p + "norm1.weight"], sem[p + "norm1.bias"], eps=1e-6)
        qkv = F.linear(y, sem[p + "attn.qkv.weight"], sem[p + "attn.qkv.bias"])
        y = attention(qkv, heads)
        y = F.linear(y, sem[p + "attn.proj.weight"], sem[p + "attn.proj.bias"])
        hidden = hidden + y * sem[p + "ls1.gamma"]
        y = F.layer_norm(hidden, (dim,), sem[p + "norm2.weight"], sem[p + "norm2.bias"], eps=1e-6)
        y = F.gelu(F.linear(y, sem[p + "mlp.fc1.weight"], sem[p + "mlp.fc1.bias"]), approximate="tanh")
        y = F.linear(y, sem[p + "mlp.fc2.weight"], sem[p + "mlp.fc2.bias"])
        hidden = hidden + y * sem[p + "ls2.gamma"]
    hidden = F.layer_norm(hidden, (dim,), sem["pretrained.norm.weight"], sem["pretrained.norm.bias"], eps=1e-6)
    return hidden[1:]


def dit_block(dit: dict[str, torch.Tensor], hidden: torch.Tensor, t_silu: torch.Tensor,
              layer: int, grid: tuple[int, int]) -> torch.Tensor:
    dim, heads = 1024, 16
    p = f"dit.blocks.{layer}."
    mod = F.linear(t_silu, dit[p + "adaLN_modulation.1.weight"], dit[p + "adaLN_modulation.1.bias"])
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
    y = F.layer_norm(hidden, (dim,), eps=1e-6) * (1.0 + scale_msa) + shift_msa
    qkv = F.linear(y, dit[p + "attn.qkv.weight"], dit[p + "attn.qkv.bias"])
    nt = qkv.shape[0]
    q = qk_norm(qkv[:, :dim].reshape(nt, heads, 64), dit[p + "attn.q_norm.weight"], dit[p + "attn.q_norm.bias"])
    k = qk_norm(qkv[:, dim:2 * dim].reshape(nt, heads, 64), dit[p + "attn.k_norm.weight"], dit[p + "attn.k_norm.bias"])
    qkv = torch.cat([q.reshape(nt, dim), k.reshape(nt, dim), qkv[:, 2 * dim:]], dim=-1)
    y = attention(qkv, heads, grid)
    y = F.linear(y, dit[p + "attn.proj.weight"], dit[p + "attn.proj.bias"])
    hidden = hidden + gate_msa * y
    y = F.layer_norm(hidden, (dim,), eps=1e-6) * (1.0 + scale_mlp) + shift_mlp
    y = F.gelu(F.linear(y, dit[p + "mlp.fc1.weight"], dit[p + "mlp.fc1.bias"]), approximate="tanh")
    y = F.linear(y, dit[p + "mlp.fc2.weight"], dit[p + "mlp.fc2.bias"])
    return hidden + gate_mlp * y


def pixel_shuffle_tokens(x: torch.Tensor, g_h: int, g_w: int) -> torch.Tensor:
    dim = x.shape[-1] // 4
    x = x.reshape(g_h, g_w, 2, 2, dim)
    return x.permute(0, 2, 1, 3, 4).reshape(g_h * 2 * g_w * 2, dim)


def dit_step(dit: dict[str, torch.Tensor], latent: torch.Tensor, cond: torch.Tensor,
             semantics: torch.Tensor, t_cur: float, dtype: torch.dtype) -> torch.Tensor:
    dim = 1024
    proc_h, proc_w = latent.shape
    lo = (proc_h // 16, proc_w // 16)
    hi = (proc_h // 8, proc_w // 8)
    inp = torch.cat([latent[None], cond], dim=0).unsqueeze(0).to(dtype)
    hidden = F.conv2d(inp, dit["dit.x_embedder.proj.weight"], dit["dit.x_embedder.proj.bias"],
                      stride=16).flatten(2).transpose(1, 2).squeeze(0)
    ts = sinusoidal_embed(t_cur, 256, latent.device).to(dtype)
    t_emb = F.silu(F.linear(ts, dit["dit.t_embedder.mlp.0.weight"], dit["dit.t_embedder.mlp.0.bias"]))
    t_emb = F.linear(t_emb, dit["dit.t_embedder.mlp.2.weight"], dit["dit.t_embedder.mlp.2.bias"])
    t_silu = F.silu(t_emb)
    for l in range(12):
        hidden = dit_block(dit, hidden, t_silu, l, lo)
    hidden = torch.cat([hidden, semantics.to(hidden.dtype)], dim=-1)
    hidden = F.silu(F.linear(hidden, dit["dit.proj_fusion.0.weight"], dit["dit.proj_fusion.0.bias"]))
    hidden = F.silu(F.linear(hidden, dit["dit.proj_fusion.2.weight"], dit["dit.proj_fusion.2.bias"]))
    hidden = F.linear(hidden, dit["dit.proj_fusion.4.weight"], dit["dit.proj_fusion.4.bias"])
    hidden = pixel_shuffle_tokens(hidden, lo[0], lo[1])
    for l in range(12, 24):
        hidden = dit_block(dit, hidden, t_silu, l, hi)
    mod = F.linear(t_silu, dit["dit.final_layer.adaLN_modulation.1.weight"],
                   dit["dit.final_layer.adaLN_modulation.1.bias"])
    shift, scale = mod.chunk(2, dim=-1)
    hidden = F.layer_norm(hidden, (dim,), eps=1e-6) * (1.0 + scale) + shift
    pred = F.linear(hidden, dit["dit.final_layer.linear.weight"], dit["dit.final_layer.linear.bias"])
    pred = pred.reshape(hi[0], hi[1], 8, 8).permute(0, 2, 1, 3).reshape(proc_h, proc_w)
    return pred.float()


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    rgb = read_ppm(Path(args.image))
    h, w = rgb.shape[:2]
    proc_h, proc_w = ((h + 15) // 16) * 16, ((w + 15) // 16) * 16
    t_load0 = time.perf_counter()
    sem_cpu = load_state(Path(args.sem))
    dit_cpu = load_state(Path(args.ppd))
    sem = to_device_state(sem_cpu, device, dtype, ("pretrained.",))
    dit = to_device_state(dit_cpu, device, dtype, ("dit.",))
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    t_load1 = time.perf_counter()
    t0 = time.perf_counter()
    semantics = sem_forward(sem, rgb, proc_h, proc_w, dtype, device)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    t1 = time.perf_counter()
    img = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0 - 0.5
    cond = resize_chw(img, proc_h, proc_w)
    latent = torch.from_numpy(c_randn(proc_h * proc_w).reshape(proc_h, proc_w)).to(device=device)
    t2 = time.perf_counter()
    timesteps = [1000.0, 750.0, 500.0, 250.0]
    for i, t_cur in enumerate(timesteps):
        t_next = timesteps[i + 1] if i + 1 < len(timesteps) else 0.0
        pred = dit_step(dit, latent, cond, semantics, t_cur, dtype)
        t_ratio, s_ratio = t_cur / 1000.0, t_next / 1000.0
        pred_x0 = latent - t_ratio * pred
        pred_x_t = latent + (1.0 - t_ratio) * pred
        latent = (1.0 - s_ratio) * pred_x0 + s_ratio * pred_x_t
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    t3 = time.perf_counter()
    depth = latent + 0.5
    if (proc_h, proc_w) != (h, w):
        depth = resize_chw(depth[None], h, w).squeeze(0)
    out = depth.detach().float().cpu().numpy()
    if args.npy:
        np.save(args.npy, out)
    print(f"load: {(t_load1 - t_load0) * 1000:.1f} ms")
    print(f"semantic: {(t1 - t0) * 1000:.1f} ms")
    print(f"dit: {(t3 - t2) * 1000:.1f} ms")
    print(f"total_inference: {(t3 - t0) * 1000:.1f} ms")
    print(f"range: [{out.min():.6f}, {out.max():.6f}], mean={out.mean():.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppd", default="/home/syoyo/work/gemm/vlm-ptx/models/ppd/ppd.pth")
    ap.add_argument("--sem", default="/home/syoyo/work/gemm/vlm-ptx/models/ppd/depth_anything_v2_vitl.pth")
    ap.add_argument("-i", "--image", default="/home/syoyo/work/gemm/vlm-ptx/cuda/ppd/street.ppm")
    ap.add_argument("--npy", default="/tmp/ppd_ref.npy")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    args = ap.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
