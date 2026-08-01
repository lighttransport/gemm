# RDNA4 VLM Vision Encoder — HIP vs PyTorch/ROCm

Performance comparison of the `rdna4/vlm` HIP vision encoder against a
PyTorch/ROCm reference, on **identical model weights and GPU**.

Date: 2026-05-15 · GPU: AMD Radeon RX 9070 XT (gfx1201) · dtype: BF16

## Model under test

Both paths run the **same vision tower** — the Qwen3-VL-30B-A3B / Qwen3.6-27B
visual encoder:

| param | value |
|-------|-------|
| depth (blocks) | 27 |
| hidden_size    | 1152 |
| intermediate_size (ffn) | 4304 |
| num_heads      | 16 |
| **head_dim**   | **72** (1152 / 16 — not a power of 2) |
| patch_size     | 16 |
| spatial_merge  | 2 |
| temporal_patch | 2 |
| params         | 539 M |

- **HIP**: `./test_hip_vision` on
  `/mnt/disk1/models/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf`
- **PyTorch**: `bench_pytorch_vision.py` loads the HF visual tower
  (`Qwen/Qwen3-VL-30B-A3B-Instruct`, shard 13 — all `model.visual.*` weights),
  class `Qwen3VLMoeVisionModel`, transformers 5.5.4, torch 2.11.0+rocm7.2.

Both use synthetic square inputs, warmed up, steady-state timed. Token counts
match exactly (`tokens = (N/16)² / 4`).

## Results — after fix (WMMA flash-attention enabled by default)

| image  | tokens | HIP **before** ms | HIP **after** ms | PyTorch ms | speedup vs before | **HIP vs PyTorch** |
|-------:|-------:|------------------:|-----------------:|-----------:|------------------:|-------------------:|
| 512²   |   256  |             306.4 |         **34.4** |       48.7 |             8.9×  | **1.42× faster**   |
| 1024²  |  1024  |            3233.0 |        **190.3** |      219.6 |            17.0×  | **1.15× faster**   |
| 2048²  |  4096  |           50796.5 |       **1831.3** |     5400.9 |            27.7×  | **2.95× faster**   |

After fix, HIP tok/s: 512²=7435, 1024²=5380, 2048²=2237.
HIP: `--warmup 3 --iters 10 --bf16`. PyTorch: `--warmup 5 --iters 20 --dtype bf16`,
back-to-back submission, single sync (mean ≈ min ≈ b2b, <2% spread).

> **The HIP vision encoder is now 1.15–2.95× faster than the PyTorch/ROCm
> reference** on the same model and GPU — up from 6–15× *slower*.

Correctness: WMMA path vs the scalar-F32 path, 1024² — cosine **0.99994**,
rel_l2 1.06e-2 (small expected BF16-vs-F32 delta). PASS.

Note: the comparison *favors* HIP — its GGUF mmproj has deepstack disabled,
while the PyTorch tower runs the full model including the 3 deepstack mergers
(layers 8/16/24).

## Root cause (original 6–15× slowdown)

The HIP codebase **already had** BF16 WMMA flash-attention kernels that
**already supported head_dim = 72** (they pad head_dim to `HD_PAD = 80` in LDS
with `if (d < head_dim)` bounds checks). But the attention dispatcher in
`hip_vision_encoder.c` only selected a WMMA kernel when the `HIP_VLM_FA`
environment variable was explicitly set. With no env var, head_dim = 72 fell
through to the scalar `flash_attn_dyn_f32` — an O(n²) scalar kernel ~10× slower.
The WMMA path was dead code purely because of the env gate, which is why
attention scaled super-quadratically (1024²→2048² was 15.7×).

## Fix

`hip_vision_encoder.c`, attention dispatch (~line 3945): added a `fa_auto`
branch — when `HIP_VLM_FA` is unset and `head_dim <= 80`, default to
`flash_attn_wmma_bf16_4w_pre` (BQ=64, 4-wave, double-buffered pre-packed K/V).
`HIP_VLM_FA=tiled` still forces the old scalar path. ~5 lines; all buffers,
kernel handles, and the K/V pre-pack transpose were already wired.

## Reproduce

```sh
cd rdna4/vlm
# HIP
./test_hip_vision \
  /mnt/disk1/models/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf \
  --image-size 1024 --warmup 3 --iters 10 --bf16

# PyTorch/ROCm reference (HF weights cached under /mnt/disk1/models/hf_cache)
VENV=/mnt/disk1/work/gemm/main/rdna4/trellis2/.venv
MD=/mnt/disk1/models/hf_cache/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/*
$VENV/bin/python bench_pytorch_vision.py --model-dir $MD \
  --sizes 512,1024,2048 --warmup 5 --iters 20 --dtype bf16
```
