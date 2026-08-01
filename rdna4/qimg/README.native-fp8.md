# Native FP8 Reference Setup

This directory owns a local `uv` environment for Qwen-Image FP8 diagnostics.
Use Python 3.12 because the ROCm 7.2.2 PyTorch wheels are `cp312`.

```bash
cd rdna4/qimg
UV_CACHE_DIR=$PWD/.uv-cache uv venv --python 3.12 --clear .venv
UV_CACHE_DIR=$PWD/.uv-cache uv sync
UV_CACHE_DIR=$PWD/.uv-cache uv pip install \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/triton-3.6.0+rocm7.2.2.git4ed88892-cp312-cp312-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torch-2.11.0+rocm7.2.2.lw.git4e323059-cp312-cp312-linux_x86_64.whl
```

Verify outside Codex sandbox if `/dev/kfd` is hidden:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'
```

Current expected result on the RX 9070 XT box:

```text
2.11.0+rocm7.2.2.git4e323059 7.2.53211 True
AMD Radeon RX 9070 XT
```

## Qwen-Image FP8 Reference Modes

Keep these modes distinct when comparing qimg against PyTorch/ComfyUI:

- `comfy-default`: the Qwen-Image DiT safetensors file stores selected weights
  as `float8_e4m3fn`, but normal linear layers dequantize/cast those weights
  to the active compute dtype and run BF16/FP16/FP32 math. Current ComfyUI marks
  Qwen-Image inference dtypes as BF16/FP32, so do not treat the stock workflow
  as all-FP8xFP8.
- `comfy-fast-fp8`: ComfyUI's opt-in fast FP8 matrix-multiplication path. It
  clamps activations to the FP8 E4M3 range, casts activations to FP8, uses the
  stored FP8 weights, applies unit input/weight scales, and returns the result
  in the input compute dtype. In qimg this is exposed as
  `--fast fp8_matrix_mult`. The local PyTorch scripts use the
  `comfy-fast-fp8` name; the older `SCALE_MODE=comfy` spelling remains an alias.
- `qimg-quality-gated`: qimg's quality-safe mode. **Default since 2026-05-23:**
  for FP8 checkpoints qimg automatically routes the perceptually-lossless 48 dB
  tier (`img_mlp_fc1`, blocks >=8, clamp activation quant) to FP8xFP8 WMMA; all
  other GEMMs stay BF16xFP8. This is the 29%-of-pool operating point measured at
  49.67 dB latent / 54.7 dB PNG PSNR (indistinguishable from pure BF16xFP8).
  Opt out to pure BF16xFP8 (51.70 dB) with `--fast none` or
  `QIMG_FP8_WMMA_BF16=1`. `--fast fp8_matrix_mult` switches to the aggressive
  all-eligible ComfyUI path (scale=1, ~24 dB) and can be narrowed with explicit
  label/block selectors while probing quality.

The local Qwen-Image DiT checkpoint is raw FP8 storage. Its safetensors header
has no scale metadata (`scale`, `input_scale`, `weight_scale`, `fp8`, or
`quant` keys were absent in the 2026-05-20 inspection). Representative tensors
such as `img_in.weight` and `transformer_blocks.0.attn.to_q.weight` are
`F8_E4M3`; qimg and the local PyTorch probes therefore must supply any
activation scaling policy themselves.

Low-risk pinned checks:

```bash
.venv/bin/python tools/native_fp8_ref.py compare-latents
.venv/bin/python tools/native_fp8_ref.py quant-sweep --input ../../ref/qwen_image/final_latent_packed_256.bin --cols 64
```

Native FP8 GEMM probes:

```bash
.venv/bin/python tools/native_fp8_ref.py gemm-probe --m 256 --n 256 --k 256 --scale-div 512
.venv/bin/python tools/native_fp8_ref.py gemm-probe --m 256 --k 64 --scale-div 512 \
  --w-safetensors /mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
  --tensor img_in.weight
.venv/bin/python tools/native_fp8_ref.py gemm-probe --m 128 --k 3072 --n 3072 --scale-div 512 \
  --w-safetensors /mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
  --tensor transformer_blocks.0.attn.to_q.weight
```

Pinned BF16xFP8 quality / traffic check:

```bash
QIMG_FP8_WMMA=1 ./test_hip_qimg --generate \
  --dit /mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
  --height 256 --width 256 --steps 20 \
  --init-bin ../../ref/qwen_image/init_latent_256.bin \
  --txt-bin ../../ref/qwen_image/apple_text_256.bin \
  --sigmas-bin ../../ref/qwen_image/sigmas_256.bin \
  --ref-final ../../ref/qwen_image/final_latent_packed_256.bin \
  --path-stats --mem-stats
```

Current pinned result: latent cosine `0.999971`, PSNR `51.70 dB`,
denoise `24.1 s`, GEMM traffic `1723 GB / 73.1 TF`.

ComfyUI-style fast FP8xFP8 check:

```bash
./test_hip_qimg --generate --fast fp8_matrix_mult \
  --dit /mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
  --height 256 --width 256 --steps 20 \
  --init-bin ../../ref/qwen_image/init_latent_256.bin \
  --txt-bin ../../ref/qwen_image/apple_text_256.bin \
  --sigmas-bin ../../ref/qwen_image/sigmas_256.bin \
  --ref-final ../../ref/qwen_image/final_latent_packed_256.bin \
  --path-stats --mem-stats
```

Current pinned result: latent cosine `0.979988`, PSNR `24.59 dB`,
denoise `20.3 s`, GEMM traffic `984 GB / 73.1 TF`, extra persistent
activation-FP8 scratch `3.0 MB`.

Quality-target gated FP8xFP8:

```bash
./test_hip_qimg --generate ... --fp8-quality-target-db 50
./test_hip_qimg --generate ... --fast fp8_matrix_mult --fp8-quality-target-db 50 \
  --fp8-fp8-allow img_in
```

By default (since 2026-05-23), qimg routes the quality-safe 48 dB FP8xFP8 tier
automatically: `img_mlp_fc1` in blocks >=8 runs activation-FP8 x weight-FP8 WMMA
(clamp activation quant), every other eligible GEMM stays BF16xFP8. This is
49.67 dB latent / 54.7 dB PNG — perceptually identical to pure BF16xFP8 (51.70
dB) — and cuts denoise ~13–15% (1024² 8.94->7.78 s/step, 256² 1.13->0.96).
The early fc1 layers (blocks 0..7) carry the intrinsic e4m3 activation-mantissa
error (floors ~44 dB) so they stay BF16xFP8.

Opt out to pure BF16xFP8 with `--fast none` or `QIMG_FP8_WMMA_BF16=1`.
`--fast fp8_matrix_mult` switches to the aggressive all-eligible ComfyUI path
(activation scale=1, ~24 dB). Label/block selectors (`--fp8-fp8-allow`,
`--fp8-fp8-block-min/max`) narrow the fast path for quality probes.

As of the 2026-05-20 quality-gate pass, the full 20-step pinned run exits
nonzero when `--fp8-quality-target-db` and `--ref-final` are both set and the
final latent PSNR is below target. The conservative no-allowlist path remains
the current quality-safe mode:

```text
no allowlist: fp8xfp8_wmma=0, bf16xfp8_wmma=7240,
              latent cosine=0.999971, PSNR=51.70 dB, PASS
```

Do not allowlist `img_in` by default. It only enables 20 FP8xFP8 GEMMs in the
full run, but still breaks the denoise trajectory:

```text
--fp8-fp8-allow img_in: fp8xfp8_wmma=20,
                        latent cosine=0.829002, PSNR=14.60 dB, FAIL
```

Single-label 20-step pinned sweeps show no full-label FP8xFP8 allowlist clears
the 50 dB target yet:

```text
label          PSNR dB   status
img_k           48.26    FAIL
img_q           44.34    FAIL
img_mlp_fc1     34.40    FAIL
img_mlp_fc2     31.65    FAIL
img_attn_out    28.55    FAIL
img_v           23.45    FAIL
img_in          14.60    FAIL
```

The useful opening is block-gated `img_k`. Blocks `0..8` pass at 50.59 dB;
blocks `0..9` fail at 49.95 dB, and all blocks fail at 48.26 dB:

```bash
rdna4/qimg/test_hip_qimg --generate ... --fast fp8_matrix_mult \
  --fp8-quality-target-db 50 --fp8-fp8-allow img_k \
  --fp8-fp8-block-min 0 --fp8-fp8-block-max 8
```

Changing the per-row activation scale divisor is not a fix either. A standalone
final-latent quant sweep prefers `div=448` over `512`, but the actual DiT run
with `--fp8-act-scale-div 448` regresses to `cosine=0.954461`, `PSNR=21.14 dB`.
The failure is trajectory sensitivity from layerwise FP8xFP8 error, not only
final-latent quantization noise.

Use the serial sweep helper to test candidate labels without concurrent ROCm
processes fighting over VRAM:

```bash
python3 rdna4/qimg/tools/sweep_fp8fp8_quality.py --include-none
python3 rdna4/qimg/tools/sweep_fp8fp8_quality.py \
  --labels img_q,img_k,img_v,img_attn_out,img_mlp_fc1,img_mlp_fc2
python3 rdna4/qimg/tools/sweep_fp8fp8_quality.py \
  --labels img_k --block-min 0 --block-max 8 \
  --reference-mode qimg-quality-gated
```

The helper writes `summary.json`, `summary.csv`, and one log per candidate to
`/tmp/qimg_fp8fp8_sweep` by default. It runs all selected candidates, then
exits nonzero if any candidate misses the configured quality target. Each row
also records the reference mode label (`comfy-default`, `comfy-fast-fp8`, or
`qimg-quality-gated`) so later reports do not mix Comfy's default storage path
with its opt-in fast FP8xFP8 path.

Activation scale experiments:

```bash
./test_hip_qimg --generate ... --fast fp8_matrix_mult
./test_hip_qimg --generate ... --fast fp8_matrix_mult --fp8-act-scale-mode clamp
```

`--fast fp8_matrix_mult` is the qimg CLI spelling for the `comfy-fast-fp8`
diagnostic (`scale=1`, clamp/cast). It saturates later qimg activations in the
full denoise run: latent cosine `0.301305`, PSNR `4.20 dB`. The experimental
`clamp` mode keeps scale 1 only while the row fits FP8 range and otherwise uses
`row_max/448`; it improves the full run to cosine `0.986565`, PSNR `25.93 dB`,
still far below the 50 dB target without BF16 fallback/gating. Image artifact:
`apple_fp8fp8_clamp_256.png`.

## Maximizing quality-safe FP8xFP8 coverage (2026-05-20)

Goal: move as many WMMA-eligible GEMMs as possible to the ~2x FP8xFP8 path
while keeping the pinned 256x256 20-step run at or above the 50 dB latent gate.
Baseline (BF16xFP8, no FP8xFP8) is `psnr_peak=51.70 dB`, so the entire
degradation budget to stay >=50 dB is only ~1.7 dB. The WMMA-eligible compute
pool at 256x256 is `bf16xfp8_wmma=69.58 TF`; every TF moved to FP8xFP8 runs at
roughly 2x. Use `tools/run_one.sh <tag> <args...>` for single, serial probes
(one 20 GB checkpoint load at a time — do not run these concurrently).

**Per-GEMM TF weight** (256x256, all 60 blocks): `img_mlp_fc1` and
`img_mlp_fc2` are 23.19 TF each (33% of the pool each, 67% combined);
`img_q`/`img_k`/`img_v`/`img_attn_out` are 5.80 TF each. So the FFN dominates
the achievable speedup and attention projections are minor.

**`clamp` is the universal-best activation mode** for the selective path
(per-label, all blocks):

```text
label          comfy(scale=1)   per-row(div512)   clamp
img_mlp_fc1        34.40            42.43          43.91
img_mlp_fc2        31.65            -9.96          (poison)
img_k              48.26            45.12          48.11
img_q              44.34            41.54          45.64
img_v              23.45            34.28          (lossy)
img_attn_out       28.55            19.58          (lossy)
```

`clamp` matches or beats both other modes everywhere, so always pass
`--fp8-act-scale-mode clamp` with `--fast fp8_matrix_mult` for quality-safe
work. `comfy` scale=1 is exact for in-range rows (good for the well-behaved
`img_k`) but saturates outliers; per-row helps outlier-heavy rows but quantizes
the scale; `clamp` adaptively picks scale=1 when the row fits FP8 and
`row_max/448` otherwise. `img_mlp_fc2` is unusable (its input is the post-GELU
FFN intermediate with huge dynamic range — `-9.96 dB` even on its own).

**Layer-depth sensitivity is the dominant effect, and it is opposite for FFN
vs attention.** FP8 error does not accumulate uniformly across the 60 DiT
layers; it localizes:

- `img_mlp_fc1` damage lives in the *early* layers. All-blocks, 0..29, 0..14,
  and 0..7 all floor at ~44 dB (the kept early layers set the floor); skipping
  them lifts quality:

  ```text
  img_mlp_fc1 clamp   block range   PSNR     captured FP8xFP8 TF
                      0..59         43.91    23.19  (FAIL)
                      8..59         49.67    20.10  (FAIL, ~48 dB safe)
                      30..59        50.07    11.60  (PASS 50 dB)
                      45..59        50.37     5.80  (PASS, more margin)
  ```

- `img_k`/`img_q` are the reverse: safe on *early* layers, damaging on late
  ones (`img_k` 0..8 = 50.59 PASS upstream; on 30..59 it degrades).

Because the global `--fp8-fp8-block-min/max` range applies to *all* allowed
labels at once, you cannot capture "fc1 on late layers" and "k on early layers"
in one run. Combining is destructive at this budget:

```text
img_mlp_fc1,img_k,img_q clamp 30..59       45.07 dB  (FAIL; k/q hate late layers)
all img labels          clamp 45..59       34.58 dB  (FAIL; v/attn_out lossy)
```

**Recommended quality-safe operating points** (256x256, 50 dB gate):

```bash
# Max coverage that still passes 50 dB: fc1 on the deep half of the stack.
./test_hip_qimg --generate ... --fast fp8_matrix_mult \
  --fp8-act-scale-mode clamp --fp8-fp8-allow img_mlp_fc1 \
  --fp8-fp8-block-min 30 --fp8-quality-target-db 50
# -> 50.07 dB PASS, 11.60 TF FP8xFP8 (17% of the eligible pool)

# Same with more headroom (45..59), fewer TF:
#   --fp8-fp8-block-min 45  -> 50.37 dB, 5.80 TF
# Relaxed 48 dB target unlocks fc1 8..59 -> 20.10 TF (29% of the pool).
```

**Bottom line:** the 50 dB gate is the binding constraint, not throughput.
Quality-safe FP8xFP8 caps at ~17% of WMMA-eligible compute (fc1 on deep
layers); the 67%-TF FFN pair is mostly off-limits because fc1's early layers
and all of fc2 break the gate, and attention projections add little TF and have
opposite layer-depth tolerance.

### Block-wise activation scaling does NOT help (2026-05-20, measured)

Tested directly via the extended `--fp8-quant-stats` (set
`QIMG_FP8_QUANT_STATS_GBLK=<group>`), which now reports the per-row `mae`
alongside `mae_blk<g>` (per-row, per-K-group scale) and `mae_tensor` (single
scale) for the real activations. For `img_mlp_fc1` the three are bit-identical
across all 60 layers:

```text
block=0   mae=0.050695  mae_blk32=0.050695  mae_tensor=0.050695  zeros=0.46%
block=29  mae=0.12354   mae_blk32=0.12354   mae_tensor=0.12354   zeros=0.04%
block=59  mae=0.38337   mae_blk32=0.38334   mae_tensor=0.3841    zeros=0.00%
```

Reason: e4m3 is *floating point* with ~constant relative precision (3 mantissa
bits), so finer scale granularity only helps when it prevents clipping
(`sat=0.00%`) or underflow (`zeros<0.5%`) — neither occurs here. The quantized
activation tensor is identical at any granularity, so the GEMM output is
identical. The early-layer fc1 floor (~44 dB) is intrinsic e4m3
activation-mantissa error amplified through layer depth, not a scaling problem.
This is an INT8 intuition that does not transfer to FP8. The same argument
rules out the "outlier-split BF16 lane" idea — there are no range outliers to
split.

The only remaining levers for *coverage* are therefore (a) **layer-depth
gating** (done: fc1 deep layers, 17% at 50 dB) and (b) **the quality threshold**
(48 dB unlocks fc1 8..59 = 29%). Improving the activation quantizer is not a
viable lever. (Kernel *throughput* is a separate axis — see below: our
hand-written kernel runs at ~60-70% of the vendor FP8 GEMM.)

### The 50 dB latent gate is too strict — 48 dB is perceptually lossless (2026-05-20)

Decoded both operating points through the VAE and compared the PNGs (256x256,
`--vae .../qwen_image_vae.safetensors -o out.png`):

```text
comparison                     PNG PSNR   mean|d|   max|d|   pix>5/255
baseline vs FP8 (fc1 8..59)     54.72 dB   0.21      8        0.00%
baseline vs diffusers-ref       46.63 dB   0.82      18       0.10%
FP8 vs diffusers-ref            46.10 dB   0.87      17       0.14%
```

The baseline-vs-FP8 image delta (54.72 dB, max 8/255, zero pixels off by >5) is
~8 dB tighter than the gap the BF16xFP8 HIP port *already* has to the true
diffusers reference (46.63 dB). The 48 dB latent point perturbs the rendered
image less than the port already differs from PyTorch — visually identical.

**Recommendation:** the 50 dB *latent* gate is well inside existing port noise;
ship the 48 dB latent point to roughly double quality-safe FP8 coverage:

```bash
./test_hip_qimg --generate ... --fast fp8_matrix_mult \
  --fp8-act-scale-mode clamp --fp8-fp8-allow img_mlp_fc1 --fp8-fp8-block-min 8 \
  --fp8-quality-target-db 48
# -> 49.67 dB latent, 20.10 TF FP8xFP8 (29% of the eligible pool), PNG indistinguishable
```

## FP8 GEMM throughput vs ComfyUI fast-fp8 (2026-05-20)

ComfyUI's `--fast fp8_matrix_mult` calls `torch._scaled_mm` (scale=1), which on
gfx1201/ROCm 7.2.2 dispatches a hipBLASLt/Tensile FP8 kernel. qimg's FP8xFP8
path instead uses a hand-written HIPRTC WMMA kernel
(`gemm_fp8_fp8w_perrow_pgr2`) because hipBLASLt's *C++* FP8 entry point is
broken on this stack. Two low-memory microbenches (one GEMM at a time — the
full PyTorch pipeline OOMs a 16 GB card) on the real Qwen-Image DiT shapes
(hidden=3072, mlp_h=12288), GEMM-only (activation cast excluded):

- `tools/bench_comfy_fast_fp8.py` — `torch._scaled_mm` vs bf16 `F.linear`.
- `tools/bench_qimg_fp8_kernel.c` — qimg's `gemm_fp8_fp8w_perrow_pgr2`.

```text
                       M=256 (256x256)            M=4096 (1024x1024)
gemm (NxK)        bf16   comfy   qimg          bf16   comfy   qimg   qimg/comfy
attn q/k/v/out    69.1   111.7   70.2          119.6  228.8  138.7      0.61x
mlp_fc1 (12288)   85.0   178.1   83.6          127.5  221.9  154.7      0.70x
mlp_fc2 (K=12288) 87.4   169.4   89.0          114.2  211.9   91.8      0.43x
mod (18432)       90.9   194.3   82.6          126.1  224.0  149.6      0.67x
        (TF/s; backend check mm0: extracted-vendor 214.8 ~= _scaled_mm 208.5)
```

Findings:
- ComfyUI fast-fp8 gives a clean ~1.7-2.1x GEMM speedup over its own bf16
  default and saturates the FP8 ceiling (~210-229 TF/s at M=4096).
- qimg's hand-written kernel reaches ~60-70% of that on attn/fc1/mod. fc1 (the
  GEMM we actually FP8-ify for quality) is the best case at 154.7 TF/s (0.70x).
- `mlp_fc2` (deep K=12288) is qimg's weak spot (0.43x), but it is poison for
  quality anyway (-9.96 dB) and never enabled — so the throughput gap that
  matters in practice is the ~0.70x on fc1.
- Closing the gap means either fixing the hipBLASLt C++ FP8 dispatch (so qimg
  can call the vendor kernel directly, as torch does) or further WMMA tuning;
  `rdna4/fp8` notes the hand-written kernel is near the limit of the
  scheduling/LDS levers tried so far (~44% of the 8-wave WMMA microbench peak).

**Net:** ComfyUI's fast path is faster at the GEMM level but applies fp8 to all
layers -> 4.20 dB (unusable). qimg trades raw GEMM throughput for a quality-safe
gate; the open perf item is the ~0.70x kernel gap on fc1, independent of the
coverage/quality work above.

## Vendor (Tensile) FP8 kernel dispatch — opt-in x1.0 path (2026-05-21)

`QIMG_FP8_VENDOR=1` makes the FP8xFP8 path dispatch the extracted vendor
(hipBLASLt/Tensile) FP8 GEMM instead of the hand-written
`gemm_fp8_fp8w_perrow_pgr2` — the same kernel class `torch._scaled_mm` /
ComfyUI use. This reaches GEMM parity (x1.0): on qimg shapes it is 1.5-2.6x the
hand-written kernel (fc1 4096x12288x3072: 154.7 -> 233.5 TF/s).

- Loads `rdna4/fp8/fp8_kernel_gfx1201.co` (build artifact; regenerate via
  `rdna4/fp8/extract_fp8_kernel.sh`). Path override: `QIMG_FP8_VENDOR_CO`.
  Launched through `hipModuleLaunchKernel` with the 172-byte Tensile kernarg
  (`qimg_launch_fp8_vendor`); per-column F32 bias fused via the kernel's bias
  slot (BIAS_TYPE=0). Falls back to the hand-written kernel if the .co is
  absent.
- End-to-end (1024x1024, all-img FP8, comfy): denoise 88.0s -> 79.8s (+9%).
  The win is bounded because 1024^2 denoise is attention-bound, not
  FP8-GEMM-bound; the FP8 GEMM is only part of the step.
- **Quality caveat (why it is opt-in, not default):** the vendor kernel takes a
  *scalar* activation scale, so it forces scale=1 quant. That is ~1.5 dB below
  the hand-written `clamp` per-row path on knife-edge configs (fc1 30..59:
  48.62 dB vendor vs 50.07 dB hand-written; fc1 8..59: 48.15 dB). It still
  clears the 48 dB perceptually-lossless tier, but would silently break the
  documented 50 dB configs, so it is OFF by default.
- **Per-tensor clamp scale does NOT recover the quality (tried 2026-05-21).**
  Implemented the global-max-abs reduce + per-tensor `clamp` scale fed to the
  vendor `scaleB`, expecting the per-tensor == per-row MAE equivalence to make
  it quality-neutral. It did not: fc1 30..59 gave 47.38 dB — *worse* than even
  scalar scale=1 (48.62), vs hand-written per-row clamp 50.07. The MAE
  equivalence holds for activation *reconstruction* but not for the GEMM output
  / denoise trajectory: per-row clamp leaves small-max rows exact (s=1) while a
  single per-tensor scale rescales every row, and the 20-step trajectory is
  sensitive to that. So the vendor (scalar-scale) kernel tops out at ~48.62 dB
  and cannot match the hand-written per-row path; it stays **opt-in** (plain
  scale=1 is its best mode). Reverted the per-tensor machinery.

## Rowwise-scale vendor kernel — identified, not wired (2026-05-21)

To make the vendor path default-on with per-row quality, the kernel must support
a per-row activation scale. hipBLASLt has one and torch._scaled_mm uses it:

```
Cijk_Alik_Bljk_F8BS_BH_Bias_HA_S_SABV_SAV_UserArgs_MT128x128x32_..._MIWT4_4_..._PGR1
```

(`F8BS`/`SABV` = scale-A-B-Vector, vs our scalar `F8SS`/`SAB`/MT128x128x64.)
Confirmed via rocprofv3 that torch._scaled_mm with `scale_a=[M,1]` dispatches it,
cos=0.999999 — **but it only outputs BF16** (`hipblaslt rowwise _scaled_mm only
supports BFloat16 output`). It lives in a separate library
(`TensileLibrary_B8B8_..._SAV_..._Alik_Bljk_gfx1201.co`). Our existing F8SS
kernel's `_SAV_` slot cannot be used by just passing a scaleAlphaVec pointer —
that produced garbage (cos 0.29); the rowwise control-word flags were not
captured.

Wiring it in is a full second extraction (extract the F8BS ELF, capture its
*different* kernarg ABI via dump_kernarg_shim — vector scaleA/B + SABV control
words, the F8SS offsets/magic do not transfer — plus a BF16->F32 conversion
pass since qimg is F32). Deferred: the payoff is bounded because 1024^2 denoise
is attention-bound, so even a perfect default-on rowwise vendor gives only ~9%
e2e (and less for quality-safe coverage). The opt-in scalar F8SS path
(`QIMG_FP8_VENDOR=1`, 48 dB tier) remains the shipped x1.0-GEMM option.
