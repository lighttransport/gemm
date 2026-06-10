# SVDQuant unit tests — status & RDNA4 resume

Self-contained, model-free validation of the **SVDQuant** forward (SmoothQuant λ
+ rank-128 SVD low-rank + 4-bit residual) across **INT4** and **NVFP4**, both
**W4A16** and **W4A4**. CPU and CUDA are done and green; RDNA4 (HIP) is the next
target.

Branch: `svdquant-unit-tests` (commits `ba378f5`, `be78cc5`) — **local, unpushed**.

---

## Status

| Target | Dir | State | Result |
|--------|-----|-------|--------|
| PyTorch ref driver | `ref/svdquant/` | ✅ done | dumps 4 cases, all cos>0.998 vs fp |
| CPU | `cpu/svdquant/` | ✅ done | 4/4 PASS, rel_L2(impl,y_svdq) ~5.5e-8 (gate 1e-5) |
| CUDA | `cuda/svdquant/` | ✅ done | 4/4 PASS (SGEMM ~3e-7 gate 2e-4; native fp4 HW ~2e-7 gate 1e-2) |
| **RDNA4 (HIP)** | `rdna4/svdquant/` | ⏳ **TODO** | — |

Key findings (the payoff of the harness):
- CPU `sq_quant_act_int4_g64` uses `rintf` (round-half-to-even) → **bit-matches
  `torch.round`**; INT4-W4A4 activation drift vs reference = **exactly 0.0**.
- The sm_120a `fp4_w4a4` HW kernel's on-device e4m3+e2m1 activation quant
  **bit-matches** numpy `float8_e4m3fn`+bucketize within f32 noise (~2e-7).
- `--cross-check` drives deepcompressor's real `LowRankBranch` (exact
  `torch.linalg.svd`): ours (randomized `svd_lowrank`) is within **~0.03%** of the
  exact residual lower bound, subspaces agree ~3%.
- `--real-nunchaku` decodes a real 3072² DiT weight + real activation from
  `nunchaku_ref_dump.*.safetensors` (low-rank exact via the installed Nunchaku
  packer; int4-main byte-swizzle is **nunchaku-version-specific**, so Ŵ vs real-y
  is only rel~0.60 — documented, not chased). Both CPU+CUDA still PASS on it
  (quant floor rises to ~0.08–0.13 as real weights are harder than Gaussian).

---

## The contract (what RDNA4 must reproduce)

Forward every implementation reproduces (from `ref/svdquant/gen_svdquant_ref.py`,
itself the math of `rdna4/qimg/tools/svdquant_from_bf16.py:reconstruct_and_check`):

```
y = act(x / lam) @ R_dec^T  +  (x @ lora_down_emit^T) @ lora_up^T  +  bias
```
- residual term uses smoothed `x/lam`; **low-rank uses RAW x**; `lora_down_emit =
  lora_down/lam` (dumped already folded).
- `R_dec` = decoded 4-bit residual: INT4 signed `[-8,7]` group-64, or NVFP4 e2m1
  group-16 × e4m3 micro-scales × per-row `wcwt`.
- `act(·)` = identity (W4A16) or 4-bit per-token-group quant→dequant (W4A4).

Reference dumps: `ref/svdquant/dumps/*.npy` (regen: `cd ref/svdquant && python3
gen_svdquant_ref.py --out dumps`). **f32/i32/u8 only** (the `common/npy_io.h`
loader reads exactly `f4`/`i4`/`u1`). `dims.npy` = i32 `[OUT,IN,TOK,RANK]`.
Per-case prefix `<fmt>_<scope>_` ∈ {int4,nvfp4}×{w4a16,w4a4}: `smooth`,
`lora_up`(OUT×R), `lora_down`(R×IN, /lam), `y_svdq`(the target); INT4 adds
`qint4`(u8 OUT×IN/2)+`wscale`(f32 OUT×IN/64); NVFP4 adds `qw`(i32 OUT×IN/8)+
`ws`(u8 OUT×IN/16)+`wcwt`(f32 OUT); W4A4 adds `xr_dq`(f32 TOK×IN). Shared: `x`,
`W`, `bias`, `y_fp`.

The decode + forward already exist in C: **reuse `cpu/svdquant/svdquant_cpu.h`**
(`sq_unpack_int4_residual`, `sq_unpack_nvfp4_residual`, `sq_e2m1_decode`,
`sq_ue4m3_decode`, `sq_quant_act_int4_g64`, `sq_smooth_div`, `sq_forward`) for the
host side — it's portable C, no CUDA. The CUDA test
(`cuda/svdquant/test_cuda_svdquant.c`) is the structural template.

---

## RDNA4 resuming prompt (copy-paste to a fresh session)

> Build `rdna4/svdquant/` — a HIP/ROCm validation of the SVDQuant forward against
> the existing PyTorch reference dumps in `ref/svdquant/dumps`, mirroring
> `cuda/svdquant/`. Read `svdquant.md` and the memory `project_svdquant_tests.md`
> first.
>
> Reuse, don't reinvent:
> - Host decode/quant + the f64 oracle forward: `cpu/svdquant/svdquant_cpu.h`
>   (portable C — include it directly).
> - `.npy` loader + metrics: `common/npy_io.h` (`npy_load`, `npy_max_abs_f32`).
> - HIP plumbing: `rdna4/rocew.{c,h}` (HIPRTC runtime kernel compile — **no
>   hipcc**; gcc only), `rdna4/hip_runner_common.h`, `rdna4/hip_kernels_common.h`.
> - **Existing INT4 SVDQuant HIP kernels** in `rdna4/qimg/hip_qimg_kernels.h`:
>   `dequant_int4_logical_main_f32` (int4 logical→f32, line ~1322),
>   `gemm_int4w_bf16a_wmma_t` (fused W4A16 int4×bf16 WMMA, line ~1611). The runner
>   `rdna4/qimg/hip_qimg_runner.c:op_linear_int4w_bf16a` already does the full
>   INT4 W4A16 SVDQuant forward (smooth-div + int4 main + low-rank + bias) — read
>   it as the reference wiring.
> - GEMM helper: there is no rocBLAS wrapper analogous to `cublasew`; RDNA4 GEMMs
>   are HIPRTC kernels. For a correctness test, the simplest path is a small
>   HIPRTC dequant kernel (int4/nvfp4 residual → f32) + a plain f32 WMMA/tiled
>   GEMM kernel for the residual and the two low-rank GEMMs (or reuse an existing
>   f32 GEMM kernel from `hip_qimg_kernels.h`).
>
> Plan (staged, mirror cuda/svdquant gates):
> 1. **INT4 W4A16** first — decode residual to f32 (host via `svdquant_cpu.h` or a
>    HIPRTC kernel), residual GEMM + 2 low-rank GEMMs on the GPU, add bias; gate
>    rel_L2(impl, `int4_w4a16_y_svdq`) ≤ 2e-4. This is the RDNA4 happy path.
> 2. **INT4 W4A4** — quantize `x/lam` per-token group-64 (host `sq_quant_act_
>    int4_g64`, or a kernel), then the same; gate ≤ 2e-4.
> 3. **NVFP4 W4A16 / W4A4** — RDNA4 (gfx12) has **FP8 and INT8 WMMA builtins**
>    (`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`,
>    `..._i32_16x16x16_iu8_w32_gfx12`) but **NO native FP4 block-scale MMA** like
>    sm_120a. So decode the NVFP4 residual to f32 (host `sq_unpack_nvfp4_residual`)
>    and run the same f32 GEMM path; gate ≤ 2e-4. (A native FP4 path is NOT
>    expected on RDNA4 — document that, don't force it.)
>
> Layout: `rdna4/svdquant/{test_hip_svdquant.c, Makefile, README.md}`. Makefile:
> gcc, `-I.. -I../../common -I../../cpu/svdquant`, link `../rocew.c`, `-ldl -lm`
> (see `rdna4/qimg/Makefile`). Run: `./test_hip_svdquant ../../ref/svdquant/dumps`,
> exit 0 iff all gates pass; also run on `--real-nunchaku` dumps. Verify both, then
> stop and report (do NOT push; ask first per the repo rule).

---

## RDNA4 gotchas

- **No hipcc / no FP4 HW.** gfx12 (RX 9070) has FP8+INT8 WMMA but no NVFP4
  block-scale MMA → NVFP4 cases run as software-decode + f32 GEMM (a correctness
  test, not the perf path). INT4 is the natural RDNA4 SVDQuant format.
- `common/npy_io.h` reads only f4/i4/u1 — the dumps already honor this.
- INT4 nibble decode is signed `[-8,7]` (`u -= 16*(u>=8)`); group sizes differ
  (INT4 g64, NVFP4 g16); low-rank uses RAW x, residual uses x/λ.
- The `rdna4/qimg` INT4 path is **W4A16** with a *logical* (de-swizzled) layout
  identical to the dumps' `qint4`/`wscale`/`lora_*`/`smooth`/`bias` — so its
  kernels can be reused with minimal glue.
- Reference regen needs torch+safetensors+numpy (already present); `--cross-check`
  needs the `ref/svdquant/deepcompressor` clone; `--real-nunchaku` needs the
  installed Nunchaku packer at `/home/syoyo/src/nunchaku`.
