# SAM 3D Body — RDNA4/HIP port journal

Mechanical port of `cuda/sam3d_body/` to AMD RDNA4 (RX 9070 XT, gfx1201)
via HIPRTC + `rocew`. The CUDA port history lives in
`doc/sam3d-body.md`.

## Port adaptations

- Drop `cuda_hip_compat.h`; include `../rocew.h`,
  `../hip_runner_common.h`, `../hip_kernels_common.h` instead.
- Mechanical `cu*` → `hip*` rename across runner, kernels and tests.
- `hipDeviceSynchronize` everywhere — `hipCtxSynchronize` returns
  `hipErrorNotSupported` (801) on ROCm 7.x.
- `MODELS ?= /mnt/disk1/models` (CUDA Makefile points at
  `/mnt/disk01/models` which doesn't exist on this host).
- Backbone precision: bf16 / fp16 only (mirrors CUDA — `fp32` is
  rejected by `sam3d_body_create`).

## BF16-WMMA encoder GEMM (opt-in, default OFF)

The encoder's 9 per-block GEMMs (QKV / proj / FC1·FC2 / SwiGLU w1·w2·w3) use
`gemm_tiled_f16_f32` (F16 weights, scalar shared-mem tiling). When
`SAM3D_GEMM_WMMA=1` (and gfx1200/1201), they route to a BF16-WMMA kernel
`gemm_f16w_bf16a_wmma_t` instead. Default OFF.

- Kernel: appended to `hip_sam3d_body_kernels.h` — a **byte-for-byte structural
  copy** of the kernel validated in `rdna4/sam3d` (and originally `rdna4/hy3d`),
  with the *identical* param layout to `gemm_tiled_f16_f32`
  (`{Y, W(f16), X, bias, n_out, n_in, n_tok}`). F16 weights via `half_to_float`,
  X+W RNE-truncated to bf16, F32 accumulate, `n_in % 16 == 0` required (all
  encoder dims qualify: 1280/3840/5120), gfx12-guarded with an empty `#else`.
- Dispatch: `sb_gemm_tiled()` in the runner reads `n_out/n_in/n_tok` from the
  shared packed param buffer and picks WMMA vs scalar + geometry. Handle
  `fn_gemm_wmma` is a *soft* BIND (NULL → scalar fallback). Debug:
  `SAM3D_GEMM_WMMA_DEBUG=1` prints the first dispatch decision.
- **Decoder GEMMs left scalar on purpose**: the 2 `gemm_f32_bias` sites are a
  145-token GEMM and a 1-row GEMV — too small for WMMA to help.

### Status — validated on RX 9070 XT / gfx1201 (2026-05-31)

- Host build clean (`make test_hip_sam3d_body`, BUILD_RC=0; 9/9 encoder launches
  routed through `sb_gemm_tiled`).
- HIPRTC compiles the kernel for gfx1201 and the symbol resolves.
- WMMA path **engages**: `SAM3D_GEMM_WMMA_DEBUG=1` →
  `[sb_gemm_tiled] use_wmma=1 fn=<non-null> n_in=1280 -> WMMA`.
- **e2e mesh validated** (dinov3 backbone, fujisan.jpg, `--bbox 0 0 512 512`),
  WMMA vs default scalar:
  - topology identical: V=18439 both.
  - vertex cosine **0.99999970**; per-vertex L2 max 0.0026, mean 0.00078,
    median 0.00042 on a 1.51-unit model — **zero** verts with L2 > 0.01.
    Tight, no outliers; expected bf16, matches the precision profile of the
    GPU-validated sam3d kernel.
- Decoder GEMMs intentionally left scalar (145-tok GEMM + 1-row GEMV).
- **Speedup** (dinov3, fujisan.jpg, gfx1201, 2 runs each):
  encoder blocks **666.5 → 219.1 ms = 3.04×**; encoder total 687.8 → 240.9 ms
  = 2.86×. (`verify_dinov3` not runnable here — needs the CUDA-generated
  `/tmp/sam3d_body_ref/*.npy` dumps, absent on this host.)

## MHR final-call threading fix (decoder)

MHR skinning (`sam3d_body_mhr_forward`, CPU/OpenMP) runs once per decoder layer
+ once post-loop. The dominant cost is a 166M-FMA dense matvec (55317×3000) in
`pose_correctives`, gated `n_threads > 1`. The per-layer calls passed
`SAM3DB_MHR_THREADS()` but the **post-loop call passed `n_threads=0`** (serial).
Fixed to pass `SAM3DB_MHR_THREADS()` too (runner ~L2368).
- Isolated effect (`decoder final: mhr` timer): **128.8 → ~26 ms** (~4.5× on
  that call, ~100 ms saved). Output **bit-identical** (max vertex diff 0.0 vs
  pre-fix — threading only parallelizes independent rows, inner sum unchanged).
- Caveat: full-pipeline wall-clock is noisy run-to-run (CPU/GPU clock+contention
  swings norm_heads/layers by 5×); only the isolated `final: mhr` field is a
  clean before/after. Remaining MHR cost is the 6 in-loop calls (~20 ms each,
  already threaded) — further wins need the GPU `mhr_blend`/`mhr_lbs` kernels
  (exist + verified) wired into the loop (deferred, "Step 7").

## BF16-WMMA flash-attention, dinov3 encoder (opt-in, default OFF)

The dinov3 encoder attention used the scalar `flash_attn_tiled_f32`
(`../hip_kernels_common.h`, one thread/query row, F32). With `SAM3D_FA_WMMA=1`
(gfx12, head_dim==64) it routes to a new BF16-WMMA kernel
`flash_attn_tiled_wmma_f32`. ViT-H is unaffected — it uses `sdpa_f32` (hd=80).

- Kernel: appended to `hip_sam3d_body_kernels.h`. **Identical signature/IO** to
  the scalar kernel (`out, qkv, K_t, V_t, n_tok, dim, n_heads, head_dim, scale`):
  Q from interleaved `qkv[qi*3*dim + h*64]`, K_t/V_t per-head `[H,n_tok,64]`,
  out `[n_tok, H*64]` — so dispatch is a clean A/B swap. FA2 online softmax in
  **F32** (m_i/l_i/S all f32); only QK^T and P·V matmul *inputs* are bf16
  (RNE-truncated). Block 128 = 4 waves×32, each wave 16 q rows (Br=64), Bc=16.
  Adapted from qimg `flash_attn_sa_wmma_f32` (hd=128→64: k-loop 8→4, O-tiles
  8→4, LDS halved, interleaved-qkv/per-head-K_t load).
- Dispatch: soft handle `fn_fa_wmma` + `sb_use_fa_wmma()` (env `SAM3D_FA_WMMA`)
  at the FA launch (~runner L1880). Block 64→128, shmem→0 (static LDS).
- Why it's safe vs the fa2 cos≈0 hazard: that bug is the 16-wave bf16-*softmax*
  path; here softmax is F32, matching the fa2-validated 1-wave-safe pattern and
  the in-production qimg/trellis2 kernels.

### Validation (gfx1201)

- **Standalone numeric harness** `test_fa_wmma.c` (build via `make test_fa_wmma`
  after adding the rule, or the gcc line in the file) compiles both kernels into
  one HIPRTC module and compares WMMA vs the scalar kernel on synthetic Q/K/V,
  including the real dinov3 config and partial-tile edge cases:
  ```
  n_tok=1029 heads=20 : cosine 0.99999723  max_abs 3.1e-4  PASS
  n_tok=768  heads=16 : cosine 0.99999733  max_abs 5.9e-4  PASS
  n_tok=130  heads=2  : cosine 0.99999736  max_abs 4.9e-4  PASS  (partial q+kv)
  n_tok=64   heads=1  : cosine 0.99999757  max_abs 6.4e-4  PASS  (single tile)
  ```
- **Perf** (encoder *blocks*, GEMM-WMMA on for both, 2 runs each):
  FA-scalar 216.6/216.9 ms → FA-WMMA 189.7/186.1 ms ≈ **−14% blocks (~30 ms)**.
  Smaller than the GEMM win — attention is a minor FLOP share of the encoder.
- **e2e mesh** (FA-WMMA vs FA-scalar, both GEMM-WMMA, dinov3 fujisan.jpg):
  V=18439 both, vertex **cosine 0.99999966**, max vertex diff 1.8e-3 — bf16
  level, consistent with the standalone harness. Default OFF; flip on once you
  want the ~30 ms.

## GPU MHR pose_correctives matvec (opt-in, default OFF) — validated gfx1201

MHR skinning (CPU/OpenMP) is the dominant remaining decoder cost. Its hot path
is a single 166M-FMA dense matvec `out[55317] = LW[55317,3000] @ h[3000]`
(bias-free) in `pose_correctives`, run 7× (per decoder layer + final).

- **Mechanism:** `common/sam3d_body_mhr.h` gained an optional plain-C callback
  hook `pc_matvec_fn` on the assets struct; `pose_correctives` calls it instead
  of the CPU matvec when set (no HIP types leak into the shared CPU header). The
  runner installs `sb_pc_matvec_gpu` (runs `gemm_f32_bias` N=1, D_in=3000,
  D_out=55317, b=NULL on resident LW) via `sb_mhr_gpu_setup` when
  `SAM3D_MHR_GPU=1`. Constant is `S3DM_N_PC_H`(=3000); matvec is bias-free
  (`pc_linear_weight` only — there is no `pc_linear_bias`).
- **VRAM guard:** LW is 633 MiB resident; only engages if `hipMemGetInfo`
  reports ≥ LW + ~1 GiB free, else stays on CPU (an unguarded reservation OOM'd
  the downstream pipeline → `rc=4` when the box was VRAM-starved). Debug:
  `SAM3D_MHR_GPU_DEBUG=1`.
- **Isolated bench (`bench_pc_matvec`):** GPU matvec vs CPU reference cosine 1.0,
  max_abs 1.3e-6; **5.92×** (CPU 24.1 ms → GPU 4.08 ms/call, LW resident).
- **e2e correctness:** GPU-MHR vs CPU-MHR mesh **cosine 1.00000000, max vertex
  diff 0.0 — bit-identical**. (Pipeline is deterministic: CPU-vs-CPU re-run also
  0.0.)
- **e2e perf** (dinov3 fujisan.jpg, all-WMMA, single cold run = real single-shot
  inference): decoder **`mhr` 525.5 → 115.7 ms**, decoder **total 751.9 →
  237.1 ms**. Most of the CPU cost is a **419 ms layer-0 cold-start** (OpenMP
  thread spin-up + first-touch of the cold 633 MiB LW); GPU layer-0 is 21.8 ms
  because LW is pre-resident — that vanished cold-start is dispositive proof the
  hook engaged. Warm steady-state per-call is closer (CPU ~20–24 ms vs GPU
  ~18–21 ms) since the non-matvec MHR stages (blend/face/skin) stay on CPU; the
  big win is eliminating the cold-start that single-shot inference always pays.

## Decoder norm_heads: drop dead full-token LayerNorm

`hip_sam3d_body_debug_run_norm_and_heads` LN'd all N_Q=145 tokens into a
`tokens_n` buffer, but that buffer is **never read** — only `pose_raw`/`cam_raw`
(computed from row 0, the pose token) feed downstream. The function already
supports `tokens_norm=NULL` (LN's 1 row via `row0_scratch`). Both call sites
(per-layer ×6 + final) now pass NULL.
- **Output bit-identical** (max vertex diff 0.0 vs pre-change) — pure dead-work
  removal: 145-row LN → 1-row LN, the kept row 0 is computed identically.
- Effect on the `norm_heads` timer is hard to quote cleanly — it's CPU/OpenMP
  and dominated by cold-cache/parallel-region overhead, not the LN arithmetic
  (it swings ~13–95 ms run-to-run independent of this change). The change is
  strictly less work (eliminates 144/145 rows × 6 calls of LN) and never slower.

**Follow-up — serialize the now-trivial norm_heads.** With `tokens_norm==NULL`
the function does a 1-row LN + two single-row head matvecs (~2.6 MFLOP) yet still
spawned a 32-thread `omp parallel for` (the LN) and called `cpu_gemm_f32` with
`n_threads=32` (the heads) — pure fork/join overhead, which *is* the 2–31 ms
noise. Gate to `n_threads=1` when `n_rows==1` (`cpu_gemm_f32` splits output rows,
so thread count never changes results). **Result: decoder `norm_heads`
125.7 → 4.9 ms** (steady; noise gone), mesh **bit-identical** (max vertex diff
0.0). Decoder total ~388 → ~240 ms.

## End-to-end

```
./test_hip_sam3d_body \
    --safetensors-dir /mnt/disk1/models/sam3d-body/safetensors \
    --mhr-assets      /mnt/disk1/models/sam3d-body/safetensors \
    --image /tmp/sam-3d-objects/notebook/images/human_object/image.png \
    --bbox 200 100 1400 800 \
    -o /tmp/hip_sam3d_body.obj
```

Writes `V=18439 F=36874` (~1.2 MiB OBJ) on the human_object sample.

## Verifier status (refdir = `/tmp/sam3d_body_ref`)

| Verifier              | max_abs    | gate       |
|-----------------------|------------|------------|
| `verify_dinov3`       | 5.43e-01   | 1.5e+00    |
| `verify_ray_cond`     | 1.72e-05   | 2.0e-03    |
| `verify_build_tokens` | 3.81e-06   | 5.0e-04    |
| `verify_decoder_layer`| 6.68e-06   | 2.0e-03    |
| `verify_kp_update`    | 5.96e-06   | 5.0e-04    |
| `verify_mhr_head`     | 6.68e-06   | 1.0e-04    |
| `verify_decoder`      | 8.11e-06   | 5.0e-03    |
| `verify_mhr`          | 3.05e-05   | 5.0e-04    |

`verify_decoder` and `verify_mhr` need `--mhr-assets[-dir]
/mnt/disk1/models/sam3d-body/safetensors` in addition to
`--safetensors-dir`.

### Not green

- `verify_vith` — needs `vith_input.npy`, not in the current
  `/tmp/sam3d_body_ref` dump. `verify_dinov3` already covers the
  backbone path.
- `verify_end_to_end` — scaffold only, mirrors the CUDA status (the
  upstream pipeline isn't wired into the verifier yet).

## Reproducing the dumps

The `/tmp/sam3d_body_ref/` dumps come from a CUDA host running
`ref/sam3d-body/`'s python helpers; same constraint as `sam3d/`
(`pytorch3d`/`xformers`/etc. are CUDA-only Linux wheels). Generate on a
CUDA box and `rsync` the dump to this AMD host.
