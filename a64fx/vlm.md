# a64fx/vlm — Status & Optimization Notes

A64FX SVE vision encoder for Qwen3-VL (drop-in replacement for
`vision_encode()`). This page tracks the current state, the fused
dual-conv2d patch-embedding work, the measured performance profile, and
the ranked optimization opportunities.

Reference model for all numbers below: **Qwen3VL-2B-Instruct-F16**
(dim=1024, heads=16, blocks=24, ffn=4096, deepstack=3 @ blocks 5/11/17,
proj=2048, patch=16), `~/fujisan.jpg` 384×256 → 384 patches → **96 merged
tokens**, `--dtype fp16`.

---

## 1. Fused dual-conv2d patch-embedding (latest work)

`kernels/conv2d_sve.{c,h}` + `tools/test_conv2d_sve.cpp`, wired into
`patch_embed_gemm_mt` in `src/vit_a64fx.c`.

- The Qwen3-VL patch embed is two 16×16×3 convs (stride == kernel size)
  summed. Stride == kernel ⇒ patches never overlap ⇒ the conv is a plain
  block GEMM: `C[p,d] = Σ_k (K0[d,k]+K1[d,k])·patch[p,k] + b[d]`.
- `conv2d_sve_full` **reuses the proven production microkernel**
  `micro_kernel_fp32_8x3_unroll4` (8 patches × 48 channels, 4× K-unroll).
  It gathers tile pixels *directly* into the 8×48 A layout
  (`A_packed[k*8+m]`) and runs the merged-K0+K1 BTP GEMM + bias in a single
  OMP region (`collapse(2)` over M-block × N-block). This skips the old
  path's intermediate `[n_patches, ks]` buffer, the separate `pack_A`
  pass, and the separate `add_bias` pass.
- **Bit-identical** to the old gather+`gemm_fp32_BTP`+bias path
  (verified `maxerr=0`), and stays bit-identical across 12T/48T
  (norm 455.6237).

### ⚠️ A64FX SVE microkernel erratum (why the hand-written kernel was abandoned)

A hand-written 16×8, then 8×8, conv2d microkernel (8-wide `ld1rw` A
broadcasts + 8-FMA inner loop) **mis-executes on this A64FX** even though
the disassembly is verified byte-correct:

- **Data-dependent accumulation loss** — e.g. a K=4 all-ones case returns
  only the last k-step's contribution instead of the full sum.
- **Phantom OOB row-store** — the epilogue writes one extra 32-byte row
  past the C buffer (an 8-store kernel writes a phantom 9th row; a
  16-store kernel writes a phantom 17th row), corrupting the heap.

The store side traces to a **16-distinct-rolling-row-base limit**: an asm
function that stores to 16 distinct row-bases writes a phantom row; 8
row-bases is safe (exactly the production 8×48 kernel's shape). The
FMA-side accumulation loss is a second, still-unrooted symptom of the
same class. **Do not hand-roll a wide (`ld1rw`×8) SVE GEMM inner loop on
this part** — reuse the 8×48 kernel (24 accumulators, 4 `ld1rw` × 3
`ld1w` FMA grouping, 8 row-base stores).

---

## 2. Current performance profile

`VLM_STAGE_TIMING=1`, 48 threads, no CMG (CMG is *slower* here — see
§3.1). Reflects the **nb-outer GEMM schedule** (new default, §3.1).

| Stage | CPU-s | Notes |
|---|---|---|
| **ffn_down** (`BT_d`) | ~0.5 s | 96×4096 → 96×1024, ×24 blocks — GEMM, dominant |
| **ffn_up** (`BT_u`) | ~0.45 s | 96×1024 → 96×4096, ×24 blocks — GEMM |
| **qkv** (`BT_qkv`) | ~0.35 s | 96×1024 → 96×3072, ×24 blocks — GEMM |
| **attn** | 0.83 s CPU / **~7% wall** | QK^T + AV, fp32, well-parallelized (see §3.2) |
| **deepstack** (fc1/fc2) | ~0.15 s | 3 layers, 96×2048 GEMMs |
| attn_out (`BT_o`) | ~0.10 s | 96×1024 → 96×1024, ×24 — GEMM |
| **patch_embed** (fused) | ~1 ms wall | **not a bottleneck** |
| layernorm / gelu / mrope / pos / mm_proj | ~0.1 s | elementwise / small |

**Total ≈ 0.16–0.32 s (median ~0.32 s) at 48T no-CMG.**

> ⚠️ **This shared Fugaku node has ~1.5–2× run-to-run variance** (HBM/NUMA
> state, background load): the total swings 0.16 s (fast) to 0.46 s (slow)
> for identical commands. **Always A/B with `--bench ≥ 8` back-to-back** and
> trust the *relative* delta (which is stable), not a single absolute number.
> The per-stage `%` from one `VLM_STAGE_TIMING` run is unreliable for the same
> reason — an early run made `attn` look like 21% when it is really ~7%.

Headline: **the transformer GEMMs (ffn_down/up, qkv, attn_out, deepstack)
are the dominant cost** and are already at their W-HBM-bound limit after the
nb-outer schedule (§3.1). Attention is a distant ~7% and fp32-FMA-bound
(§3.2). patch_embed (the fused-conv2d target) is 0.5% and was never the
bottleneck.

### The GEMM is W-traffic bound, and that is now fixed (mostly)

A standalone micro-benchmark of `gemm_fp16_BTP` (ffn_up shape, 96×1024→
4096) shows the *kernel* is efficient: with W LLC-warm it hits ~2.7
TFLOP/s (≈ compute-bound), but with 24 different W matrices (the real
VLM, one per block) it drops to ~1 TFLOP/s, and in-situ to ~0.3 TFLOP/s.
The gap is **W re-streaming**: the old **mb-outer** tile loop made each of
the 12 M-blocks re-read the whole 8 MB layer W from HBM (12× = 96 MB per
GEMM). The **nb-outer** schedule (below) cuts that to reading W once per
GEMM, which recovers most of the gap.

---

## 3. Optimization opportunities (ranked)

### 3.1 ✅ GEMM W re-streaming — FIXED via the nb-outer schedule

**Done.** `gemm_fp16_BTP` / `gemm_bf16_BTP` now use an **nb-outer** tile
loop by default: each core owns a slice of N and streams the small A
(96×1024×4 B = 384 KB, L2-resident) across all 12 M-blocks, so the
W-slice is read **once per GEMM** instead of once per M-block. This is a
pure schedule change (swap the `collapse(2)` loop order, bounds swapped so
the pragma still directly precedes the loop) — no kernel change.

- **Measured: ~28% faster on the whole VLM** (back-to-back `--bench 8`,
  48T fp16, 384×256): nb-outer median ~0.32 s (~300 tok/s) vs mb-outer
  ~0.44 s (~217 tok/s). The delta is stable even as absolute numbers swing
  with node state (see the §2 variance note).
- Toggle: `VLM_GEMM_NB=0` reverts to the old mb-outer for A/B. Bit-
  identical output either way (norm 455.6237).
- The `collapse(2)` must directly precede the `for` — an `if/else` around
  two worksharing loops is an **invalid** OpenMP construct (the Fujitsu
  libfjomp aborts in `__kmpc_for_static_init_4`). Swap the *bounds*
  instead.

**Tested and rejected:** per-core W replicas (CMG, `VLM_NUMA=4` +
`numactl -C 12-59 -m 4-7`) is *slower* here (0.35 s vs 0.24 s) — each
node's ~768 MB W-replica is far larger than the LLC, so it just moves HBM
reads local and adds replication/mbind overhead. **Remaining GEMM ideas**
(lower payoff now): larger M-tile (16-row compute + two 8-row store passes
to dodge the §1 erratum), and fp16 activations (§3.2).

### 3.2 Attention (~7% — fp32-FMA-bound, low priority)

Profiled with `VLM_ATTN_PROFILE=1` (per-phase CPU-seconds, stable across
runs): **QK^T 0.40 s + AV 0.38 s = 94%**, softmax 0.05 s, extract 0.003 s;
sum ~0.83 s CPU. It is **well-parallelized** (the CPU-sum stays ~0.75–0.83 s
at 12–48 threads while wall-clock scales cleanly → only ~7% of the encode at
48T). The QK^T kernel (`qk_vert_8q_48k`) already runs at **full fp32 FMA
peak** (24 `svmla` per d-iter = 16 lane-FMA/cycle), so there is no fp32
headroom — the only lever is **fp16 QK^T/AV** (2× FMA rate → ~3% overall,
but hard to measure given the §2 variance, and AV mixes an fp32 score with an
fp16 V so it needs a fp32-accumulate trick). Low priority: ~7% ceiling.

> Note: an earlier single-run `VLM_STAGE_TIMING` showed attn at ~21%; that was
> node-state variance (see §2). The stable sub-profile puts it at ~7%.

### 3.3 Store the activations in fp16 (A is currently fp32) — likely small

`hidden` / `Y` / `ffn_buf` are `float` (fp32). In the nb-outer schedule A is
the *streamed* operand, but it is only 384 KB and stays **L2-resident**, so
halving it to fp16 cuts L2 (not HBM) traffic — modest. The 2× fp16×fp16 FMA
rate does **not** help because the in-situ GEMM is **W-HBM-bound, not
compute-bound** (W is read from HBM once per GEMM; the FMA pipe idles waiting
on W). So fp16-A is expected to be a small win; measure before investing.

### 3.4 Elementwise / glue (~6%, low priority)

layernorm, gelu, mrope, pos_emb, mm_proj are each a few % or less. Only worth
touching if 3.1–3.3 are exhausted. The layernorm SVE kernel already exists
(`norm_sve.c`); gelu is fused into the ffn stages.

### 3.5 What is *not* worth doing

- **Further patch_embed work.** It is 0.4% and bit-identical to the
  prior GEMM path. Done.
- **A wider hand-rolled SVE GEMM kernel.** The §1 erratum makes wide
  `ld1rw`×8 inner loops unreliable; the 8×48 kernel is the safe ceiling.
- **Chasing single-run stage-timing deltas.** The §2 node variance (~1.5–2×)
  swamps any <5% change; A/B with `--bench ≥ 8` back-to-back or trust the
  stable CPU-second sub-profiles (`VLM_ATTN_PROFILE=1`) instead.

---

## 4. Build / run / validate

```sh
cd a64fx/vlm
make CC=fcc OPENMP=1            # -> build/vlm_runner, build/tensor_diff

M=~/models/Qwen3VL-2B-Instruct-F16.gguf
MM=~/models/mmproj-Qwen3VL-2B-Instruct-F16.gguf

# current best on this node (no CMG — see 3.1):
OMP_NUM_THREADS=48 ./build/vlm_runner $M $MM ~/fujisan.jpg \
    --dtype fp16 --threads 48 --bench 3

# stage breakdown:
VLM_STAGE_TIMING=1 OMP_NUM_THREADS=48 ./build/vlm_runner $M $MM ~/fujisan.jpg \
    --dtype fp16 --threads 48 --bench 1 2>&1 | grep -A16 "stage timings"

# fused-conv2d unit test (correctness vs scalar + micro-bench):
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -Ikernels -Iinclude \
    -c kernels/conv2d_sve.c -o build/conv2d_sve.o
as  -march=armv8.2-a+sve -o build/micro_kernel_fp32_8x3.o kernels/micro_kernel_fp32_8x3.S
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -Ikernels -Iinclude \
    -c kernels/pack_matrices.c -o build/pack_matrices.o
FCC -O3 -march=armv8.2-a+sve -std=c++17 -fopenmp -Ikernels -Iinclude \
    -c tools/test_conv2d_sve.cpp -o build/test_conv2d_sve.o
FCC -O3 -fopenmp -o build/test_conv2d_sve build/test_conv2d_sve.o \
    build/conv2d_sve.o build/pack_matrices.o build/micro_kernel_fp32_8x3.o -lm
./build/test_conv2d_sve 384 384 1     # expect PASS, max abs err ~1e-5
```

### Known issues (not from the fused-conv2d work)

- **CPU reference does not build** — `common/transformer.h:1847: 'xi8'
  undeclared` (and `:1849 'inv'`), so `make REF=1` + `tensor_diff`
  (the gold-standard dump comparison) is unavailable until that is fixed.
- **Stale norm reference** — the readme documents `norm=455.7341`; the
  current build gives `455.6237` (bit-identical across 12T/48T). The gap
  is a stale reference (older model/build), **not** a regression — the
  fused patch_embed is provably bit-identical to the prior path.
- **`getenv` build break (fixed)** — `common/ggml_dequant.h:1553` calls
  `getenv` without `stdlib.h`; `src/vit_a64fx.c` now includes `<stdlib.h>`
  before pulling in `ggml_dequant.h`.
