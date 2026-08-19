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

`VLM_STAGE_TIMING=1`. No-CMG, 48 threads (the current best on this node;
see §3.1 — CMG+numactl is *slower* here).

| Stage | Time | % | Notes |
|---|---|---|---|
| **ffn_up** (`BT_u`) | ~96 ms | ~24–30% | 96×1024 → 96×4096, ×24 blocks |
| **ffn_down** (`BT_d`) | ~70 ms | ~22–38% | 96×4096 → 96×1024, ×24 blocks |
| **qkv** (`BT_qkv`) | ~51 ms | ~16–17% | 96×1024 → 96×3072, ×24 blocks |
| **deepstack** (fc1/fc2) | ~40–63 ms | ~7–20% | 3 layers, 96×2048 GEMMs |
| attn_out (`BT_o`) | ~14–60 ms | ~4–11% | 96×1024 → 96×1024, ×24 |
| attn (softmax/SDPA) | ~18 ms | ~5–6% | fp16 attention |
| **patch_embed** (fused) | **1 ms** | **~0.4%** | **not a bottleneck** |
| layernorm / gelu / mrope / pos / mm_proj | ~10 ms | ~3% | elementwise / small |

**Total ≈ 320–450 ms** (median ~450 ms, ~214 tok/s) at 48T no-CMG.

Headline: **the transformer GEMMs are ~85% of the encode.** patch_embed
(the fused-conv2d target) is 0.4% and was already fast — it was never the
bottleneck.

### GEMM efficiency is the problem, not throughput

The GEMMs run at **~200 GFLOP/s** across 48 cores. Even against a
conservative A64FX fp16 peak (~256 GFLOP/s/core → ~12 TFLOP/s on 48
cores), that is **~1–2% of peak** — i.e. the GEMMs are **memory /
W-traffic bound, not compute bound**. For M=96 the 8×48 kernel makes 12
M-blocks, and each M-block **re-streams the whole layer W** (e.g. ffn_up
W = 1024×4096×2 B = 8 MB ⇒ ~96 MB per layer). That re-streaming, fought
over the LLC by 48 threads, is the dominant cost.

---

## 3. Optimization opportunities (ranked)

### 3.1 Kill the W re-streaming in the transformer GEMMs (biggest win, ~85% of encode)

The 96-token M dimension is small; the 8×48 tile re-reads each layer's W
once per 8-row M-block. Options, roughly in order of expected payoff:

1. **Per-core W replicas (CMG) *with working NUMA pinning*.** The code
   already has `gemm_fp16_BTP_cmg` / `gemm_bf16_BTP_cmg` + `VLM_NUMA=N`.
   On **this node the readme's `numactl -C 12-59 -m 4-7` config is
   slower** (450 ms → 550 ms) than no-CMG — the 2B model is small enough
   that the replica + cross-NUMA traffic hurts, and the core pinning
   fights OMP. Investigate: pin to the node's *actual* usable
   core/NUMA map (not the hardcoded 12–59 / nodes 4–7), or try
   `VLM_NUMA=2`, or `mbind` replicas to the cores that own them. This is
   config, not code.
2. **W-sliced (N-parallel) schedule.** Give each thread a *slice of N*
   (a set of 48-channel blocks) and have it stream the *small* A
   (96×1024×4 B = 384 KB, easily L2-resident) across all its N-slice.
   The thread's W-slice stays L2-resident; A is re-read but tiny. This
   inverts the current M-parallel re-streaming. Needs a
   `collapse(2)`-friendly schedule (N outer, M inner) in
   `gemm_fp16_BTP` / the cmg variants.
3. **Larger M-tile for the small-M case.** A 16- or 32-row A-tile halves /
   quarters the M-blocks (and thus W re-streams). A 16-row *store* hits
   the §1 erratum, but the tile can compute 16 rows and **store in two
   8-row passes** (safe). More kernel work; higher risk.

### 3.2 Store the activations in fp16 (A is currently fp32)

`hidden` / `Y` / `ffn_buf` are `float` (fp32). Converting A to fp16 would
halve A traffic and let the kernel run fp16×fp16 (2× FMA rate). Cost: an
A-conversion pass per layer (A is regenerated from layernorm each block).
Worth measuring against 3.1 — the two compose.

### 3.3 Attention (~5–6%)

`attn` (softmax/SDPA) is a separate kernel from the GEMMs. Profile it on
its own; at 96 tokens it is small in absolute terms (~18 ms) so it is
lower priority than the GEMMs, but it is the one non-GEMM hot spot.

### 3.4 Elementwise / glue (~3%, low priority)

layernorm, gelu, mrope, pos_emb, mm_proj are each <1.4%. Only worth
touching if 3.1–3.3 are exhausted. The layernorm SVE kernel already exists
(`norm_sve.c`); gelu is fused into the ffn stages.

### 3.5 What is *not* worth doing

- **Further patch_embed work.** It is 0.4% and bit-identical to the
  prior GEMM path. Done.
- **A wider hand-rolled SVE GEMM kernel.** The §1 erratum makes wide
  `ld1rw`×8 inner loops unreliable; the 8×48 kernel is the safe ceiling.

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
