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
| **attn** | 0.83 s CPU / **~7% wall** | QK^T + AV, fp32. ⚠️ **only** parallel with the **OpenMP** backend — the C11-thrd default serializes it (~65% wall). See the build gotcha in §3.3. |
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
nb-outer schedule (§3.1). Attention is fp32-FMA-bound (§3.2) and ~7% *only
with the OpenMP build* (the C11-thrd default serializes it to ~65% wall —
see the build gotcha in §3.3, now fixed by defaulting `make CC=fcc` to
OpenMP). patch_embed (the fused-conv2d target) is 0.5% and was never the
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

### 3.2 Attention (fp32-FMA; load-bound, ~27% of the int8 encode at 48T)

Profiled with `VLM_ATTN_PROFILE=1` (per-phase CPU-seconds, stable across
runs): **QK^T + AV = ~87%**, softmax ~6%, extract ~2%. It is
**well-parallelized** across 16 heads × 12 q-tiles (192 units / 48 threads)
*only with the OpenMP backend* — the C11-thrd default serializes it (see the
build gotcha in §3.3). Post-int8 (GEMMs fast), attention is **~27% of the
encode at 48T**.

**Benchmarked (standalone, per core, `tools/bench_attn.c`):** the fp32 FMA
floor (8 independent chains, no loads) is **49.8 GFLOP/s**; the full attention
(QK^T+AV, K/V 96 KB/head in L2) is **29.3 GFLOP/s = 58% of the FMA floor** —
so it is **load-bound, not FMA-bound**: the 8 scalar Q loads + 3 K-vector loads
per QK^T d-iter (and 4 att + 4 V per AV row) bound it, not the 24 `svmla`.
In-situ it drops to **~19% of the FMA floor** (a further ~3.2× memory/
scheduling penalty from 48 threads' K/V contending for L2/LLC), so the kernel
headroom (58%→100%) does not fully translate end-to-end.

> **int8 attention tested and rejected (precision).** A precision probe
> (`tools/attn_int8_probe2.c`, per-row int8 Q/K/attn, V per-head, run on real
> per-block qkv dumps) measured the int8 attention-output rel-L2 vs the fp32
> path. **Full int8** (int8 QK^T + int8 AV) is too lossy: the per-block error
> grows with depth as the softmax sharpens — block 0 **1.1% avg / 2.4% worst**,
> block 12 **3.1% / 7.4%**, block 23 **3.0% / 10.2%** (the int8 AV quantizes
> the peaked attention weights and drops the tail). That would add a several-%
> norm delta on top of the int8 GEMM's 0.79% — not acceptable. **QK^T-only
> int8** (int8 scores, fp32 softmax + fp32 AV) is safe (~**1.0%** worst across
> all blocks) but only speeds the QK^T (~12% of the encode) → **~1.07×
> end-to-end** — marginal for one new SDOT kernel + Q/K quantize. So int8
> attention is not worth it; the attention stays fp32.

> **fp16 attention investigated and ruled out (A64FX hardware limit).** A
> fast fp16 dot product needs `FMLA Z.S, P/M, Z.H, Z.H` (fp16×fp16 → **fp32**
> accumulate). That instruction is **FEAT_SVE_FP16**, which the A64FX does
> **not** implement — `/proc/cpuinfo` lists only `sve` (no `svefp16`/`sve2`),
> and `as -march=armv8.2-a+sve` rejects it. Measured FMA rates confirm the
> split: fp16-**accumulate** (`FMLA Z.H`) is 2× fp32 (32 vs 16 lanes, same
> cycles/iter), but fp16 accumulate over a 64-term QK^T is numerically
> marginal (~0.1–1% score error → borderline softmax). The GEMM's
> fp16-load+`FCVT`+fp32-FMA path gives **no FMA gain** (fp32 rate) and adds
> FCVT overhead — and the attention is FMA-bound, so fp16's halved memory
> traffic does not apply. Net: fp16 attention would be a ~1–2% marginal gain
> at real precision cost; not worth it. The GEMMs hit the same wall (A is
> fp32, so they use FCVT and are fp32-FMA-rate limited too).

> Note: an earlier single-run `VLM_STAGE_TIMING` showed attn at ~21%; that was
> node-state variance (see §2). The stable sub-profile puts it at ~7%.

### 3.3 ⭐ INT8 SDOT GEMM — the big remaining lever (~3–5× on GEMMs)

A64FX has **INT8 SDOT** (`sdot z.s, z.b, z.b`, 3-operand — no predicate; the
4-operand `p/m` form does NOT assemble on this binutils 2.30). Peak is
**512 GOPS/core** (2 FPU × 2 SDOT/cy × 64 int8-MAC @ 2 GHz) = **8× the fp32-FMA
MAC rate** (32 MAC/cy). **INT16 has no dot product** in SVE — and on this node
+ binutils 2.30 even `smla`/`smlal`/`smull` won't assemble (see the int16
section below) → **quantize to int8, not int16.** (An int16 path was still
built, via a hi/lo int8 split, for precision comparison: it is near-exact but
**slower than fp16**, so it is not a production win.)

Benchmarked (`int8-new/bench_int8_nb.c`: nb-outer + prepacked B + 48T, **24
different W**, same conditions as the fp16 `vlm/tools/bench_gemm.c`):

| GEMM (VLM shape)        | fp16 nb-outer | int8 nb-outer | speedup |
|-------------------------|---------------|---------------|---------|
| ffn_up   96×4096×1024   | 0.784 ms      | 0.283 ms      | **2.8×** |
| ffn_down 96×1024×4096   | 1.071 ms      | 0.196 ms      | **5.5×** |
| qkv      96×3072×1024   | 0.737 ms      | 0.245 ms      | **3.0×** |

The win = (a) int8 W is **half the bytes** of fp16 → half the HBM traffic (the
GEMMs are W-HBM-bound), and (b) the 8× compute rate. The existing `int8-new`
driver is **mb-outer + re-packs B per K-chunk** → only ~5 GOPS for M=96 (1% of
peak); the nb-outer + prepacked schedule is what reaches ~2800 GOPS/call (48T).

**Integrated** (`--dtype int8`). `kernels/int8_gemm.c` +
`kernels/kernel_6x4_int8.S` (the 6×4 kernel made to **accumulate** so K>256
correctly sums over K/256 chunks). Weights are quantized to int8 **per output
channel** (per-n) + pre-packed at cache build; activations are quantized
**per row** (per-m, no global-max barrier) per GEMM. Dequant: `C[m][n] =
C_i32[m][n] · a_scale[m] · w_scale[n]`.

Result on this node (384×256, 96 tokens, `--bench 16` median; node speed
varies session-to-session, so the *relative* columns are what matter):

| dtype    | output norm | Δ vs fp16 | vs fp16 (12T / 24T / 48T) | tok/s int8 (12T/24T/48T) |
|----------|-------------|-----------|---------------------------|--------------------------|
| fp16     | 455.6237    | ref       | 1.00×                     | 187 / 340 / 474          |
| **int8** | 452.0263    | **0.79%** | **3.0× / 3.1× / 2.7×**    | **567 / 1050 / 1274**    |
| int16    | 458.2751    | 0.58%     | ~0.7× (43% slower)        | —                        |

> **⚠️ Build gotcha (fixed this session, big win):** the encode is
> **dominated by attention (~27–65% of the wall)**, and attention runs on
> `vlm_parallel_for`. The Makefile historically defaulted to the **C11-thrd
> backend** (`OPENMP ?=` empty), whose worker threads **cluster on a few cores
> on A64FX and scale only ~1.4× over 48 cores** (vs ~25× for OpenMP) — so the
> attention was effectively **serial**, capping the whole encode at ~72 tok/s
> (48T) no matter how fast the GEMMs were. `make CC=fcc` now **defaults
> `OPENMP=1`** (fcc supports `-fopenmp`, and the OpenMP backend spreads the
> threads): the encode scales 52 → **1274 tok/s** (1T → 48T) and int8/fp16 is
> **3.0× / 3.1× / 2.7×**. `OPENMP=0` forces the old C11 path. **Always build
> with OpenMP** (`make CC=fcc`, now the default) or the attention silently
> serializes.

**This session took int8 from 1.53× (pre-session, scalar quant + separate
dequant) to ~2.7–3.1× faster than fp16** — via the fused GEMM+dequant kernel
(§3.3b), the SVE activation quantize (§3.3c), and fixing the serial attention
build above. The ratio is *higher* at fewer threads (less HBM contention →
the HBM-efficient int8 W wins more). The norm is unchanged throughout
(bit-identical int8 path: 452.0263 at every thread count; verified bit-
identical across 1T/48T via `--dump` + md5 of all 255 tensors).

**int8 is the production win** (1.53× faster, small 0.79% norm delta). The
whole-VLM win is well below the 3–5× standalone-GEMM win. `INT8_STEP_PROF=1`
(kernels/int8_gemm.c) breaks the int8 GEMM-BTP into per-row-A quant (~22%),
A-pack (~1%), SDOT GEMM (~71%), dequant (~6%). The SDOT GEMM runs ~5× slower
in-situ than standalone (≈500 vs ≈2400 GOPS) because **each block's W is
streamed from HBM once** (24 distinct W, nb-outer reads W a single time) —
the same regime that bounds the fp16 path, so the int8/fp16 ratio is
preserved. Ruled out (measured): OMP team creation ≈0.4 µs/region and temp-
buffer page faults ≈0.1 µs — both negligible; the per-step in-situ numbers are
HBM-contended and node-variance-noisy, so read the split, not the absolute ms.
Further whole-VLM gains need the SDOT GEMM closer to HBM-roofline (it is already
memory-bound on W) — fusing the dequant into the kernel (drop the int32 C
buffer) is **done** (§3.3b); bigger wins are int8 on the LLM side.

**Per-stage validation** (`--dump` + `tensor_diff`, enabled by the §Known-
issues build fixes — the fp16 A64FX output is the reference proxy). Confirms
int8/int16 are *correct* (pure quantization noise, no corruption): error is
0.0000 before the first GEMM (patch_embed/pos_emb/ln1), appears at `qkv`, and
accumulates smoothly through the 24-block residual — `block_out` rmse vs fp16:

    block  0      6      9     12     18     23
    int8   2.0e-2 3.8e-2 4.3e-2 4.6e-1 5.7e-1 1.7e+1
    int16  4.9e-3 9.4e-3 1.4e-2 2.4e-1 3.2e-1 7.5e+0

The steps at blocks ~11 and ~22 are the **deepstack injections** (layers
5/11/17) + final merge, which add large residual terms that amplify the
(correct) noise — not a bug (no isolated spike). int16 is ~4× smaller per
block, ~2× at the final embedding.

### 3.3a INT16 (hi/lo int8 split) — near-exact, but dominated by fp16

A64FX has **no 16-bit dot product**, and this node's **binutils 2.30 does not
even assemble** the SVE 16-bit integer multiplies (`smla` → "unknown
mnemonic", `smlal`/`smulla`/`smull` → fail; `fmla z.s, z.h, z.h` needs
FEAT_SVE_FP16, which A64FX lacks). So int16 is done by splitting each int16
into a high int8 + low int8 and expanding into **3 int8 SDOT GEMMs**
(`kernels/int8_gemm.c: gemm_int16_BTP`):

    x16 = xhi*256 + xlo_s + 128
    Σ a16·b16 = 2¹⁶·Σahi·bhi + 2⁸·Σahi·blo + 2⁸·Σalo·bhi
              + 2¹⁵·Σahi + 2⁷·Σalo + 2¹⁵·Σbhi + 2⁷·Σblo + 2¹⁴·K
    (the alo·blo term is dropped: a √K random walk vs the K·signal, ~1e-5 rel.)

i.e. 3 int8 GEMMs + 2 per-row + 2 per-col reductions + an **int64** combine
(SVE has no int64 accumulate, so the combine is scalar; the GEMMs stay SVE).

Standalone GEMM (96×4096×1024, 48T, vs fp32 ref): **int16 rel(L2)=2.6e-5**
vs **int8 rel(L2)=5.6e-3** → int16 is **~215× more accurate** and effectively
exact. But it costs **3× the int8 GEMM**, so in the VLM it lands **0.70×
fp16 (slower)** while only marginally closer to fp16 than int8 (0.58% vs
0.79% norm delta). **Conclusion: int16 is dominated by fp16** (slower and
less accurate) — keep it as a precision/validation reference, not a
production path. One bug found while integrating: the dispatch shared the
int8 `w_scale` slot, which is NULL under `--dtype int16` → segfault; fixed by
giving int16 its own scale param.

Two bugs found while integrating (both fixed):
- **A-pack race** — `pack_A_6x256` was called with `M-mb*6` rows (packing
  multiple 6-row tiles into one slot), overwriting neighbouring slots; masked
  at 1 thread, corrupted the 2nd M-tile under threads. Fixed to pack exactly
  one 6-row tile per (mb,kc).
- The 6×4 kernel **clobbers z8–z31**; `int8_gemm.c`'s SVE dequant keeps values
  live across the call, so the kernel now saves/restores the callee-saved SVE
  vectors (z8–z15, z28–z31).

### 3.3b Fused int8 GEMM + dequant (`kernel_int8_6x4_fused`) — DONE, ~2.5× on the GEMM

The last in-situ int8 cost was the non-fused pipeline's int32 **C buffer**
(alloc + memset + per-kc-chunk write + read) and the separate SVE **dequant
pass** (~6% of the GEMM-BTP per `INT8_STEP_PROF`). `kernel_int8_6x4_fused`
(`kernels/kernel_6x4_int8.S`) removes both: the kernel zeroes 24
accumulators, loops over **all** K/256 chunks in registers, then dequantizes
`Y = C·(a_scale[m]·w_scale[n])` and stores the 6×64 float tile in-kernel.
`gemm_int8_BTP_fused` (`kernels/int8_gemm.c`) is the driver (same quantize +
pack; falls back to `gemm_int8_BTP` when N%64≠0 or K%256≠0); the int8 VLM
path (`vit_gemm_bias_BT_int8_mt`) now calls it.

**Bit-identical** to the non-fused path: the unit test
(`tools/test_int8_gemm.c fused`) compares `gemm_int8_BTP_fused` vs
`gemm_int8_BTP` element-exactly (1T and 48T, incl. odd M=49), and the VLM
end-to-end norm is unchanged: **452.0263** (int8) / 455.6237 (fp16). The
epilogue reproduces the dequant's exact FP order (`t = a_scale*w_scale` per
column, then `C*t`) — a different order is only ULP-correct, not bit-exact.

Standalone GEMM (96×1024×4096, 48T; bit-identical to non-fused):

| threads | non-fused | fused    | speedup |
|---------|-----------|----------|---------|
| 48      | 0.325 ms  | 0.127 ms | **2.6×** |
| 12      | 0.640 ms  | 0.359 ms | **1.8×** |

End-to-end (384×256, 96 tokens, 48T): int8 546 → **660–810 tok/s**, i.e. the
int8/fp16 ratio rises **1.53× → 1.8–2.3×** (back-to-back A/B; the spread is
session/node variance). 12T: ~429 tok/s.

Bugs found while integrating (all fixed — each one masked at 1 call / 1 tile):
- **Truncated K loop** — the first draft copied only 2 of the original body's
  8 unrolled SDOT groups, so the fused kernel computed 1/4 of K — and ran ~4×
  *faster*. A speedup that is too good is a bug; diff the unroll count.
- **`x30` (link register) clobbered** — the epilogue used `w30` as the
  valid-row counter; `ret` then jumped to the row count. Epilogue scratch
  must be caller-saved (`x9`).
- **`x25` clobbered without save** — a callee-saved reg used for `mr` while
  the prologue only saves `x19–x24`; the *first* kernel call worked, later
  calls corrupted the caller's loop vars → wrong/garbage tiles ("only tile
  (0,0) correct") and an ASan heap-buffer-overflow in the test. This is why
  SVE kernel bugs can look shape-dependent (they depend on which register the
  compiler parked the caller's state in).
- **Tail-row store** — the epilogue must store only `mr` valid rows (Y is M
  rows, not MB×6); the kernel takes `mr` as an arg and guards each row's
  stores. The non-fused path hides this by padding C to MB×6.
- **Y alignment** — the in-kernel `st1w … [x, #i, mul vl]` stores need
  64-byte-aligned Y; the VLM's `xcalloc_f`/`xmalloc_f` now return
  `aligned_alloc(64, …)` (SVE-friendly for all paths).
- **a_scale tail** — the epilogue reads 6 a_scales per tile, so the driver
  pads `a_scale` to MB×6 floats.

### 3.3c SVE activation quantize — DONE, ~1.2× end-to-end

After fusing the dequant, `INT8_STEP_PROF` showed the per-row activation
quantize (step 1 of the int8 GEMM) was the **single biggest int8 cost**: it
read each row's fp32 input twice (max + quantize) with a **scalar**
`fabsf`/`lroundf` loop — **54% of a GEMM call** (0.365 ms) vs 46% for the SDOT
GEMM itself. `quant_row_sve` (`kernels/int8_gemm.c`) vectorizes it: SVE
`svabs`+`svmax` for the row max, `svmul` + **`svrinta`** (round ties *away*
from zero = exactly `lroundf`) + clamp [−127,127] for the quantize, `svtbl`
byte-extract for the int8 store. Because `svrinta` reproduces `lroundf`
bit-for-bit, the int8 output — and the VLM norm (**452.0263**) — are
**unchanged**.

Result: the quantize step drops **0.365 → 0.132 ms/call (2.8×)**, from 54% to
~21% of the GEMM call. End-to-end back-to-back A/B (scalar vs SVE quant, 48T,
384×256, `--bench 3`): **1.105× / 1.306× / 1.162×** (~**1.2×** on the whole
encode). Now the SDOT GEMM (W-HBM-bound) is the dominant int8 cost again.

Note: the old arm_sve.h (clang-7 / binutils 2.30) lacks `svcvtn_s8_f32_x`
(narrowing fp32→int8), so the byte store uses the `svcvt_s32_f32` + `svtbl`
extract trick (as in `tf_quantize_f32_to_int8`).

### 3.3d Pre-broadcast A kernel rewrite — TRIED, reverted (slower)

The fused kernel's per-group A path is 6× `ldr w` (scalar) + 6× `mov z.s,w`
(broadcast the 4 A-bytes to all 16 int32 lanes). Tight isolated probes (A/B
const in registers, single core) showed the A path is the main non-SDOT
overhead: **floor 512 / A-path 348 / B-path 494 GFLOP/s**. So I tried
**pre-broadcasting A**: pack each 4-byte A group already tiled to 16 lanes
(24576 B vs 1536 B per kc-chunk = **16×**), so the kernel does one `ld1b`/row
(no `ldr w`, no `mov`). In the isolated probe that A-load hit **512** (floor),
so the idea looked like a ~1.5× on the GEMM compute.

**It does not transfer to the real kernel — it's a loss.** End-to-end A/B
(384×256, int8, 48T, `--bench 4`): PB **0.90–0.94×** (slower); single-thread
GEMM (OMP_NUM_THREADS=1): PB **1.31× slower** (2.95→3.87 ms on ffn_up). The
isolated probe was misleading: its A buffer was tiny (L1-resident), so only
the `mov` cost showed. In the real kernel the **16× A data** is streamed and
the A-pack write is 16× bigger → the A path goes memory-bound and the net is
negative. (Reverted; `VLM_INT8_PB` removed.)

Two measurements worth keeping from this:

- **The real single-core GEMM is ~271 GFLOP/s (ffn_up) = 53% of the 512
  SDOT floor** — not the "~100/core" in-situ number, which is the **48-core
  aggregate** (W-HBM-bound). ⚠️ `taskset -c N` does **not** pin OpenMP
  threads; per-core GEMM benchmarks need `OMP_NUM_THREADS=1` or the "per-core"
  figure is actually the multi-core aggregate (looks absurdly above peak).
- **A B-pack rewrite won't help**: B (W) is already well-cached — one HBM
  read per nb-tile (~4 MB minimum for ffn_up), the ~16 re-reads per nb hit
  L2/LLC. The kernel is compute-bound on the A path, not B-memory-bound, and
  the A `mov` broadcast can't be removed without the 16× A-data penalty
  (above). So the fused int8 GEMM kernel is **near its practical limit**;
  the remaining ~47% to the SDOT floor is the A-load/mov + B-load + loop
  issue overhead, which the register file (32 Z: 24 accum + 6 A + 2 B) won't
  let us amortize away.

### 3.4 Store the activations in fp16 (A is currently fp32) — likely small

`hidden` / `Y` / `ffn_buf` are `float` (fp32). In the nb-outer schedule A is
the *streamed* operand, but it is only 384 KB and stays **L2-resident**, so
halving it to fp16 cuts L2 (not HBM) traffic — modest. The 2× fp16×fp16 FMA
rate does **not** help because the in-situ GEMM is **W-HBM-bound, not
compute-bound** (W is read from HBM once per GEMM; the FMA pipe idles waiting
on W). So fp16-A is expected to be a small win; measure before investing.

### 3.5 Elementwise / glue (~6%, low priority)

layernorm, gelu, mrope, pos_emb, mm_proj are each a few % or less. Only worth
touching if 3.1–3.3 are exhausted. The layernorm SVE kernel already exists
(`norm_sve.c`); gelu is fused into the ffn stages.

### 3.6 What is *not* worth doing

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
make CC=fcc                     # OpenMP is the default for fcc (required — the
                                 # C11-thrd path serializes attention; OPENMP=0
                                 # forces it). -> build/vlm_runner, build/tensor_diff

M=~/models/Qwen3VL-2B-Instruct-F16.gguf
MM=~/models/mmproj-Qwen3VL-2B-Instruct-F16.gguf

# current best on this node (no CMG — see 3.1):
OMP_NUM_THREADS=48 ./build/vlm_runner $M $MM ~/fujisan.jpg \
    --dtype fp16 --threads 48 --bench 3

# int8 (W8A8, ~1.5× faster, norm 452.03 vs fp16 455.62 — see 3.3):
OMP_NUM_THREADS=48 ./build/vlm_runner $M $MM ~/fujisan.jpg \
    --dtype int8 --threads 48 --bench 3

# int16 (hi/lo int8 split; near-exact but SLOWER than fp16 — see 3.3a):
OMP_NUM_THREADS=48 ./build/vlm_runner $M $MM ~/fujisan.jpg \
    --dtype int16 --threads 48 --bench 3

# int16 GEMM unit test (vs fp32 ref; expect rel(L2) ~2.6e-5; 1T == 48T):
make CC=fcc OPENMP=1
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -fopenmp -Ikernels -I. \
    -o /tmp/ti tools/test_int8_gemm.c kernels/int8_gemm.c kernels/kernel_6x4_int8.S -lm
OMP_NUM_THREADS=48 /tmp/ti        # int16 correctness
OMP_NUM_THREADS=48 /tmp/ti bench  # int8 vs int16 GEMM speed
OMP_NUM_THREADS=48 /tmp/ti 8      # int8 correctness

# int8 GEMM unit test (vs fp32 ref; expect rel(L2) ~0.0056, 1T == 48T):
make CC=fcc OPENMP=1
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -fopenmp -Ikernels -I. \
    -o /tmp/ti tools/test_int8_gemm.c kernels/int8_gemm.c kernels/kernel_6x4_int8.S -lm
OMP_NUM_THREADS=48 /tmp/ti

# fused int8 GEMM unit test (expect BIT-IDENTICAL vs non-fused, 0 differ):
OMP_NUM_THREADS=48 /tmp/ti fused           # 1T too: OMP_NUM_THREADS=1
OMP_NUM_THREADS=48 /tmp/ti fused bench     # fused vs non-fused GEMM speed

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

- **CPU reference build (fixed)** — `common/transformer.h:1848` used `xi8`
  / `inv` in the scalar 1-row tail that sat *outside* the
  `#if defined(__ARM_FEATURE_SVE)` block where they were declared, so any
  non-SVE build (the CPU reference) failed with `'xi8' undeclared`. Fixed by
  hoisting the x-quantize (portable `tf_quantize_f32_to_int8`) + `inv` above
  the SVE `#if`. Separately, the `ref:` target was missing `-D_GNU_SOURCE`
  (this glibc only defines the sized `__CPU_*_S` macros otherwise, so
  `CPU_ZERO`/`CPU_SET` in transformer.h's NUMA binder linked as bare symbols).
  `make ref` now builds `cpu/vlm/test_vision`. Note the reference **dump**
  flag `VLM_DUMP_REFERENCE` is stale (no code honours it), so a full
  `tensor_diff` dump comparison still needs the dump hooks re-wired in the CPU
  vision path — but the reference binary itself builds and runs.
  (Verified: the SVE VLM numerics are unchanged — fp16 455.6237 / int8
  452.0263 / int16 458.2751.)
- **Stale norm reference** — the readme documents `norm=455.7341`; the
  current build gives `455.6237` (bit-identical across 12T/48T). The gap
  is a stale reference (older model/build), **not** a regression — the
  fused patch_embed is provably bit-identical to the prior path.
- **`clock_gettime(CLOCK_MONOTONIC)` is unreliable after tight SVE loops** on
  this node — it returns a huge (wrong) delta (observed ~1e6–1e7 s) around a
  sustained SVE microbenchmark, while `CNTVCT_EL0` is correct. It is **not**
  affected in the VLM runner (SVE kernels are interspersed with C; the runner's
  clock_gettime matches CNTVCT exactly — verify with `VLM_CNTCHECK=1`).
  **Use `CNTVCT_EL0` for any tight-loop microbenchmark** (`bench_gemm.c` and the
  int8 benches already do).
- **`getenv` build break (fixed)** — `common/ggml_dequant.h:1553` calls
  `getenv` (for the `TF_*` env toggles) but never included `<stdlib.h>` →
  "implicit declaration" warning under `-std=c11 -Wall`. Fixed by adding
  `#include <stdlib.h>` to `ggml_dequant.h` itself (a header should include
  what it uses; the transitive include via `gguf_loader.h` did not reliably
  reach the use site).
