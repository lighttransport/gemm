# Local qlair kernel optimization — attention decode (2026-07-03, 0 node-hours)

Bench: `qlair/kernels/attn_decode_bench.c` (GLM5.2 absorb flash-attention inner loop, real dims
kv_lora=512, qk_rope=64, NOWN=16 heads, NSEL=128 ctx). Built cross-gcc SVE, cycle-profiled under qlair.
Build: `aarch64-linux-gnu-gcc -O2 -march=armv8.2-a+sve -fno-math-errno -fno-tree-vectorize -static`
(`-fno-tree-vectorize` stops the init loop's `%` auto-vectorizing into an SVE int op qlair lacks;
the kernel uses explicit intrinsics so it's unaffected). Run: `qlair -n 6G attn_decode_bench.elf`.

## Result: multi-accumulator dot breaks the FMA dependency chain
| variant | isolated pure dot (2048×512) | speedup |
|---|---|---|
| dot1 (single acc — current `glm5_dot_f32_opt`) | 623 µs | 1.00× |
| **dot4 (4 acc)** | **315 µs** | **1.98×** |
| dot8 (8 acc) | 270 µs | 2.31× |

Full attention inner loop (dot + online-softmax expf/axpy/scale): dot1 1485 µs → **dot4 1100 µs (1.35×)**,
numerically identical (ok=1, rel err 2.4e-6 = FP reassociation only).

## Why (qlair A64FX profiler, dot1 single-acc)
- **Data-dependency stalls: 1.28M of 1.8M cycles (~71%)** — the loop-carried `svmla` accumulator can't
  hide the A64FX FMA latency (~9 cyc, 2 FLA pipes). IPC 1.02, 3.7% of peak FLOPS. Classic latency-bound.
- 4 independent accumulators fill the pipes → ~2× on the dot; 8 gives a bit more (diminishing).

## Takeaways
- **dot4 is the sweet spot** (1.98×, 4 registers; dot8's extra 0.33× isn't worth the register pressure).
- The dot is ~42% of the attention inner loop → dot4 alone = **1.35× on attention**. To reach the "2×
  compute" ceiling lever, also optimize the **axpy/scale online-softmax accumulation** (~58% of the loop,
  another svmla-heavy path) — next kernel task.
- Maps to `glm5_dot_f32_opt` in `common/glm5_impl.h:100` (add a 4-accumulator path behind the existing
  `use_sve` flag). Validate via K0 (1-node native calibration) then K1 (32n end-to-end A/B).

## qlair notes (raise sim accuracy)
- Cycle mode (default, NOT `--native`) is required to see this win (`--native` = instruction-count timing,
  blind to FMA latency). Needs `-n 6G` (bench is 1.8M instr; default cap 1M).
- Cycle→ns not yet calibrated to real hardware — K0 (native 1-node kernel probe + `--friction`) sets that;
  until then trust the RATIOS (1.98×), not the absolute ns.
- qlair cycle-mode SVE gaps hit so far: auto-vectorized integer `%` (init), NEON scalar `shl d` (float
  printf ×const). Avoid via `-fno-tree-vectorize` + integer-only reporting. gcc SVE FMA/load/whilelo/faddv
  all fine.

## IMPLEMENTED (2026-07-03)
- **4-accumulator dot landed** in `glm5_dot_f32_opt` (`common/glm5_impl.h:100`), default ON,
  `GLM5_DOT_ACC4=0` restores single-acc for the A/B. **All local decode gates PASS** with it
  (glm5_sim: TEST1 lockstep, TEST2 batched-vs-single bit-identical, TEST3 MTP, TEST4 prefill-chunk).
- **axpy NOT changed** — tested `axpy4` (4-way unroll) in the bench: **1.03× (no win)**. The online-
  softmax AXPY has no loop-carried reduction (independent iterations), so it's throughput/memory-bound
  and unrolling doesn't help. Leaving it avoids useless complexity. The ~58% non-dot part of the attn
  loop would need a STRUCTURAL change (head-batched GEMV / fewer softmax passes) — deferred, higher risk.
- Timing fixed to real ns via `CNTFRQ_EL0` (qlair CNTVCT = **2 GHz**, not 1 GHz) → native (K0) and
  qlair ns are now directly comparable. Ratios unchanged (dot4 1.98× isolated / 1.35× full loop).

## K0 submitted (job 49420133, 1 node, ~2 nh)
`pjsub_glm5_kernel_probe_1n.sh` runs the SAME bench NATIVELY (fcc) → real A64FX ns vs qlair cycle ns
→ set `--friction` to <=10% (raises qlair sim accuracy). Then the local kernel loop is fully trusted.
Reference qlair ns (2 GHz, uncalibrated): full-loop dot1 742µs / dot4 550µs; pure dot dot1 312 / dot4 157 / dot8 135 µs.

## K0 CALIBRATION RESULT (job 49420133, native 1-node, cntfrq=100 MHz)

| kernel | NATIVE A64FX | qlair (2 GHz) | agree? |
|---|---|---|---|
| pure dot dot4/dot1 | **1.95×** | 1.98× | ✅ within 1.5% (abs within ~8%) |
| pure dot dot8/dot1 | 2.05× | 2.31× | ~ok |
| **axpy4/axpy1** | **1.93×** | **1.03×** | ❌ qlair badly wrong |
| full attn loop dot4 | 1.18× | 1.35× | qlair optimistic |

### qlair accuracy boundary (the calibration verdict)
- **Compute / FMA-latency-bound kernels (the dot): qlair is ACCURATE** — ratio within ~2%, absolute
  within ~8%. Trust qlair for these; no friction needed.
- **Memory-pipeline-bound kernels (store/load-bound AXPY): qlair is UNRELIABLE** — it saw 1.03× where
  the real hardware gives **1.93×** (store→load / store-buffer effects qlair's memory model misses).
  **Use native probes (K0-style) for memory-bound kernel decisions.** A single `--friction` scalar
  can't reconcile both (compute-accurate + memory-inaccurate), so no global friction was set; instead:
  **qlair for compute-bound ratios, native probes for memory-bound.**

### Consequence
- **axpy4 ADDED to `glm5_axpy_f32`** (bit-identical, native 1.93×) — my earlier "skip axpy" call was
  based on the wrong qlair number; K0 corrected it. This is the payoff of the native calibration.
- Combined dot4+axpy4 ≈ **1.24× on the attention inner loop** natively (attn is ~20–25% of the token
  → ~4–5% end-to-end). To reach the full "2× compute" lever, the **int8 GEMV kernel** (qkv 20 + shared 20
  + router 14 + o_proj 11 ms/tok — bigger than attn's 26) needs the SAME multi-accumulator treatment on
  its int8 SDOT. That is the next, higher-impact kernel target (`glm5_mv` / int8 matvec).
- KERNBENCH (native, th=1): wq_a M=1 13.7 GB/s, wq_b 8.8, wq_b_s8 7.0 — feeds `decode_sim.BW_NODE`
  (single-thread; the runner uses 48 threads → scale up).

## int8 GEMV investigation (glm5_matvec_int8_8row) — mostly a NEGATIVE result

Bench: `qlair/kernels/int8_gemv_bench.c` (real kernel, 2048×6144 gs=128, per-token GEMV shape).
qlair profile of the BASELINE: **IPC 2.30, HBM 3.7 GB/s (1.6% — NOT memory-bound), L2 hit 99%**, 12.2 GB/s
weight throughput. Data-dependency stalls are the top category but IPC 2.3 is already decent.

Tried `matvec_int8_8row_u2` (c-loop unrolled 2×, 16 FMA chains vs 8) → **0.98× (NO win)** under qlair.
Interpretation: the kernel already has **8-way ILP** (8 output rows) so it's NOT FMA-latency-bound like
the single-accumulator attention dot was. The real cost is **op throughput of the w8a16 dequant** —
every weight byte needs load → widen(u8→u32) → convert(u32→f32) → FMA; the converts compete with the
FMAs on the FP pipes, and multi-accumulator can't reduce the op *count*. Inherent to w8a16 (f32 acts).

**K0b NATIVE VERDICT (job 49420161, decisive):**
| variant | native ns | GB/s | vs baseline |
|---|---|---|---|
| baseline 8row | 1,919,510 | 6.6 | 1.00× |
| u2 (2× c-unroll) | 2,545,830 | 4.9 | **0.75× (SLOWER)** |

u2 is clearly WORSE natively (16 accumulators spill). qlair said 0.98×, native says 0.75× — qlair again
optimistic/mis-ranking (and ~1.9× too fast on absolute). **Native confirms: DO NOT change `glm5_int8.h`;
the baseline 8-row kernel is near-optimal.**

**CONCLUSION: the int8 GEMV is convert-throughput-bound (inherent to w8a16 f32-activations) — NO easy
kernel headroom.** The compute lever is the **attention dot/axpy only (~1.24× on attn ≈ ~4–5 % e2e)**.
The "2× compute" ceiling is OPTIMISTIC for int8 — reaching it needs a format change (w8a8 int8·int8 SDOT
= activation quantization, accuracy-affecting) or a bf16 BFDOT path, both out of scope for a kernel tweak.
Net: **batching (1.6–1.7×) and overlap remain the dominant decode levers; kernels give a modest e2e win.**
