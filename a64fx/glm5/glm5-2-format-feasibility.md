# GLM-5.2 decode — format-change feasibility (breaking the int8 convert-bound compute wall)

The campaign (`glm5-2.md`) showed decode is bottlenecked by the **int8 w8a16 GEMV**, which is
**convert-throughput-bound**: every weight byte is `load → widen(u8→u32) → convert(u32→f32) → f32 FMA`,
and the converts compete with the FMAs on the FP pipes (K0b: ~6% of peak, IPC 2.3, HBM 1.6%). Comm
levers are exhausted (lean-AR ~0, overlap broken, batching plateaus at ~1.67× because it amortizes comm
but not this compute). **The only lever that breaks the wall is changing the compute format.** This
evaluates the two candidates the user named — w8a8 and BFDOT — plus FP16.

## TL;DR / recommendation

- **BFDOT is NOT available on A64FX.** A64FX is **Armv8.2-A + SVE**; the SVE BF16 dot (`BFDOT`/`BFMMLA`,
  FEAT_BF16) is **Armv8.6**. bf16 on A64FX is storage-only (widen to f32 to compute) → no compute win,
  2× memory. **Ruled out.**
- **w8a8 SDOT is the viable path.** A64FX *does* have SVE `SDOT`/`UDOT` (int8·int8→int32, 4 MACs/lane in
  one instruction) — its peak-throughput integer path. It **eliminates the per-element convert** and packs
  **4× the MACs/instruction**. Expected **~4–6× on the GEMV kernel**, which makes decode **comm-bound
  again → batching re-engages** → the high ceiling (~20–30 tok/s aggregate) reopens.
  **Cost:** quantize activations to int8 (accuracy risk + calibration) + a kernel rewrite.
- **FP16** is a smaller, lower-risk fallback (~1.5–2× FMA rate, no int8-activation accuracy risk) but stays
  convert-bound and costs 2× weight memory — modest, secondary.

**Recommend:** prototype the **w8a8 SDOT GEMV locally in qlair** (qlair is accurate for compute-bound
kernels, per K0) to confirm the ~4–6×, run a **small w8a8 accuracy eval**, and only then commit the
model-side activation-quant work. High payoff, real accuracy risk — gate on the eval.

---

## The wall (recap, from K0b)

`glm5_matvec_int8_8row` per weight element: `svld1ub` + `svcvt_f32_u32` + `svmla`. The convert and the
FMA both issue on the two FP pipes → ~half the pipe slots are converts, not MACs. 8 output rows already
give 8-way ILP (not latency-bound), so multi-accumulator doesn't help (native u2 was 0.75×). The op count
per MAC is the ceiling. To go faster you must **remove the convert** and/or **pack more MACs per op** —
exactly what SDOT does.

## Option A — w8a8 (int8 weights × int8 activations) via SVE SDOT  ★ recommended to prototype

**Instruction:** `svdot_s32(acc, a_i8, b_i8)` — signed int8 × signed int8, 4 products summed into each
int32 lane. At 512-bit SVE = 16 int32 lanes → **64 int8 MACs per instruction**, no convert. (A64FX issues
it on the FLA/FLB pipes; ~2/cycle.) Compare the current path: 16 f32 MACs/instr **plus** a competing
convert. SDOT ≈ **4× the MAC density and 0 converts** → **~4–6× on the GEMV** (confirm in qlair).

**Kernel shape (per 8-row block, mirrors the current one):**
1. **Weights → signed int8 once at load.** The checkpoint is offset-binary (`b = q+128`); re-bias to signed
   `q = b-128` at load and store int8 (keeps the 1-byte footprint, no accuracy change). Per-group f32
   `weight_scale` unchanged.
2. **Activations → int8 per token (dynamic).** `sx = max|x| / 127`; `x_q = round(x/sx)` (clamp ±127).
   Per-token scalar scale (cheap; one pass over `x[H]`). Optionally per-channel for accuracy (below).
3. **GEMV:** for each group, `acc_i32 += svdot(w_i8_row, x_q_i8)`; then `y_row += acc_i32 * weight_scale_g * sx`.
   One int32→f32 convert **per group** (not per element) → convert cost drops ~128×.

**Signedness note:** SDOT is signed×signed and UDOT unsigned×unsigned; A64FX has **no USDOT** (mixed,
FEAT_I8MM/8.6). Hence re-bias weights to *signed* int8 at load (step 1) so both operands are signed → plain
`svdot_s32`. Clean, no per-element correction term needed.

**Accuracy (the real risk):** w8a8 with **per-token dynamic activation quant** is the standard "full int8"
inference. Large MoE models can have **activation outliers** (esp. in the gate/router and some down-proj
inputs) that clip under a single per-token scale → quality loss. Mitigations, in increasing effort:
per-token *per-channel* activation scales; SmoothQuant-style weight↔activation scale migration (offline,
one-time); or **mixed precision** — keep the router gate (argmax-critical, small) and any outlier-heavy
layer in w8a16, w8a8 the rest. **Must be measured** (perplexity / logit cosine / a task eval) before trusting.

**Memory:** unchanged (1-byte weights) + tiny int8 activation buffers. Fits wherever int8 fits (≥32n).

**Effort:** kernel rewrite (~the size of `glm5_int8.h`) + an activation-quant pass + load-time re-bias +
the accuracy eval + optional mixed-precision plumbing. Medium–large, dominated by the accuracy work.

## Option B — FP16 (fallback, secondary)

A64FX has full-rate **FP16 SVE arithmetic** (`svmla` on `svfloat16_t`, ~2× the FP32 FMA throughput).
Path: keep int8 weights, convert `int8 → f16` (instead of f32), accumulate in f16 (or f32 via widening).
- **Win:** the f16 FMA is ~2× faster, and f16 has 32 lanes/vector → more MACs/instr than f32's 16.
- **But:** still one convert per element (convert-bound persists, just cheaper target), and f16
  accumulation over H=6144 risks precision loss (use f32 accumulate → back to 16-lane). Net likely
  **~1.3–1.8× on the GEMV**, less than SDOT, but **no activation-quant accuracy risk**.
- Or full **w16a16 in f16** (weights f16): 2× memory, ~2× FMA, no convert — but 2× bandwidth may bite at
  higher M. Modest.

## Option C — bf16 BFDOT  ✗ ruled out on A64FX

`BFDOT`/`BFMMLA` (bf16·bf16→f32 dot) is **FEAT_BF16, Armv8.6-A**. **A64FX is Armv8.2-A and lacks it.** A
bf16 weight on A64FX must be widened to f32 (a `lsl #16`) then f32-FMA'd — i.e. the same convert+FMA
structure as int8, but with **2× the weight bytes/bandwidth**. No compute win, strictly worse memory.
(BFDOT would be the right answer on a Neoverse V1/V2 / Grace, not on Fugaku.)

## Quantified end-to-end estimate

From the 32n bd=0 profile, the int8-GEMV stages are ~**56% of the token** (qkv 20 + shared 20 + router 14
+ o_proj 10 + experts 9 + dense 2 ms/tok = ~75 ms), attention ~20% (partly f32 dot/axpy, not int8 GEMV),
comm ~30–40%.

| path | GEMV speedup | single-stream e2e | with batching (comm amortized) |
|---|---|---|---|
| current int8 w8a16 | 1.0× | 1.75 tok/s/slot | 1.67× at M=8 (comm-limited, plateaus) |
| **w8a8 SDOT** | **~4–6×** | **~1.3–1.7×** (comm becomes the floor) | **compounds: decode is comm-bound again → batching scales → ~2–3× aggregate, ~20–30 tok/s** |
| FP16 | ~1.3–1.8× | ~1.15–1.4× | modest |

The strategic point: **the convert-bound compute is *why* batching plateaued** (K3). Remove it with SDOT
and decode returns to the comm-bound regime where batching (and MTP) actually scale — that is what reopens
the original ~20 tok/s target, not any single kernel factor in isolation.

## Validation plan (node-hour-economized, matches the campaign method)

1. **Local qlair SDOT kernel bench (0 nh).** Add `int8_gemv_sdot` to `qlair/kernels/int8_gemv_bench.c`
   (svdot_s32, signed weights, int8 x). Compare cycles vs the current kernel under `qlair -n` (cycle
   mode). **qlair is accurate for compute-bound kernels** (K0: dot 1.98× vs 1.95×), so this confirms the
   ~4–6× locally. Cross-check numerics vs the scalar w8a8 reference.
2. **One native probe (~2 nh)** — K0-style, run the SDOT bench with `fcc` on 1 node to lock the absolute
   (qlair mis-ranks only memory-bound kernels; this is compute-bound, so it should agree).
3. **Accuracy eval (local + small job).** w8a8 vs w8a16 on a handful of real prompts: logit cosine +
   argmax-agreement per layer, and a short task eval. Decide per-token vs per-channel vs mixed-precision.
   This gates everything — a w8a8 that loses quality isn't worth the speed.
4. **If (1)+(3) pass:** implement the kernel + activation-quant behind a flag (`GLM5_W8A8=1`, default off),
   land the load-time re-bias, and run a real-weight decode A/B at 32n (reuse the K3 harness) → measure the
   e2e win + re-run the batching curve (which should now scale).

## Bottom line

- **w8a8 SDOT is the one A64FX-viable lever that can break the compute wall** and re-open the high decode
  ceiling; its blocker is accuracy, not the ISA.
- **BFDOT is not on A64FX** — drop it from consideration for Fugaku.
- Do the **local qlair SDOT bench + a small accuracy eval first** (cheap, decisive) before committing the
  larger activation-quant + mixed-precision engineering.
