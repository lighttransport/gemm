# WS6 — Q8 small-M NaN: reproduced, root-caused, fixed (single node)

**Status:** reproduced single-node, root cause identified, fixed + validated. Not committed.

## Root cause (NOT the doc's "uninitialized read" hypothesis)
The Q8 activation scale is `amax / 127` and was stored as **fp16** in both
`ds4f_quant_x_sdot_into` (`common/ds4f_impl.h`) and the sibling `tf_quant_x_sdot_blocks`
(`common/transformer.h`). When a 64-element activation block has
`amax > 127 * 65504 ≈ 8.3e6`, `ggml_fp32_to_fp16(scale)` **overflows to +Inf**. In the kernel
(`matvec_sdot_8row` / `_3x`) the dequant `sc = w_scale * Inf = Inf`, and for a **zero-dot lane**
`acc += 0 * Inf = NaN` → NaN poisons the whole output. This is:
- **Q8-specific** (only the Q8 path stores an fp16 activation scale; bf16 dense has none → "bf16 dense always works"),
- **magnitude-dependent** → looks **nondeterministic** across runs/inputs,
- reachable: the synthetic model's last-hidden `||x|| = 2.5e8`, and the verify path feeds
  **un-normalized** activations (e.g. the o-proj input / accumulated residual) to Q8 GEMMs.
  (Plain *decode* matvecs take RMSNorm'd, bounded inputs → never overflow → "decode is fine".)

The doc's hypothesis (uninitialized/tail read in the small-M kernel) is **not** the cause: the
kernel and the `__thread` xscratch are correct by inspection for K%64==0 (xs buffer = cap/64
always covers any non-growing call; reads in-bounds; accumulators initialized).

## Single-node reproduction
`tools/ws6_q8_stress.c` hammers the Q8 GEMM/decode path at M=1,2,3 with adversarial X. Mode 3
(one ~1e8 spike per 64-block) reproduces immediately:
```
!! NaN/inf: iter=0 shape=[8192,4096] M=1 mode=3 bad=8192
DONE iters=3000  total NaN/inf outputs=60825600  -> REPRODUCED
```
(`ds4f_gemm_test` never triggers it — its random X has amax ~ O(1), far below the 8.3e6 threshold,
which is why M=1,2 Q8 cases already passed there.)

## Fix — fp32 activation scale (the int8 quants are unchanged)
Store the activation scale as **fp32** instead of fp16 (the *weight* scale stays fp16 — weights are
bounded). The int8 values `xq` are unchanged (`inv = 127/amax` is fp32), so this is a pure
scale-precision/overflow fix, and it makes the spike case *correct*, not merely finite
(`127 * (spike/127) = spike`). Files:
- `common/ggml_dequant.h`: `matvec_sdot_8row` / `matvec_sdot_8row_3x` — `xscale`/`xs0..2`
  `const uint16_t*` → `const float*`; drop `ggml_fp16_to_fp32`.
- `common/ds4f_impl.h`: `ds4f_quant_x_sdot_into` writes `float`; `ds4f_xs_buf` /
  `ds4f_q8_xscratch` → `float` (sizeof(float)); the 3 caller decls (decode matvec, mv_bd_worker,
  gemm worker) `uint16_t *xs` → `float *xs`.

## Validation (single node, native fcc)
| gate | result |
|------|--------|
| `tools/ws6_q8_stress.c` (the repro) | **NaN/inf = 0** (was 60,825,600); relL2(q8 vs bf16, normal X) 9.7e-3 |
| `ds4f_gemm_test` | **ALL OK 205/205**; Q8 M=1,2 argmax-exact, relL2 ~8.6e-3 (slightly better than fp16), nonfinite=0 |
| `ds4f_exact_test` / `ds4f_tierb2_test` | 5e-8 / 2e-6 (unchanged) |
| `ds4f_runner` Q8 dense decode | builds, NaN=0 |

## Caveats / follow-ups (main session, 11n)
1. The fix **changes Q8 decode numerics slightly** (fp16→fp32 scale precision) — a "coherent"
   change. gemm_test shows argmax is not flipped on its shapes, but the committed `DS4F_Q8_DENSE`
   decode path should get an **11n token-identical A/B vs bf16** to confirm token-identity still
   holds. (Normal decode inputs don't overflow, so only sub-1e-3 scale rounding changes.)
2. Whether this is the **complete** fix for the 11n `ds4f_forward_verify` NaN needs the multi-node
   verify path to confirm (it's a real bug fixed regardless; the verify's un-normalized activations
   make it the likely cause). After 11n confirms verify is NaN-free under Q8, the
   `DS4F_SPEC/GEMM_DECODE + Q8_DENSE` guard in `ds4f_ep_runner.c` can be removed (left in place here;
   that file also holds unrelated in-progress MTP work, so I did not touch it).
3. The sibling `tf_quant_x_sdot_blocks` (`transformer.h`, legacy GPT path) has the **same latent
   bug** — out of WS6 scope (different model path), noted for completeness.

## Artifacts
- `tools/ws6_q8_stress.c` — reproducer (mode 3 = spike). `OMP_NUM_THREADS=12 taskset -c 12-23 /tmp/ws6 12 1 3000`.
