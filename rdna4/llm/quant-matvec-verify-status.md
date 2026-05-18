# Quant Matvec Kernels — A/B Verification Status

**Date:** 2026-05-17; updated 2026-05-18 for Q5_K MW decode path
**Tool:** `./test_hip_llm --verify-quant-kernels`
**Method:** Identical raw quant bytes fed to the HIP `matvec_<type>_f32`
kernel and to the CPU `dequantize_row_<type>` + scalar dot product. Random
bytes generated deterministically (xorshift seeded by ggml_type). Rows where
either side produces NaN/Inf (from random bytes accidentally encoding FP16
NaN scales) are filtered before computing rel-L2 — typically ≥95% of rows
remain valid. Threshold: rel_l2 < 1e-4 (FP reduction-order noise floor).

## Results (RX 9070 XT, n_rows=64, n_cols=512)

| type     | rel_l2     | max_abs   | result |
|----------|-----------:|----------:|:------:|
| Q4_K     | 2.27e-07   | 7.5e-01   |  PASS  |
| Q5_K     | 1.73e-07   | 1.5e+00   |  PASS  |
| Q6_K     | 1.72e-07   | 4.3e+00   |  PASS  |
| Q4_0     | 7.73e-08   | 9.8e-03   |  PASS  |
| Q4_1     | 9.82e-08   | 2.3e-02   |  PASS  |
| Q5_0     | 1.20e-07   | 2.0e-02   |  PASS  |
| Q5_1     | 1.18e-07   | 4.7e-02   |  PASS  |
| IQ2_XXS  | 4.44e-08   | 9.4e-02   |  PASS  |
| IQ2_XS   | 1.33e-07   | 9.4e-02   |  PASS  |
| IQ2_S    | 1.34e-07   | 5.0e-01   |  PASS  |
| IQ3_XXS  | 1.23e-07   | 5.0e-01   |  PASS  |
| IQ3_S    | 2.21e-07   | 1.0e+00   |  PASS  |
| IQ1_S    | 3.89e-07   | 6.3e-02   |  PASS  |
| IQ1_M    | 5.19e-07   | 1.3e-01   |  PASS  |
| IQ4_NL   | 1.13e-07   | 1.3e-01   |  PASS  |
| IQ4_XS   | 2.08e-07   | 6.0e+00   |  PASS  |
| TQ1_0    | 2.14e-07   | 3.9e-03   |  PASS  |
| TQ2_0    | 2.53e-07   | 1.1e-02   |  PASS  |

**18 PASS, 0 FAIL, 0 SKIP.**

## What this closes

The IQ/TQ quant types above were ported into the HIP runner via the
`rdna4/llm: port IQ2_XS / IQ2_S / IQ3_S / IQ4_XS dequant kernels` family of
commits (foundation + 27B 9.2x), but were only exercised through end-to-end
generation. E2E covers correctness *coarsely* — a small per-type bug can
hide as long as overall generation quality stays "reasonable" (sampling
hides per-logit drift; rmsnorm + softmax + temperature mask small
divergences). This standalone A/B closes that window: byte-identical
inputs, both paths must agree to FP reduction noise.

The 2026-05-17 update adds K-quant and legacy Q4/Q5 coverage for decode
kernel-internals work, so the verifier now covers the 9B Q4_K_XL path as
well as the 27B IQ3_XXS path.

The `max_abs` column reflects absolute dot-product deltas on rows where the
FP16 scale field of a random block happens to land at a large finite
magnitude (e.g. IQ4_XS hits ~6.0). Per-row *relative* error stays at FP
rounding noise — confirmed by the rel_l2 column.

## Reproducibility

```sh
cd rdna4/llm
make test_hip_llm
./test_hip_llm --verify-quant-kernels

# Isolated kernel timing; no model needed
./test_hip_llm --bench-quant-matvec Q4_K 4096 4096 500 5
./test_hip_llm --bench-quant-matvec Q5_K 248320 5120 20 3
```

No model needed. Standalone runs in well under a second per type.

## Notes for future kernel work

If a new quant matvec kernel is added (or an existing one rewritten for
perf — e.g. the kernel-internals attack discussed in
`decode-graph-capture-audit.md` / `perf-vs-peak.md` Tier-4), add a row to
the `cases[]` table in `run_verify_quant_kernels` (test_hip_llm.c). The
runner-side `hip_llm_verify_quant_matvec` already accepts a function-pointer
callback for the CPU reference dequantizer — no runner change needed unless
a new GGML type is added to the matvec dispatcher.

`LLM_QUANT_MATVEC_OPT=1` currently enables an experimental one-warp-per-row
Q4_K path. It is left off by default because microbench and 9B decode timing
showed it was not a net win. On RX 9070 XT with `Q4_K 4096x4096`, serial
median timing was 0.0755 ms for the default kernel versus 0.0799 ms for the
one-warp path (`--bench-quant-matvec Q4_K 4096 4096 500 5`).

`matvec_q5_K_mw_f32` is the default Q5_K path as of 2026-05-18. It maps one
warp to each row and packs 8 rows per block, which is a much better launch
shape for the full-vocab logits projection. `LLM_Q5_K_MW=0` disables it for
A/B and rollback. On RX 9070 XT with `Q5_K 248320x5120`, median timing
improved from 7.096 ms/launch (`LLM_Q5_K_MW=0`) to 1.787 ms/launch with the
default path (`--bench-quant-matvec Q5_K 248320 5120 20 3`). End-to-end
decode improved from 67.785 to 63.404 ms/tok on Qwen3.6-27B IQ3_XXS and
from 45.179 to 36.372 ms/tok on Qwen3.5-9B Q4_K_XL, with identical A/B
first/last decoded token ids.

`matvec_q6_K_f32` uses a 64-thread launch as of 2026-05-19. This keeps the
same one-row-per-block arithmetic and only reduces idle lanes/reduction
overhead for decode-sized rows. On RX 9070 XT:

| shape | before | after |
|---|---:|---:|
| `Q6_K 5120x5120` | 0.143 ms | **0.0899 ms** |
| `Q6_K 17408x5120` | 0.426 ms | **0.2895 ms** |
| `Q6_K 5120x17408` | 0.558 ms | **0.4696 ms** |

The same final binary passed `--verify-quant-kernels` with 18/18 PASS.
