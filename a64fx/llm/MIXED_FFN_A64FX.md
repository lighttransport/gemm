# Mixed-format FFN decode — 2026-09-12

Continuation of [the GSQ single-node validation](QWEN38_GSQ_A64FX.md), on
interactive job 51563926 / a25-2212c, 48 A64FX cores at 2.0 GHz. Same staged
Qwen3.8-27B GSQ weights, `common/fujisan.jpg`, `explain the image` prompt,
384x256 encoder input, 109 prompt tokens, greedy seed 1.

## Changes

- `mixed_iq_decode.h`: fused SVE lookup/FP32 dot products for IQ2_XXS,
  IQ2_XS, IQ2_S, IQ3_S, IQ1_S, IQ1_M and Q2_K. No temporary dequantized row
  and no additional activation quantization on the default path. Existing
  IQ3_XXS activation quantization is unchanged.
- Specialize format branches at constant-type call sites. Decode IQ2_XXS
  indices in registers, avoiding a store-to-load round-trip through the stack.
- IQ1 kernels represent the original `grid +/- 1/8` exactly as
  `(8*grid +/- 1)/8`. Q2_K retains its original scale/min arithmetic.
- `--mixed-iq-q8` explicitly opts into Q8 activations for the four IQ2/IQ3_S
  formats. Inputs are packed once per worker row slice, never cached by
  sampled input values. This option changes token choices; it is not the
  default and is not a lossless optimization.
- `--mixed-ffn-cache-mib N` optionally expands eligible FFN codebooks into
  signed bytes and original FP32 scales (320 bytes per 256 weights). Every
  reconstructed weight must equal the original dequantized value exactly.
  Original model weights remain intact for prefill and fallback. Allocation
  respects both the explicit budget and a 6 GiB `MemAvailable` reserve.
  This single-request runner releases the packed vision cache first and
  reports cache construction separately from prefill/decode timings.
- Fix missing shared-layout dependencies in the vision Makefile. Otherwise
  adding a field to `qtensor` could leave `vit_a64fx.o` with an incompatible
  `vision_model` layout and crash during vision cache construction.

## Reproduce

Build on login1, not the benchmark node:

```sh
mkdir -p tmp/mixed-ffn
TMPDIR="$PWD/tmp/mixed-ffn" make -C a64fx/llm CC=fccpx OPENMP=1 llm_runner
TMPDIR="$PWD/tmp/mixed-ffn" fccpx -Nclang -O3 -march=armv8.2-a+sve \
    -ffp-contract=fast -fopenmp -D_GNU_SOURCE -ffunction-sections -fdata-sections \
    -Wno-unused-function a64fx/llm/test_mixed_tokenmajor.c \
    -Wl,--gc-sections -lm -lpthread -lhwb -o tmp/mixed-ffn/test_mixed
```

Run the numerical test on the allocated node, once without and once with the
optional fourth `q8` argument:

```sh
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
    tmp/mixed-ffn/test_mixed \
    /local/u14346/qwen38-gsq/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf 4096
```

Run inference detached, with only one model/benchmark active:

```sh
NUMA_INTERLEAVE=1 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
TF_PREFILL_EMBD_GEMM=1 TF_BATCH_SCRATCH_REUSE=1 TF_DPROF=1 TF_DUMP_LOGITS=1 \
timeout 240s a64fx/llm/build/llm_runner \
    /local/u14346/qwen38-gsq/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf \
    /local/u14346/qwen38-gsq/mmproj-Qwen3.8-27B-BF16.gguf common/fujisan.jpg \
    --prompt 'explain the image' --vit-dtype bf16 --max-seq 512 --max-gen 96 \
    --vit-threads 48 --llm-threads 48 --no-deepstack --prefill-gemm --seed 1
```

Append `--mixed-ffn-cache-mib 13000` to test the lossless expanded cache.
Append `--mixed-iq-q8` only when intentionally testing lossy activations.
Do not set `OMP_WAIT_POLICY=active`; passive also substantially slowed the
vision encoder in this allocation, so the measurements use the runtime default.

The compute-side shared-filesystem cache sometimes exposed older contents at
an edited pathname. Tests used unique executable/script names and verified
SHA256 on both hosts; cross-compilation used the frontend's current sources.
`llio_transfer --purge` did not refresh the affected script in this allocation.
Do not assume an edited launch script is visible without checking its contents.

## Numerical and kernel checks

Real GGUF rows from all ten non-IQ3_XXS matrix formats pass independent weight
dequantization plus double-precision dot references. Cases cover N=1 matvec,
N=3/9 prefill, changed/zero inputs, nonzero row slices, empty row slices,
padded strides and a K=37 FP32 tail. Cache construction additionally checks
every reconstructed weight in all 4096 tested rows of each eligible format.
The Q8 test's oracle uses reconstructed Q8 activations; this is deliberately
separate from the unquantized-input oracle.

The existing IQ3 token-major suite also passes, including input mutation and
the pool-restart regression (48 distinct pinned cores, caller affinity restored).
A native non-SVE build succeeds with the existing unrelated
`matvec_bf16_8row_pv` implicit-declaration warning. A64FX cross-builds are
warning-free; `git diff --check` passes.

Representative 48-thread matvecs, M=4096, real model weights, FP32 activations:

| Format | K | Before (ms) | Fused (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| IQ2_S | 5120 | 1.368 | 0.297 | 4.6x |
| IQ3_S | 5120 | 1.195 | 0.207 | 5.8x |
| IQ2_XS | 17408 | 5.279 | 0.737 | 7.2x |
| IQ2_XXS | 17408 | 3.693 | 0.608 | 6.1x |
| IQ1_S | 5120 | 1.017 | 0.186 | 5.5x |
| IQ1_M | 5120 | 1.259 | 0.218 | 5.8x |
| Q2_K | 5120 | 1.058 | 0.217 | 4.9x |

These are small, repeatedly accessed matrices, not memory-controller bandwidth
measurements or a claim of these speedups for the whole model.

## End-to-end checkpoints

Decode excludes the first generated token, which is supplied by prefill.

| Configuration | Decode forwards / seconds | Decode tok/s | Accuracy check |
| --- | ---: | ---: | --- |
| Start of this pass | 95 / 102.069 | 0.93 | Baseline |
| All seven fused formats, no extra cache, FP32 activations | 95 / 27.549 | 3.45 | 96/96 baseline token IDs |
| All seven fused formats + lossless cache, FP32 activations | 95 / 22.214 | 4.28 | 96/96 baseline token IDs |
| All seven fused formats + cache + optional Q8 | 95 / 20.296 | 4.68 | 96/96 on this prompt; still lossy |
| Earlier fused revision + lossless cache | 31 / 7.851 | 3.95 | 32/32 baseline token IDs |
| Earlier fused revision + cache + optional Q8 | 95 / 21.920 | 4.33 | Coherent, token choices differ |

The default run's full prefill is 11.24 s (9.7 tok/s); its encoder takes
about 0.11 s. Output begins: “The image shows a majestic, snow-capped mountain
under a clear blue sky.” It continues coherently about the mountain's shape,
snow, forested slopes and foreground. Greedy agreement on this image/prompt
does not prove identical logits or equivalence on all inputs.

The final Q8 run happens to match all 96 baseline token IDs for this prompt,
but earlier Q8 configurations diverged. The activation conversion is lossy
regardless of this particular greedy match and remains opt-in.

The final FP32 cache run is **4.6x** the baseline decode rate, with 96/96
matching tokens and finite, non-degenerate logits. Its FFN gate/up plus down
time is 10.543 s / 95 forwards (111 ms/token), versus 730 ms/token before
this pass, a **6.6x** FFN improvement. The four-row cached-kernel experiment
did not improve the small-matrix timings consistently and was not retained.

The expanded cache stores 119 tensors / 12.347 GiB; construction took
3.65–3.75 s in these checkpoints, outside the quoted timings. With the
cache and original model resident, available memory remained above 7 GiB.
The monitored Q8 run had ~22.37 GiB in `HugetlbPages`; process `read_bytes`
stayed at 16,881,156,096 throughout sampled decode: no storage reads during
that interval. A detached monitor terminates the benchmark if available
memory falls below 6 GiB. CMG-local placement was slower (3.12 tok/s with
FP32 cache over 95 forwards), so interleaving is retained.

The no-cache default spends 10.787 s in FFN gate/up and 5.252 s in FFN down
over 95 forwards: 58% of decode, down from about 68% at baseline. SSM, QKV,
the output head and synchronization remain substantial. At 3.45 tok/s, the
original 9.6734 GB active packed-weight numerator is 33.37 GB/s, about 3.3%
of the 1024 GB/s peak cited in the preceding report. This is packed-equivalent
throughput, not actual measured HBM traffic; expanded-cache runs have a
different traffic volume.

**30+ decode tok/s is not reached.** The target requires further work across
the decode path, not just eliminating disk faults or speeding up mmproj.

Logs, launch scripts and preserved comparison binaries are in
`tmp/mixed-ffn/`, notably `baseline.log`, `test_final_f32.log`,
`test_final_q8.log`, `test_iq3_regression.log`, `final_nocache_image.log`,
`final_nocache_output.txt`, `cache_f32_abi_image.log`, `cache_q8_image.log`,
`final_cache_image.log`, `final_cache_output.txt`, and the corresponding
memory logs. `final_cache_q8_image.log` / `final_cache_q8_output.txt` contain
the final optional-Q8 run. Tested comparison binary SHA256:
`d436e74811e9a60948f4d3b5a663138e25fb7ec41cc9a48eae09a9da478e2009`.

The rebuilt normal-path `a64fx/llm/build/llm_runner` has SHA256
`d67d0caa3116dcc2096bdfe57c6bcd9945bbfb72445caf751307f608ff52a7dd`,
verified on both hosts. Its final FP32-cache smoke test (`delivery_cache_image.log`)
passed 32/32 baseline tokens: 31 forwards / 7.164 s = 4.33 tok/s,
3.719 s cache setup, 7.407 GiB available. Invalid cache budgets and the
unsupported stdio/cache combination return an error before model loading.
No inference/test process remains running; the interactive allocation is
left available for continuation until its six-hour limit.
