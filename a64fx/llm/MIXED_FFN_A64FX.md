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

## Follow-up: decode barriers and worker-local palette cache

The same allocation now has an explicit `--decode-barrier` runner option:
`default`, `flat-spin`, `cmg-spin`, `cmg-wait`, or `flat-wait`. Sense loads
and stores use acquire/release atomics. CMG state has a 256-byte stride on
AArch64; padding alone did not improve the benchmark. Default behavior is
unchanged, and the spin policies are opt-in.

With the existing expanded FFN cache, append:

```sh
--mixed-ffn-cache-mib 13000 --decode-barrier flat-spin
```

The 96-token image test completed 95 decode forwards in 18.855 s:
**5.04 tok/s, 96/96 baseline greedy token IDs**. This is 18% faster than
the earlier 4.28 tok/s FP32-cache checkpoint, not a result for the new
four-bit cache. Profile totals were FFN gate/up 6537.1 ms, FFN down
3395.7 ms, SSM prepare 188.8 ms, SSM scan 205.7 ms, and head 1429.7 ms.
The output remains a coherent description of the mountain image.

`--iq4-cache-mib N` is a separate, opt-in, decode-only representation of
IQ2_XXS, IQ2_XS, IQ2_S, IQ3_S, and IQ3_XXS weights. Four-bit indices refer
to the original signed palette; this is not re-quantization to GGML IQ4.
Original per-16 FP32 scales are retained, using 192 bytes per 256 weights
instead of the expanded cache's 320 bytes. FP32 activation precision is
unchanged except for IQ3_XXS, which already uses Q8 activations in the
baseline. `--mixed-iq-q8` still explicitly opts into lossy activations for
the other supported formats.

The builder covers eligible FFN, SSM, and attention projections, keeps the
source tensors intact, and allocates one slab inside each pinned worker.
Each worker temporarily selects the default local memory policy, then
restores its original policy. A user budget and a 6 GiB `MemAvailable`
reserve bound allocation, including an allowance for huge-page rounding.
Packing or headroom failure discards the new representation before publishing
it. The new and old cache options are mutually exclusive; neither is
supported in stdio serving mode.

Bounded tests pass for all five formats: 4096 real rows per format, exact
weight reconstruction, changed and zero activations, empty/nonzero row
slices, output padding, and an independent double-precision dot reference.
The worker-local builder and dispatch agree with the contiguous cache on
every output tested; a repeated cache request allocates nothing. Worst
scaled dot error was 1.33e-8 (threshold 2e-6). Representative FP32 direct
versus worker-local cache times were 0.310/0.114 ms for IQ2_S and
0.736/0.337 ms for IQ2_XS. These reused-matrix microbenchmarks do not
establish whole-model throughput or measured HBM traffic.

Build these bounded tests on the frontend and run on the compute node:

```sh
mkdir -p tmp/llm-goal
for test in test_iq4_decode_cache test_decode_sync; do
    TMPDIR="$PWD/tmp/llm-goal" fccpx -Nclang -O3 -march=armv8.2-a+sve \
        -ffp-contract=fast -fopenmp -D_GNU_SOURCE -ffunction-sections \
        -fdata-sections -Wno-unused-function "a64fx/llm/$test.c" \
        -Wl,--gc-sections -lm -lpthread -lhwb -o "tmp/llm-goal/$test"
done

# Compute node, with no concurrent inference/benchmark:
NUMA_INTERLEAVE=1 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
    tmp/llm-goal/test_iq4_decode_cache \
    /local/u14346/qwen38-gsq/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf
for mode in 0 1 2 3 4; do
    OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
        tmp/llm-goal/test_decode_sync "$mode"
done
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
    tmp/llm-goal/test_decode_sync stream
```

Barrier ordering checks pass at 4, 12, and 48 workers. Flat spin reduced
the 48-worker microbenchmark from about 140 us (baseline hierarchical wait)
to about 14 us. The bounded read test measured about 380 GB/s with a
main-thread interleaved allocation versus 737 GB/s when pinned workers
allocated their own slabs. These are synthetic read rates, not decode rates.

Evidence is in `tmp/llm-goal/`: `barrier_modes_image.log`,
`barrier_modes_output.txt`, `iq4local_test.log`, `stream.log`, and
`workeralloc.log`. The next section records the subsequent full-model
four-bit-cache validation. **30+ decode tok/s remains unachieved.**

## Follow-up: full-model IQ cache and FP32 palette lookups

The saved post-commit runs in `tmp/llm-goal/iq4_full_image.log` and
`iq4_q8_image.log` complete the earlier cache validation. The FP32-activation
run produces all 96 baseline greedy tokens at **6.18 tok/s** (95 forwards /
15.371 s). It caches 278 tensors / 12.360 GiB. The optional Q8 run reaches
8.17 tok/s (95 / 11.623 s), but matches only the first 77 tokens before
diverging. It caches 287 tensors / 12.810 GiB, so those timings also differ
in cache coverage. Q8 activations remain explicitly lossy and opt-in.
The sampled minimum available memory is 7.077 and 6.900 GiB respectively.

The next kernel revision keeps each signed palette in an FP32 SVE register,
eliminating repeated integer-to-float conversions after table lookups. These
small integer palette values are exactly representable in FP32. The native
IQ4_XS dot kernel additionally uses four independent accumulation chains;
weight and activation precision stay unchanged, but the FP32 addition order
changes. Greedy agreement on the measured prompt does not imply bit-identical
logits or agreement on every prompt.

Fresh bounded tests on job **51562789**, initial node **a25-4009c**, pass:

- `test_mixed_tokenmajor`: 42 passing checks in each of FP32 and optional-Q8
  modes; maximum scaled error 3.96e-8 against the independent double reference.
- Added synthetic IQ4_XS coverage uses K=256, 768, and 5120, nonlinear palette
  codes, signed six-bit scales, zero block scales, and changed/zero activations.
- `test_iq4_decode_cache`: all five formats pass packing and both activation
  modes, including worker-local dispatch, repeated construction, row slices,
  and output padding. Maximum scaled error is 1.33e-8.

The pending 16-row panel experiment was also run on this node. Its weight
reconstruction and both activation modes pass for all five formats, but it
does not justify production integration. For K=17408, row/panel FP32 times
are 0.753/0.812 ms (IQ2_XS) and 0.752/0.840 ms (IQ2_XXS); corresponding Q8
times are 0.177/0.417 and 0.177/0.426 ms. The smaller K=5120 Q8 matrices
improve modestly. The panel remains an experimental local microbenchmark;
the runner continues to use worker-local rows. Evidence:
`tmp/llm-goal/iq4_panel_v2_job51562789.log`, binary SHA256
`c20ca2239991ec18f42b7adabe250b9f1375c99d5d67d24bba63776d38c01636`.

### Paired full-model measurement

The continuation comparison uses one 48-core A64FX node at 2.0 GHz, with
the same 12,000 MiB cache budget and flat-spin barrier for both binaries.
Both cache **264 tensors / 11.708 GiB**, skipping 49. The model and mmproj
were staged with `stage_model.sh` using its `SRC` and `DST` overrides
(1 GiB chunks with fsync). Main-model weights use anonymous HBM allocation.
Each run is detached and guarded by a five-minute timeout and a 6 GiB
`MemAvailable` floor; no other inference ran concurrently.

| Revision | Decode forwards / seconds | Decode tok/s | Baseline greedy IDs | Minimum available GiB |
| --- | --- | --- | --- | --- |
| `297d590e` runner | 95 / 15.421 | 6.16 | 96/96 | 7.757 |
| FP32 palette + IQ4_XS accumulation chains | 95 / 14.510 | 6.55 | 96/96 | 7.728 |

This paired measurement improves throughput by **6.3%**. Output-head time
drops from 1365.8 to 960.7 ms across the 95 forwards; FFN gate/up goes from
6160.8 to 6036.7 ms, and FFN down from 2912.0 to 2744.6 ms. Logits remain
finite and non-degenerate. Process `read_bytes` stays constant throughout
the recorded decode samples (18 before, 17 after), at 16,241,852,416 and
16,196,894,720 bytes respectively. These are cumulative process I/O counters,
not model sizes or measured HBM traffic.

Frontend builds (Fujitsu TCSDS 1.2.43):

```sh
mkdir -p tmp/llm-continue-20260912
export TMPDIR="$PWD/tmp/llm-continue-20260912"
make -C a64fx/llm llm_runner CC=fccpx OPENMP=1
for test in test_mixed_tokenmajor test_iq4_decode_cache; do
    fccpx -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
        -fopenmp -D_GNU_SOURCE -ffunction-sections -fdata-sections \
        -Wno-unused-function "a64fx/llm/$test.c" \
        -Wl,--gc-sections -lm -lpthread -lhwb \
        -o "tmp/llm-continue-20260912/$test"
done
```

Bounded commands on the compute node:

```sh
export TMPDIR=/local/u14346/codex-tmp NUMA_INTERLEAVE=1
export OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores
M=/local/u14346/qwen38-gsq/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf
tmp/llm-continue-20260912/test_mixed_tokenmajor "$M" 4096
tmp/llm-continue-20260912/test_mixed_tokenmajor "$M" 4096 q8
tmp/llm-continue-20260912/test_iq4_decode_cache "$M"
```

The detached guardian `tmp/llm-continue-20260912/run_validation_v1.py`
records exact command arrays and runs this full-model command for each
preserved binary, with the same OpenMP/NUMA settings as above:

```sh
TF_PREFILL_EMBD_GEMM=1 TF_BATCH_SCRATCH_REUSE=1 TF_DPROF=1 TF_DUMP_LOGITS=1 \
  a64fx/llm/build/llm_runner "$M" \
  /local/u14346/qwen38-gsq/mmproj-Qwen3.8-27B-BF16.gguf common/fujisan.jpg \
  --prompt 'explain the image' --vit-dtype bf16 --max-seq 512 --max-gen 96 \
  --vit-threads 48 --llm-threads 48 --no-deepstack --prefill-gemm --seed 1 \
  --iq4-cache-mib 12000 --decode-barrier flat-spin
```

Logs and memory samples are in `tmp/llm-continue-20260912/`:
`validation.log` ends in `VALIDATION PASS`; `mixed_f32_output.txt`,
`mixed_q8_output.txt`, and `iq4_cache_output.txt` contain bounded results;
`before.log`, `after.log`, and their `_memory.jsonl` files contain the paired
measurement. The old binary is `llm_runner_before` (SHA256
`607467bbbd18f43db6efaf55e9c018a4a11719d6fe5f9f18a1b11ad4a939d2fe`);
the tested candidate is `build/llm_runner` (SHA256
`965e32e922a9b9e7587f58c6d0b03a195a510486393a4c976268a2369ce15f7d`).
Binary and guardian hashes were compared on the frontend and compute node
before launch. The bounded builds introduce no warnings; the isolated VLM
rebuild reports the existing `_GNU_SOURCE` redefinition warnings in
`cmg_pool.c` and `test_cmg_pool.c`.

The delivered `a64fx/llm/build/llm_runner` has SHA256
`525056586be12373877d0d521ccac3e1f27a6bfdba53bda1dd1330a39e053186`,
verified on both hosts. Its `.text`, `.rodata`, and `.data` sections are
byte-identical to the measured candidate. At completion no validation or
inference process remains, and available memory has returned to 29.76 GiB.
The existing bridge for job 51562789 is reachable at local port 42393
(login reverse port 32393); the staged model files remain on its initial
node for continuation until the allocation ends.
