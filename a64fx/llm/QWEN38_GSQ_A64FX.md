# Qwen3.8 GSQ single-node continuation — 2026-09-12

Later mixed-format decode optimization results are in
[MIXED_FFN_A64FX.md](MIXED_FFN_A64FX.md).

## Allocation and reproducibility

Interactive PJM job **51563926**, compute node **a25-2212c**, started
2026-09-12 13:08:43 JST, six-hour limit, 48 compute cores at 2.0 GHz,
normal (non-eco) mode. Submitted directly from login1 with `pjsub --interact`,
resource group `int`, project `hp250467`, one node and 87 GiB local scratch.
The supervised bash-over-HTTP bridge is reachable from this frontend at
`http://127.0.0.1:32394`; compute-side listener is `127.0.0.1:21264`.
The frontend and compute node share this checkout; no source deployment is
required between them. The allocation remains running after testing.

Build on the frontend (not the compute benchmark):

```sh
mkdir -p tmp/iq3-validation
TMPDIR="$PWD/tmp/iq3-validation" make -C a64fx/llm CC=fccpx OPENMP=1 llm_runner
```

Stage on the allocated node, using bounded direct-I/O staging:

```sh
a64fx/llm/stage_gguf_shards.sh \
    /home/u14346/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf \
    /local/u14346/qwen38-gsq
a64fx/llm/stage_gguf_shards.sh \
    /home/u14346/models/qwen38/27b/gsq/mmproj-Qwen3.8-27B-BF16.gguf \
    /local/u14346/qwen38-gsq
```

Measured staging: 9626 MiB main model in 183 seconds, 888 MiB mmproj in
16 seconds. Main weights load into anonymous huge pages (default runner
behavior; do not pass `--mmap`). The mmproj source is mapped during cache
construction; its compute kernels use the packed cache.

Run detached on the compute node, with no competing benchmark:

```sh
NUMA_INTERLEAVE=1 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
TF_PREFILL_EMBD_GEMM=1 TF_BATCH_SCRATCH_REUSE=1 TF_DPROF=1 TF_DUMP_LOGITS=1 \
a64fx/llm/build/llm_runner \
    /local/u14346/qwen38-gsq/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf \
    /local/u14346/qwen38-gsq/mmproj-Qwen3.8-27B-BF16.gguf common/fujisan.jpg \
    --prompt 'explain the image' --vit-dtype bf16 --max-seq 512 --max-gen 96 \
    --vit-threads 48 --llm-threads 48 --no-deepstack --prefill-gemm --seed 1
```

The 640x427 source is resized to 384x256: 384 vision patches become 96
merged image embeddings. The complete LLM prompt has 109 tokens.

## Findings and changes

- The filename does **not** describe every weight format. Tensor data totals
  10,083,362,816 bytes; IQ3_XXS accounts for 2,043,166,720 bytes. IQ3_S,
  IQ2_S, IQ4_XS and several other IQ/K formats account for most of the rest.
  Optimizing only IQ3_XXS cannot accelerate most of this model.
- Removed an incorrect activation cache from IQ3 token-major GEMM. Testing
  only eight samples of a reused input buffer missed real activation changes.
  Each invocation now packs its own inputs once and shares the pack over rows.
  Restored `memcpy` metadata reads: a 98-byte IQ3 block does not guarantee
  four-byte alignment. Integer scaled accumulation avoids redundant adds.
- Generic mixed-format prefill now expands a weight row once, reuses its SVE
  vectors across four tokens, and uses the existing OpenMP team. Previously
  it used scalar dot loops and raw pthreads inheriting the pinned primary's
  single-core mask.
- `NUMA_INTERLEAVE=1` now also binds the decode pool to the compute cores,
  including after prefill recreates it. Before this fix all 47 recreated
  workers inherited CPU 12. The caller's prior affinity is restored after
  dispatch; the interleaved memory policy is unchanged.
- IQ4_XS matvec uses a register-resident nonlinear codebook and SVE table
  lookup instead of expanding each group through a scalar temporary array.
- The short text prefix is batched without computing unused logits. The
  final prefill vocabulary projection runs after restoring the decode pool,
  instead of silently running single-threaded.
- The embedding-prefill API preserves its `const` input even at dense stride.
  Added missing SVE guards around fused IQ3 calls for non-SVE builds.
- Anonymous GGUF contexts initialize their unowned descriptor to `-1`;
  NUMA setup must not mistake descriptor zero for an open model file.
- Added optional per-projection tracing (`TF_PREFILL_TRACE`) and decode
  forward timing that excludes the first token supplied by prefill.

## Kernel validation

```sh
for test in test_iq3_tokenmajor test_mixed_tokenmajor; do
    TMPDIR="$PWD/tmp/iq3-validation" fccpx -Nclang -O3 \
        -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
        -ffunction-sections -fdata-sections -Wno-unused-function \
        a64fx/llm/$test.c -Wl,--gc-sections -lm -lpthread \
        -o tmp/iq3-validation/$test
done
# Execute these two commands on the compute node:
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
    tmp/iq3-validation/test_iq3_tokenmajor
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
    tmp/iq3-validation/test_mixed_tokenmajor \
    /local/u14346/qwen38-gsq/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf
```

IQ3: all 12 shape/thread/stride cases pass, including input mutation, zero
input, token tails and output padding. Error is at most about 1.8e-8 after
scaling by `max(1, sum(abs(reference products)))`, versus a 2e-6 tolerance.
The old sampled cache fails these cases (up to 0.00564 scaled error).
The reference independently dequantizes weights and uses double accumulation
with the same Q8 activation quantization; this does not claim equivalence to
unquantized FP32 activations.

For M=1024, K=5120, N=32, 48 threads: IQ3 batch 0.354–0.365 ms versus
0.787–0.793 ms for separate SVE dots (2.17–2.22x). These are small, warm-cache
kernel benchmarks, not end-to-end or HBM bandwidth measurements.

Mixed formats: real rows from all ten other 2D weight formats pass double
reference comparisons for N=1 matvec and N=3/9 prefill, changed and zero
inputs, padded strides, plus K=37 FP32 vector tails. The 13-row comparisons
have scaled errors below 4e-8. Representative 48-thread, N=96 prefill kernels
reach 685–794 effective GFLOP/s. IQ4_XS M=1024/K=5120 matvec is 0.082 ms.

The pool restart regression passes twice: 48 distinct pinned compute cores,
and caller affinity restored. A64FX cross builds are warning-free. A native
non-SVE compile also succeeds, with an existing unrelated BF16-PV implicit
declaration warning. `git diff --check` passes.

## End-to-end measurements

Before mixed-format dispatch changes, the first layer's QKV projection took
22.752 s, SSM gate 13.667 s, and FFN gate/up 38.683/38.729 s; the bounded
180-second run timed out before finishing layer zero.

| Configuration | Full 109-token prefill | Decode |
| --- | ---: | ---: |
| Mixed SVE prefill, before pool affinity fix | 22.26 s | Workers collapsed onto one core; stopped |
| Plus pool affinity fix | 20.40 s | 32 generated tokens / 36.52 s (0.88 gen tok/s) |
| Plus IQ4 SVE lookup | 15.49 s | 31 forwards / 33.389 s (0.93 decode tok/s) |
| Plus batched prefix and parallel prefill vocabulary head | 11.24 s (9.7 tok/s) | 95 forwards / 102.069 s (0.93 decode tok/s) |
| Same final binary, token-by-token prefill control | 116.55 s (0.94 tok/s) | 31 forwards / 33.403 s (0.93 decode tok/s) |

The final batched path improves full-prompt prefill by **10.37x** against the
same-binary token-loop control. Its first **32/32 generated token IDs match**
the control exactly. Logit values are not bit-identical; this is greedy-token
agreement for this prompt, not a general proof of model equivalence. Both
runs use the real image, same resolution, same weights and same greedy prompt.

The IQ4 change reduces the two prefill vocabulary projections from 5.908 s
to 1.176 s before the subsequent pool-restoration change. The final visual
LLM prefill segment takes 8.14 s for 96 embeddings (11.79 tok/s).
Restoring the pool before those vocabulary projections reduces their combined
time again, to 66.21 ms. Final output has 96 generated tokens; all 96 reported
logit vectors are finite and non-degenerate (`nan=0`, `inf=0`, `all_equal=no`).

The image output is coherent, beginning: “The image shows a majestic,
snow-capped mountain under a clear blue sky. The mountain has a symmetrical,
conical shape ...”. This is visual/prompt coherence, separate from numerical
kernel validation.

Residency during final decode: `HugetlbPages=12460032 kB` (~11.88 GiB),
ordinary RSS ~95 MiB, and `MemAvailable` ~18.0 GiB. `/proc/PID/io` read_bytes
remains 16,856,449,024 between generated steps 3 and 49: no storage reads
during that decode interval. On this system ordinary RSS alone omits the
anonymous hugepage allocation and is not a residency measure.

The measured 31-forward decode profile has 15.738 s FFN gate/up and 6.907 s
FFN down: about 68% of decode time. Mixed IQ2/IQ3_S expansion and projection
work remains the main optimization target, not model staging or mmproj.
The requested 60+ prefill / 30+ decode tok/s targets are **not reached**.

## Peak-relative efficiency (estimated work, not hardware counters)

Fujitsu specifies 6.144 TFLOP/s FP32, 24.576 TOP/s INT8, and 1024 GB/s HBM2
bandwidth at normal-mode 2.0 GHz. See the [Fugaku specifications](https://global.fujitsu/en-global/technology/research/fugaku/specifications)
and [A64FX datasheet](https://www.fujitsu.com/downloads/SUPER/a64fx/a64fx_datasheet_en.pdf).
BF16 storage here is expanded for FP32 arithmetic; the FP16 peak is not the
appropriate denominator for the vision encoder.

For P=384 patches, D=1152, F=4304, L=27 blocks, Q=96 merged embeddings,
merged width U=4608 and output width E=5120, dominant vision operations are:

```text
L * (8*P*D*D + 4*P*D*F + 4*P*P*D)  # block projections and attention
+ 2*P*D*768                         # folded patch projection
+ 2*Q*(U*U + U*E)                   # final projector
= 343.334 GFLOP
```

The complete encoder takes 0.106–0.113 s (850–903 merged embeddings/s),
or about 3.04–3.24 effective TFLOP/s, **49–53%** of FP32 peak. This excludes
one-time cache construction (~5.96 s) and model loading. Norms, activation
functions and padding overhead are not included in the operation numerator.
The final two-layer projector alone is 8.607 GFLOP; stage timing rounds its
elapsed time to 0.003 s (2.4% of the 0.106-second encode). The projector does
not perform autoregressive decode.

The warm-cache IQ3 batch microbenchmark is ~0.92–0.95 effective TOP/s, about
3.7–3.9% of INT8 peak. The mixed FP32 prefill microbenchmarks achieve about
11–13% of FP32 peak. These effective operations do not count quantization or
codebook-expansion instructions as useful GEMM work.

Active trunk/head 2D tensors total 9,673,400,320 packed bytes after excluding
input embeddings and the unused NextN block. At 0.9307 decode forwards/s,
that is **9.00 GB/s effective weight throughput**, or **0.88%** of peak HBM2
bandwidth. It is not a measured memory-controller rate: redundant traffic,
dequantization, cache reuse and state work are not represented. The 80%
target is not met; decode is far from exhausting HBM bandwidth.

Logs and exact bounded launch scripts for this allocation are retained in
`tmp/iq3-validation/`: `image_trace.log`, `image_affinity.log`, `image_iq4.log`,
`final.log`, `final_output.txt`, `reference.log`, and the two kernel-test logs.
