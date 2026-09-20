# Qwen3.8 DSpark on A64FX

This directory is a standalone C11 implementation of the RadixArk
Qwen3.8-27B DSpark drafter.  It does not run or verify the 27B target.  A
caller supplies the accepted target model's post-layer features for layers
5, 19, 33, 47, and 61; DSpark returns seven greedy proposals, their selected
logits, and confidence-head probabilities.  Target verification consumes the
anchor plus those seven proposals (width eight) and remains authoritative.

## Numerical and state contract

- Draft and embedding weights are BF16 storage with FP32 activation and
  accumulation. Persistent K/V is BF16.
- The target LM head uses the exact stored NVFP4 E2M1 codes, E4M3 K=16
  scales, and `weight_scale_2` against FP32 activations (W4A16).
  `input_scale` is validated but intentionally unused.
- The seven noise rows are the anchor followed by six MASK tokens. They see
  the full accepted prefix and all seven current rows noncausally.
- Markov correction is sequential: row zero is conditioned on the anchor and
  every later row is conditioned on the preceding proposal.
- `dspark_state_propose` does not change persistent state. Append only the
  target features for tokens accepted by the verifier. `truncate` and `reset`
  are available for transactional integrations.

The public API is in `dspark.h`. Model weights are immutable and may be shared
by multiple states; calls operating on one state must be serialized.

## Build and test

Portable scalar build:

```sh
make -C a64fx/dspark clean test CC=cc CFLAGS='-O3 -std=c11'
```

A64FX SVE1 build:

```sh
make -C a64fx/dspark clean all CC=fcc \
  CFLAGS='-Nclang -O3 -std=c11 -march=armv8.2-a+sve -ffp-contract=fast'
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/dspark/test_dspark
```

The Makefile puts compiler temporaries in `a64fx/dspark/tmp`; it never uses
`/tmp`.

## Validate the downloaded checkpoints

Header validation reads no tensor payloads:

```sh
DRAFT=$HOME/models/qwen38/radixark/Qwen3.8-27B-DSpark
TARGET=$HOME/models/qwen38/radixark/Qwen3.8-27B-NVFP4
./a64fx/dspark/validate_dspark --headers "$DRAFT" "$TARGET"
```

The full validator loads resident weights, appends two deterministic target
feature rows, and executes one complete proposal:

```sh
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/dspark/validate_dspark --full "$DRAFT" "$TARGET"
```

The default state capacity is 8192 tokens (about 160 MiB of BF16 K/V). A
caller may explicitly request up to 262144 tokens (about 5 GiB). Resident
draft, embedding, and panel-only LM-head weights occupy about 6.7 GiB.

On the shared model filesystem, initial resident loading is I/O-bound and may
take many minutes. The loader uses parallel `pread`, anonymous first-touch
placement, and `POSIX_FADV_DONTNEED`; copy or stage only through `/local`, not
`/tmp`, when repeated load benchmarks are required.

## Independent golden

`export_golden.py` uses the checkpoint-shipped PyTorch implementation and the
official SGLang semantics pinned at revision
`3a64faa1f22a86abd37a759c84267d929e820d5b`. Run it on a machine with PyTorch,
Transformers 5.8.1, and safetensors:

```sh
python3 a64fx/dspark/export_golden.py "$DRAFT" "$TARGET" \
  a64fx/dspark/tmp/qwen38_golden.json
./a64fx/dspark/validate_dspark --golden \
  a64fx/dspark/tmp/qwen38_golden.json --full "$DRAFT" "$TARGET"
```

The fixture uses deterministic FP32 target features, the real checkpoint
weights, and the same W4A16 LM-head contract. Token IDs must match exactly;
confidence uses 2e-3 absolute tolerance and selected logits use 2e-3 relative
tolerance.

## Benchmark

```sh
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/dspark/bench_dspark "$DRAFT" "$TARGET"
```

Set `DSPARK_PROFILE=1` on a validator or benchmark run to print opt-in phase
timings for each proposal. This is a diagnostic flag only; it does not select
a production implementation:

```sh
DSPARK_PROFILE=1 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/dspark/validate_dspark --full "$DRAFT" "$TARGET"
```

The output separates embedding, normalization, QKV, attention, output
projection, FFN, NVFP4 LM-head, and sequential Markov/confidence costs.

The dominant SVE kernels are width-1-through-8 BF16 GEMM, GQA attention, and
a width-seven NVFP4 output-panel kernel. The width-seven paths stream each
weight matrix once rather than decomposing the proposal into 4+2+1 passes.

The validated two-chain FP32 accumulation path measured on the current A64FX
node with one accepted context row as follows:

| Threads | SVE proposal | Scalar proposal | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 1746.9 ms | 96160.3 ms | 55.05x |
| 12 | 204.2 ms | 8083.5 ms | 39.59x |
| 48 | 69.4 ms | 2074.2 ms | 29.87x |

All seven token IDs matched. Maximum confidence difference was `1.24037e-4`
and maximum selected-logit relative difference was `4.05867e-4`. The
benchmark treats `2e-4` confidence error and `5e-4` selected-logit relative
error as failures. A production-length synthetic BF16 GEMM additionally
measured `1.45e-6` relative L2 and `2.30e-6` scaled maximum error.
