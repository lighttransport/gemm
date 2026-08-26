# K3 optimization findings: rejected approaches

This file records optimization attempts that should not be repeated without
new evidence. It complements `k3-resume.md`; the entries here are deliberately
organized by failure mode and include the live-run gate.

## Measurement contract

The decisive test is a real 12-rank layer-2 decode in boost-eco mode, using 47
OpenMP threads per rank, 256 generated tokens, row-aligned expert sharding,
BF16 communication, and the current async/sparse communication flags. A probe
is retained only when it passes the output validator and preserves the live
`hidden_hash` and generated IDs. Local kernel tests alone are insufficient.

Retained reference:

```text
decode_tok_s=1000.549618
hidden_hash=f70045a0852c325d
```

The target is below 0.8 ms/layer; the retained result is still above target.
Timing is noisy in boost-eco mode, so single-run gains smaller than the
observed spread are not evidence of improvement.

## Routed MXFP4 expert path

| Probe | Live result | Gate | Decision |
|---|---:|---|---|
| W1/W3 paired kernel with 16 live SVE accumulators | 936.23 tok/s | hash preserved | Rejected: register pressure/traffic outweighed shared activation work |
| Lower-pressure one-row W1/W3 pairing | 970.90 tok/s | hash preserved | Rejected: no gain |
| Paired four-row W1/W3 kernel, shared latent load | 553.05 tok/s | `f70045a0852c325d` | Rejected; removed |
| MXFP4 loop unroll 4 | 992.97 tok/s | hash preserved | Rejected; unroll 2/default retained |
| MXFP4 prefetch distance 0 | 936.23 tok/s | hash preserved | Rejected |
| MXFP4 prefetch distance 8 | 949.80 tok/s | hash preserved | Rejected |
| MXFP4 prefetch distance 32 | 985.50 tok/s | hash preserved | Rejected; distance 16 retained |
| Eight routed-down W2 software prefetch streams | 724.15 tok/s | hash preserved | Rejected; local 256-channel shard is too short |
| Routed-down block-major expert loop | 956.73 tok/s | hash changed | Rejected; expert-major order retained |
| Fused routed-down accumulator reduced from 8 rows to 4 | 978.15 tok/s | hash preserved | Rejected |
| Per-CMG compact MXFP4 arena | 1.129 ms/layer; expert 0.301 ms | hash preserved | Rejected; page packing did not improve locality |
| Cached NUMA page-owner W1/W3 scheduler | expert 0.298 ms | hash changed | Rejected; ownership model is invalid for these slices |
| Balanced owner scheduler | 1.175 ms/layer; expert 0.317 ms | hash changed | Rejected; static schedule retained |
| Replication cap 24 instead of 16 | 956.73 tok/s | hash preserved | Rejected; extra replicated traffic loses |
| W1/W3 pointer arrays hoisted per worker | 630.15 tok/s | hash changed | Rejected; per-task selection has ownership/state assumptions |
| Routed-down scalar bases/route weight hoisted | 587.77 tok/s | hash preserved | Rejected |
| Worker-to-CMG index cached in TLS | 704.69 tok/s | hash preserved | Rejected |
| Scale routed activation once after SiTU | decode neutral; prefill 799 -> 533 layer tok/s | hash changed | Rejected; per-load scaling retained |
| Prefill tile threshold 16 | M=1024 critical 5.90K tok/s | correctness passed | Rejected; slower than 8/4 |

For prefill, threshold 4 is retained only as a 1K-chunk recommendation: 5,972
critical-rank tok/s versus 5,863 for threshold 8, but it regresses M=64/256.

The MXFP4 e8m0 scale LUT is **not** a rejected change. It is retained: it is
bit-identical and removes the GPR-to-FPR scale conversion cost. The remaining
expert bottleneck is primarily the access pattern of many short streams, not
the scale conversion.

## BF16 projection path

| Probe | Live result | Gate | Decision |
|---|---:|---|---|
| `K3_BF16_ROWS=4` | 963.76 tok/s (earlier run) | hash preserved | Rejected; fewer streams did not offset extra calls |
| BF16 sparse-row direct A2A allgather | 970.90 tok/s | hash preserved | Rejected as neutral; async latent disabled |
| `TF_BF16PV_PREFETCH=1` | 560.14 tok/s | hash preserved | Rejected; isolated bandwidth gain does not survive short concurrent streams |
| `K3_BF16_PREFETCH=128…2048` | within noise | hash preserved | Rejected; hardware prefetch is sufficient |

An isolated BF16 benchmark showed `TF_BF16PV_PREFETCH=1` improving one
7168-column stream from 21.9 to 24.5 GB/s. The full decode result is the
authoritative result; do not infer a model-level gain from that microbenchmark.

## Communication and overlap

| Probe | Live result | Gate | Decision |
|---|---:|---|---|
| Per-token latent collective pthread | 364.09 tok/s | hash preserved | Rejected; startup/synchronization dominates |
| Persistent pinned latent worker | 492.75 tok/s | hash preserved | Rejected |
| OpenMP-team latent/expert overlap | 978.15 tok/s | hash preserved | Rejected; overlap is not profitable on this topology |
| Start shared-hidden reduction during routed expert work | 910.22 tok/s; 1.190 ms/layer | hash preserved | Rejected; second reduction costs more |
| Compact-trailer collective | deadlock/no output | unavailable | Rejected; fixed max-count trailer required |
| Corrected trailer clearing variant | no decode output | unavailable | Rejected |
| 12-rank sparse allgather, `--ar-groups 1` | 879.68 tok/s | hash changed | Rejected; 6+2 topology is required |
| `K3_COMM_A2A=1` | 1.178 ms/layer | hash preserved | Rejected |
| `K3_COMM_ROBUST=0/1` | 1.163/1.162 ms/layer | hash preserved | Neutral; default 2 retained |
| `K3_COMM_POLL_SPINS=16` | 1.162 ms/layer | hash preserved | Neutral; default 4 retained |
| `--ar-groups=3/4/6` | 1.263/1.313/1.162 ms/layer | valid where reported | Rejected; default 2 retained |
| `moe_shard_layout=replicated` | no layer-level change | hash preserved | Rejected at 12 nodes; saved collective time is consumed by extra streaming |

Retained after the table was written: deferred TCQ/MRQ completion, late shared
reduction, and the large-vector recursive-halving/allgather form.  The latter
is scoped by layer phase because MLA final-output reduction regressed while
attention and KDA final output improved.

The four collectives are close to the measured uTofu latency floor. Reducing
collective count alone is therefore not a reliable 12-node lever.

## KDA and general scheduling

| Probe | Result | Decision |
|---|---|---|
| Overlap three KDA state convolutions with independent projection | 897.75 tok/s | Rejected; extra team/row scheduling costs more |
| Fused decay plus `k·state` SVE pass | standalone 15.2 vs 20.8 us; live 985.50 tok/s | Rejected; no end-to-end gain |
| In-place replacement of KDA convolution copies | real SVE runner exited before output | Rejected; input/output scratch contract is required |
| Persistent OpenMP team per layer | +1.2%, within noise | Rejected |
| Fused KDA projection/conv/decay team | isolated front 0.137 -> 0.122 ms; endpoint 0.931 ms vs 0.919 ms control | Rejected; selector removed |
| Serial FEXPA gated RMSNorm (128 weights repeated/head) | 0.900 ms vs 0.889 ms control | Rejected; parallel scalar path retained |
| `K3_CMG_LOCAL=1` | 2.5% worse | Rejected; page split imbalance exceeds locality gain |
| First-touch NUMA placement | 3.505 vs 1.500 ms | Rejected; interleave is load-bearing |
| Q8_0 rows-in-lanes repack | 4.73x at one thread, 0.91x at 47 | Rejected |
| 8-row Q8_0 blocking | 20% worse | Rejected; remains opt-in only where explicitly requested |
| Serial fast path for one-thread quant matvec | 6% worse in two tests | Rejected |

## Do not confuse these with failures

The following are retained or useful baselines, not rejected experiments:

- e8m0 f32 LUT in the MXFP4 kernel: bit-identical and retained.
- MXFP4 prefetch distance 16: best matched result, retained.
- Sparse-row async latent path: retained as the current communication path.
- BF16 PV repacking: retained; it improves the projection kernel without
  changing the model output.
- BF16 transport and SiTU/FEXPA settings: quality/performance choices that
  require the existing model-output gate; they are not interchangeable with
  exact-f32 communication.

When revisiting a rejected idea, change the underlying evidence first (topology,
weight placement, compiler output, or measured phase) and record a new matched
run here rather than retrying the same knob.

## IQ1/Q2 quantized layer work (boost-eco, 2026-08-08)

The staged real-weight layer benchmark used `K3_QUANT_KERNEL=sve-q8`,
`OMP_NUM_THREADS=47`, and 12 ranks.  The layer bench is a projection/dequant
proxy, not a full generated-token run; boost-eco timing has substantial
run-to-run spread.

| Package | Earlier matched proxy | Later matched proxy | Decision |
|---|---:|---:|---|
| IQ1_S | 7.047 ms/layer (3 reps) | 7.374 ms/layer (10 reps, rank 0) | Keep only code changes that pass matched full-runner tests; no target claim |
| IQ2_XS | 8.217 ms/layer (3 reps) | 8.362 ms/layer (10 reps, rank 0) | Same; target not reached |

Retained implementation work:

- IQ1_S sixteen-row SVE-Q8 path now computes the shared `sum(x)` with SVE
  reductions instead of a scalar 32-element loop.  The local reference gate
  passes; the isolated real-layer measurements are too noisy to claim a fixed
  percentage gain.
- The full runner prepares one activation quantization workspace per unique
  `(input, columns, quant type, mode)` and reuses it across projection tasks.
  IQ1/IQ2 tensors use sixteen-row tasks when their row count permits it; other
  paths remain eight-row.  This removes repeated activation scans and avoids
  the per-task workspace allocation, but it was not independently measurable
  in the layer proxy.

Rejected probe:

- Replacing the IQ1/IQ2 second-half gather-index arithmetic with a shifted tile
  base and only four persistent index vectors passed all local correctness
  tests, but two direct real-weight runs measured about 9.1 ms/layer for both
  IQ1 and IQ2.  It is reverted.  Do not retry this exact four-index/base-shift
  shape without new compiler or hardware evidence.

Current hard limit: the measured IQ1/IQ2 layer proxy remains multiple
milliseconds, far above the requested 0.25/0.50 ms targets.  Reaching those
targets requires a different kernel/dataflow (for example a fused multi-
projection or packed weight layout), not another small gather/unroll tweak.

## Compact IQ decode cache and row-paired assembly (2026-08-26)

The different dataflow is now implemented and measured on one A64FX CMG with
12 cores.  The GGUF payload remains unchanged; a one-time, lossless decode
cache is built after loading weights:

- IQ1_S pairs output rows 0..15 and 16..31 in the low/high nibbles of each
  byte.  Sign extension produces two 16-row SDOT operands without ZIP/TBL.
- IQ2_XS stores a four-bit semantic index for the exact six-value grid
  `+/-{8,25,43}`.  Two SVE TBL instructions expand the paired rows in
  registers; no weight approximation is introduced.
- FP16 block scales are converted once to FP32, Q8 activation sums are
  prepared once, and FP32 accumulation occurs at every K=256 block.
- Assembly preloads the next 512-byte group before applying the current
  scales, covering SVE load latency without a second register bank.

Correctness (`k3_quant_kernel_test`) passes.  Pair32 versus the established
per-row Q8 path is `1.025e-7` relative L2 for IQ1_S and `8.906e-8` for
IQ2_XS.  Packed and unpacked paths have identical activation-quantization
quality.

Real layer-1 expert planes were staged from `~/models/k3/{iq1,q2}`.  The
measurement flattens the eight decode-selected expert planes into one OpenMP
dispatch and pins cores 12-23 to CMG 1:

```sh
OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores \
K3_QUANT_KERNEL=sve-q8 K3_QUANT_PACKED=1 K3_QUANT_PACKED_PAIR=1 \
K3_EXPERT_FLAT=1 K3_ACTIVE_EXPERTS=8 K3_ONLY_TENSOR=ffn_gate_exps \
numactl --physcpubind=12-23 --membind=4 ./k3_gguf_layer_bench \
  /local/u14346/k3-iq-decode-iq1/rank000.manifest \
  /local/u14346/k3-iq-decode-iq1/rank000.blob 501
```

| Real projection | IQ1_S GFLOP/s | IQ2_XS GFLOP/s |
|---|---:|---:|
| gate, 24576 x 3584 | 720.6 | 686.9 |
| up, 24576 x 3584 | 717.7 | 674.6 |
| down, 28672 x 3072 | 677.7 | 609.8 |

For gate, native GGUF decode measured 30.6/22.9 GFLOP/s and the older
byte-expanded packed cache measured 190.0/363.2 GFLOP/s (IQ1/IQ2).  Thus the
row-paired kernels are 23.5x/30.0x faster than native and 3.79x/1.89x faster
than the byte-packed cache.

FAPP reports 15.29 GB/s per core for IQ1 and 15.84 GB/s per core for IQ2
(about 183 and 190 GB/s per CMG) with IPC 1.31 and 1.38.  The kernels are
simultaneously close to the available memory stream and constrained by SVE
decode/SDOT issue; software separation of decode from consumers and explicit
prefetch both regressed.

The 800 GFLOP/s target is above the lossless roof at the previously measured
214 GB/s HBM rate once metadata is counted.  IQ1 streams 0.546875 bytes per
weight (nibbles, signed group scale, FP32 block scale), giving a 782.6
GFLOP/s roof before activation metadata and loop/dispatch cost.  IQ2's two
half-group scales make its exact roof lower.  At a true 230 GB/s stream the
corresponding IQ1 ideal is 841 GFLOP/s, but the measured 720.6 result already
uses about 92% of the 214 GB/s combined-stream roof.

### Lossless IQ1 two-bit row-quad cache

IQ1_S is ternary, so the signed-nibble cache still carried one redundant bit
per weight. A new 64-row cache packs four signed two-bit values into each
byte. The assembly kernel extracts the four fields with LSL/ASR, broadcasts
each Q8 activation once, and issues four vector SDOTs. Unlike the rejected
64-row composition of two pair32 tiles, it retains one sequential weight
stream and halves weight traffic from 0.5 to 0.25 byte/weight. The GGUF
payload and numerical format are unchanged.

`k3_quant_kernel_test` reports `1.028e-7` relative L2 against the established
IQ1 per-row Q8 path. On the same pinned 12-core CMG command above, replacing
`K3_QUANT_PACKED_PAIR=1` with `K3_QUANT_PACKED_QUAD2=1` gives:

| Real projection | Pair32 GFLOP/s | Quad64 two-bit GFLOP/s |
|---|---:|---:|
| gate, 24576 x 3584 | 720.6 | 980.4 |
| up, 24576 x 3584 | 717.7 | 977.9 |
| down, 28672 x 3072 | 677.7 | 972.7 |

The gate result is a 36.1% improvement and passes the 800 GFLOP/s target.
Three additional gate measurements were stable at roughly 975--983 GFLOP/s.

IQ2_XS cannot use the same representation because its exact grid has six
values. Three matched alternatives were rejected: vector rather than indexed
SDOT was about 3% slower, packing its scale side stream to nibbles added too
much decode work, and composing two pair32 weight streams damaged hardware
prefetch. The exact semantic-nibble pair32 IQ2 kernel therefore remains the
best measured path.
