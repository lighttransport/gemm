# Qwen3.8-27B NVFP4 on one A64FX node

Job `51865934`, node `d01-0210s`, 48 workers on CPUs 12–59, four CMG-local
anonymous HBM arenas, 2 MiB pages. Model:
`/local/u14346/Qwen3.8-27B-NVFP4-Quality-v2.gguf` (16,058,504,512 bytes).
The model has 866 tensors; the dense projections are GGUF NVFP4 (type 40).

## Bottleneck and changes

The initial exact four-row SVE path decoded at 1.04 tok/s on this node.
`TF_NULL_GEMM=1` streamed 14.754 GB/token at 243.4 GB/s and reached
16.497 tok/s, pointing to decode arithmetic rather than memory capacity.
Disassembly showed four calls to `ldexpf` per four-row FP4 subblock. A bounded
sample of 24,313,856 UE4M3 scale bytes found 24,305,496 with exponent zero.
The normal-only bit conversion experiment had missed that dominant case.

`tf_nvfp4_scale_fast` now converts all 256 scale codes without libm. The
first implementation used seven subnormal constants and IEEE bit construction
for normal codes; an exhaustive 256-code float-bit comparison against the
reference conversion passed. The exact four-row full-model result rose to
2.838 tok/s over 128 tokens. A later 1 KiB table of those exact values removed
the remaining scalar branches and bit construction from the decode loop.

The persistent decode path now groups eight rows per call, sharing activation
loads while keeping each row's original low-nibble/high-nibble FP32 FMA order.
Prefetching each of those eight row streams eight 64-value blocks ahead
reduced the full decode from 35.388 to 29.624 seconds under identical
non-profiled settings. Four-block prefetch measured 29.649 seconds, within
run-to-run noise of the eight-block result. Sixteen-block prefetch
regressed to 30.926 seconds / 4.139 tok/s and was discarded.
The shared diagnostic counter is accumulated locally and updated atomically
once per matrix slice. The opt-in W4A8 path requantizes each input rather than
reusing a scratch address and four sampled values across tokens.

## Measurements and validation

| Path | 128-token decode | Token IDs | Selected-logit error |
| --- | ---: | ---: | ---: |
| Exact four-row reference with fast scales | 2.838 tok/s (45.110 s) | reference | — |
| Exact eight-row, original FMA order | **3.617 tok/s (35.391 s)** | **128/128** | **0 bitwise** |
| Exact eight-row, eight-block prefetch | **4.321 tok/s (29.624 s)** | **128/128** | **0 bitwise** |
| Exact eight-row, prefetch and scale table | **4.464 tok/s (28.674 s)** | **128/128** | **0 bitwise** |
| Combined runner, exact default | **4.698 tok/s (27.243 s)** | **128/128** | **0 bitwise** |
| Combined runner, `--nvfp4-fast` | **4.858 tok/s (26.350 s)** | **128/128** | **0.000395 max** |

The first two runs used prompt `hi`, `--max-seq 256 --max-gen 128 --spec-k 0`,
48 threads, `TF_DUMP_TOKENS=1 TF_DPROF=1`, the same staged GGUF, and the
four-CMG environment in `run_qwen38_nvfp4_cmg4.sh`. The prefetch comparison
used the same inputs with `TF_DUMP_TOKENS=1` and profiling disabled in both
cases: saved baseline 35.388 s / 3.617 tok/s, four-block prefetch 29.649 s /
4.317 tok/s, eight-block prefetch 29.624 s / 4.321 tok/s. The eight-block
token comparison was:

```sh
python3 a64fx/llm/test_qwen38_token_trace.py \
  /local/u14346/q27b-ref128.log /local/u14346/q27b-prefetchdist8-128.log \
  --tokens 128 --max-logit-error 0
# PASS: tokens=128/128, selected_logit_max_abs=0, rms=0
```

The scale-table run used the same 128-token settings without profiling and
repeated at 28.676 and 28.674 seconds, both with 128/128 token IDs and zero
selected-logit error against the exact reference.

`--nvfp4-fast` selects a full-width nibble-interleaved SVE reduction for the
eight-row decode path. It changes FP32 accumulation order; the exact kernel
remains the default. The combined runner matched all 128 token IDs on `hi`
within the strict 0.001 selected-logit tolerance (maximum 0.000395). On a
second prompt, `Explain why a compass needle points north even though Earth
is not a perfect bar magnet.`, a fast-only build reached 4.987 tok/s versus
4.461 exact, with 128/128 IDs matching and maximum selected-logit error
0.000280. The fast-only build reached 4.995 tok/s on `hi`; the combined
opt-in runner is somewhat slower, so its 4.858 tok/s result is the usable
headline. A third prompt requesting Python binary-search-tree code was run for
256 generated tokens with the combined runner: exact 4.676 tok/s, fast
4.836 tok/s, **256/256 IDs matched**, and selected-logit maximum error
0.000870. Across these three prompts 512/512 generated IDs matched. This does
not prove token identity for every prompt or a longer context.

The original eight-row stage costs were approximately 123.6 ms/token for FFN
gate/up, 49.1 for FFN down, 44.2 for SSM input, 15.0 for SSM output, and
20.5 for the vocabulary head. The kernel improvement is still far from the
40 tok/s goal. The 128-token result establishes this prompt and short context;
it does not establish longer-context or other-prompt performance.

An earlier full-width build without the later lookup/prefetch changes reached
3.962 tok/s; 128/128 token IDs matched but selected logits differed by up to
0.00256, above the repository's 0.001 validation tolerance. That build was
discarded; the current opt-in build passes the strict tolerance on the two
prompts measured above.

The opt-in row-major W4A8 path kept all 128 token IDs on this prompt, but
ran at 3.085 tok/s (41.493 s) and changed selected logits by up to 0.13878.
Its activation scratch now always requantizes the current input, so there is
no cross-token reuse through a sampled-address cache. The row-major W4A8
path is not a performance improvement.

`bench_nvfp4_packed.c` explores a K-major eight-row W4A8 layout with SVE
`TBL` + `SDOT`. It predecodes the eight UE4M3 scales per subblock to FP32;
the packed bytes are 4/3 the original GGUF weight bytes. The benchmark now
reserves one expanded arena and packs it **backward in place** with only a
30 KiB tile scratch, avoiding a second full weight allocation. Before the
serial packing pass, 48 workers first-touch the 2 MiB pages so the later
static decode partition reads CMG-local HBM. Without that placement, packed
read throughput collapsed to 77.8 GB/s original-byte equivalent; with it,
the final first/last-tile-checked trial streamed 0.503 GB packed in 0.973 ms:
517.2 GB/s physical or 387.9 GB/s original-byte equivalent. The synthetic
131,072-row by 5,120-column matrix packed in 1.091 seconds; first and last
eight-row tiles had relative L2 error 0.001628 against a scalar FP32 oracle.
FP16 predecoded scales reduced storage but measured 375.6 GB/s original-byte
equivalent. This is a synthetic kernel measurement, not end-to-end model speed
or token validation. At 14.754 GB of weights per token, 387.9 GB/s would
bound a perfect packed model at about 26.3 tok/s before non-matvec work;
reaching 40 tok/s still requires a faster format/kernel and full-model
integration of the in-place layout.

A metadata-only scan of the staged GGUF found 371 eligible NVFP4 tensors:
13.740 GB original bytes become 18.320 GB packed. Including the remaining
tensors, the expanded data arena would be approximately **20.627 GB** versus
16.048 GB today, within the node's 32 GB HBM before runtime buffers. The
largest original eligible tensor is 50.1 MB. Full-model integration still
needs reverse-order movement across GGUF tensors and remapping every loaded
`qtensor.data` pointer; this scan is a capacity calculation, not a successful
model repack. The following experiment completed that integration on the
replacement job described below.

## Packed full-model decode (job 51869188)

On node `l01-3209c`, `--nvfp4-packed` reserves the expanded anonymous GGUF
arena, first-touches it across four CMGs, repacks the 371 NVFP4 tensors in
reverse offset order with one tile of scratch, and updates loaded tensor
pointers. The pack took 7.88 seconds; the 20.627 GB arena left approximately
10 GB `MemAvailable`. The SVE kernel uses `TBL` to decode FP4 nibbles and
`SDOT` for eight output rows. It quantizes activations in groups of four to
keep the tested token traces identical to exact decode. The old 16-value
groups diverged on the compass prompt at token 7. Eight-value groups diverged
at token 17 and offered no speed benefit, so both were discarded.

| Prompt | Exact | Packed | Token IDs | Max selected-logit error |
| --- | ---: | ---: | ---: | ---: |
| `hi`, 128 generated | 4.654 tok/s | 8.454 tok/s | 128/128 | 0.061661 |
| Compass explanation, 128 generated | ~4.65 tok/s | 8.446 tok/s | 128/128 | 0.032745 |
| C Fibonacci function, 256 generated | 4.668 tok/s | 8.332 tok/s | 256/256 | 0.146685 |

The three completed traces match **512/512 generated token IDs**. This is
bounded validation, not a guarantee for all prompts. The packed path remains
opt-in because activation quantization changes logits. It is approximately
1.8 times the exact decode rate, still far below 40 tok/s. On the compass
run, FFN gate/up and down take 29.8 and 30.1 ms/token; the Q6_K vocabulary
head alone takes 20.6 ms/token. The head cost caps throughput below 49 tok/s
even if all other stages were free. Further progress requires a substantially
faster packed projection kernel and head.

Validation used the existing `test_qwen38_token_trace.py` with exact and
packed logs, `--tokens 128` or `256`, and `--max-logit-error 0.2`. The longer
code-prompt logs are `/local/u14346/q27b-exact-code256.log` and
`/local/u14346/q27b-packed-per4-code256.log`. Reproduce with the same command
below and append `--nvfp4-packed`; this option requires 48 NUMA workers,
anonymous weights, a dense NVFP4 model, and `--spec-k 0`.

Vectorizing the four-value activation conversion while retaining the same
scalar group maxima and scales reduced the profiled `hi` run from 118.3 to
110.7 ms/token (8.45 to 9.03 tok/s). The non-profiled compass and code runs
reached 8.980 and 8.891 tok/s. Against the earlier scalar four-value packed
logs, all 512 token IDs and selected logits matched **bitwise**. The code
prompt remained 256/256 against exact decode with 0.146685 maximum selected-
logit error. The vectorized conversion is the retained packed implementation.

## Reproduce

```sh
make -C a64fx/llm qwen38_runner CC=fcc OPENMP=1
TF_DPROF=1 TF_DUMP_TOKENS=1 \
  a64fx/llm/run_qwen38_nvfp4_cmg4.sh \
  /local/u14346/Qwen3.8-27B-NVFP4-Quality-v2.gguf \
  --prompt hi --max-seq 256 --max-gen 128 --spec-k 0
```

Append `--nvfp4-fast` to opt into the faster reduction. The benchmark numbers
above omit `TF_DPROF`; setting it prints stage costs but changes timing.

The build used Fujitsu `fcc -Nclang -O3 -march=armv8.2-a+sve -Kfast`.
Known TLS debug-relocation linker warnings remain.
