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

`tf_nvfp4_scale_fast` now converts all 256 scale codes without libm: seven
subnormal constants, a zero/sentinel case, and an IEEE exponent/mantissa
construction for normal codes. An exhaustive 256-code float-bit comparison
against the reference conversion passed. The exact four-row full-model
result rose to 2.838 tok/s over 128 tokens.

The persistent decode path now groups eight rows per call, sharing activation
loads while keeping each row's original low-nibble/high-nibble FP32 FMA order.
The shared diagnostic counter is accumulated locally and updated atomically
once per matrix slice. The opt-in W4A8 path requantizes each input rather than
reusing a scratch address and four sampled values across tokens.

## Measurements and validation

| Path | 128-token decode | Token IDs | Selected-logit error |
| --- | ---: | ---: | ---: |
| Exact four-row reference with fast scales | 2.838 tok/s (45.110 s) | reference | — |
| Exact eight-row, original FMA order | **3.617 tok/s (35.391 s)** | **128/128** | **0 bitwise** |

Both runs used prompt `hi`, `--max-seq 256 --max-gen 128 --spec-k 0`, 48
threads, `TF_DUMP_TOKENS=1 TF_DPROF=1`, the same staged GGUF, and the
four-CMG environment in `run_qwen38_nvfp4_cmg4.sh`. The token comparison was:

```sh
python3 a64fx/llm/test_qwen38_token_trace.py \
  /local/u14346/q27b-ref128.log /local/u14346/q27b-8row-ordered128.log \
  --tokens 128 --max-logit-error 0
# PASS: tokens=128/128, selected_logit_max_abs=0, rms=0
```

The final eight-row stage costs are approximately 123.6 ms/token for FFN
gate/up, 49.1 for FFN down, 44.2 for SSM input, 15.0 for SSM output, and
20.5 for the vocabulary head. The kernel improvement is still far from the
40 tok/s goal. The 128-token result establishes this prompt and short context;
it does not establish longer-context or other-prompt performance.

A full-width nibble-interleaved variant reached 3.962 tok/s but changed the
FP32 reduction order; 128/128 token IDs matched while selected logits differed
by up to 0.00256, above the repository's 0.001 validation tolerance. It was
not retained.

## Reproduce

```sh
make -C a64fx/llm qwen38_runner CC=fcc OPENMP=1
TF_DPROF=1 TF_DUMP_TOKENS=1 \
  a64fx/llm/run_qwen38_nvfp4_cmg4.sh \
  /local/u14346/Qwen3.8-27B-NVFP4-Quality-v2.gguf \
  --prompt hi --max-seq 256 --max-gen 128 --spec-k 0
```

The build used Fujitsu `fcc -Nclang -O3 -march=armv8.2-a+sve -Kfast`.
Known TLS debug-relocation linker warnings remain.
