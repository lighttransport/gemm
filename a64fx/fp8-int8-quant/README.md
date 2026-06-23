# GLM-5.2 FP8 to INT8 Quantization Experiment

Purpose: evaluate whether A64FX SVE 1.0 `sdot` int8 kernels can accurately model GLM-5.2 FP8 dense weights. The current host is x64 plus Tesla V100, so this directory contains host-side quantization/error analysis and a CUDA DP4A int8 throughput proxy. The A64FX code is kept as a target-side skeleton for later build/run on real hardware.

The GLM-5.2 FP8 checkpoint stores dense weights as `F8_E4M3` plus `*.weight_scale_inv` FP32 block scales. The model config says the FP8 block size is `[128,128]`; the analyzer mirrors that layout.

The analyzer also supports the original BF16 checkpoint at `~/models/glm5.2`; BF16 weights are widened exactly with `uint16 << 16`.

## Recommendation

Use `int8 AWQ block128` as the first A64FX `sdot` candidate. It keeps the 128x128 block-scale layout needed for efficient tiled int8 kernels and, in the sampled GLM/M3 reports, roughly halves relative L2 versus plain block int8. Use `SmoothQuant block128` as the simpler fallback when AWQ clipping/saturation is undesirable.

There is no proven universal tensor threshold for quantize-or-not decisions. Treat the following as engineering gates until end-to-end logit or perplexity validation is available:

- Green for int8: synthetic output rel L2 below `0.02`, cosine above `0.9998`, saturation below `0.5%`, and no large max-error outliers.
- Yellow for int8: output rel L2 `0.02-0.04` or saturation `0.5-1.0%`; require end-to-end validation or a per-tensor exception.
- Red for int8: output rel L2 above `0.04`, saturation above `1.0%`, SQNR below about `30 dB`, or sensitive tensors such as embeddings, lm_head, router/gating, norms, and first/last layers.
- For quick batch screening before matmul tests: `i8_awq` weight rel L2 below `0.005` is a candidate, `0.005-0.01` needs inspection, and above `0.01` should stay fp16/bf16 or move to int16 unless full-model quality passes.

Reserve int16 for tensors that fail int8 quality gates. It is nearly lossless in the samples, but doubles bandwidth and does not use A64FX int8 `sdot`. Do not prioritize SVDQuant for the first A64FX kernel: rank-4 residual compensation did not beat Smooth/AWQ and adds residual storage plus extra compute.

Do not use plain int4/fp4 block quantization for GLM/M3 dense weights as a first target. In the current samples, FP4 E2M1-style block quantization is better than symmetric int4, but both are far outside the int8 error gates. TurboQuant is more relevant for KV-cache compression than static weights; evaluate it with attention-score and attention-output drift rather than weight reconstruction metrics.

## Build

```sh
make -C a64fx/fp8-int8-quant
make -C a64fx/fp8-int8-quant cuda   # optional, needs nvcc
```

No Python packages are required.

## Analyze One Tensor

```sh
./a64fx/fp8-int8-quant/quant_analyze \
  --model ~/models/glm52-fp8 \
  --tensor model.layers.0.self_attn.q_a_proj.weight \
  --rows 512 \
  --cols 4096 \
  --svd-rank 4
```

## Sweep Representative Tensors

```sh
ROWS=256 COLS=2048 SVD_RANK=4 \
  ./a64fx/fp8-int8-quant/scripts/sweep_glm52_representative.sh
```

The sweep covers layer-0 dense attention/MLP tensors and layer-3 expert-0 MLP tensors, writes a CSV, and prints a per-tensor best-method summary.

## Batch Weight Reports

```sh
./a64fx/fp8-int8-quant/scripts/batch_quant_report.py \
  --model ~/models/glm5.2 \
  --out a64fx/fp8-int8-quant/report_glm52_bf16_all_weights_sample.csv \
  --rows 4 \
  --cols 64 \
  --max-elements-per-tensor 256 \
  --methods i8_block128,i8_awq,i16_block128,fp16
```

This script uses only Python stdlib. It parses safetensors headers, mmap-reads each shard, samples each 2D `.weight` tensor without writing temporary weight files, and emits CSV features plus quantization error columns. Remove `--max-tensors` for an unattended full pass; keep small row/column caps for large MoE checkpoints.

Useful knobs:

- `--scheme tensor|row|block|block_mse|block_p99|int4|fp4|smooth|awq|svd|i16|i16_row|i16_block|i16_smooth|i16_awq|i16_svd|fp16|all`
- `--block 128` for the intended A64FX int8 block scale granularity
- `--smooth-alpha 0.5`
- `--act-stat path.txt` for SmoothQuant activation maxima, one float per input column. If omitted, a weight-only proxy is used.
- `--rows N` and `--cols N` cap the sampled submatrix to keep analysis fast.
- `--x-rows N` also reports synthetic activation matmul error for `X * W^T`, with per-token int8 activation quantization.
- `--fp16-chunks 16,32,64` sets K chunk sizes for the fp16 widened-to-fp32 matmul proxy.
- `--csv out.csv` appends machine-readable metrics.
- `--dump-int8 prefix` writes sampled row-major int8 weights for kernel experiments.

## Parameter Search

`scripts/search_quant_configs.py` performs a bounded streaming search over quantization settings without writing temporary tensors:

```sh
./a64fx/fp8-int8-quant/scripts/search_quant_configs.py \
  --model ~/models/glm52-fp8 \
  --out a64fx/fp8-int8-quant/search_glm52_fp8_allconfigs_100.csv \
  --rows 32 \
  --cols 256 \
  --max-elements-per-tensor 8192 \
  --max-tensors 100 \
  --topk 0 \
  --threads 1
```

The search covers symmetric tensor/row/block scaling for 3/4/5/6/8/16 bits, clipped block scaling, column-scaled block variants, FP4 E2M1, NF4, and FP16. Output includes error metrics plus simple implementation-cost features: bytes per weight, scale count, estimated dequant ops per value, and the likely A64FX path.

Run one model search at a time. Earlier parallel launches mmaped multiple checkpoints and were stopped by OOM; the current script keeps only one sampled tensor per worker and either all 229 config rows or a bounded top-k heap in memory.

The raw search CSV includes `quant_ms` for each tensor/config. `search_summary.csv` aggregates this into mean/median/p95 processing time columns. These are host Python fitting/simulation times, not A64FX kernel timings.

CPU worker parallelism is controlled by `--threads`. The default is half of `os.cpu_count()`; on this host that is 36 workers. Use `--threads 1` for sequential timing or a smaller explicit value when memory pressure matters. Model-level wall-clock summaries are written with `--summary-json` and can be aggregated into `search_model_timing.csv`.

## CUDA Proxy

```sh
./a64fx/fp8-int8-quant/int8_dp4a_bench 4096 4096 4096 50
./a64fx/fp8-int8-quant/int16_cuda_bench 1024 1024 2048 20
```

This is not an A64FX performance predictor. It checks the packed-int8 data path and gives a V100 DP4A baseline while the actual target is unavailable.

## Quantization Modes

- `block`: symmetric int8 per 128x128 weight block, matching GLM FP8 block geometry and the likely A64FX scale layout.
- `block_mse`: symmetric 128x128 block int8 with per-block clipping selected by a small MSE search.
- `block_p99`: symmetric 128x128 block int8 clipped to the per-block 99.9th percentile absolute value.
- `int4`: symmetric signed int4 per 128x128 weight block.
- `fp4`: FP4 E2M1-style signed codebook per 128x128 weight block.
- `smooth`: SmoothQuant-style column scale migration. It quantizes `W * s_col` and reconstructs `q / s_col`; activation max statistics can be supplied externally.
- `awq`: AWQ-style column equalization. It grid-searches several activation/weight scaling exponents, then uses MSE-clipped block quantization.
- `svd`: SVDQuant-style low-rank residual compensation on top of block int8. The implementation uses power iteration on the sampled residual matrix so it stays dependency-free.
- `i16`, `i16_row`, `i16_block`, `i16_smooth`, `i16_awq`, `i16_svd`: signed symmetric int16 variants. `--x-rows` uses int16 activations for these schemes.
- `fp16`: IEEE fp16 storage. With `--x-rows`, synthetic activations are also stored as fp16; each K chunk is widened to fp32 and accumulated with fp32 FMA.

Primary metrics are RMSE, MAE, max absolute error, relative L2, cosine similarity, SQNR dB, and saturation percentage.

With `--x-rows`, each `scheme+x` line compares float `X * W^T` against dequantized int8-activation `X` times the quantized weight approximation. The synthetic activations are deterministic uniform values with sparse outliers; this is a kernel/error proxy, not a substitute for calibration data.

## A64FX Direction

The intended target kernel contract is:

```text
Y = X_int8 dot W_int8^T
dequant scale = activation_scale[token/block] * weight_scale[row_block,col_block]
accumulate int32, convert to f32/bf16 after scaling
```

On A64FX, use SVE `sdot` with row/column blocking aligned to 128 columns where possible. Keep the quantization metadata generated here byte-simple: raw int8 rows plus FP32 block scales.

For int16, A64FX does not use the same int8 `sdot` instruction. The plausible path is SVE halfword multiply-accumulate into int32, for example `smlalb/smlalt`-style kernels. This is expected to trade throughput and bandwidth for much higher fidelity.

For fp16, A64FX does not have a dot-product instruction equivalent to int8 `sdot`; model it as fp16 load, widen chunks to fp32, and use fp32 FMA/reduction.

## KV Cache Quantization

`scripts/kv_cache_quant.py` is a synthetic attention proxy for KV-cache compression. It compares scalar `int4`, scalar `fp4`, and TurboQuant-style randomized-rotation scalar quantization:

```sh
./a64fx/fp8-int8-quant/scripts/kv_cache_quant.py \
  --seq 2048 \
  --dim 128 \
  --out a64fx/fp8-int8-quant/kv_cache_quant_2048x128.csv
```

The TurboQuant proxy uses a random sign transform plus FWHT rotation before scalar quantization, then inverse rotation. It is an offline measurement proxy for MSE-style TurboQuant behavior, not a production KV-cache kernel and not the full QJL residual-correction path.
