# Initial GLM-5.2 FP8 -> INT8 Results

Host: x64 analysis binary plus Tesla V100 CUDA proxy.

Command:

```sh
./a64fx/fp8-int8-quant/quant_analyze \
  --model ~/models/glm52-fp8 \
  --tensor model.layers.0.self_attn.q_a_proj.weight \
  --scheme all \
  --rows 512 \
  --cols 4096 \
  --block 128 \
  --svd-rank 4
```

Tensor:

```text
model.layers.0.self_attn.q_a_proj.weight
full shape: 2048 x 6144
sample: 512 x 4096
FP8 scale tensor: model.layers.0.self_attn.q_a_proj.weight_scale_inv
```

Metrics:

| scheme | RMSE | MAE | max abs | rel L2 | cosine | SQNR dB | sat |
|---|---:|---:|---:|---:|---:|---:|---:|
| tensor | 3.1753809e-4 | 2.7472574e-4 | 5.4976251e-4 | 5.8657291e-2 | 0.9982845186 | 24.634 | 0.000% |
| row | 6.9683315e-5 | 4.5321403e-5 | 5.4976251e-4 | 1.2872265e-2 | 0.9999171704 | 37.807 | 0.025% |
| block 128x128 | 1.3755998e-4 | 1.1399071e-4 | 5.4550171e-4 | 2.5410796e-2 | 0.9996772739 | 31.900 | 0.008% |
| smooth, weight proxy | 9.4852527e-5 | 8.0066385e-5 | 5.4550171e-4 | 1.7521653e-2 | 0.9998465863 | 35.128 | 0.008% |
| svd rank 4 residual | 1.3642622e-4 | 1.1293268e-4 | 7.2053866e-4 | 2.5201363e-2 | 0.9996825726 | 31.972 | 0.008% |

Takeaway for this slice: per-row int8 is the most accurate simple mode, but 128x128 block int8 is the better A64FX-friendly baseline because it matches the FP8 block geometry. SmoothQuant-style column migration improves the block baseline substantially even without real activation maxima. Rank-4 residual compensation is only a small relative-L2 improvement here and increases the worst element error, so it needs broader tensor coverage before it is worth carrying into the SVE kernel design.

## CUDA DP4A Proxy

Toolkit: `/usr/local/cuda/bin/nvcc`

Visible GPUs:

```text
2 x Tesla V100-PCIE-32GB
Driver 580.126.09, CUDA 13.0 runtime visible through nvidia-smi
```

Commands:

```sh
make -C a64fx/fp8-int8-quant cuda
./a64fx/fp8-int8-quant/int8_dp4a_bench 1024 1024 2048 20
./a64fx/fp8-int8-quant/int8_dp4a_bench 2048 2048 4096 20
```

Results:

```text
M=1024 N=1024 K=2048 iters=20 time=63.589 ms int8_ops=1.351 TOPS
M=2048 N=2048 K=4096 iters=20 time=451.216 ms int8_ops=1.523 TOPS
```

This CUDA kernel is intentionally simple: one output element per thread using `__dp4a`. It validates the int8 path and gives a local V100 baseline, not a tuned GPU GEMM ceiling.

Simple int16 CUDA proxy:

```sh
./a64fx/fp8-int8-quant/int16_cuda_bench 1024 1024 2048 20
```

```text
M=1024 N=1024 K=2048 iters=20 time=235.456 ms int16_ops=0.365 TOPS
```

This is a naive scalar int16 CUDA kernel, included only as a local sanity proxy.

## Representative Tensor Sweep

Command:

```sh
ROWS=256 COLS=2048 SVD_RANK=4 \
  ./a64fx/fp8-int8-quant/scripts/sweep_glm52_representative.sh \
  ~/models/glm52-fp8 \
  a64fx/fp8-int8-quant/glm52_representative_sweep.csv
```

Summary by relative L2 error:

| tensor | best | best rel L2 | block rel L2 | smooth rel L2 | row rel L2 | svd rel L2 |
|---|---:|---:|---:|---:|---:|---:|
| layer0 mlp down | smooth | 0.008888 | 0.012110 | 0.008888 | 0.009041 | 0.011947 |
| layer0 mlp gate | row | 0.008404 | 0.012890 | 0.010186 | 0.008404 | 0.012720 |
| layer0 mlp up | row | 0.008354 | 0.011291 | 0.008980 | 0.008354 | 0.011135 |
| layer0 kv_a | row | 0.010177 | 0.025231 | 0.013449 | 0.010177 | 0.024713 |
| layer0 kv_b | row | 0.008380 | 0.016947 | 0.010288 | 0.008380 | 0.016523 |
| layer0 o_proj | smooth | 0.008729 | 0.011861 | 0.008729 | 0.014096 | 0.011664 |
| layer0 q_a | row | 0.010621 | 0.023915 | 0.016169 | 0.010621 | 0.023552 |
| layer0 q_b | smooth | 0.013691 | 0.026539 | 0.013691 | 0.018749 | 0.024477 |
| layer3 expert0 down | smooth | 0.008708 | 0.013305 | 0.008708 | 0.011299 | 0.013131 |
| layer3 expert0 gate | row | 0.010102 | 0.018496 | 0.012645 | 0.010102 | 0.018202 |
| layer3 expert0 up | row | 0.010061 | 0.019554 | 0.013296 | 0.010061 | 0.019242 |

Observations:

- Per-row int8 is the best simple quantizer for 7/11 sampled tensors, with rel L2 around 0.0084-0.0106 on most dense/expert projections.
- SmoothQuant-style block int8 is best for 4/11 tensors and improves the 128x128 block baseline on every sampled tensor. It cuts block rel L2 by about 20-48%, with the biggest gains on q/kv projections.
- Rank-4 SVD residual correction only slightly improves block rel L2 in this sample and often increases max absolute error. It is not yet compelling for the A64FX kernel path unless higher rank or activation-aware residual targeting changes the tradeoff.
- For A64FX SVE `sdot`, the practical next target is `smooth + block128`: it keeps scale metadata coarse enough for efficient tiled SDOT while recovering much of the row-scale accuracy.

## Synthetic Matmul Error

Command:

```sh
./a64fx/fp8-int8-quant/quant_analyze \
  --model ~/models/glm52-fp8 \
  --tensor model.layers.0.self_attn.q_a_proj.weight \
  --scheme all \
  --rows 512 \
  --cols 4096 \
  --block 128 \
  --svd-rank 4 \
  --x-rows 16
```

The `+x` lines compare float `X * W^T` with per-token int8 activation quantization plus the weight approximation. Synthetic activations are deterministic and include sparse outliers.

| scheme | weight rel L2 | output rel L2 | output cosine | output SQNR dB |
|---|---:|---:|---:|---:|
| tensor | 0.058657 | 0.060514 | 0.9981737437 | 24.363 |
| row | 0.012872 | 0.024101 | 0.9997096541 | 32.359 |
| block 128x128 | 0.025411 | 0.032166 | 0.9994825353 | 29.852 |
| smooth block | 0.017522 | 0.026282 | 0.9996547033 | 31.607 |
| svd rank 4 | 0.025201 | 0.032050 | 0.9994862758 | 29.883 |

This reinforces the weight-only conclusion under an SDOT-like path: activation quantization adds error, but the relative ordering stays stable. Smooth block quantization is materially better than plain block quantization while keeping the A64FX-friendly scale granularity.

## Additional Quantizers

Added methods:

- `block_mse`: per-128x128 block clipping selected by a small MSE search.
- `block_p99`: per-128x128 block 99.9th percentile clipping.
- `awq`: AWQ-style column equalization grid over several activation/weight exponents, followed by MSE-clipped 128x128 block quantization.

Representative sweep command:

```sh
ROWS=256 COLS=2048 SVD_RANK=4 \
  ./a64fx/fp8-int8-quant/scripts/sweep_glm52_representative.sh \
  ~/models/glm52-fp8 \
  a64fx/fp8-int8-quant/glm52_representative_sweep_v2.csv
```

Best relative L2 from the expanded sweep:

| tensor | best | best rel L2 | row | block | smooth | awq |
|---|---:|---:|---:|---:|---:|---:|
| layer0 mlp down | awq | 0.007171 | 0.009041 | 0.012110 | 0.008888 | 0.007171 |
| layer0 mlp gate | row | 0.008404 | 0.008404 | 0.012890 | 0.010186 | 0.008448 |
| layer0 mlp up | awq | 0.007566 | 0.008354 | 0.011291 | 0.008980 | 0.007566 |
| layer0 kv_a | awq | 0.009546 | 0.010177 | 0.025231 | 0.013449 | 0.009546 |
| layer0 kv_b | awq | 0.007103 | 0.008380 | 0.016947 | 0.010288 | 0.007103 |
| layer0 o_proj | awq | 0.006965 | 0.014096 | 0.011861 | 0.008729 | 0.006965 |
| layer0 q_a | row | 0.010621 | 0.010621 | 0.023915 | 0.016169 | 0.012082 |
| layer0 q_b | awq | 0.008709 | 0.018749 | 0.026539 | 0.013691 | 0.008709 |
| layer3 expert0 down | awq | 0.007044 | 0.011299 | 0.013305 | 0.008708 | 0.007044 |
| layer3 expert0 gate | awq | 0.009519 | 0.010102 | 0.018496 | 0.012645 | 0.009519 |
| layer3 expert0 up | awq | 0.009958 | 0.010061 | 0.019554 | 0.013296 | 0.009958 |

Additional observations:

- `awq` wins 9/11 representative slices and is close on the remaining 2. It is now the best A64FX-oriented candidate because it retains block quantization but reaches or beats per-row error.
- `awq` saturation is higher, roughly 0.43-0.52% in these samples. That needs matmul/perplexity validation before adopting it globally.
- `block_mse` gives small rel-L2 gains over plain block, but clipping creates large worst-element errors. It may still be useful if output error improves, but it is not as strong as AWQ.
- `block_p99` lowers MAE but badly hurts rel-L2 and max error on these FP8 weights. It is not a good default.

## BF16 Original Weights

The original BF16 checkpoint is available at `~/models/glm5.2`. The analyzer now supports both checkpoint families:

- `~/models/glm52-fp8`: `F8_E4M3` weight plus FP32 `weight_scale_inv`.
- `~/models/glm5.2`: raw `BF16` weight, widened exactly to f32 for analysis.

Representative BF16 q-proj command:

```sh
./a64fx/fp8-int8-quant/quant_analyze \
  --model ~/models/glm5.2 \
  --tensor model.layers.0.self_attn.q_a_proj.weight \
  --scheme all \
  --rows 256 \
  --cols 2048 \
  --block 128 \
  --svd-rank 4 \
  --x-rows 16 \
  --csv a64fx/fp8-int8-quant/glm52_bf16_qaproj_i8_i16.csv
```

BF16 q-proj results:

| scheme | weight rel L2 | output rel L2 | output SQNR dB | sat |
|---|---:|---:|---:|---:|
| row int8 | 0.010634 | 0.022221 | 33.065 | 0.050% |
| block int8 | 0.023784 | 0.030605 | 30.284 | 0.006% |
| smooth int8 | 0.016247 | 0.025718 | 31.795 | 0.006% |
| awq int8 | 0.012135 | 0.023421 | 32.608 | 0.395% |
| i16 | 0.000153 | 0.000168 | 75.487 | 0.000% |
| i16 row | 0.000041 | 0.000087 | 81.191 | 0.050% |
| i16 block | 0.000091 | 0.000117 | 78.600 | 0.006% |
| i16 awq | 0.000047 | 0.000089 | 81.020 | 0.394% |

BF16 representative sweep:

```sh
ROWS=256 COLS=2048 SVD_RANK=4 \
  ./a64fx/fp8-int8-quant/scripts/sweep_glm52_representative.sh \
  ~/models/glm5.2 \
  a64fx/fp8-int8-quant/glm52_bf16_representative_sweep.csv
```

Expanded BF16 sweep takeaway:

- Int8 quantization from BF16 looks very similar to int8 quantization from the FP8 checkpoint on the sampled tensors. The FP8 checkpoint is already a close model of the BF16 weights for these slices.
- `awq` remains the best int8-oriented block method for BF16 weights.
- Int16 quantization is essentially lossless at this scale: representative `i16_awq` weight rel L2 is around `2.7e-5` to `4.7e-5`, and `i16_block` is around `4.3e-5` to `1.2e-4`.
- With `--x-rows`, int16 schemes use int16 activations. For q-proj, output rel L2 is around `8.7e-5` to `1.7e-4`, far below the int8 output error.
- The cost is hardware: int16 doubles weight/activation bandwidth versus int8 and does not use A64FX int8 `sdot`; it needs SVE halfword multiply-accumulate kernels.

## FP16 Path

The analyzer now supports:

```sh
--scheme fp16 --x-rows 16 --fp16-chunks 16,32,64,128,256,512
```

This stores weights as IEEE fp16. For `+xM` output metrics, synthetic activations are also stored as fp16, then each K chunk of `M` elements is widened to fp32 and accumulated with fp32 FMA. This models the A64FX direction where fp16 has no `fdot` equivalent and must use widened FMA/reduction.

### q_a_proj, 256x2048 sample

BF16 original source:

| scheme | rel L2 | SQNR dB |
|---|---:|---:|
| fp16 weight | 1.7617e-7 | 135.081 |
| fp16+x16 | 1.8414e-4 | 74.697 |
| fp16+x32 | 1.8413e-4 | 74.697 |
| fp16+x64 | 1.8414e-4 | 74.697 |
| fp16+x128 | 1.8414e-4 | 74.697 |
| fp16+x256 | 1.8414e-4 | 74.697 |
| fp16+x512 | 1.8414e-4 | 74.697 |

FP8 checkpoint source:

| scheme | rel L2 | SQNR dB |
|---|---:|---:|
| fp16 weight | 2.1208e-4 | 73.470 |
| fp16+x16 | 2.8264e-4 | 70.975 |
| fp16+x32 | 2.8263e-4 | 70.976 |
| fp16+x64 | 2.8264e-4 | 70.975 |
| fp16+x128 | 2.8264e-4 | 70.975 |
| fp16+x256 | 2.8264e-4 | 70.975 |
| fp16+x512 | 2.8264e-4 | 70.975 |

### o_proj, 256x2048 sample

BF16 original source:

| scheme | rel L2 | SQNR dB |
|---|---:|---:|
| fp16 weight | 3.2579e-7 | 129.741 |
| fp16+x16 | 1.8053e-4 | 74.869 |
| fp16+x32 | 1.8054e-4 | 74.869 |
| fp16+x64 | 1.8053e-4 | 74.869 |
| fp16+x128 | 1.8053e-4 | 74.869 |
| fp16+x256 | 1.8052e-4 | 74.869 |
| fp16+x512 | 1.8053e-4 | 74.869 |

FP8 checkpoint source:

| scheme | rel L2 | SQNR dB |
|---|---:|---:|
| fp16 weight | 1.8708e-4 | 74.559 |
| fp16+x16 | 2.5750e-4 | 71.784 |
| fp16+x32 | 2.5750e-4 | 71.784 |
| fp16+x64 | 2.5751e-4 | 71.784 |
| fp16+x128 | 2.5750e-4 | 71.784 |
| fp16+x256 | 2.5751e-4 | 71.784 |
| fp16+x512 | 2.5751e-4 | 71.784 |

FP16 observations:

- BF16 -> FP16 weight storage is almost exact on these slices because the weights are small and fit well in fp16.
- FP8 -> FP16 is still very accurate but not exact because dequantized FP8 values with FP32 block scales are not always fp16-representable.
- Chunk size from 16 to 512 has negligible effect here when each chunk is widened and accumulated in fp32. Error is dominated by fp16 storage of activations/weights, not fp32 reduction order.
- FP16 output error is much lower than int8 and comparable to int16 tensor-scale output error, but the A64FX compute path is fp32 FMA after widening, not an `sdot`-like packed dot path.

## Batch CSV Reports

Added:

```text
scripts/batch_quant_report.py
```

It reads safetensors directly with Python stdlib and mmap, samples each 2D `.weight` tensor in streaming shard order, and writes one CSV row per tensor with:

- tensor metadata: model, shard, dtype, full shape, sampled shape, scale availability
- sample features: absmax, mean abs, RMS, zero percentage
- quant errors for selected methods: `i8_block128`, `i8_awq`, `i16_block128`, `fp16`

Commands run:

```sh
# Interrupted after 21,900 tensors because a full uncapped GLM5.2 MoE pass is long.
./a64fx/fp8-int8-quant/scripts/batch_quant_report.py \
  --model ~/models/glm5.2 \
  --out a64fx/fp8-int8-quant/report_glm52_bf16_all_weights_sample.csv \
  --rows 4 --cols 64 --max-elements-per-tensor 256 \
  --methods i8_block128,i8_awq,i16_block128,fp16

./a64fx/fp8-int8-quant/scripts/batch_quant_report.py \
  --model ~/models/glm52-fp8 \
  --out a64fx/fp8-int8-quant/report_glm52_fp8_weights_sample_2k.csv \
  --rows 4 --cols 64 --max-elements-per-tensor 256 --max-tensors 2000 \
  --methods i8_block128,i8_awq,i16_block128,fp16

./a64fx/fp8-int8-quant/scripts/batch_quant_report.py \
  --model ~/models/m3 \
  --out a64fx/fp8-int8-quant/report_m3_bf16_weights_sample_2k.csv \
  --rows 4 --cols 64 --max-elements-per-tensor 256 --max-tensors 2000 \
  --methods i8_block128,i8_awq,i16_block128,fp16

./a64fx/fp8-int8-quant/scripts/batch_quant_report.py \
  --model ~/models/m3-fp8 \
  --out a64fx/fp8-int8-quant/report_m3_fp8_weights_sample_2k.csv \
  --rows 4 --cols 64 --max-elements-per-tensor 256 --max-tensors 2000 \
  --methods i8_block128,i8_awq,i16_block128,fp16
```

Note: `~/models/m30fp8` was not present by that exact name; the installed FP8 M3 model is `~/models/m3-fp8`.

Aggregate summary, written to `batch_report_summary.csv`:

| report | rows | method | mean rel L2 | median rel L2 | p95 rel L2 |
|---|---:|---|---:|---:|---:|
| glm5.2 partial | 21900 | i8_block128 | 0.006982 | 0.006853 | 0.008583 |
| glm5.2 partial | 21900 | i8_awq | 0.003097 | 0.003095 | 0.003338 |
| glm5.2 partial | 21900 | i16_block128 | 2.705e-5 | 2.656e-5 | 3.319e-5 |
| glm5.2 partial | 21900 | fp16 | 8.975e-9 | 0.0 | 5.635e-8 |
| glm52-fp8 2k | 2000 | i8_block128 | 0.007083 | 0.006961 | 0.008405 |
| glm52-fp8 2k | 2000 | i8_awq | 0.003170 | 0.003171 | 0.003411 |
| glm52-fp8 2k | 2000 | i16_block128 | 2.710e-5 | 2.684e-5 | 3.281e-5 |
| glm52-fp8 2k | 2000 | fp16 | 1.800e-4 | 2.020e-4 | 2.447e-4 |
| m3 2k | 2000 | i8_block128 | 0.007077 | 0.006940 | 0.008879 |
| m3 2k | 2000 | i8_awq | 0.003182 | 0.003179 | 0.003415 |
| m3 2k | 2000 | i16_block128 | 2.733e-5 | 2.680e-5 | 3.362e-5 |
| m3 2k | 2000 | fp16 | 5.243e-7 | 0.0 | 0.0 |
| m3-fp8 2k | 2000 | i8_block128 | 0.006639 | 0.006578 | 0.007651 |
| m3-fp8 2k | 2000 | i8_awq | 0.003189 | 0.003186 | 0.003420 |
| m3-fp8 2k | 2000 | i16_block128 | 2.624e-5 | 2.603e-5 | 3.034e-5 |
| m3-fp8 2k | 2000 | fp16 | 6.261e-7 | 0.0 | 0.0 |

Batch observations:

- Across GLM and M3 samples, `i8_awq` roughly halves relative L2 versus plain 128-block int8.
- `i16_block128` is consistently near `2.6e-5` to `2.7e-5` mean rel L2.
- BF16 source tensors often round exactly to FP16 in the small sampled regions, hence zero median fp16 error in several reports.
- GLM FP8 -> FP16 has nonzero fp16 error around `1.8e-4` mean rel L2 because FP8 dequantized values include FP32 block scales.

## SmoothQuant/SVDQuant on INT8 and INT16

The analyzer now exposes both SmoothQuant-style scaling and SVD residual compensation for int8 and int16:

```sh
./a64fx/fp8-int8-quant/quant_analyze \
  --model ~/models/glm5.2 \
  --tensor model.layers.0.self_attn.q_a_proj.weight \
  --scheme all \
  --rows 256 --cols 2048 --block 128 --svd-rank 4 --x-rows 16

./a64fx/fp8-int8-quant/quant_analyze \
  --model ~/models/glm52-fp8 \
  --tensor model.layers.0.self_attn.q_a_proj.weight \
  --scheme all \
  --rows 256 --cols 2048 --block 128 --svd-rank 4 --x-rows 16
```

q_a_proj 256x2048 summary:

| source | scheme | weight rel L2 | output rel L2 | sat |
|---|---|---:|---:|---:|
| BF16 | smooth int8 | 0.016247 | 0.025718 | 0.006% |
| BF16 | awq int8 | 0.012135 | 0.023421 | 0.395% |
| BF16 | svd int8 rank4 | 0.023419 | 0.030304 | 0.006% |
| BF16 | i16_smooth | 6.300e-5 | 9.958e-5 | 0.006% |
| BF16 | i16_awq | 4.692e-5 | 8.892e-5 | 0.394% |
| BF16 | i16_svd rank4 | 8.944e-5 | 1.165e-4 | 0.006% |
| FP8 | smooth int8 | 0.016169 | 0.025900 | 0.008% |
| FP8 | awq int8 | 0.012082 | 0.023072 | 0.435% |
| FP8 | svd int8 rank4 | 0.023552 | 0.030569 | 0.008% |
| FP8 | i16_smooth | 6.232e-5 | 9.985e-5 | 0.008% |
| FP8 | i16_awq | 4.607e-5 | 9.025e-5 | 0.434% |
| FP8 | i16_svd rank4 | 9.231e-5 | 1.196e-4 | 0.008% |

Takeaway:

- SmoothQuant helps both int8 and int16 block quantization, but AWQ-style scaling is better on this q-proj slice and in the broader sampled reports.
- Rank-4 SVD residual correction is not compelling here. It slightly improves plain block relative L2, but remains much worse than Smooth/AWQ for int8 and is unnecessary for int16.
- Int16 accuracy is effectively a fallback precision path. It is far more accurate than int8, but it doubles bandwidth and misses the A64FX int8 `sdot` throughput target.

## Quantize-or-Not Thresholds

There is no proven universal threshold that decides whether a tensor should be quantized. A useful cutoff must be calibrated against end-to-end logit/perplexity drift, and sensitive tensors can violate a weight-only rule even with low relative error.

Practical gates for this A64FX experiment:

- Green for int8: synthetic output rel L2 below `0.02`, cosine above `0.9998`, saturation below `0.5%`, and no large max-error outliers.
- Yellow for int8: output rel L2 `0.02-0.04` or saturation `0.5-1.0%`; require end-to-end validation or keep per-tensor exceptions.
- Red for int8: output rel L2 above `0.04`, saturation above `1.0%`, SQNR below roughly `30 dB`, or known sensitive roles such as embeddings, lm_head, router/gating, norms, and first/last layers.
- For cheap batch screening before matmul tests: `i8_awq` weight rel L2 below `0.005` is a good candidate, `0.005-0.01` needs inspection, and above `0.01` should stay fp16/bf16 or move to int16 unless end-to-end quality says otherwise.

Recommendation: use int8 AWQ block128 as the main A64FX `sdot` candidate, SmoothQuant block128 as the simpler fallback, and reserve fp16/bf16 or int16 for tensors that fail the gates. Do not carry SVDQuant into the first A64FX kernel unless a later end-to-end run shows a quality gain large enough to pay for the residual storage and extra compute.

## INT4/FP4 Weight Quantization

Added analyzer modes:

- `int4`: symmetric signed int4 per 128x128 block.
- `fp4`: FP4 E2M1-style signed codebook per 128x128 block.

Focused q_a_proj command:

```sh
./a64fx/fp8-int8-quant/quant_analyze \
  --model ~/models/glm52-fp8 \
  --tensor model.layers.0.self_attn.q_a_proj.weight \
  --scheme fp4 \
  --rows 256 --cols 2048 --block 128 --x-rows 16
```

q_a_proj 256x2048 summary:

| source | scheme | weight rel L2 | output rel L2 | output cosine |
|---|---|---:|---:|---:|
| BF16 | int4 block128 | 0.376702 | 0.368497 | 0.936933 |
| BF16 | fp4 block128 | 0.245572 | 0.243350 | 0.971214 |
| FP8 | int4 block128 | 0.378765 | 0.373704 | 0.934652 |
| FP8 | fp4 block128 | 0.245534 | 0.242640 | 0.970749 |

Streaming 1k tensor screens:

```sh
./a64fx/fp8-int8-quant/scripts/batch_quant_report.py \
  --model ~/models/glm52-fp8 \
  --out a64fx/fp8-int8-quant/report_glm52_fp8_int4_fp4_1k.csv \
  --rows 4 --cols 64 --max-elements-per-tensor 256 --max-tensors 1000 \
  --methods i8_awq,int4_block128,fp4_block128

./a64fx/fp8-int8-quant/scripts/batch_quant_report.py \
  --model ~/models/m3-fp8 \
  --out a64fx/fp8-int8-quant/report_m3_fp8_int4_fp4_1k.csv \
  --rows 4 --cols 64 --max-elements-per-tensor 256 --max-tensors 1000 \
  --methods i8_awq,int4_block128,fp4_block128
```

| report | method | mean rel L2 | median rel L2 | p95 rel L2 |
|---|---|---:|---:|---:|
| GLM52 FP8 1k | i8_awq | 0.003174 | 0.003177 | 0.003412 |
| GLM52 FP8 1k | int4_block128 | 0.127890 | 0.125003 | 0.158262 |
| GLM52 FP8 1k | fp4_block128 | 0.111712 | 0.111216 | 0.124453 |
| M3 FP8 1k | i8_awq | 0.003192 | 0.003188 | 0.003426 |
| M3 FP8 1k | int4_block128 | 0.123136 | 0.122627 | 0.143970 |
| M3 FP8 1k | fp4_block128 | 0.112425 | 0.112077 | 0.123177 |

Takeaway: plain 4-bit weight quantization is not a good first A64FX path for these dense weights. FP4 is better than symmetric int4, but both are far beyond the int8 quality gates and much worse than AWQ int8. A useful 4-bit weight path would need stronger methods than plain block scaling, for example activation-aware/groupwise scaling, mixed precision, or tensor exceptions.

## TurboQuant-Style KV Cache Proxy

TurboQuant is a KV-cache compression method, not a static weight quantizer. The paper uses randomized rotation and scalar quantizers, with an optional QJL residual correction path for unbiased inner products. The local script implements an MSE-style randomized sign plus FWHT rotation proxy and measures attention-score and attention-output drift:

```sh
./a64fx/fp8-int8-quant/scripts/kv_cache_quant.py \
  --seq 2048 --dim 128 \
  --out a64fx/fp8-int8-quant/kv_cache_quant_2048x128.csv
```

Synthetic 2048-token, 128-dim KV-cache result:

| method | bits/value | key rel L2 | value rel L2 | score rel L2 | softmax KL | attention output rel L2 | output cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| int4 | 4 | 0.138583 | 0.126560 | 0.160889 | 0.016235 | 0.245757 | 0.970818 |
| fp4 | 4 | 0.115294 | 0.111080 | 0.121506 | 0.008839 | 0.192872 | 0.981923 |
| turbo4 | 4 | 0.115638 | 0.116161 | 0.116209 | 0.007407 | 0.172638 | 0.985627 |
| turbo3 | 3 | 0.270028 | 0.270964 | 0.270749 | 0.040000 | 0.419757 | 0.925852 |

Takeaway: TurboQuant-style rotation improves attention metrics over scalar int4/fp4 at the same 4-bit budget in this proxy, especially attention output rel L2. The 3-bit proxy is much more aggressive and should be treated as experimental until evaluated on real KV activations and full-model quality.

## Exhaustive Parameter Search

The first broad attempt launched four searches in parallel and was stopped because the combined mmap/process footprint could trigger OOM. The retry uses one process at a time, a smaller per-tensor sample, and bounded top-k retention. This stayed around tens of MB RSS.

Search script:

```sh
./a64fx/fp8-int8-quant/scripts/search_quant_configs.py \
  --model ~/models/glm52-fp8 \
  --out a64fx/fp8-int8-quant/search_glm52_fp8_allconfigs_100.csv \
  --rows 32 --cols 256 --max-elements-per-tensor 8192 \
  --max-tensors 100 --topk 0
```

Additional bounded top-k searches were run for:

```text
search_glm52_fp8_top.csv
search_glm52_bf16_top.csv
search_m3_fp8_top.csv
search_m3_bf16_top.csv
```

The search covers:

- symmetric tensor, row, and block quantization for 3/4/5/6/8/16 bits
- block sizes 16, 32, 64, 128, and 256
- clipping ratios from 0.80 to 1.0 for 4/5/6/8-bit block quantization
- column-scaled block variants with alpha 0.0, 0.25, 0.5, 0.75, and 1.0
- FP4 E2M1, NF4, and FP16
- simple dequant-cost features: bytes per weight, scales per 1k values, estimated dequant ops per value, and likely A64FX path

### GLM52 FP8 full-config frontier

Source: `search_glm52_fp8_allconfigs_100.csv`, 100 tensors, 229 configs/tensor, 32x256 sample.

| class | best config | bytes/weight | scales/1k | ops/value | mean rel L2 | p95 rel L2 | likely path |
|---|---|---:|---:|---:|---:|---:|---|
| best accuracy | int16 row | 2.000 | 3.906 | 1 | 2.84e-5 | 3.46e-5 | SVE i16 MAC |
| best int16 block128 | int16 block128 | 2.000 | 0.244 | 1 | 3.96e-5 | 7.38e-5 | SVE i16 MAC |
| fp16 storage | fp16 | 2.000 | 0.000 | 0 | 1.80e-4 | 2.32e-4 | widen + fp32 FMA |
| best int8 accuracy | int8 column-scale block128 alpha0 | 1.000 | 31.494 | 2 | 0.00557 | 0.00597 | A64FX SDOT |
| best cheap int8 block32 | int8 block32 | 1.000 | 0.977 | 1 | 0.00878 | 0.01521 | A64FX SDOT |
| recommended int8 block128 | int8 block128 | 1.000 | 0.244 | 1 | 0.01011 | 0.01865 | A64FX SDOT |
| int8 block256 clipped | int8 clip0.975 block256 | 1.000 | 0.122 | 1 | 0.01057 | 0.02078 | A64FX SDOT |
| best 6-bit | 6-bit column-scale block128 alpha0 | 0.750 | 31.494 | 2 | 0.02317 | 0.02473 | unpack to i8/dot |
| cheap 6-bit block128 | 6-bit clipped block128 | 0.750 | 0.244 | 2 | 0.03820 | 0.07036 | unpack to i8/dot |
| best 5-bit | 5-bit column-scale block128 alpha0 | 0.625 | 31.494 | 2 | 0.04709 | 0.05086 | unpack to i8/dot |
| best 4-bit | 4-bit column-scale block128 alpha0 | 0.500 | 31.494 | 2 | 0.10010 | 0.10841 | unpack to i8/dot |
| best NF4 | NF4 block16 | 0.500 | 3.906 | 3 | 0.10352 | 0.12882 | lookup + FMA |
| best 3-bit | 3-bit row | 0.375 | 3.906 | 2 | 0.30814 | 0.37486 | unpack to i8/dot |

Processing-time columns are included in `search_summary.csv` as `mean_quant_ms`, `median_quant_ms`, and `p95_quant_ms`. These are host Python fitting/simulation times for one 32x256 sampled tensor. They are useful for comparing preprocessing cost and method complexity in this tool, but they are not A64FX kernel runtime.

Selected GLM52-FP8 processing times:

| config | mean rel L2 | p95 rel L2 | mean quant ms | median quant ms | p95 quant ms |
|---|---:|---:|---:|---:|---:|
| fp16 | 1.80e-4 | 2.32e-4 | 3.971 | 3.941 | 4.160 |
| int16 block128 | 3.96e-5 | 7.38e-5 | 7.230 | 7.182 | 7.426 |
| int8 block32 | 0.00878 | 0.01521 | 7.429 | 7.397 | 7.623 |
| int8 block64 | 0.00942 | 0.01721 | 7.337 | 7.299 | 7.523 |
| int8 block128 | 0.01011 | 0.01865 | 7.304 | 7.251 | 7.482 |
| int8 column-scale block128 alpha0 | 0.00557 | 0.00597 | 9.921 | 9.850 | 10.117 |
| 6-bit column-scale block128 alpha0 | 0.02317 | 0.02473 | 9.923 | 9.863 | 10.106 |
| NF4 block16 | 0.10352 | 0.12882 | 22.525 | 22.566 | 22.655 |

Timing observations:

- Plain int8 block32/64/128 all take about `7.3-7.4 ms` per sampled tensor in this Python tool; block128 is slightly faster and has lower scale metadata.
- Column-scale modes improve error but cost about `10 ms`, roughly 35% more processing time in this implementation, and require extra dequant work.
- NF4/FP4-style lookup codebook quantization is much slower in the Python fitting path and still much less accurate than int8.
- FP16 is fastest in the tool because it is just scalar conversion, but its A64FX compute path is widened fp32 FMA rather than packed dot product.

### Whole-model search timing

`search_quant_configs.py` now supports `--threads N`. The default is half of `os.cpu_count()`; on this host `os.cpu_count()` is 72, so the default CPU path uses 36 worker processes. Use `--threads 1` for sequential timing.

Timed command shape:

```sh
./a64fx/fp8-int8-quant/scripts/search_quant_configs.py \
  --model ~/models/glm52-fp8 \
  --out a64fx/fp8-int8-quant/search_glm52_fp8_cpu.csv \
  --rows 32 --cols 256 --max-elements-per-tensor 8192 \
  --max-tensors 100 --topk 20 \
  --summary-json a64fx/fp8-int8-quant/search_glm52_fp8_cpu.json
```

The model-level timing summary is written to JSON and aggregated in `search_model_timing.csv`.

| mode | model | threads | tensors | configs | elapsed sec | configs/sec |
|---|---|---:|---:|---:|---:|---:|
| sequential | glm52-fp8 | 1 | 100 | 22900 | 299.459 | 76.47 |
| sequential | glm5.2 | 1 | 100 | 22900 | 296.540 | 77.22 |
| sequential | m3-fp8 | 1 | 100 | 22900 | 295.384 | 77.53 |
| sequential | m3 | 1 | 100 | 22900 | 298.100 | 76.82 |
| cpu default | glm52-fp8 | 36 | 100 | 22900 | 303.303 | 75.50 |
| cpu default | glm5.2 | 36 | 100 | 22900 | 307.242 | 74.53 |
| cpu default | m3-fp8 | 36 | 100 | 22900 | 300.495 | 76.21 |
| cpu default | m3 | 36 | 100 | 22900 | 301.382 | 75.98 |

CPU timing observation: for this bounded `32x256`, 100-tensor search, 36-worker multiprocessing does not improve wall-clock time. It is slightly slower than sequential because each tensor task is small and process/shard setup plus result serialization dominate. The threaded CPU path should be more useful for larger per-tensor samples or a heavier config space; for the current safe screening setup, `--threads 1` is the better default for reproducible timing and low memory pressure, even though the script default follows the requested half-core policy.

Top-k searches on GLM52 BF16, M3 FP8, and M3 BF16 showed the same ordering:

- lowest error: int16 row/block or fp16, depending on source representation
- best SDOT-compatible path: int8 with column scaling if extra dequant multiply and many column scales are acceptable
- best cheap SDOT path: plain int8 block32/block64/block128
- 6-bit can be made moderate-error but needs unpacking and does not naturally beat int8 SDOT for A64FX throughput
- 4-bit and 3-bit are not viable for dense weights without a stronger mixed-precision or activation-aware method

### Updated recommendation from search

For A64FX, prefer this order:

1. `int8 block128` as the first production kernel target: simple scale metadata, one dequant multiply, direct `sdot`, mean rel L2 about `0.0101` and p95 about `0.0187` on the GLM52-FP8 search sample.
2. `int8 block32` or `block64` if quality is more important than scale metadata volume: mean rel L2 about `0.0088-0.0094`, but 2-4x more block scales than block128.
3. `int8 column-scale block128 alpha0` only as an accuracy mode: mean rel L2 about `0.0056`, but it needs per-column scaling plus block scales and an extra dequant multiply.
4. `int16 block128` for tensors failing int8 gates: much lower error, but 2x weight bandwidth and no int8 `sdot`.
5. Avoid 3/4/5-bit dense-weight paths for the first A64FX implementation. Even the best 4-bit/NF4 variants are around `0.10` mean rel L2 in the search, far outside the int8 gates.
