# GLM-5.3-Flash decode kernels on A64FX

This file has two parts:
- The inventory of every GLM-5.3-Flash kernel, with the per-rank shapes of the production 12-node layout: 1 rank per node, the 6-head rank, native UD-Q4_K_XL.
- The kernel cores in this directory, with their measured efficiency.

**Measurement harness.** The harness and its campaign data live in the clair repository, in `a64fx/llm-guided-opt/sim-accuracy/glm53f/` (README.md, STATUS.md). Every number below is a median over at least 10 samples on a quiet A64FX node at 2.0 GHz with eco_state=0 and `XOS_MMM_L_PAGING_POLICY=demand:demand:demand`. The binary is a clang-built dynamic glibc ELF, the same one QLAIR replays.

## Model constants

| Constant | Value |
|---|---|
| hidden | 4096 |
| Layers | 45, plus 1 MTP layer |
| vocab | 154880 |
| KDA layers | 34. 64 heads × 128; q/k/v 8192; conv 4; f and g low-rank 128; state 128×128 FP32 per head |
| Sparse MLA + DSA layers | 11 (l%4==3). 64 heads; q_lora 1536; kv_lora 512; qk_nope 256; v 256; no RoPE; indexer 32×128, top-2048, pools of 4 |
| Dense FFN layers | 0–2, 12288 wide |
| MoE layers | 42. 288 experts, top-8, sigmoid+bias, inter 2048, 1 shared |
| mHC | 4 streams, Sinkhorn 20 |

## Inventory: one decode token, per rank

| # | Kernel | Global W (out×in) | Per-rank decode W | Format | Calls/token | MB/token | Class (decode / prefill) |
|---|---|---|---|---|---|---|---|
| 1 | Embedding row + broadcast | 154880×4096 | 12907 rows | F32 | 1 | 0.016 | copy |
| 2 | mHC pre: RMS(16384), GEMV, Sinkhorn, mix | 24×16384 | replicated | BF16 | 90 | 70.8 | latency / bandwidth |
| 3 | mHC post | – | 4×4096 | F32 | 90 | small | latency |
| 4 | KDA q/k/v + f_a, g_a, beta | 3×8192×4096, 2×128×4096, 64×4096 | 3×768×4096, 2×128×4096, 6×4096 | Q8_0R | 34 | 11.8/layer | bandwidth / SDOT GEMM |
| 5 | KDA conv4 + SiLU | 3×8192×4 | 3×768×4 | BF16 | 34 | small | scan |
| 6 | KDA f_b, g_b | 8192×128 ×2 | 768×128 ×2 | Q8_0R | 34 | 0.22/layer | bandwidth |
| 7 | KDA recurrence, L2norm, gated RMSNorm | 128×128 state per head | 6 heads | F32 | 34 | 0.39/layer | FMLA |
| 8 | KDA o_proj | 4096×8192 | 4096×768 | Q8_0R | 34 | 3.54/layer | bandwidth / GEMM |
| 9 | MLA q_a + RMSNorm | 1536×4096 | replicated | Q8_0R | 11 | 7.08/layer | bandwidth |
| 10 | MLA q_b; kv_a + RMSNorm | 16384×1536; 512×4096 | 1536×1536; 512×4096 | Q8_0R | 11 | 2.65 + 2.36 | bandwidth |
| 11 | DSA indexer: wq_b, wk, compress, weights_proj | 4096×1536, 128×4096, 128×4096, 32×4096 | replicated | BF16 | 11 | 15/layer | bandwidth |
| 12 | Index score + top-k | L/4 × 32 × 128 | split across ranks beyond 2K | F32 | 11 | 0.5 KB/pool | bandwidth / select |
| 13 | Latent gather + absorbed MLA | k_b / v_b 64×256×512 | 6 heads, ns ≤ 2051 | BF16 / Q8_0R | 11 | 2.45 + cache | FMLA / bandwidth |
| 14 | MLA o_proj | 4096×16384 | 4096×1536 | Q8_0R | 11 | 7.08/layer | bandwidth / GEMM |
| 15 | Dense FFN (layers 0–2) | 2×12288×4096, 4096×12288 | 2×1024×4096, 4096×1024 | Q8_0R | 3 | 14.2/layer | bandwidth / GEMM |
| 16 | Router top-8 of 288 | 288×4096 | replicated | BF16 | 42 | 2.36/layer | bandwidth + select |
| 17 | Routed expert part: gate/up, SwiGLU, down | 288×(2×2048×4096 + 4096×2048) | per part 512×4096 + 4096×256; about 5.33 parts per token | Q4_K; Q5_K/Q6_K down | 42 | about 10.1/layer | bandwidth / SDOT GEMM (M≈14) |
| 18 | Shared expert | 2×2048×4096, 4096×2048 | 2×192×4096, 4096×192 | Q8_0R | 42 | 2.65/layer | bandwidth |
| 19 | Allreduce 4096 FP32 | – | – | – | 90 | 0.016 | communication |
| 20 | Head GEMV + argmax | 154880×4096 | 12907×4096 | F32 | 1 | 211.5 | bandwidth |
| 21 | Activation quantize (Q8_0 / Q8_K) | – | 4096–8192 | – | about 200 | – | latency |

Per rank, one token reads about 1.9 GB of weights:

| KDA | Sparse | Routed | Head | Shared | Router | mHC | Dense |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 530 MB | 402 MB | 425 MB | 212 MB | 111 MB | 99 MB | 71 MB | 43 MB |

## Kernel cores

Each core computes output rows `[r0, r1)` of one GEMV. Cores contain no OpenMP and no libc calls (`glm53f_kern.h`).

| Core | File | Description |
|---|---|---|
| `gk_q8_0r_v0`, `gk_q4_k_v0`, `gk_q5_k_v0`, `gk_q6_k_v0`, `gk_f32_v0` | `glm53f_kern_v0.c` | Verbatim production loops (`glm53f_iq_bridge.c`, `glm53f_target_head_12n.c`) |
| `gk_q8_0r16_v1` | `glm53f_kern_q8r16.c` | Lossless 16-row panel repack of Q8_0 (1.125 B/weight). One SDOT lane per output row, indexed SDOT against LD1RQB. No FADDV. |
| `gk_q8_0r16_v2` | same | As v1, with two 4-deep SDOT chains per block |
| `gk_q8_0r16_v3` | same | Plain SDOT against LD1RW broadcast. Indexed SDOT is 2 uops on A64FX. |
| `gk_q8_0r16_v3pf{4k,16k,64k}` | same | v3 plus L2 software prefetch of the weight stream, 4/16/64 KiB ahead |
| `gk_q4_kp16_v1` | `glm53f_kern_q4k16.c` | Lossless 16-row Q4_K panel repack (152 B per 256 columns per row, +5.6% bytes; scales unpacked to bytes, `d`/`dmin` as f32). The nibbles are laid out so one AND or LSR yields each row-lane SDOT operand. Scale and min sums are exact in int32; per-32 activation sums are precomputed. |
| `gk_q4_kp16_v1pf` | same | v1 plus L2 software prefetch 16 KiB ahead |
| `gk_q5_kp16_v1`, `_v1pf` | `glm53f_kern_q5k16.c` | Lossless 16-row Q5_K panels: 184 B per 256 columns per row (+4.5%). The Q4_KP16 nibble plane is kept, plus one 64 B high-bit plane per sub-block (bit t of byte r·4+i is the fifth bit of k-group t). The high part is a separate SDOT on `(plane >> t) & 1`, scaled by 16 in int32. |
| `gk_q5_kp16_v2pf` | same | 8 two-deep SDOT chains instead of 4. Slower: 7.5 vs 8.5 B/cycle from L1, so Q5 is bound by FP-op count, not latency. |
| `gk_q6_kp16_v1`, `_v1pf` | `glm53f_kern_q6k16.c` | Lossless 16-row Q6_K panels: 212 B per 256 columns per row (+1%). Per 16-weight sub-block: 2 nibble vectors, one 2-bit high plane, 16 int8 scales; f32 `d` per row. The −32 offset is an exact int32 term using per-16 activation sums. |

## Measured efficiency: HBM-streaming, per-rank shapes (job 51943789)

**Roofs** (measured by the harness itself):
- Read: 36.4 GB/s on 1 core, 224.6 GB/s on 1 CMG, 900.8 GB/s on 48 threads.
- SDOT and FMLA fp32/fp16: 99.6–99.9% of peak at 1/12/48 threads.

| Shape | Thr | v0 GB/s (% roof) | v3pf16k GB/s (% roof) | Speed-up |
|---|---:|---:|---:|---:|
| KDA q/k/v 2304×4096 | 1 | 6.7 (18%) | 33.2 (91%) | 5.0× |
| KDA q/k/v 2304×4096 | 12 | 71.6 (32%) | 200 (89%) | 2.8× |
| KDA q/k/v 2304×4096 | 48 | 260 (29%) | 664 (74%)¹ | 2.6× |
| Head 12907×4096, F32 v0 → Q8_0R16 | 48 | 652 (72%) | 770 (85%) | 4.2× in time |

¹ The 48-thread KDA region lasts only about 16 µs, and its CV is 15–18%. Stream start-up and ramp dominate at that size. Closing the gap needs cross-kernel prefetch in a persistent layer chain; a single kernel cannot do it.

Routed-expert gate/up, Q4_K, 8 parts (4096×4096). GB/s counts the bytes actually streamed; the panel format is +5.6%.

| Thr | v0 cycles | v1pf cycles | v1pf GB/s (% roof) | Speed-up |
|---:|---:|---:|---:|---:|
| 1 | 4.80 M | 0.94 M | 21.2 (58%) | 5.1× |
| 12 | 422 k | 106 k | 188 (84%) | 4.0× |
| 48 | 121 k | 35.8 k | 557 (62%)¹ | 3.4× |
| 48, decode-average 5.33 parts (2736 rows) | 83.7 k | 28.5 k | 467 | 2.9× |

Q4_KP16 v1 from L1 runs at 11.7 B/cycle, 5.5× v0. A fully unrolled first version spilled Z registers, about 80 STR/LDR Z per super-block, and reached only 6.1 B/cycle.

Routed-expert down projection, 8 parts (32768×256, K=256). v1 means v1pf unless noted.

| Kernel | Thr | v0 cycles | v1 cycles | v1 % roof | Speed-up |
|---|---:|---:|---:|---:|---:|
| Q5_K → Q5_KP16 | 1 | 5.31 M | 0.94 M | 35% | 5.7× |
| Q5_K → Q5_KP16 | 12 | 450 k | 94 k | 57% | 4.8× |
| Q5_K → Q5_KP16 | 48 | 121 k | 31 k | 43% | 3.9× |
| Q6_K → Q6_KP16 | 1 | 5.48 M | 0.82 M | 47% | 6.7× |
| Q6_K → Q6_KP16 | 12 | 464 k | 78 k | 79% | 5.9× |
| Q6_K → Q6_KP16 | 48 | 125 k | 29 k | 53% | 4.3× |

From L1:
- Q5_KP16 runs at 8.5 B/cycle and Q6_KP16 at 8.7, against 1.1 and 1.3 for v0.
- Q5 remains bound by FP-op count, at about 49 ops per 16-row × 32-weight sub-block. It is below the 9.4 B/cycle per core needed to saturate a CMG.

Production v0 kernels at 48 threads:

| Kernel | GB/s | % of roof |
|---|---:|---:|
| Q8_0R MLA q_a | 247 | 27 |
| Q8_0R o_proj 4096×768 | 161 | 18 |
| Q4_K gate/up | 156 | 17 |
| Q5_K down | 97 | 11 |
| Q6_K down | 111 | 12 |

All v0 kernels are issue-bound, not bandwidth-bound. From L1:

| Kernel | B/cycle |
|---|---:|
| Q8_0R v0 | 9.6 |
| Q4_K v0 | 2.0 |
| Q5_K v0 | 1.1 |
| Q6_K v0 | 1.2 |

Reaching the node roof needs about 9.4 B/cycle per core.

## Next

1. **K-quant panels.** All three are done (Q4/Q5/Q6_KP16). Q5 is still FP-op bound at 57% of the CMG roof. Options: 2 panels per iteration to share LD1RW broadcasts; drop the `svdup_s32(0)` MOVIs by chaining the first SDOT; or a lossy INT8-128 path behind the quality gate.
2. **Persistent layer chain.** Prefetch the next matrix during the current one to remove the 48-thread ramp.
3. **Rewire the production path.** Have `glm53f_iq_bridge.c` call these cores, repack Q8_0R to Q8_0R16 at load, and convert the head from F32 to Q8_0R16 losslessly.

## Two-panel variants and layer chains (2026-09-28)

**Two-panel kernels.** These process 32 rows per iteration and share the LD1RW activation broadcasts:
- `gk_q8_0r16_v4pf`: 31.2 B/cycle from L1 (v3: 24.4).
- `gk_q4_kp16_v3pf`: 15.4 B/cycle (v1: 12.3).
- `gk_q5_kp16_v3pf`: no gain; it is FP-op bound.

**Per-rank decode GEMV chain for one token** (clair `sim-accuracy/glm53f/chain.c`). The chain runs all 270 dependent GEMVs of the 12-node layout with production OpenMP hardware barriers. It excludes allreduce, attention cores, the KDA recurrence and mHC. At 48 threads:

| Kernel set | ms per token | % of node read roof |
|---|---:|---:|
| Production kernels | 8.68 | 21.7% |
| Panel kernels + 16 KiB next-stage prefetch | 2.92 | 59.3% |

That is 2.97× faster. The two-panel kernels do not improve the chain: it is bound by per-stage start-up (about 10 µs stages, 1.4 µs barrier). Use 64 KiB pages or per-thread 2 MiB-aligned slices: XOS large pages shared across CMGs cost 1.8×.
