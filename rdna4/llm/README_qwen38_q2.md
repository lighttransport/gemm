# Qwen3.8 Flash Next Q2_0 on RX 9070 XT

Build and launch from the repository root:

```sh
make -C rdna4/llm test_hip_llm
rdna4/llm/run_qwen38_flash_next_q2_rocm.sh --bench --decode 8
```

The launcher uses the two GGUF shards in `/mnt/disk1/models/q38nf/` and a
262144 token context. `QWEN38_Q2_MODEL` can select another first shard.
Additional arguments pass through to `test_hip_llm`.
The launcher selects `/opt/rocm/lib` when it exists, so a system-installed
older ROCm library does not take precedence over the gfx1201 runtime.

The 26.8 GiB IQ4_NL PLE table stays on the model SSD. Sixteen reader workers
fetch its selected rows through a bounded 64 MiB page cache, using direct I/O
when available. The 31.6 GiB Q2_0 routed experts are staged into anonymous
CPU RAM in 8 MiB reads, dropping the source page cache after each read. The
GPU keeps a depth weighted LRU expert cache capped at 8 GiB and reserves at
least 1.5 GiB of reported free VRAM for execution. Each routed layer needs at
least ten cache slots. Selected misses execute on the CPU and one cold expert
per layer is promoted asynchronously. Routing evaluates all ten selected
experts; there is no resident only approximation. Full attention KV uses
scaled INT8 storage.

Cached Q2_0 experts run with two grouped GPU launches per layer, one for
gate/up and one for down projection. CPU misses use an AVX2 dot product that
transposes each activation block once for reuse across expert rows. Both paths
keep the exact top-ten routing and the same greedy output sequence.

The launcher defaults to scalar prefill and disables host registration and
the older mapped miss path. The cache capacity depends on free VRAM at load
time. Close other GPU applications if the loader reports insufficient space.
The model needs roughly 32 GiB of available system RAM for experts plus other
loader and OS allocations.

The disk reader and Q2_0 format can be checked without a GPU:

```sh
make -C rdna4/llm test_qwen4_q2_ssd
rdna4/llm/test_qwen4_q2_ssd \
  /mnt/disk1/models/q38nf/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
```

On an RX 9070 XT with about 4.9 GiB already occupied by another workload,
the 256K configuration selected a 1.07 GiB expert cache and completed a
six-token prefill plus two-token decode with 17.3%/29.2% expert-cache hits.
The cached and CPU-only expert runs produced the same sampled sequence hash.
These short-run rates do not predict long-context throughput under a different
GPU load.

## 1K prompt measurement

Run from the repository root:

```sh
rdna4/llm/run_qwen38_flash_next_q2_rocm.sh --bench \
  --prompt-file tmp/qwen38_1k_prompt.txt --prefill-len 1024 -n 1024 \
  --decode 32 -s 1152
```

With `xmrig` also using roughly 29 CPU cores on the Ryzen 9 3950X, the RX
9070 XT completed the 1024-token prefill at 3.28 tok/s and decoded 32 tokens
at 3.35 tok/s. After `xmrig` stopped, the same command reached **7.14
prefill tok/s** and **6.33 decode tok/s**. The clean run spent 9.50 s in CPU
expert work during 143.34 s of prefill, and 0.35 s during 5.06 s of decode;
the scalar GPU/PLE chain now dominates. GPU expert-cache hits were 85.6% /
86.4% for prefill / decode, with 7.97 GiB of cached experts and 14.0 GiB
peak VRAM use. The sequence hash `b9f867f533408c06` matched both earlier
runs. Logs are `tmp/qwen38_q2_1k_final.log` and
`tmp/qwen38_q2_1k_clean.log`.

The scalar prefill measured here had no grouped Q2_0 matrix-matrix path. Strata's
[technical details](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)
describe quantized grouped prompt kernels, streaming experts ahead of the
next layer, and an MTP draft model for speculative decode. Its reported Q2_0
1K prefill is 494 tok/s; 1,308 tok/s is for a 32K prompt. Its 1K decode is
84 tok/s with MTP enabled. This repository has only the two base model GGUF
shards and no matching Flash-Next MTP weights. The 60 decode / 1,200 prefill
targets are not met by this implementation.

An explicit single-tile batched profile is available with the standard
`test_hip_llm` build:

```sh
make -C rdna4/llm test_hip_llm
LLM_BMAX=1024 \
  rdna4/llm/run_qwen38_flash_next_q2_rocm.sh --bench \
  --prompt-file tmp/qwen38_1k_prompt.txt --prefill-len 1024 -n 1024 \
  --decode 32 -s 1152 --qwen4-batched-prefill
```

The opt-in batch path reached **8.97 prefill tok/s** and **6.35 decode
tok/s** on the same 1K prompt, with the scalar sequence hash
`b9f867f533408c06`. Peak VRAM use was 14.65 GiB. The gain is limited because
SSM layers and most attention layers still execute row by row to preserve
the verified exact output. Forcing all layers into the existing batched path
changed the sequence and did not improve prefill speed, so it remains a
diagnostic only. The GPU was on a 304 W power cap and manual performance
level for both measurements.

## MTP and exact-path follow-up

The `shared-Q8_0` NextN sidecar from
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md)
passed the runner's Qwen4 NextN schema inspection and loaded beside this Q2_0
trunk. Exact scalar MTP with a three-token draft preserved the 1K prompt's
sequence hash, accepting most drafts, but measured **5.89 decode tok/s** and
**6.78 prefill tok/s**. Its verifier still executes target tokens sequentially.
The experimental grouped verifier preserved the sequence on a 32-token prompt
but decoded only **3.90 tok/s**. Neither is a faster default for this model.

`LLM_QWEN4_PLE_PROFILE=1` reports the cumulative SSD gather and stream-wait
time every 128 PLE calls. On a 1,024-token scalar prefill, the 16-worker SSD
gather took 610 ms and its preceding stream wait took 293 ms, together only
0.65% of the 139.07 s prefill. That run reached 7.36 prefill tok/s and 7.33
decode tok/s for eight decode tokens. The PLE table is not the current
throughput limiter. During the run, ROCm reported manual performance mode and
memory clock level 0 (96 MHz), even while GPU memory activity was 69%.

The standalone device-memory probe reproduces this clock's bandwidth limit:

```sh
make -C rdna4/llm tmp/bench_hip_mem_bw
LD_LIBRARY_PATH=/opt/rocm/lib rdna4/llm/tmp/bench_hip_mem_bw
```

Four consecutive runs measured 40.77–40.88 GiB/s. After changing the
manual memory level, rerun this probe and the 1K model command with the same
model and context settings to quantify the hardware effect.
On this card, `sudo rocm-smi --setmclk 5` selects the highest listed memory
level while keeping performance mode manual. The benchmark launcher does not
change GPU clock settings.

After the host-side GPU clock reset, the same device-memory probe measured
**568.21 GiB/s** (16 copies of 256 MiB). `rocm-smi` could not report a clock
or performance level at that point, but HIP completed the probe and the model
benchmark. This is 13.9 times the earlier measured device bandwidth.

The Qwen SSD profile now uses the existing vectorized F16 matvec kernel for
its F16 projections. `LLM_QWEN4_F16_LLAMA=0` restores the original kernel;
an explicit `LLM_DECODE_WMMA=1` also retains its own path. On the 1K prompt,
this raised scalar throughput from 7.12/6.28 to **7.53 prefill / 6.85 decode
tok/s** with the same 32-token hash `b9f867f533408c06`. An independent
11-token prompt with 16 forced decode tokens also matched its baseline hash
`fab0d2285407b753` and measured 7.49/7.61 versus 6.86/7.29 tok/s.

With the vectorized F16 path, `LLM_QWEN4_EXACT_GPU_TOPK=1` and
`LLM_QWEN4_EXACT_PRE_GRAPHS=1` raised the same 1K run to **7.81 prefill /
6.97 decode tok/s**, retaining its hash; 47 prefix graphs captured. A
separate 11-token coding prompt with 16 forced decode tokens also retained
its baseline hash `fab0d2285407b753` with both options enabled. A
nonblocking Q2 cache-promotion experiment kept the 32-token hash but did not
improve the short run and lowered cache hits, so the required promotion wait
remains in place.

The fastest parity-checked profile combines the opt-in 1K batch prefill with
the default vectorized F16 kernel and both exact options above. It measured
**10.27 prefill / 7.07 decode tok/s** for 1,024 prompt and 32 decode tokens,
with the same `b9f867f533408c06` hash and 14.65 GiB peak VRAM use. The GPU
still reported 96 MHz memory clock in manual mode. This is the current
throughput result, well below the 1,200/60 tok/s target.
The standard build also passed the same combined profile at 10.38/7.03
tok/s and the identical hash, so a separate HIPBLASLt build is unnecessary.

After the clock reset, that exact standard-build command reached **31.93
prefill / 27.60 decode tok/s**, with the same 32-token hash
`b9f867f533408c06`, 14.646 GiB peak VRAM, and `Result: PASS`. The prefill
processed 1,024 tokens in 32.072 s; decode generated 32 tokens in 1.159 s.
The MoE cache hit rate was 85.6% during prefill and 86.6% during decode, with
35.77 GiB and 1.13 GiB of host-to-device expert transfers respectively.
The clock reset improved the two rates by 3.08 and 3.93 times, but the
1,200/60 tok/s targets remain unmet. The current Qwen4 attention batch cap
is layer 2, whereas the first attention layer in this model is layer 3; the
SSM batch gate is also off. Thus the layer bodies still run token by token.
The batch path amortizes MoE work, but batching the layer bodies requires
numerical parity work before it can replace this path.
`LLM_DECODE_WMMA=1` on the reset clock measured 30.44/27.03 tok/s and kept
the same hash, so it is not a speed improvement for this workload. Logs are
`tmp/qwen38_q2_stock_batch_gpu_reset1024.log` and
`tmp/qwen38_q2_stock_batch_gpu_reset_wmma1024.log`.

An extended bandwidth probe measured 26.36 GiB/s for 256 MiB pinned H2D
copies, but expert-sized 512 KiB copies reached 20.58 GiB/s pinned and
14.09 GiB/s pageable. The exact 1K run copied 35.77 GiB of experts, which
would take roughly 2.5 s at the measured pageable rate if serialized. The
1200 tok/s prefill goal allows only 0.85 s for the whole 1K prompt, so the
current expert traffic alone exceeds that budget. Kernel tracing of a
128-token prompt plus eight decode tokens recorded 245,530 dispatches, with
`matvec_f16_llama_f32` the largest GPU kernel cost. The trace includes setup;
it is evidence of the scalar launch count, not a phase-isolated timing.

`LLM_QWEN4_Q2_CACHE_PROFILE=1` selects an opt-in expert-cache allocation
derived from replaying the exact 1K prompt's routed IDs. The replay matched
the runner's 420,817 cache hits at the existing 6,190-slot allocation and
projected 427,102 hits with the same slot total. The measured run reached
**32.58 prefill / 28.60 decode tok/s**, retained hash `b9f867f533408c06`,
and used 14.664 GiB peak VRAM. Hits rose to 426,977 (86.9%) but H2D was
still 35.68 GiB: one cold expert is refilled on almost every layer/token.
This prompt-specific allocation remains opt-in because its benefit on other
requests has not been established. A separate prefill-balance control was
slower at 31.40/26.56 tok/s with 83.9% prefill hits.

`LLM_MOE_CPU_DECODE_REFILLS_PER_LAYER=0` keeps the prefill's resident cache
fixed during decode and evaluates every cold route on the CPU. With the Q2
cache profile above, two exact 1K/32 runs measured **31.06 and 31.75 decode
tok/s**, both with hash `b9f867f533408c06`; decode expert H2D fell from
1.17 GiB to zero. The second run reached 31.94 prefill tok/s, 85.3% decode
cache hits, and 14.664 GiB peak VRAM. The switch remains opt-in because a
longer continuation may churn away from the fixed resident set. At a 9,000
MiB cache budget (`--moe-cache-mb 9000`) it reached 33.37 prefill / 30.80
decode tok/s and used 15.468 GiB peak VRAM, leaving 836 MiB free.

The diagnostic Q2_0 batch path now recognizes Q2_0 projections and supports
cache layers with more than 128 slots. Q5_0 and IQ4_NL BF16 dequantizers let
the wider batched body run. Batching through layer 3 reached 32.59 prefill
tok/s on 1K but changed the output hash and stopped after 18 decoded tokens.
On a 128-token logit comparator, the layer-3 path had relative L2 difference
0.345 from scalar. Scalar MoE reduced it to 0.058, and scalar projections to
0.048; wider batching also changed the hash. These gates remain diagnostic.

### Grouped Q2_0 prefill diagnostic after the clock reset

The opt-in staged path now has native Q2_0 grouped gate/up and down kernels.
It groups routed assignments by expert and copies cold experts through two
staging banks. A bounded arithmetic check compares one staged assignment in
each layer against the existing selected-expert Q2_0 kernels, including the
default buffer reuse:

```sh
LLM_BMAX=128 LLM_QWEN4_BATCH_SSM=1 \
  LLM_QWEN4_BATCH_ATTN_MAX_LAYER=47 LLM_QWEN4_BATCH_PLE_FFN=1 \
  LLM_MOE_COPY_PIPELINE=1 LLM_QWEN4_Q2_STAGE_CHECK=1 \
  rdna4/llm/run_qwen38_flash_next_q2_rocm.sh --bench \
  --prompt-file tmp/qwen38_1k_prompt.txt --prefill-len 128 -n 128 \
  --decode 0 -s 256 --qwen4-batched-prefill \
  --qwen4-prefill-staging --qwen4-prefill-stage-mb 512
```

All 48 checked assignments matched bit for bit for both the 640 gate/up
outputs and 2,560 down outputs. The 1K diagnostic reached **157.80 prefill
tok/s** with 256-thread grouped kernels and the copy pipeline, moving 20.35
GiB of experts in 143 waves. Peak VRAM was 15,190 MiB. The 128- and
512-thread variants reached 151.58 and 145.06 tok/s, respectively. The
corresponding logs are in `tmp/qwen38_q2_reset_q2stage_pipeline*.log` and
`tmp/qwen38_q2_reset_stage_check_alias128.log`.

The full batched layer path changes the output, so these rates are diagnostic.
Its 1K prefill profile spent about 0.72 s in the scalar PLE state work and
roughly 0.11–0.16 s in each batched MoE FFN. Batching only the PLE layer's
MoE produced the exact 1K/32 sequence when both
`LLM_QWEN4_BATCH_ROUTER_SCALAR=1` and
`LLM_QWEN4_BATCH_HC_SCALAR=1` were set, but reached only 30.30 prefill /
25.98 decode tok/s. Without those precision controls it changed the sequence.
The exact scalar path remains the default. Disabling prefix graphs did not
improve 32-token decode: it reached 31.69 tok/s at a 9,000 MiB cache budget,
with the exact hash `b9f867f533408c06`.

### Ordered-state 1K prefill with grouped MoE

The opt-in `LLM_QWEN4_BATCH_SCALAR_STATE_FFN=1` path runs every attention,
SSM, and PLE state transition in token order, then batches each layer's MoE
FFN. The F16 HC batch and bounded router tile retain the scalar reference's
greedy sequence on the measured 1K prompt. The
Q2_0 stage can group up to eight assignments for one expert into a weight-
reuse tile; the per-layer selected-expert check matched all 48 sampled gate/up
and down outputs bit for bit with an eight-assignment tile. A four-assignment
tile measured best in the exact 1K/32 run:

```sh
LLM_BMAX=1024 LLM_QWEN4_BATCH_ATTN=0 \
  LLM_QWEN4_BATCH_SCALAR_STATE_FFN=1 \
  LLM_QWEN4_BATCH_ROUTER_SCALAR=1 LLM_QWEN4_BATCH_ROUTER_TILE=32 \
  LLM_QWEN4_BATCH_HC_EXACT_F16=1 LLM_QWEN4_BATCH_HC_PREMIX=1 \
  LLM_MOE_COPY_PIPELINE=1 LLM_QWEN4_Q2_STAGE_TILE_TASKS=4 \
  LLM_QWEN4_STAGE_PROMOTE=1 LLM_MOE_CPU_DECODE_REFILLS_PER_LAYER=0 \
  rdna4/llm/run_qwen38_flash_next_q2_rocm.sh --bench \
  --prompt-file tmp/qwen38_1k_prompt.txt --prefill-len 1024 -n 1024 \
  --decode 32 -s 1152 --moe-cache-mb 9000 --qwen4-batched-prefill \
  --qwen4-prefill-staging --qwen4-prefill-stage-mb 512 --qwen4-kv-quant none
```

This reached **70.62 prefill / 32.65 decode tok/s**, with hash
`b9f867f533408c06` and 15,872 MiB peak VRAM (432 MiB free). The cache
starts empty; `LLM_QWEN4_STAGE_PROMOTE=1` uses observed routes to retain
experts for decode. Without promotion, decode fell to 18.11 tok/s. Batched
F16 HC projections and layer-entry premixing keep the scalar F16 matvec
reduction order, while 32-row router groups retain the single-row WMMA tile.
All 248,320 logits matched bit for bit against the earlier staged path
at both 128 and 1,024 prompt tokens. This compares two versions of the
grouped MoE path, not grouped MoE against scalar MoE. With the same 9,000 MiB
cache budget, grouped MoE versus scalar prefill differed by 1.55% relative
logit L2 at 512 tokens and 5.03% at 1,024 tokens. At 128 tokens, using scalar MoE with the batched F16
HC path matched all scalar logits bit for bit; grouped MoE with scalar HC
still differed by 2.49% relative L2. The grouped route therefore remains an
opt-in quality experiment despite the matching 1K greedy sequence. In paired
1K grouped-MoE runs, these changes raised
prefill from 55.57 to 68.55 tok/s; the 70.62 result above used the same
settings with 32 decoded tokens. A 512+512 streamed run changed the 32-token
greedy hash to `5d1fa86db31017c9` and reached 42.01/33.60 tok/s, so this
grouped-MoE profile is not validated across an external chunk boundary. The
same streamed schedule with scalar MoE matched all scalar 1K logits bit for
bit, which isolates the difference to grouped MoE arithmetic rather than
the batched-to-scalar state handoff.

For numerical comparisons, use the same cache budget. Scalar and batched-HC
1K logits matched bit for bit at 9,000 MiB when both used scalar MoE. A scalar
run with a 10,200 MiB cache differed from the 9,000 MiB scalar run by 2.97%
relative logit L2 because different experts executed on CPU versus GPU.
The cache budget is therefore part of the numerical configuration.

The parity-preserving 1K batched-HC/scalar-MoE profile reached **38.60 prefill
/ 33.49 decode tok/s** at 9,000 MiB, compared with **30.86 prefill tok/s**
for scalar prefill at the same cache budget. It uses
`LLM_QWEN4_BATCH_SCALAR_STATE_FFN=1`, `LLM_QWEN4_BATCH_ATTN=0`,
`LLM_QWEN4_BATCH_HC_EXACT_F16=1`, `LLM_QWEN4_BATCH_HC_PREMIX=1`,
`LLM_MOE_PREFILL_SCALAR=1`, `LLM_BMAX=1024`, and
`--qwen4-batched-prefill`, without staging.

An optional `LLM_QWEN4_Q2_EXPERT_PROFILE=/path/to/expert-profile.bin` accepts
the model-matched STRP ranking format and preloads cache slots at model load.
In the full-batch diagnostic, preload plus an eight-assignment tile reduced
1K expert H2D from 20.35 to 14.30 GiB and reached 167.42 tok/s, but that
full-batch path still changes the generated sequence. With the exact split,
static preload did not improve the combined 1K/32 result enough to recommend
it over route-based promotion. A 10,000 MiB cache plus 1K batched buffers
exceeded 16 GiB VRAM; 9,500 MiB left only 24 MiB free and was slower.

The 1,200/60 tok/s goal remains unmet. In the full-batch 1K GPU trace,
grouped Q2_0 gate/up and down consumed about 2.33 s together, expert H2D
about 1.69 s, and attention about 0.49 s. Weight reuse lowered grouped
kernel time in the profiler but did not remove the other costs. The exact
decode trace attributed about 2.3 ms/token to the selected Q2_0 kernels,
5.4 ms/token to F16 matvecs, and 3.0 ms/token to full attention; these are
GPU times and exclude host gaps. Reaching the target requires a larger
change to state/attention execution and the quantized expert matrix path.

### Fastest exact decode profile measured so far

At a 1,152-token maximum context, F16 KV needs only about 30 MiB more VRAM
than scaled INT8 KV and preserves the same 1K/32 greedy sequence. The
following opt-in profile reached **33.85 prefill / 36.68 decode tok/s** with
hash `b9f867f533408c06`, 91.2% decode cache hits, and **16,004 MiB peak
VRAM** (300 MiB free):

```sh
LLM_QWEN4_Q2_CACHE_PROFILE=1 LLM_MOE_CPU_DECODE_REFILLS_PER_LAYER=0 \
  LLM_QWEN4_EXACT_GPU_TOPK=1 \
  rdna4/llm/run_qwen38_flash_next_q2_rocm.sh --bench \
  --prompt-file tmp/qwen38_1k_prompt.txt --prefill-len 1024 -n 1024 \
  --decode 32 -s 1152 --moe-cache-mb 10200 --qwen4-kv-quant none
```

The 10,200 MiB cache leaves little room for other GPU processes. F16 KV is
appropriate only for this bounded context profile; the launcher retains
scaled INT8 KV for its 262K default. Eight CPU threads measured 34.39 decode
tok/s and 117.71 ms in CPU miss work, versus 16 threads' 35.06 tok/s and
97.54 ms in the earlier control. HC graphs measured 33.80 tok/s with the 9,000 MiB control and
did not help.

Exact MTP with the matching local Q8_0 NextN sidecar accepted most
three-token drafts but reached 29.03 tok/s with scalar target verification
and 23.38 tok/s with the window verifier; both retained the same sequence.
The current window verifier still replays enough scalar work that speculation
does not close the 60 tok/s target. The detailed phase trace is enabled by
`LLM_QWEN4_PROFILE_DECODE_PHASES=1` with `LLM_GRAPH_DISABLE=1`; it found
roughly 18.9 ms across state/attention phases, 14.2 ms across MoE phases,
and 1.3 ms across HC combines per profiled decode token. This trace adds
stream barriers and is diagnostic rather than an end-to-end latency sum.

```sh
LLM_BMAX=1024 LLM_QWEN4_EXACT_GPU_TOPK=1 \
  LLM_QWEN4_EXACT_PRE_GRAPHS=1 \
  rdna4/llm/run_qwen38_flash_next_q2_rocm.sh --bench \
  --prompt-file tmp/qwen38_1k_prompt.txt --prefill-len 1024 -n 1024 \
  --decode 32 -s 1152 --qwen4-batched-prefill
```

Keep the launcher's default `OMP_NUM_THREADS=16` on the Ryzen 9 3950X.
At 32 prompt + 8 decode tokens, an eight-thread control measured 7.39/8.20
tok/s, and 32 threads fell to 2.00/2.05 tok/s while preserving the output
hash. The 32-thread run spent 11.76 s in CPU expert work during 15.96 s of
prefill, showing severe oversubscription.

`rocprofv3 --kernel-trace --stats` on the vectorized F16 32+8 run recorded
73,695 GPU dispatches and 3.91 s of kernel time during a 5.56 s measured
request. F16 matvec was largest (11,817 calls, 1.24 s), followed by Q3_K
matvec (0.49 s), IQ4_XS matvec (0.45 s), and F32 matvec (0.34 s). A
128-thread F16 launch retained the hash but was slower than 256 threads.
The trace and PLE profile point to the scalar GPU projection chain and its
memory traffic as the remaining bottleneck under the observed clock setting.

Combining exact GPU router top-k, 47 captured prefix graphs, and WMMA decode
also preserved the 1K sequence hash. That run measured **7.29 prefill tok/s**
and **6.47 decode tok/s**. The gain is small enough to leave these diagnostic
switches explicit. Logs are `tmp/qwen38_q2_mtp1024_32.log`,
`tmp/qwen38_q2_mtp_window32_8.log`, and
`tmp/qwen38_q2_1k_exact_tuned_combo.log`.
