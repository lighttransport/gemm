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

The present scalar prefill has no grouped Q2_0 matrix-matrix path. Strata's
[technical details](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)
describe quantized grouped prompt kernels, streaming experts ahead of the
next layer, and an MTP draft model for speculative decode. Its reported Q2_0
1K prefill is 494 tok/s; 1,308 tok/s is for a 32K prompt. Its 1K decode is
84 tok/s with MTP enabled. This repository has only the two base model GGUF
shards and no matching Flash-Next MTP weights. The 60 decode / 1,200 prefill
targets are not met by this implementation.

An explicit single-tile batched profile is available with a build that enables
the batch dispatcher:

```sh
make -C rdna4/llm TARGET=tmp/test_hip_llm_batch HIPBLASLT=1 \
  ROCM_LIB=/opt/rocm/lib tmp/test_hip_llm_batch
QWEN38_Q2_RUNNER=rdna4/llm/tmp/test_hip_llm_batch LLM_BMAX=1024 \
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
nonblocking Q2 cache-promotion experiment kept the 32-token hash but did not
improve the short run and lowered cache hits, so the required promotion wait
remains in place.

The fastest parity-checked profile combines the opt-in 1K batch prefill with
the default vectorized F16 kernel and both exact options above. It measured
**10.27 prefill / 7.07 decode tok/s** for 1,024 prompt and 32 decode tokens,
with the same `b9f867f533408c06` hash and 14.65 GiB peak VRAM use. The GPU
still reported 96 MHz memory clock in manual mode. This is the current
throughput result, well below the 1,200/60 tok/s target.

Combining exact GPU router top-k, 47 captured prefix graphs, and WMMA decode
also preserved the 1K sequence hash. That run measured **7.29 prefill tok/s**
and **6.47 decode tok/s**. The gain is small enough to leave these diagnostic
switches explicit. Logs are `tmp/qwen38_q2_mtp1024_32.log`,
`tmp/qwen38_q2_mtp_window32_8.log`, and
`tmp/qwen38_q2_1k_exact_tuned_combo.log`.
