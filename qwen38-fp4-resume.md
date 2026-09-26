# Qwen3.8-27B NVFP4 on one Fugaku A64FX node: resume note

## Goal and correctness contract

Reach **40+ emitted tokens/s end to end** for a single request using the
Qwen3.8-27B NVFP4 model on **one** 48-core A64FX node. The user explicitly
allows approximate computation in the timed path and an **untimed, full,
unapproximated serial target replay** afterward. Compare every emitted
`(position, token ID)`; a spot check or only comparing final text is
insufficient. Do not call an approximate verifier exact. Keep prompt, model,
KV dtype, tokenizer, and generation options aligned between timed and replay
runs. The 40+ target has **not** been met.

## Current state (2026-09-24 JST)

- Checkout: local `/mnt/nvme02/work/gemm/qwen38-27b`; Fugaku
  `~/work/gemm/qwen38-27b`. Last committed work is `2d1403b7` (integer
  coefficient experiments). There are **uncommitted** edits to
  `a64fx/llm/Makefile`, `a64fx/llm/qwen38_runner.c`,
  `common/transformer.h`, and new
  `a64fx/llm/qwen38_nvfp4_i16_super.c`. Scratch work under `tmp/q27b-*`
  is untracked. Inspect `git status` before modifying anything.
- Model: `~/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf` (check
  actual storage path before staging); allocation-local staged copy is
  `/local/u14346/qwen38-nvfp4/Qwen3.8-27B-NVFP4-Quality-v2.gguf`,
  16,058,504,512 bytes. `/local` is wiped when an allocation ends.
- The last observed compute allocation was PJM `51877011` on `a25-0103c`,
  started about 2026-09-23 14:16 UTC for six hours. **Check whether it is
  still alive before use**; do not assume its staged model or bridge survives.
  Local bridge port was 42573. The existing bridge client is
  `tmp/q27b-roofline/remote.py`; send command lines to its stdin. Follow
  `a64fx/remote-dev-procedure.md` and project-local connection config for
  login1/login3 forwarding and job relaunch. Do not use `/tmp`; use `/local`
  on the node or repo `tmp/`.
- Current validated setup uses `a64fx/llm/run_qwen38_nvfp4_cmg4.sh` with
  `TF_KV_DTYPE=f32`, `--nvfp4-exact-tiled --nextn-exact-tiled
  --q6-exact-head --spec-k 3 --spec-verify --draft-head-rows 65536`. Set
  `TF_DUMP_TOKENS=1` for replay comparison. The NextN persistence settings
  used in recent runs are listed below. The timed path has 32 experimental
  approximate gate sidecars via `--i16-super-gates 32`; the serial replay
  must omit this flag and use `--spec-k 0`.

## Performance and roofline evidence

The full inventory and experiment history are in
`a64fx/llm/qwen38_nvfp4_roofline.md`. Active compact trunk + vocabulary
payload is **14.754 GB/token**, giving an optimistic 830 GB/s STREAM-based
floor of **17.8 ms/token** before state/draft/dispatch. A 40 tok/s target
allows **25 ms per emitted token**. The current exact tiled setup retains the
compact NVFP4 trunk and has a separate 1.589 GB predecoded Q6 head; the
expanded packed format would stream 19.283 GB of trunk per token and is a
poor fit for the target. Isolated packed-weight read scan reached about
509 GB/s, whereas compact exact N=3 gate computation reached only about
0.50 ms for a 50.135 MB matrix (~100 GB/s source rate). A64FX PMU data
showed FP/execution completion waits dominating load waits. The compact
N=3 kernel is **decode/arithmetic limited**, not already HBM limited.

The isolated experimental K-major 64-row signed-byte coefficient sidecar
(`tmp/q27b-roofline/bench_coeff_supertile16.c`) quantizes each of three
activations to INT16, uses two SVE `SDOT`s per candidate for low/high bytes,
and handles rare larger UE4M3 scales via FP32 corrections. It expands a
50.135 MB gate to **89.129 MB** but reached **0.279-0.303 ms** in a warm
microbenchmark with active OpenMP workers, roughly 300+ GB/s physical
stream rate. Synthetic relative L2 output error was `1.37653e-05`, maximum
absolute difference `4.42564e-06`. The approximate INT8 variant reached
~0.18 ms but had much larger relative L2 error (~0.011). These isolated
rates did **not** translate into a material end-to-end gain.

With 32 gate sidecars on the matrix-multiplication prompt, 32 tokens took
**2.794 s = 11.453 tok/s**, 13 verification rounds, 20/26 proposed draft
positions accepted. Projection profile: total 553.5 ms, output 163.1 ms,
FFN gate/up **820.7 ms**, FFN down **426.6 ms**. A comparable exact K=3 run
was **10.995 tok/s** and FFN gate/up ~843.9 ms: only ~23 ms of aggregate
gate/up saving is supported, while run-to-run draft/attention variation
affects the headline rate. `KMP_BLOCKTIME=5` made the full run worse:
**11.059 tok/s**, FFN gate/up 821.1 ms. Global `OMP_WAIT_POLICY=ACTIVE`
also hurt NextN draft and is not a solution.

An independent 64-token `hi` run with 32 sidecars finished at
**6.380 s = 10.031 tok/s**, 29 rounds, 35/58 draft positions accepted.
Profile: target verifier 5223.6 ms, draft 951.9 ms, commit 190.0 ms;
projection 1234.6 ms, output 363.3 ms, FFN gate/up 1831.4 ms, FFN down
952.5 ms. Its log is `/local/u14346/q27b-i16super32-hi64.log` if that
allocation remains alive. **A fresh exact serial replay of this `hi` trace
has not yet been run or compared.** The earlier 32-token
matrix-multiplication trace did match all 32 `(position, ID)` pairs against
both an existing and a fresh unapproximated serial run. That is the current
end-to-end token-identity evidence for the new sidecar.

## Experimental code and immediate cleanup

`a64fx/llm/qwen38_nvfp4_i16_super.c` packs the first N FFN gate matrices
into 64-output, K-major signed-byte sidecars. For UE4M3 scale bytes `d<=10`,
`FP4_code*d` is exactly representable as an INT8 coefficient multiplied by
`2^-10`; larger scales use sparse full-FP32 correction records. The original
compact weights remain resident for serial and draft paths. Runner flag
`--i16-super-gates N` is opt-in and currently restricted to exact tiling,
K=3 spec verify. `common/transformer.h` dispatches the sidecar only for
registered N=3 target projections. 32 gates add about **2.85 GB** of weight
storage. Avoid blindly packing all 64 gates on a 32 GB HBM node; monitor
`MemAvailable`, and guard against dropping below 6 GB.

Before treating this as production code or committing:

1. Run fresh **64-token unapproximated serial replay** of `hi` with F32 KV,
   exact tiled trunk/NextN and exact Q6 head; compare all `(position, ID)`
   pairs by regex. Check that the existing 64-token sidecar log is complete.
2. Review accumulation overflow. Current SVE `SDOT` accumulators are INT32
   across full K, and the correction `z + 128*weight_sum` is also INT32.
   An adversarial activation/weight pattern could overflow. Add a bounded
   reduction or a justified bound and test cancellation/large activations.
3. Fix portability: `<omp.h>` and `omp_get_*` are unconditional in the new
   file although default non-OpenMP builds may use the same Makefile. Check
   `-Wall -Wextra -Wpedantic`, sequential build, and Fujitsu OpenMP build.
4. Measure a **real loaded model matrix** output against the compact exact
   kernel with error metrics, not only synthetic matrix error and final IDs.
   Add an appropriate `test_*.c` or bounded probe. Verify rare-scale
   handling, packing indices, row tails/shape contracts, and allocation
   failure paths. Re-sync the local file after cleanup; the remote copy may
   still have a redundant `_GNU_SOURCE` definition warning.
5. Update `a64fx/llm/qwen38_nvfp4_roofline.md` with the sidecar experiment,
   exact commands, identity results, and current limits. Run `git diff
   --check` and relevant builds/tests. Commit a coherent validated unit;
   report the hash. **Do not push** without an explicit per-action push
   request (AGENTS.md).

## Reproduction on an active allocation

Deploy changed sources with targeted `rsync -av` to
`fugaku3:work/gemm/qwen38-27b/`; build with
`make -C a64fx/llm qwen38_runner CC=fcc OPENMP=1`. The full run shape:

```sh
cd ~/work/gemm/qwen38-27b
export TMPDIR=/local/u14346/tmp TF_KV_DTYPE=f32 TF_DUMP_TOKENS=1
export TF_NEXTN_FFN_PERSIST=1 TF_NEXTN_BLOCK_PERSIST=1
export TF_NEXTN_FULL_PERSIST=1 TF_NEXTN_ATTN_BLOCK_PERSIST=1
export TF_NEXTN_QKV_PERSIST=1 TF_NEXTN_INLINE_ARGMAX=1
M=/local/u14346/qwen38-nvfp4/Qwen3.8-27B-NVFP4-Quality-v2.gguf
a64fx/llm/run_qwen38_nvfp4_cmg4.sh "$M" --prompt hi \
  --max-seq 256 --max-gen 64 --spec-k 3 --spec-verify \
  --draft-head-rows 65536 --nvfp4-exact-tiled \
  --nextn-exact-tiled --q6-exact-head --i16-super-gates 32 \
  > /local/u14346/q27b-i16super32-hi64.log 2>&1
```

For the serial replay, keep model/prompt/sequence/KV/tiling/Q6 settings;
replace the speculative flags with `--spec-k 0`, omit
`--i16-super-gates`, and log to a new file. Parse using
`re.finditer(r'qwen38: token n=(\d+) pos=(\d+) id=(\d+)', text)` because
decoded text and log records can share lines. Assert equal length and every
position/ID pair, and record any first divergence.

## Fresh research and kernel redesign prompt

> Continue work on single-node Fugaku A64FX Qwen3.8-27B NVFP4 decode.
> Target **40+ end-to-end emitted tokens/s** for one request. The timed path
> may approximate, but run a separate **full, unapproximated serial target
> replay** and prove that *every* emitted `(position, token ID)` matches.
> First inspect this note, `a64fx/llm/qwen38_nvfp4_roofline.md`, current
> source/diff, remote development procedure, and live job/bridge state.
> Do not assume an isolated microbenchmark predicts full-model speed.
>
> Start from a measured per-stage roofline for **K=3 target verification**:
> actual bytes read, effective GB/s, arithmetic/decode work, OpenMP wake
> overhead, CMG locality, NextN draft cost, Q6 head, verification-round
> count, and 25 ms/emitted-token budget. Use hardware counters or compiler
> assembly when deciding whether each stage is bandwidth, instruction,
> latency, or synchronization limited. Research A64FX SVE/SVE2 capability
> and current fast low-bit GEMV/GEMM designs from primary sources as needed.
> In particular, redesign **the whole dominant projection path** rather
> than further tweaking only the first 32 FFN gates: gate/up, down, SSM
> QKV/output, attention projections, and head. Explore compact fused
> nibble+scale decode versus bounded-memory K-major signed-byte/INT16
> formats, scalable per-CMG tiling, persistent worker teams, and ways to
> avoid duplicate resident weight copies. Quantify whether each design can
> fit 32 GB HBM and reach the required bytes/s and rounds/token.
>
> Implement the highest-leverage design, test matrix error against the
> real exact kernel, then run multiple prompt traces and full serial token
> replay. Report measured end-to-end speed, stage times, acceptance,
> memory headroom, and exact ID comparisons. If the 40+ budget is
> physically or algorithmically out of reach under these constraints,
> give a quantified bound and best measured result instead of claiming
> success. Preserve work in a focused commit; do not push without a new
> explicit user request.
