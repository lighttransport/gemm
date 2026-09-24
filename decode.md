# Qwen3.8-27B FP4 decode on A64FX: remaining work and resume prompt

Updated: 2026-09-24 JST. Implementation baseline: `8cb9f2c9`
(`Add bandwidth-limited 6-bit FP4 execution layout`).

## Objective and current boundary

Reach **40+ end-to-end tokens/s for one decode stream on one 48-core A64FX
node**, with validated output correctness. The user allows weight repacking
and 5-/6-bit payload expansion, performed while loading weights staged in
`/local`. Establish bandwidth behavior in local qlair and collect measurements
and profiles on actual A64FX. Use interactive PJM allocations by default.

**Achieved:** a synthetic fused dequantization + W4A8 projection kernel is
bandwidth-limited on native A64FX. **Not achieved:** the simulator's 200 GB/s
gate, real-model validation/integration of this layout, or 40 tok/s decode.
The current new kernel is N=1; historical K=3 speculative-verifier results
are a different workload and must not be substituted for serial decode.

## Verified checkpoint

See [the full layout study](a64fx/llm/qwen38_nvfp4_expanded.md) for build/run
commands, tables, counter interpretation, and limitations.

| Execution layout | Tile bytes, including scales | Total bits/weight | Native GB/s per CMG | Time, 6144 x 5120 |
| --- | ---: | ---: | ---: | ---: |
| Compact integer-scale FP4 | 352 | 5.5 | 147.167 | 146.954 us |
| Signed 5-bit payload | 416 | 6.5 | 181.912 | 140.502 us |
| Half-predecoded 6-bit payload | 480 | 7.5 | 218.995 | 134.666 us |

- A tile contains 8 output rows x 64 input columns. The 6-bit payload
  expands half the FP4 codes into signed bytes, eliminating half the lookups.
  It costs 36.4% more bytes than the 352-byte compact representation and
  reduces projection time by 6–9%; bandwidth alone overstates the speedup.
- Four CMGs reach **852–859 GB/s**, 97–99% of matched scan bandwidth, on
  17408 x 5120 and 15360 x 5120. Every row passes the synthetic checks.
- FAPP PA17 confirms **220.57 GB/s** HBM reads for 6-bit compute versus
  226.26 GB/s for its scan. These are CMG-wide counters observed in
  overlapping worker windows: **do not sum the 12 worker readings**.
- Fixed a placement bug: pin initialization to the owning CMG **before
  allocation**, then bind/touch and verify physical page nodes. Successful
  `mbind` without migration did not move already populated pages. Misplaced
  weights reduced a scan from about 225 to 120 GB/s.
- Randomized tests pass on QEMU and native Fujitsu builds. Expanded outputs
  equal the compact integer-scale kernel bitwise; source-reference tests use
  integer-ratio scales. This does **not** validate arbitrary real scales.
- The repacks retain the existing integer-scale path's scale-ratio rounding.
  Their input is the existing **384-byte intermediate tile**, not raw GGUF
  bytes. An actual GGUF loader adapter is still needed.

## Files and artifacts

- Kernels/repack: `a64fx/llm/qwen38_nvfp4_expanded_a8.c`;
  baseline: `a64fx/llm/qwen38_nvfp4_packed8_iscale_a8.c`.
- Correctness: `a64fx/llm/test_qwen38_nvfp4_expanded_a8.c`.
- Benchmark: `a64fx/llm/bench_qwen38_nvfp4_qlair.c`.
  Modes: `packed8_iscale1`, `packed5_1`, `packed6_1`, and matching
  `_stream` controls. Enable `Q38_QLAIR_CHECK_ALL=1` and native
  `Q38_QLAIR_DIAG_PLACEMENT=1`.
- Make targets in `a64fx/llm/Makefile`: `qwen38_nvfp4_layout_bench`,
  `qwen38_nvfp4_expanded_test`; `Q38_FAPP=1` builds a separate
  `bench_qwen38_nvfp4_qlair_fapp` executable.
- Native logs, assembly, raw FAPP CSVs, and Makefile verification:
  `tmp/q38-expanded-20260924/hw-51891531-local/`.
  `hw-51891531/` contains earlier remote-placement measurements; do not mix
  those with the corrected local-HBM baseline.
- Simulator logs/binaries: `tmp/q38-expanded-20260924/`.
  These scratch artifacts are untracked; preserve them or explicitly archive
  the required evidence before cleaning anything.
- Background: [roofline/history](a64fx/llm/qwen38_nvfp4_roofline.md),
  [remote procedure](a64fx/remote-dev-procedure.md), and
  [older resume note](qwen38-fp4-resume.md). The older note's commit,
  uncommitted-work list, active allocation, and bridge state are stale.
  `a64fx/llm/decode.md` concerns GLM-5.1 and is unrelated to this handoff.

## Remaining items, in order

1. **Resolve the simulator discrepancy.** Use
   `~/work/clair/a64fx/build-inference/qlair` in `~/work/clair/a64fx`.
   Clang kernels currently predict compact/5-bit/6-bit throughput of
   105.95/116.23/134.63 GB/s; the 6-bit scan predicts 227.58 GB/s.
   Imported Fujitsu compute assembly with Clang setup passes numerical
   checks but predicts only 30.42 GB/s for 6-bit (43.05 for compact).
   Full Fujitsu setup/repack assembly fails earlier in emulation. Minimize
   execution failures separately from timing discrepancies; compare actual
   instruction counts, scheduling, spills, and memory behavior. The 5-bit
   leaf's single-register pre/post-index stack save/restore also caused
   repeated calls in qlair; `-fno-omit-frame-pointer
   -mno-omit-leaf-frame-pointer` was a working workaround. Add focused
   regressions before any simulator fix. Do not retune the model merely to
   force 200 GB/s, or present hardware results as simulator results.

2. **Validate real weights before enabling the execution layout.** Confirm
   the model path and tensor inventory. Probe bounded slices from multiple
   projection types/layers against the unapproximated source kernel. Measure
   scale-ratio rounding error, rejected/large ratios, relative L2 and maximum
   absolute output error, activation-quantization error, and worst-case
   integer accumulation bounds. Cover zero scales, all codes/signs,
   activation extremes, shape contracts, and invalid inputs. Keep an exact
   fallback for unsupported tensors/scales; do not label W4A8 or rounded
   scale multipliers exact simply because synthetic tests pass.

3. **Inventory memory and implement bounded load-time repacking.** Compute
   exact per-tensor final bytes, metadata, head/NextN weights, KV/state,
   scratch, and peak staging memory before expanding the whole model.
   Compare 352/416/480-byte choices by latency and resident bytes. Read
   staged `/local` GGUF data in bounded chunks, convert through bounded
   intermediate tiles, and write directly to final CMG-local anonymous
   storage. Drop source page cache as it is consumed. Do not retain two
   full resident models or a full expanded intermediate. Handle errors and
   partial allocations safely; verify sampled page placement and monitor
   `MemAvailable` with a conservative 6 GB floor for new full-model trials.

4. **Integrate an opt-in N=1 decode path.** Inspect `qwen38_runner.c`,
   `common/transformer.h`, and existing tensor dispatch before editing.
   Start with a measured real projection, then cover the dominant gate/up,
   down, attention/SSM projections as supported. Reuse persistent workers,
   preserve CMG ownership, and avoid runtime repacks and per-token
   dequantization buffers. Measure activation preparation, synchronization,
   and reductions as well as the fused kernel. Preserve existing reference
   paths and keep speculative N>1 dispatch distinct.

5. **Measure actual serial decode and correctness.** Re-establish current
   exact FP4 and available FP8 baselines with the same prompt/context,
   generation settings, thread placement, and timing boundary. Prior
   ~11 tok/s FP4 versus user-reported 20+ tok/s FP8 is not a controlled
   comparison of this new kernel. Profile target projections, attention/SSM,
   head, dispatch, and memory traffic. Report serial N=1 tokens/s separately
   from speculative emitted tokens/s. Use multiple prompt traces and enough
   generated tokens for stable repeated measurements. If the timed path
   approximates, run an untimed **full unapproximated serial replay**, compare
   every emitted `(position, token ID)`, and report the first divergence;
   matrix tolerance or matching final text alone is insufficient.

6. **Close the 25 ms/token budget using measured stages.** Recompute the
   full-model byte/latency budget after layout selection; 852–859 GB/s from
   synthetic matrices does not establish a full-model token rate. Optimize
   remaining dominant work and only then evaluate optional NextN/speculative
   paths as separately labeled experiments. Deliver exact commands, repeated
   timing, memory peak, FAPP evidence, and correctness results. If 40 tok/s is
   not reached, report the measured rate and quantified limiting stages.

## Remote restart and safety

The measured interactive allocation was **51891531**, node **a27-0006c**,
`freq=2000,eco_state=0`; it was explicitly released at the end of the kernel
study. There is no reusable shell from that job. Check `pjstat` before a new
allocation and follow the interactive-default procedure through `ssh -tt
fugaku1`. Do not assume allocation-local files survive.

Local checkout: `/mnt/nvme02/work/gemm/qwen38-27b`.
Remote checkout: `~/work/gemm/qwen38-27b`.
Previous source bundle and shared reports:
`~/work/gemm/qwen38-27b/tmp/q38-expanded-20260924/`.
Historical model source (verify):
`~/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf`.

Use targeted `rsync`, stage large weights in bounded chunks with writeback,
and set build temporaries under `/local` on the node or repository `tmp/`
locally. Never use `/tmp`. Keep long/full-model trials detached within an
appropriate allocation so memory pressure cannot strand the control shell.
Copy results to shared storage before releasing the allocation. Preserve
unrelated work; no `git push` without a new explicit per-action request.

## Copy-ready resuming prompt

```text
Resume Qwen3.8-27B FP4 single-stream decode optimization on one A64FX node.
Read AGENTS.md, root decode.md, a64fx/llm/qwen38_nvfp4_expanded.md,
a64fx/llm/qwen38_nvfp4_roofline.md, and a64fx/remote-dev-procedure.md.
Inspect git status/current code; baseline implementation is commit 8cb9f2c9.

Goal: 40+ end-to-end tok/s for one decode stream. Repacking at load time
from staged /local weights and 5-/6-bit payload expansion are allowed.
The new 6-bit-payload kernel already measures 219 GB/s/CMG, FAPP confirms
221 GB/s HBM reads, and four CMGs reach 852–859 GB/s. This uses 480 bytes
per tile (7.5 bits/weight including scales), versus compact's 352 bytes;
actual projection time improves only 6–9%. Hardware success is synthetic,
not real-model integration or a demonstrated 40 tok/s result.

First resolve/minimize the local qlair execution and timing discrepancies
without forcing a desired bandwidth number. Then validate real scale and
activation errors, audit the 32 GB HBM budget, and integrate bounded
load-time repacking into final NUMA-local storage with exact fallbacks.
The repack API consumes 384-byte intermediate tiles, not raw GGUF bytes.
Do not keep full source and expanded copies resident simultaneously.

Use interactive Fugaku jobs by default and FAPP for actual bottlenecks.
Job 51891531 was released; check pjstat and allocate afresh. Pin the owning
CMG before allocation/first touch and verify physical page placement.
Keep all temporary work in /local or repo tmp/, never /tmp. Preserve logs.

Validate N=1 serial decode first, separately from K=3/speculative tests.
Compare controlled FP4/FP8 baselines, complete stage timings, and memory
headroom. For any approximate timed path, run a separate unapproximated
serial replay and compare every emitted position/token ID. Do not claim
exactness from synthetic checks or infer token rate from kernel GB/s.
Implement and verify the highest-leverage remaining work, document the
measured result and any unmet gate, and commit coherent changes. Do not push.
```
