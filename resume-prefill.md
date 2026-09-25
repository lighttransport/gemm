# Resume: optimized multi-node prefill for Qwen3.8-27B on A64FX

The plan is in `/home/syoyo/.claude/plans/idempotent-frolicking-bengio.md`.
Code is in `a64fx/llm/q38p/`: `q38p_prefill.inc` is included into
`q38d/q38d_engine.c` and enabled with `Q38P=1`. The kernel generator is
`gen_pf.py`, which produces `q38p_kern.S`.

## Goal (user decisions)

- An optimized prefill that scales over 1-12 A64FX nodes; focus on 12, 6 and 4 nodes.
- Prefill and decode time-share the same nodes, e.g. 12-node prefill plus 3 TP4 decode groups.
- The metric is **steady-state prefill throughput** with several prompts in
  flight; single-prompt latency is reported as well.
- **Upper bound:** the 16-bit arithmetic peak, 6.144 T MAC/s per node
  (48 cores x 2 GHz x 64 int16 MAC/cycle).
  - Work per token is 24.35 G MAC of weights, plus 0.10 G attention at
    L=1024 and 0.11 G SSM.
  - Bound: 250 tok/s per node. The target is 90% of it:

| nodes | bound tok/s (L=1024) | 90% target |
| ---: | ---: | ---: |
| 1 | 250 | 225 |
| 4 | 1000 | 900 |
| 6 | 1501 | 1351 |
| 12 | 3001 | 2701 |

- A different weight layout from decode is allowed as long as everything
  fits in HBM2 (about 28 GB usable per node).
- Correctness gate: decode after prefill must match the F32 reference
  256/256 (`compare_tokens.py` against `/local/q38/ref-f32.log`).

## Done

**Phase 0 findings** (microbenchmarks in `tmp/q38p/ub`, `tmp/q38p/wstat`):
- **int16 SDOT (`sdot z.d, z.h, z.h`) runs at 2/cycle with 9-cycle latency,**
  so the 16-bit peak is real. The indexed SDOT/FMLA forms run at 1/cycle.
- **Every NVFP4 row of all 371 matrices spans ≤ 6 binades of per-16 scales.**
  Each weight row is therefore exactly int16 (≤ 12*15*2^6) times 2^(b_row-21).
- **Microkernel** `q38p_mk_r4t5` (32 rows x 5 tokens, weight tile in L1):
  - Accuracy: exact against a scalar reference.
  - Speed: 89.4% on 1 core, 87.9% on 48 cores, including the per-512-column
    epilogue (convert to fp64 and fma into memory).
  - Without the epilogue it reaches about 93.4%.
  - Transposed layouts, and more LD1RD or vector loads, are worse. The
    vector-load port (about one 64 B load per cycle) is the limit.
- **Streaming from L2** with a 4 KB prefetch and one epilogue per full K
  reaches about 90%. That would need a per-token activation scale, which is
  riskier for accuracy.

**Phase 1 (single node) engine, committed as 1c71d4d6:**
- Activations are int16 per (token, 512 columns). Accumulation is exact
  int64 per block and fp64 across blocks.
- F4 tiles expand on the fly from the decode F4 groups (SVE, bit-exact,
  including the K-chunked down matrix). Q4_K/Q8_K matrices (alpha/beta,
  attention k/v) are converted once to int16 tiles.
- GEMMs use dynamic 32-row tiles.
- The SSM core runs decode's `ssm_head_io` per token, one head per worker.
- Attention runs per (KV head, token) with decode's kernels.
- KV cache, SSM state and conv history are left in decode's layout; decode
  continues unchanged. The head for the last token uses `head_argmax`.
- Self-test `Q38P_TEST=1` checks the tile expansion against decode weights.

**Results** (1 node, FP4, 1024 prompt + 256 decode, all 256/256):

| version | prefill tok/s | note |
| --- | ---: | --- |
| old decode path | 41.8 | |
| v1 | 122.8 | 1200 tokens computed because of chunk padding |
| v2 | 142.9 | occupied groups only, vector epilogue, chunk 480 |
| v3 | **150.2** | tail splitting (`Q38P_TAIL=1`) and faster activation packing, commit 41025342 |
| v4 | **154.7** | token-major GEMM output stores, job 51917132 |

- The max logit diff against F32 is 0.886 (decode path: 0.197). It is
  identical across chunk sizes.
- v3 breakdown at chunk 480 (6.82 s):
  - gemm 5.81 s: worker-mean kernel 4.80, expand 0.41, the rest is epilogue,
    imbalance and barriers
  - norm+quant 0.20
  - ssm_core 0.31
  - attn 0.15
  - The kernel MAC rate is 84.6% of peak.
- The overall rate is 60% of the bound.

**2026-09-26 continuation, job 51917132:** At chunk 480, the GEMM
worker-mean profile before the change was expansion 0.407 s, kernel 4.793 s,
output epilogue 0.467 s, zeroing 0.020 s, dispatch 0.003 s and barrier waits
0.069 s. Reordering the epilogue to write all 32 adjacent rows of one token
before moving to the next cut epilogue time to 0.304 s and GEMM wall time
from 5.795 to 5.668 s. Throughput rose from 151.2 to 154.7 tok/s, with
256/256 decode agreement and unchanged maximum logit difference (0.8857).
Chunk 640 tied at 154.8 tok/s; chunk 1025 was slower at 152.3 tok/s, so
chunk 480 remains the measured setting. At 4096 prompt tokens, the same
change produced 142.2 tok/s and matched a fresh F32 reference 256/256
(maximum logit difference 0.5504). Attention took 4.791 s of the 28.809 s
prefill, up from 0.150 s at 1024 tokens. Logs are preserved in
`tmp/q38p/job51917132/`. The final compiled source repeated the 1024-token
gate at 154.5 tok/s and 256/256 agreement.

**Phase 2 pipeline prototype, job 51917132:** `Q38P_MPI` adds balanced
mixer/FFN stage cuts, MPI residual handoff, repeated independent prompts and
an optional state gather to rank 0 for full decode validation. The verified
8-prompt, 1024-token steady rates at chunk 160 are **275.1 tok/s on 2 nodes,
411.3 on 3, and 544.9 on 4**. The 3-node cuts split layers at units 43 and
85. All three configurations passed 256/256 decode against F32 after the
state gather. On 4 nodes, single-prompt latency was 2.55 s and end-to-end
throughput including fill/drain was 521.7 tok/s. Build/run instructions and
the chunk sweep are in `a64fx/llm/q38p/README.md`.

**Single-node 200 tok/s investigation, job 51917132:** A fresh 1024-token
baseline was 154.6 tok/s (6.624 s); GEMM was 5.668 s, including 4.820 s in
the int16 kernel, 0.408 s in F4 expansion and 0.314 s in output epilogues.
The 200 tok/s budget is 5.120 s, so raising the kernel from 84% to 90% of
peak alone would only save about 0.31 s. The remaining phases also need
large reductions. The following are exploratory measurements at chunk 480;
only the alias-safe panel copy was retained:

| Experiment | Prefill tok/s | Decode vs F32 | Finding |
| --- | ---: | ---: | --- |
| FP32 output epilogue for F4 | 153.8 | not run | Slower than baseline |
| Weight prefetch, 1024 B | 154.6 | residual hash unchanged | No full-engine gain |
| Token-major accumulator layout | 154.7 | residual hash unchanged | No full-engine gain |
| Precomputed F4 base vectors | 154.2 | 256/256 | Within noise |
| 1024-column K blocks | 134.1 | 5/256 | Kernel rate fell from 84% to 71% of peak |
| INT8 rounding of both operands | 153.4 | 0/256 | Insufficient accuracy |
| INT8 rounding of activations only | 155.0 | 4/256 | Insufficient accuracy |
| INT8 rounding of F4 weights only | 154.3 | 0/256 | Insufficient accuracy |
| Power-of-two activation scales | 154.4 | 129/256 | Insufficient accuracy |
| One activation scale per full K | 154.5 | 256/256 | Accurate, no speed gain alone |
| Int64 accumulation across K with one scale | 154.0 | 256/256 | GEMM faster, other cost offset it |

An `-O3` build initially appeared to reach 162.1 tok/s but diverged at the
first generated token. Per-layer hashes located the first difference in the
activation panel before layer 0 GEMM: an `int16_t` buffer had been read and
written through `uint64_t` pointers, violating C aliasing rules. Replacing
those accesses with `memcpy` restored the exact O2 panel and 256/256 decode
agreement under `-O3`. The corrected `-O3` run was 154.1 tok/s. This was a
false optimization result caused by missing work, not a usable speedup.
At 4096 tokens, the corrected `-O3` build reached 142.8 tok/s and matched
the fresh F32 reference 256/256 (maximum logit difference 0.5504).

## Remaining items

### Single node (toward 225 tok/s)
1. Continue reducing the GEMM gap. The epilogue is now about 0.30 s;
   barriers 0.07 s and zeroing 0.02 s. Profile per-GEMM imbalance next.
2. Kernel: gain the last points (epilogue cost about 4%).
   - `movprfx` from a zero register instead of `dup` zeroing.
   - Overlap group boundaries.
   - Keep the 512-column activation block: 1024 columns reduced the kernel
     rate to 71% of peak and failed the 256-token decode gate.
   - Try a per-token scale with outlier columns handled separately.
     This is accuracy-sensitive; check logit diffs.
3. Tile expansion is 0.41 s (about 8.5% of the kernel). Optimize it, or use
   a larger chunk. For multi-node, pre-expand int16 tiles in HBM: 4.1 GB per
   node at 12 stages, so no expansion at all.
4. SSM core (0.31 s): batch the per-token loop (chunked delta rule / WY), or
   overlap it with GEMMs. For now only 48 heads run on 48 workers, per token.
5. Attention (0.15 s at L=1024, grows with L): block queries so K/V are
   reused from L1/L2 (flash-style, several queries per K block).
6. norm+quant (0.20 s): fuse with the GEMM epilogue or residual add.
7. L=4096 passed 256/256 on job 51917132. Test the FP6 image (the path
   currently assumes an FP4 model with a Q6_K embedding).

### Multi-node (Phases 2-4)
8. Run 6- and 12-node allocations. The stage partition code supports these
   counts, but this allocation had four nodes. Report steady-state throughput
   and single-prompt latency.
9. Replace the working MPI handoff with uTofu Put plus MRQ completion.
   `utofu-tests/pp_handoff_bench.c` measured 3.2 us for a 12 KB hop and
   6.3 GB/s per link. MPI nonblocking sends without a progress thread were
   slower here, so the prototype uses blocking sends.
10. Stage-shard the FP4 image and pre-expand int16 tiles (48.7 GB total),
    instead of replicating the entire FP4 image on every node. Check that
    each stage plus its decode TP4 shard fits in about 28 GB HBM2.
11. Replace the rank-0 single-node decode validation path with handoff to
    co-resident TP4 decode groups. Re-shard KV, SSM state and conv history,
    then alternate prefill chunks with decode steps on the same nodes.
12. Generalize `q38d_tp.h` beyond four ranks for TP collectives if required;
    the MPI pipeline itself already supports 2–12 stages.

## How to build and run (1 node)

```
# cross build (local)
$(cat tmp/q38-fast/build2.cmd) a64fx/llm/q38p/q38p_kern.S -o tmp/q38-fast/q38p_vN
rsync -a tmp/q38-fast/q38p_vN fugaku1:work/gemm/qwen38-27b/tmp/q38-fast/
# run through the bridge (tmp/q38-fast/r.py); helper tmp/q38p/run1.sh defines
#   pf <tag> ENV...  -> prefill tok/s, decode tok/s, token match
(echo "BIN=q38p_vN; MD5=$(md5sum tmp/q38-fast/q38p_vN | cut -c1-32)"; cat tmp/q38p/run1.sh;
 echo 'pf x Q38P=1 Q38P_CHUNK=480') | python3 tmp/q38-fast/r.py 2400
```

Environment switches:

| switch | effect |
| --- | --- |
| `Q38P=1` | enable the batched prefill |
| `Q38P_CHUNK=N` | prompt chunk size (rounded to a multiple of 5) |
| `Q38P_TAIL=0/1` | GEMM tail splitting off/on |
| `Q38P_TEST=1` | expansion self-test |

- Kernel microbenchmark: `a64fx/llm/q38p/bench_mk.c` (build it with
  `q38p_kern.S`).
- Remote runs: `export TMPDIR=/local/q38/tmp` before fcc.
- Wait for the shared filesystem with an md5 loop before building (r.py
  runs remotely).

## Resume prompt

> Continue the A64FX Qwen3.8-27B prefill work: read resume-prefill.md and the
> plan in /home/syoyo/.claude/plans/idempotent-frolicking-bengio.md. The
> latest verified state includes a 2–4-node MPI pipeline with 256/256
> decode agreement. At 4 nodes and chunk 160 it sustains 544.9 tok/s across
> eight prompts; see `a64fx/llm/q38p/README.md`.
> Continue with the remaining items:
> - single-node GEMM gap, kernel epilogue, expansion, SSM, attention and
>   norm costs, toward 225 tok/s (90% of the 250 tok/s/node int16 bound);
> - measure the pipeline on 6 and 12 nodes (steady-state targets 1351 and
>   2701 tok/s), then stage-shard weights and replace MPI handoff with uTofu;
> - move the validated rank-0 decode handoff to co-resident TP4 groups.
>
> Keep every change 256/256 against the F32 reference and commit each unit.
> Do not use /tmp and do not push.
