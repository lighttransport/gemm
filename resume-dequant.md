# Goal: finish Qwen3.8 rank-local k-quant decode integration

Continue the Qwen3.8-27B UD-Q4_K_XL A64FX work from the verified Q5R/IQ4R
layouts and rank-local sidecar builder to a safe, measured `tp_runner` decode
path. The finished path must preserve compact weights for prefill and fallback,
load only the local rank's decode sidecar, retain CMG-local row ownership, and
pass exact 128- and 256-token greedy-output gates before promotion.

Do not reset the branch or discard unrelated work. Do not push without explicit
permission in the current user request.

## Definition of done

The goal is complete only when all of the following are true:

1. A real rank-local `Q38TP` stage can be planned and converted to a versioned
   `Q38KQC1` sidecar with bounded memory and I/O.
2. The TP loader validates source identity, layout version, shapes, offsets,
   entry-table hashes, and payload hashes before making a cache entry usable.
3. Decode dispatch uses Q5R/IQ4R only for validated compatible tensors and
   falls back to the unchanged compact kernel for every unsupported, missing,
   invalid, or tail case. Prefill continues to use its required compact form.
4. Cache pages are first-touched or mapped consistently with the persistent
   pool's row/CMG ownership; the eight-row scheduler neither crosses ownership
   boundaries nor drops tail rows.
5. Compact and cached paths produce identical greedy token hashes for the same
   prompts at 128 and 256 generated tokens. Q5R's small floating-point
   accumulation differences are not grounds to waive this token gate.
6. A clean-node run records per-rank stage sizes, peak/steady `MemAvailable`,
   sidecar load time, total decode tok/s, and compact-versus-cache performance.
7. `qwen-q8.md` contains exact commands and representative correctness,
   memory, and performance output, and the focused work is committed without
   unrelated files.

## Verified starting point

The initial Q5 layout benchmark is commit `b383fc0b`, the shared exact-cache
milestone is `7d37cd3c`, the rank-local sidecar builder milestone is
`8cf312cf`, and strict real-sidecar validation is `cfcdf019`. The current
branch may advance beyond those commits; they are landmarks, not reset
targets.

Relevant files:

- `a64fx/llm/bench_qwen38_kquants.c`
- `a64fx/llm/kquant_decode_cache.h`
- `a64fx/llm/test_qwen38_kquant_cache.c`
- `a64fx/llm/qwen38_kquant_stage.[ch]`
- `a64fx/llm/qwen38_kquant_load.[ch]`
- `a64fx/llm/test_qwen38_kquant_stage.c`
- `a64fx/llm/qwen38_tp_stage.h`
- `a64fx/llm/Makefile`
- `qwen-q8.md`, `Q5_K row-interleaved decode probe` and child sections

The real model used for the isolated benchmark is:

```text
/home/u14346/models/qwen38/27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
```

Q5R interleaves eight rows by 256 columns, expands Q5 values once, and retains
the original affine scales/minima. It occupies 280 bytes per original
row/block versus 176 bytes for Q5_K. IQ4R expands the original nonlinear
palette exactly and occupies 272 bytes per row/block versus 136 bytes for
IQ4_XS. Both layouts have version 1.

Validated real layer-0 results at 2.0 GHz and 48 threads:

| Projection | Compact A8 | Row-interleaved | Speedup/effective BW |
|---|---:|---:|---:|
| Q5_K up, 17408 x 5120 | 1.100 ms | 0.244 ms | 4.5x / 251.5 GB/s |
| Q5_K down, 5120 x 17408 | 0.881 ms | 0.265 ms | 3.3x / 230.9 GB/s |
| IQ4_XS gate, 17408 x 5120 | 0.466 ms | 0.213 ms | 2.2x / 222.3 GB/s |

IQ4R is bit-identical to native A8 over wave, sparse, high-dynamic-range, and
deterministic random inputs. Q5R retains native-A8 NRMSE; observed differences
from native A8 were `2.83e-8`--about `1e-6` normalized RMS depending on shape
and pattern, with a worst recorded absolute difference of `1.53e-5`.

The metadata scan reports:

```text
model tensors=866 tensor_bytes=17.912GB Q5_K=325/12.936GB Q5R_eligible=325/20.581GB Q5R_delta=7.644GB projected=25.557GB
IQ4_XS=65/3.078GB IQ4R_eligible=65/6.155GB IQ4R_delta=3.078GB Q5R_IQ4R_projected=28.634GB
```

A combined single-node replacement leaves only about 3.4 GB before runtime
state, and an additive cache cannot fit. The selected design therefore builds
a separate sidecar from each already-sharded compact `Q38TP` rank file.

`qwen38_kquant_stage` writes `rankNN.kquant` through a PID-qualified partial
file and atomic rename. Its header records rank/size, source stage identity,
layout version, tensor metadata, and source/cache checksums. Conversion holds
only one compact tensor and its packed result at a time, uses positioned I/O,
and calls `POSIX_FADV_DONTNEED`. `--plan` performs no output writes. Valid
sidecars are reused; inconsistent header metadata or hashes cause a rebuild.

The bounded synthetic stage test verifies plan/build/reuse, exact Q5R/IQ4R
payload bytes, all relevant hashes, and rebuild after deliberate header
corruption:

```text
SENTINEL qwen38_kquant_stage=OK entries=2 q5r=2240 iq4r=2176 reuse=1 corrupt_rebuild=1 loader_rejects=9
```

The independent read-only loader validates the compact source identity, fixed
headers, entry-table hashes, unique names, type/format pairs, local shapes,
monotonic non-overlapping extents, source checksums, and every payload hash.
It uses a fixed 1 MiB hashing buffer, evicts validation reads, and maps the
sidecar only after every entry passes. Tests require rejection of bad magic,
version, layout, truncation, offset, duplicate name, source entry, source file
size, and payload data. The runtime attachment described below now wires this
loader into the TP runner and persistent-pool dispatch without weakening those
checks.

No real compact `rank00.blob` was present under `/local/u14346` during the
sidecar-builder milestone. A later bounded rank-0 run built and checked the
real mixed-Q4 artifact. All four TP4 metadata plans are symmetric: 866
compact entries, a 5.758 GB compact file, 390 cache entries, a 7.810 GB
sidecar, and 13.568 GB combined files per rank. Rank 0 produced exact file
sizes of `5,757,905,920` and `7,809,826,816` bytes. Full payload validation
passed in 31.635 seconds, builder reuse passed, and `MemAvailable` remained
about 30.8 GB. Ranks 1--3 still require physical builds on their own nodes.

The decode-only runtime attachment is now implemented. `tp_runner` accepts
`--kquant-stage DIR`; every rank validates its local compact identity and all
sidecar hashes, then votes before retaining the mapping. Cache pointers are
attached only after compact prefill and a second all-rank vote. Any validation
or attachment mismatch coherently returns all ranks to compact dispatch. The
persistent pool statically partitions complete eight-row groups, while partial
or tail extents explicitly use the compact path. Validation reads are evicted
before mapping and `MADV_RANDOM` limits readahead beyond worker-owned ranges.
Synthetic coverage includes uneven three-worker ownership, unaligned ranged
dispatch, a 15-row compact fallback, invalid format rejection, and model
attach/detach. Fujitsu builds of the focused test and full runner pass; the
multi-node token and performance gates have not run yet.

## Constraints and safety rules

- Read `AGENTS.md` and this entire file before changing code.
- Use `/local/u14346/codex-research` for compiler scratch and temporary staged
  test data. Never use `/tmp`.
- Never `cat` or make an interactive full copy of a multi-gigabyte model.
  Keep model and stage access lazy or bounded, use positioned/chunked I/O, and
  discard page cache as work progresses.
- Run `qwen38_kquant_stage --plan` and establish a per-rank compact + cache +
  KV/state/scratch budget before building or loading real sidecars.
- Monitor `MemAvailable`; do full-load speed work detached or in a batch job.
- Production tuning selection must be a runner argument, not a new production
  environment-variable switch. Environment variables may remain diagnostics.
- Preserve compact prefill and correctness fallbacks. Never silently accept a
  cache whose source identity, format, dimensions, offsets, or checksum fails.
- Preserve all unrelated dirty-worktree files. In particular,
  `common/transformer.h` already contains unrelated user changes. Inspect its
  diff before editing it and keep this work separable; if runtime attachment
  cannot be isolated safely, stop and ask rather than overwriting or staging
  those changes.

## Remaining work, in order

1. **Complete real staging on all TP4 nodes.** Rank 0 is built and fully hash-
   validated; metadata plans cover ranks 0--3. On a four-node allocation, build
   the compact stage and sidecar locally on ranks 1--3, run
   `qwen38_kquant_check` on every rank, and record validation time plus
   `MemAvailable`. Do not copy rank 0's `/local` files between nodes.

2. **Run the integrated smoke test on TP4.** Use the explicit
   `--kquant-stage` argument with all four local compact/sidecar pairs. Confirm
   every rank reports `prepared` before prefill and `decode attach OK` after
   prefill. Exercise one missing/corrupt sidecar on a disposable synthetic or
   staged copy and confirm the all-rank vote selects compact fallback. Do not
   corrupt the validated rank artifacts.

3. **Run correctness acceptance.** Use the same prompt and settings for compact
   and cached paths. Require exact token hashes at 128 and 256 generated tokens.
   Also retain the isolated multi-pattern tests so an end-to-end pass cannot
   conceal a tensor-level regression.

4. **Measure and document.** On clean nodes record load/stage timings, peak
   and steady memory, forward/decode timing, total tok/s, and per-rank storage.
   Add the exact multi-node commands and representative output to `qwen-q8.md`,
   run `git diff --check`, commit only focused follow-up files, and report the
   hash. Do not push.

## Revalidation commands

```sh
mkdir -p /local/u14346/codex-research
TMPDIR=/local/u14346/codex-research \
  make -B -C a64fx/llm \
  qwen38_kquant_bench qwen38_kquant_test \
  qwen38_kquant_stage qwen38_kquant_stage_test qwen38_kquant_check \
  CC=fcc OPENMP=1

OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/llm/build/test_qwen38_kquant_cache

OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/llm/build/test_qwen38_kquant_stage \
  ./a64fx/llm/build/qwen38_kquant_stage \
  /local/u14346/codex-research/kquant-stage-test

Q38TP_RANK=0 Q38TP_SIZE=4 \
  ./a64fx/llm/build/qwen38_kquant_check \
  /local/u14346/codex-research/qwen38-q4-tp4 \
  /local/u14346/codex-research/qwen38-q4-tp4-kquant

OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  numactl --interleave=all ./a64fx/llm/build/bench_qwen38_kquants \
  /home/u14346/models/qwen38/27b/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  7 blk.0.ffn_gate.weight wave
```

For an existing real compact stage, substitute its actual directories and
rank count only after verifying them:

```sh
Q38TP_RANK=0 Q38TP_SIZE=4 ./a64fx/llm/build/qwen38_kquant_stage \
  --plan /local/u14346/ACTUAL_COMPACT_STAGE \
  /local/u14346/ACTUAL_KQUANT_STAGE
```

## Resume prompt

```text
Continue the active goal in resume-dequant.md: finish and accept safe rank-local
Q5R/IQ4R decode for Qwen3.8 on A64FX. Read AGENTS.md and the whole goal file
first, inspect git status/diffs, and preserve every unrelated dirty-worktree
change. The exact layouts, kernel tests, versioned Q38KQC1 sidecar builder,
strict loader, explicit `tp_runner --kquant-stage DIR` attachment, all-rank
fallback votes, decode-only activation, and persistent eight-row ownership are
implemented. Real rank 0 is built and hash-validated; metadata plans cover all
four symmetric TP4 ranks. Revalidate the focused tests, then continue at the
first unfinished item: physically stage and check ranks 1--3 in a four-node
allocation, then run the integrated TP4 smoke test.

Do not create a full single-node additive cache. Plan real per-rank memory
before conversion, use bounded I/O and /local/u14346/codex-research, preserve
compact prefill/fallback, and verify the `prepared`/post-prefill `decode attach
OK` diagnostics on every rank. common/transformer.h may retain unrelated user
edits, so never overwrite or stage them. Require exact 128/256-token compact
versus cached greedy hashes plus clean-node sidecar-load, memory, bandwidth, and
tok/s evidence. Update qwen-q8.md, commit only focused files, report the commit
hash, and do not push.
```
