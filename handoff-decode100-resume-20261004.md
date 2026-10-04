# GLM-5.3 Flash decode-100 — resume handoff (A64FX, 12 nodes)

Snapshot: **2026-10-04 ~23:45 JST**. Written for resuming on another PC with another coding agent.
Detailed per-campaign history: `handoff-decode100-20261004.md` (same directory). Earlier plan:
`~/.claude/plans/vectorized-napping-pearl.md` on the original PC (summarized under "Strategy" below).

## Goal and status
- Goal: **≥100 delivered decode tok/s** for single-stream GLM-5.3 Flash 4-bit on 12 A64FX nodes (TP12).
- Qualified (promoted) recipe is unchanged: **35.46 decode / 412.27 prefill tok/s**. Everything below is opt-in,
  default-off, and **not promoted** (promotion needs 8K and 32K qualification).
- **Best timed so far: 53.35 tok/s**, MTP draft depth 2, 512-token context, option set "BEST" (below).
  Plain (non-MTP) decode ≈ 35.5–36.1 tok/s (≈28 ms/token).

## Repository and code state
- Local repo (original PC): `/mnt/nvme02/work/gemm/glm53f`, branch `glm53f`, remote `git@github.com:lighttransport/gemm`.
- **Not pushed.** Local `glm53f` is many commits ahead of `origin/glm53f` (a90f7972). To move to another PC, the
  owner must push (agents may not push without explicit per-action permission) or copy the repo.
- Commits of this session (newest first), all under `a64fx/glm5/`:

| Commit | What | Status |
|---|---|---|
| 14987b73 | `--kda-decode-pipeline heads` (per-head KDA pipeline) + `glm53f_native_matvec_rows`; in-model mHC phase report (diagnostic build) | **untested** — j2s gates it |
| 21b1b999 | `--env GLM53F_NAME=VALUE` (setenv inside every rank) | parser test PASS |
| 29e140f0 | `GLM53F_TARGET_GAP` per-rank min/mean/max ledger + mHC site split | working (j2q) |
| f04f4443 | `--snapshot-copy deferred` (MTP verify KDA states in per-layer slots; restore only accepted) | d2/d3 state BIT_EXACT; +2–3% |
| f1d0d1f7 | `--mhc-post-kernel chunked` (batch mHC post in 256-col tasks) | +1.2/+3.4% in two pairs |
| eab714d8 | `--snapshot-copy parallel` | +7% (superseded by deferred) |
| 59e8cf0e | `--kda-verify-kernel columns64` | d2 state BIT_EXACT; +1–2% |
| 24f5dfd9, 9e385cd4 | ROWS4 expert interleave, R16 two-chain tile | no gain (kernel not ILP-bound) |
| earlier | see `handoff-decode100-20261004.md` (fusions, placement, local-gram mHC, front/router verify) | |

- Unrelated pre-existing local modifications (leave untouched): `a64fx/glm5/test_glm53f_mhc_local.c`,
  `glm53f_runtime.h` etc. listed in `git status` from before this session, plus many untracked `tmp/` and `q38fn/` files.

## Remote environment (Fugaku)
- SSH alias `fugaku1` (login node). Remote checkout: `$HOME/work/gemm/glm53f-strata-20261001`
  (`/vol0006/mdt0/data/hp250467/work/gemm/glm53f-strata-20261001`). Group `hp250467`.
- **The remote tree is synced by `rsync` of individual files, not git.** It already contains all files of the commits
  above. On a new PC: clone/copy the repo, then rsync changed `a64fx/glm5/*` files to the remote before building.
- Allocation: job **PJM52167472**, 12 nodes, interactive bash-over-http bridge, **ends ~01:00 JST 2026-10-05**.
  After it ends a new job must be submitted and **re-staged** (`tmp/decode100-20261004/stage-j2.sh`; staging copies
  weights to node-local `/local/glm53f-q4-*-<jobid>`; takes a while).
- Bridge (original PC only): `tmp/bash-http-glm53f-decode2/launch.sh` (submits the job + tunnel; local port 42449 →
  login port 32449). Run commands on the compute node with
  `printf '%s\n' 'set +e; ...' | python3 tmp/bash-http-glm53f-decode2/remote.py`.
  If it says "no such session", move `tmp/bash-http-glm53f-decode2/session` aside and rerun (new session).
  A new PC needs its own bridge: `a64fx/tools/bash-over-http/run_bash_http_interactive.sh` with the env vars in
  `launch.sh` (REMOTE, FRONTEND_SSH_TARGET=login1.fugaku.r-ccs.riken.jp, NODES=12, NODE_SPEC=2x3x2, ...).
- Builds: on the login node with cross compilers (`mpifccpx -Nclang`), e.g.
  `ssh fugaku1 'cd work/gemm/glm53f-strata-20261001 && bash tmp/decode100-20261004/build-<tag>.sh'`.
  New build = copy the previous script with `sed` renaming the tag (`build-pipe20-v30.sh` is the latest; v29
  `build-mhct19-v29.sh` adds `-DGLM53F_MHC_PHASE_TIMING`, diagnostic only). Binaries land in
  `a64fx/glm5/build/candidate-decode100-<tag>/`. **Never rebuild or overwrite an existing candidate.**

## How campaigns run
- `tmp/decode100-20261004/driver-j2v.sh WAITPID ARMS.json TAG BINARY` — waits for WAITPID (previous driver) to exit,
  refuses if another MPI job is running, runs native checks, then `campaign-j2.py`.
- Arms JSON keys: `pairs` (alternating A/B vs legacy at 512 ctx), `arms` {name: [cli...]},
  `exact_8k` {name: [cli...]} (8K full-state BIT_EXACT gate vs legacy), `mtp_state` {name: [depth, [cli...]]}
  (state-gated MTP exactness), `mtp` {name: [depth, [cli...]]} (timed MTP). Use `--env GLM53F_X=1` (v28+) instead
  of `ENV:` entries: plain environment variables are **not** forwarded to ranks by mpiexec.
- Results: `tmp/decode100-20261004/driver-<tag>.log` (lines `DECODE100_*`), `results-<tag>/performance.json`
  (incl. MTP draft/verify/replay seconds), rank logs `tmp/glm53f-q4-52167472/benchmark-decode100-<tag>-<arm>-*.<rank>`
  containing `GLM53F_TARGET_PROFILE`, `GLM53F_TARGET_GAP`, `GLM53F_KDA_DECODE_DETAIL`, `GLM53F_MHC_INMODEL`.
- Rules: one MPI job at a time; never concurrent `remote.py` calls; never edit a driver while it runs; no `set -e`
  in the persistent bridge shell; no `/tmp` (use repo `tmp/`); never `p4 submit`; no `git push` without permission;
  commit coherent units with trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Option sets
- MTP CLI (added by campaign): `--speculation mtp --draft-depth N --spec-policy always ...`.
- **BEST (53.35 tok/s, d2):** `--weight-placement cmg-experts --mhc-kernel local-gram --mhc-prefetch next
  --kda-decode-kernel columns --mhc-batch-tail parallel --kda-verify-kernel columns64 --sparse-verify-kernel front
  --moe-verify-router batch --mhc-verify-kernel team --snapshot-copy deferred --mhc-post-kernel chunked`
- Do **not** use: `--act-header-kernel cached` (SIGSEGV at 8K), `--kda-decode-kernel columns16` (−4%),
  `--kda-verify-kernel columns` (16-col, regression). Fusions (`--*-layer-kernel fused`) are exact but give ~0.
- Run-to-run noise is ±2–4%: only alternating repeated pairs count.

## Where the time goes (plain decode, 28 ms/token, rank logs of j2q)
| Part | ms/token | Notes |
|---|---|---|
| mHC sites | 6.6 | attention site 0.05; **FFN site 3.6** (~86 µs, includes fused router); **end-of-layer 3.0** (~70 µs). Local work, no MPI. Weights only ~66 MB/token (bf16) → latency-bound. local-gram (2× faster standalone) does not change it in-model — cause unknown, j2r measures it. |
| KDA (31 layers) | 7.1 | per layer ≈228 µs: proj 60, conv 20, fb+norm+decay 21, recurrence 20, gb+rmsnorm 21, o_proj 14, allreduce 45 (≈2× pure latency). Small steps are head-local but run as team phases with only 6 busy threads. |
| MoE | 8.0 | router 0.76, local 4.9 (~107 GB/s; Q4_K/Q5_K kernel ≈2.1 instr/byte, IPC ~0.5), allreduce 2.8 (67 µs × 42) |
| Sparse | 4.5 | front 1.7 (156 GB/s), index 0.6, MLA 1.3, o_proj 0.33 (~500 GB/s), allreduce 0.8 |
| head/embed/dense | ~0.9 | |
- Cross-rank skew is small: per-phase min/mean/max within 10–15%.
- Bandwidth floor (≈800 GB/s HBM + ~15 µs allreduces) is ≈4 ms/token: the gap is ~7×, not 2×.
- MTP: verify ≈92% of MTP time; acceptance 42/43 (d2), 47/51 (d3); verify ≈17 ms/position vs 28 ms plain.
  100 tok/s with MTP needs ≈7 ms/position → plain ≈15 ms/token is the realistic target (MTP ≈1.6×).
- Routed experts are 15 GB/rank (Q8 would not fit in 29 GB free). Expert kernel experiments (R16, ROWS4) showed it
  is not dependency-bound.

## Bug found (fixed in the commit after c08cd0c3)
- `glm53f_mhc_fused_sync_on()` returned `e && *e && atoi(e)` (0/1), so **`--mhc-kernel local` and `local-gram` never
  ran in-model**; every "local-gram" model result in both handoffs actually measured the fused-sync (mode 1) path.
  Fixed to return the mode. v31 (`candidate-decode100-mfix21-v31`) is the first binary where local-gram really runs;
  j2t gates it (8K exact + pair + MTP d2 best ± KDA head pipeline). The standalone kernel is ~23 µs vs ~48 µs.

## In flight at snapshot time (job ends ~01:00)
- **j2r** (v29, `-DGLM53F_MHC_PHASE_TIMING`): plain legacy vs best; read `GLM53F_MHC_INMODEL` (rank 0 log of the
  best arm) and compare per-call kernel time with the 70–86 µs site times → tells whether mHC time is inside the
  kernel (cold misses / barriers) or around it (dispatch, router, non-team serial code). Diagnostic only.
- **j2s** (v30): `--kda-decode-pipeline heads` 8K exact gate (`exact_8k pipe`) + 2 alternating pairs
  (`pipe`, `best-pipe`). If not BIT_EXACT, suspect `glm53f_native_matvec_rows` row-group alignment or act prep.
- **j2t** (v31, queued after j2s): local-gram 8K exact gate, one pair vs legacy, MTP d2 BEST and BEST + head pipeline.
- Read with: `ssh fugaku1 'grep -h DECODE100 work/gemm/glm53f-strata-20261001/tmp/decode100-20261004/driver-j2[rst].log'`.

## Strategy / next steps (in priority order)
1. Finish j2r/j2s analysis. If the head pipeline is exact and faster, add it to BEST and run MTP d2/d3.
2. **mHC (6.6 ms):** find the in-model overhead from j2r. Candidates: router inside the FFN site (move it out or make
   it Q8), cold `fn`/streams after MoE evicts L2 (prefetch during the preceding allreduce), serial non-team code
   around `glm53f_mhc_post_pre_sve`. Target ≈2 ms.
3. **KDA (7.1 ms):** head pipeline (above); projections at 60 µs → requantize bf16-derived projections to Q8_0R16
   (Q8 matvec reaches ~350 GB/s); overlap the 45 µs allreduce with the next mHC. Target ≈3 ms.
4. **MoE allreduce skew (2.8 ms):** balanced expert slicing across all 12 ranks (needs restaging; plan P3 in the
   earlier handoff). Shared expert overlapped with the routed allreduce.
5. Then MTP on top (BEST options); deeper draft only if verify becomes sublinear.
6. Structural fallback: 4 MPI ranks/node (one per CMG) — CMG-local barriers/memory, needs TP48 restaging.
