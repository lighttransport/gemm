# GLM53F A64FX optimization handoff

Snapshot: **2026-10-04 14:50 JST**. Recheck live status before acting; the detached jobs continue after this handoff.

## Goal and current outcome

Continue GLM5.3F optimization on **12 A64FX nodes**, targeting **100+ delivered decode tok/s and 2000+ prefill tok/s**. Weight prefill this round; speculative decoding is also authorized. Study `/home/syoyo/work/Strata`, branch `glm53f`, commit `3bbb469`, for transferable techniques.

Local checkout: `/mnt/nvme02/work/gemm/glm53f`, branch `glm53f`. Remote checkout: `$HOME/work/gemm/glm53f-strata-20261001` on Fugaku. The qualified recipe remains `candidate-capacity4096-v1`, **35.462134 decode / 412.273634 prefill tok/s**. Neither target is met. New kernels remain opt-in; **nothing has been pushed**.

Start with this document, then `resume-strata.md`, `a64fx/glm5/GLM53F_STRATA.md`, and recent git log. The long resume has historical allocation/owner references; the live snapshot below takes precedence.

## Rules and authorization

- Read repository `AGENTS.md` and `a64fx/remote-dev-procedure.md`. User supplied the same repository guidelines in conversation.
- Do not use `/tmp`. Use repo `tmp/`; use `/local` on compute nodes for staged model images.
- **One MPI job at a time in our allocation. Never make concurrent `remote.py` calls.** Login-host cross-builds, rsync and bounded read-only audits can coexist with staging/MPI.
- Do not rebuild or overwrite immutable candidate binaries. Use a fresh candidate directory for changes.
- Do not restage while the current staging owner is live. Model files are large: use canonical chunked stagers, not whole-file `cp`/`cat` or unbounded reads. `/local` is wiped when an allocation expires.
- Never put `set -e` in the persistent bridge shell. Scope launch checks as `( set -e; ... )`; guard optional log reads with `|| true`.
- Commit coherent units freely; report hashes. **No git push without explicit per-action permission in the current request.** There is no push authorization.
- Leave unrelated existing work untouched, including `a64fx/remote-dev-procedure.md`, Q38FN changes, `common/q38fn_spec.h`, CUDA tools and unrelated logs/binaries. Do not stage the entire tree or scratch directories.
- Do not spawn agents unless explicitly authorized by the user or applicable instructions. No delegation was requested.
- Do not confuse component speedups, static assembly counts, speculative acceptance, or a three-trial screen with confirmed model throughput. Preserve failed/rejected evidence.

## Live allocation and bridge

| Item | Value |
|---|---|
| Our job | **PJM52159552**, 12 nodes, compact `2x3x2` |
| Mode | normal 2000 MHz, eco0 |
| Start / expiry | October 4 **14:05:14 / 20:05:14 JST** |
| New model run cutoff | **19:35 JST**: campaign needs 600 seconds before its 19:45 guard |
| Compute host | `e25-4008c` |
| Bridge | local **42448** → login1 **32448** → compute **21264** |
| Control directory | `tmp/bash-http-glm53f-prefill5/` |
| SSH control socket | repo `tmp/cm-prefill5` |
| tmux socket/session | `tmp/tmux-glm53f/pp1.sock` / `glm53f-prefill5` |
| Launcher / log | `tmp/bash-http-glm53f-prefill5/launch.sh` / `launch.log` |

Separate **four-node PJM52158976 is unrelated; do not touch it**. Previous PJM52138572 expired and its `/local` images are gone.

Use the bridge serially, from local repository root (SSH/network generally needs sandbox escalation):

```bash
printf '%s\n' 'set +e; cd "$HOME/work/gemm/glm53f-strata-20261001"; ps -p 127,459,1052 -o pid,etime,args; tail -n 12 tmp/strata-router-tiles-20261004/stage-driver-v1.log; tail -n 12 tmp/strata-router-tiles-20261004/driver-packed-v1.log; tail -n 12 tmp/strata-router-eight-20261004/driver-v3.log; date' |
  python3 tmp/bash-http-glm53f-prefill5/remote.py
```

The client timeout is 50 seconds; exit124 means no exit marker, not necessarily job failure. Verify owners/PASS logs before restarting anything. `remote.py` creates a session if its local `session` file is absent. If a session expires, archive that pointer and create a new one; detached jobs survive. Do not relaunch MPI because the HTTP session vanished.

For read-only login-host access:

```bash
ssh fugaku1 'pjstat'
ssh fugaku1 'cd "$HOME/work/gemm/glm53f-strata-20261001"; tail -n 3 tmp/glm53f-q4-52159552/routed-stage-router-tiles5-stage-130-1.*.0'
```

If this task's local tunnel needs recreation, use `REMOTE_PORT=32448` (the tunnel script does **not** use `FRONTEND_PORT`):

```bash
CONTROL_DIR="$PWD/tmp/bash-http-glm53f-prefill5" \
CONTROL_PATH="$PWD/tmp/cm-prefill5" REMOTE=fugaku1 \
REMOTE_PORT=32448 LOCAL_PORT=42448 \
  a64fx/tools/bash-over-http/open_local_tunnel.sh
```

The short socket path avoids Unix socket path length limits. Only repair this task's socket/forward, not other users' tunnels.

## Detached owners: do not duplicate them

| Owner | State at snapshot | Work / dependencies |
|---|---|---|
| **127** | active; sole MPI owner | target + MTP staging, `tmp/strata-router-tiles-20261004/stage-v1.sh` |
| **459** | waiting | original 12×16 router campaign; waits for127 terminal and successful staging, checks idle MPI |
| **1052** | waiting | final multi-shape campaign; waits for459 terminal and `ROUTER_TILES_NATIVE_PASS`, checks idle MPI and cutoff |

At14:50 rank0 routed staging logged **layer35, 12145655808 bytes**. The bridge node reported `MemAvailable=31021120 kB`, `Dirty=9472 kB`, `Writeback=0`; this is a single-node snapshot, not a global bound. Active MPI is `glm53f_q2_stage` from `candidate-v15` into `/local/glm53f-q4-routed-52159552`.

Stage log: `tmp/strata-router-tiles-20261004/stage-driver-v1.log`, with terminal sentinel **`MTP_BATCH_STAGE_PASS`**. Target stage log is `stage.log`; MTP stage log is `mtp-stage.log`. They may remain quiet while per-rank MPI logs advance.

Images are `/local/glm53f-q4-*-52159552`, plus `/local/glm53f-mtp-routed-52159552` and `/local/glm53f-mtp-shared-52159552`. Source GGUF is `$HOME/models/glm53f-gguf-wide/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf` (six shards); base safetensors are under `$HOME/models/glm53f`.

Canceled **idle** waiters337,788,943 never launched MPI.337 targeted a discarded legacy-router prototype.788/943 were unlaunched follow-up revisions. Their logs/binaries remain in scratch. Do not resume them.

## Current router experiments

The tuned default router uses packed BF16 weights and6 tokens×48 experts. An initial four-token legacy-router optimization was ineligible for this path and was removed before MPI. Its discarded source/build logs are scratch-only.

Current CLI: **`--moe-router-prefill legacy|tiles12|tiles8|unroll1`**, default `legacy`.

- Mode1 `tiles12`:12 tokens×16 experts,12 vector accumulators. Same packed layout, sequential key FMAs.
- Mode2 `tiles8`:8×32,16 accumulators. Separate pointers handle16-expert halves crossing packed48-expert boundaries; no repack.
- Mode3 `unroll1`: original6×48 tile with key-loop unrolling disabled using guarded Clang/GCC pragmas.
- Runtime eligibility: packed router mode1, SVE16 lanes, tokens>8 for tiles8 and>6 otherwise. Diagnostic router mode2 and nonpacked paths keep legacy. Internal numeric selector is `GLM53F_MOE_ROUTER_TILES12`; its name is historical.
- No new allocation, collective or weight copy. Decode small batches retain the original route.

Files: `glm53f_moe_grouped_native.h`, `glm53f_expert_decode_12n.c`, `glm53f_runtime.h`, `test_glm53f_router_tiles.c`, `test_glm53f_prefill_config.c` under `a64fx/glm5/`.

**Native correctness/performance are still unmeasured at this snapshot.** FCC fast/conservative builds are warning-clean; strict host CLI, Python campaign syntax, bash driver syntax and `git diff --check` pass.

### Original 12×16 campaign (owner459)

- Immutable binary: `a64fx/glm5/build/candidate-router-tiles-packed-v1/`.
- Source provenance: commit **`3d74197b`**, not current source.
- Scratch: `tmp/strata-router-tiles-20261004/`.
- Build/driver/campaign: `build-packed-v1.sh`, `driver-packed-v1.sh`, `campaign-packed-v1.py`.
- Driver log: `driver-packed-v1.log`; native outputs: `results/native-{fast,conservative}-{1,47}.log`; model outputs: `results-model/`.
- 72 cases/rank/configuration ×12 ranks×4 configurations =3456 expected exact cases. Tests include all12 token tails,23/24/25 and47/48/49 boundaries,511/512/513/4096-token production shapes, K16/31/64/4096, all288 logits/canaries, zeros and varied signed finite bits.
- Seven alternating component pairs at512/4096,2 calls/mode,47 threads, MPI-rank maximum. Both sizes need≥1.05 fast and≥1.02 conservative before model timing.
- 17 model runs: four short128/full8049 state gates, frozen control, two three-trial screens, ten alternating one-trial pairs. Other options remain legacy.
- Sentinels: `ROUTER_TILES_NATIVE_PASS`, then **`ROUTER_TILES_MODEL_FULL_PASS`**. Performance status: `router_model_confirmation_complete`.

### Expanded comparison (owner1052)

- Immutable binary: `a64fx/glm5/build/candidate-router-tiles-eight-v3/`.
- Source provenance: commit **`ed1d05a6`**.
- Scratch: `tmp/strata-router-eight-20261004/`.
- Build/driver/campaign: `build-v3.sh`, `driver-v3.sh`, `campaign-v3.py`; selection script `gate-native-v2.py`.
- Driver log: `driver-v3.log`; native/model directories have the same names as above, under this scratch root.
- Same72 cases, now comparing all three alternatives:3456 case instances /10368 expected bit-equivalence comparisons. These numbers are expected, not passing results yet.
- Seven rotating timing sweeps, four modes,512/4096 tokens, two calls/mode, both math modes at47 threads.
- Gate writes `native-ratios.json` and, if qualified, `selected-native.json`. Each eligible alternative must pass both size/math thresholds; greatest fast4096 ratio wins. The model campaign reads the selected **actual CLI option**.
- Owner1052 can run after459 passes native correctness but fails its component timing gate. It cannot run if459 fails native correctness.
- Model labels contain inherited **`tiles4`** strings even though the actual CLI is selected tiles12/tiles8/unroll1. Inspect `MTP_BATCH_CONFIG` argv and `native_selection`; do not mislabel results or rename live artifacts.

FCC static full-shape probe SVE spill store/reload counts: original4/6, tiles12 7/7, tiles8 4/6, unroll1 0/0. They exclude ABI callee saves and come from standalone probes; **not dynamic traffic or speedup measurements**. Source/assembly are in follow-up scratch; evidence is `a64fx/glm5/strata-router-shapes-20261004.json`.

## What to do next

1. Recheck allocation, owner commands and both driver logs. Do not assume the snapshot is current. Keep the queue serial; no fresh MPI until relevant owner is terminal and no MPI remains.
2. Let native gates finish. If arithmetic fails, preserve logs and isolate the failing shape/tail before model runs. If timing gates fail, record rejection; do not weaken gates to manufacture a gain.
3. Collect completed small artifacts with rsync, **excluding large state files**:

```bash
rsync -a --exclude='*.state*' \
  fugaku1:work/gemm/glm53f-strata-20261001/tmp/strata-router-tiles-20261004/ \
  tmp/strata-router-tiles-20261004/
rsync -a --exclude='*.state*' \
  fugaku1:work/gemm/glm53f-strata-20261001/tmp/strata-router-eight-20261004/ \
  tmp/strata-router-eight-20261004/
```

4. Model results need fresh final reporting scripts. Adapt `tmp/strata-mla-logits-20261003/record_values_model.py` / `tmp/strata-mtp-restore-20261004/record_model.py`; verify historical source via `git show COMMIT:path`, not current files for the older binary. Preserve full commands, binary hashes, all ID counts/reference equality, trial completion, memory minima, paired ratios and profiles. Do not claim a model gain from native component timing.
5. Audit cross-option short/full state files **on the login host** with1MiB reads, not by copying them locally or loading whole files. Adapt `tmp/strata-mtp-restore-20261004/audit_states.py` or `tmp/strata-mla-logits-20261003/audit_states_values_model.py`. Compare legacy versus inherited `tiles4` labels, all12 ranks, equal nonempty sizes and complete129-ID counts. Record byte equality and SHA256. Internal warmup/trial state PASS alone does not replace cross-option audit.
6. Archive unique rank0 profile logs from remote `tmp/glm53f-q4-52159552/`: `benchmark-router-tiles-*.0` for459, `benchmark-router-eight-*.0` for1052. Keep campaigns separate. Compare corresponding **timed**, not warmup, profiles. Require one archived profile per completed model run.
7. Finalize/update the two JSON evidence files, `GLM53F_STRATA.md` and `resume-strata.md`, including failures and terminal owner state. Commit only owned files and report hash. No push.
8. No promotion from a small screen. A candidate clearing model gains still needs remaining long-context/qualification checks, including32K when applicable, before replacing the qualified recipe. Current router model scripts cover8K, not32K.
9. Once these measurements finish, prioritize larger prefill costs (MoE/collectives or PP correctness isolation). Do not start speculative architectural changes merely because the current jobs are waiting.

## Recent completed results to preserve

| Work | Evidence / result |
|---|---|
| MLA shared-cache scores | `64536808`;22 runs/38 trials; five pairs+1.1114% decode/+0.1222% prefill; exact IDs/states; no promotion |
| Prescaled six-head MLA values | `5fbc0f66`; native helper1.506071× fast/1.298130× conservative;77760 fixture instances including4320 direct-helper checks |
| Combined values+split6+packed embedding |23 runs/41 trials; five pairs+2.9427% prefill/+0.7268% decode; medians401.299720→413.108692 prefill,34.025137→34.272419 decode; no promotion |
| MTP rejection-only target restore | `9b3d9a1d`;1792 native cases/rank,7 model runs/13 trials; long screen34.780039→35.609712 decode (+2.3855%), prefill−0.5229%; exact IDs/raw states, no promotion |
| Completed reports | `48854841`: `strata-mla-values-model-20261004.json`, `strata-mtp-rejection-restore-model-20261004.json` |
| Router12×16 / expanded follow-up | `3d74197b` / `ed1d05a6`; currently queued, not qualified |

MTP acceptance500/523 (~95.6%) did not produce a large model gain. Rejection-only restore timed prefill+decode49.179866s versus plain49.286350s at1024 outputs (~0.22% unpaired advantage; load/IO excluded). Do not treat acceptance as speedup or these timed phases as full serving latency.

Many prior prototypes were rejected (dense tile64, whole expert group192, coarse expert panels, and others); see long resume before repeating them.

PP3×TP4 is implemented but **not qualified**: short real-model IDs first differ at index19, with early-layer rounding/quantization divergence. Dense virtual-TP12 reconstruction and isolated KDA components pass, but full early-layer mHC replay/isolation is still needed. See `strata-cross-layout-isolation-20261003.json`; no PP throughput claim is valid.
