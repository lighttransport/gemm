# K3 72-node Resume Notes

## Result (latest run)

- 2026-07-31: `pjsub_k3_probe_72n.sh` with PJM job ID `49868424`
- Outcome: **FAILED**
- Evidence:
  - [pjsub_k3_probe_72n.sh.49868424.out](/vol0006/mdt0/data/hp250467/work/gemm/k3/pjsub_k3_probe_72n.sh.49868424.out):  
    `K3_JOB_TOTAL elapsed_s=8 rc=5`
  - Same output shows:
    - `K3_RESULT status=FAIL reason=non-finite`
    - `tokens_completed=0`
  - Per-rank status files all report numeric failure:
  - `a64fx/k3/logs/probe-72n-49868424/dummy/k3_rank*.status`
  - every rank is `state=numeric-failed reason=non-finite`
- Net result: no useful 72-node decode text was produced in this run.
- 2026-07-31: `pjsub_k3_scale_16n.sh` with PJM job ID `49881342`
- Outcome: **PASS**
- Run type: local pool probe pack (dummy + kda + moe + expert-tp + attention, 16 nodes, 00:30 limit)
- Evidence:
  - [pjsub_k3_scale_16n.sh.49881342.out](/vol0006/mdt0/data/hp250467/work/gemm/k3/pjsub_k3_scale_16n.sh.49881342.out)
  - Dummy gate: `K3_RESULT status=PASS reason=complete tokens_completed=64`
  - KDA probe: `K3 multi-node KDA probe: PASS`
  - MoE probe: `K3 multi-node MoE probe: PASS`
  - Expert-TP probe: `K3 multi-node expert-TP probe: PASS`
  - Attention probe: `K3 RESULT status=PASS reason=complete tokens_completed=256` (layer 3, MLA)

## What to resume

- This was a TP72 probe run (dummy + staged real decode + prefill validation).  
  It failed in the **dummy/probe gate** phase (`non-finite`), so full TP72 real decoding did not establish a successful baseline.

- Resume by rerunning the 72-node probe with reduced run-time and shorter probe:

```bash
cd /vol0006/mdt0/data/hp250467/work/gemm/k3
pjsub a64fx/k3/pjsub_k3_probe_72n_short.sh
pjsub a64fx/k3/pjsub_k3_probe_96n_short.sh
```

- After rerun, archive the new outputs and update this file with:
  - updated job id
  - dummy gate status (`pass`/`numeric-failed`)
  - real decode `K3_RUN ... tokens_completed`
  - `K3_PROFILE` summary

## Follow-up scale results observed locally

- `pjsub_k3_scale_24n.sh` job `49881344`: **PASS**. Dummy, KDA, MoE,
  expert-TP (`24/24` ranks), and attention (`24/24`, 256 tokens) all passed;
  attention checksum disagreement was `0.000e+00`.
- `pjsub_k3_scale_32n.sh` job `49881346`: **PASS**. Dummy, KDA, MoE,
  expert-TP (`32/32` ranks), and attention (`32/32`, 256 tokens) all passed;
  attention checksum disagreement was `0.000e+00`.
- No local output file is currently present for the queued `48`, `72`, or
  `96`-node job IDs above, so those results remain unverified here.

## Current 12-node cache stress

- `TP12`, dummy layers `1-3`, 4,096 tokens, job `49879093`: cache save
  passed `12/12` with zero checksum disagreement, 20 MiB MLA cache, and
  44.46 MiB peak pool.
- Restoring the same cache passed `12/12` with checksum `+4.427299894e+02`
  and the same KDA/MLA health sizes; all 4,096 tokens were restored before
  decode.
- `make -C a64fx/k3 test` passed the native kernel and runtime memory/pool
  gates after the cache error-path changes.

## Additional 12-node cache/context coverage

- Job `49879093`, TP4 over 12 ranks (three independent context groups),
  completed a one-token dummy decode with `12/12` pass markers and zero
  checksum disagreement.
- A missing shared-cache shard exited with `rc=5` in four seconds and
  published `12/12 state=load-failed reason=cache-missing` statuses.
- A one-byte payload mutation exited with `rc=5` in three seconds and
  published `12/12 state=load-failed reason=cache-corrupt` statuses.
- A truncated shard exited with `rc=5` in four seconds and published `12/12
  state=load-failed reason=cache-corrupt` statuses after normalizing short-read
  `EOF` before the collective error reduction. This fixes the prior case where
  a negative local error could be hidden by healthy ranks' zero in a max
  reduction.
- The focused launcher, adapter, server, CLI, queue, and OpenAI compatibility
  suites pass `113` tests; the FP32/BF16 cache-format matrix is also covered.

## Additional 12-node communication coverage

- Job `49879093`, TP12, layer 3 MLA, four-token runs passed `12/12` with zero
  disagreement for hierarchical `ar-groups=3` plus ACK/robust-1, hierarchical
  deterministic/robust-2, flat ACK+deterministic/robust-1, and hierarchical
  ACK+prefetch configurations.
- With `K3_DEBUG_COMM_DROP_N=7` and `--comm-ack 1`, the injected-loss run
  completed eight MLA tokens, passed `12/12`, and retained zero disagreement.
  This exercises actual ACK/retransmit recovery rather than only option parsing.

## Prefix-cache continuation coverage

- Job `49879093`, TP12, layer 3 MLA: a direct 128-token run passed `12/12`
  with checksum `+4.395680055e+02`.
- Saving a 64-token prefix and loading it into a 128-token destination now
  continues from token 64, completes `128/128` on all ranks, and produces the
  exact same checksum `+4.395680055e+02` as the direct run.
- The cache loader now validates serialized MLA dimensions using the file's
  own capacity and expands per-head token strides when restoring a shorter
  prefix into a larger destination. Same-size and BF16/FP32 compatibility
  checks remain enforced.
- The same TP12 64-to-128 continuation was repeated with FP32 cache storage:
  `12/12` passed and matched the direct FP32 baseline checksum
  `+4.395681073e+02` exactly.
- A mixed TP12 three-layer run (`2` KDA + `1` MLA) also passed the full
  64-to-128 continuation: direct and restored checksums both equal
  `+4.428643176e+02` with `12/12` pass markers and zero disagreement. This
  exposed and fixed a KDA q/k layout bug: KDA consumes 128-wide per-head
  vectors, while the synthetic preparer had been writing them at the MLA
  192-wide stride, allowing stale tail data to corrupt restored continuations.
- The same mixed continuation with FP32 MLA cache storage passed `12/12` with
  zero disagreement and exact direct/restored checksum `+4.428642457e+02`.
- With 1 MiB asynchronous prefetch, ACK, deterministic hierarchical reduction,
  and mixed KDA+MLA layers enabled together, the TP12 64-to-128 continuation
  passed `12/12` with zero disagreement and exact checksum `+4.428643176e+02`.
- The same concurrent mixed continuation was repeated with TP4 contexts across
  all 12 ranks (three independent groups). Every group matched its direct
  128-token checksum exactly, with `12/12` pass markers and zero disagreement.
- With deterministic payload drops (`K3_DEBUG_COMM_DROP_N=7`) added to the
  prefetch/ACK configuration, TP12 mixed continuation still passed `12/12`:
  direct and restored checksums both equal `+4.428643176e+02`, with zero
  disagreement and successful profile output.
- llmgr now maps OpenAI `prompt_cache_key` to a hashed, model/layout-scoped
  shared K3 cache directory when explicit paths are absent; it only attempts a
  load after all expected shards exist. Explicit paths and non-K3 adapters are
  unchanged. The complete launcher + llmgr regression set passes `117` tests,
  including an endpoint-level prompt-cache-key submission check.

## Current queued reruns (2026-07-31)

- `pjsub_k3_probe_72n_short.sh` submitted as job `49881338` (72 nodes, 00:30:00, queue)
- `pjsub_k3_probe_96n_short.sh` submitted as job `49881340` (96 nodes, 00:30:00, queue)
- `pjsub_k3_scale_24n.sh` submitted as job `49881344` (24 nodes, 00:30:00, queue)
- `pjsub_k3_scale_32n.sh` submitted as job `49881346` (32 nodes, 00:30:00, queue)
- `pjsub_k3_scale_48n.sh` submitted as job `49881348` (48 nodes, 00:30:00, queue)

## Local 12-node module verification performed during this run

- `a64fx/k3/logs/quick-12n-module-20260731-154514`
  - `make -C a64fx/k3 test` → PASS
  - `run_k3_ep.sh --mode dummy --nodes 12 --layer 0 --tokens 1024` → PASS (12/12)
  - `run_kda_probe_mpi.sh --nodes 12 --layer 0 --head 0` → PASS
  - `run_moe_probe_mpi.sh --nodes 12 --layer 1 --experts-per-rank 4 --threads 48` → PASS
  - `run_expert_tp_probe_mpi.sh --nodes 12 --layer 1 --experts 16 --threads 48` → PASS
  - `run_k3_ep.sh --mode real --nodes 12 --layer 3 --tokens 256 --mla-cache-bf16` → PASS

## 2026-07-31 12-node cache continuation expansion

- FP32 mixed KDA+MLA (`2 KDA + 1 MLA`) with hierarchical AR groups, robust-1 ACK/retransmit, deterministic reduction, 1 MiB prefetch, and injected payload drops (`K3_DEBUG_COMM_DROP_N=7`) passed baseline/save/restore on all 12 ranks. The 64-token FP32 cache restored into the 128-token target with exact checksum `+4.428642457e+02`, zero disagreement, and `pass_markers=12/12`. Results: `a64fx/k3/logs/prefix-fp32-drop-1785496327`.
- BF16 mixed KDA+MLA with the same transport stress and `--no-fused-team` passed baseline/save/restore on all 12 ranks. The 64-token cache restored into the 128-token target with exact checksum `+4.428643176e+02`, zero disagreement, and `pass_markers=12/12`. Results: `a64fx/k3/logs/prefix-nofused-1785496357`.
- Cache format rejection: a valid 12-rank BF16 64-token cache was loaded with FP32 settings. All ranks rejected it as `cache-incompatible`; launcher returned `rc=5` with `pass_markers=0/12` and no collective timeout. Results: `a64fx/k3/logs/cache-incompat-1785496424`.
- FP32 TP4 multi-context continuation: 12 nodes formed three independent TP4 groups. Under hierarchical robust-1 ACKs, deterministic reduction, 1 MiB prefetch, and injected drops, baseline, 64-token save, and 128-token restore all passed for every group (`12/12`, zero disagreement). Restored checksums matched baselines: group 0 `+4.428642098e+02`, group 1 `-4.333240691e+02`, group 2 `-4.435292319e+02`. Results: `a64fx/k3/logs/prefix-fp32-tp4-1785496455`.
- OpenAI cache API coverage expanded: `prompt_cache_key` now has tests for model/layout isolation, incomplete shard suppression, and `/v1/responses` endpoint propagation. Focused server suite: `48` passed; combined launcher/llmgr suite: `120` passed.
- FP32 flat robust-2 continuation without ACKs: hierarchical-independent flat allreduce (`ar_groups=0`), deterministic reduction, robust-2 polling, and 1 MiB prefetch passed baseline, 64-token save, and 128-token restore on 12 ranks. Restore exactly matched baseline checksum `+4.428642533e+02`, with zero disagreement and `pass_markers=12/12`. Results: `a64fx/k3/logs/prefix-fp32-flat-r2-1785496598`.
- TP4 multi-context corruption isolation: after saving three FP32 context groups, one payload byte in group 0 shard `..._g000_r000.bin` was changed. Restore rejected only group 0 as `cache-corrupt`; groups 1 and 2 passed, yielding `rc=5` and `pass_markers=8/12` without a collective hang. Results: `a64fx/k3/logs/cache-corrupt-tp4-1785496641/load-payload`.
- Cache-path reuse: the same 12-rank BF16 cache directory was saved at 128 tokens, overwritten by a 64-token save, then restored at 128 tokens. Restore matched the fresh baseline checksum `+4.428643176e+02`, with `12/12` markers and zero disagreement. Results: `a64fx/k3/logs/cache-reuse-1785496714`.
- Longer prefix expansion: under hierarchical robust-2 ACKs, deterministic reduction, 2 MiB prefetch, and injected drops, a 128-token BF16 cache restored into a 256-token target on 12 nodes. Restore matched the fresh 256-token baseline checksum `+4.432389704e+02`, with `12/12` markers and zero disagreement. Results: `a64fx/k3/logs/prefix-256-1785496774`.
- Partial layer-window continuation: layer range `[2,4)` (`1 KDA + 1 MLA`) passed 12-node BF16 baseline, 64-token save, and 128-token restore with robust-1 ACKs, deterministic hierarchical reduction, 1 MiB prefetch, and injected drops. Restore matched baseline checksum `+4.416268038e+02`, with `12/12` markers and zero disagreement. Results: `a64fx/k3/logs/prefix-layer2-1785496822`.
- Nonzero layer-window FP32 continuation: layer range `[3,5)` (`1 KDA + 1 MLA`) passed baseline, 64-token save, and 128-token restore on 12 nodes with robust-2 ACKs, deterministic hierarchical reduction, 1 MiB prefetch, and injected drops. Restore matched baseline checksum `+4.416997178e+02`, with `12/12` markers and zero disagreement. Results: `a64fx/k3/logs/prefix-mla-layer3-1785496867`.
- KDA-only edge case: single layer range `[1,2)` (`1 KDA + 0 MLA`, zero MLA cache) passed BF16 baseline, 64-token save, and 128-token restore on 12 nodes with robust-1 ACKs, deterministic hierarchical reduction, 1 MiB prefetch, and injected drops. Restore matched baseline checksum `+4.255938671e+02`, with `12/12` markers and zero disagreement. Results: `a64fx/k3/logs/prefix-layer1-only-1785496909`.
- Second KDA-only window: FP32 layer range `[4,5)` (`1 KDA + 0 MLA`, zero MLA cache) passed baseline, 64-token save, and 128-token restore on 12 nodes with robust-2 ACKs, deterministic hierarchical reduction, 1 MiB prefetch, and injected drops. Restore matched baseline checksum `+4.255762701e+02`, with `12/12` markers and zero disagreement. Results: `a64fx/k3/logs/prefix-layer4-only-1785496964`.
- TP6 topology coverage: 12 nodes formed two independent TP6 groups. BF16 baseline, 64-token save, and 128-token restore passed with robust-1 ACKs, deterministic hierarchical reduction, 1 MiB prefetch, and injected drops. Restored checksums matched group baselines: group 0 `+4.428643334e+02`, group 1 `-4.333238839e+02`; all phases had `12/12` markers and zero disagreement. Results: `a64fx/k3/logs/prefix-tp6-1785497005`.
- TP3 topology coverage: 12 nodes formed four independent TP3 groups. FP32 baseline, 64-token save, and 128-token restore passed with robust-2 ACKs, deterministic hierarchical reduction (`ar_groups=3`), 1 MiB prefetch, and injected drops. Restored checksums matched group baselines: group 0 `+4.428642015e+02`, group 1 `-4.333240835e+02`, group 2 `-4.435292525e+02`, group 3 `+4.428068897e+02`; every phase had `12/12` markers and zero disagreement. Results: `a64fx/k3/logs/prefix-tp3-1785497061`.
- No-ACK transport coverage: robust-2 hierarchical BF16 run with deterministic reduction, 1 MiB prefetch, injected drop (`K3_DEBUG_COMM_DROP_N=7`), and `comm_ack=0` still completed on 12 nodes with `12/12` markers, zero disagreement, and checksum `+4.406585343e+02`. The injected-drop point is tolerated without retransmission in this configuration. Results: `a64fx/k3/logs/comm-drop-noack-1785497104/run`.
- Launcher regression coverage: added a test for rejecting `--ar-groups=1` with `--tp-nodes=3`, matching the live validation boundary. Launcher suite: `28` passed; combined launcher/llmgr suite: `121` passed.
- TP2 topology coverage: 12 nodes formed six independent TP2 groups, the highest context count tested. BF16 baseline, 64-token save, and 128-token restore passed with robust-1 ACKs, deterministic hierarchical reduction, 1 MiB prefetch, and injected drops; every phase had `12/12` markers and zero disagreement. Restored checksums matched group baselines: g0 `+4.428643017e+02`, g1 `-4.333238721e+02`, g2 `-4.435292589e+02`, g3 `+4.428068983e+02`, g4 `-4.320930310e+02`, g5 `-4.435552872e+02`; peak pool was `146.23 MiB`. Results: `a64fx/k3/logs/prefix-tp2-1785497194`.
- TP1 maximum-context coverage: 12 nodes formed twelve independent TP1 contexts (`ar_groups=0`, flat allreduce). BF16 baseline, 64-token save, and 128-token restore passed with robust-2 ACKs, deterministic reduction, 1 MiB prefetch, and injected drops. Every phase had `12/12` markers and zero disagreement; peak pool usage was `289.88 MiB`. Results: `a64fx/k3/logs/prefix-tp1-1785497242`.
- Final validation snapshot for this 12-node pass: native `make -C a64fx/k3 test` passed all kernel correctness and runtime memory/pool checks; combined launcher/llmgr suite passed `121` tests.
- Combined load/save pipeline: a 64-token BF16 source cache was loaded while decoding to 128 tokens and simultaneously saved to a new destination. A subsequent 128-token restore from that destination passed with zero additional decode tokens, checksum `+4.428643176e+02`, `12/12` markers, and zero disagreement. Results: `a64fx/k3/logs/cache-load-save-pipeline-1785497338`.
- Nondeterministic reduction coverage: BF16 12-node baseline, 64-token save, and 128-token restore passed with `comm_deterministic=0`, robust-2 ACKs, deterministic-independent hierarchical transport, 1 MiB prefetch, and injected drops. Restore matched baseline checksum `+4.428643176e+02`, with `12/12` markers and zero disagreement. Results: `a64fx/k3/logs/prefix-nondeterministic-1785497382`.
- In-place prompt-cache lifecycle: a 64-token BF16 cache was loaded and expanded to 128 tokens while saving back to the same directory, then restored from that same path with zero decode tokens. All phases passed `12/12`, zero disagreement, and checksum `+4.428643176e+02`. Results: `a64fx/k3/logs/cache-inplace-pipeline-1785497459`.
- FP32 in-place prompt-cache lifecycle: 64-token FP32 state was loaded and expanded to 128 tokens while saving back to the same directory, then restored with zero decode tokens. All phases passed `12/12`, zero disagreement, and checksum `+4.428642457e+02`. Results: `a64fx/k3/logs/cache-inplace-fp32-1785497499`.
- FP32 no-prefetch control: with `prefetch_mib=0`, robust-2 ACKs, deterministic hierarchical reduction, and injected drops, 12-node baseline, 64-token save, and 128-token restore all passed. Restore matched baseline checksum `+4.428642457e+02`, with `12/12` markers and zero disagreement. Results: `a64fx/k3/logs/prefix-fp32-noprefetch-1785497546`.
- Minimum polling coverage: robust-2 BF16 in-place 64-to-128 cache expansion with `comm_poll_spins=1`, ACKs, deterministic hierarchical reduction, and prefetch passed on 12 nodes. Final full-cache verification decoded zero tokens, with `12/12` markers, zero disagreement, and checksum `+4.428643176e+02`. Results: `a64fx/k3/logs/prefix-poll1-1785497589`.
- Maximum polling coverage: robust-2 FP32 in-place 64-to-128 cache expansion with `comm_poll_spins=1024`, ACKs, deterministic hierarchical reduction, and prefetch passed on 12 nodes. Final verification decoded zero tokens, with `12/12` markers, zero disagreement, and checksum `+4.428642457e+02`. Results: `a64fx/k3/logs/prefix-poll1024-1785497632`.
- Cache capacity safety: a valid 128-token FP32 cache was requested for a 64-token target. All 12 ranks rejected it as `cache-incompatible`; launcher returned `rc=5` with `pass_markers=0/12` and no hang. Results: `a64fx/k3/logs/cache-too-large-1785497675`.
- Layer-scope isolation: a BF16 cache saved for `[1,4)` was requested for `[2,4)`. Layer-qualified shard naming prevented reuse; all ranks reported `cache-missing`, launcher returned `rc=5` with `0/12` markers, and no hang occurred. Results: `a64fx/k3/logs/cache-layer-mismatch-1785497707`.
- Prompt-cache key isolation expanded to cover `np` and `tp_np` changes in addition to model, layer count, and cache dtype. Focused server suite: `48` passed; combined launcher/llmgr suite: `121` passed.
- Non-fused FP32 in-place pipeline: with `fused_team=0`, a 64-token cache expanded in place to 128 tokens and then restored with zero decode tokens. All phases passed `12/12`, zero disagreement, and checksum `+4.428642457e+02`. Results: `a64fx/k3/logs/cache-inplace-nofused-fp32-1785497794`.
- Responses API cache-hit coverage: added an integration test proving `/v1/responses` maps a complete K3 `prompt_cache_key` directory to identical `cache_load`/`cache_save` paths. Focused server suite: `49` passed; combined launcher/llmgr suite: `122` passed.
- Responses API partial-cache safety: added an endpoint test proving an 11-of-12 K3 shard directory is not loaded, while a new `cache_save` path is still supplied. Focused server suite: `50` passed; combined launcher/llmgr suite: `123` passed.
- Combined TP4/non-fused/FP32 coverage: three TP4 contexts ran same-directory 64-to-128 expansion with `fused_team=0`, robust-2 ACKs, deterministic hierarchical reduction, and prefetch. Final verification decoded zero tokens; all phases had `12/12` markers and zero disagreement. Restored checksums matched groups: g0 `+4.428642098e+02`, g1 `-4.333240691e+02`, g2 `-4.435292319e+02`. Results: `a64fx/k3/logs/cache-inplace-tp4-nofused-fp32-1785497983`.
- Fused-thread override coverage: FP32 in-place 64-to-128 cache expansion with `fused_threads=2` passed on 12 nodes; final verification decoded zero tokens, with `12/12` markers, zero disagreement, and checksum `+4.428642457e+02`. Results: `a64fx/k3/logs/cache-inplace-fused2-1785498025`.
- Fixed cache collective diagnostic propagation in `a64fx/k3/k3_ep_runner.c`: collective load/reduction failures now set the reported code to `EIO` instead of leaving a healthy rank's local `rc=0`. Native `make -C a64fx/k3 test` passed; mixed-generation 12-node restore now reports `rc=5 reason=cache-collective` and exits without hanging. Results: `a64fx/k3/logs/cache-mixed-generation-1785498085/load-fixed`.
- Post-fix regression snapshot: native `make -C a64fx/k3 test` passed all kernel correctness and runtime memory/pool checks; combined launcher/llmgr suite passed `123` tests.
- Fixed automatic prompt-cache completeness scoping in `a64fx/llmgr/llmgr_server.py`: `prompt_cache_key` reuse now counts only canonical shards matching current layer range, world-node count, and thread layout, preventing stale historical shards from triggering a load. `/kv stats` broad inventory behavior remains unchanged. Focused server suite: `51` passed; combined launcher/llmgr suite: `124` passed.
- Automatic AR-group coverage: BF16 in-place 64-to-128 cache expansion with `ar_groups=auto`, robust-1 ACKs, deterministic hierarchical reduction, and prefetch passed on 12 nodes. Final verification decoded zero tokens, with `12/12` markers, zero disagreement, and checksum `+4.428643176e+02`. Results: `a64fx/k3/logs/cache-inplace-argroups-auto-1785498374`.
- TP4 automatic AR-group coverage: `ar_groups=auto` selected flat reduction (`ar_groups=0`) for three TP4 contexts. BF16 same-directory 64-to-128 expansion and zero-token verification passed with robust-2 ACKs, deterministic reduction, and prefetch; all phases had `12/12` markers and zero disagreement. Restored checksums: g0 `+4.428643385e+02`, g1 `-4.333238602e+02`, g2 `-4.435292370e+02`. Results: `a64fx/k3/logs/cache-inplace-tp4-auto-1785498453`.

## 12-node coverage index

- Topologies: TP1/12 contexts, TP2/6, TP3/4, TP4/3, TP6/2, and TP12/1; all exercised cache save/restore, with TP4 and TP12 also covering automatic AR selection.
- Cache formats: BF16 and FP32 MLA state; KDA-only windows, mixed KDA+MLA windows, layer-scoped caches, 128-to-256 expansion, same-path in-place updates, and load/save pipelines.
- Transport: flat and hierarchical reductions, robust-1 and robust-2 polling, ACK/retransmit on and off, deterministic and nondeterministic reductions, polling bounds 1 and 1024, prefetch on/off, injected drops, and fused/non-fused team execution.
- Failure safety: missing, truncated, payload-corrupt, mixed-generation, incompatible dtype/capacity, stale-layout, and partial-shard cases fail without hangs and report actionable status.
- API: OpenAI Chat Completions and Responses prompt-cache key mapping, complete-cache reuse, partial-cache suppression, topology/layout isolation, in-place persistence, and native-request field filtering.
- Software baseline: native K3 tests pass; combined launcher/llmgr suite currently passes `124` tests.
- Added reusable `a64fx/k3/run_12n_cache_matrix.sh` harness for the standard dummy 12-node in-place lifecycle. It supports environment overrides for TP topology, BF16/FP32, token counts, AR grouping, polling, ACKs, deterministic reduction, and prefetch. Default harness run passed save, same-directory expansion, and zero-token verification with `12/12` markers; results: `a64fx/k3/logs/cache-matrix-1785498595`.
- Harness override validation: `run_12n_cache_matrix.sh` passed with `DTYPE=fp32`, `TP_NODES=4`, `AR_GROUPS=auto`, robust-2, ACKs, and injected drops. Three TP4 contexts completed same-directory expansion and zero-token verification with `12/12` markers and zero disagreement; results: `a64fx/k3/logs/cache-matrix-override-1785498650`.
- Harness input guard: `DTYPE=fp8` is rejected locally with `rc=2` before any runner launch or cache creation.

- Harness portability validation: invoked `a64fx/k3/run_12n_cache_matrix.sh` from `/tmp` with `CACHE_TOKENS=32`, `TOKENS=64`. The default TP12 BF16 source save, in-place expansion, and zero-token verification passed `12/12` with zero disagreement. Results: `a64fx/k3/logs/cache-matrix-cwd-1785498741`.
- Harness oracle hardening: `run_12n_cache_matrix.sh` now normalizes relative result roots to absolute paths, captures each phase log, and requires `rc=0`, all rank status markers, at least one group `K3_RESULT`, zero disagreement, and the expected token count (`0` for full-cache verification). Added `a64fx/k3/test_cache_matrix_script.py` covering a complete lifecycle and missing-rank rejection. The real relative-root TP12 BF16 `32->64` run passed all phases with `12/12` markers and zero disagreement: `a64fx/k3/logs/cache-matrix-oracle-pass-1785499083`. Focused launcher/llmgr plus harness suite: `95` tests passed.
- Harness negative-result coverage: the reusable harness test now also rejects a complete rank set with nonzero `disagreement`; the dedicated harness suite passes `3/3`.
- Cache artifact-integrity coverage: the matrix harness now requires exactly `NODES` canonical cache shard files after save and in-place phases, catching silent local cache-save failures that aggregate launcher status can miss. TP12 BF16 `32->64` real run passed all lifecycle phases, `12/12` rank markers, exactly 12 shards, zero disagreement, and zero-token verification. Results: `a64fx/k3/logs/cache-matrix-artifact-1785499214`.

## Laguna S-2.1 12-node decode status (as of 2026-08-01)

- The latest 12-node Laguna S-2.1 long-context quality checkpoints available in this
  tree are the following July 30 runs (all from 12-node int4):
  - `a64fx/laguna-s21/gen_20260730-221203` (16K+ context, 16,377 prefill tokens, decode 123 tok at 12.1 tok/s)
  - `a64fx/laguna-s21/gen_20260730-222040` (32K+ context, 32,767 prefill tokens, decode 123 tok at 7.8 tok/s)
  - `a64fx/laguna-s21/gen_20260730-230226` (32K+ context, 32,767 prefill tokens, decode 123 tok at 9.9 tok/s)
  - `a64fx/laguna-s21/gen_20260730-232203` (32K+ context, 32,767 prefill tokens, decode 123 tok at 10.0 tok/s)
  - `a64fx/laguna-s21/gen_20260730-234401` (32K+ context, 32,767 prefill tokens, decode 123 tok at 17.0 tok/s)
  - `a64fx/laguna-s21/gen_20260731-005642` (65K+ context, 65,525 prefill tokens, decode 123 tok at 12.7 tok/s)
- Generation quality on those long-context runs was consistent:
  - Sampling line in all runs: `temp=0.70 top_k=20 top_p=0.95 min_p=0.00`.
  - `lockstep: all ... token picks agreed across ranks` in all runs.
  - Decoded output from `gen.ids` reconstructs to valid C++ solutions for the
    requested fenced-program recovery task in each run (exactly the expected
    `assert`/`cout` block with `314159`, `cobalt-orchid`, `271828`).
- A job from `output.49899304` remains the most recent non-user-facing run artifact in
  this workspace and is a staging regression case: `laguna_s21_stage` was invoked with
  `--expert-groups` against a binary lacking that flag support.

### Recommended resume/next pass command (12 nodes)

- If you want a clean rerun with current code on 12 nodes, generate prompts with the
  repository helper first (16K or 32K target lengths):
  - `python3 a64fx/laguna-s21/tools/make_long_context.py --target 16384 --out /shared/laguna-16k.ids --metadata /shared/laguna-16k.meta`
  - `python3 a64fx/laguna-s21/tools/make_long_context.py --target 32768 --out /shared/laguna-32k.ids --metadata /shared/laguna-32k.meta`
- Then run a quality pass (set `MODEL`/`STAGE` paths explicitly in your environment if needed):
  - `cd /vol0006/mdt0/data/hp250467/work/gemm/k3/a64fx/laguna-s21`
  - `./run_laguna_s21_12n.sh generate --fp8 --ids /shared/laguna-32k.ids --chat \"You are a meticulous senior software engineer...\" --max-new 123 --no-stage --quality-cpp --prompt-cache /shared/laguna-system-prefix.ids --maxpos 34823 --np 12`
- Capture job output and `laguna_rank00.txt` as:
  - `output_dir/a64fx/laguna-s21/gen_<timestamp>/laguna_rank00.txt`
  - `output_dir/a64fx/laguna-s21/gen_<timestamp>/gen.ids` (decode/quality validation).
- For a no-stage rerun, ensure all ranks already share valid `STAGE` blobs from the
  matching `--expert-groups` configuration before launching.

## Revisit attempt (2026-08-01 17:00 JST)

- I rechecked the 12-node Laguna S-2.1 path from this workspace and confirmed:
  - Existing long-context 12-node runs remain valid (`16k/32k/65k` checkpoints listed above),
    with sampling at `temp=0.70 top_k=20 top_p=0.95`, lockstep agreement, and expected
    C++ recovery output.
  - `output.49899304` is still a known stale regression artifact from an earlier
    staging/build mismatch (`--expert-groups` passed to a stager binary that did not accept it).
- In this environment I fixed the stager flag parity so `laguna_s21_stage` now accepts
  `--expert-groups` and emits staged headers including that value. This closes the
  local CLI mismatch we were hitting during stage prep.
- A true 12-rank interactive launch is not available in this session’s MPI context, so
  a full authorized 12-node decode quality run could not be executed end-to-end here.
- Resume command for next authorized run (12-node interactive/job allocation):
  - `cd /vol0006/mdt0/data/hp250467/work/gemm/k3/a64fx/laguna-s21`
  - `./run_laguna_s21_12n.sh stage --np 12 --model-dir /home/u14346/models/laguna-s21-int4 --stage-dir /local/$USER/laguna-s21-int4-ep12 --nshards 15 --expert-groups 1`
  - `./run_laguna_s21_12n.sh generate --np 12 --no-stage --model-dir /home/u14346/models/laguna-s21-int4 --ids /shared/laguna-32k.ids --chat \"You are a meticulous senior software engineer...\" --max-new 123 --quality-cpp --prompt-cache /shared/laguna-system-prefix.ids --maxpos 65536`


## 12-node Laguna S-2.1 revisit status (2026-08-01 17:37 JST)

- 12-node interactive execution is still not runnable from this chat session without a real multi-node PJM allocation.
- I created fresh long-context inputs for local reuse:
  - `a64fx/laguna-s21/long16k.ids` (`16377` tokens, target `16384`)
  - `a64fx/laguna-s21/long32k.ids` (`32767` tokens, target `32768`)
- Metadata produced:
  - `a64fx/laguna-s21/long16k.meta`
  - `a64fx/laguna-s21/long32k.meta`
- Recommended restart command on a 12-node allocation (with fresh stage from fp8 or int4 as needed):
  - `cd /vol0006/mdt0/data/hp250467/work/gemm/k3/a64fx/laguna-s21`
  - `./run_laguna_s21_12n.sh generate --fp8 --no-stage --np 12 --ids /vol0006/mdt0/data/hp250467/work/gemm/k3/a64fx/laguna-s21/long32k.ids --chat "You are a meticulous senior software engineer" --max-new 123 --quality-cpp --prompt-cache /tmp/laguna-system-prefix.ids --sample --temp 0.70 --top-p 0.95 --seed 305441741 --maxpos 34823`
  - For 16K prefix, use `long16k.ids` and `--maxpos 20000` (or larger)
- Historical 12-node result checks to carry forward:
  - Last successful long-context checkpoints remain `temp=0.70 top_k=20 top_p=0.95` with lockstep agreement and valid C++ recovery text (e.g. `314159 cobalt-orchid 271828`) in `gen_20260730-*` and `gen_20260731-005642`.
- Note: the most recent transient local attempts in this session (`gen_20260801-1737xx`) are incomplete/inconclusive due environment/runtime interruptions; use the above fresh IDs + job allocation for a clean validation.

## 12-node Laguna S-2.1 revisit check (2026-08-01 session)

- Completed quick launcher validation for the 12-node path in this session:
  - `./run_laguna_s21_12n.sh self-test --fp8 --np 12` passes (`Laguna S21 ABI self-test: PASS`).
- No successful clean end-to-end 12-node generate run was produced in this chat session because the interactive environment does not provide a real multi-node allocation.
- There are multiple stale/incomplete 12-node `gen_20260801-17xxxx` artifacts that are not valid quality baselines:
  - `gen_20260801-155151`: staging artifacts missing (`rankXX.manifest` not found) due `/local/u14346/laguna-s21-ep12` mismatch
  - `gen_20260801-1723xx` and `gen_20260801-173732`: wrong checkpoint tensor set (`missing tensor model.layers.1.mlp.experts.*`) or stale manifest (`/local/u14346/laguna-s21-ep1`)
  - `gen_20260801-00xx`/`-16xx`: timeout/rpc-synchronization failures (`rc=110`) during `bcast/wait`
  - `gen_20260801-083047/48` etc: missing local prompt-id files (`/tmp/laguna-8k.ids`)
- The last valid long-context quality baselines still in-tree remain the July 30/31 12-node runs listed in prior section (16k, 32k, 65k cases), all with `temp=0.70 top_k=20 top_p=0.95` and lockstep agreement.

Next clean rerun (12-node allocation):

```bash
cd /vol0006/mdt0/data/hp250467/work/gemm/k3/a64fx/laguna-s21
./run_laguna_s21_12n.sh stage --fp8 --np 12 --model-dir /home/u14346/models/laguna-s21-fp8   --stage-dir /local/$USER/laguna-s21-fp8-ep12 --nshards 24 --expert-groups 1

./run_laguna_s21_12n.sh generate --fp8 --np 12 --no-stage --model-dir /home/u14346/models/laguna-s21-fp8   --stage-dir /local/$USER/laguna-s21-fp8-ep12 --ids /vol0006/mdt0/data/hp250467/work/gemm/k3/a64fx/laguna-s21/long16k.ids   --chat "You are a meticulous senior software engineer" --sample --temp 0.70 --top-p 0.95 --seed 305441741   --max-new 123 --quality-cpp --prompt-cache /tmp/laguna-system-prefix.ids --maxpos 17000

# 32K pass (optional)
./run_laguna_s21_12n.sh generate --fp8 --np 12 --no-stage --model-dir /home/u14346/models/laguna-s21-fp8   --stage-dir /local/$USER/laguna-s21-fp8-ep12 --ids /vol0006/mdt0/data/hp250467/work/gemm/k3/a64fx/laguna-s21/long32k.ids   --chat "You are a meticulous senior software engineer" --sample --temp 0.70 --top-p 0.95 --seed 305441741   --max-new 123 --quality-cpp --prompt-cache /tmp/laguna-system-prefix.ids --maxpos 36000
```

Required for clean resume:

- Use a writable shared path for prompt/id and prompt-cache files visible on all nodes.
- Ensure `--stage` artifacts are regenerated once per allocation if stale/missing.
- Keep `--no-stage` only after a successful prior stage pass with matching `--nshards` + `--expert-groups`.
