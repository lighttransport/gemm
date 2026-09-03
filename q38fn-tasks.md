# Qwen3.8 27B A64FX BF16 TP4 optimization tasks

## Objective and gates

- Hardware: four A64FX nodes, one TP rank/node, 48 pinned threads/rank.
- Execution: launch the uTofu binary directly with `mpiexec`; no `pjsub`.
- Targets: non-MTP decode at least 40 tok/s; exact K=5 MTP at least 60 tok/s.
- Exact hashes: 128 tokens
  `6b136ca08910eb2b47a820d1be2efc5eed1a38c7bf8f8a43de009c6b29f274c2`;
  256 tokens
  `7b86e9830096198c4066689d487ad18b3cd6efbad02626494a0d3fb9460d2f14`.
- Clean-run gate: at least 800 GB/s effective weight bandwidth on every node,
  no more than 3% rank wall-time skew, stable memory below 32 GB/node, 256-byte
  communication alignment, and no inter-CMG weight access.
- Final acceptance: three consecutive exact 256-token runs meeting the target.

The reference MTP result is 53.43 tok/s on clean nodes with replicated NextN,
68.79 ms verification, 18.26 ms drafting, 0.9318 acceptance, and about 868
GB/s/node. Degraded interactive-node measurements are diagnostics only.

## Phase 0: reproduce and profile

- [ ] Re-stage the BF16 TP4 trunk and replicated/sharded NextN images under
  `/local/u14346/` after every node/session restart.
- [ ] Reproduce non-MTP and MTP baselines with direct `mpiexec` on clean nodes.
- [ ] Record per-rank wall time, effective GB/s, collective time, verifier and
  draft time, acceptance, peak memory, and output hash.
- [ ] Use `TF_DPROF=1` to split DeltaNet input/core/output, attention QKV/core/
  output, FFN gate-up/down, vocabulary head, barriers, and all-reduces.
- [ ] A/B replicated versus sharded NextN with otherwise identical settings.

## Phase 1: non-MTP decode to 40+ tok/s

- [x] Route persistent Qwen SSM/attention output and FFN-down collectives
  through the exact reduce-plus-residual callback. The old path remains active
  when projection/collective overlap is requested.
- [ ] Verify that path against both exact hashes and A/B `TP_AR_FUSED_ADD=0`.
  Both token hashes pass; the adjacent bandwidth-degraded 128-token run
  improved 26.19 to 26.83 tok/s (+2.4%). The clean-node gate remains.
- [ ] Sweep BF16-PV prefetch by shape: K=4352 at 8/12/16, and K=5120/6144 at
  4/6/8/12. Promote only a repeatable whole-model gain.
- [ ] Reduce persistent DeltaNet dispatch/barrier cost by grouping independent
  QKV/gate/alpha/beta work while preserving CMG-owned row ranges.
- [ ] Profile the vocabulary head and test local per-rank argmax plus a small
  winner exchange; keep the full-logit route as exact control.
- [ ] Revisit projection/collective overlap only with CMG-local buffers and no
  lost weight-stream worker or reordered exact fold.

Exit only after three clean, exact 256-token runs reach 40 tok/s and beat an
adjacent control.

## Phase 2: MTP to 60+ tok/s

- [ ] Rebaseline the combined exact one-round TP4 fold, fused batch residual,
  compact single-pass BF16 4x5 verifier, K=5, and sharded NextN. At clean-node
  bandwidth the combined changes may already cross 60 tok/s.
- [ ] Measure verifier and proposer separately. At observed acceptance, the
  60 tok/s budget is roughly 75--80 ms total per accepted K=5 round.
- [ ] Keep full-NextN asynchronous drafting disabled: two concurrent weight
  streams contend for the same HBM and previously missed the overlap window.
- [ ] If proposer time remains exposed, build a persistent NextN K-step chain
  retaining workers and recurrent scratch, eliminating wakeups and redundant
  preparation without rereading trunk weights.
- [ ] Tune BF16-PV shapes separately for the K=5 verifier and single-vector
  proposer; do not infer proposer settings from verifier results.
- [ ] Test local vocabulary winner exchange only with exact tie-breaking.
- [ ] Overlap only small communication/preparation work, never two full weight
  streams on the same CMGs.

Exit only after three clean, exact 256-token runs reach 60 tok/s. Report median
and minimum throughput, not only the best run.

## Canonical commands

```sh
cd a64fx/llm

TP_SIZE=4 TP_NEXTN_SHARD=1 \
  TP_STAGE_DIR=/local/u14346/qwen38-bf16-tp4-nextnshard \
  bash run_qwen38_bf16_tp4.sh stage-mtp

TP_SIZE=4 TP_NEXTN_SHARD=0 TP_MAXGEN=256 \
  bash run_qwen38_bf16_tp4.sh bench

TP_SIZE=4 TP_NEXTN_SHARD=1 TP_SPEC_K=5 TP_MAXGEN=256 \
  TP_AR_DETERMINISTIC=1 TP_AR_A2A_TREE=1 TP_AR_FUSED_ADD=1 \
  TF_BF16PV_MTP5_FUSED=1 \
  bash run_qwen38_bf16_tp4.sh mtp-sustained
```

The launcher expands these modes to direct `mpiexec`. Save complete per-rank
output for each accepted or rejected A/B in the ledger.

## Result ledger

| Date | Change/config | Clean? | Exact? | tok/s | Verify ms | Draft ms | GB/s/rank | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| historical | replicated NextN K=5 | yes | yes | 53.43 | 68.79 | 18.26 | ~868 | reference |
| pending | combined optimized MTP | - | - | - | - | - | - | clean rebaseline |
| 2026-09-03 | persistent decode reduce+add | no | 128/256 yes | 26.83 vs 26.19 (128); 31.40 (256) | n/a | n/a | 540--617 (256) | +2.4% adjacent; clean pending |
