# Calibrating the local decode simulators from real A64FX runs

The decode optimization loop runs LOCALLY (decode_sim.py + the qlair A64FX simulator with
`--ranks 8`) to save node-hours. This doc maps the outputs of the calibration job
(`pjsub_glm5_ar_probe_8n.sh`, 8 nodes x ~25 min) onto the simulator knobs, so one cheap job
re-anchors both models.

## Submit + collect

```
ssh fugaku 'cd ~/work/gemm/glm5-1 && pjsub a64fx/glm5/pjsub_glm5_ar_probe_8n.sh'
ssh fugaku 'grep -hE "ARPROBE|KERNBENCH|SELFCHECK|cbatch:|TOKEN DIFF" \
    ~/work/gemm/glm5-1/a64fx/glm5/ar_probe_run_<jobid>/*rank00.txt \
    ~/work/gemm/glm5-1/a64fx/glm5/pjsub_glm5_ar_probe_8n.sh.<jobid>.out'
```

Repeat at other node counts by copying the script and changing `node=/proc=` (16/32 give the
log2(N) round-scaling decode_sim assumes; 96 anchors the production point).

## Phase 1 — ARPROBE → decode_sim.py comm term

`ARPROBE,sum,N=8,robust=1,bf16=0,M=1,bytes=24576,us_per_ar=...,round_us=...`

- `us_per_ar` at robust=1, M=1 is THE production decode AR cost. Feed it:
  `decode_sim.recalibrate(ar_ms=us_per_ar/1000, n_ref=8)` — or, preferred once several N are
  measured, fit `ar = init + rounds*round_ms` across N ∈ {8,16,32,96} and call
  `recalibrate(round_ms=..., init_ms=...)`.
- robust=2 rows quantify the LEAN path win (`TP_AR_ROBUST=2`, the m1_tput(round_ms) lever) —
  this is the number that decides whether bf16 M=1 ≥ 2 tok/s is reachable.
- robust=0 rows bound the tax floor (passive spin; unsafe at scale, reference only).
- The M sweep gives the marginal per-byte cost → check `UTOFU_BW` (1/8e9 s/B assumed).
- `ARPROBE,token78,...,ms_per_token_comm=X` is directly `1000/X` comm-bound tok/s for a
  78-AR decode token — compare with `m1_tput()` before trusting any lever stack.
- `argmax` vs `argmax_n32`: per-stream vs batched head merge — validates cutting M head
  ARs to 1 in the batched head.

## Phase 1 → qlair Tofu model

qlair (`tools/qlair/tofu/qlair-tofu.hh`) charges `1400ns + bytes/6.3GB/s` per put and has no
multi-node topology. Compare `ARPROBE` per-AR against the same tp_ar_8rank run under
`qlair --ranks 8`: the DIFFERENCE is the real-cluster term (incast, OS jitter, robust drain)
qlair cannot see. If the 2-node `us_per_ar` at robust=0 with small M is far from
2×rounds×~1.5µs, revisit the put constants (`cycles_put`); otherwise leave qlair as the
wire-floor model and keep the cluster term in decode_sim only.

## Phase 2 — KERNBENCH → decode_sim BW_NODE + qlair kernel check

`KERNBENCH,shape,rows,cols,M,ms_per_call,GBs,GBs_per_stream`

- decode_sim `BW_NODE` (currently 300e9) := the M=1 `GBs` averaged over the per-token shape
  mix (attention + shared + router + owned experts + head). Update the constant, rerun
  `python3 decode_sim.py`, and the bandwidth-bound ceiling + all predictions re-anchor.
- The M sweep is the batched-decode compute model: `sum(shape ms at M)/M` = compute
  ms/token/stream. If GBs saturates before M=32, batching stops amortizing weight reads
  there — cap the M recommendation accordingly.
- qlair cross-check: run the same binary shapes under `qlair -p` (cycle mode) on the dev
  host and compare GBs — target ≤10% error (QLAIR_VERIFICATION.md methodology).

## Phase 3 — SELFCHECK

`BATCH_SELFCHECK ... MATCH` on real SVE hardware with real 8-rank comm = the batched-MLA
kernel is production-safe at M=1 (the qlair TEST2 result already covers M>1 bit-identity;
a real-weight M>1 A/B at 96n stays on the job-validation list).

## Phase 4 — cbatch A/B → end-to-end batched-decode win

- `TOKEN DIFF` lines must be absent (bd=0 and bd=1 emit identical streams).
- `cbatch: service decode ... agg tok/s` ratio bd=1/bd=0 at 8n, 8 slots, 8-layer synthetic
  ≈ the comm-amortization factor. decode_sim comparison: scale `n_allreduce` to the
  synthetic layer count (8 MoE-ish ARs/token) and N=8 — predicted ratio ≈
  `pred_tok_s(8, M=8) / (8 * pred_tok_s(8, 1))` with the ARPROBE-recalibrated constants.

## After calibration

Update `decode_sim.py` constants (BW_NODE, UTOFU_ROUND/INIT via `recalibrate()` defaults or
inline), re-run the report, and refresh the lever stack in BATCHED_DECODE_PLAN.md /
the job-validation list with the re-anchored predictions.
