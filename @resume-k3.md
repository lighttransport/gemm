# K3 resume handoff

Last refreshed: 2026-08-05, during live 12-node interactive job `50000128`
(12 nodes, 6 h, started 18:35).

**Environment note that costs time if forgotten:** this session runs on a
compute node. `pjsub` / `pjstat` are **not** on PATH here. Everything to do with
the batch queue goes through `ssh login1 '<cmd>'` (verified working, lands on
`fn01sv01`).

---

## 1. Batch queue state

| job | what | nodes | state |
|---|---|---|---|
| `50005954` | `pjsub_k3_full_96n_short_1h.sh`, the real full-precision run, 3 h, 87Gi, fixed `-x` | 96 | QUE |
| `50004855` | canary c7: 96n, 80Gi + bare `-x` | 96 | QUE |
| `50004857` | canary c8: 96n, 87Gi, no bare `-x` | 96 | QUE |
| `49996689`, `49997180` | IQ1 GGUF text smoke jobs | 32 | QUE |

Superseded: `50001400` (deleted — it still carried the bare `-x` directives).

Backfill matters: the estimated start times are pessimistic. The 96-node
canaries were estimated 08/10 and actually ran the same evening, ~1.5 h after
submission.

## The GATE CHECK root cause: bare `#PJM -x NAME`

`49973647` (and `49959412`, `49973486`) ended `ST=ERR LAST=QUE REASON=GATE CHECK
PC=3`, `JOB START DATE: -`, with no stdout and no log directory. The script
creates its log root within seconds of starting, so the body never ran. This was
**not** a runtime, numerical, or collective failure.

Cause: the script carried 17 **bare `#PJM -x NAME`** directives (no `=value`).
`pjsub` accepts them at submit time, then the job fails the gate check. The
supported form is `-x NAME=value`.

Canary evidence (`a64fx/k3/canary/`, outputs archived in
`a64fx/k3/logs/quant-bench-live12-50000128/`):

| canary | job | nodes | directives | result |
|---|---|---|---|---|
| c1 | 50001019 | 1 | 87Gi, no bare `-x` | ran |
| c2 | 50001024 | 1 | 80Gi, no bare `-x` | ran |
| c3 | 50001026 | 1 | 87Gi, no retention_state | ran |
| c4 | 50001030 | 1 | no `--llio` | ran |
| **c5** | **50001241** | **96** | **87Gi + bare `-x`** | **ERR / GATE CHECK** |
| **c6** | **50001247** | **96** | **80Gi, no bare `-x`** | **ran** |
| **c7** | **50004855** | **96** | **80Gi + bare `-x`** | **ERR / GATE CHECK** |
| **c8** | **50004857** | **96** | **87Gi, no bare `-x`** | **ran** |
| c9 | 50005790 | 1 | command-line `-x NAME=value` | ran, values propagated |

The 96-node 2x2 is unambiguous:

|            | bare `-x`            | no bare `-x`     |
|------------|----------------------|------------------|
| **80Gi**   | c7 -> GATE CHECK     | c6 -> ran        |
| **87Gi**   | c5 -> GATE CHECK     | c8 -> ran        |

Bare `-x` is necessary and sufficient; **localtmp size is irrelevant**. c8 is
exactly the configuration job 50005954 now uses. The failure only reproduces at
96 nodes — 1-node canaries pass with the same directives, which is why the first
bisect round was inconclusive and why an early guess that the 80Gi->87Gi bump
was to blame was wrong.

`pjacl` puts the localtmp ceiling at 89278Mi (87.18 GiB), so 87Gi is legal.

### What the scripts now do

Both `pjsub_k3_full_96n_short_1h.sh` and `pjsub_k3_full_96n.sh`:

- carry **no** bare `#PJM -x NAME` directives (only `-x PJM_LLIO_GFSCACHE=/vol0004`);
- use `--llio localtmp-size=87Gi`;
- take launch-time overrides as `-x NAME=value` on the pjsub command line,
  falling back to the `${VAR:-default}` values in the script body. Verified by
  canary c9 that this actually propagates — the old comment claiming pjsub
  "silently used the shell defaults" was describing the broken bare form.

The short script also now uses `elapse=03:00:00` (one hour never could have
worked: `49931198` spent 2885 s of 3600 s on rank-local staging and was killed
mid-validation; `49922938` spent 3103 s then hit `barrier timeout from rank 46`
on all 96 ranks), a `full-96n-short-3h-*` log root, and staging `CHUNK_MIB` 32
(`K3_STAGE_CHUNK_MIB` overrides) since 16 GB in 2885 s is ~5.6 MB/s per rank.

Submit line for the current job:

```bash
cd /vol0006/mdt0/data/hp250467/work/gemm/k3
pjsub --no-check-directory \
  -x K3_PROFILE=1 -x K3_THREADS=47 -x K3_BARRIER_ITERS=128 \
  -x K3_PREFILL_TOKENS=256 -x K3_NEW_TOKENS=256 -x K3_PREFILL_CHUNK=64 \
  -x K3_COMM_DETERMINISTIC=1 -x K3_COMM_BF16=0 -x K3_COMM_ROBUST=2 \
  -x K3_COMM_POLL_SPINS=4 -x K3_COMM_A2A=0 -x K3_COMM_A2A_MAX=8192 \
  -x K3_PREFETCH_MIB=0 -x K3_AR_GROUPS=16 -x K3_MOE_SHARD_LAYOUT=row-aligned \
  -x K3_MODEL_DIR=/home/u14346/models/kimi-k3 \
  a64fx/k3/pjsub_k3_full_96n_short_1h.sh
```

(Note: a non-interactive `ssh login1 '<cmd>'` starts in `$HOME`, so the `cd` is
required or pjsub cannot open the script.)

### Open on the full-precision path

- `49931198/run.rank.*` reports `tp_ar: ... deterministic=0` although the job
  was submitted with `K3_COMM_DETERMINISTIC=1`. Not yet traced through
  `k3_full_runner.c`. Do not draw checksum conclusions until it is.
- Best measured full-model figures remain `49931198`: prefill 1.709 tok/s,
  decode 1.699 tok/s at 96 nodes — far from the 10 tok/s target. Partial EP
  numbers are not evidence for that target.

---

## 2. Quantized path — current state

Full detail and tables:
`a64fx/k3/logs/quant-bench-live12-50000128/SUMMARY.md`.

**The two downloaded GGUF packages are misnamed.** Neither
`Kimi-K3-UD-IQ1_M-*` nor `Kimi-K3-UD-Q2_K_XL-*` contains a single IQ1_M or Q2_K
tensor. Across all shards both are entirely {F32, Q8_0, IQ1_S, IQ2_XXS/XS,
IQ3_XXS} — all already supported. Q8_0 dominates by count (1116 tensors: every
attention/dense projection plus `output.weight`).

Single-layer projection rate with real weights (layer 3, rank 0 of 12,
47 threads), default kernel path:

| package | at session start | now |
|---|---|---|
| IQ1 | 20.84 tok/s | **141.2** |
| Q2 | ~20.8 tok/s | **120.4** |

(Later figures were taken on a different node after a session restart; treat
absolutes as node-specific and compare only same-node before/after pairs.)

Roughly 5x, and ~34x versus the scalar reference the default entry point was
actually taking. These are single-layer projection proxies, **not** full-model
decode rates.

Measurement hygiene: the bench's first run is cold-start dominated (32.9 ms vs
13.2 ms warm). Always discard it and prefer best-of-N; this node also shows
occasional interference outliers.

What changed, in order of payoff:

1. `k3_quant_matvec()` no longer routes to the scalar reference; default kernel
   mode is now `sve-q8`.
2. Q8_0 uses the **int16**-activation kernel in both SVE modes. A 32-element
   Q8_0 block is exactly one 512-bit vector of int16 but only half a vector of
   int8, so `svdot_s64` on int16 beats `svdot_s32` on int8 here. Also 130x more
   accurate (rel_l2 3.8e-3 → 2.9e-5).
3. IQ kernels: whole 8-byte grid entries move as doublewords; sign application
   is one `svmul_s8` against a `[256][8]` ±1 table; and **two 32-element groups
   share one 64-lane `svdot`**, with each group's integer scale applied by
   `svsel` on lane index (IQ2_XS packs four half-group scales across 16 lanes).
   IQ2_XXS's `(0.5+k)*0.25` is `(1+2k)*0.125`, i.e. integer, which is what lets
   it join.
4. `cols % 256` workspace guard → `cols % 32`. Every Q8_0 tensor narrower than
   256 columns was failing `k3_quant_workspace_prepare`, so `matvec_mode`
   returned -1 and the tensor silently fell to the scalar path. `attn_k_b`
   (4096x128) was the case in this layer, and the bench had been reporting
   `ms=0.000` for it — a failed call, not a fast one.
5. INT8 KV attention QK is a real `svdot_s32` (query quantized once per call);
   5.444 → 3.979 ms on the whole attention call at tokens=16384, qk=576. PV
   stays fp32. Accuracy cost is real but small: `mla-int8` kernel-test error
   1.390e-03 → 1.642e-03.

6. **Vector-domain accumulation** (found with `perf annotate`, which put
   `fmov` off `svaddv` and its dependent load at 17% of cycles): IQ kernels
   accumulate scaled int32 lanes across a whole 256-element block, and Q8_0
   folds each block's scale in with `svmla` on doubles, so each does one
   horizontal reduction per block/row instead of per group.

**Dead ends worth not repeating** (all measured, all losses):

- `svtbl` does *not* apply to the IQ grid lookup. It permutes bytes within a
  vector, and the grids are 256–65536 entries, so the lookup must stay a memory
  access. The `k3_mxfp4_group_svtbl` precedent does not carry over — 16 entries.
- Merged grid+sign LUT for IQ2_XXS: -9%. But the *reverse* change for IQ2_XS
  (dropping its merged table for grid+sign) is also a loss, -7%. The two
  formats want opposite choices; table size does not predict which.
- A `threads <= 1` serial fast path in `k3_quant_matvec_ws`: -6%, two different
  ways. The inline dispatch chain lets the compiler hoist the mode/type tests
  out of the row loop and neither rewrite preserves that.
- 8-row Q8_0 blocking: -20% (kept behind `K3_Q8_ROWS8=1`).

8-row Q8_0 blocking is implemented but **default-off** (`K3_Q8_ROWS8=1`).
Measured: it loses ~20% (`output.weight` 1.58 → 1.91 ms).

Not done: the `_a16` variants of the IQ kernels still use the original scalar
staging. They are the accuracy fallback, not the default path.

---

## 3. Verified regression state

- `make -C a64fx/k3 test` exits 0 (kernel, quantized-kernel, runtime
  memory/pool, full-runner python).
- 12-node end-to-end: `run_k3_ep.sh --mode dummy --layers 1 --layer 3
  --cache-tokens 16384 --mla-cache-int8 --threads 47` →
  `K3_RESULT status=PASS`, `checksum=+3.650037202e+02`,
  `disagreement=0.000e+00`, `pass_markers=12/12`.
- Every IQ type's `rel_l2` is unchanged to the last digit across all the kernel
  rewrites, so they are semantics-preserving rather than accuracy trades.

### Earlier multi-node results (still valid)

- `49959732`: 96-node dummy + bounded real layer-1 smoke passed 96/96.
- `49961479` / `49961480` / `49961481`: 24 / 32 / 48-node matrices passed;
  48n included windows `[0,4)`, `[4,12)`, `[12,24)`, `[88,93)`, real layer 1
  and real layer 92.
- `49966258`: 64-node deterministic TP64 matrix passed after the 12-slot fix.
- `49964849`: real layer-92 profile ~`0.388 ms/layer` in the bounded runner.
- `TP_AR_NSTEP` is 13 (12 deterministic receive slots), enough for TP96
  hierarchical 6x16; flat deterministic N96 would need 14 and is not the
  production layout.

---

## 4. Resume prompt

```text
Resume K3 work in /vol0006/mdt0/data/hp250467/work/gemm/k3. Read @resume-k3.md
first.

pjsub/pjstat are not on PATH from a compute node — use `ssh login1 '<cmd>'`.

Batch: job 50005954 is the 96-node full-precision run (3 h, 87Gi, corrected -x
directives), with canaries 50004855/50004857 confirming the GATE CHECK cause. Do
not cancel or resubmit them unless explicitly directed. When one starts, inspect
a64fx/k3/logs/full-96n-short-3h-<id> and require topology, barrier preflight,
all-rank staging, full 93-layer generation, validation and decoded output, then
record real prefill/decode tok/s against the 10 tok/s target. Do not claim
10+ tok/s from partial EP tests.

Kernels: the quantized path was heavily optimized on 12 nodes (see section 2
and logs/quant-bench-live12-50000128/SUMMARY.md). Any further kernel work
should start from a fresh perf/PMU profile, not from assumption — several
"obvious" wins here measured as losses (8-row Q8_0 blocking, int8-over-int16
for Q8_0, svtbl for IQ grids).

Use fcc/FCC natively on A64FX compute nodes, fccpx/FCCpx when cross-compiling.
Preserve unrelated dirty workspace changes and do not push.
```
