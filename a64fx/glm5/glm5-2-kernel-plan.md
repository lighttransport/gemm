# GLM-5.2 decode → ceiling: node-hour-economized probe/test/optimize plan

Goal: reach the practical decode ceiling (~35–37 tok/s aggregate / ~4.5 tok/s single-stream, per
`glm5-2.md`) by attacking the two levers the profile identified — **attention compute (the dominant,
non-scaling term)** and **compute/comm overlap** — while spending as few node-hours as possible.

Principle: **iterate LOCALLY in qlair (cycle-accurate, 0 node-hours); spend node-hours only to
(a) calibrate qlair once, and (b) validate a finished optimization batch.** Small 1-node native
profile jobs are the calibration anchor.

Budget so far: ~145 / 1000 nh. This plan targets the ceiling in **< 250 additional nh**.

---

## A. Local qlair kernel-optimization loop (0 node-hours per iteration)

The hot kernel (from the real profile, `GLM5_P_ATTN`, `glm5_impl.h:1437`): the **absorb** flash-attention
inner loop — for each of `nsel` KV positions × `nown` heads: a **512-dim f32 dot** (`glm5_dot_f32_opt`)
+ a **512-dim online-softmax scale/AXPY** over `ctx`. It does NOT shard across nodes (why attn grows
26→48 ms/tok from 32n→96n), so the ONLY lever is making the kernel faster — ideal for local work.

Toolchain (same as the tp_ar / glm5_sim harnesses):
```
aarch64-linux-gnu-gcc -O3 -march=armv8.2-a+sve -ffast-math -o k.elf kernel.c
~/work/clair/clair/build/qlair --profile-markers --profile-report k.json k.elf   # cycle-accurate
```
Wrap the kernel in `__qlair_sim_start()/__qlair_sim_end()` markers; cycles→ms at 2.0 GHz (2.2 boost).

### Optimizations to try (each measured in qlair, ranked by expected win)
1. **Multi-accumulator dot/AXPY** (2–4 independent `svmla` accumulators) — breaks the FMA
   dependency chain (A64FX FMA latency ~9 cyc, 2 FLA pipes → single-acc is ~4× off peak). Biggest win.
2. **Head-batched GEMV** — replace `nown` separate 512-dots with one `[nown×kv_lora]·[kv_lora]` GEMV
   (better reuse of the shared `kv[]` load; amortize loads across heads).
3. **KV latent load / dequant** (`glm5_load_latent_kv`, bf16→f32) — fuse the load into the dot;
   prefetch next position's KV.
4. **Fuse scale+AXPY** in the online-softmax update; hoist the `expf` (SVE `expf` approx already exists).
5. **int8/bf16 qa·kv** where accuracy allows (the dot is currently f32).

### Local validation each iteration
- `ok=1` exact-recompute self-check + `gcc -O0` numeric cross-check (per `qlair/kernels/README.md`).
- Record cycles/token for attention; project the end-to-end tok/s via `decode_sim.py` (attn ms/tok knob).

---

## B. qlair calibration (ONE small native job, ~1–3 nh)

qlair's A64FX cycle model needs one real anchor for the decode kernels (the `qlair/kernels` are scalar
`-O0` upper bounds; we need the real SVE path). 1-node native job:
- `pjsub_glm5_kernel_probe_1n.sh` (NEW): run the SAME attention/dot/AXPY microbench binaries **natively**
  on 1 A64FX node (fcc `-O3 -march=armv8.2-a+sve`), report cycles (CNTVCT) + ns/call for the real dims.
- Also re-run `bdecode_kern_bench` (already in the ar_probe job) for the GEMV shapes.
- Compare native ns vs qlair cycle→ns; set qlair `--friction` so the error is ≤10% (QLAIR_VERIFICATION
  methodology). After this, **trust qlair for all kernel iteration**.
- Cost: 1 node × ~10 min = **~0.2 nh** (round up ~1–3 nh with retries). Re-run only if the kernel shape
  changes materially.

---

## C. Overlap lever (mostly local reasoning + one A/B job)

`GLM5_COMM_OVERLAP` (comm thread runs the AR concurrent with compute; infra exists,
`glm5_ep_runner.c:421`) was OFF in the J0 profile. Enabling it in the cbatch decode path is a code
change validated by:
- Local: confirm the decode path spawns/uses the comm thread (read + a qlair `--ranks` functional run).
- Job: one 32n int8 cbatch A/B, `GLM5_COMM_OVERLAP=0` vs `1`, short ctx (reuse `pjsub_glm5_cbatch_int8_32n.sh`
  + the env). ~20 nh. Expect the comm/compute serial→overlap gain (up to ~1.5–2× once compute is halved).

---

## D. Economized job ladder (validation only — iteration is local)

| # | job | nodes | ~nh | validates |
|---|---|---|---|---|
| K0 | qlair calib (native kernel probe, 1n) | 1 | ~2 | qlair cycle accuracy ≤10% → local loop trusted |
| K1 | optimized-attn A/B (int8 cbatch, 32n) | 32 | ~20 | the local attn speedup holds end-to-end (attn ms/tok ↓) |
| K2 | overlap A/B (`GLM5_COMM_OVERLAP` 0 vs 1, 32n) | 32 | ~20 | the overlap lever |
| K3 | batched-M-cap fix + bit-identity A/B (32n) | 32 | ~20 | raise M>4; slot bit-identity fixed |
| K4 | combined best (attn+overlap+batched+MTP) int8 (32n) | 32 | ~25 | the ceiling, int8 short-ctx @32n |
| K5 | bf16 decode profile + best config (≥96n) | 96 | ~50 | bf16 precision track (doesn't fit <96n) |
| K6 | node-count confirm (48n done; optional 24n) | 24 | ~15 | efficiency floor |

Total ≈ **150 nh** (+contingency → <250). Everything else (kernel iteration, projections) is local.

Gates: K0 before trusting local numbers; K1/K2/K3 each land one lever; K4 is the headline; K5 is the
second precision track. Run ≤2 concurrent; every tok/s paired with a bit-identity check.

---

## E. Sequence

1. **Local**: build attention microbench + baseline qlair profile → multi-accumulator + GEMV opts,
   measure speedup locally. (now)
2. **K0**: 1-node native calibration → set qlair friction, confirm ≤10%.
3. Land the attn kernel change behind an env flag; **K1** 32n A/B.
4. Enable overlap; **K2** 32n A/B. Fix batched M-cap + bit-identity; **K3**.
5. **K4** combined @32n = the ceiling number. **K5** bf16 track.
6. Refresh `decode_sim.py` / `glm5-2.md` after each landed lever.
