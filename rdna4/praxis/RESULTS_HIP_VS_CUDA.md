# HIP (RX 9070 XT) vs CUDA (RTX 5060 Ti) closed-loop validation

Question: does the ROCm/HIP port of the praxis-1-droid runner give the same closed-loop task success as the CUDA runner?
Both run the real `praxis-1-droid` checkpoint with the **same settings** ("fast": mixed experts, no video guidance,
skip-last-video, `--rebase 8`, 2 Euler steps) in praxis's own MuJoCo DROID sim, on the **same seeds** (paired per seed).
The HIP policy runs on an AMD host through `ssh` (`PRAXIS_SERVER_CMD`, see `README.md` 'Paths and environment'); rendering and
the sim run on the NVIDIA machine for both backends.

Statistics: per-task exact McNemar on the paired outcomes; the pooled difference (HIP minus CUDA, in percentage points) has a
Newcombe 95% interval (and a seed-bootstrap interval). Tool: `cuda/praxis/eval/pool_pair.py`.

## Phase 1: seeds 100-111 (n = 12 per task)

| task | CUDA | HIP | only-CUDA / only-HIP | McNemar p | HIP-CUDA (pts), 95% CI |
|---|---|---|---|---|---|
| lift_cube | 11 | 10 | 2 / 1 | 1.00 | -8.3 [-37.3, +21.3] |
| cube_on_plate | 10 | 10 | 1 / 1 | 1.00 | 0.0 [-25.8, +25.8] |
| move_block | 11 | 11 | 0 / 0 | 1.00 | 0.0 [-20.2, +20.2] |
| block_in_bowl | 10 | 10 | 1 / 1 | 1.00 | 0.0 [-25.8, +25.8] |
| stack_block | 10 | 8 | 3 / 1 | 0.63 | -16.7 [-45.0, +14.9] |
| press_button | 9 | 8 | 2 / 1 | 1.00 | -8.3 [-34.0, +18.6] |
| **pooled (72 pairs)** | **61 (85%)** | **57 (79%)** | **9 / 5** | **0.42** | **-5.6 [-16.0, +4.8]** (bootstrap [-15.3, +4.2]) |

Per-seed outcome agreement 81%; mean ticks to success 188 (CUDA) vs 190 (HIP).

## Phase 2: fresh seeds 112-135 (n = 24 per task)

| task | CUDA | HIP | only-CUDA / only-HIP | McNemar p | HIP-CUDA (pts), 95% CI |
|---|---|---|---|---|---|
| lift_cube | 23 | 23 | 0 / 0 | 1.00 | 0.0 [-12.6, +12.6] |
| cube_on_plate | 23 | 21 | 3 / 1 | 0.63 | -8.3 [-27.2, +9.7] |
| move_block | 19 | 20 | 2 / 3 | 1.00 | +4.2 [-14.8, +23.1] |
| **pooled (72 pairs)** | **65 (90%)** | **64 (89%)** | **5 / 4** | **1.00** | **-1.4 [-10.4, +7.4]** (bootstrap [-9.7, +6.9]) |

Agreement 88%; mean ticks to success 177 (CUDA) vs 185 (HIP).

## Both phases pooled (144 pairs, fast settings)

CUDA 126/144 (88%), HIP 121/144 (84%); discordant 14 only-CUDA / 9 only-HIP; exact McNemar p = 0.41;
difference -3.5 pts, Newcombe 95% [-10.2, +3.2], bootstrap [-9.7, +3.5]; agreement 84%; ticks 183 vs 188.
Seeds needed to exclude a +-10 point difference at this discordance rate: about 62 pairs per comparison.

## Full variant (video guidance 4, decoded predicted video), seeds 100-111

| task | CUDA | HIP | only-CUDA / only-HIP |
|---|---|---|---|
| cube_on_plate | 11 | 10 | 2 / 1 |
| move_block | 11 | 12 | 0 / 1 |
| **pooled (24 pairs)** | **22 (92%)** | **22 (92%)** | **2 / 2** (p = 1.00; difference 0.0 [-18.5, +18.5]) |

## Reading

- No per-task difference is statistically significant, and the pooled differences have 95% intervals that include 0
  (-5.6 pts in phase 1, -1.4 pts in phase 2, -3.5 pts over both).
- The data do **not** prove equivalence: a true HIP deficit of up to about 10 points (pooled upper-lower bound -10.2) is not excluded.
  The point estimate leans slightly towards CUDA (14 vs 9 discordant seeds), which is consistent with either chance or a small
  cost of the 4-bit expert weights on AMD (no FP4 hardware: W4A16).
- A direct numerical replay on recorded sim observations found no stage with a bug signature: both servers are deterministic
  run to run, and the HIP-vs-CUDA action differences are inside each backend's own error against a torch fp32 oracle
  (see `README.md`).
- Per-call latency (fast): about 3.5-3.8 s on HIP including the ssh pipe, 4.0-4.5 s on CUDA.

## Caveats

- Simulated tasks only; N = 12-24 per task. Earlier-checkpoint numbers in praxis's own sim documentation are not comparable.
- The AMD host's GPU also drives its desktop; runs were gated on desktop VRAM, CPU temperature and kernel-log health (see
  `README.md`). Rendering for both backends ran on the same NVIDIA GPU.
- The CUDA side of phase 2 ran while a CPU/GPU miner was active on the same machine (see `cuda/praxis/README.md`,
  'Co-existence with a miner'); this changes latency, not success.

## Reproduce

```
# CUDA side (local GPU): fast variant, seeds 100-111 per task, and 112-135 (cuda/praxis/eval)
bash cuda/praxis/eval/e1.sh            # A_<task> runs (seeds 100-111)
bash cuda/praxis/eval/e3_cuda_extra.sh # X_<task> runs (seeds 112-135)
# HIP side: same flags with the policy on the AMD host
PRAXIS_SERVER_CMD='ssh <amd-host> "cd /path/to/repo && OMP_NUM_THREADS=8 rdna4/praxis/hip_praxis_server_full ..."' \
  python cuda/praxis/run_sim_cuda.py --config <sim_droid.yaml> --vae-mode reload --expert-fmt mixed \
  --cfg-video 1.0 --skip-last-video --set task.name=<task> --set "episodes.seeds=[100,...,111]" --set output.dir=<out>
# paired analysis
python cuda/praxis/eval/pool_pair.py lift_cube=<cuda_dir>:<hip_dir>[,<hip_dir2>] cube_on_plate=...
```
