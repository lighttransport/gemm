# Evaluation results (N=12, seeds 100-111) - generated

Generated from `/mnt/disk01/tmp/praxis_sim2/` with `summarize.py` (see cuda/praxis/README.md 'Evaluation (N=12)'). Success = the sim's check, Wilson 95% CI. Latency = per Session.step call; part of the E2 runs shared the GPU with another job's EGL rendering (server_sample_ms is unaffected: see README). 'peak GB' = cuMemGetInfo-based, includes other processes; the server's own peak is 7.24 GB (A, mixed) / 6.88 GB (NVFP4).

## E1: variants (A mixed fast rebase 8, B NVFP4, C full cfg video 4 + decoded video, D rebase 0)

| run | success | Wilson 95% CI | mean ticks (succ) | call mean/p50 s | boundary jump mm | step CV | peak GB sample/encode | failed seeds |
|---|---|---|---|---|---|---|---|---|
| A_lift_cube | 11/12 | 65-99% | 152 | 4.17/3.96 | 5.5 | 0.46 | 7.72/7.72 | [110] |
| A_cube_on_plate | 10/12 | 55-95% | 205 | 4.00/3.94 | 6.1 | 0.45 | 7.73/7.73 | [102, 109] |
| A_move_block | 11/12 | 65-99% | 175 | 4.00/3.94 | 5.7 | 0.46 | 7.73/7.73 | [109] |
| A_block_in_bowl | 10/12 | 55-95% | 196 | 4.02/3.97 | 6.0 | 0.45 | 7.73/7.73 | [107, 109] |
| A_block_out_of_bowl | 3/12 | 9-53% | 196 | 4.15/3.98 | 5.5 | 0.46 | 10.71/9.04 | [100, 101, 102, 103, 105, 107, 108, 109, 110] |
| A2_block_out_of_bowl | 3/12 | 9-53% | 185 | 4.03/4.00 | 5.6 | 0.46 | 7.72/7.72 | [100, 101, 102, 103, 104, 107, 108, 109, 110] |
| A_stack_block | 10/12 | 55-95% | 250 | 4.03/3.99 | 5.5 | 0.47 | 7.74/7.74 | [109, 111] |
| A_press_button | 9/12 | 47-91% | 152 | 4.01/3.96 | 4.9 | 0.48 | 7.73/7.73 | [102, 107, 110] |
| B_lift_cube | 11/12 | 65-99% | 174 | 3.31/3.16 | 5.4 | 0.46 | 7.36/7.36 | [110] |
| B_cube_on_plate | 9/12 | 47-91% | 191 | 3.21/3.19 | 5.8 | 0.44 | 7.37/7.37 | [103, 105, 106] |
| B_move_block | 10/12 | 55-95% | 199 | 3.19/3.16 | 5.5 | 0.46 | 7.37/7.37 | [102, 107] |
| B_block_in_bowl | 11/12 | 65-99% | 210 | 3.19/3.17 | 5.9 | 0.45 | 7.37/7.37 | [102] |
| C_cube_on_plate | 11/12 | 65-99% | 164 | 8.23/8.31 | 5.7 | 0.47 | 7.76/6.33 | [104] |
| C_move_block | 11/12 | 65-99% | 173 | 8.23/8.31 | 5.9 | 0.46 | 7.77/6.34 | [104] |
| D_cube_on_plate | 8/12 | 39-86% | 186 | 4.01/3.97 | 25.1 | 0.45 | 7.73/7.73 | [102, 103, 104, 109] |
| D_move_block | 11/12 | 65-99% | 179 | 4.01/3.97 | 28.7 | 0.46 | 7.73/7.73 | [102] |

## E2: knob sweep (block_out_of_bowl, cube_on_plate); S4 = 4 Euler steps, CFG = action CFG scale, SG = Savitzky-Golay window (order 3), P1/P2 = alternative prompts

| run | success | Wilson 95% CI | mean ticks (succ) | call mean/p50 s | boundary jump mm | step CV | peak GB sample/encode | failed seeds |
|---|---|---|---|---|---|---|---|---|
| S4_block_out_of_bowl | 5/12 | 19-68% | 204 | 7.58/7.61 | 5.0 | 0.46 | 7.74/7.74 | [102, 103, 107, 108, 109, 110, 111] |
| CFG15_block_out_of_bowl | 6/12 | 25-75% | 195 | 6.88/6.88 | 5.9 | 0.46 | 7.74/7.74 | [101, 104, 106, 107, 108, 110] |
| CFG20_block_out_of_bowl | 6/12 | 25-75% | 208 | 6.86/6.84 | 5.9 | 0.45 | 7.88/7.88 | [101, 103, 105, 107, 109, 110] |
| SG5_block_out_of_bowl | 4/12 | 14-61% | 231 | 4.05/4.01 | 5.1 | 0.45 | 7.74/7.74 | [100, 101, 102, 103, 104, 105, 108, 110] |
| SG9_block_out_of_bowl | 2/12 | 5-45% | 240 | 4.04/4.00 | 4.1 | 0.46 | 7.75/7.75 | [100, 102, 103, 104, 105, 106, 107, 108, 109, 110] |
| SG13_block_out_of_bowl | 7/12 | 32-81% | 264 | 4.05/4.01 | 4.0 | 0.45 | 7.74/7.74 | [100, 101, 102, 108, 110] |
| S4CFG15_block_out_of_bowl | 4/12 | 14-61% | 180 | 14.05/14.19 | 5.0 | 0.45 | 7.74/7.74 | [102, 103, 106, 107, 108, 109, 110, 111] |
| P1_block_out_of_bowl | 5/12 | 19-68% | 233 | 4.04/4.00 | 5.6 | 0.46 | 7.74/7.74 | [100, 101, 102, 103, 105, 108, 110] |
| P2_block_out_of_bowl | 5/12 | 19-68% | 241 | 4.01/3.97 | 5.4 | 0.46 | 7.74/7.74 | [100, 103, 105, 106, 107, 109, 110] |
| S4_cube_on_plate | 9/12 | 47-91% | 176 | 7.57/7.60 | 5.5 | 0.46 | 7.73/7.73 | [103, 106, 109] |
| CFG15_cube_on_plate | 10/12 | 55-95% | 199 | 6.87/6.86 | 6.4 | 0.45 | 7.73/7.73 | [105, 109] |
| CFG20_cube_on_plate | 10/12 | 55-95% | 180 | 6.87/6.86 | 6.5 | 0.45 | 7.73/7.73 | [109, 110] |

## Paired exact McNemar (first = baseline, same seeds)

```
A_block_out_of_bowl vs A2_block_out_of_bowl: paired n=12: only-first-succeeds 1, only-second-succeeds 1, exact McNemar p=1.000
A_block_out_of_bowl vs S4_block_out_of_bowl: paired n=12: only-first-succeeds 1, only-second-succeeds 3, exact McNemar p=0.625
A_block_out_of_bowl vs CFG15_block_out_of_bowl: paired n=12: only-first-succeeds 2, only-second-succeeds 5, exact McNemar p=0.453
A_block_out_of_bowl vs CFG20_block_out_of_bowl: paired n=12: only-first-succeeds 0, only-second-succeeds 3, exact McNemar p=0.250
A_block_out_of_bowl vs SG5_block_out_of_bowl: paired n=12: only-first-succeeds 1, only-second-succeeds 2, exact McNemar p=1.000
A_block_out_of_bowl vs SG9_block_out_of_bowl: paired n=12: only-first-succeeds 2, only-second-succeeds 1, exact McNemar p=1.000
A_block_out_of_bowl vs SG13_block_out_of_bowl: paired n=12: only-first-succeeds 0, only-second-succeeds 4, exact McNemar p=0.125
A_block_out_of_bowl vs S4CFG15_block_out_of_bowl: paired n=12: only-first-succeeds 2, only-second-succeeds 3, exact McNemar p=1.000
A_block_out_of_bowl vs P1_block_out_of_bowl: paired n=12: only-first-succeeds 0, only-second-succeeds 2, exact McNemar p=0.500
A_block_out_of_bowl vs P2_block_out_of_bowl: paired n=12: only-first-succeeds 1, only-second-succeeds 3, exact McNemar p=0.625
A_cube_on_plate vs S4_cube_on_plate: paired n=12: only-first-succeeds 2, only-second-succeeds 1, exact McNemar p=1.000
A_cube_on_plate vs CFG15_cube_on_plate: paired n=12: only-first-succeeds 1, only-second-succeeds 1, exact McNemar p=1.000
A_cube_on_plate vs CFG20_cube_on_plate: paired n=12: only-first-succeeds 1, only-second-succeeds 1, exact McNemar p=1.000
A_cube_on_plate vs B_cube_on_plate: paired n=12: only-first-succeeds 3, only-second-succeeds 2, exact McNemar p=1.000
A_cube_on_plate vs C_cube_on_plate: paired n=12: only-first-succeeds 1, only-second-succeeds 2, exact McNemar p=1.000
A_cube_on_plate vs D_cube_on_plate: paired n=12: only-first-succeeds 2, only-second-succeeds 0, exact McNemar p=0.500
A_move_block vs B_move_block: paired n=12: only-first-succeeds 2, only-second-succeeds 1, exact McNemar p=1.000
A_move_block vs C_move_block: paired n=12: only-first-succeeds 1, only-second-succeeds 1, exact McNemar p=1.000
A_move_block vs D_move_block: paired n=12: only-first-succeeds 1, only-second-succeeds 1, exact McNemar p=1.000
A_lift_cube vs B_lift_cube: paired n=12: only-first-succeeds 0, only-second-succeeds 0, exact McNemar p=1.000
A_block_in_bowl vs B_block_in_bowl: paired n=12: only-first-succeeds 1, only-second-succeeds 2, exact McNemar p=1.000
```
