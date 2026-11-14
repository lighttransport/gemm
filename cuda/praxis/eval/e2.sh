#!/bin/bash
# E2: knob sweep on block_out_of_bowl (A baseline 3/12), seeds 100-111, one knob at a time vs the E1 A_* baselines.  bash cuda/praxis/eval/e2.sh [stage]
source /home/syoyo/work/gemm/praxis/cuda/praxis/eval/common.sh
T=block_out_of_bowl
stage=${1:-1}
if [ $stage = 1 ]; then
  fast S4_$T $T mixed --num-steps 4          # (1) 4 Euler steps
  fast CFG15_$T $T mixed --cfg-scale 1.5     # (2) action CFG (extra unconditional action pass)
  fast CFG20_$T $T mixed --cfg-scale 2.0
  fast SG9_$T $T mixed --savgol 9            # (3) Savitzky-Golay (order 3)
  fast SG5_$T $T mixed --savgol 5
  fast SG13_$T $T mixed --savgol 13
fi
if [ $stage = 2 ]; then   # combos + validation on cube_on_plate + prompt variants (best knobs: 4 steps, action CFG 1.5/2.0)
  fast S4CFG15_$T $T mixed --num-steps 4 --cfg-scale 1.5
  fast S4_cube_on_plate cube_on_plate mixed --num-steps 4
  fast CFG15_cube_on_plate cube_on_plate mixed --cfg-scale 1.5
  fast CFG20_cube_on_plate cube_on_plate mixed --cfg-scale 2.0
  PROMPT="Take the red block out of the bowl." fast P1_$T $T mixed
  PROMPT="Pick up the red block and put it on the table." fast P2_$T $T mixed
fi
if [ $stage = 3 ]; then   # repeat of the unchanged baseline = run-to-run noise floor of the sim (same seeds, same settings)
  fast A2_$T $T mixed
fi
