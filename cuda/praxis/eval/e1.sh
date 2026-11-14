#!/bin/bash
# E1: paired evaluation, seeds 100-111. Sequential, one server at a time.  bash cuda/praxis/eval/e1.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for t in lift_cube cube_on_plate move_block block_in_bowl; do fast A_$t $t mixed; done          # A: mixed, cfg video 1, skip-last-video, rebase 8
for t in cube_on_plate move_block; do fast D_$t $t mixed --rebase 0; done                       # D: praxis' own chunk behaviour
for t in lift_cube cube_on_plate move_block block_in_bowl; do fast B_$t $t nvfp4; done          # B: NVFP4 W4A4
for t in cube_on_plate move_block; do full C_$t $t mixed; done                                  # C: mixed, cfg video 4, decoded video (rebase 8)
for t in block_out_of_bowl stack_block press_button; do fast A_$t $t mixed; done                # A on the extra tasks
