#!/usr/bin/env python3
"""bf16 noise floor: distance of the bf16 oracle and of our CUDA run to the fp32 oracle (gen_ref.py --fp32 --out tmp/cosmos3_ref_fp32)."""
import numpy as np, sys
import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--fp32", default="tmp/cosmos3_ref_fp32"); ap.add_argument("--bf16", default="tmp/cosmos3_ref"); ap.add_argument("--free", default="tmp/cosmos3_out"); ap.add_argument("--tf", default="tmp/cosmos3_out_tf"); A = ap.parse_args()
R, B, O, T = A.fp32, A.bf16, A.free, A.tf
def rel(a, b):
    a = np.load(a).astype(np.float64).ravel(); b = np.load(b).astype(np.float64).ravel(); return np.linalg.norm(a-b)/np.linalg.norm(a)
names = ["s0_cond_layer00_gen","s0_cond_layer09_gen","s0_cond_layer18_gen","s0_cond_layer27_gen","s0_cond_layer00_und","s0_cond_layer13_und","s0_cond_layer27_und","s0_cond_pred_vision","s0_cond_pred_action","s0_uncond_pred_vision","step0_vision_velocity_cfg","step0_action_velocity_cfg","step3_vision_velocity_cfg","step3_action_velocity_cfg","final_action","final_video_latent"]
print(f"{'tensor':32s} bf16oracle-vs-fp32   ours(tf)-vs-fp32   ours(free)-vs-fp32")
for n in names:
    f = f"{R}/{n}.npy"
    t = f"{T}/{n}.npy"; o = f"{O}/{n}.npy"
    import os
    print(f"{n:32s} {rel(f, f'{B}/{n}.npy'):.3e}      {rel(f, t) if os.path.exists(t) else float('nan'):.3e}       {rel(f, o) if os.path.exists(o) else float('nan'):.3e}")
