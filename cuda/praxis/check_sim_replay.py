#!/usr/bin/env python3
"""M4: replay a recorded SIM observation (run_sim_cuda.py --dump-obs) through PraxisClient (session seed 0, so call 2 uses seed 1 like gen_real_ref.py)
and diff everything against the torch reference of the SAME observation (gen_real_ref.py --obs-npz ... --descriptor praxis.sim --calls 2).
  ~/.venv/bin/python -I cuda/praxis/check_sim_replay.py --obs /mnt/disk01/tmp/praxis_sim/m4_obs/obs_call1.npz --ref /mnt/disk01/tmp/praxis_sim/m4_ref [--oracle DIR]"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import praxis_client as pc
from praxis import single_arm as SA
ap = argparse.ArgumentParser()
ap.add_argument("--obs", required=True); ap.add_argument("--ref", required=True)
ap.add_argument("--oracle", default=None); ap.add_argument("--prompt", default="Put the red cube on the blue plate."); ap.add_argument("--expert-fmt", default="mixed")
a = ap.parse_args()
L = lambda p: np.load(p).astype(np.float32)
rel = lambda x, y: float(np.linalg.norm(np.asarray(x, np.float64) - y) / (np.linalg.norm(y) + 1e-30))
z = np.load(a.obs)
env = {"PRAXIS_EXPERT_FMT": a.expert_fmt, "PRAXIS_PACK_CACHE": "/mnt/disk1/praxis/praxis-1-droid.nvfp4.packcache"}
pipe = pc.PraxisClient("/mnt/disk1/praxis/praxis-1-droid", env=env, vae_mode="reload", log_path="/mnt/disk01/tmp/praxis_sim/server_m4.log")
sess = pipe.session(prompt=a.prompt, seed=0, max_context_chunks=1, control_mode="droid_impedance", exec_horizon=24)
pipe.warm_text([a.prompt])
obs = lambda n: {"ee_pose": z["ee_pose"][:n], "exterior_image": z["exterior"][:n], "wrist_image": z["wrist"][:n], "grip": z["grip"][:n], "task": a.prompt, "descriptor": "praxis.sim"}
r1 = sess.step(obs(1)); r2 = sess.step(obs(22))
d = f"{a.ref}/inputs_call2"
print("anchor rel-L2 %.3e  clean0 %.3e  proprio_all %.3e  boundary_poses equal %s  pose_now diff %.2e" % (
    rel(r2.inputs["anchor"].numpy(), L(d + "/anchor.npy")), rel(r2.inputs["cleans"][0].numpy(), L(d + "/clean0.npy")),
    rel(r2.inputs["proprio_all"].numpy(), L(d + "/proprio_all.npy")), np.allclose(r2.inputs["boundary_poses"], np.load(d + "/boundary_poses.npy"), atol=1e-6),
    np.abs(r2.inputs["pose_now"] - np.load(d + "/pose8_now.npy")).max()))
print("seed ours", r2.seed, "depth", r2.depth)
ra, rr = r2.action_rel[:, :3] * 1e3, L(a.ref + "/action_rel.npy")[:, :3] * 1e3
print("pred_action vs torch bf16: %.3e   action_rel xyz rel %.3e" % (rel(r2.pred_action_norm.numpy(), L(a.ref + "/pred_action.npy")), rel(ra, rr)))
for tag, x in (("ours", ra), ("torch-bf16 ref", rr)):
    st = np.linalg.norm(np.diff(x, axis=0), axis=1)
    print("  %-15s |row0| %.1f |row1| %.1f |row2| %.1f mm; step median %.1f mm; first steps %s" % (tag, *np.linalg.norm(x[:3], axis=1), np.median(st), np.round(st[:4], 1)))
if a.oracle:
    ro = L(a.oracle + "/action_rel.npy")[:, :3] * 1e3
    st = np.linalg.norm(np.diff(ro, axis=0), axis=1)
    print("  %-15s |row0| %.1f |row1| %.1f |row2| %.1f mm; step median %.1f mm" % ("torch-fp32", *np.linalg.norm(ro[:3], axis=1), np.median(st)))
    print("ours vs fp32 oracle: pred_action %.3e  xyz rows rel %.3e (rms %.1f mm);  torch-bf16 vs fp32: pred_action %.3e xyz rel %.3e (rms %.1f mm)" % (
        rel(r2.pred_action_norm.numpy(), L(a.oracle + "/pred_action.npy")), rel(ra, ro), np.sqrt(((ra - ro) ** 2).sum(1).mean()),
        rel(L(a.ref + "/pred_action.npy"), L(a.oracle + "/pred_action.npy")), rel(rr, ro), np.sqrt(((rr - ro) ** 2).sum(1).mean())))
pipe.close()
