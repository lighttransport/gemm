#!/usr/bin/env python3
"""Noise-floor / backend comparison on recorded sim observations: every obs file is replayed (fresh session, seed 0: call 1 = 1 row, call 2 = 22 rows, identical
noise) through backends listed as tag=cmd ('local' = the CUDA server; anything else = a PRAXIS_SERVER_CMD), possibly several times each.  Prints the pairwise
rel-L2 of the sampled action tokens, of the un-normalised xyz and of the grip channel, per obs, so the run-to-run difference of ONE backend can be compared to the
cross-backend difference.
  check_repro.py --obs o1.npz o2.npz --run cuda_a=local --run cuda_b=local --run hip_a="ssh ..." --run hip_b="ssh ..." [--oracle DIR for the first obs]"""
import argparse, itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import praxis_client as pc
PRAXIS_ROOT = os.environ.get("PRAXIS_ROOT", os.path.expanduser("~/praxis"))
PRAXIS_OUT = os.environ.get("PRAXIS_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tmp"))
ap = argparse.ArgumentParser()
ap.add_argument("--obs", nargs="+", required=True); ap.add_argument("--run", action="append", default=[])
ap.add_argument("--prompt", default="Pick up the red cube."); ap.add_argument("--oracle", default=None)
ap.add_argument("--ckpt", default=PRAXIS_ROOT + "/praxis-1-droid"); ap.add_argument("--outdir", required=True, help="one <tag>.pkl per finished run (resumable: finished tags are skipped); the report covers every pkl in the dir")
a = ap.parse_args()
rel = lambda x, y: float(np.linalg.norm(np.asarray(x, np.float64) - np.asarray(y, np.float64)) / (np.linalg.norm(np.asarray(y, np.float64)) + 1e-30))
import pickle
os.makedirs(a.outdir, exist_ok=True)
for spec in a.run:
    tag, cmd = spec.split("=", 1)
    if os.path.exists(f"{a.outdir}/{tag}.pkl"): print("skip", tag); continue
    if cmd == "local": os.environ.pop("PRAXIS_SERVER_CMD", None)
    else: os.environ["PRAXIS_SERVER_CMD"] = cmd
    env = {"PRAXIS_EXPERT_FMT": "mixed", "PRAXIS_PACK_CACHE": PRAXIS_ROOT + "/praxis-1-droid.nvfp4.packcache"}
    pipe = pc.PraxisClient(a.ckpt, env=env, vae_mode="reload", log_path=f"{PRAXIS_OUT}/praxis_sim_hip/server_repro_{tag}.log")
    out = []
    for of in a.obs:
        z = np.load(of)
        sess = pipe.session(prompt=a.prompt, seed=0, max_context_chunks=1, control_mode="droid_impedance", exec_horizon=24)
        pipe.warm_text([a.prompt])
        ob = lambda n: {"ee_pose": z["ee_pose"][:n], "exterior_image": z["exterior"][:n], "wrist_image": z["wrist"][:n], "grip": z["grip"][:n], "task": a.prompt, "descriptor": "praxis.sim"}
        r1 = sess.step(ob(1)); r2 = sess.step(ob(22))
        out.append([(r.pred_action_norm.numpy().copy(), r.action_rel.copy(), r.inputs["anchor"].numpy().copy()) for r in (r1, r2)])
    pipe.close()
    pickle.dump(out, open(f"{a.outdir}/{tag}.pkl", "wb"))
import glob
res = {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob(a.outdir + "/*.pkl"))}
tags = list(res)
for oi, of in enumerate(a.obs):
    for ci in (0, 1):
        print(f"== {os.path.basename(of)} call{ci+1}")
        for t1, t2 in itertools.combinations(tags, 2):
            A, B = res[t1][oi][ci], res[t2][oi][ci]
            print(f"  {t1:8s} vs {t2:8s}: pred_action {rel(A[0], B[0]):.3e}  xyz {rel(A[1][:, :3], B[1][:, :3]):.3e} (rms {np.sqrt(((A[1][:, :3]-B[1][:, :3])**2).sum(1).mean())*1e3:.1f} mm)  grip {np.abs(A[1][:, 9]-B[1][:, 9]).max():.3f} max|d|  anchor {rel(A[2], B[2]):.2e}")
