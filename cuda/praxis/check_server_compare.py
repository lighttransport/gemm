#!/usr/bin/env python3
"""Run the SAME recorded sim observation (run_sim_cuda.py --dump-obs) through two servers (call 1 = 1 frame, call 2 = 22 frames, session seed 0, identical noise)
and diff their outputs: the local CUDA server and a remote one given by PRAXIS_SERVER_CMD (e.g. the HIP server on the AMD host).  Sequential, one server at a time.
  PRAXIS_SERVER_CMD='ssh <amd-host> "..."' ~/.venv/bin/python -I cuda/praxis/check_server_compare.py --obs .../m4_obs/obs_call1.npz [--oracle .../m4_ref_fp32]"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import praxis_client as pc
PRAXIS_ROOT = os.environ.get("PRAXIS_ROOT", os.path.expanduser("~/praxis"))
PRAXIS_OUT = os.environ.get("PRAXIS_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tmp"))
ap = argparse.ArgumentParser()
ap.add_argument("--obs", required=True); ap.add_argument("--oracle", default=None)
ap.add_argument("--prompt", default="Put the red cube on the blue plate."); ap.add_argument("--expert-fmt", default="mixed")
ap.add_argument("--ckpt", default=PRAXIS_ROOT + "/praxis-1-droid")
a = ap.parse_args()
rel = lambda x, y: float(np.linalg.norm(np.asarray(x, np.float64) - np.asarray(y, np.float64)) / (np.linalg.norm(np.asarray(y, np.float64)) + 1e-30))
z = np.load(a.obs)
obs = lambda n: {"ee_pose": z["ee_pose"][:n], "exterior_image": z["exterior"][:n], "wrist_image": z["wrist"][:n], "grip": z["grip"][:n], "task": a.prompt, "descriptor": "praxis.sim"}
remote = os.environ.pop("PRAXIS_SERVER_CMD", None)
if not remote:
    sys.exit("set PRAXIS_SERVER_CMD")
res = {}; dec = {}
for tag, cmd in (("cuda", None), ("remote", remote)):
    if cmd: os.environ["PRAXIS_SERVER_CMD"] = cmd
    else: os.environ.pop("PRAXIS_SERVER_CMD", None)
    env = {"PRAXIS_EXPERT_FMT": a.expert_fmt, "PRAXIS_PACK_CACHE": PRAXIS_ROOT + "/praxis-1-droid.nvfp4.packcache"}
    pipe = pc.PraxisClient(a.ckpt, env=env, vae_mode="reload", log_path=f"{PRAXIS_OUT}/praxis_sim_hip/server_cmp_{tag}.log")
    sess = pipe.session(prompt=a.prompt, seed=0, max_context_chunks=1, control_mode="droid_impedance", exec_horizon=24)
    pipe.warm_text([a.prompt])
    r1 = sess.step(obs(1)); r2 = sess.step(obs(22))
    t0 = __import__("time").monotonic(); fr = sess.decode_video(r2).numpy(); dms = (__import__("time").monotonic() - t0) * 1e3
    dec[tag] = (fr, dms, dict(pipe.server.last))
    res[tag] = (r1, r2, [dict(t) for t in pipe.timings])
    pipe.close()
L = lambda p: np.load(p).astype(np.float32)
for ci, nm in ((0, "call1"), (1, "call2")):
    c, h = res["cuda"][ci], res["remote"][ci]
    xyz_c = np.stack([np.asarray(x["robot_command"]["pose"])[:3] for x in c.commands]) * 1e3
    xyz_h = np.stack([np.asarray(x["robot_command"]["pose"])[:3] for x in h.commands]) * 1e3
    print(f"{nm}: anchor/clean inputs rel-L2 hip-vs-cuda {rel(h.inputs['anchor'].numpy(), c.inputs['anchor'].numpy()):.3e}; pred_action {rel(h.pred_action_norm.numpy(), c.pred_action_norm.numpy()):.3e}; "
          f"pred_video {rel(h.pred_video_latent.numpy(), c.pred_video_latent.numpy()):.3e}; action_rel {rel(h.action_rel, c.action_rel):.3e}; xyz max|diff| {np.abs(xyz_h-xyz_c).max():.1f} mm rms {np.sqrt(((xyz_h-xyz_c)**2).sum(1).mean()):.1f} mm")
    if a.oracle and ci == 1:
        o = L(a.oracle + "/pred_action.npy")
        print(f"  vs fp32 oracle (call2): cuda {rel(c.pred_action_norm.numpy(), o):.3e}  hip {rel(h.pred_action_norm.numpy(), o):.3e}")
for tag in ("cuda", "remote"):
    t = res[tag][2][-1]
    print(f"{tag}: call2 server_sample_ms {t['server_sample_ms']:.0f} total_s {t['total_s']:.2f} encode_ms {t['encode_ms']:.0f} peak sample/encode {t['peak_gb_sample']:.2f}/{t['peak_gb_encode']:.2f}")
d_c, d_h = dec["cuda"][0].astype(np.float32), dec["remote"][0].astype(np.float32)
print(f"DECODE of each server's own predicted video: frames {d_h.shape}, mean|diff| {np.abs(d_h-d_c).mean():.2f}/255, max {np.abs(d_h-d_c).max():.0f}; cuda {dec['cuda'][1]:.0f} ms (server {dec['cuda'][2]['ms']:.0f}, peak {dec['cuda'][2]['peak_gb']:.2f} GB), remote {dec['remote'][1]:.0f} ms (server {dec['remote'][2]['ms']:.0f}, rtt {dec['remote'][2]['rtt_ms']:.0f}, peak {dec['remote'][2]['peak_gb']:.2f} GB)")
# decode of the SAME latent (cuda's predicted video) on both servers is the VAE-only comparison: done in the vae parity tests (18/18 on each backend)
