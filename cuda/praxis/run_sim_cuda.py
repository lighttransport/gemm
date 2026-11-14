#!/usr/bin/env python3
"""Closed-loop praxis MuJoCo sim (scripts/run_sim.py of praxis-1-oss) with OUR CUDA policy server instead of the PyTorch pipeline.

  MUJOCO_GL=egl PRAXIS_EXPERT_FMT=mixed PRAXIS_PACK_CACHE=$PRAXIS_ROOT/praxis-1-droid.nvfp4.packcache \
    ~/.venv/bin/python -I cuda/praxis/run_sim_cuda.py --config $PRAXIS_ROOT/praxis-1-oss/configs/sim_droid.yaml \
      --set task.name=lift_cube --set task.prompt=null --set "episodes.seeds=[0,1,2]" --set output.dir=$PRAXIS_OUT/praxis_sim/lift_cube_mixed

Everything of the sim is praxis's own (scene, tasks, env, loop.run_episode, control mode droid_impedance, exec_horizon 24, History stack, MP4 writer); only
`policy.type=model` is replaced: ModelPolicy is built on a PraxisClient (cuda/praxis/praxis_client.py) instead of PraxisPipeline. Options (before --set):
  --policy model|scripted|hold   (scripted/hold = the sim's own, for smoke tests)
  --expert-fmt mixed|nvfp4|nvfp4a16|bf16 (bf16 = parity-only streamed bf16 experts, ~13 GB device)   --vae-mode keep|reload   --no-decode (no predicted-video decode)   --skip-last-video   --cfg-video X
"""
import argparse
import json
import numpy as np
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import praxis_client as pc  # noqa: E402  (puts the praxis source tree on sys.path)

# praxis.sim sets the OpenGL backend of MuJoCo: it must load before mujoco
import praxis.sim  # noqa: E402,F401
from praxis.sim import loop  # noqa: E402
from praxis.sim.config import load_config  # noqa: E402
PRAXIS_ROOT = os.environ.get("PRAXIS_ROOT", os.path.expanduser("~/praxis"))

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--policy", default="model")
ap.add_argument("--checkpoint", default=PRAXIS_ROOT + "/praxis-1-droid")
ap.add_argument("--expert-fmt", default="mixed")
ap.add_argument("--vae-mode", default="keep")
ap.add_argument("--no-decode", action="store_true")
ap.add_argument("--skip-last-video", action="store_true")
ap.add_argument("--cfg-video", type=float, default=None)
ap.add_argument("--num-steps", type=int, default=None, help="Euler steps per chunk (default: the config's 2)")
ap.add_argument("--cfg-scale", type=float, default=None, help="text guidance of the ACTIONS (config 1.0 = off; needs the unconditional pass)")
ap.add_argument("--savgol", type=int, default=0, help="opt-in Savitzky-Golay window (praxis serve smoothing, window 9 order 3 are its defaults) over each chunk's relative xyz + grip; 0 = off")
ap.add_argument("--sg-order", type=int, default=3)
ap.add_argument("--stream", type=int, default=2)
ap.add_argument("--server-cmd", default=None, help="opt-in remote policy server: complete shell command (stdin/stdout = binary protocol), e.g. ssh <amd-host> '...hip_praxis_server_full ...'; sets PRAXIS_SERVER_CMD (CUDA default unchanged)")
ap.add_argument("--rebase", type=int, default=8, help="boundary mitigation, DEFAULT 8 (NOT praxis chunked behaviour; --rebase 0 = praxis behaviour): shift the new chunk's xyz so its first row continues the previous command (last + last step), decaying linearly to 0 over this many rows")
ap.add_argument("--dump-obs", default=None, help="dir: save the observation stack of every call (obs_call<N>.npz) for reference replays")
a = ap.parse_args()
if a.server_cmd:
    os.environ["PRAXIS_SERVER_CMD"] = a.server_cmd

paths = [f"model.checkpoint={a.checkpoint}", f"model.vae_weights={PRAXIS_ROOT}/wan2.2-ti2v/Wan2.2_VAE.pth", f"model.text_encoder={PRAXIS_ROOT}/qwen3.5-2b",
         "viewer.enabled=false", f"policy.type={a.policy}"]
if a.no_decode or a.skip_last_video:
    paths.append("policy.decode_video=false")
cfg = load_config(a.config, paths + a.set)
out_dir = Path(cfg.output.dir)
out_dir.mkdir(parents=True, exist_ok=True)


class CudaModelPolicy(loop.ModelPolicy):
    """loop.ModelPolicy on a PraxisClient; the prompt of the task is taken at reset and its text conditioning is warmed on the server."""

    def __init__(self, pipe, session_cfg, decode_video):
        super().__init__(pipe, "", session_cfg, decode_video)
        self._warm = set()
        self.last = None

    def reset(self, env, seed):
        self.prompt = env.prompt
        if self.prompt not in self._warm:
            t0 = time.monotonic()
            self.pipe.warm_text([self.prompt])
            self._warm.add(self.prompt)
            print(f"[text] warmed {self.prompt!r} in {time.monotonic() - t0:.1f} s (server peak {self.pipe.server.last['peak_gb']:.2f} GB over ctx)", flush=True)
        super().reset(env, seed)
        self.ncall = 0
        self.last = None

    def act(self, history):
        if a.dump_obs:
            os.makedirs(a.dump_obs, exist_ok=True)
            o = history.obs_dict()
            np.savez(f"{a.dump_obs}/obs_call{self.ncall}.npz", exterior=o["image.exterior"], wrist=o["image.wrist"], ee_pose=o["robot_state.ee_pose"], grip=o["grip"])
        self.ncall += 1
        out = super().act(history)
        if a.rebase > 0:
            xyz = np.stack([np.asarray(c["robot_command"]["pose"])[:3] for c in out.commands]).astype(np.float64)
            if self.last is not None:
                last, vel = self.last
                off = (last + vel) - xyz[0]
                for k, c in enumerate(out.commands):
                    w = max(0.0, 1.0 - k / a.rebase)
                    pose = np.array(c["robot_command"]["pose"], dtype=np.float32)
                    pose[:3] += (w * off).astype(np.float32)
                    c["robot_command"]["pose"] = pose
            new = np.stack([np.asarray(c["robot_command"]["pose"])[:3] for c in out.commands]).astype(np.float64)
            self.last = (new[-1], new[-1] - new[-2])
        return out


policy = None
pipe = None
if a.policy == "model":
    s = cfg.sampler
    pipe = pc.PraxisClient(
        a.checkpoint, sampler=pc.SamplerSettings(num_steps=a.num_steps or s.num_steps, t_shift=s.t_shift, cfg_scale=a.cfg_scale if a.cfg_scale is not None else s.cfg_scale,
                                                 sg_window=a.savgol, sg_order=a.sg_order,
                                                 cfg_video_scale=a.cfg_video if a.cfg_video is not None else s.cfg_video_scale,
                                                 skip_last_video=a.skip_last_video),
        env=({"PRAXIS_BF16_MMAP": "1"} if a.expert_fmt == "bf16" else
             {"PRAXIS_EXPERT_FMT": a.expert_fmt, "PRAXIS_PACK_CACHE": os.environ.get("PRAXIS_PACK_CACHE", PRAXIS_ROOT + "/praxis-1-droid.nvfp4.packcache")}),
        vae_mode=a.vae_mode, stream=a.stream, log_path=str(out_dir / "server.log"))
    policy = CudaModelPolicy(pipe, cfg.session, cfg.policy.decode_video)
try:
    result = loop.run(cfg, viewer=None, policy=policy)
    print(f"success {sum(e.success for e in result.episodes)}/{len(result.episodes)} episodes")
    print("outputs:", Path(result.out_dir).resolve())
    for ep in result.episodes:
        print(f"  seed {ep.seed}: success {ep.success} ticks {ep.ticks} calls {ep.calls} depths {ep.call_depths} mean call {ep.mean_call_s:.2f} s video {ep.video}")
    if pipe is not None:
        tm = pipe.timings
        info = pipe.server.info()
        summary = dict(n_calls=len(tm), server_info=info, calls=tm)
        (out_dir / "calls.json").write_text(json.dumps(summary, indent=1))
        print("server:", info)
finally:
    if pipe is not None:
        pipe.close()
