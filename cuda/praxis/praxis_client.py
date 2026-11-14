#!/usr/bin/env python3
"""Python client of cuda/praxis/praxis_server: a drop-in stand-in for praxis.pipeline.PraxisPipeline (CPU side only).

The observation -> commands logic is praxis's own, imported unchanged from the praxis source tree (approved, it is the target model's repo):
  * praxis.single_arm.SingleArmServeSession does the whole session state machine: tile canvas (fit rule, antialiased resize), executed-frame stream
    (ContiguousStreamContext, max_context_chunks), anchor/clean latents, boundary poses, proprio normalisation, the client-frame <-> flange-frame
    pose conversion, rot6d <-> matrix, relative-to-current commands -> absolute robot_command/cartesian_pos (R_flat9) + target_grip + timestamps.
  * only the three heavy calls are redirected to the CUDA server: `encode` (Wan2.2 VAE), sampling (Qwen conditioning + window_joint sampler) and the
    optional video decode. The video/action noise is drawn here with praxis.sampler.seeded_normal (same torch CPU generator as the reference), so the
    server consumes bit-identical noise.
The torch used here is CPU-only (tensors stay on the CPU, the GPU belongs to the server).

    pipe = PraxisClient(checkpoint="$PRAXIS_ROOT/praxis-1-droid")            # spawns cuda/praxis/praxis_server
    sess = pipe.session(prompt="Pick up the red cube.", seed=0, max_context_chunks=1, control_mode="droid_impedance", exec_horizon=24)
    res = sess.step(obs)          # obs as praxis.single_arm expects (stacks of frames, ee_pose, grip, descriptor); res.commands = robot_command rows
    frames = sess.decode_video(res)
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PRAXIS_ROOT = os.environ.get("PRAXIS_ROOT", os.path.expanduser("~/praxis"))
PRAXIS_OUT = os.environ.get("PRAXIS_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tmp"))
PRAXIS_SRC = os.environ.get("PRAXIS_SRC", PRAXIS_ROOT + "/praxis-1-oss")
if PRAXIS_SRC not in sys.path:
    sys.path.insert(0, PRAXIS_SRC)

import torch  # noqa: E402

from praxis import single_arm  # noqa: E402
from praxis.robot_io import NormStats, text_encoder_prompt  # noqa: E402
from praxis.sampler import ACTION_SALT, VIDEO_SALT, seeded_normal  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MAGIC_Q, MAGIC_R = 0x51585250, 0x52585250


@dataclass
class SamplerSettings:
    """The sampler settings of configs/sim_droid.yaml (window_joint, euler)."""

    num_steps: int = 2
    t_shift: float = 12.5
    cfg_scale: float = 1.0  # action guidance
    cfg_video_scale: float = 4.0
    sg_window: int = 0  # opt-in: Savitzky-Golay smoothing (praxis serve smoothing=savgol, defaults 9/3) of each chunk's relative xyz + grip; 0 = off (praxis' chunked sim behaviour)
    sg_order: int = 3
    skip_last_video: bool = False  # server skips the last video step (no predicted video); only when the video is not decoded


@dataclass
class StepResult:
    commands: list
    action_rows_norm: np.ndarray
    action_rel: np.ndarray
    pred_video_latent: torch.Tensor | None
    pred_action_norm: torch.Tensor
    depth: int
    seed: int
    prompt: str
    inputs: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)


class ServerError(RuntimeError):
    pass


class PraxisServer:
    """The server process and its binary protocol (see cuda/praxis/praxis_server.c)."""

    def __init__(self, checkpoint, *, env=None, log_path=None, vae_dir=None, qwen_dir=None, stream=2, vae_mode="keep", trim=2,
                 text_cache=None, binary=None):
        binary = binary or str(REPO / "cuda/praxis/praxis_server")
        cmd = [binary, "--policy", str(checkpoint), "--stream", str(stream), "--vae-mode", vae_mode, "--trim", str(int(trim))]
        if vae_dir:
            cmd += ["--vae", str(vae_dir)]
        if qwen_dir:
            cmd += ["--qwen", str(qwen_dir)]
        if text_cache:
            cmd += ["--text-cache", str(text_cache)]
        e = dict(os.environ)
        e.update(env or {})
        self.log = open(log_path or PRAXIS_OUT + "/praxis_sim/server.log", "ab")
        # opt-in remote server: PRAXIS_SERVER_CMD is the COMPLETE shell command that starts a server speaking this protocol on its stdin/stdout (e.g. through
        # ssh to the AMD host); the local `cmd`/`env` above are then ignored (give the policy path, --stream, --vae-mode, PRAXIS_EXPERT_FMT=... in the command).
        remote = os.environ.get("PRAXIS_SERVER_CMD")
        if remote:
            self.proc = subprocess.Popen(remote, shell=True, cwd=str(REPO), env=e, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        else:
            self.proc = subprocess.Popen(cmd, cwd=str(REPO), env=e, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.last = {}

    def _rd(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise ServerError("praxis_server closed the pipe (see its log)")
            buf += chunk
        return bytes(buf)

    def call(self, op, payload=b""):
        t0 = time.monotonic()
        self.proc.stdin.write(struct.pack("<IIQ", MAGIC_Q, op, len(payload)) + payload)
        self.proc.stdin.flush()
        magic, status, ln = struct.unpack("<IiQ", self._rd(16))
        if magic != MAGIC_R:
            raise ServerError("bad response magic")
        if status:
            raise ServerError(f"praxis_server op {op} failed with status {status}")
        body = self._rd(ln)
        ms, aux_ms, peak_gb, after_gb = struct.unpack("<4d", body[:32])
        self.last = dict(op=op, ms=ms, aux_ms=aux_ms, peak_gb=peak_gb, after_gb=after_gb, rtt_ms=(time.monotonic() - t0) * 1e3)
        return body[32:]

    def encode(self, u8_3thw: np.ndarray):
        _, T, H, W = u8_3thw.shape
        body = self.call(1, struct.pack("<3i", T, H, W) + np.ascontiguousarray(u8_3thw).tobytes())
        c, tl, h, w = struct.unpack("<4i", body[:16])
        return np.frombuffer(body, dtype=np.float32, offset=16).reshape(c, tl, h, w).copy()

    def warm_text(self, prompts):
        pl = struct.pack("<i", len(prompts))
        for p in prompts:
            b = p.encode()
            pl += struct.pack("<i", len(b)) + b + b"\0" * (-len(b) % 4)
        self.call(3, pl)

    def sample(self, prompt, uncond, K, steps, skipv, t_shift, cfg_v, cfg_a, anchor, clean, proprio, noise_v, noise_a):
        pb, ub = prompt.encode(), uncond.encode()
        h, w = anchor.shape[-2:]
        pl = struct.pack("<5i", K, h, w, steps, int(skipv)) + struct.pack("<3d", t_shift, cfg_v, cfg_a)
        pl += struct.pack("<i", len(pb)) + pb + b"\0" * (-len(pb) % 4) + struct.pack("<i", len(ub)) + ub + b"\0" * (-len(ub) % 4)
        for a in (anchor, clean, proprio, noise_v, noise_a):
            if a is not None:
                pl += np.ascontiguousarray(a, dtype=np.float32).tobytes()
        body = self.call(2, pl)
        c, f, h, w, a_dim = struct.unpack("<5i", body[:20])
        vid = np.frombuffer(body, dtype=np.float32, count=c * f * h * w, offset=20).reshape(c, f, h, w).copy()
        act = np.frombuffer(body, dtype=np.float32, count=f * a_dim, offset=20 + 4 * c * f * h * w).reshape(f, a_dim).copy()
        return vid, act

    def decode(self, latent_cthw: np.ndarray, keep: int):
        c, tl, h, w = latent_cthw.shape
        body = self.call(4, struct.pack("<4i", tl, h, w, keep) + np.ascontiguousarray(latent_cthw, dtype=np.float32).tobytes())
        n, H, W = struct.unpack("<3i", body[:12])
        return np.frombuffer(body, dtype=np.uint8, offset=12).reshape(n, H, W, 3).copy()

    def info(self):
        return self.call(5).decode()

    def close(self):
        try:
            self.proc.stdin.write(struct.pack("<IIQ", MAGIC_Q, 6, 0))
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        self.log.close()


class _Policy:  # minimal stand-in for PraxisPipeline.policy
    def __init__(self, config):
        self.config = config


class PraxisClient:
    """Stand-in for PraxisPipeline: same attributes the sim loop and Session use (policy.config, embodiment, stats, session())."""

    def __init__(self, checkpoint, sampler: SamplerSettings | None = None, server: PraxisServer | None = None, **server_kw):
        self.checkpoint = str(checkpoint)
        self.config = json.load(open(Path(checkpoint) / "config.json"))
        self.policy = _Policy(self.config)
        self.embodiment = str(self.config["embodiment"])
        if self.embodiment != "single_arm":
            raise ValueError("praxis_client supports the single-arm checkpoint")
        norm = self.config["normalization"]
        self.stats = NormStats(**{k: np.asarray(norm[k], dtype=np.float32) for k in ("action_q01", "action_q99", "proprio_mean", "proprio_std")})
        self.sampler = sampler or SamplerSettings()
        self.server = server or PraxisServer(checkpoint, **server_kw)
        self.device = torch.device("cpu")
        self.timings: list[dict] = []

    def session(self, prompt="", seed=0, max_context_chunks=1, control_mode=None, exec_horizon=24, sampler=None, **_ignored):
        return Session(self, prompt, seed, max_context_chunks, control_mode, exec_horizon, sampler or self.sampler)

    def warm_text(self, prompts):
        self.server.warm_text([text_encoder_prompt(p) for p in prompts] + [self.config["text_encoder"]["blank_caption"]])

    def close(self):
        self.server.close()


class Session:
    def __init__(self, pipe: PraxisClient, prompt, seed, max_context_chunks, control_mode, exec_horizon, sampler):
        self.pipe, self.sampler, self.seed = pipe, sampler, int(seed)
        cfg = pipe.config
        v = cfg["video"]
        self.uncond = cfg["text_encoder"]["blank_caption"]
        self.enc_ms = 0.0
        self.enc_rtt_ms = 0.0
        self.state = single_arm.SingleArmServeSession(
            encode=self._encode, stats=pipe.stats, device="cpu", prompt=prompt, max_context_chunks=max_context_chunks,
            fps=float(cfg["chunking"].get("grid_hz", single_arm.FPS)), dims=single_arm.q99_dims(cfg),
            control_mode=control_mode or "droid_impedance", exec_horizon=exec_horizon,
            n_chunks=int(cfg["chunking"].get("n_chunks", single_arm.N_CHUNKS)),
            tile_size=(v["tile_height"], v["tile_width"]), aspect_tol=v.get("aspect_tol"), antialias=v.get("antialias", False))

    def _encode(self, clip):  # clip (1,3,T,H,W) float in [0,255] (integers) -> latent (1,192,T',h,w)
        u8 = clip[0].round().clamp(0, 255).to(torch.uint8).contiguous().numpy()
        lat = self.pipe.server.encode(u8)
        self.enc_ms += self.pipe.server.last["ms"]
        self.enc_rtt_ms += self.pipe.server.last["rtt_ms"]
        self.last_encode = dict(self.pipe.server.last)
        return torch.from_numpy(lat)[None]

    @torch.no_grad()
    def step(self, obs, executed_rows=None):
        t0 = time.monotonic()
        self.enc_ms = 0.0
        self.enc_rtt_ms = 0.0
        self.last_encode = {}
        inputs = self.state.prepare(obs, executed_rows)
        t_prep = time.monotonic() - t0
        K = len(inputs.cleans) + 1
        cfg = self.pipe.config
        lat = int(cfg["chunking"]["latents_per_chunk"])
        c, h, w = inputs.anchor.shape[1], inputs.anchor.shape[3], inputs.anchor.shape[4]
        seed = self.seed + inputs.chunk_index
        a_dim = int(cfg["action_expert"]["action_patch_dim"])
        noise_v = seeded_normal((1, c, K * lat, h, w), seed, VIDEO_SALT).numpy()
        noise_a = seeded_normal((K * lat, a_dim), seed, ACTION_SALT).numpy()
        clean = torch.cat(inputs.cleans, dim=2).numpy()[0] if inputs.cleans else None
        s = self.sampler
        t1 = time.monotonic()
        vid, act = self.pipe.server.sample(inputs.prompt, text_encoder_prompt(self.uncond), K, s.num_steps, s.skip_last_video, s.t_shift,
                                           s.cfg_video_scale, s.cfg_scale, inputs.anchor.numpy()[0], clean, inputs.proprio_all.numpy()[0],
                                           noise_v[0], noise_a)
        samp = dict(self.pipe.server.last)
        t_samp = time.monotonic() - t1
        actions = torch.from_numpy(act)[None]  # (1, 6, 40) normalised action tokens
        video = None if s.skip_last_video else torch.from_numpy(vid)[None]
        commands = self.state.finish(inputs, actions)
        rows, rel = self.state.decode_actions(inputs, actions)
        if s.sg_window and len(rel) >= s.sg_window > s.sg_order:  # same as praxis.serve.policy._smoothed (xyz + grip, rotation untouched, grip clipped)
            from scipy.signal import savgol_filter
            from praxis import robot_io
            sm = robot_io.denormalize_rows(rows, self.state.stats, self.state.dims)
            for dim in single_arm.Q99_DIMS:
                sm[:, dim] = savgol_filter(sm[:, dim], s.sg_window, s.sg_order)
            rel = robot_io.clip_grip(sm, single_arm.GRIP_DIMS)
            commands = self.state.rel_to_wire(rel, inputs.pose8_now)
            if self.state.exec_horizon is not None:
                commands = commands[: self.state.exec_horizon]
        timing = dict(prepare_s=t_prep, encode_ms=self.enc_ms, encode_rtt_ms=self.enc_rtt_ms, sample_rtt_ms=samp["rtt_ms"], sample_s=t_samp, server_sample_ms=samp["ms"], server_aux_ms=samp["aux_ms"],
                      peak_gb_sample=samp["peak_gb"], peak_gb_encode=self.last_encode.get("peak_gb"), total_s=time.monotonic() - t0, depth=inputs.depth)
        self.pipe.timings.append(timing)
        return StepResult(commands=commands, action_rows_norm=rows, action_rel=rel, pred_video_latent=video, pred_action_norm=actions,
                          depth=inputs.depth, seed=seed, prompt=inputs.prompt,
                          inputs=dict(anchor=inputs.anchor, cleans=inputs.cleans, proprio=inputs.proprio, proprio_all=inputs.proprio_all,
                                      boundary_poses=inputs.boundary_poses, pose_now=inputs.pose8_now),
                          timing=timing)

    @torch.no_grad()
    def decode_video(self, result: StepResult):
        """uint8 frames (n, 3, H, W): one per command row (praxis Session.decode_video)."""
        latent = result.pred_video_latent
        if latent is None:
            raise ValueError("this call skipped the predicted video (skip_last_video)")
        cleans = result.inputs["cleans"]
        if cleans:
            latent = torch.cat([*cleans, latent.to(cleans[0].dtype)], dim=2)
        frames = self.pipe.server.decode(latent[0].numpy(), len(result.action_rows_norm))
        t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        return t[-len(result.action_rows_norm):][: len(result.commands)]
