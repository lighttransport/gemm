#!/usr/bin/env python3
"""Standalone Qwen-Image 2.1 web demo server.

The server intentionally shells out to the checked-in native and reference
drivers.  This keeps the web process small and prevents two model copies from
being resident in the 12--16 GB CUDA device at once.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
DEFAULT_MODEL = Path("/mnt/disk1/models/qimg-21")
DEFAULT_QUANT = ROOT / "tmp/qimg21-int8-package"
DEFAULT_PYTHON = ROOT / "tmp/qimg21-ref-venv/bin/python"
DEFAULT_FAST = ROOT / "cuda/qimg21/test_cuda_qimg21_fast"
DEFAULT_FAST_PACKAGES = {"int8": Path("/mnt/disk1/models/qimg-21-fast/int8-smooth-a0.6"),
                         "nvfp4": Path("/mnt/disk1/models/qimg-21-fast/nvfp4-svd-a0.5-m")}
# Fast CUDA denoiser presets (test_cuda_qimg21_fast --preset) and the
# pack_fast.py weight package each needs.
FAST_PRESETS = {"low8": "int8", "low8-fp4": "nvfp4", "fast12": "int8", "accurate": None}
# The demo's names for the fast runner's attention kernels.
FAST_ATTENTION = {"sage": "sage", "flash": "flash", "exact": "cutlass-efficient"}
# Output size limits per pixel side. The fast runner bounds its own memory, so
# the caps are about what the other stages can afford: the parity harness and the
# PyTorch reference stay at 1024, an untiled fast run goes to 2048 (the VAE
# decode tiles itself above 1024), and a tiled coarse-to-fine refine reaches 4096.
SIZE_LIMIT_REFERENCE = 1024
SIZE_LIMIT_FAST = 2048
SIZE_LIMIT_TILED = 4096
# Where the PyTorch reference may run. `cuda` and `rocm` name a PyTorch build as
# much as a device: each GPU vendor needs its own Torch, and the CPU route works
# with either. Only cuda and rocm can be compared against the native runner
# numerically; a CPU reference runs different kernels on different hardware, so
# that pairing is about composition rather than agreement.
REFERENCE_DEVICES = ("cuda", "rocm", "cpu")
REFERENCE_PARITY_DEVICES = {"cuda", "rocm"}
# Tiled coarse-to-fine refine, from cuda/qimg21/native_generate.py. A refine
# tile is composed as its own canvas, so the tile is a quality dial, not only a
# memory one: the default is the largest that fits the preset's budget, found by
# the driver. Everything left at None lets the driver choose.
TILE_DEFAULTS = {"tile_overlap": 8, "vae_tile_overlap": 8, "vae_tile_bleed": 2}
MAX_BODY = 32 * 1024
# A job's progress is polled rather than streamed, so a client can attach late
# or drop off without leaving a thread behind. Steps are returned from a cursor
# because a long run would otherwise resend its whole history several times a
# second.
PROGRESS_TTL = 900.0
# A torch import is slow but bounded; a hung interpreter must not hang /api/health.
PROBE_TIMEOUT = 120.0
MAX_EVENTS = 4096

# Which pipeline stage a subprocess is, from the driver's "+ <command>" line.
# The order matters: the text encoder is also invoked for the condition image.
STAGE_PATTERNS = (
    ("test_cuda_qimg21_fast --serve", "load resident denoiser"),
    ("test_cuda_qimg21_vae --serve", "load resident VAE"),
    ("reference.py --serve", "load resident PyTorch reference"),
    ("test_cuda_qimg21_vae_encode", "encode condition image"),
    ("test_cuda_qimg21_vision", "encode condition image"),
    ("test_cuda_qimg21_text", "encode prompt"),
    ("make_native_fixture.py", "sample noise"),
    ("test_cuda_qimg21_fast", "denoise"),
    ("test_cuda_qimg21_native", "denoise"),
    ("test_hip_qimg21_native", "denoise"),
    ("test_cuda_qimg21_vae", "decode"),
    ("test_hip_qimg21_vae", "decode"),
    ("reference.py", "PyTorch reference"),
)


class Progress:
    """Turns a pipeline log into stage and step events with timings.

    Everything here is a projection of what the runners already print, so the
    numbers are the runners' own: a step's duration is the denoiser's measured
    device time, not a difference of poll timestamps. The accumulated figure is
    the sum of the step times the denoiser reported, which is the number worth
    showing -- wall clock includes weight streaming and is tracked separately.
    """

    STEP = re.compile(r"^fast: step (\d+)/(\d+) sigma=([0-9.]+)(?: ([0-9.]+) ms)?$")
    LAYER = re.compile(r"^text: layer (\d+)/(\d+)\b")
    TILE = re.compile(r"^fast: tile (\d+)/(\d+)\b")
    DECODE = re.compile(r"^qimg21-vae: tile (\d+)/(\d+)\b")
    STAGE_DONE = re.compile(r"^\s+\(([^:]+): ([0-9.]+) s\)\s*$")
    PLAN = re.compile(r"^fast: plan (.+)$")
    REFINE = re.compile(r"^fast: tiled refine (.+)$")
    PASS = re.compile(r"^(base pass|refine pass|refine tile|decoding .+)$")
    # "timing: <phase> <seconds> s[ (<detail>)]", printed by every component.
    TIMING = re.compile(r"^timing: (.+?) ([0-9]+(?:\.[0-9]+)?) s(?: \((.*)\))?$")

    def __init__(self, started: float):
        self.started = started
        self.stage: str | None = None
        self.stage_started = started
        self.accum_ms = 0.0          # measured denoiser step time so far
        self.stage_steps = 0
        self.total_steps = 0
        self.notes: list[str] = []
        self.done = False

    def feed(self, line: str, now: float) -> dict | None:
        """One log line in, at most one event out."""
        line = line.rstrip("\n")
        if line.startswith("+ "):
            self.stage = _stage_name(line)
            self.stage_started = now
            return {"kind": "stage", "stage": self.stage, "command": line[2:].split(" ")[0].rsplit("/", 1)[-1]}
        if (match := self.TIMING.match(line)):
            return {"kind": "timing", "stage": self.stage, "label": match.group(1),
                    "seconds": float(match.group(2)), "detail": match.group(3),
                    "elapsed_ms": (now - self.started) * 1000.0}
        if (match := self.STAGE_DONE.match(line)):
            return {"kind": "stage_done", "stage": match.group(1),
                    "seconds": float(match.group(2)), "elapsed_ms": (now - self.started) * 1000.0}
        if (match := self.STEP.match(line)):
            index, total, sigma = int(match.group(1)), int(match.group(2)), float(match.group(3))
            ms = float(match.group(4)) if match.group(4) else None
            if ms is not None:
                self.accum_ms += ms
            self.stage_steps += 1
            self.total_steps += 1
            return {"kind": "step", "stage": self.stage, "index": index, "total": total,
                    "sigma": sigma, "ms": ms, "accum_ms": self.accum_ms,
                    "elapsed_ms": (now - self.started) * 1000.0}
        if (match := self.LAYER.match(line)):
            return {"kind": "layer", "stage": self.stage, "index": int(match.group(1)),
                    "total": int(match.group(2))}
        if (match := self.TILE.match(line)):
            return {"kind": "tile", "stage": self.stage, "index": int(match.group(1)),
                    "total": int(match.group(2)),
                    "elapsed_ms": (now - self.started) * 1000.0}
        if (match := self.DECODE.match(line)):
            return {"kind": "tile", "stage": "decode", "index": int(match.group(1)),
                    "total": int(match.group(2)),
                    "elapsed_ms": (now - self.started) * 1000.0}
        for pattern, kind in ((self.PLAN, "plan"), (self.REFINE, "plan"), (self.PASS, "plan")):
            if match := pattern.match(line):
                if len(self.notes) < 12:
                    self.notes.append(line)
                return {"kind": kind, "text": line}
        return None

    def finish(self) -> None:
        self.done = True


def timing_breakdown(log: Path, default_stage: str) -> list[dict]:
    """One run's phases, read back from its log: each component's own
    "timing:" lines, grouped under the pipeline stage that printed them, plus
    each stage's wall time as the driver measured it."""
    if not log.is_file():
        return []
    tracker = Progress(0.0)
    tracker.stage = default_stage
    rows: list[dict] = []
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        event = tracker.feed(line, 0.0)
        if not event:
            continue
        if event["kind"] == "timing":
            rows.append({"stage": event["stage"] or default_stage, "label": event["label"],
                         "seconds": event["seconds"], "detail": event["detail"]})
        elif event["kind"] == "stage_done":
            rows.append({"stage": tracker.stage or default_stage, "label": "stage total",
                         "seconds": event["seconds"], "detail": event["stage"], "total": True})
    return rows


def generation_seconds(rows: list[dict]) -> float | None:
    """Image generation alone: the denoising loop, summed over passes."""
    spans = [row["seconds"] for row in rows if "(image generation)" in row["label"]]
    return sum(spans) if spans else None


def _stage_name(command_line: str) -> str:
    for needle, name in STAGE_PATTERNS:
        if needle in command_line:
            return name
    return "pipeline"


def _tail(log: Path, progress, stop: threading.Event, interval: float = 0.15) -> None:
    """Forward newly written log lines until `stop`, never splitting a line."""
    offset = 0
    pending = ""
    while not stop.is_set():
        try:
            with log.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                chunk = stream.read()
                offset = stream.tell()
        except OSError:
            chunk = ""
        if chunk:
            pending += chunk
            *lines, pending = pending.split("\n")
            for line in lines:
                progress(line)
        stop.wait(interval)
    # One last pass so the tail of the log is not dropped.
    try:
        with log.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(offset)
            chunk = stream.read()
    except OSError:
        chunk = ""
    for line in (pending + chunk).split("\n"):
        if line.strip():
            progress(line)


# Per-step previews. Each denoising step's dumped latent becomes a picture
# without running the VAE: see cuda/qimg21/fit_latent_preview.py.
PREVIEW_WEIGHTS = ROOT / "cuda/qimg21/latent_preview.npy"
PREVIEW_MAX_SIDE = 384
PREVIEW_INTERVAL = 0.3


def flow_sigmas(steps: int, tokens: int) -> list[float]:
    """The FlowMatch schedule the runners use (qimg21_flow_sigmas in
    test_cuda_qimg21_native.c): steps + 1 sigmas ending at 0."""
    import math
    mu = tokens * (0.9 - 0.5) / (8192.0 - 256.0) + 0.5 - (0.9 - 0.5) / (8192.0 - 256.0) * 256.0
    emu = math.exp(mu)
    if steps == 1:
        return [1.0, 0.0]
    sigmas = [emu / (emu + (1.0 / (1.0 - i / steps) - 1.0)) for i in range(steps)]
    scale = (1.0 - sigmas[-1]) / (1.0 - 0.02)
    return [1.0 - (1.0 - sigma) / scale for sigma in sigmas] + [0.0]


def denoised_estimate(latent, previous, sigma_from: float, sigma_to: float):
    """The clean latent a step is heading for. Step i moves x from sigma_from
    to sigma_to along one velocity v, so v = dx / dsigma and x0 = x - sigma_to v.
    A preview of x0 shows the picture as it is being decided, where x itself is
    mostly noise until the last few steps."""
    if previous is None or sigma_to <= 0.0 or sigma_from == sigma_to:
        return latent
    velocity = (latent - previous) / (sigma_to - sigma_from)
    return latent - sigma_to * velocity


_preview_weights = None


def preview_image(latents, h_tokens: int, w_tokens: int) -> str | None:
    """Normalized [tokens, 64] latents -> a PNG data URL, or None when the
    preview map is not installed."""
    global _preview_weights
    import numpy as np
    from io import BytesIO
    from PIL import Image

    if _preview_weights is None:
        if not PREVIEW_WEIGHTS.is_file():
            return None
        _preview_weights = np.load(PREVIEW_WEIGHTS).astype(np.float32)
    weights = _preview_weights
    patch = int(round((weights.shape[1] // 3) ** 0.5))
    # The map reads each token's 3x3 neighbourhood (fit_latent_preview.features).
    grid = np.pad(latents.astype(np.float32).reshape(h_tokens, w_tokens, 64), ((1, 1), (1, 1), (0, 0)),
                  mode="edge")
    near = [grid[dy:dy + h_tokens, dx:dx + w_tokens] for dy in range(3) for dx in range(3)]
    x = np.concatenate(near, axis=2).reshape(h_tokens * w_tokens, 9 * 64)
    x = np.hstack([x, np.ones((x.shape[0], 1), np.float32)])
    rgb = (x @ weights).reshape(h_tokens, w_tokens, patch, patch, 3).transpose(0, 2, 1, 3, 4)
    rgb = np.clip(rgb.reshape(h_tokens * patch, w_tokens * patch, 3) * 127.5 + 127.5, 0, 255)
    image = Image.fromarray(rgb.astype(np.uint8))
    if max(image.size) > PREVIEW_MAX_SIDE:
        image.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE), Image.BILINEAR)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class StepPreviews:
    """Watch a run's step dumps and emit one preview event per new step.

    The runners already write step_NNN.npy after every step for parity work,
    so this reads what is there rather than asking them for anything new. A
    file caught half-written fails to load and is simply retried next poll.
    `grids` lists the (h_tokens, w_tokens, steps) a dump may belong to: a tiled
    run's base pass writes a smaller grid with its own step count.
    """

    def __init__(self, source: str, step_dir: Path, initial: Path | None,
                 grids: list[tuple[int, int, int]], emit, interval: float = PREVIEW_INTERVAL):
        self.source, self.step_dir, self.initial = source, step_dir, initial
        self.grids, self.emit, self.interval = grids, emit, interval
        self.seen: set[str] = set()
        self.latents: dict[int, object] = {}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10.0)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            self.sweep()
            self.stop_event.wait(self.interval)
        self.sweep()  # the last step lands just before the runner exits

    def _load(self, path: Path):
        import numpy as np
        try:
            value = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            return None
        if value.size % 64 or not np.isfinite(value).all():
            return None
        return value.reshape(-1, 64)

    def sweep(self) -> None:
        try:
            names = sorted(path.name for path in self.step_dir.glob("step_*.npy"))
        except OSError:
            return
        for name in names:
            if name in self.seen:
                continue
            try:
                index = int(name[5:8])
            except ValueError:
                self.seen.add(name)
                continue
            latent = self._load(self.step_dir / name)
            if latent is None:
                return  # still being written; keep order and retry
            self.seen.add(name)
            grid = next((g for g in self.grids if g[0] * g[1] == latent.shape[0]), None)
            if grid is None:
                continue
            h, w, steps = grid
            previous = self.latents.get(index - 1)
            if index == 0 and self.initial is not None and self.initial.is_file():
                previous = self._load(self.initial)
            if previous is not None and previous.shape != latent.shape:
                previous = None
            self.latents = {index: latent}
            sigmas = flow_sigmas(steps, h * w)
            if index + 1 >= len(sigmas):
                continue
            try:
                image = preview_image(denoised_estimate(latent, previous, sigmas[index], sigmas[index + 1]),
                                      h, w)
            except Exception:  # noqa: BLE001 - a preview never breaks a run
                image = None
            if image:
                self.emit({"kind": "preview", "source": self.source, "index": index + 1,
                           "total": steps, "sigma": sigmas[index + 1], "image": image})


# A stage "agrees" with the reference above this cosine. Looser than the
# compare.py BF16 gate on purpose: this marks where a picture starts to go wrong,
# not whether a kernel change passes parity.
DIVERGENCE_COSINE = 0.999


def _parity_helpers():
    """cuda/qimg21/compare.py is a script, not a package; load it by path so the
    demo measures with exactly the math the parity gate uses."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("qimg21_compare", ROOT / "cuda/qimg21/compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_runs(ref_dir: Path, native_work: Path, native_image: Path) -> dict:
    """Numbers for a compare: where along the pipeline the native run leaves the
    reference. Text embeddings, then every denoising step, then the decoded
    picture; the first stage below DIVERGENCE_COSINE is the one to look at."""
    import numpy as np
    from PIL import Image

    helpers = _parity_helpers()
    stages: list[dict] = []

    def measure(name: str, reference: Path, candidate: Path) -> None:
        if not reference.is_file() or not candidate.is_file():
            stages.append({"stage": name, "missing": True})
            return
        try:
            cosine, rel_l2 = helpers._cosine_error(np.load(reference), np.load(candidate))
        except ValueError as exc:
            stages.append({"stage": name, "error": str(exc)})
            return
        stages.append({"stage": name, "cosine": cosine, "rel_l2": rel_l2})

    measure("initial latents", ref_dir / "initial_latents.npy", native_work / "latents.npy")
    measure("text embeddings", ref_dir / "prompt_embeds.npy",
            native_work / "prompt" / "prompt_embeds.npy")
    for name in helpers._step_names(ref_dir):
        measure(f"step {int(name[5:8])}", ref_dir / name, helpers._step_path(native_work, name))

    image: dict = {}
    ref_rgba = ref_dir / "reference_rgba.npy"
    if ref_rgba.is_file() and native_image.is_file():
        ref = np.load(ref_rgba)[..., :3].astype(np.float64)
        got = np.asarray(Image.open(native_image).convert("RGB")).astype(np.float64)
        if ref.shape != got.shape:
            image = {"error": f"shape mismatch: {ref.shape} vs {got.shape}"}
        else:
            diff = got - ref
            rmse = float(np.sqrt(np.mean(diff * diff)))
            image = {"mae": float(np.mean(np.abs(diff))), "rmse": rmse,
                     "psnr": 20 * np.log10(255.0 / max(rmse, 1e-12)),
                     "max_abs": float(np.max(np.abs(diff)))}
            cosine, rel_l2 = helpers._cosine_error(ref, got)
            stages.append({"stage": "decoded image", "cosine": cosine, "rel_l2": rel_l2})

    first = next((s["stage"] for s in stages
                  if "cosine" in s and s["cosine"] < DIVERGENCE_COSINE), None)
    return {"stages": stages, "image": image, "first_divergence": first,
            "threshold": DIVERGENCE_COSINE}


# A resident denoiser that has sat unused this long gives its VRAM back.
RESIDENT_IDLE = 900.0
RESIDENT_START_TIMEOUT = 300.0


def _end_process(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def spawn_resident(command: list[str], log: Path, ready: str, sink, lines: list[str],
                   timeout: float = RESIDENT_START_TIMEOUT) -> subprocess.Popen | None:
    """Start one resident process and follow its log until it prints a line
    starting with `ready` or exits. Its output goes to `sink` and `lines` as it
    arrives, so the run that pays for the start shows where the time went."""
    stream = log.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    finally:
        stream.close()
    lines.append("+ " + " ".join(command))
    if sink:
        sink(lines[-1])
    offset, started, is_ready = 0, time.monotonic(), False
    while time.monotonic() - started < timeout:
        try:
            with log.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
        except OSError:
            chunk = ""
        for line in chunk.splitlines():
            lines.append(line)
            if sink:
                sink(line)
            is_ready = is_ready or line.startswith(ready)
        if is_ready or process.poll() is not None:
            break
        time.sleep(0.1)
    program = command[1] if Path(command[0]).name.startswith("python") else command[0]
    lines.append(f"  ({Path(program).name}: "
                 f"{time.monotonic() - started:.1f} s)")
    if sink:
        sink(lines[-1])
    if not is_ready:
        _end_process(process)
        return None
    return process


def send_resident(socket_path: Path, args: list[str], log: Path, progress=None) -> int | None:
    """One run on a resident server (the socket protocol the runners share):
    the reply streams into `log` -- and through it to `progress` -- as a
    one-shot run's output would. Returns the status, or None when the server
    could not be reached or went away mid-run."""
    import socket as socketlib
    if any("\t" in arg or "\n" in arg for arg in args):
        return None
    try:
        conn = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_STREAM)
        conn.connect(str(socket_path))
    except OSError:
        return None
    status = None
    log.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    with log.open("w", encoding="utf-8") as stream, conn:
        reader = None
        if progress:
            reader = threading.Thread(target=_tail, args=(log, progress, stop), daemon=True)
            reader.start()
        try:
            conn.sendall(("\t".join(args) + "\n").encode("utf-8"))
            pending = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                pending += chunk
                *lines, pending = pending.split(b"\n")
                for raw in lines:
                    line = raw.decode("utf-8", errors="replace")
                    if line.startswith("fast-serve: status "):
                        status = int(line.split()[2])
                    stream.write(line + "\n")
                stream.flush()
        except OSError:
            status = None
        finally:
            stop.set()
            if reader:
                reader.join(timeout=5.0)
    return status


class ResidentReference:
    """A PyTorch reference kept loaded (reference.py --serve, --offload
    resident). Its weights live in pinned host memory; each run puts the
    resident transformer blocks and the VAE on the device and takes them off
    again, so between runs it holds only its CUDA context (about 1 GB) and the
    native runners keep the GPU. Loading takes about 40 s, once."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.process: subprocess.Popen | None = None
        self.key: tuple | None = None
        self.socket: Path | None = None
        self.last_used = 0.0
        self.lock = threading.Lock()

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        with self.lock:
            _end_process(self.process)
            self.process, self.key = None, None
            if self.socket is not None:
                self.socket.unlink(missing_ok=True)

    def ensure(self, python: Path, model: Path, sink=None, start_log: Path | None = None) -> Path | None:
        key = (str(python), str(model))
        if not Path(python).exists() or not Path(model).is_dir():
            return None
        with self.lock:
            if self.alive() and self.key == key:
                self.last_used = time.monotonic()
                return self.socket
            _end_process(self.process)
            self.directory.mkdir(parents=True, exist_ok=True)
            self.socket = self.directory / "reference.sock"
            command = [str(python), "cuda/qimg21/reference.py", "--serve", str(self.socket), "--model", str(model),
                       "--device", "cuda", "--dtype", "bf16", "--offload", "resident"]
            lines: list[str] = []
            self.process = spawn_resident(command, self.directory / "reference.log", "reference: serving on",
                                          sink, lines)
            if start_log is not None:
                start_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            if self.process is None:
                self.key = None
                return None
            self.key = key
            self.last_used = time.monotonic()
            return self.socket

    def reap_if_idle(self, idle: float = RESIDENT_IDLE) -> bool:
        with self.lock:
            if self.alive() and time.monotonic() - self.last_used > idle:
                _end_process(self.process)
                self.process, self.key = None, None
                return True
        return False

    def state(self) -> dict:
        with self.lock:
            return {"running": self.alive(), "idle_s": round(time.monotonic() - self.last_used, 1)
                    if self.alive() else None}


class ResidentDenoiser:
    """At most one test_cuda_qimg21_fast --serve, kept loaded between runs.

    Loading the transformer is most of a short run's setup, and a refine or a
    new seed at the same size does not need it again. The process is keyed by
    everything its memory plan depends on -- binary, model, preset and package,
    grid, and whether CFG doubles the batch -- and replaced when a run needs a
    different one. It holds several GB, so any other kind of run (the PyTorch
    reference, a tiled refine, the parity harness) stops it first.
    """

    # The resident VAE decoder holds about 3 GB after its first decode; past
    # 1024^2 the denoiser needs that room, and the decode tiles itself anyway.
    VAE_MAX_TOKENS = 64

    def __init__(self, directory: Path):
        self.directory = directory
        self.process: subprocess.Popen | None = None
        self.key: tuple | None = None
        self.socket: Path | None = None
        # The VAE decoder rides along with the denoiser: same lifetime, own socket.
        self.vae_process: subprocess.Popen | None = None
        self.vae_socket_path: Path | None = None
        self.last_used = 0.0
        self.lock = threading.Lock()

    @staticmethod
    def key_for(demo: "Demo", cfg: dict) -> tuple | None:
        """The setup a run needs, or None when a resident process cannot serve it."""
        preset = cfg.get("preset")
        if (cfg["backend"] not in ("cuda", "rocm") or not preset or cfg["mode"] != "native" or cfg["upscale"] > 1.0
                or cfg["quantized"] or not demo.model.is_dir() or not demo.preset_available(preset, cfg["backend"])):
            return None
        kind = FAST_PRESETS[preset]
        package = str(demo.fast_packages[kind]) if kind else ""
        binary = demo.fast_binary(cfg["backend"])
        return (str(binary), binary.stat().st_mtime_ns, str(demo.model), preset, package,
                cfg["height"] // 16, cfg["width"] // 16, bool(cfg["negative_prompt"]),
                FAST_ATTENTION.get(cfg.get("attention") or "", ""))

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def vae_alive(self) -> bool:
        return self.vae_process is not None and self.vae_process.poll() is None

    def stop(self) -> None:
        with self.lock:
            self._stop()

    _end = staticmethod(_end_process)

    def _stop(self) -> None:
        self._end(self.process)
        self._end(self.vae_process)
        self.process = self.vae_process = None
        for path in (self.socket, self.vae_socket_path):
            if path is not None:
                path.unlink(missing_ok=True)
        self.key = None

    def _spawn(self, command: list[str], log: Path, ready: str, sink, lines: list[str]):
        return spawn_resident(command, log, ready, sink, lines)

    def vae_socket(self, demo: "Demo", cfg: dict, sink=None, start_log: Path | None = None) -> Path | None:
        """The resident VAE decoder's socket, starting it next to a running
        resident denoiser if needed. None means decode one-shot."""
        with self.lock:
            if not self.alive() or self.key is None:
                return None
            if max(cfg["height"], cfg["width"]) // 16 > self.VAE_MAX_TOKENS:
                return None
            if self.vae_process is not None and self.vae_process.poll() is None:
                return self.vae_socket_path
            if self.key[0].find("test_hip_") >= 0:
                return None
            binary = ROOT / "cuda/qimg21/test_cuda_qimg21_vae"
            if not binary.is_file():
                return None
            self.vae_socket_path = self.directory / "vae.sock"
            lines: list[str] = []
            command = [str(binary), "--serve", str(self.vae_socket_path), "--model", str(demo.model / "vae"),
                       "--conv", "cudnn", "--tf32"]
            self.vae_process = self._spawn(command, self.directory / "resident-vae.log",
                                           "qimg21-vae: serving", sink, lines)
            if start_log is not None:
                with start_log.open("a", encoding="utf-8") as handle:
                    handle.write("\n".join(lines) + "\n")
            return self.vae_socket_path if self.vae_process is not None else None

    def ensure(self, demo: "Demo", cfg: dict, sink=None, start_log: Path | None = None) -> Path | None:
        """The socket of a resident process for this run, starting one if
        needed. Startup output goes to `sink` and `start_log`, so the run that
        paid for the load still shows where the time went. None means run
        one-shot."""
        key = self.key_for(demo, cfg)
        with self.lock:
            if key is None:
                self._stop()
                return None
            if self.alive() and self.key == key:
                self.last_used = time.monotonic()
                return self.socket
            self._stop()
            self.directory.mkdir(parents=True, exist_ok=True)
            self.socket = self.directory / "fast.sock"
            _, _, model, preset, package, h, w, cfg_batch, attention = key
            command = [str(key[0]), "--serve", str(self.socket), "--preset", preset, "--model", model,
                       "--height-tokens", str(h), "--width-tokens", str(w),
                       "--serve-cfg", "1" if cfg_batch else "0"]
            if attention:
                command += ["--attention", attention]
            if package:
                command += ["--quant-package", package]
            lines: list[str] = []
            self.process = self._spawn(command, self.directory / "resident.log", "fast: serving ", sink, lines)
            if start_log is not None:
                start_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            if self.process is None:
                self._stop()
                return None
            self.key = key
            self.last_used = time.monotonic()
            return self.socket

    def reap_if_idle(self, idle: float = RESIDENT_IDLE) -> bool:
        with self.lock:
            if self.alive() and time.monotonic() - self.last_used > idle:
                self._stop()
                return True
            if self.process is not None and not self.alive():
                self._stop()
        return False

    def state(self) -> dict:
        with self.lock:
            if not self.alive() or self.key is None:
                return {"running": False}
            _, _, _, preset, _, h, w, cfg_batch, attention = self.key
            return {"running": True, "preset": preset, "attention": attention or "preset default",
                    "width": w * 16, "height": h * 16, "cfg": cfg_batch,
                    "vae": self.vae_alive(), "idle_s": round(time.monotonic() - self.last_used, 1)}


class Demo:
    def __init__(self, model: Path, quant: Path, python: Path, work: Path,
                 native: Path, host: str, port: int,
                 *, native_rocm: Path | None = None,
                 python_rocm: Path | None = None,
                 python_cpu: Path | None = None,
                 fast: Path = DEFAULT_FAST,
                 fast_packages: dict[str, Path] | None = None):
        self.model = model.resolve()
        self.quant = quant.resolve()
        # Preserve a venv/uv launcher symlink; Path.resolve() would collapse it
        # to the host interpreter and lose the environment's Torch packages.
        self.python = python.absolute()
        self.python_rocm = (python_rocm or python).absolute()
        # A CPU reference needs no PyTorch build of its own, but allow one to be
        # named so a box without a GPU-hosted reference environment still works.
        self.python_cpu = (python_cpu or python).absolute()
        self.work = work.resolve()
        self.native = native.resolve()
        self.native_rocm = (native_rocm or native).resolve()
        self.fast = fast.resolve()
        self.fast_packages = {kind: path.resolve()
                              for kind, path in (fast_packages or DEFAULT_FAST_PACKAGES).items()}
        self.host, self.port = host, port
        self.lock = threading.Lock()
        # job id -> {"progress": Progress, "events": [...], "started": t}
        self.jobs: dict[str, dict] = {}
        self.jobs_lock = threading.Lock()
        # interpreter path -> probed torch build; see probe_torch()
        self._probes: dict[str, dict] = {}
        self.resident = ResidentDenoiser(self.work.parent / "qimg21-resident")
        self.reference_server = ResidentReference(self.work.parent / "qimg21-resident")

    def fast_binary(self, backend="cuda") -> Path:
        return self.fast if backend == "cuda" else ROOT / "rdna4/qimg21/test_hip_qimg21_fast"

    def preset_available(self, preset: str, backend="cuda") -> bool:
        kind = FAST_PRESETS[preset]
        return (backend != "rocm" or preset != "low8-fp4") and self.fast_binary(backend).is_file() and (
            kind is None or (self.fast_packages.get(kind, Path("/nonexistent")) / "manifest.json").is_file())

    def reference_python(self, device: str) -> Path:
        """Which interpreter runs the PyTorch reference for a device.

        CUDA and ROCm need different PyTorch builds, so each gets its own
        environment. CPU runs on whichever build is already there: a CUDA build
        executes CPU kernels perfectly well, and asking for a third environment
        only adds a way to be wrong.
        """
        return {"rocm": self.python_rocm, "cpu": self.python_cpu}.get(device, self.python)

    def probe_torch(self, python: Path) -> dict:
        """Ask one interpreter what it is, once per process.

        Importing torch costs seconds, and the health endpoint needs the answer
        on every call, so the result is cached. The cache is keyed by
        interpreter rather than by device because the CPU and CUDA routes share
        an interpreter by default and would otherwise import torch twice for the
        same answer.
        """
        cached = self._probes.get(str(python))
        if cached is not None:
            return cached
        result = {"ok": False, "reason": "", "torch": None, "python": str(python)}
        if not python.is_file():
            result["reason"] = f"no interpreter at {python}"
            self._probes[str(python)] = result
            return result
        probe = ("import json, torch;"
                 "print(json.dumps({'torch': torch.__version__,"
                 "'hip': getattr(torch.version, 'hip', None),"
                 "'cuda': bool(torch.cuda.is_available())}))")
        try:
            done = subprocess.run([str(python), "-c", probe], capture_output=True,
                                  text=True, timeout=PROBE_TIMEOUT, env=dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core/lib:"+os.environ.get("LD_LIBRARY_PATH", "")))
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["reason"] = f"could not probe {python}: {exc}"
            self._probes[str(python)] = result
            return result
        if done.returncode:
            detail = (done.stderr or "").strip().splitlines()
            result["reason"] = (f"{python} has no usable torch: "
                                f"{detail[-1] if detail else 'import failed'}")
            self._probes[str(python)] = result
            return result
        try:
            build = json.loads(done.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            result["reason"] = f"{python} printed no torch version"
            self._probes[str(python)] = result
            return result
        result.update(ok=True, torch=build["torch"], hip=build["hip"], gpu=build["cuda"])
        self._probes[str(python)] = result
        return result

    def torch_build(self, device: str) -> dict:
        """Whether the reference can run on `device` here, and why not if not.

        A CUDA build answers torch.cuda.is_available() == True just like a ROCm
        one does, so availability is a property of the build and the requested
        device together, not of the machine alone.
        """
        python = self.reference_python(device)
        build = self.probe_torch(python)
        result = {"available": False, "reason": "", "torch": build["torch"],
                  "python": str(python)}
        if not build["ok"]:
            result["reason"] = build["reason"]
        elif device == "cpu":
            # Any build runs CPU kernels; that is the whole point of the route.
            result["available"] = True
        elif bool(build["hip"]) != (device == "rocm"):
            # A ROCm build reports a HIP version and a CUDA one does not, and
            # both answer torch.cuda.is_available() the same way, so this is the
            # only thing that tells the two builds apart.
            installed = "ROCm" if build["hip"] else "CUDA"
            wanted = "ROCm" if device == "rocm" else "CUDA"
            result["reason"] = (f"torch {build['torch']} is a {installed} build; {device} "
                                f"needs a {wanted} build")
        elif not build["gpu"]:
            result["reason"] = f"torch {build['torch']} sees no GPU"
        else:
            result["available"] = True
        return result

    def reference_devices(self) -> dict[str, dict]:
        return {device: self.torch_build(device) for device in REFERENCE_DEVICES}

    def native_components(self, backend: str) -> dict[str, Path]:
        """Return the complete native subprocess set for one backend."""
        root = ROOT / ("cuda/qimg21" if backend == "cuda" else "rdna4/qimg21")
        return {
            "transformer": self.native if backend == "cuda" else self.native_rocm,
            "text": root / ("test_cuda_qimg21_text" if backend == "cuda" else "test_hip_qimg21_text"),
            "vision": root / ("test_cuda_qimg21_vision" if backend == "cuda" else "test_hip_qimg21_vision"),
            "vae": root / ("test_cuda_qimg21_vae" if backend == "cuda" else "test_hip_qimg21_vae"),
            "vae_encode": root / ("test_cuda_qimg21_vae_encode" if backend == "cuda" else "test_hip_qimg21_vae_encode"),
            "attention": root / ("libq21_cutlass_attention.so" if backend == "cuda" else "libq21_hip_attention.so"),
        }

    @staticmethod
    def _optional_int(request: dict, key: str, low: int, high: int, what: str):
        """None passes through so the driver can pick; otherwise bound-checked."""
        value = request.get(key)
        if value is None or value == "":
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer") from None
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high} ({what})")
        return value

    def _validate(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        prompt = request.get("prompt", "").strip()
        if not prompt or len(prompt) > 4000:
            raise ValueError("prompt must contain 1--4000 characters")
        # A request that names a backend but omits mode is a native request.
        # Keep the legacy mode=cuda/rocm shorthand for older clients.
        mode = request.get("mode", "native" if "backend" in request else "cuda")
        backend = request.get("backend", "cuda")
        # Preserve the original API where mode=cuda meant native CUDA.  New
        # callers should use backend=cuda|rocm and mode=native|reference|compare.
        if mode in {"cuda", "rocm"}:
            backend = mode
            mode = "native"
        if backend not in {"cuda", "rocm"}:
            raise ValueError("backend must be cuda or rocm")
        if mode not in {"native", "reference", "compare"}:
            raise ValueError("mode must be native, reference, or compare")
        # Where the PyTorch reference runs. Defaulting to the native backend
        # keeps every existing request pointing where it always did: a CUDA
        # request gets the CUDA reference, an ROCm one the ROCm reference.
        reference_device = request.get("reference_device") or backend
        if reference_device not in REFERENCE_DEVICES:
            raise ValueError("reference_device must be one of " + ", ".join(REFERENCE_DEVICES))
        if mode in {"reference", "compare"}:
            # Refuse here with the reason rather than letting the run die on an
            # import error or a missing GPU several stages later.
            build = self.torch_build(reference_device)
            if not build["available"]:
                raise ValueError(f"the {reference_device} PyTorch reference is unavailable: "
                                 f"{build['reason']}")
        quantized = bool(request.get("quantized", False))
        preset = request.get("preset") or None
        if preset is not None:
            if preset not in FAST_PRESETS:
                raise ValueError("preset must be one of " + ", ".join(FAST_PRESETS))
            if backend == "rocm" and preset == "low8-fp4":
                raise ValueError("NVFP4 is unavailable on RDNA4; use low8 or fast12")
            if quantized:
                raise ValueError("a fast preset selects its own weights; leave quantized off")

        # Tiled coarse-to-fine refine: a base pass at 1/upscale, then a refine
        # that resamples it onto the requested size and denoises one tile at a
        # time. It needs the fast runner (the parity harness has no tile path)
        # and the PyTorch reference cannot do it, so a compare would be a
        # comparison of two different things.
        try:
            upscale = float(request.get("upscale") or 1.0)
        except (TypeError, ValueError):
            raise ValueError("upscale must be a number") from None
        if not 1.0 <= upscale <= 4.0:
            raise ValueError("upscale must be between 1 (no refine) and 4")
        tiled = upscale > 1.0
        if tiled:
            if backend not in ("cuda", "rocm") or preset is None:
                raise ValueError("a tiled refine needs a CUDA fast preset; pick one under "
                                 "'CUDA denoiser'")
            if mode != "native":
                raise ValueError("a tiled refine has no PyTorch reference to compare against; "
                                 "use native mode")
        width = int(request.get("width", 256)); height = int(request.get("height", 256))
        steps = int(request.get("steps", 2)); seed = int(request.get("seed", 42))
        if width < 256 or height < 256 or width % 32 or height % 32:
            raise ValueError("width and height must be at least 256 and divisible by 32")
        if tiled:
            for side in (height, width):
                base = max(32, round(side / upscale))
                if base % 32:
                    raise ValueError(f"upscale {upscale} leaves a {base} px base for a {side} px "
                                     "side, which is not a multiple of 32")
        fast_native = backend in ("cuda", "rocm") and preset is not None and mode == "native"
        limit = SIZE_LIMIT_TILED if tiled else SIZE_LIMIT_FAST if fast_native else SIZE_LIMIT_REFERENCE
        if width > limit or height > limit:
            what = "a tiled refine" if tiled else "a fast CUDA run" if fast_native else "this run"
            raise ValueError(f"width and height must be at most {limit} for {what}")
        if steps < 1 or steps > 40:
            raise ValueError("steps must be between 1 and 40")
        negative = request.get("negative_prompt", "")
        if not isinstance(negative, str) or len(negative) > 4000:
            raise ValueError("negative_prompt must be at most 4000 characters")

        smallest = min(height // 16, width // 16)
        tile_tokens = self._optional_int(request, "tile_tokens", 1, smallest,
                                         "a refine tile covers the whole output grid")
        tile_overlap = self._optional_int(request, "tile_overlap", 0, 1024, "latent tokens")
        vae_tile = self._optional_int(request, "vae_tile", 1, smallest, "a decode tile spans the image")
        vae_tile_overlap = self._optional_int(request, "vae_tile_overlap", 0, 1024, "latent tokens")
        vae_tile_bleed = self._optional_int(request, "vae_tile_bleed", 0, 1024, "latent tokens")
        if tile_overlap is None:
            tile_overlap = TILE_DEFAULTS["tile_overlap"]
        if tile_tokens is not None and tile_overlap >= tile_tokens:
            raise ValueError("tile_overlap must be smaller than tile_tokens")
        # The decode overlap and bleed only mean anything against a decode tile
        # size; reject them on their own rather than quietly dropping them, since
        # the driver would pick a different pair.
        if vae_tile is None and (vae_tile_overlap is not None or vae_tile_bleed is not None):
            raise ValueError("vae_tile_overlap and vae_tile_bleed need an explicit vae_tile; "
                             "leave vae_tile empty to let the driver choose all three")
        if vae_tile is not None:
            # Each kept pixel must be at least a bleed inside its tile, and a
            # neighbour's kept interior has to reach it, so the overlap must be
            # at least twice the bleed or the tiles leave gaps.
            if vae_tile_overlap is None:
                vae_tile_overlap = TILE_DEFAULTS["vae_tile_overlap"]
            if vae_tile_bleed is None:
                vae_tile_bleed = TILE_DEFAULTS["vae_tile_bleed"]
            if vae_tile_bleed > vae_tile_overlap // 2:
                raise ValueError("vae_tile_bleed must be at most half of vae_tile_overlap")
        try:
            refine_strength = float(request.get("refine_strength", 0.5))
        except (TypeError, ValueError):
            raise ValueError("refine_strength must be a number") from None
        if not 0.0 < refine_strength <= 1.0:
            raise ValueError("refine_strength must be greater than 0 and at most 1")
        base_steps = self._optional_int(request, "base_steps", 1, 100, "steps per pass")
        refine_seed = self._optional_int(request, "refine_seed", 0, 2 ** 63, "a seed")
        if refine_seed is None:
            refine_seed = 0
        # Refine an earlier native result with a longer schedule: its final
        # latents are renoised to step K = keep * steps with the noise that run
        # started from, and only steps K..N run. keep 0 is a fresh run.
        # How the CUDA PyTorch reference holds its weights: a loaded server with
        # as many transformer blocks on the device as fit (fast, the fair
        # comparison), or the original one-shot run that streams every module
        # at each use (least device memory).
        # Attention kernel for a fast preset. Its default (Sage: INT8 Q.K, FP8
        # P.V for low8/fast12/low8-fp4) is the fastest; "exact" is PyTorch's
        # memory-efficient kernel, which halves INT8's distance to the
        # reference trajectory for about 17% more denoise time.
        attention = request.get("attention") or None
        if attention is not None:
            if attention not in FAST_ATTENTION:
                raise ValueError("attention must be one of " + ", ".join(FAST_ATTENTION))
            if preset is None:
                raise ValueError("attention is a fast-preset option")
        reference_offload = request.get("reference_offload") or "resident"
        if reference_offload not in {"resident", "sequential"}:
            raise ValueError("reference_offload must be resident or sequential")
        restart_from = request.get("restart_from") or None
        restart_keep = 0.0
        if restart_from is not None:
            if not isinstance(restart_from, str) or not re.fullmatch(r"[0-9a-f]{32}", restart_from):
                raise ValueError("restart_from must be a job id from an earlier result")
            if backend not in ("cuda", "rocm") or preset is None or mode != "native" or upscale > 1.0:
                raise ValueError("refining with more steps needs a CUDA fast preset in native mode, "
                                 "without the tiled refine")
            try:
                restart_keep = float(request.get("restart_keep", 0.5))
            except (TypeError, ValueError):
                raise ValueError("restart_keep must be a number") from None
            if not 0.0 <= restart_keep <= 0.95:
                raise ValueError("restart_keep must be between 0 and 0.95")
            source = self.work / restart_from
            work = source / "cuda-work"
            if not (work / "native_latents.npy").is_file() or not (work / "latents.npy").is_file():
                raise ValueError("that result is gone or was not a native CUDA run; generate it again")
            try:
                earlier = json.loads((source / "request.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                earlier = {}
            if (earlier.get("width"), earlier.get("height")) != (width, height):
                raise ValueError("refining keeps the picture's size: use the width and height it was made at")
        return {"prompt": prompt, "negative_prompt": negative.strip(),
                "mode": mode, "backend": backend, "reference_device": reference_device,
                "width": width, "height": height, "steps": steps, "seed": seed,
                "quantized": quantized, "preset": preset,
                # Per-step timings come from the fast runner's own CUDA events,
                # so this is a hint it can only honour with a preset selected.
                "profile_steps": bool(request.get("profile_steps")) and preset is not None,
                "upscale": upscale, "base_steps": base_steps, "tile_tokens": tile_tokens,
                "tile_overlap": tile_overlap, "refine_strength": refine_strength,
                "refine_seed": refine_seed, "vae_tile": vae_tile,
                "vae_tile_overlap": vae_tile_overlap, "vae_tile_bleed": vae_tile_bleed,
                "restart_from": restart_from, "restart_keep": restart_keep,
                "reference_offload": reference_offload, "attention": attention}


    def _run(self, command: list[str], cwd: Path, log: Path,
             env: dict[str, str] | None = None, progress=None) -> None:
        """Run one child, streaming its output lines to `progress` as it goes.

        The child already writes everything to `log`, so progress is read back
        from that file rather than through a pipe: the driver and the denoiser
        interleave their own stderr with the child's, and a second channel would
        lose the ordering that makes the log readable. A tail thread advances a
        byte offset, so a partial line never surfaces twice.
        """
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            env = dict(os.environ) if env is None else dict(env)
            env["LD_LIBRARY_PATH"] = "/opt/rocm/core/lib:" + env.get("LD_LIBRARY_PATH", "")
            env["TMPDIR"] = str(ROOT / "tmp/qimg21-runtime")
            Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, env=env)
            if progress:
                stop = threading.Event()
                reader = threading.Thread(target=_tail, args=(log, progress, stop), daemon=True)
                reader.start()
                try:
                    code = process.wait()
                finally:
                    stop.set()
                    reader.join(timeout=5.0)
            else:
                code = process.wait()
        if code:
            tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"inference exited with {code}: {tail}")

    def _native(self, cfg: dict, out: Path, progress=None,
                initial_latents: Path | None = None) -> tuple[Path, Path]:
        backend = cfg["backend"]
        python = self.python if backend == "cuda" else self.python_rocm
        if backend == "rocm" and not python.is_file():
            # The native ROCm path only needs NumPy and Pillow. A ROCm Torch
            # environment is required for reference mode, but not generation.
            python = Path(sys.executable)
        native = self.native if backend == "cuda" else self.native_rocm
        image = out / f"{backend}.png"
        work = out / f"{backend}-work"
        attention = "cutlass-efficient" if backend == "cuda" else "wmma-fused"
        native_vae = (backend == "cuda" or
                      (ROOT / "rdna4/qimg21/test_hip_qimg21_vae").is_file())
        command = [str(python), "cuda/qimg21/native_generate.py", "--backend", backend,
                   "--model", str(self.model),
                   "--prompt", cfg["prompt"], "--height", str(cfg["height"]),
                   "--width", str(cfg["width"]), "--steps", str(cfg["steps"]),
                   "--seed", str(cfg["seed"]), "--dtype", "bf16",
                   "--native-bin", str(native),
                   "--native-attention", attention, "--native-normalization", "vector4",
                   "--native-rope", "host-table-exact", "--work-dir", str(work), "--out", str(image)]
        # A repeated prompt reuses its text embedding instead of streaming the
        # 14 GB text encoder again; the key covers the prompt, the checkpoint
        # and the encoder build.
        command += ["--prompt-cache", str(self.work.parent / "qimg21-prompt-cache")]
        preset = cfg.get("preset")
        if preset:
            # The fast runner's preset sets budget, weights and attention.
            kind = FAST_PRESETS[preset]
            if not self.preset_available(preset, backend):
                raise RuntimeError(f"preset {preset} is unavailable: build `make -C cuda/qimg21 fast`"
                                   + (f" and provide the {kind} package" if kind else ""))
            at = command.index("--native-bin")
            command[at:at + 8] = ["--native-bin", str(self.fast_binary(backend)), "--runner", "fast", "--preset", preset]
            if cfg.get("attention"):
                command += ["--native-attention", FAST_ATTENTION[cfg["attention"]]]
            if kind:
                command += ["--quant-package", str(self.fast_packages[kind])]
            if cfg["profile_steps"]:
                command += ["--profile-steps"]
        if native_vae:
            command.insert(command.index("--native-bin"), "--native-vae")
        if initial_latents is not None:
            # A compare starts the runner from the noise the reference drew, so
            # the two pictures differ only by what each implementation computes,
            # not by how each one happens to turn a seed into noise.
            command += ["--initial-latents", str(initial_latents)]
        if cfg["negative_prompt"]:
            command += ["--negative-prompt", cfg["negative_prompt"], "--true-cfg-scale", "4.0"]
        if cfg["quantized"]:
            if not self.quant.is_dir():
                raise RuntimeError(f"quantized package is unavailable: {self.quant}")
            command += ["--quantized-transformer", str(self.quant)]
            if backend == "cuda":
                command += ["--int8-tensor-core", "--int8-bf16-tail-blocks", "16"]
        # Coarse-to-fine: a base pass at 1/upscale, then a tiled refine up to
        # the requested size. --tile-tokens and --vae-tile are left off when the
        # client did not pin them, so the driver picks the largest tile the
        # preset's budget allows rather than the demo second-guessing it.
        if cfg["upscale"] > 1.0:
            command += ["--upscale", str(cfg["upscale"])]
            if cfg["base_steps"] is not None:
                command += ["--base-steps", str(cfg["base_steps"])]
            if cfg["tile_tokens"] is not None:
                command += ["--tile-tokens", str(cfg["tile_tokens"])]
            command += ["--tile-overlap", str(cfg["tile_overlap"]),
                        "--refine-strength", str(cfg["refine_strength"]),
                        "--refine-seed", str(cfg["refine_seed"])]
        if cfg["vae_tile"]:
            command += ["--vae-tile", str(cfg["vae_tile"]),
                        "--vae-tile-overlap", str(cfg["vae_tile_overlap"]),
                        "--vae-tile-bleed", str(cfg["vae_tile_bleed"])]
        # uv-managed reference environments can expose the host interpreter as
        # sys.executable from a child process; carry the selected interpreter
        # explicitly to the fixture helper so it retains Torch/CUDA imports.
        socket_path = self.resident.ensure(self, cfg, progress, out / "resident-start.log")
        if socket_path is not None:
            command += ["--resident-socket", str(socket_path)]
            vae_socket = self.resident.vae_socket(self, cfg, progress, out / "resident-start.log")
            if vae_socket is not None:
                command += ["--resident-vae-socket", str(vae_socket)]
        if preset:
            # TF32 VAE convolutions: 22% faster, and the decoded image stays
            # at 52.7 dB against the BF16 reference VAE, as in F32.
            command += ["--vae-tf32"]
        if cfg.get("restart_from"):
            source = self.work / cfg["restart_from"] / "cuda-work"
            command += ["--initial-latents", str(source / "latents.npy"),
                        "--restart-from", str(source / "native_latents.npy"),
                        "--restart-step", str(self.restart_step(cfg))]
        env = os.environ.copy()
        env["QIMG21_PYTHON"] = str(python)
        log = out / f"{backend}.log"
        self._run(command, ROOT, log, env=env, progress=progress)
        return image, log

    @staticmethod
    def restart_step(cfg: dict) -> int:
        """The step a refine restarts at: `restart_keep` of the schedule is
        taken as done, so at least one step always runs."""
        return min(cfg["steps"] - 1, max(0, round(cfg.get("restart_keep", 0.0) * cfg["steps"])))

    @staticmethod
    def _summary(log: Path) -> list[str]:
        """The lines a user actually wants from a run: the memory plan, the tile
        geometry the driver chose, and the per-stage timings. A demo that only
        shows the picture hides exactly the numbers that explain it."""
        wanted = ("fast: plan ", "fast: tiled refine ", "fast: prefill ",
                  "fast: weights ready", "qimg21-vae: ", "base pass:", "refine tile:",
                  "refine pass:")
        if not log.is_file():
            return []
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return [line for line in lines if line.startswith(wanted)][-12:]

    def _reference(self, cfg: dict, out: Path, progress=None) -> Path:
        device = cfg["reference_device"]
        backend = cfg["backend"]
        python = self.reference_python(device)
        # The reference is named by the device that produced it: a CUDA and a CPU
        # picture of the same prompt are different claims, and a compare that
        # showed one under the other's label would be misleading. One directory
        # per device, because the driver writes reference.png and run.json by
        # fixed name and a second run would overwrite the first one's record.
        dump = out / "reference" / device
        command = [str(python), "cuda/qimg21/reference.py", "--model", str(self.model),
                   "--device", device,
                   "--prompt", cfg["prompt"], "--height", str(cfg["height"]),
                   "--width", str(cfg["width"]), "--steps", str(cfg["steps"]),
                   "--seed", str(cfg["seed"]), "--dtype", "bf16", "--sdpa-backend", "efficient",
                   "--dump-dir", str(dump)]
        # The initial noise is cheap to save and does not change the picture.
        # A compare starts the runner from it, and the step-0 preview needs it.
        command += ["--dump-initial-latents"]
        if cfg["negative_prompt"]:
            command += ["--negative-prompt", cfg["negative_prompt"], "--true-cfg-scale", "4.0"]
        log = out / f"reference-{backend}-{device}.log"
        if device == "cuda" and cfg.get("reference_offload", "resident") == "resident":
            socket_path = self.reference_server.ensure(python, self.model, progress, out / "reference-start.log")
            if socket_path is not None:
                status = send_resident(socket_path, command[2:] + ["--offload", "resident"], log, progress)
                if status == 0:
                    return dump / "reference.png"
        else:
            # The one-shot run streams its own weights; the server's idle
            # context is the only thing to give back.
            self.reference_server.stop()
        self._run(command, ROOT, log, progress=progress)
        # The driver names its own output; do not invent a second name for it.
        return dump / "reference.png"

    @staticmethod
    def _data_url(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    def generate(self, request: dict, job_id: str | None = None) -> dict:
        cfg = self._validate(request)
        job = self.work / uuid.uuid4().hex
        job.mkdir(parents=True, exist_ok=False)
        # What was asked for, so a later refine can check it continues the
        # same picture.
        (job / "request.json").write_text(json.dumps(cfg), encoding="utf-8")
        started = time.monotonic()
        results: dict = {"request": cfg, "job": job.name}

        # A client that supplies an id can attach to the run; one that does not
        # just waits for the response, so progress is strictly additive.
        if job_id:
            self._begin(job_id, started)
        try:
            with self.lock:
                if ResidentDenoiser.key_for(self, cfg) is None:
                    # Everything else needs the VRAM a resident denoiser holds.
                    self.resident.stop()
                compare = cfg["mode"] == "compare"
                backend = cfg["backend"]
                ref_path = native_path = None
                # A compare runs the reference first so the native runner can
                # start from the very noise the reference drew: on any reference
                # device, the two pictures then share every input.
                if cfg["mode"] in {"reference", "compare"}:
                    device = cfg["reference_device"]
                    self._say(job_id, f"Qwen PyTorch reference ({device})")
                    dump = job / "reference" / device
                    side_start = time.monotonic()
                    with self._previews(job_id, "reference", dump, dump / "initial_latents.npy", cfg):
                        ref_path = self._reference(cfg, job, self._progress(job_id, "PyTorch reference"))
                    ref_ms = round((time.monotonic() - side_start) * 1000)
                    self._say(job_id, f"Qwen PyTorch reference ({device}) complete")
                    breakdown = (timing_breakdown(job / "reference-start.log", "load resident PyTorch reference")
                                 + timing_breakdown(job / f"reference-{backend}-{device}.log",
                                                    "PyTorch reference"))
                    entry = {"image": self._data_url(ref_path), "device": device,
                             "torch": self.torch_build(device)["torch"],
                             "elapsed_ms": ref_ms, "timings": breakdown,
                             "generation_s": generation_seconds(breakdown)}
                    if device not in REFERENCE_PARITY_DEVICES and compare:
                        # A CPU reference runs different kernels on different
                        # hardware, so a side-by-side with the native runner is a
                        # look at both pictures, not a parity result. Say so on the
                        # result instead of letting the pairing imply agreement.
                        entry["note"] = (f"the {device} reference runs different kernels on "
                                         "different hardware: compare this for composition, "
                                         "not for numerical agreement")
                    results["reference"] = entry
                if cfg["mode"] in {"native", "compare"}:
                    latents = ref_path.parent / "initial_latents.npy" if compare else None
                    if latents is not None and not latents.is_file():
                        latents = None
                    self._say(job_id, f"Qwen {backend.upper()} native")
                    work = job / f"{backend}-work"
                    side_start = time.monotonic()
                    with self._previews(job_id, "native", work / "steps", work / "latents.npy", cfg):
                        native_path, log = self._native(cfg, job, self._progress(job_id, "encode prompt"),
                                                        initial_latents=latents)
                    native_ms = round((time.monotonic() - side_start) * 1000)
                    self._say(job_id, f"Qwen {backend.upper()} native complete")
                    breakdown = (timing_breakdown(job / "resident-start.log", "load resident denoiser")
                                 + timing_breakdown(log, "encode prompt"))
                    results[backend] = {"image": self._data_url(native_path),
                                        "log": self._summary(log), "elapsed_ms": native_ms,
                                        "timings": breakdown,
                                        "generation_s": generation_seconds(breakdown)}
                    if cfg.get("restart_from"):
                        results[backend]["restart"] = {"from": cfg["restart_from"],
                                                       "step": self.restart_step(cfg),
                                                       "steps": cfg["steps"]}
                    if compare:
                        results["reference"]["matched_noise"] = latents is not None
                if compare:
                    try:
                        results["compare"] = compare_runs(ref_path.parent,
                                                          job / f"{backend}-work", native_path)
                    except Exception as exc:  # noqa: BLE001 - metrics never hide the images
                        results["compare"] = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._end(job_id)
        results["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return results

    def _previews(self, job_id: str | None, source: str, step_dir: Path, initial: Path,
                  cfg: dict) -> StepPreviews | contextlib.nullcontext:
        """Previews for one run, only when a client is attached to watch them."""
        if not job_id:
            return contextlib.nullcontext()
        h, w = cfg["height"] // 16, cfg["width"] // 16
        grids = [(h, w, cfg["steps"])]
        if cfg.get("upscale", 1.0) > 1.0:
            base_h = max(32, round(cfg["height"] / cfg["upscale"])) // 16
            base_w = max(32, round(cfg["width"] / cfg["upscale"])) // 16
            grids.insert(0, (base_h, base_w, cfg.get("base_steps") or cfg["steps"]))
        return StepPreviews(source, step_dir, initial, grids,
                            lambda event: self._record(job_id, event))

    # ---- progress registry ----

    def _begin(self, job_id: str, started: float) -> None:
        with self.jobs_lock:
            now = time.monotonic()
            for stale in [key for key, value in self.jobs.items() if now - value["started"] > PROGRESS_TTL]:
                del self.jobs[stale]
            self.jobs[job_id] = {"progress": Progress(started), "events": [],
                                 "base": 0, "started": now}

    def _end(self, job_id: str | None) -> None:
        if not job_id:
            return
        with self.jobs_lock:
            record = self.jobs.get(job_id)
            if record:
                record["progress"].finish()

    def _say(self, job_id: str | None, message: str) -> None:
        """A message with no log line behind it, for the driver's own phases."""
        if job_id:
            self._record(job_id, {"kind": "message", "text": message})

    def _record(self, job_id: str, event: dict) -> None:
        with self.jobs_lock:
            record = self.jobs.get(job_id)
            if not record:
                return
            record["events"].append(event)
            if len(record["events"]) > MAX_EVENTS:
                # Drop from the front, but count what went: the cursor is
                # absolute, so a client that has not polled yet still resumes
                # from the right place instead of re-reading shifted events.
                drop = len(record["events"]) - MAX_EVENTS
                del record["events"][:drop]
                record["base"] += drop

    def _progress(self, job_id: str, stage: str | None = None):
        """A line sink bound to one job, carrying the wall clock with it.
        `stage` names what runs until the log's first command line does: the
        driver's prompt cache check, or the whole PyTorch reference."""
        if stage:
            with self.jobs_lock:
                record = self.jobs.get(job_id)
                if record:
                    record["progress"].stage = stage

        def sink(line: str) -> None:
            with self.jobs_lock:
                record = self.jobs.get(job_id)
                tracker = record["progress"] if record else None
            if tracker is None:
                return
            event = tracker.feed(line, time.monotonic())
            if event:
                self._record(job_id, event)
        return sink

    def progress_state(self, job_id: str, since: int = 0) -> dict | None:
        """The events after `since`, plus the totals a client needs to render
        a header without keeping the whole history."""
        with self.jobs_lock:
            record = self.jobs.get(job_id)
            if not record:
                return None
            tracker = record["progress"]
            # `base` events were trimmed before the client got to them, so a
            # cursor older than that resumes at the oldest one still held.
            events = record["events"][max(0, since - record["base"]):]
            cursor = record["base"] + len(record["events"])
            snapshot = {"cursor": cursor, "stage": tracker.stage, "done": tracker.done,
                        "accum_ms": tracker.accum_ms, "total_steps": tracker.total_steps,
                        "elapsed_ms": (time.monotonic() - record["started"]) * 1000.0,
                        "notes": list(tracker.notes), "events": events}
        return snapshot

class Handler(BaseHTTPRequestHandler):
    server_version = "qwen-image21-demo/1.0"

    def _json(self, status: int, value: dict) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _progress(self, parsed) -> None:
        """Answer a poll. Bad input is a 400, not a dropped connection: a
        polling client that gets nothing back cannot tell a typo from a dead
        server, and would keep retrying a request that can never work."""
        demo: Demo = self.server.demo  # type: ignore[attr-defined]
        query = parse_qs(parsed.query)
        job = (query.get("job") or [""])[0]
        if not job or len(job) > 64 or not all(c.isalnum() or c in "-_" for c in job):
            return self._json(400, {"ok": False,
                                    "error": "job must be 1--64 characters of [A-Za-z0-9_-]"})
        try:
            since = int((query.get("since") or ["0"])[0])
        except ValueError:
            return self._json(400, {"ok": False, "error": "since must be an integer"})
        state = demo.progress_state(job, max(0, since))
        self._json(200 if state else 404,
                   state if state else {"ok": False, "error": "unknown or expired job"})

    def do_GET(self) -> None:
        demo: Demo = self.server.demo  # type: ignore[attr-defined]
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/progress":
            return self._progress(parsed)
        if path == "/api/health":
            builds = demo.reference_devices()
            components = {backend: {name: item.is_file()
                                    for name, item in demo.native_components(backend).items()}
                          for backend in ("cuda", "rocm")}
            self._json(200, {"ok": True, "model": str(demo.model),
                             "quantized_available": demo.quant.is_dir(),
                             "presets": {name: demo.preset_available(name) for name in FAST_PRESETS},
                             "presets_by_backend": {backend: {name: demo.preset_available(name, backend) for name in FAST_PRESETS} for backend in ("cuda", "rocm")},
                             "size_limit": {"reference": SIZE_LIMIT_REFERENCE,
                                             "fast": SIZE_LIMIT_FAST, "tiled": SIZE_LIMIT_TILED},
                             "native": {backend: all(components[backend].values())
                                        for backend in ("cuda", "rocm")},
                             "native_components": components,
                             # Probed once: reference_devices() caches per interpreter,
                             # but building it twice here would read as two probes.
                             "reference": {device: build["available"] for device, build
                                           in builds.items()},
                             "reference_detail": builds,
                             "resident": demo.resident.state(),
                             "reference_server": demo.reference_server.state()})
            return
        if path in {"/", "/index.html"}:
            data = (WEB / "qwen_image21.html").read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/generate":
            self._json(404, {"ok": False, "error": "not found"}); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY: raise ValueError("request body is too large")
            request = json.loads(self.rfile.read(length))
            job_id = request.get("job") if isinstance(request, dict) else None
            if job_id is not None and (not isinstance(job_id, str) or not job_id
                                       or len(job_id) > 64
                                       or not all(c.isalnum() or c in "-_" for c in job_id)):
                raise ValueError("job must be 1--64 characters of [A-Za-z0-9_-]")
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from server.vhuman.gpu import file_lock
            backend = request.get("backend", "cuda")
            with file_lock(ROOT / "tmp/pixal3d/device-locks" / f"{backend}-0.lock", 900):
                result = self.server.demo.generate(request, job_id)  # type: ignore[attr-defined]
            self._json(200, {"ok": True, **result})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:  # inference errors are shown in the UI
            self._json(500, {"ok": False, "error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[qwen-image21] {self.address_string()} {fmt % args}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--quant-package", type=Path, default=DEFAULT_QUANT)
    ap.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    ap.add_argument("--native", type=Path, default=ROOT / "cuda/qimg21/test_cuda_qimg21_native")
    ap.add_argument("--native-rocm", type=Path,
                    default=ROOT / "rdna4/qimg21/test_hip_qimg21_native")
    ap.add_argument("--python-rocm", type=Path,
                    default=ROOT / "tmp/vhuman-rocm-venv/bin/python")
    ap.add_argument("--python-cpu", type=Path, default=None,
                    help="interpreter for the CPU PyTorch reference; defaults to --python, "
                         "since a CUDA build also runs CPU kernels")
    ap.add_argument("--fast", type=Path, default=DEFAULT_FAST, help="fast CUDA denoiser for presets")
    ap.add_argument("--int8-package", type=Path, default=DEFAULT_FAST_PACKAGES["int8"],
                    help="pack_fast.py int8-smooth package (low8, fast12)")
    ap.add_argument("--nvfp4-package", type=Path, default=DEFAULT_FAST_PACKAGES["nvfp4"],
                    help="pack_fast.py nvfp4-svd package (low8-fp4)")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "tmp/qimg21-web-jobs")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()
    if not args.model.is_dir(): ap.error(f"model directory not found: {args.model}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    demo = Demo(args.model, args.quant_package, args.python, args.work_dir, args.native,
                args.host, args.port, native_rocm=args.native_rocm,
                python_rocm=args.python_rocm, python_cpu=args.python_cpu, fast=args.fast,
                fast_packages={"int8": args.int8_package, "nvfp4": args.nvfp4_package})
    server = ThreadingHTTPServer((args.host, args.port), Handler); server.demo = demo  # type: ignore[attr-defined]
    print(f"Qwen Image 2.1 demo: http://{args.host}:{args.port}")

    def reap() -> None:
        # Never while a run holds the device: the run is what keeps it warm.
        while True:
            time.sleep(30)
            if demo.lock.acquire(blocking=False):
                try:
                    if demo.resident.reap_if_idle():
                        print("resident denoiser idle; stopped to free VRAM", flush=True)
                    if demo.reference_server.reap_if_idle():
                        print("resident PyTorch reference idle; stopped", flush=True)
                finally:
                    demo.lock.release()
    threading.Thread(target=reap, daemon=True).start()
    # A plain kill must still stop the resident denoiser (the finally below).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        demo.resident.stop()
        demo.reference_server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
