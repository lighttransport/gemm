#!/usr/bin/env python3
"""Standalone Qwen-Image 2.1 web demo server.

The server intentionally shells out to the checked-in native and reference
drivers.  This keeps the web process small and prevents two model copies from
being resident in the 12--16 GB CUDA device at once.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
DEFAULT_MODEL = Path("/mnt/nvme01/models/qimg-21")
DEFAULT_QUANT = ROOT / "tmp/qimg21-int8-package"
DEFAULT_PYTHON = ROOT / "tmp/qimg21-ref-venv/bin/python"
DEFAULT_FAST = ROOT / "cuda/qimg21/test_cuda_qimg21_fast"
DEFAULT_FAST_PACKAGES = {"int8": Path("/mnt/nvme01/models/qimg-21-fast/int8-smooth-a0.6"),
                         "nvfp4": Path("/mnt/nvme01/models/qimg-21-fast/nvfp4-svd-a0.5-m")}
# Fast CUDA denoiser presets (test_cuda_qimg21_fast --preset) and the
# pack_fast.py weight package each needs.
FAST_PRESETS = {"low8": "int8", "low8-fp4": "nvfp4", "fast12": "int8", "accurate": None}
# Output size limits per pixel side. The fast runner bounds its own memory, so
# the caps are about what the other stages can afford: the parity harness and the
# PyTorch reference stay at 1024, an untiled fast run goes to 2048 (the VAE
# decode tiles itself above 1024), and a tiled coarse-to-fine refine reaches 4096.
SIZE_LIMIT_REFERENCE = 1024
SIZE_LIMIT_FAST = 2048
SIZE_LIMIT_TILED = 4096
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
MAX_EVENTS = 4096

# Which pipeline stage a subprocess is, from the driver's "+ <command>" line.
# The order matters: the text encoder is also invoked for the condition image.
STAGE_PATTERNS = (
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


class Demo:
    def __init__(self, model: Path, quant: Path, python: Path, work: Path,
                 native: Path, host: str, port: int,
                 *, native_rocm: Path | None = None,
                 python_rocm: Path | None = None,
                 fast: Path = DEFAULT_FAST,
                 fast_packages: dict[str, Path] | None = None):
        self.model = model.resolve()
        self.quant = quant.resolve()
        # Preserve a venv/uv launcher symlink; Path.resolve() would collapse it
        # to the host interpreter and lose the environment's Torch packages.
        self.python = python.absolute()
        self.python_rocm = (python_rocm or python).absolute()
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

    def preset_available(self, preset: str) -> bool:
        kind = FAST_PRESETS[preset]
        return self.fast.is_file() and (
            kind is None or (self.fast_packages.get(kind, Path("/nonexistent")) / "manifest.json").is_file())

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
        quantized = bool(request.get("quantized", False))
        preset = request.get("preset") or None
        if preset is not None:
            if preset not in FAST_PRESETS:
                raise ValueError("preset must be one of " + ", ".join(FAST_PRESETS))
            if backend != "cuda":
                raise ValueError("fast presets are CUDA only")
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
            if backend != "cuda" or preset is None:
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
        fast_native = backend == "cuda" and preset is not None and mode == "native"
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
        return {"prompt": prompt, "negative_prompt": negative.strip(),
                "mode": mode, "backend": backend,
                "width": width, "height": height, "steps": steps, "seed": seed,
                "quantized": quantized, "preset": preset,
                # Per-step timings come from the fast runner's own CUDA events,
                # so this is a hint it can only honour with a preset selected.
                "profile_steps": bool(request.get("profile_steps")) and preset is not None,
                "upscale": upscale, "base_steps": base_steps, "tile_tokens": tile_tokens,
                "tile_overlap": tile_overlap, "refine_strength": refine_strength,
                "refine_seed": refine_seed, "vae_tile": vae_tile,
                "vae_tile_overlap": vae_tile_overlap, "vae_tile_bleed": vae_tile_bleed}


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

    def _native(self, cfg: dict, out: Path, progress=None) -> tuple[Path, Path]:
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
        preset = cfg.get("preset")
        if preset:
            # The fast runner's preset sets budget, weights and attention.
            kind = FAST_PRESETS[preset]
            if not self.preset_available(preset):
                raise RuntimeError(f"preset {preset} is unavailable: build `make -C cuda/qimg21 fast`"
                                   + (f" and provide the {kind} package" if kind else ""))
            at = command.index("--native-bin")
            command[at:at + 8] = ["--native-bin", str(self.fast), "--runner", "fast", "--preset", preset]
            if kind:
                command += ["--quant-package", str(self.fast_packages[kind])]
            if cfg["profile_steps"]:
                command += ["--profile-steps"]
        if native_vae:
            command.insert(command.index("--native-bin"), "--native-vae")
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
        env = os.environ.copy()
        env["QIMG21_PYTHON"] = str(python)
        log = out / f"{backend}.log"
        self._run(command, ROOT, log, env=env, progress=progress)
        return image, log

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
        backend = cfg["backend"]
        python = self.python if backend == "cuda" else self.python_rocm
        image = out / "reference" / f"{backend}.png"
        command = [str(python), "cuda/qimg21/reference.py", "--model", str(self.model),
                   "--prompt", cfg["prompt"], "--height", str(cfg["height"]),
                   "--width", str(cfg["width"]), "--steps", str(cfg["steps"]),
                   "--seed", str(cfg["seed"]), "--dtype", "bf16", "--sdpa-backend", "efficient",
                   "--dump-dir", str(image.parent)]
        if cfg["negative_prompt"]:
            command += ["--negative-prompt", cfg["negative_prompt"], "--true-cfg-scale", "4.0"]
        self._run(command, ROOT, out / f"reference-{backend}.log", progress=progress)
        return image

    @staticmethod
    def _data_url(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    def generate(self, request: dict, job_id: str | None = None) -> dict:
        cfg = self._validate(request)
        job = self.work / uuid.uuid4().hex
        job.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        results: dict = {"request": cfg, "job": job.name}

        # A client that supplies an id can attach to the run; one that does not
        # just waits for the response, so progress is strictly additive.
        if job_id:
            self._begin(job_id, started)
        try:
            with self.lock:
                if cfg["mode"] in {"native", "compare"}:
                    backend = cfg["backend"]
                    self._say(job_id, f"Qwen {backend.upper()} native")
                    path, log = self._native(cfg, job, self._progress(job_id))
                    self._say(job_id, f"Qwen {backend.upper()} native complete")
                    results[backend] = {"image": self._data_url(path), "log": self._summary(log)}
                if cfg["mode"] in {"reference", "compare"}:
                    self._say(job_id, "Qwen PyTorch reference")
                    path = self._reference(cfg, job, self._progress(job_id))
                    self._say(job_id, "Qwen PyTorch reference complete")
                    results["reference"] = {"image": self._data_url(path)}
        finally:
            self._end(job_id)
        results["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return results

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

    def _progress(self, job_id: str):
        """A line sink bound to one job, carrying the wall clock with it."""
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
            components = {backend: {name: item.is_file()
                                    for name, item in demo.native_components(backend).items()}
                          for backend in ("cuda", "rocm")}
            self._json(200, {"ok": True, "model": str(demo.model),
                             "quantized_available": demo.quant.is_dir(),
                             "presets": {name: demo.preset_available(name) for name in FAST_PRESETS},
                             "size_limit": {"reference": SIZE_LIMIT_REFERENCE,
                                             "fast": SIZE_LIMIT_FAST, "tiled": SIZE_LIMIT_TILED},
                             "native": {backend: all(components[backend].values())
                                        for backend in ("cuda", "rocm")},
                             "native_components": components,
                             "reference": {"cuda": demo.python.is_file(),
                                           "rocm": demo.python_rocm.is_file()}})
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
                    default=ROOT / "tmp/qimg21-rocm-venv/bin/python")
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
                python_rocm=args.python_rocm, fast=args.fast,
                fast_packages={"int8": args.int8_package, "nvfp4": args.nvfp4_package})
    server = ThreadingHTTPServer((args.host, args.port), Handler); server.demo = demo  # type: ignore[attr-defined]
    print(f"Qwen Image 2.1 demo: http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
