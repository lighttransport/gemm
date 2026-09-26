"""NativeBackend: GenRequest -> cuda/qimg21/native_generate.py.

One condition image (the native text encoder and editing layout take one
today), SDEdit through --init-image/--strength, masks through --mask (latent
blending in the fast runner plus pixel paste-back). The model is not held by
this process; what makes repeated views cheap is:
- --condition-cache / --prompt-cache: the reference image's VAE and vision
  encodes, multimodal prompt embeddings and seed noise are computed once and
  reused (keyed by content);
- resident processes: with resident=True (the default) this backend starts
  its own test_cuda_qimg21_fast --serve, sized for the view's grid, CFG and
  condition tokens, and a resident VAE decoder, and keeps them for every
  later view of that setup; explicit sockets (the demo server's) take
  precedence. A request a resident process cannot take runs one-shot.
  The one-shot encoders need the device memory the resident processes hold,
  so a request that has to run one -- a reference not seen yet (its encodes
  are then cached), or an init image (encoded every time) -- stops them and
  runs one-shot.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from .backends import BackendError, GenRequest, GenResult

HERE = Path(__file__).resolve().parents[1]          # cuda/qimg21
ROOT = HERE.parents[1]                              # repo root
DEFAULT_MODEL = Path("/mnt/nvme01/models/qimg-21")
FAST = HERE / "test_cuda_qimg21_fast"
PACKAGES = {"int8": "/mnt/nvme01/models/qimg-21-fast/int8-smooth-a0.6",
            "nvfp4": "/mnt/nvme01/models/qimg-21-fast/nvfp4-svd-a0.5-m"}
PRESET_WEIGHTS = {"low8": "int8", "low8-fp4": "nvfp4", "fast12": "int8", "accurate": None}
ATTENTION = {"sage": "sage", "flash": "flash", "exact": "cutlass-efficient"}
VAE = HERE / "test_cuda_qimg21_vae"
RESIDENT_START_TIMEOUT = 600.0
# The resident VAE decoder holds about 3 GB after its first decode; past 1024^2
# the decode tiles itself one-shot and the denoiser needs the room.
RESIDENT_VAE_MAX_TOKENS = 64
TIMING = re.compile(r"^timing: (.+) ([0-9]+(?:\.[0-9]+)?) s(?: \((.*)\))?$")


class _Resident:
    """One resident server process, keyed by the setup it was sized for.

    The server polls its parent pid and exits with this process, so a crashed
    run leaves nothing on the GPU; close() stops it earlier."""

    def __init__(self, directory: Path, name: str, ready: str):
        self.directory, self.name, self.ready = directory, name, ready
        self.process: subprocess.Popen | None = None
        self.key = None
        self.socket = directory / f"{name}.sock"
        self.log = directory / f"{name}.log"
        self.starts = 0

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def ensure(self, key, command: list[str]) -> tuple[Path | None, float]:
        """(socket or None, seconds spent starting it)."""
        if self.alive() and self.key == key:
            return self.socket, 0.0
        self.stop()
        started = time.perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.log.open("w") as stream:
            self.process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        while time.perf_counter() - started < RESIDENT_START_TIMEOUT:
            text = self.log.read_text(errors="replace")
            if any(line.startswith(self.ready) for line in text.splitlines()):
                self.key = key
                self.starts += 1
                return self.socket, time.perf_counter() - started
            if self.process.poll() is not None:
                break
            time.sleep(0.05)
        self.stop()
        return None, time.perf_counter() - started

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process, self.key = None, None
        self.socket.unlink(missing_ok=True)


class NativeBackend:
    name = "native"
    max_references = 1
    OPTIONS = ("model", "preset", "attention", "condition_resolution", "cache_dir", "keep_work",
               "resident_socket", "resident_vae_socket", "python", "resident")

    def __init__(self, model=DEFAULT_MODEL, preset: str | None = "fast12", attention: str | None = None,
                 condition_resolution: int = 1024, cache_dir=None, keep_work: bool = False,
                 resident_socket=None, resident_vae_socket=None, python=None, resident: bool = True):
        if preset is not None and preset not in PRESET_WEIGHTS:
            raise BackendError(f"preset must be one of {', '.join(PRESET_WEIGHTS)}, got {preset!r}")
        if attention is not None and attention not in ATTENTION:
            raise BackendError(f"attention must be one of {', '.join(ATTENTION)}, got {attention!r}")
        if not 256 <= condition_resolution <= 1024:
            raise BackendError("condition_resolution must be in [256, 1024] (the native encoder's range)")
        self.model = Path(model).resolve()
        self.preset, self.attention = preset, attention
        self.condition_resolution = condition_resolution
        self.cache_dir = Path(cache_dir) if cache_dir else ROOT / "tmp/qimg21-prompt-cache"
        self.keep_work = keep_work
        self.resident_socket, self.resident_vae_socket = resident_socket, resident_vae_socket
        self.python = str(python or sys.executable)
        self.resident = resident and preset is not None
        self._home = ROOT / "tmp" / f"qimg21-i23d-{os.getpid()}"
        self._fast = _Resident(self._home, "fast", "fast: serving ")
        self._vae = _Resident(self._home, "vae", "qimg21-vae: serving")
        self._encoded: set[tuple[str, int]] = set()   # (reference sha256, condition resolution)
        self._pending: dict[tuple[str, int], list[str]] = {}   # prompts to encode with a reference's next run

    def condition_size(self, image) -> tuple[int, int, int]:
        """(resolution, width, height) of a reference's condition image.

        The pipeline resizes a reference to resolution^2 pixels at its own
        aspect ratio (sides rounded to 32); the native encoder takes at most
        1024 per side, so tall or wide references get a lower resolution."""
        from PIL import Image
        with Image.open(image) as im:
            width, height = im.size
        ratio = width / height
        resolution = self.condition_resolution
        while True:
            w = round((resolution * resolution * ratio) ** 0.5 / 32) * 32
            h = round((resolution * resolution / ratio) ** 0.5 / 32) * 32
            if max(w, h) <= 1024 or resolution <= 256:
                return resolution, w, h
            resolution -= 16

    def fit_condition_resolution(self, image) -> int:
        return self.condition_size(image)[0]

    def _encode_key(self, request: GenRequest):
        if not request.references:
            return None
        from .imageops import sha256_file
        return sha256_file(request.references[0]), self.fit_condition_resolution(request.references[0])

    def needs_encoder(self, request: GenRequest) -> bool:
        """Whether the driver will run a one-shot encoder: VAE/vision for a
        new reference or an init image, or a batched text-encoder pass."""
        if request.init_image is not None and (request.strength < 1.0 or request.mask is not None):
            return True
        key = self._encode_key(request)
        return key is not None and (key not in self._encoded or bool(self._pending.get(key)))

    def prepare(self, requests) -> None:
        """Note the prompts of requests to come: the next run with their
        reference encodes all of them in one text-encoder pass (weights
        streamed once, results bitwise the single-prompt ones) into the
        prompt cache, so the other runs skip the text encoder."""
        for request in requests:
            key = self._encode_key(request)
            if key is not None:
                pending = self._pending.setdefault(key, [])
                if request.prompt not in pending:
                    pending.append(request.prompt)

    def resident_sockets(self, request: GenRequest) -> tuple[str | None, str | None, list[dict]]:
        """(denoiser socket, VAE socket, startup timings) for a request,
        starting or re-sizing this backend's resident processes as needed."""
        if self.resident_socket or self.resident_vae_socket or not self.resident or not FAST.is_file():
            return self.resident_socket, self.resident_vae_socket, []
        if self.needs_encoder(request):
            self._fast.stop()
            self._vae.stop()
            return None, None, []
        h, w = request.height // 16, request.width // 16
        condition = 0
        if request.references:
            _, cw, ch = self.condition_size(request.references[0])
            condition = (cw // 16) * (ch // 16)
        weights = PRESET_WEIGHTS[self.preset]
        package = PACKAGES[weights] if weights else ""
        attention = ATTENTION.get(self.attention or "", "")
        key = (FAST.stat().st_mtime_ns, str(self.model), self.preset, package, h, w,
               bool(request.negative_prompt), attention, condition)
        command = [str(FAST), "--serve", str(self._fast.socket), "--preset", self.preset, "--model", str(self.model),
                   "--height-tokens", str(h), "--width-tokens", str(w),
                   "--serve-cfg", "1" if request.negative_prompt else "0",
                   "--serve-condition-tokens", str(condition)]
        if attention:
            command += ["--attention", attention]
        if package:
            command += ["--quant-package", package]
        timings = []
        fast, seconds = self._fast.ensure(key, command)
        if seconds:
            timings.append({"label": "start resident denoiser", "seconds": round(seconds, 3)})
        vae = None
        if fast and VAE.is_file() and max(h, w) <= RESIDENT_VAE_MAX_TOKENS:
            vae, seconds = self._vae.ensure((VAE.stat().st_mtime_ns, str(self.model)),
                                            [str(VAE), "--serve", str(self._vae.socket), "--model",
                                             str(self.model / "vae"), "--conv", "cudnn", "--tf32"])
            if seconds:
                timings.append({"label": "start resident VAE decoder", "seconds": round(seconds, 3)})
        return (str(fast) if fast else None), (str(vae) if vae else None), timings

    @staticmethod
    def available(model=DEFAULT_MODEL) -> bool:
        return FAST.is_file() and Path(model).is_dir()

    def command(self, request: GenRequest, work: Path, sockets=None) -> list[str]:
        """The native_generate.py invocation for a request. Its only side
        effect is the prepared prompt list, written into `work`."""
        cmd = [self.python, str(HERE / "native_generate.py"), "--backend", "cuda", "--model", str(self.model),
               "--prompt", request.prompt, "--height", str(request.height), "--width", str(request.width),
               "--steps", str(request.steps), "--seed", str(request.seed), "--dtype", "bf16",
               "--native-vae", "--prompt-cache", str(self.cache_dir), "--work-dir", str(work / "run"),
               "--out", str(request.out)]
        if self.preset:
            cmd += ["--runner", "fast", "--preset", self.preset, "--vae-tf32"]
            weights = PRESET_WEIGHTS[self.preset]
            if weights:
                cmd += ["--quant-package", PACKAGES[weights]]
            if self.attention:
                cmd += ["--native-attention", ATTENTION[self.attention]]
        batch = self._pending.get(self._encode_key(request)) if request.references else None
        if batch and [p for p in batch if p != request.prompt]:
            import json
            listing = work / "prompt_batch.json"
            listing.write_text(json.dumps([p for p in batch if p != request.prompt]))
            cmd += ["--prompt-batch", str(listing)]
        if request.references:
            cmd += ["--image", str(Path(request.references[0]).resolve()),
                    "--condition-resolution", str(self.fit_condition_resolution(request.references[0])),
                    "--condition-cache", str(self.cache_dir)]
        if request.negative_prompt:
            cmd += ["--negative-prompt", request.negative_prompt, "--true-cfg-scale", str(request.true_cfg_scale)]
        if request.init_image is not None and (request.strength < 1.0 or request.mask is not None):
            cmd += ["--init-image", str(Path(request.init_image).resolve()), "--strength", str(request.strength)]
        if request.mask is not None:
            cmd += ["--mask", str(Path(request.mask).resolve())]
        fast, vae = sockets if sockets is not None else (self.resident_socket, self.resident_vae_socket)
        if fast:
            cmd += ["--resident-socket", str(fast)]
        if vae:
            cmd += ["--resident-vae-socket", str(vae)]
        return cmd

    def generate(self, request: GenRequest) -> GenResult:
        # One condition image natively: a mask is enforced (blending plus
        # paste-back) but cannot also be shown as a second reference.
        mask_shown = False
        request = replace(request, mask_as_reference=False).validate(self.max_references)
        if (request.mask is not None or request.strength < 1.0) and not self.preset:
            raise BackendError("init images and masks need a fast preset (--runner fast)")
        Path(request.out).parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        work = Path(tempfile.mkdtemp(prefix="qimg21-native-", dir=ROOT / "tmp"))
        log = work / "native.log"
        env = {**os.environ, "QIMG21_PYTHON": self.python}
        try:
            fast, vae, timings = self.resident_sockets(request)
            with log.open("w") as stream:
                code = subprocess.run(self.command(request, work, (fast, vae)), cwd=ROOT, stdout=stream,
                                      stderr=subprocess.STDOUT, env=env).returncode
            text = log.read_text(errors="replace")
            if code or not Path(request.out).is_file():
                raise BackendError(f"native generation failed ({code}): {text[-3000:]}")
            key = self._encode_key(request)
            if key is not None:
                self._encoded.add(key)
                self._pending.pop(key, None)
            timings += [{"label": m.group(1), "seconds": float(m.group(2))}
                        for m in map(TIMING.match, text.splitlines()) if m]
            resident = {"requests": text.count("+ resident "),
                        "fallbacks": text.count("resident: run not taken")}
        finally:
            if not self.keep_work:
                import shutil
                shutil.rmtree(work, ignore_errors=True)
        return GenResult(Path(request.out), time.perf_counter() - started, self.name,
                         {"timings": timings, "resident": resident, "mask_shown_as_reference": mask_shown,
                          "preset": self.preset, "attention": self.attention,
                          "condition_resolution": self.condition_resolution if request.references else None})

    def close(self) -> None:
        self._fast.stop()
        self._vae.stop()
        import shutil
        shutil.rmtree(self._home, ignore_errors=True)
