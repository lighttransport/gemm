"""Native model orchestration, image preparation and atomic video packaging."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tmp/hv15-native/build/hv15n"
ROCM_RUNNER = ROOT / "tmp/video-rocm/hv15-build/hv15n_rocm"
UPSTREAM = "60783e704160023913bee78f0b47036d393d4dfa"


class Cancelled(RuntimeError):
    pass


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def model_manifest(model, task, preset, *, verify=True):
    model = Path(model).resolve()
    manifest = json.loads((model / "model.json").read_text())
    if manifest.get("schema") != "hunyuan_video15.model.v1":
        raise ValueError("unsupported model manifest")
    if task not in ("i2v", "t2v") or preset not in ("quality", "fast12") or (task == "t2v" and preset == "fast12"):
        raise ValueError("unsupported task/preset")
    names = [manifest["checkpoints"][f"{preset}_{task}"]]
    names += [manifest["components"][k] for k in ("vae", "qwen", "byt5", "tokenizer")]
    if task == "i2v":
        if manifest.get("vision_profile") != "google_siglip_so400m_14_384":
            raise ValueError("this port requires the Google SigLIP profile")
        names.append(manifest["components"]["vision"])
    receipts = {}
    for name in dict.fromkeys(names):
        path = (model / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(model) or not path.is_file():
            raise ValueError("invalid component path")
        receipt = manifest.get("sources", {}).get(name, {})
        if not receipt.get("sha256") or not receipt.get("revision"):
            raise ValueError(f"missing pinned source receipt: {name}")
        if receipt.get("bytes") != path.stat().st_size or path.with_suffix(path.suffix + ".aria2").exists():
            raise ValueError(f"incomplete component: {name}")
        if verify and digest(path) != receipt["sha256"]:
            raise ValueError(f"component checksum mismatch: {name}")
        receipts[name] = receipt
    return manifest, receipts


def prepare_image(image, out):
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(image) as opened:
        rgba = ImageOps.exif_transpose(opened).convert("RGBA")
    background = Image.new("RGBA", rgba.size, (96, 96, 96, 255))
    background.alpha_composite(rgba)
    rgb = background.convert("RGB")
    scale = max(480 / rgb.width, 848 / rgb.height)
    resized = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.Resampling.LANCZOS)
    left, top = round((resized.width - 480) / 2), round((resized.height - 848) / 2)
    prepared = resized.crop((left, top, left + 480, top + 848))
    image_path = out / "input.png"
    prepared.save(image_path)
    vision = prepared.resize((384, 384), Image.Resampling.BICUBIC)
    pixels = (np.asarray(vision, dtype=np.float64) * (1 / 255)).astype(np.float32)
    pixels = ((pixels - np.float32(.5)) / np.float32(.5)).transpose(2, 0, 1)
    pixel_path = out / "vision_pixels.f32"
    pixels.astype("<f4").tofile(pixel_path)
    return image_path, pixel_path


def run_process(command, *, cancel=None, progress=None, log=None, on_start=None):
    scratch = ROOT / "tmp/hv15-native/subprocess-scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, TMPDIR=str(scratch))
    environment.setdefault("CUDA_CACHE_PATH", str(scratch / "cuda-cache"))
    process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True, bufsize=0, cwd=ROOT, env=environment)
    reader = selectors.DefaultSelector()
    reader.register(process.stdout, selectors.EVENT_READ)
    pending, tail = b"", []
    def received(raw):
        line = raw.decode("utf-8", errors="replace")
        tail.append(line)
        del tail[:-40]
        if log:
            log.write(line)
            log.flush()
        fields = line.split()
        if len(fields) == 3 and fields[0] == "PROGRESS" and progress:
            progress(int(fields[1]), int(fields[2]))
    try:
        if on_start:
            on_start(process.pid)
        while True:
            if cancel and cancel.is_set():
                raise Cancelled("cancelled")
            if not reader.select(.1):
                if process.poll() is not None:
                    continue
                continue
            block = os.read(process.stdout.fileno(), 65536)
            if not block:
                if pending:
                    received(pending)
                break
            pending += block
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                received(line + b"\n")
            if len(pending) > 65536:
                received(pending)
                pending = b""
        code = process.wait()
        if code:
            raise RuntimeError(f"native process failed ({code}): {''.join(tail)[-4000:]}")
    finally:
        reader.close()
        process.stdout.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


@contextmanager
def device_lock(device, cancel=None):
    path = ROOT / "tmp/pixal3d/device-locks" / f"rocm-{device}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        while True:
            if cancel and cancel.is_set():
                raise Cancelled("cancelled while waiting for AMD device")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(.1)
        yield


def amd_process_vram(pid):
    # DRM counters repeat across descriptors; take the maximum for each device.
    devices = {}
    for path in Path(f"/proc/{pid}/fdinfo").glob("*"):
        fields = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
        if "drm-memory-vram" in fields:
            value, unit = fields["drm-memory-vram"].split()
            factor = {"KiB": 1 / 1024, "MiB": 1, "bytes": 1 / 1048576}[unit]
            key = fields.get("drm-pdev", fields.get("drm-client-id", "device"))
            devices[key] = max(devices.get(key, 0), int(value) * factor)
    if devices:
        return sum(devices.values())
    counters = list(Path(f"/sys/class/kfd/kfd/proc/{pid}").glob("vram_*"))
    return sum(int(p.read_text()) for p in counters) / 1048576 if counters else None


class MemorySampler:
    def __init__(self, backend="cuda"):
        self.backend = backend
        self.stop = threading.Event()
        self.thread = None
        self.vram, self.rss = None, None
    def start(self, pid):
        def sample():
            while not self.stop.is_set():
                try:
                    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            self.rss = max(self.rss or 0, int(line.split()[1]) / 1024)
                    if self.backend == "rocm":
                        value = amd_process_vram(pid)
                        if value is not None:
                            self.vram = max(self.vram or 0, value)
                    else:
                        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
                        values = [int(parts[1]) for line in result.stdout.splitlines()
                                  if len(parts := [p.strip() for p in line.split(",")]) == 2
                                  and parts[0] == str(pid) and parts[1].isdigit()]
                        if values:
                            self.vram = max(self.vram or 0, sum(values))
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
                self.stop.wait(.2)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)


def package_frames(frames, out, *, cancel=None, log=None, count=81, width=480, height=848):
    from PIL import Image
    paths = sorted(frames.glob("frame_*.ppm"))
    if [p.name for p in paths] != [f"frame_{i:05d}.ppm" for i in range(count)]:
        raise ValueError("native runner did not produce exactly the requested consecutive frames")
    for path in paths:
        with Image.open(path) as image:
            if image.size != (width, height) or image.mode != "RGB":
                raise ValueError("invalid native RGB frame")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    run_process([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "24",
        "-i", frames / "frame_%05d.ppm", "-frames:v", str(count), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", out / "clip.mp4"], cancel=cancel, log=log)
    with Image.open(paths[0]) as image:
        image.save(out / "poster.png")


def generate(*, model, out, prompt, task="i2v", preset="quality", image=None, negative_prompt="",
             seed=42, device=0, vram_budget_mib=14336, gemm="repo", gemm_fallback=None,
             runner=None, allow_experimental=False, keep_frames=False, noise_file=None,
             dump_dir=None, cancel=None, progress=None, backend="cuda", aotriton_bridge=None):
    if backend not in ("cuda", "rocm"):
        raise ValueError("backend must be cuda or rocm")
    runner = Path(runner) if runner else (ROCM_RUNNER if backend == "rocm" else RUNNER)
    gemm_fallback = gemm_fallback or ("error" if backend == "rocm" else "cublas")
    vendor = "hipblas" if backend == "rocm" else "cublas"
    if gemm not in ("repo", vendor) or gemm_fallback not in ("error", vendor):
        raise ValueError("GEMM choice does not match the backend")
    if not allow_experimental:
        raise ValueError("native inference requires --allow-experimental until GPU parity is established")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1 or not prompt or len(prompt.encode()) > 4096:
        raise ValueError("invalid prompt or seed")
    if (task == "i2v") != bool(image):
        raise ValueError("I2V requires a portrait; T2V omits it")
    memory = next((int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                   if line.startswith("MemTotal:")), 0)
    if memory < 60 * 1024**2:
        raise ValueError("the initial block-offload profile requires a 64 GB host (at least 60 GiB usable RAM)")
    bridge_receipt = None
    if aotriton_bridge:
        aotriton_bridge = Path(aotriton_bridge).resolve()
        if backend != "rocm" or not aotriton_bridge.is_file():
            raise ValueError("AOTriton bridge requires ROCm and an existing shared library")
        bridge_receipt = {"path": str(aotriton_bridge), "bytes": aotriton_bridge.stat().st_size,
                          "sha256": digest(aotriton_bridge)}
    manifest, receipts = model_manifest(model, task, preset)
    # Freeze provenance before executing: an independent coding agent may be
    # rebuilding files while this long-running process is active.
    runner_sha256 = digest(runner)
    source_hashes = {p.name: digest(p) for pattern in ("*.cpp", "*.hpp", "*.h", "*.py", "Makefile")
                     for p in Path(__file__).parent.glob(pattern)}
    shared_names = (("rdna4/rocew.c", "rdna4/rocew.h", "rdna4/video_common/hip_platform.hpp", "rdna4/video_common/aotriton_bridge.h",
                     "rdna4/video_common/aotriton_bridge.cpp",
                     "rdna4/video_common/kernels.hpp", "rdna4/video_common/gemm.hip",
                     "rdna4/video_common/attention.hip", "rdna4/video_common/flex_attention.hip",
                     "rdna4/video_common/precision.hip", "rdna4/video_common/timestep_basis.hpp",
                     "rdna4/video_common/make_kernels.py",
                     "rdna4/hunyuan_video15_native/gpu_hip.cpp", "common/safetensors.h")
                    if backend == "rocm" else ("cuda/gemm/cuda_gemm_ptx_kernels.h", "cuda/cuew.c",
                    "cuda/cuew.h", "cuda/cublasew.c", "cuda/cublasew.h", "common/safetensors.h"))
    shared_sources = {name: digest(ROOT / name) for name in shared_names}
    image_sha256 = digest(image) if image else None
    noise_sha256 = digest(noise_file) if noise_file else None
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    active_dump, sampler = None, None
    try:
        frames = out / "frames"
        frames.mkdir()
        command = [runner, "--generate", "--model", Path(model).resolve(), "--task", task, "--preset", preset,
            "--prompt", prompt, "--negative-prompt", negative_prompt, "--seed", seed, "--device", device,
            "--vram-budget-mib", vram_budget_mib, "--gemm", gemm, "--gemm-fallback", gemm_fallback,
            "--offload", "block", "--allow-experimental", "--out-dir", frames]
        if aotriton_bridge:
            command += ["--aotriton-bridge", aotriton_bridge]
        if image:
            prepared, pixels = prepare_image(image, out)
            command += ["--image", prepared, "--vision-pixels", pixels]
        if noise_file:
            command += ["--noise-file", Path(noise_file).resolve()]
        if dump_dir:
            dump_dir = Path(dump_dir).resolve()
            if dump_dir.exists() and any(dump_dir.iterdir()):
                raise ValueError("dump directory must be new or empty")
            active_dump = dump_dir
            command += ["--dump-dir", dump_dir]
        sampler = MemorySampler(backend)
        started = time.monotonic()
        with (out / "runner.log").open("w") as log:
            try:
                if backend == "rocm":
                    with device_lock(device, cancel):
                        run_process(command, cancel=cancel, progress=progress, log=log, on_start=sampler.start)
                else:
                    run_process(command, cancel=cancel, progress=progress, log=log, on_start=sampler.start)
            finally:
                sampler.close()
            metrics = json.loads((frames / "runner_metrics.json").read_text())
            if metrics.get("backend") != "hv15n_" + backend:
                raise ValueError("runner did not identify the repository native backend")
            package_frames(frames, out, cancel=cancel, log=log)
        if cancel and cancel.is_set():
            raise Cancelled("cancelled")
        metrics.update(wall_seconds=time.monotonic() - started, sampled_peak_vram_mib=sampler.vram,
                       sampled_peak_host_rss_mib=sampler.rss, memory_fit="unverified" if sampler.vram is None
                       else ("pass" if sampler.vram <= vram_budget_mib else "fail"))
        if metrics["memory_fit"] == "fail":
            raise RuntimeError("sampled native VRAM exceeded the requested budget")
        result = {"schema": "hunyuan_video15.video.v1", "backend": "hv15n_" + backend + "_experimental",
            "upstream_reference_revision": UPSTREAM, "runner_sha256": runner_sha256,
            "runtime_sources": source_hashes, "shared_sources": shared_sources,
            "task": task, "preset": preset, "prompt": prompt, "negative_prompt": negative_prompt,
            "frames": 81, "fps": 24, "width": 480, "height": 848, "seed": seed,
            "steps": 12 if preset == "fast12" else 50, "cfg": 1 if preset == "fast12" else 6,
            "flow_shift": 7 if preset == "fast12" else 5,
            "gemm": gemm, "gemm_fallback": gemm_fallback, "aotriton_bridge": bridge_receipt, "model": manifest, "verified_components": receipts,
            "vision_profile": manifest.get("vision_profile") if image else None,
            "image_sha256": image_sha256, "noise_sha256": noise_sha256,
            "vram_budget_mib": vram_budget_mib, "metrics": metrics, "parity": "unverified"}
        atomic_json(out / "metrics.json", metrics)
        atomic_json(out / "manifest.json", result)
        if not keep_frames:
            shutil.rmtree(frames)
        return result
    except BaseException as error:
        if active_dump and active_dump.is_dir():
            try:
                metrics_path = out / "frames/runner_metrics.json"
                failure_metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
                atomic_json(active_dump / "failure.json", {"scope": "failed_generation_diagnostic",
                    "error": str(error), "runner_sha256": runner_sha256, "runtime_sources": source_hashes,
                    "metrics": failure_metrics, "sampled_peak_vram_mib": sampler.vram if sampler else None,
                    "sampled_peak_host_rss_mib": sampler.rss if sampler else None, "parity": "unverified"})
            except (OSError, ValueError):
                pass  # Diagnostic failures must not prevent partial-output cleanup.
        shutil.rmtree(out, ignore_errors=True)
        raise


def main(default_backend="cuda"):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("model", "out", "prompt"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--task", choices=("i2v", "t2v"), default="i2v")
    parser.add_argument("--preset", choices=("quality", "fast12"), default="quality")
    parser.add_argument("--image")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vram-budget-mib", type=int, default=14336)
    parser.add_argument("--gemm", choices=("repo", "cublas", "hipblas"), default="repo")
    parser.add_argument("--gemm-fallback", choices=("cublas", "hipblas", "error"))
    parser.add_argument("--runner")
    parser.add_argument("--backend", choices=("cuda", "rocm"), default=default_backend)
    parser.add_argument("--allow-experimental", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--noise-file")
    parser.add_argument("--dump-dir")
    parser.add_argument("--aotriton-bridge", help="optional standalone ROCm FP16 attention bridge")
    args = parser.parse_args()
    print(json.dumps(generate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
