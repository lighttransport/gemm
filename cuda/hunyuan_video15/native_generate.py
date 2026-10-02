#!/usr/bin/env python3
"""Native inference orchestration and FFmpeg packaging; no torch imports."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "cuda/hunyuan_video15/test_cuda_hunyuan_video15"
PIN = "3f8527a46c54ecf4cb4ed6003da8e8982283c73c"

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Cancelled(RuntimeError):
    pass

def run_process(cmd, *, cancel=None, progress=None, log=None, on_start=None):
    """Keep cancellation responsive during loading, denoising and encoding."""
    process = subprocess.Popen([str(x) for x in cmd], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
    if on_start:
        on_start(process.pid)
    output, pending = [], b""
    reader = selectors.DefaultSelector()
    reader.register(process.stdout, selectors.EVENT_READ)
    def line_received(raw):
        line = raw.decode("utf-8", errors="replace")
        output.append(line)
        del output[:-50]
        if log:
            log.write(line); log.flush()
        fields = line.strip().split()
        if len(fields) == 3 and fields[0] == "PROGRESS" and progress:
            progress(int(fields[1]), int(fields[2]))
    try:
        while True:
            if cancel and cancel.is_set():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise Cancelled("cancelled")
            if not reader.select(0.1):
                continue
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                if pending:
                    line_received(pending)
                break
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                line_received(line + b"\n")
            if len(pending) > 65536:
                line_received(pending)
                pending = b""
        code = process.wait()
        if code:
            raise RuntimeError(f"process failed ({code}): {''.join(output)[-4000:]}")
    finally:
        reader.close()
        process.stdout.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

class VramSampler:
    """Sample memory for the owned native PID, not unrelated GPU processes."""
    def __init__(self):
        self.peak = None
        self.peak_host_rss_mib = None
        self.stop = threading.Event()
        self.thread = None
    def start(self, pid):
        def sample():
            while not self.stop.is_set():
                try:
                    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            memory = int(line.split()[1]) / 1024
                            self.peak_host_rss_mib = max(self.peak_host_rss_mib or 0, memory)
                            break
                except (OSError, ValueError):
                    pass
                try:
                    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
                    for line in result.stdout.splitlines():
                        fields = [v.strip() for v in line.split(",")]
                        if len(fields) == 2 and fields[0] == str(pid) and fields[1].isdigit():
                            memory = int(fields[1])
                            self.peak = max(self.peak or 0, memory)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                self.stop.wait(0.2)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)


def load_manifest(model, task, preset):
    model = Path(model).resolve()
    manifest = json.loads((model / "model.json").read_text())
    if manifest.get("schema") != "hunyuan_video15.model.v1":
        raise ValueError("unsupported model manifest")
    names = [manifest["checkpoints"][f"{preset}_{task}"]]
    names.extend(manifest["components"][k] for k in ("vae", "qwen", "byt5", "vision", "tokenizer"))
    for name in names:
        path = (model / name).resolve()
        if not isinstance(name, str) or not path.is_relative_to(model) or not path.is_file():
            raise ValueError(f"missing or unsafe model component: {name}")
    if not manifest.get("sources"):
        raise ValueError("model provenance receipts are required")
    return manifest

def encode_frames(frames, out, count, *, cancel=None, log=None):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for MP4 packaging")
    files = sorted(frames.glob("frame_*.ppm"))
    if len(files) != count or any(p.name != f"frame_{i:05d}.ppm" for i, p in enumerate(files)):
        raise RuntimeError("runner returned an incomplete frame sequence")
    run_process([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-framerate", "24", "-i", frames / "frame_%05d.ppm", "-frames:v", count,
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out / "clip.mp4"], cancel=cancel, log=log)
    run_process([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", files[0], "-frames:v", "1", out / "poster.png"], cancel=cancel, log=log)

def prepare_portrait(image, out, width, height):
    """Non-neural preprocessing: official Lanczos crop, SigLIP PIL bicubic + 0.5 normalization."""
    from PIL import Image
    import numpy as np
    with Image.open(image) as source:
        if "A" in source.getbands() or "transparency" in source.info:
            rgba = source.convert("RGBA")
            source = Image.alpha_composite(Image.new("RGBA", rgba.size, (96, 96, 96, 255)), rgba).convert("RGB")
        else:
            source = source.convert("RGB")
        scale = max(width / source.width, height / source.height)
        rw, rh = round(source.width * scale), round(source.height * scale)
        resized = source.resize((rw, rh), Image.Resampling.LANCZOS)
        prepared = resized.crop(((rw - width) / 2, (rh - height) / 2,
                                 (rw + width) / 2, (rh + height) / 2))
        prepared.save(out / "input.png")
        pixels = np.asarray(prepared.resize((384, 384), Image.Resampling.BICUBIC), dtype=np.float32)
        pixels = (pixels.astype(np.float64) * (1 / 255)).astype(np.float32)
        pixels = ((pixels - np.float32(0.5)) / np.float32(0.5))
        pixels.transpose(2, 0, 1).astype("<f4").tofile(out / "siglip_pixels.f32")
    return out / "input.png", out / "siglip_pixels.f32"


def generate(*, model, image, prompt, out, task="i2v", preset="quality", frames=81,
             seed=42, runner=RUNNER, width=480, height=848, device=0,
             vram_budget_mib=14336, allow_experimental=False, keep_frames=False,
             cancel=None, progress=None):
    if not allow_experimental:
        raise ValueError("full pipeline parity is unverified; enable experimental video explicitly")
    manifest = load_manifest(model, task, preset)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    frame_dir = out / "frames"
    frame_dir.mkdir()
    cmd = [runner, "--generate", "--model", model, "--task", task, "--preset", preset,
        "--prompt", prompt, "--frames", frames, "--seed", seed, "--width", width,
        "--height", height, "--device", device, "--vram-budget-mib", vram_budget_mib,
        "--offload", "block", "--allow-experimental", "--out-dir", frame_dir]
    started = time.monotonic()
    try:
        runner_sha256 = file_sha256(runner)
        library = ROOT / "tmp/hunyuan-video15-native/build/bin/libstable-diffusion.so"
        library_sha256 = (file_sha256(library) if Path(runner).resolve() == RUNNER.resolve()
                          and library.is_file() and not os.environ.get("LD_LIBRARY_PATH") else None)
        overlay_sha256 = {name: file_sha256(Path(__file__).parent / name)
            for name in ("patch_native.py", "siglip.hpp", "vision_projection.hpp", "cuda_attention.cu",
                         "cuda_attention.h", "cuda_math.cu", "cuda_math.h", "cuda_overlay.cmake", "progress.cpp", "progress.h", "vae_spatial_tiling.hpp",
                         "conditioning_inputs.hpp", "dump.hpp")}
        if image:
            prepared, pixels = prepare_portrait(image, out, width, height)
            cmd.extend(["--image", prepared, "--vision-pixels", pixels])
        sampler = VramSampler()
        with (out / "runner.log").open("w") as log:
            try:
                run_process(cmd, cancel=cancel, progress=progress, log=log, on_start=sampler.start)
            finally:
                sampler.close()
            encode_frames(frame_dir, out, frames, cancel=cancel, log=log)
        if cancel and cancel.is_set():
            raise Cancelled("cancelled")
        attention_counts = re.findall(r"HV15 precise attention calls: (\d+)",
                                      (out / "runner.log").read_text(errors="replace"))
        metrics = {"wall_seconds": time.monotonic() - started, "peak_vram_mib": sampler.peak,
                   "peak_host_rss_mib": sampler.peak_host_rss_mib,
                   "precise_attention_calls": int(attention_counts[-1]) if attention_counts else None,
                   "peak_vram_status": "sampled_native_pid_200ms" if sampler.peak is not None else "not_measured",
                   "within_budget": sampler.peak <= vram_budget_mib if sampler.peak is not None else None, "parity": "unverified"}
        provenance = {"schema": "hunyuan_video15.video.v1", "backend": "native_cuda_experimental",
            "native_revision": PIN, "runner_sha256": runner_sha256,
            "native_default_library_build_sha256": library_sha256,
            "conditioning_overlay": "siglip_so400m_v2_ieee_f32",
            "task": task, "preset": preset, "frames": frames, "fps": 24, "seed": seed,
            "width": width, "height": height, "prompt": prompt,
            "alpha_background_rgb": [96, 96, 96],
            "vae_settings": {"temporal_tiling": False, "spatial_tile_pixels": 128,
                             "spatial_overlap": 0.25, "spatial_stride": "fixed",
                             "spatial_edges": "clipped", "spatial_blend": "linear_vertical_then_horizontal",
                             "attention": "dense_causal"},
            "vision_profile": manifest.get("vision_profile"),
            "vision_precision": {"weights": "float16", "activations": "float32",
                                 "patch_im2col": "float32", "cublas_math": "ieee_float32"},
            "image_sha256": hashlib.sha256(Path(image).read_bytes()).hexdigest() if image else None,
            "vram_budget_mib": vram_budget_mib, "managed_vram_mib": vram_budget_mib - 3072,
            "rng": "matched_reference_override" if os.environ.get("HV15_NOISE_F32") else "native_cuda_philox",
            "noise_override_sha256": hashlib.sha256(Path(os.environ["HV15_NOISE_F32"]).read_bytes()).hexdigest() if os.environ.get("HV15_NOISE_F32") else None,
            "overlay_sha256": overlay_sha256,
            "steps": 12 if preset == "fast12" else 50,
            "cfg": 1 if preset == "fast12" else 6, "flow_shift": 7 if preset == "fast12" else 5,
            "model": manifest, "metrics": metrics}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        # Manifest is the completion marker. Failed/cancelled jobs never publish it.
        partial = out / "manifest.json.partial"
        partial.write_text(json.dumps(provenance, indent=2) + "\n")
        partial.replace(out / "manifest.json")
        if not keep_frames:
            shutil.rmtree(frame_dir)
        return provenance
    except BaseException:
        shutil.rmtree(out)
        raise

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="new artifact directory")
    ap.add_argument("--task", choices=("i2v", "t2v"), default="i2v")
    ap.add_argument("--preset", choices=("quality", "fast12"), default="quality")
    ap.add_argument("--image")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--frames", type=int, choices=(81, 121), default=81)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--vram-budget-mib", type=int, default=14336)
    ap.add_argument("--runner", default=str(RUNNER))
    ap.add_argument("--allow-experimental", action="store_true")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()
    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())
    try:
        generate(**vars(args), cancel=cancel, progress=lambda s, n: print(f"PROGRESS {s} {n}", flush=True))
    except Cancelled:
        return 130
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
