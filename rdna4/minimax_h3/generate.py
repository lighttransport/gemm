"""Run native H3 INT8 ConvRot on RDNA4 or CUDA and publish a silent 24 fps MP4."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hv15_video_tools", ROOT / "cuda/hunyuan_video15_native/generate.py")
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)
RUNNER = ROOT / "tmp/video-rocm/h3-build/h3_rocm"
CUDA_RUNNER = ROOT / "tmp/video-cuda/h3-build/h3_cuda"
DEFAULT_MODEL = {"rocm": "/mnt/disk01/models/h3/weights", "cuda": "/mnt/nvme01/models/h3/weights"}
UPSTREAM = "2472a20bd291451acc303917059ab14dfc380478"
COMPONENTS = ("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
              "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
              "vae/minimax_h3_video_vae_fp16.safetensors", "tokenizer/tokenizer.json")


def generate(*, model=None, out, prompt, width=1344, height=768,
             frames=124, steps=40, seed=42, device=0, vram_budget_mib=14336, runner=None,
             allow_experimental=False, keep_frames=False, noise_file=None, audio_noise_file=None,
             dump_dir=None, convrot_hipblas=None, bf16_hipblas=1, aotriton_bridge=None, vae_hipblas=0,
             cancel=None, progress=None, compress_dumps=False, fp32_hipblas=1, backend="rocm", cudnn_attention=None,
             cudnn_library=None):
    if backend not in ("cuda", "rocm"):
        raise ValueError("backend must be cuda or rocm")
    if cudnn_attention not in (None, "off", "auto") and not Path(cudnn_attention).is_file():
        raise ValueError("cudnn_attention must be off, auto or a bridge library path")
    if cudnn_library and (cudnn_attention in (None, "off") or not Path(cudnn_library).is_file()):
        raise ValueError("cudnn_library must be an existing file and requires cudnn_attention")
    if backend == "rocm" and cudnn_attention not in (None, "off"):
        raise ValueError("cuDNN attention is CUDA-only")
    if backend == "cuda" and aotriton_bridge:
        raise ValueError("the AOTriton bridge is ROCm-only")
    model = model or DEFAULT_MODEL[backend]
    if convrot_hipblas is None:  # CUDA fuses the factorized rotation by default
        convrot_hipblas = 0 if backend == "cuda" else 1
    runner = runner or (CUDA_RUNNER if backend == "cuda" else RUNNER)
    backend_name = f"minimax_h3_{backend}_experimental"
    if not allow_experimental:
        raise ValueError("native H3 requires --allow-experimental until full GPU parity is established")
    if not prompt or len(prompt.encode()) > 4096 or type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("invalid prompt or seed")
    if width > 2048 or height > 2048 or width < 64 or height < 64 or width % 32 or height % 32 or width * height > 1344 * 768:
        raise ValueError("dimensions must be multiples of 32, >=64, with area <=1344*768")
    if frames < 5 or frames > 362 or (frames - 5) % 17 or not 2 <= steps <= 100:
        raise ValueError("frames must be 17*n+5 in 5..362; steps must be 2..100 sigma-grid points")
    if not 4096 <= vram_budget_mib <= 14336 or device < 0:
        raise ValueError("invalid device or VRAM budget")
    model, out, runner = Path(model).resolve(), Path(out).resolve(), Path(runner).resolve()
    stage = out.with_name(out.name + ".partial")
    if out.exists() or stage.exists():
        raise FileExistsError("output and staging directories must be new")
    if type(convrot_hipblas) is not int or convrot_hipblas not in (0, 1):
        raise ValueError("convrot_hipblas must be 0 or 1")
    if type(bf16_hipblas) is not int or bf16_hipblas not in (0, 1):
        raise ValueError("bf16_hipblas must be 0 or 1")
    if type(fp32_hipblas) is not int or fp32_hipblas not in (0, 1):
        raise ValueError("fp32_hipblas must be 0 or 1")
    if type(vae_hipblas) is not int or vae_hipblas not in (0, 1):
        raise ValueError("vae_hipblas must be 0 or 1")
    bridge_receipt = None
    if aotriton_bridge:
        aotriton_bridge = Path(aotriton_bridge).resolve()
        if not aotriton_bridge.is_file():
            raise ValueError("AOTriton bridge must be an existing shared library")
        bridge_receipt = {"path": str(aotriton_bridge), "bytes": aotriton_bridge.stat().st_size,
                          "sha256": video.digest(aotriton_bridge)}
    if type(compress_dumps) is not bool or (compress_dumps and not dump_dir):
        raise ValueError("dump compression requires a capture directory")
    receipts = {}
    for name in COMPONENTS:
        path = (model / name).resolve()
        if not path.is_relative_to(model) or not path.is_file() or path.with_suffix(path.suffix + ".aria2").exists():
            raise ValueError(f"missing or incomplete H3 component: {name}")
        receipts[name] = {"bytes": path.stat().st_size, "sha256": video.digest(path)}
    sources = {str(p.relative_to(ROOT)): video.digest(p) for directory in
               (ROOT / "rdna4/minimax_h3", ROOT / "rdna4/video_common")
               for p in directory.iterdir() if p.suffix in (".cpp", ".h", ".hpp", ".hip", ".py") or p.name == "Makefile"}
    platform = (("cuda/cuew.c", "cuda/cuew.h", "cuda/cublasew.c", "cuda/cublasew.h",
                 "cuda/fa2/cuda_fa2_kernels.h", "cuda/hunyuan_video15_native/gpu.cpp",
                 "cuda/minimax_h3/Makefile") if backend == "cuda" else
                ("rdna4/rocew.c", "rdna4/rocew.h", "rdna4/hunyuan_video15_native/gpu_hip.cpp"))
    for name in (*platform,
                 "cuda/hunyuan_video15_native/host.hpp", "cuda/hunyuan_video15_native/tokenizer.hpp",
                 "cuda/hunyuan_video15_native/gpu.hpp", "cuda/hunyuan_video15_native/loader.cpp",
                 "cuda/hunyuan_video15_native/kernels.hpp", "common/safetensors.h"):
        sources[name] = video.digest(ROOT / name)
    sources["ref/minimax_h3_native/captures.py"] = video.digest(ROOT / "ref/minimax_h3_native/captures.py")
    provenance = {"upstream_reference_revision": UPSTREAM, "runner_sha256": video.digest(runner),
                  "runtime_sources": sources, "verified_components": receipts,
                  "aotriton_bridge": bridge_receipt}
    memory = next((int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                   if line.startswith("MemTotal:")), 0)
    if memory < 60 * 1024**2:
        raise ValueError("H3 block offload requires a 64 GB host (at least 60 GiB usable RAM)")
    stage.mkdir(parents=True)
    sampler = video.MemorySampler(backend)
    started = time.monotonic()
    try:
        raw = stage / "frames"
        raw.mkdir()
        command = [runner, "--generate", "--allow-experimental", "--model", model, "--prompt", prompt,
                   "--width", width, "--height", height, "--frames", frames, "--steps", steps,
                   "--seed", seed, "--device", device, "--vram-budget-mib", vram_budget_mib,
                   "--convrot-hipblas", convrot_hipblas, "--bf16-hipblas", bf16_hipblas,
                   "--vae-hipblas", vae_hipblas, "--fp32-hipblas", fp32_hipblas, "--out-dir", raw]
        if aotriton_bridge:
            command += ["--aotriton-bridge", aotriton_bridge]
        if cudnn_attention:
            command += ["--cudnn-attention", cudnn_attention]
        if cudnn_library:
            command += ["--cudnn-library", Path(cudnn_library).resolve()]
        for flag, value in (("--noise-file", noise_file), ("--audio-noise-file", audio_noise_file), ("--dump-dir", dump_dir)):
            if value:
                path = Path(value).resolve()
                if flag == "--dump-dir":
                    if path.exists() and any(path.iterdir()):
                        raise ValueError("dump directory must be new or empty")
                    path.mkdir(parents=True, exist_ok=True)
                    video.atomic_json(path / "provenance.json", provenance)
                command += [flag, path]
        with (stage / "runner.log").open("w") as log:
            with video.device_lock(device, cancel):
                try:
                    video.run_process(command, cancel=cancel, progress=progress, log=log, on_start=sampler.start)
                finally:
                    sampler.close()
            metrics = json.loads((raw / "metrics.json").read_text())
            if metrics.get("backend") != backend_name or metrics.get("int8_wmma_calls", 0) <= 0:
                raise ValueError("runner did not execute the native H3 INT8 WMMA backend")
            video.package_frames(raw, stage, count=frames, width=width, height=height, cancel=cancel, log=log)
        if compress_dumps:
            capture_spec = importlib.util.spec_from_file_location(
                "h3_capture_tools", ROOT / "ref/minimax_h3_native/captures.py")
            capture_tools = importlib.util.module_from_spec(capture_spec)
            capture_spec.loader.exec_module(capture_tools)
            try:
                capture_tools.compress(dump_dir, cancel.is_set if cancel else None)
            except InterruptedError:
                raise video.Cancelled("cancelled")
        if cancel and cancel.is_set():
            raise video.Cancelled("cancelled")
        metrics.update(wall_seconds=time.monotonic() - started, sampled_peak_vram_mib=sampler.vram,
                       sampled_peak_host_rss_mib=sampler.rss,
                       memory_fit="unverified" if sampler.vram is None else "pass" if sampler.vram <= vram_budget_mib else "fail")
        if metrics["memory_fit"] == "fail":
            raise RuntimeError("native process exceeded its VRAM budget")
        result = {"schema": "minimax_h3.video.v1", "backend": backend_name, **provenance,
                  "checkpoint": "ref2va_pruned_int8_convrot", "references": [], "prompt": prompt,
                  "width": width, "height": height, "frames": frames, "fps": 24, "seed": seed,
                  "sigma_grid_points": steps, "euler_updates": steps - 1,
                  "video_shift": 12, "audio_shift": 3, "joint_audio_denoising": True, "audio_output": False,
                  "noise_sha256": video.digest(noise_file) if noise_file else None,
                  "audio_noise_sha256": video.digest(audio_noise_file) if audio_noise_file else None,
                  "vram_budget_mib": vram_budget_mib, "convrot_hipblas": convrot_hipblas,
                  "bf16_hipblas": bf16_hipblas, "vae_hipblas": vae_hipblas, "fp32_hipblas": fp32_hipblas,
                  "cudnn_attention": cudnn_attention or "off",
                  "dump_compression": "gzip" if compress_dumps else "none",
                  "metrics": metrics, "parity": "unverified"}
        video.atomic_json(stage / "manifest.json", result)
        video.atomic_json(stage / "metrics.json", metrics)
        if not keep_frames:
            shutil.rmtree(raw)
        stage.replace(out)
        return result
    except BaseException as error:
        sampler.close()
        if dump_dir and Path(dump_dir).is_dir():
            try:
                video.atomic_json(Path(dump_dir) / "failure.json", {**provenance, "error": str(error),
                    "sampled_peak_vram_mib": sampler.vram, "sampled_peak_host_rss_mib": sampler.rss,
                    "parity": "unverified"})
            except (OSError, ValueError):
                pass
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(default_backend="rocm"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("cuda", "rocm"), default=default_backend)
    p.add_argument("--model", help="model directory (default depends on --backend)")
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", required=True)
    for name, default in (("width", 1344), ("height", 768), ("frames", 124), ("steps", 40), ("seed", 42),
                          ("device", 0), ("vram-budget-mib", 14336), ("convrot-hipblas", None), ("bf16-hipblas", 1),
                          ("vae-hipblas", 0), ("fp32-hipblas", 1)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--runner")
    p.add_argument("--cudnn-attention", help="CUDA only: off, auto or libh3_cudnn.so path (opt-in DiT SDPA)")
    p.add_argument("--cudnn-library", help="CUDA only: libcudnn.so.9 path for --cudnn-attention")
    for name in ("noise-file", "audio-noise-file", "dump-dir", "aotriton-bridge"):
        p.add_argument("--" + name)
    p.add_argument("--allow-experimental", action="store_true")
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--compress-dumps", action="store_true",
                   help="losslessly compress completed F32 captures to reduce disk use")
    print(json.dumps(generate(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
