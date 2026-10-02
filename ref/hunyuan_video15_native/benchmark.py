"""Time the pinned independent PyTorch graphs against native capture timings.

Fast12 is measured end-to-end; quality samples the first N steps of its exact
50-step CFG-6 schedule. A quality extrapolation is never a full-run measurement.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cuda/hunyuan_video15_native"))
from generate import Cancelled, MemorySampler, atomic_json, digest, model_manifest, package_frames
from validate_quality import LOCK, device_lock


def statistics_seconds(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("timing samples must be nonempty, positive and finite")
    return dict(samples=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values), values=list(values))


def native_timing(run, captures):
    pause_path = run / "performance_pause.json"
    pause_record = json.loads(pause_path.read_text()) if pause_path.exists() else {}
    pauses = pause_record.get("intervals", [pause_record] if pause_record else [])
    def interval(begin, end):
        return end - begin - sum(max(0, min(end, p["ended_unix"]) - max(begin, p["started_unix"])) for p in pauses)
    times = []
    for path in captures.glob("latent_step_*.json"):
        index = int(path.stem.rsplit("_", 1)[1])
        times.append((index, path.with_suffix(".f32").stat().st_mtime))
    times.sort()
    if not times or [n for n, _ in times] != list(range(len(times))):
        raise ValueError("native timings require consecutive completed captures")
    durations = [interval(a[1], b[1]) for a, b in zip(times, times[1:])]
    result = dict(scope="completed_capture_mtime_intervals", completed_steps=len(times),
                  warm_step_seconds=statistics_seconds(durations) if durations else None,
                  recorded_pause_seconds=sum(p["duration_seconds"] for p in pauses))
    if (run / "manifest.json").is_file():
        manifest = json.loads((run / "manifest.json").read_text())
        result.update(manifest_sha256=digest(run / "manifest.json"), metrics=manifest["metrics"],
                      generation_and_packaging_seconds=manifest["metrics"]["wall_seconds"])
        begin = (captures / "noise_input.f32").stat().st_mtime
        final = (captures / "latent_final.f32").stat().st_mtime
        decoded = (captures / "vae_decoded.f32").stat().st_mtime
        result.update(denoising_seconds=interval(begin, final), decode_seconds=interval(final, decoded),
                      active_generation_and_packaging_seconds=manifest["metrics"]["wall_seconds"] - result["recorded_pause_seconds"])
    return result


def proc_record(pid):
    proc = Path("/proc") / str(pid)
    parts = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    command = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    return dict(state=parts[0], parent=int(parts[1]), started=parts[19], command=command)


@contextmanager
def gpu_reservation(pid, run, report, write, cancel):
    if not pid:
        with device_lock(cancel):
            yield
        return
    native = proc_record(pid)
    command = native["command"]
    if "--out-dir" not in command or Path(command[command.index("--out-dir") + 1]).resolve() != run / "frames":
        raise ValueError("suspended PID does not own the specified native run")
    parent = proc_record(native["parent"])
    holder = parent["parent"]
    lock_stat = LOCK.stat()
    lock_id = f"{os.major(lock_stat.st_dev):02x}:{os.minor(lock_stat.st_dev):02x}:{lock_stat.st_ino}"
    if not any(line.split()[1:6] == ["FLOCK", "ADVISORY", "WRITE", str(holder), lock_id]
               for line in Path("/proc/locks").read_text().splitlines()):
        raise ValueError("native job does not hold the shared GPU reservation")
    pause = dict(native_pid=pid, native_start_ticks=native["started"], lock_holder_pid=holder,
                 started_unix=time.time(), started_monotonic=time.monotonic(), native_command=command)
    report["native_pause"] = pause
    write()
    os.kill(pid, signal.SIGSTOP)
    try:
        deadline = time.monotonic() + 30
        idle = 0
        while idle < 2:
            if cancel.is_set():
                raise Cancelled("cancelled while draining native GPU work")
            if time.monotonic() > deadline:
                raise RuntimeError("GPU did not become idle after native suspension")
            if proc_record(pid)["state"] != "T":
                cancel.wait(.25)
                continue
            sample = subprocess.check_output(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], text=True)
            rows = [line.split() for line in sample.splitlines() if line.strip() and not line.lstrip().startswith("#")]
            owned = next((row for row in rows if row[1] == str(pid)), None)
            utilization = 0 if owned is None or owned[3] == "-" else int(owned[3])
            report["native_pause"]["process_utilization_before_reference"] = sample
            idle = idle + 1 if utilization == 0 else 0
            cancel.wait(.25)
        report["native_pause"]["idle_confirmed"] = True
        write()
        yield
    finally:
        try:
            if proc_record(pid)["started"] == native["started"]:
                os.kill(pid, signal.SIGCONT)
                pause["native_resumed"] = True
        except FileNotFoundError:
            pause["native_resumed"] = False
        pause.update(ended_unix=time.time(), duration_seconds=time.monotonic() - pause["started_monotonic"])
        path = run / "performance_pause.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        intervals = previous.get("intervals", [previous] if previous else [])
        intervals.append(pause)
        atomic_json(path, dict(native_pid=pid, native_start_ticks=native["started"], intervals=intervals,
                               total_paused_seconds=sum(p["duration_seconds"] for p in intervals)))
        write()


def denoise(model_dir, manifest, generation, conditioning, noise, captures, measured_steps, cancel, progress):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
    from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
    from ref.hunyuan_video15_native.verify import save
    start = time.monotonic()
    config = {k: v for k, v in json.loads((model_dir / manifest["reference_configs"][generation["preset"] + "_i2v"]).read_text()).items()
              if not k.startswith("_")}
    config["attn_mode"] = "torch"
    with torch.device("meta"):
        model = HunyuanVideo_1_5_DiffusionTransformer(**config)
    state = load_file(str(model_dir / manifest["checkpoints"][generation["preset"] + "_i2v"]))
    for name in list(state):
        if ".img_attn_qkv." in name or ".txt_attn_qkv." in name:
            value = state.pop(name)
            for suffix, chunk in zip(("q", "k", "v"), value.chunk(3, dim=0)):
                state[name.replace("_qkv.", f"_{suffix}.")] = chunk
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval().requires_grad_(False)
    def before(child, unused):
        if cancel.is_set():
            raise Cancelled("cancelled reference denoising")
        child.to(device="cuda", dtype=torch.float16)
    def after(child, unused, output):
        child.to(device="cpu", dtype=torch.float16)
    for _, child in model.named_children():
        for block in (child if isinstance(child, torch.nn.ModuleList) else [child]):
            block.register_forward_pre_hook(before)
            block.register_forward_hook(after)
    latent = torch.from_numpy(noise.copy()).to("cuda")
    save(captures, "noise_input", latent)
    condition, mask = torch.zeros_like(latent), torch.zeros_like(latent[:, :1])
    condition[:, :, :1] = conditioning["vae_encoded"].to("cuda")
    mask[:, :, 0] = 1
    vision, text, glyph = [conditioning[n].to("cuda") for n in ("siglip_hidden", "qwen_hidden", "byt5_hidden")]
    glyph_mask = torch.full(glyph.shape[:2], int(bool(torch.any(glyph))), dtype=torch.int64, device="cuda")
    steps, shift, cfg = (12, 7, 1) if generation["preset"] == "fast12" else (50, 5, 6)
    negative = conditioning["qwen_negative_hidden"].to("cuda") if cfg != 1 else None
    scheduler = FlowMatchDiscreteScheduler(shift=shift, reverse=True, solver="euler")
    scheduler.set_timesteps(steps, device="cuda")
    times = np.float32(1) - np.arange(steps + 1, dtype=np.float32) / np.float32(steps)
    sigmas = np.float32(shift) * times / (np.float32(1) + np.float32(shift - 1) * times)
    if not np.allclose(sigmas, scheduler.sigmas.cpu().numpy(), atol=2.e-7, rtol=0):
        raise ValueError("benchmark schedule differs from official recipe")
    scheduler.sigmas = torch.from_numpy(sigmas)
    scheduler.timesteps = (scheduler.sigmas[:-1] * 1000).to("cuda")
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - start
    durations = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for step in range(measured_steps):
            tick = time.monotonic()
            t = scheduler.timesteps[step:step + 1]
            next_t = torch.tensor([float(sigmas[step + 1] * np.float32(1000))], device="cuda")
            x = torch.cat((latent, condition, mask), dim=1)
            def predict(encoded, glyphs, glyphs_mask):
                return model(x, t, encoded, None, torch.ones(encoded.shape[:2], dtype=torch.int64, device="cuda"),
                             timestep_r=next_t if steps == 12 else None, vision_states=vision, mask_type="i2v",
                             extra_kwargs={"byt5_text_states": glyphs, "byt5_text_mask": glyphs_mask}, return_dict=False)[0]
            positive = predict(text, glyph, glyph_mask)
            if step == 0:
                save(captures, "dit_first", positive)
            if cfg != 1:
                uncond = predict(negative, torch.zeros_like(glyph), torch.zeros_like(glyph_mask))
                positive = uncond + cfg * (positive - uncond)
            latent = scheduler.step(positive, t[0], latent, return_dict=False)[0]
            save(captures, f"latent_step_{step}", latent)
            torch.cuda.synchronize()
            durations.append(time.monotonic() - tick)
            progress(durations, load_seconds)
    save(captures, "latent_final", latent)
    return latent.detach().float().cpu(), load_seconds, durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "out", "fast-native-run", "fast-native-captures", "quality-native-run", "quality-native-captures"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=ROOT / "tmp/hunyuan-video15-upstream")
    parser.add_argument("--quality-measure-steps", type=int, default=4)
    parser.add_argument("--suspend-native-pid", type=int)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if not 2 <= args.quality_measure_steps <= 50:
        parser.error("quality measurement needs 2..50 steps")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("benchmark output must be new or empty")
    args.out.mkdir(parents=True, exist_ok=True)
    scratch = args.out / "scratch"
    scratch.mkdir()
    os.environ.update(TMPDIR=str(scratch), TORCHINDUCTOR_CACHE_DIR=str(scratch / "inductor"),
                      TRITON_CACHE_DIR=str(scratch / "triton"), CUDA_CACHE_PATH=str(scratch / "cuda"))
    cancel = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda unused, unused_frame: cancel.set())
    report = dict(schema="hv15n.reference_performance.v1", status="running", pid=os.getpid(), profiles={},
                  encoder_dtype="FP32 CPU", dit_vae_dtype="FP16 CUDA", attention="official torch FlexAttention",
                  offload="complete transformer block", cpu_threads=args.threads,
                  timing_policy="wall clock with CUDA synchronization; model load, compilation and captures included",
                  benchmark_sha256=digest(Path(__file__)), verifier_sha256=digest(ROOT / "ref/hunyuan_video15_native/verify.py"))
    def write():
        atomic_json(args.out / "performance.json", report)
    write()
    try:
        with gpu_reservation(args.suspend_native_pid, args.quality_native_run, report, write, cancel):
            import numpy as np
            import torch
            from PIL import Image
            from ref.hunyuan_video15_native import verify
            torch.set_num_threads(args.threads)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
            upstream = verify.source(args.upstream)
            report.update(upstream_revision=verify.UPSTREAM, torch_version=torch.__version__,
                          gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                                       "--format=csv,noheader"], text=True).strip())
            for preset in ("fast12", "quality"):
                if cancel.is_set():
                    raise Cancelled("cancelled benchmark")
                run = args.fast_native_run if preset == "fast12" else args.quality_native_run
                native = args.fast_native_captures if preset == "fast12" else args.quality_native_captures
                baseline = json.loads((args.fast_native_run / "manifest.json").read_text())
                generation = dict(task="i2v", preset=preset, prompt=baseline["prompt"],
                                  negative_prompt=baseline["negative_prompt"])
                manifest, receipts = model_manifest(args.model, "i2v", preset)
                verify.validate_configs(args.model, manifest, receipts, generation)
                if baseline.get("preset") != "fast12" or baseline.get("model") != manifest:
                    raise ValueError("native baseline profile or pinned model differs")
                if digest(run / "input.png") != digest(args.fast_native_run / "input.png"):
                    raise ValueError("native profile portraits differ")
                profile_dir = args.out / preset
                profile_dir.mkdir()
                captures = profile_dir / "reference"
                captures.mkdir()
                raw = native / "noise_input.f32"
                info = json.loads(raw.with_suffix(".json").read_text())
                if info.get("shape") != [1, 32, 21, 53, 30] or info.get("dtype") != "float32":
                    raise ValueError("benchmark requires full 81-frame native noise")
                noise = np.fromfile(raw, dtype="<f4").reshape(info["shape"])
                if not np.isfinite(noise).all():
                    raise ValueError("nonfinite input noise")
                if digest(raw) != digest(args.fast_native_captures / "noise_input.f32"):
                    raise ValueError("native profile input noises differ")
                profile = dict(native=native_timing(run, native), stage_seconds={},
                               measured_steps=12 if preset == "fast12" else args.quality_measure_steps,
                               prescribed_steps=12 if preset == "fast12" else 50, cfg=1 if preset == "fast12" else 6,
                               flow_shift=7 if preset == "fast12" else 5, frames=81, height=848, width=480,
                               prompt=generation["prompt"], image_sha256=digest(run / "input.png"),
                               noise_sha256=digest(raw), weights={k: v["sha256"] for k, v in receipts.items()})
                report["profiles"][preset] = profile
                sampler = MemorySampler()
                sampler.start(os.getpid())
                total_start = time.monotonic()
                def stage(name, action):
                    if cancel.is_set():
                        raise Cancelled("cancelled benchmark")
                    report.update(profile=preset, stage=name)
                    write()
                    print(f"BENCH_STAGE {preset} {name}", flush=True)
                    if torch.cuda.is_initialized():
                        torch.cuda.synchronize()
                    tick = time.monotonic()
                    value = action()
                    if torch.cuda.is_initialized():
                        torch.cuda.synchronize()
                    profile["stage_seconds"][name] = time.monotonic() - tick
                    write()
                    return value
                conditioning = {}
                try:
                    for name in ("qwen_hidden", "qwen_negative_hidden", "byt5_hidden", "siglip_hidden"):
                        if name == "qwen_negative_hidden" and preset == "fast12":
                            continue
                        action = (lambda: verify.qwen(args.model, manifest, generation["prompt"])) if name == "qwen_hidden" else (
                            (lambda: verify.qwen(args.model, manifest, generation["negative_prompt"])) if name == "qwen_negative_hidden" else (
                            (lambda: verify.byt5(args.model, manifest, generation["prompt"])) if name == "byt5_hidden" else
                            (lambda: verify.siglip(args.model, manifest, run / "input.png", run / "vision_pixels.f32"))))
                        conditioning[name] = stage(name, action)
                        verify.save(captures, name, conditioning[name])
                    def encode():
                        model = verify.vae_model(args.model, manifest, upstream)
                        with Image.open(run / "input.png") as image:
                            x = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)[None, :, None] / np.float32(255)).to("cuda")
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                            return (model.encode(x * 2 - 1).latent_dist.mode() * model.scaling_factor).float().cpu()
                    conditioning["vae_encoded"] = stage("vae_encode", encode)
                    verify.save(captures, "vae_encoded", conditioning["vae_encoded"])
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    def progress(durations, load_seconds):
                        profile.update(completed_steps=len(durations), denoiser_load_seconds=load_seconds,
                                       step_seconds=list(durations), first_step_seconds=durations[0],
                                       warm_step_seconds=statistics_seconds(durations[1:]) if len(durations) > 1 else None)
                        write()
                        print(f"BENCH_STEP {preset} {len(durations)} {profile['measured_steps']} {durations[-1]:.6f}", flush=True)
                    latent, _, durations = stage("denoise", lambda: denoise(args.model, manifest, generation,
                        conditioning, noise, captures, profile["measured_steps"], cancel, progress))
                    profile.update(peak_denoise_torch_allocated_mib=torch.cuda.max_memory_allocated() / 1048576,
                                   peak_denoise_torch_reserved_mib=torch.cuda.max_memory_reserved() / 1048576)
                    warm = statistics_seconds(durations[1:])
                    profile["warm_native_over_reference"] = profile["native"]["warm_step_seconds"]["mean"] / warm["mean"]
                    if preset == "fast12":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        def decode():
                            model = verify.vae_model(args.model, manifest, upstream)
                            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                                return model.decode(latent.to("cuda") / model.scaling_factor).sample.float().cpu()
                        decoded = stage("vae_decode", decode)
                        verify.save(captures, "vae_decoded", decoded)
                        profile.update(peak_decode_torch_allocated_mib=torch.cuda.max_memory_allocated() / 1048576,
                                       peak_decode_torch_reserved_mib=torch.cuda.max_memory_reserved() / 1048576)
                        def package():
                            frames = profile_dir / "frames"
                            frames.mkdir()
                            pixels = np.rint((decoded.numpy().clip(-1, 1) + np.float32(1)) * np.float32(127.5)).clip(0, 255).astype(np.uint8)
                            for i in range(81):
                                Image.fromarray(pixels[0, :, i].transpose(1, 2, 0)).save(frames / f"frame_{i:05d}.ppm")
                            package_frames(frames, profile_dir, cancel=cancel)
                        stage("rgb_and_mp4", package)
                        profile["generation_and_packaging_seconds"] = time.monotonic() - total_start
                        profile["total_native_over_reference"] = profile["native"]["generation_and_packaging_seconds"] / profile["generation_and_packaging_seconds"]
                        # Recheck numerical correctness outside the performance interval.
                        results = verify.compare(captures, native, verify.pipeline_names(baseline))
                        frames = verify.compare_frames(captures, native)
                        profile["parity_pass"] = all(v["pass"] for v in results.values()) and len(frames) == 81 and all(v["pass"] for v in frames)
                        atomic_json(profile_dir / "parity.json", dict(results=results, decoded_frames=frames, pass_all=profile["parity_pass"]))
                        if not profile["parity_pass"]:
                            raise ValueError("benchmark full fast12 output failed numerical parity")
                    else:
                        profile.update(scope="quality_schedule_prefix_throughput",
                                       projected_50_warm_steps_seconds=50 * warm["mean"],
                                       full_quality_generation_measured=False)
                    write()
                finally:
                    sampler.close()
                    profile.update(sampled_peak_process_vram_mib=sampler.vram, sampled_peak_host_rss_mib=sampler.rss)
                    write()
                del conditioning
                torch.cuda.empty_cache()
            report.update(status="complete")
            write()
    except BaseException as error:
        report.update(status="cancelled" if isinstance(error, Cancelled) else "failed", error=str(error))
        write()
        raise


if __name__ == "__main__":
    main()
