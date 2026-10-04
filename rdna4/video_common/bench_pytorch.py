"""Measure upstream ComfyUI generation on ROCm, without a server or custom nodes.

Use an isolated ComfyUI checkout and its dependencies on PYTHONPATH. Run under
the shared video GPU lock. This benchmark does not establish numerical parity.
"""
import argparse
import gc
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("h3", "hv15"), required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--reserve-vram", type=float, default=2.0)
    parser.add_argument("--lowvram", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory already exists; choose a fresh path")
    is_h3 = args.model == "h3"
    width = args.width or (864 if is_h3 else 480)
    height = args.height or (480 if is_h3 else 848)
    requested_frames = args.frames or (360 if is_h3 else 81)
    frames = requested_frames
    if is_h3:
        while frames % 17 != 5:
            frames += 1
    steps = args.steps or (20 if is_h3 else 12)
    if args.smoke:
        width = height = 64
        frames = 5
        steps = 1
    multiple = 32 if is_h3 else 16
    if min(width, height, frames, steps) <= 0 or width % multiple or height % multiple:
        parser.error(f"positive dimensions divisible by {multiple}, frames and steps required")
    if not is_h3 and (frames - 1) % 4:
        parser.error("Hunyuan frames must be 4*n+1")
    if not is_h3 and args.image is None:
        parser.error("Hunyuan fast12 requires --image")
    weights = args.weights or Path("/mnt/disk01/models/h3/weights" if is_h3 else "/mnt/disk01/models/hv15")
    prompt = args.prompt or ("A red ball rolling on a wooden table, cinematic lighting." if is_h3 else "A person smiles naturally.")
    # Give upstream only its own explicit CLI arguments.
    sys.argv = [sys.argv[0], "--use-pytorch-cross-attention", "--reserve-vram",
                str(args.reserve_vram), "--disable-all-custom-nodes"]
    if args.lowvram:
        sys.argv += ["--lowvram", "--disable-dynamic-vram"]
    logging.basicConfig(level=logging.INFO)
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import numpy as np
    import comfy.sd as sd
    import comfy.utils as utils
    import comfy.sample as sample
    import comfy.model_management as management
    import comfy.samplers as samplers
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("a GPU-enabled ROCm PyTorch installation is required")
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    checkout = Path(sd.__file__).resolve().parents[1]
    report = {
        "scope": "upstream_comfyui_rocm_generation_performance",
        "status": "running", "model": args.model, "weights": str(weights.resolve()),
        "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(), "comfy_revision": subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "width": width, "height": height, "requested_frames": requested_frames,
        "frames": frames, "fps": 24, "video_seconds": frames / 24,
        "steps": steps, "seed": args.seed, "prompt": prompt,
        "cfg": 1, "attention": "pytorch_sdpa", "custom_nodes": False,
        "cache_acceleration": False, "audio_decode": False,
        "reserve_vram_gib": args.reserve_vram, "smoke": args.smoke,
        "lowvram": args.lowvram, "encoder_residency": "released_after_conditioning",
        "stage_seconds": {}, "step_seconds": [], "checkpoints": {},
    }
    def publish():
        partial = args.out / "metrics.json.partial"
        partial.write_text(json.dumps(report, indent=2) + "\n")
        partial.replace(args.out / "metrics.json")

    def checkpoint(path):
        path = weights / path
        report["checkpoints"][str(path)] = {"bytes": path.stat().st_size}
        return str(path)

    def timed(name, operation):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = operation()
        torch.cuda.synchronize()
        report["stage_seconds"][name] = time.perf_counter() - started
        publish()
        print("STAGE", name, report["stage_seconds"][name], flush=True)
        return result

    total_started = time.perf_counter()
    publish()
    try:
        with torch.inference_mode():
            if is_h3:
                from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift
                clip = timed("text_encoder_load", lambda: sd.load_clip(
                    [checkpoint("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")],
                    clip_type=sd.CLIPType.MINIMAX))
                positive, latent = timed("conditioning", lambda: tuple(MiniMaxH3ReferenceToVideo.execute(
                    clip, prompt, width, height, frames)))
                management.unload_all_models()
                del clip
                gc.collect()
                management.cleanup_models()
                torch.cuda.empty_cache()
                negative = positive
                model = timed("dit_load", lambda: sd.load_diffusion_model(
                    checkpoint("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors")))
                model, = MiniMaxH3SigmaShift.execute(model, 12.0, 3.0)
                vae_path = "vae/minimax_h3_video_vae_fp16.safetensors"
            else:
                from comfy_extras.nodes_hunyuan import HunyuanVideo15ImageToVideo
                from comfy_extras.nodes_model_advanced import ModelSamplingSD3
                import comfy.clip_vision as vision
                from PIL import Image
                clip = timed("text_encoder_load", lambda: sd.load_clip([
                    checkpoint("split_files/text_encoders/qwen_2.5_vl_7b.safetensors"),
                    checkpoint("split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors")],
                    clip_type=sd.CLIPType.HUNYUAN_VIDEO_15))
                positive = timed("text_encode", lambda: clip.encode_from_tokens_scheduled(clip.tokenize(prompt)))
                negative = positive
                vae_path = "split_files/vae/hunyuanvideo15_vae_fp16.safetensors"
                vae = timed("vae_load", lambda: sd.VAE(sd=utils.load_torch_file(checkpoint(vae_path)), dtype=torch.float16))
                with Image.open(args.image) as image:
                    image_tensor = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)[None]
                vision_model = timed("vision_load", lambda: vision.load(checkpoint("google_siglip/vision_fp16.safetensors")))
                vision_output = timed("vision_encode", lambda: vision_model.encode_image(image_tensor))
                positive, negative, latent = timed("image_conditioning", lambda: tuple(HunyuanVideo15ImageToVideo.execute(
                    positive, negative, vae, width, height, frames, 1,
                    start_image=image_tensor, clip_vision_output=vision_output)))
                management.unload_all_models()
                del clip, vision_model
                gc.collect()
                management.cleanup_models()
                torch.cuda.empty_cache()
                model = timed("dit_load", lambda: sd.load_diffusion_model(checkpoint(
                    "split_files/diffusion_models/hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors")))
                model, = ModelSamplingSD3().patch(model, 7.0)
            noise = sample.prepare_noise(latent["samples"], args.seed)
            # Match our H3 native grid: steps here counts updates, not points.
            if is_h3:
                base = torch.linspace(1, 0, steps + 1, dtype=torch.float32)
                sigmas = 12 * base / (1 + 11 * base)
            else:
                sigmas = samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", steps)
            step_started = time.perf_counter()
            def progress(step, denoised, current, total):
                nonlocal step_started
                torch.cuda.synchronize()
                now = time.perf_counter()
                report["step_seconds"].append(now - step_started)
                step_started = now
                publish()
                print("STEP", step + 1, total, report["step_seconds"][-1], flush=True)
            samples = timed("sampling", lambda: sample.sample_custom(
                model, noise, 1, samplers.sampler_object("euler"), sigmas, positive,
                negative, latent["samples"], callback=progress, seed=args.seed))
            management.unload_all_models()
            del model
            gc.collect()
            management.cleanup_models()
            torch.cuda.empty_cache()
            if is_h3:
                video_samples = samples.unbind()[0]
                vae = timed("vae_load", lambda: sd.VAE(sd=utils.load_torch_file(checkpoint(vae_path)), dtype=torch.float16))
            else:
                video_samples = samples
            decoded = timed("video_decode", lambda: vae.decode_tiled(
                video_samples, tile_x=16, tile_y=16, overlap=4, tile_t=7, overlap_t=2))
            if not torch.isfinite(decoded).all():
                raise RuntimeError("decoded video contains nonfinite values")
            report["decoded_shape"] = list(decoded.shape)
            actual_frames = decoded.shape[1] if decoded.ndim == 5 else decoded.shape[0]
            if actual_frames != frames:
                raise RuntimeError(f"decoded {actual_frames} frames, expected {frames}")
            from PIL import Image
            first = decoded[0, 0] if decoded.ndim == 5 else decoded[0]
            Image.fromarray((first.float().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)).save(args.out / "poster.png")
            report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["generation_wall_seconds"] = time.perf_counter() - total_started
        report["peak_torch_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        report["peak_torch_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        publish()


if __name__ == "__main__":
    main()
