"""Wan 2.2 TI2V-5B Q8_0 generation with PyTorch ROCm and repository HIP GEMM."""
import argparse
import importlib.util
import json
import os
import hashlib
from pathlib import Path
import time

from hip_runner import HipRunner, ROOT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/mnt/disk01/models/wan22"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--image", type=Path, help="Optional first frame for image-to-video")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--backend", choices=("hip", "pytorch"), default="hip")
    parser.add_argument("--vram-budget-mib", type=int, default=14336)
    parser.add_argument("--text-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--dump-latents", action="store_true")
    args = parser.parse_args()
    if args.width < 64 or args.height < 64 or args.width % 32 or args.height % 32:
        parser.error("width and height must be multiples of 32 and at least 64")
    if args.frames < 1 or (args.frames - 1) % 4 or args.steps < 1 or args.fps < 1:
        parser.error("frames must be 4*n+1; steps and fps must be positive")
    if not 1024 <= args.vram_budget_mib <= 14336:
        parser.error("VRAM budget must be between 1024 and 14336 MiB")
    if args.out.exists():
        parser.error("output directory must not exist")
    return args


def generate(args):
    import torch
    from diffusers import (AutoencoderKLWan, GGUFQuantizationConfig, WanPipeline,
                           WanImageToVideoPipeline, WanTransformer3DModel)
    from diffusers.utils import export_to_video, load_image
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("Use the ROCm environment via run.sh; AMD GPU access is required")
    spec = importlib.util.spec_from_file_location("video_tools", ROOT / "cuda/hunyuan_video15_native/generate.py")
    video = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(video)
    model = args.model.resolve()
    pipeline_dir = model / "pipeline"
    checkpoint = model / "gguf/Wan2.2-TI2V-5B-Q8_0.gguf"
    if not checkpoint.is_file() or not (pipeline_dir / "model_index.json").is_file():
        raise ValueError("Missing weights; run download.py first")
    with video.device_lock(0):
        torch.cuda.set_device(0)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(args.vram_budget_mib * 1048576 / total, .9))
        sampler = video.MemorySampler("rocm")
        sampler.start(os.getpid())
        started = time.monotonic()
        args.out.mkdir(parents=True)
        try:
            transformer = WanTransformer3DModel.from_single_file(
                str(checkpoint), config=str(pipeline_dir), subfolder="transformer",
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
                torch_dtype=torch.float16, local_files_only=True)
            runner = HipRunner() if args.backend == "hip" else None
            count = runner.install(transformer) if runner else 0
            vae = AutoencoderKLWan.from_pretrained(pipeline_dir, subfolder="vae",
                                                   torch_dtype=torch.float32, local_files_only=True)
            cls = WanImageToVideoPipeline if args.image else WanPipeline
            pipe = cls.from_pretrained(pipeline_dir, transformer=transformer, vae=vae,
                                       torch_dtype=torch.bfloat16, local_files_only=True)
            # CPU UMT5 avoids the ~11 GiB text encoder displacing the DiT.
            # Precompute both branches before attaching component offload hooks.
            pipe.text_encoder.to("cpu" if args.text_device == "cpu" else "cuda")
            with torch.inference_mode():
                embeds, negative = pipe.encode_prompt(
                    args.prompt, negative_prompt=args.negative_prompt,
                    do_classifier_free_guidance=args.guidance_scale > 1,
                    device=args.text_device if args.text_device == "cpu" else "cuda",
                    dtype=torch.float16)
            pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            embeds = embeds.to("cuda", dtype=torch.float16)
            if negative is not None:
                negative = negative.to("cuda", dtype=torch.float16)
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
            def callback(pipeline, step, timestep, values):
                latents = values["latents"]
                if not torch.isfinite(latents).all():
                    raise RuntimeError(f"Nonfinite latent at update {step}")
                if sampler.vram is not None and sampler.vram > args.vram_budget_mib:
                    raise RuntimeError("Sampled process VRAM exceeds requested budget")
                if args.dump_latents:
                    import numpy as np
                    np.save(args.out / f"latent_{step:03d}.npy", latents.float().cpu().numpy())
                print(f"update {step + 1}/{args.steps}", flush=True)
                return values
            kwargs = dict(prompt_embeds=embeds, negative_prompt_embeds=negative,
                          width=args.width, height=args.height, num_frames=args.frames,
                          num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                          generator=torch.Generator(device="cpu").manual_seed(args.seed),
                          callback_on_step_end=callback)
            if args.image:
                kwargs["image"] = load_image(str(args.image)).convert("RGB")
                pipe.register_to_config(expand_timesteps=True)
            with torch.inference_mode():
                frames = pipe(**kwargs).frames[0]
            import numpy as np
            if len(frames) != args.frames or not np.isfinite(frames).all():
                raise RuntimeError("Invalid decoded frames")
            export_to_video(frames, str(args.out / "video.mp4"), fps=args.fps)
            torch.cuda.synchronize()
            sampler.close()
            if runner and runner.calls == 0:
                raise RuntimeError("HIP backend did not execute any quantized projections")
            if sampler.vram is not None and sampler.vram > args.vram_budget_mib:
                raise RuntimeError("Sampled process VRAM exceeds requested budget")
            manifest = {"backend": f"wan22_rocm_{args.backend}", "model": str(model),
                        "quantization": "Q8_0 weight-only INT8, FP16 compute",
                        "parity": "unverified", "prompt": args.prompt,
                        "negative_prompt": args.negative_prompt,
                        "width": args.width, "height": args.height, "frames": args.frames,
                        "steps": args.steps, "seed": args.seed, "fps": args.fps,
                        "guidance_scale": args.guidance_scale, "image": str(args.image) if args.image else None,
                        "torch": torch.__version__, "rocm": torch.version.hip,
                        "device": torch.cuda.get_device_name(), "hip_modules": count,
                        "hip_projection_calls": runner.calls if runner else 0,
                        "seconds": time.monotonic() - started,
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
                        "peak_process_vram_mib": sampler.vram,
                        "vram_budget_mib": args.vram_budget_mib}
            receipt = model / "download.json"
            manifest["weights"] = json.loads(receipt.read_text()) if receipt.is_file() else None
            manifest["sources"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.iterdir()
                                   if p.suffix in (".py", ".hip", ".sh") or p.name == "Makefile"}
            manifest["shared_gemm_sha256"] = hashlib.sha256(
                (ROOT / "rdna4/video_common/gemm.hip").read_bytes()).hexdigest()
            (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps(manifest, indent=2), flush=True)
        except Exception as error:
            (args.out / "failure.json").write_text(json.dumps({"error": str(error)}, indent=2) + "\n")
            raise
        finally:
            sampler.close()


if __name__ == "__main__":
    generate(parse_args())
