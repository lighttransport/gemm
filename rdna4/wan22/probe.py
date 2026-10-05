"""Actual-weight Wan DiT parity on shared noise, text embeddings and timesteps."""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
import torch
from diffusers import GGUFQuantizationConfig, WanTransformer3DModel
from hip_runner import HipRunner, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/mnt/disk01/models/wan22"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()
    if args.width % 32 or args.height % 32 or min(args.width, args.height) < 64 or args.frames < 1 or (args.frames - 1) % 4:
        parser.error("Dimensions must be multiples of 32; frames must be 4*n+1")
    spec = importlib.util.spec_from_file_location("video_tools", ROOT / "cuda/hunyuan_video15_native/generate.py")
    video = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(video)
    with video.device_lock(0):
        torch.cuda.set_per_process_memory_fraction(.85)
        model = WanTransformer3DModel.from_single_file(
            str(args.model / "gguf/Wan2.2-TI2V-5B-Q8_0.gguf"),
            config=str(args.model / "pipeline"), subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
            torch_dtype=torch.float16, local_files_only=True).cuda()
        torch.manual_seed(42)
        f, h, w = (args.frames - 1) // 4 + 1, args.height // 16, args.width // 16
        noise = torch.randn(1, 48, f, h, w, device="cuda", dtype=torch.float16)
        text = torch.randn(1, 32, 4096, device="cuda", dtype=torch.float16)
        timestep = torch.full((1, f * (h // 2) * (w // 2)), 500., device="cuda")
        sampler = video.MemorySampler("rocm")
        sampler.start(os.getpid())
        started = time.monotonic()
        with torch.inference_mode():
            print("Running PyTorch GGUF reference", flush=True)
            expected = model(noise, timestep, text).sample.float()
            torch.cuda.synchronize()
            print("PyTorch GGUF reference completed", flush=True)
            runner = HipRunner()
            count = runner.install(model)
            print("Running repository HIP projections", flush=True)
            actual = model(noise, timestep, text).sample.float()
        torch.cuda.synchronize()
        sampler.close()
        relative = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        result = {"hip_modules": count, "hip_calls": runner.calls,
                  "relative_l2": float(relative), "cosine": float(cosine),
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
                  "peak_process_vram_mib": sampler.vram, "seconds": time.monotonic() - started,
                  "width": args.width, "height": args.height, "frames": args.frames}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        assert torch.isfinite(actual).all()
        assert relative < .02 and cosine >= .9999, result


if __name__ == "__main__":
    main()
