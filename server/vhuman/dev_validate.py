"""Explicit GPU checks after CPU-only setup; never invoked by setup_cuda.sh."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("edit8", "sm120"))
    parser.add_argument("--models-root", type=Path, default=Path("/mnt/disk1/models"))
    parser.add_argument("--candidate", type=Path, default=ROOT / "tmp/vhuman-ear-synth/reconstruction_contact")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 50 or args.device < 0:
        parser.error("steps must be 1..50 and device must be nonnegative")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("--out must be empty; preserve previous validation receipts")
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "scratch"
    scratch.mkdir()
    os.environ.update(TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch),
                      PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    os.chdir(ROOT)
    import torch
    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("A visible CUDA GPU and CUDA PyTorch environment are required")
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    expected = (8, 6) if args.stage == "edit8" else (12, 0)
    if capability != expected:
        raise RuntimeError(f"{args.stage} requires compute capability {expected}, detected {capability}")
    free, total = torch.cuda.mem_get_info(args.device)
    if args.stage == "edit8" and not 7 << 30 <= total <= 9 << 30:
        raise RuntimeError("edit8 records an actual 8 GiB GPU run")
    if free < (7 if args.stage == "edit8" else 12) << 30:
        raise RuntimeError("GPU is occupied; free it before running this validation")
    report = dict(stage=args.stage, started=datetime.now(timezone.utc).isoformat(),
                  gpu=properties.name, capability=list(capability), total_bytes=total,
                  free_bytes_before=free, torch=torch.__version__, cuda=torch.version.cuda,
                  steps=args.steps, seed=args.seed, success=False)
    report_path = out / "validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    started = time.monotonic()
    try:
        if args.stage == "sm120":
            configuration = json.loads((ROOT / "tmp/vhuman-dev/build.json").read_text())
            if configuration.get("status") != "complete" or configuration["gpu_arch"] != "sm_120":
                raise RuntimeError("Rebuild with setup_cuda.sh --phases build --gpu-arch sm_120 first")
            # Small independent CPU/CUDA comparison before the native suites.
            generator = torch.Generator().manual_seed(args.seed)
            a = torch.randn((128, 128), generator=generator, dtype=torch.float32).to(torch.bfloat16)
            b = torch.randn((128, 128), generator=generator, dtype=torch.float32).to(torch.bfloat16)
            reference = a.float() @ b.float()
            actual = (a.cuda() @ b.cuda()).float().cpu()
            relative = float(torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference))
            if relative > .01:
                raise RuntimeError(f"BF16 matmul relative error {relative} exceeds .01")
            report["bf16_relative_l2"] = relative
            subprocess.run(["make", "-C", "cuda/vhuman", "test-training"], check=True)
            subprocess.run([sys.executable, "-B", "-m", "server.vhuman.test_native_gpu_training",
                            "--gpu", "-v"], check=True)
            subprocess.run([sys.executable, "-B", "-m", "unittest",
                            "server.vhuman.test_native_image_training"], check=True)
        from .reconstruction import mv_texture
        work = out / "edit"
        mv_texture.prepare(args.candidate, work)
        torch.cuda.reset_peak_memory_stats(args.device)
        result = mv_texture.generate(work, "qwen_edit_seq", name="right", selected_views=["right"],
                    edit_backend="gguf", edit_model_root=args.models_root / "qwen-image-edit-2511",
                    offload_blocks=1, portrait_mode="raw", prompt_recipe="original",
                    steps=args.steps, seed=args.seed)
        torch.cuda.synchronize(args.device)
        report.update(result=result, peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                      peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(args.device), success=True)
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
