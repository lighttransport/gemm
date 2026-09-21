#!/usr/bin/env python3
"""Isolate FlashAttention tile boundaries using cropped, matched model Q/K/V."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from attention_replay_compare import metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-dir", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--tokens", type=int, nargs="+", default=[32, 64, 68, 128, 132, 256])
    ap.add_argument("--native-bin", type=Path, default=Path(__file__).with_name("test_bf16_attention"))
    args = ap.parse_args()
    arrays = [np.load(args.stage_dir / f"{name}.npy", allow_pickle=False)
              for name in ("rope_q", "rope_k", "v")]
    if any(a.ndim != 2 or a.shape != arrays[0].shape or a.shape[1] != 4096 or
           a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("expected matching finite F32 [tokens,4096] inputs")
    if any(n < 4 or n % 4 or n > min(len(arrays[0]), 4096) for n in args.tokens):
        raise ValueError("token counts must be multiples of four within available inputs and <=4096")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    results = dict(diagnostic_only=True, full_model_acceptance=False, torch=torch.__version__, cases=[])
    for n in args.tokens:
        directory = args.work_dir / str(n)
        directory.mkdir()
        qkv = []
        for name, array in zip(("rope_q", "rope_k", "v"), arrays):
            tensor = torch.from_numpy(array[:n]).to("cuda", torch.bfloat16)
            np.save(directory / f"{name}.npy", tensor.float().cpu().numpy())
            qkv.append(tensor.reshape(1, n, 32, 128).transpose(1, 2))
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            reference = F.scaled_dot_product_attention(*qkv).transpose(1, 2).reshape(n,4096).float().cpu().numpy()
        np.save(directory / "reference.npy", reference)
        del tensor, qkv
        torch.cuda.empty_cache()
        # A single image with no text prefix isolates the unmasked backend.
        layout = [n//4, 0, 1, *([1]*(n//4)), 4, n//4]
        (directory / "layout.txt").write_text(" ".join(map(str, layout))+"\n")
        for mode in ("--reverse64", "--flash-softmax"):
            output = directory / f"{mode[2:]}.npy"
            with (directory / f"{mode[2:]}.log").open("w") as log:
                subprocess.run([str(args.native_bin.resolve()), str(directory), str(directory/"layout.txt"),
                                str(output), mode], stdout=log, stderr=subprocess.STDOUT, check=True)
            result = metrics(reference, np.load(output, allow_pickle=False), n)
            result.update(tokens=n, mode=mode)
            results["cases"].append(result)
            print(json.dumps(result), flush=True)
        results["passed"] = all(case["passed"] for case in results["cases"])
        (args.work_dir / "results.json").write_text(json.dumps(results, indent=2)+"\n")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
