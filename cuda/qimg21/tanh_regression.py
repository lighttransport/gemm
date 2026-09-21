#!/usr/bin/env python3
"""Compare native tanh rounding with PyTorch for every finite BF16 input."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/qimg21-tanh-regression"))
    parser.add_argument("--native-bin", type=Path, default=Path(__file__).with_name("test_tanh_bf16"))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    values = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    values = values[np.isfinite(values)].copy().reshape(-1, 1)
    with torch.inference_mode():
        expected = torch.tanh(torch.from_numpy(values).cuda().bfloat16()).float().cpu().numpy()
    np.save(args.work_dir / "input.npy", values)
    np.save(args.work_dir / "reference.npy", expected)
    subprocess.run([str(args.native_bin.resolve()), str(args.work_dir / "input.npy"),
                    str(args.work_dir / "native.npy")], check=True)
    actual = np.load(args.work_dir / "native.npy", allow_pickle=False)
    if actual.shape != expected.shape or actual.dtype != np.float32 or not np.isfinite(actual).all():
        raise ValueError("invalid native tanh output")
    mismatches = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    result = {"torch": torch.__version__, "device": torch.cuda.get_device_name(),
              "inputs": values.size, "bit_mismatches": mismatches,
              "max_absolute_error": float(np.max(np.abs(actual - expected))),
              "passed": mismatches == 0}
    (args.work_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
