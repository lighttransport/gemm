#!/usr/bin/env python3
"""Profile actual PyTorch CUDA dispatch on saved Q/K/V (diagnostic only)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import scaled_dot_product_attention


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-dir", required=True, type=Path)
    ap.add_argument("--prefix-tokens", required=True, type=int)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    arrays = [np.load(args.stage_dir / f"{name}.npy", allow_pickle=False)
              for name in ("rope_q", "rope_k", "v")]
    if any(a.ndim != 2 or a.shape != arrays[0].shape or a.shape[1] != 4096 or
           not np.isfinite(a).all() for a in arrays):
        raise ValueError("expected matching finite [tokens,4096] Q/K/V arrays")
    prefix = args.prefix_tokens
    if not 0 < prefix < len(arrays[0]):
        raise ValueError("prefix must leave both text and image tokens")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    q, k, v = [torch.from_numpy(a).to("cuda", torch.bfloat16).reshape(
        1, -1, 32, 128).transpose(1, 2) for a in arrays]
    mask = torch.ones(prefix, prefix, device="cuda", dtype=torch.bool).tril()

    def run():
        text = scaled_dot_product_attention(q[:, :, :prefix], k[:, :, :prefix],
                                            v[:, :, :prefix], attn_mask=mask)
        image = scaled_dot_product_attention(q[:, :, prefix:], k, v)
        return torch.cat((text, image), dim=2).transpose(1, 2).flatten(2)

    with torch.inference_mode():
        run()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            output = run()
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.out_dir / "trace.json"))
    trace = json.loads((args.out_dir / "trace.json").read_text())
    kernels = [{"name": e["name"], "duration_us": e.get("dur"), "args": e.get("args", {})}
               for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    summary = dict(torch=torch.__version__, device=torch.cuda.get_device_name(),
                   prefix_tokens=prefix, image_tokens=len(arrays[0])-prefix,
                   cuda_kernels_captured=bool(kernels), kernels=kernels)
    np.save(args.out_dir / "attention.npy", output.float().cpu().numpy()[0])
    (args.out_dir / "dispatch.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if kernels else 1


if __name__ == "__main__":
    raise SystemExit(main())
