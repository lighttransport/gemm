#!/usr/bin/env python3
"""
inspect_ckpt.py — dump the state-dict keys / shapes / dtypes of the
sam-3d-body HF checkpoint so the C port can plan the safetensors
slicing layout.

Usage:
    python cpu/sam3d_body/inspect_ckpt.py \\
        /mnt/disk01/models/sam3d-body/dinov3/model.ckpt [--filter PREFIX]
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to .ckpt (pytorch pickle)")
    ap.add_argument("--filter", default=None,
                    help="only print keys starting with this prefix")
    ap.add_argument("--limit", type=int, default=0,
                    help="max number of keys to print (0 = all)")
    args = ap.parse_args()

    import torch
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Common wrapper shapes: {"state_dict": {...}}, {"model": {...}},
    # or the state-dict directly.
    sd = obj
    for key in ("state_dict", "model", "module"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            print(f"[inspect] unwrapped '{key}' level", file=sys.stderr)

    if not isinstance(sd, dict):
        print(f"[inspect] unexpected payload type: {type(sd)}", file=sys.stderr)
        return 1

    # Strip common wrapper prefixes (`model.`, `module.`).
    clean = {}
    for k, v in sd.items():
        kk = k
        for pfx in ("module.", "model."):
            if kk.startswith(pfx):
                kk = kk[len(pfx):]
        clean[kk] = v

    printed = 0
    by_prefix = {}
    for k, v in sorted(clean.items()):
        if args.filter and not k.startswith(args.filter):
            continue
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        dtype = v.dtype if hasattr(v, "dtype") else "?"
        print(f"  {k}  shape={shape}  dtype={dtype}")
        printed += 1
        top = k.split(".", 1)[0]
        by_prefix[top] = by_prefix.get(top, 0) + 1
        if args.limit and printed >= args.limit:
            print(f"[inspect] (truncated at {args.limit})", file=sys.stderr)
            break

    print("[inspect] summary by top-level prefix:", file=sys.stderr)
    for pfx, cnt in sorted(by_prefix.items(), key=lambda x: -x[1]):
        print(f"  {pfx:20s} {cnt} tensors", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
