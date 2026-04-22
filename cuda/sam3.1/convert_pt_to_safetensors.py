"""Convert sam3.1_multiplex.pt (flat state_dict) to .safetensors.

The CUDA sam3 loader uses safetensors (see cpu/sam3/, common/safetensors.h).
sam3.1 multiplex has two top-level prefixes:
  - detector.*  : image detector (sam3-compatible structure)
  - tracker.*   : video/object tracker (new in 3.1)
"""
import argparse
import os
import torch
from safetensors.torch import save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="/mnt/nvme02/models/sam3.1/sam3.1_multiplex.pt")
    ap.add_argument("--out", default="/mnt/nvme02/models/sam3.1/sam3.1.model.safetensors")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--split", action="store_true",
                    help="Also emit detector.safetensors and tracker.safetensors")
    args = ap.parse_args()

    print(f"loading {args.pt}")
    sd = torch.load(args.pt, map_location="cpu", weights_only=False)
    assert isinstance(sd, dict), f"expected dict, got {type(sd)}"

    cast = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    out = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            print(f"  skip non-tensor: {k}")
            continue
        t = v.detach().contiguous()
        if t.is_floating_point() and t.dtype != cast:
            t = t.to(cast)
        out[k] = t

    print(f"tensors: {len(out)}  dtype: {args.dtype}")
    if args.split:
        det = {k: v for k, v in out.items() if k.startswith("detector.")}
        trk = {k: v for k, v in out.items() if k.startswith("tracker.")}
        other = {k: v for k, v in out.items()
                 if not (k.startswith("detector.") or k.startswith("tracker."))}
        base = os.path.splitext(args.out)[0]
        if det:
            p = base + ".detector.safetensors"
            save_file(det, p); print(f"wrote {p}  ({len(det)} tensors)")
        if trk:
            p = base + ".tracker.safetensors"
            save_file(trk, p); print(f"wrote {p}  ({len(trk)} tensors)")
        if other:
            p = base + ".misc.safetensors"
            save_file(other, p); print(f"wrote {p}  ({len(other)} tensors)")
    else:
        save_file(out, args.out)
        print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
