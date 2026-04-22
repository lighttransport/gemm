"""Inspect sam3.1_multiplex.pt structure.

Usage: python inspect_ckpt.py <path-to-sam3.1_multiplex.pt>
"""
import sys
import torch

if len(sys.argv) < 2:
    sys.exit("usage: inspect_ckpt.py <sam3.1_multiplex.pt>")
PATH = sys.argv[1]

obj = torch.load(PATH, map_location="cpu", weights_only=False)
print("type:", type(obj).__name__)
if isinstance(obj, dict):
    print("top-level keys:", list(obj.keys())[:50])
    for k, v in obj.items():
        if isinstance(v, dict):
            print(f"  [{k}] dict, {len(v)} entries. sample keys:", list(v.keys())[:10])
        elif isinstance(v, torch.Tensor):
            print(f"  [{k}] tensor {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  [{k}] {type(v).__name__}: {str(v)[:200]}")

    # If there is a state_dict / model / weights key, dive in.
    for cand in ("state_dict", "model", "weights", "module"):
        if cand in obj and isinstance(obj[cand], dict):
            sd = obj[cand]
            print(f"\n=== {cand} ({len(sd)} tensors) ===")
            # show first 30 keys
            for k in list(sd.keys())[:30]:
                v = sd[k]
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)} {v.dtype}")
                else:
                    print(f"  {k}: {type(v).__name__}")
            print("  ...")
            for k in list(sd.keys())[-10:]:
                v = sd[k]
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)} {v.dtype}")
            break
