"""Inspect a SAM-3D-Objects checkpoint (either `.ckpt` torch pickle or
`.safetensors`) or an entire HF checkpoint directory.

Usage:
    python inspect_ckpt.py <path-to-.ckpt-or-.safetensors>
    python inspect_ckpt.py <path-to-directory>

Output: tensor name, shape, dtype — first 30 + last 10 unless
`--all` is passed.

The shipped sam-3d-objects ckpts are flat state_dicts (no wrapper
key), so we unwrap `state_dict` / `model` / `module` when present but
also accept the top-level dict as the weight map.
"""
import os
import sys
import argparse

def load_ckpt(path):
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        out = {}
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out

    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for wrap in ("state_dict", "model", "module", "weights"):
            if wrap in obj and isinstance(obj[wrap], dict):
                return obj[wrap]
        return obj
    return None

def describe(path, show_all):
    sd = load_ckpt(path)
    if sd is None:
        print(f"=== {path} (not a state-dict) ===")
        return
    keys = list(sd.keys())
    print(f"=== {path}  ({len(keys)} tensors) ===")
    shown = keys if show_all else keys[:30]
    def _line(k, t):
        if hasattr(t, "shape") and hasattr(t, "dtype"):
            return f"  {k}: {tuple(t.shape)} {t.dtype}"
        return f"  {k}: {type(t).__name__}"
    for k in shown:
        print(_line(k, sd[k]))
    if not show_all and len(keys) > 30:
        print(f"  ... ({len(keys) - 30} more)")
        for k in keys[-10:]:
            print(_line(k, sd[k]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--all", action="store_true",
                    help="list every tensor, not just first 30 + last 10")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        for fn in sorted(os.listdir(args.path)):
            p = os.path.join(args.path, fn)
            if fn.endswith((".ckpt", ".safetensors")) and os.path.isfile(p):
                describe(p, args.all)
    else:
        describe(args.path, args.all)

if __name__ == "__main__":
    sys.exit(main() or 0)
