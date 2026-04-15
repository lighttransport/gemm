"""Dump Real-ESRGAN x4 reference outputs for future CUDA-side validation.

Loads basicsr's RRDBNet arch (num_block=23, num_feat=64, num_grow_ch=32,
scale=4) + the RealESRGAN_x4plus.pth weights, runs a forward pass on a
single-view image, and writes:

  resrgan_input.npy    [1, 3, H, W]  float32, values in [0, 1]
  resrgan_output.npy   [1, 3, 4H, 4W] float32, values in [0, 1]
  resrgan_rdb0.npy     [1, 64, H, W]  first RRDB block output (for layer-
                                      by-layer diffing)

The hy3dpaint imageSuperNet wraps this via RealESRGANer.enhance(), which
internally does uint8 -> float in [0,1] -> model(x) -> (clamp + uint8).
We bypass the wrapper and call the raw nn.Module so the CUDA-side runner
can ignore the padding / tiling / half-precision logic.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default="/mnt/disk01/models/realesrgan/RealESRGAN_x4plus.pth")
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=128,
                    help="downscale input to this before super-res (keep "
                         "it small so the test is quick)")
    ap.add_argument("--outdir", default="/tmp/hy3d_resrgan")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # basicsr still imports the removed torchvision.transforms.functional_tensor.
    # Apply the same shim the upstream hy3dpaint utils use.
    import sys
    _repo = "/mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dpaint"
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    try:
        from utils.torchvision_fix import apply_fix
        apply_fix()
    except Exception as e:
        print(f"Warning: torchvision_fix: {e}")
    from basicsr.archs.rrdbnet_arch import RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=True)
    key = "params_ema" if "params_ema" in ckpt else (
        "params" if "params" in ckpt else None)
    sd = ckpt[key] if key else ckpt
    model.load_state_dict(sd, strict=True)
    model = model.float().eval()
    print(f"Loaded {args.weights} key={key!r} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    img = Image.open(args.image).convert("RGB").resize((args.size, args.size),
                                                        Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    print(f"input {tuple(x.shape)} min={x.min():.4f} max={x.max():.4f}")

    # Hook the first RRDB block for bisection
    taps = {}
    def h(name):
        def _h(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            taps[name] = t.detach().float().cpu().numpy()
        return _h
    model.body[0].register_forward_hook(h("rdb0"))

    with torch.inference_mode():
        y = model(x)
    print(f"output {tuple(y.shape)} min={y.min():.4f} max={y.max():.4f} "
          f"mean={y.mean():.4f} std={y.std():.4f}")

    np.save(os.path.join(args.outdir, "resrgan_input.npy"),  x.numpy())
    np.save(os.path.join(args.outdir, "resrgan_output.npy"), y.float().cpu().numpy())
    for k, v in taps.items():
        np.save(os.path.join(args.outdir, f"resrgan_{k}.npy"), v)
    print(f"Wrote {args.outdir}/resrgan_*.npy")


if __name__ == "__main__":
    main()
