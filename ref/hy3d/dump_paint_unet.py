"""Dump Hunyuan3D-2.1 paint UNet reference activations for Phase 3 validation.

Uses the inner stock SD-2.1 UNet2DConditionModel (12-ch conv_in patched at
checkpoint load time by the wrapper) but **resets attention processors to
the vanilla diffusers AttnProcessor** so we get a pure cross-attn baseline
to validate the CUDA skeleton against. The MDA / RA / MA / DINO custom
processors come back online in Phase 4.

Inputs and output are dumped as .npy:
    {prefix}_sample.npy           [B, 12, 64, 64]   float32  noisy latent
    {prefix}_timestep.npy         [B]               int64
    {prefix}_encoder_hidden.npy   [B, 77, 1024]     float32  text embed
    {prefix}_out.npy              [B, 4, 64, 64]    float32  predicted noise

Usage:
  uv run --with torch --with diffusers --with safetensors \\
      python ref/hy3d/dump_paint_unet.py \\
      --unet /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet \\
      --outdir /tmp/hy3d_paint_unet_ref \\
      [--seed 42] [--batch 1]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="/mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet")
    ap.add_argument("--outdir", default="/tmp/hy3d_paint_unet_ref")
    ap.add_argument("--prefix", default="ref")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # Build a stock UNet2DConditionModel from config, patch conv_in 4→12,
    # then load checkpoint with the unet.* prefix stripped (those are the
    # only keys we need; mda/ma/ra/dino params are skipped).
    config_path = os.path.join(args.unet, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    unet = UNet2DConditionModel(**cfg)
    unet.conv_in = torch.nn.Conv2d(
        12, unet.conv_in.out_channels,
        kernel_size=unet.conv_in.kernel_size,
        stride=unet.conv_in.stride,
        padding=unet.conv_in.padding,
        bias=unet.conv_in.bias is not None,
    )
    ckpt = torch.load(os.path.join(args.unet, "diffusion_pytorch_model.bin"),
                       map_location="cpu", weights_only=True)
    # The wrapper's init_attention() wraps each BasicTransformerBlock so that
    # the original block lives under `transformer_blocks.{i}.transformer.*`
    # while siblings `attn_multiview / attn_refview / attn_dino / *.processor.*`
    # hold the custom-attention weights. To rebuild a stock UNet2DConditionModel
    # we strip that extra `transformer.` level and drop the custom siblings.
    inner_state = {}
    SKIP_TOKENS = ("attn_multiview", "attn_refview", "attn_dino",
                    ".processor.")
    for k, v in ckpt.items():
        if not k.startswith("unet."):
            continue
        kk = k[len("unet."):]
        if any(t in kk for t in SKIP_TOKENS):
            continue
        kk = kk.replace(".transformer.", ".")
        inner_state[kk] = v
    missing, unexpected = unet.load_state_dict(inner_state, strict=False)
    # Expected: many "unexpected" keys from the dual-stream / custom procs
    # at the wrapper level — those don't belong to the inner unet. Missing
    # should be empty (every inner unet param must be present).
    if missing:
        print(f"WARNING: {len(missing)} missing keys (first 5): {missing[:5]}",
              file=sys.stderr)
    print(f"loaded inner unet: {len(inner_state)} keys, "
          f"{len(unexpected)} wrapper-only keys ignored")

    # Reset every attn processor to vanilla AttnProcessor — this is what
    # the upstream wrapper would substitute with custom processors in
    # Phase 4. For Phase 3 we want the stock SD path.
    procs = {name: AttnProcessor() for name in unet.attn_processors.keys()}
    unet.set_attn_processor(procs)

    unet = unet.to(dtype).eval()
    if torch.cuda.is_available():
        unet = unet.to("cuda")
    device = next(unet.parameters()).device
    print(f"device={device}, n_attn_procs={len(procs)}")

    # Build inputs
    torch.manual_seed(args.seed)
    B = args.batch
    sample = torch.randn(B, 12, 64, 64, dtype=dtype, device=device)
    timestep = torch.tensor([500] * B, dtype=torch.int64, device=device)
    text = torch.randn(B, 77, 1024, dtype=dtype, device=device)

    # Hook intermediates for piece-wise CUDA validation.
    intermediates = {}
    def grab(name):
        def hook(_mod, _inp, output):
            t = output
            if isinstance(t, tuple):
                t = t[0]
            if hasattr(t, "sample"):
                t = t.sample
            intermediates[name] = t.detach()
        return hook
    unet.conv_in.register_forward_hook(grab("conv_in"))
    unet.time_embedding.register_forward_hook(grab("time_emb"))
    # First down-block ResNet output and first attentions block output too.
    unet.down_blocks[0].resnets[0].register_forward_hook(grab("db0_res0"))
    unet.down_blocks[0].attentions[0].register_forward_hook(grab("db0_attn0"))

    with torch.no_grad():
        out = unet(sample, timestep, encoder_hidden_states=text).sample

    print(f"sample {tuple(sample.shape)}  range=[{sample.min():+.3f},{sample.max():+.3f}]")
    print(f"out    {tuple(out.shape)}  range=[{out.min():+.3f},{out.max():+.3f}]")

    os.makedirs(args.outdir, exist_ok=True)
    def save(name, t, keep_int=False):
        path = os.path.join(args.outdir, f"{args.prefix}_{name}.npy")
        arr = t.detach().cpu().numpy()
        if not keep_int:
            arr = arr.astype(np.float32)
        np.save(path, arr)
    save("sample", sample)
    save("timestep", timestep, keep_int=True)
    save("encoder_hidden", text)
    save("out", out)
    for name, t in intermediates.items():
        save(name, t)
    print(f"wrote {4 + len(intermediates)} .npy files to {args.outdir}")


if __name__ == "__main__":
    main()
