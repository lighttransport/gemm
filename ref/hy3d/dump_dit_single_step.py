"""Dump DiT single-forward-pass reference outputs for CUDA verification.

Usage:
    HY3D_REPO=/path/to/Hunyuan3D-2.1 \
    uv run python dump_dit_single_step.py \
        --ckpt /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt \
        [--outdir output]

HY3D_REPO must point at a clone of https://github.com/Tencent/Hunyuan3D-2.1
(or wherever `hy3dshape/models/denoisers/hunyuandit.py` and `moe_layers.py`
live). Required because the DiT class pulls in bespoke MoE layers that we
load via direct file import to avoid transitive deps.
"""
import argparse
import importlib.util
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


HY3D_REPO = os.environ.get(
    "HY3D_REPO",
    "/mnt/disk01/models/Hunyuan3D-2.1-repo",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="output")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--hy3d-repo", type=str, default=HY3D_REPO)
    parser.add_argument("--latents", type=str, default=None)
    parser.add_argument("--context", type=str, default=None)
    parser.add_argument("--timestep", type=float, default=0.5)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = ckpt["model"]

    # Load hunyuandit.py without executing hy3dshape/__init__.py (which pulls
    # in pymeshlab, gradio, etc.). We register bare namespace packages in
    # sys.modules so relative imports `from ...utils import ...` resolve to
    # real submodules without triggering the top-level __init__.py.
    import types
    if not os.path.isdir(args.hy3d_repo):
        print(f"ERROR: --hy3d-repo not a directory: {args.hy3d_repo}")
        sys.exit(1)

    # Locate package root: try v2.1 `hy3dshape/hy3dshape` then v2.0 `hy3dgen`.
    v21_pkg = os.path.join(args.hy3d_repo, "hy3dshape")
    v20_pkg = os.path.join(args.hy3d_repo, "hy3dgen")
    if os.path.isdir(os.path.join(v21_pkg, "models", "denoisers")):
        pkg_name, pkg_root = "hy3dshape", v21_pkg
    elif os.path.isdir(os.path.join(v20_pkg, "shapegen", "models", "denoisers")):
        pkg_name, pkg_root = "hy3dgen", v20_pkg
    else:
        print(f"ERROR: cannot find hy3dshape or hy3dgen package under {args.hy3d_repo}")
        sys.exit(1)

    def _stub_pkg(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod
        return mod

    _stub_pkg(pkg_name, pkg_root)
    _stub_pkg(f"{pkg_name}.models", os.path.join(pkg_root, "models"))
    _stub_pkg(f"{pkg_name}.models.denoisers", os.path.join(pkg_root, "models", "denoisers"))

    # utils has its own __init__.py that we actually want (only needs omegaconf + torch).
    import importlib
    importlib.import_module(f"{pkg_name}.utils")

    dit_mod = importlib.import_module(f"{pkg_name}.models.denoisers.hunyuandit")

    model = dit_mod.HunYuanDiTPlain(
        input_size=4096,
        in_channels=64,
        hidden_size=2048,
        context_dim=1024,
        depth=21,
        num_heads=16,
        qk_norm=True,
        text_len=1370,
        with_decoupled_ca=False,
        use_attention_pooling=False,
        qk_norm_type="rms",
        qkv_bias=False,
        use_pos_emb=False,
        num_moe_layers=6,
        num_experts=8,
        moe_top_k=2,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model = model.float().eval()  # CPU — DiT doesn't fit on small GPUs w/ activations
    if args.latents:
        lat_np = np.load(args.latents)
        latents = torch.from_numpy(lat_np if lat_np.ndim == 3 else lat_np[None, ...]).float()
    else:
        torch.manual_seed(42)
        latents = torch.randn(1, 4096, 64)
    if args.context:
        ctx_np = np.load(args.context)
        context = torch.from_numpy(ctx_np if ctx_np.ndim == 3 else ctx_np[None, ...]).float()
    else:
        if not args.latents:
            torch.manual_seed(42)
        context = torch.randn(1, 1370, 1024)
    t = torch.full((latents.shape[0],), args.timestep)

    np.save(os.path.join(args.outdir, "dit_input_latents.npy"), latents.numpy()[0])
    np.save(os.path.join(args.outdir, "dit_input_context.npy"), context.numpy()[0])

    # Hook every block to capture its output (after the block's residual stream).
    block_outs = {}
    block0_io = {}
    hooks = []
    for i, blk in enumerate(model.blocks):
        def _make_hook(idx):
            def _hook(module, inp, out):
                block_outs[idx] = out.detach().float().cpu().numpy()[0]
            return _hook
        hooks.append(blk.register_forward_hook(_make_hook(i)))

    # Block-0 submodule traces for first-failure localization.
    blk0 = model.blocks[0]

    def _save(name, tensor):
        block0_io[name] = tensor.detach().float().cpu().numpy()[0]

    hooks.append(blk0.norm1.register_forward_hook(lambda m, inp, out: _save("b0_norm1", out)))
    hooks.append(blk0.attn1.register_forward_pre_hook(lambda m, inp: _save("b0_attn1_in", inp[0])))
    hooks.append(blk0.attn1.register_forward_hook(lambda m, inp, out: _save("b0_attn1_out", out)))
    hooks.append(blk0.norm2.register_forward_hook(lambda m, inp, out: _save("b0_norm2", out)))
    hooks.append(blk0.attn2.register_forward_pre_hook(lambda m, inp: _save("b0_attn2_in", inp[0])))
    hooks.append(blk0.attn2.register_forward_hook(lambda m, inp, out: _save("b0_attn2_out", out)))
    hooks.append(blk0.attn2.to_q.register_forward_hook(lambda m, inp, out: _save("b0_attn2_to_q", out)))
    hooks.append(blk0.attn2.to_k.register_forward_hook(lambda m, inp, out: _save("b0_attn2_to_k", out)))
    hooks.append(blk0.attn2.to_v.register_forward_hook(lambda m, inp, out: _save("b0_attn2_to_v", out)))
    hooks.append(blk0.attn2.q_norm.register_forward_hook(lambda m, inp, out: _save("b0_attn2_q_norm", out)))
    hooks.append(blk0.attn2.k_norm.register_forward_hook(lambda m, inp, out: _save("b0_attn2_k_norm", out)))
    hooks.append(blk0.attn2.out_proj.register_forward_hook(lambda m, inp, out: _save("b0_attn2_out_proj", out)))
    hooks.append(blk0.norm3.register_forward_hook(lambda m, inp, out: _save("b0_norm3", out)))
    hooks.append(blk0.mlp.register_forward_pre_hook(lambda m, inp: _save("b0_mlp_in", inp[0])))
    hooks.append(blk0.mlp.register_forward_hook(lambda m, inp, out: _save("b0_mlp_out", out)))

    with torch.no_grad():
        output = model(latents, t, {"main": context})

    for h in hooks:
        h.remove()

    out_np = output.float().cpu().numpy()[0]
    np.save(os.path.join(args.outdir, "dit_output.npy"), out_np)
    print(f"  Output: {out_np.shape}, mean={out_np.mean():.6f}, std={out_np.std():.6f}")

    # Save a few block outputs for per-block error localization.
    for i in sorted(block_outs.keys()):
        if i in (0, 5, 10, 11, 14, 15, 20):
            np.save(os.path.join(args.outdir, f"dit_block_{i}.npy"), block_outs[i])
            b = block_outs[i]
            print(f"  block_{i}: shape={b.shape} mean={b.mean():+.4f} std={b.std():.4f}")

    for name, arr in sorted(block0_io.items()):
        np.save(os.path.join(args.outdir, f"dit_{name}.npy"), arr)
        print(f"  {name}: shape={arr.shape} mean={arr.mean():+.4f} std={arr.std():.4f}")

    # Timestep embedding reference
    class Timesteps(nn.Module):
        def __init__(self, dim, max_period=10000):
            super().__init__()
            self.dim = dim
            self.max_period = max_period

        def forward(self, t):
            half = self.dim // 2
            exp = -math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32) / half
            emb = t[:, None].float() * torch.exp(exp)[None, :]
            return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    t_emb = Timesteps(2048)(torch.tensor([0.5])).numpy()[0]
    np.save(os.path.join(args.outdir, "dit_timestep_embed.npy"), t_emb)
    print(f"Saved to {args.outdir}/")


if __name__ == "__main__":
    main()
