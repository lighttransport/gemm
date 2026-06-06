#!/usr/bin/env python3
"""Generate a PyTorch reference for TRELLIS.2 SC-VAE decoders.

The upstream TRELLIS.2 repo expects the FlexGEMM sparse-conv extension. This
script installs a small pure-PyTorch submanifold sparse-conv backend so tiny
shape/texture decoder parity checks can run without that extension.

Default output is suitable for CUDA texture SC-VAE verification:

  python ref/trellis2/gen_scvae_decoder_ref.py \
    --decoder /mnt/disk01/models/trellis2-4b/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors \
    --outdir ref/trellis2/dumps/tex_scvae_tiny
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_REPO = REPO_ROOT / "cpu" / "trellis2" / "trellis2_repo"
sys.path.insert(0, str(TRELLIS_REPO))


def _install_pytorch_sparse_conv_backend() -> None:
    from trellis2.modules.sparse import config as sparse_config
    import trellis2.modules.sparse.conv.conv as conv_mod
    import torch.nn as nn

    def sparse_conv3d_init(self, in_channels, out_channels, kernel_size,
                           stride=1, dilation=1, padding=None, bias=True,
                           indice_key=None):
        del indice_key
        assert stride == 1 and padding is None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
        self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
        assert self.kernel_size == (3, 3, 3)
        assert self.dilation == (1, 1, 1)
        self.weight = nn.Parameter(torch.empty((out_channels, 3, 3, 3, in_channels)))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        torch.nn.init.kaiming_uniform_(self.weight.reshape(out_channels, -1), a=5 ** 0.5)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def sparse_conv3d_forward(self, x):
        coords_np = x.coords.detach().cpu().numpy()
        index = {tuple(int(v) for v in row): i for i, row in enumerate(coords_np)}
        feats = x.feats
        out = feats.new_zeros((coords_np.shape[0], self.out_channels))
        if self.bias is not None:
            out += self.bias.to(dtype=out.dtype, device=out.device)

        for kd in range(3):
            dz = kd - 1
            for kh in range(3):
                dy = kh - 1
                for kw in range(3):
                    dx = kw - 1
                    idx = [index.get((int(b), int(z) + dz, int(y) + dy, int(xv) + dx), -1)
                           for b, z, y, xv in coords_np]
                    idx_t = torch.tensor(idx, dtype=torch.long, device=feats.device)
                    mask = idx_t >= 0
                    if not bool(mask.any()):
                        continue
                    src = feats.index_select(0, idx_t[mask])
                    w = self.weight[:, kd, kh, kw, :].to(dtype=src.dtype, device=src.device)
                    out[mask] += src @ w.t()
        return x.replace(out)

    backend = types.SimpleNamespace(
        sparse_conv3d_init=sparse_conv3d_init,
        sparse_conv3d_forward=sparse_conv3d_forward,
        sparse_inverse_conv3d_init=lambda *a, **k: (_ for _ in ()).throw(NotImplementedError()),
        sparse_inverse_conv3d_forward=lambda *a, **k: (_ for _ in ()).throw(NotImplementedError()),
    )
    conv_mod._backends["pytorch_ref"] = backend
    sparse_config.CONV = "pytorch_ref"


def _default_config_path(decoder: Path) -> Path:
    if decoder.suffix == ".safetensors":
        return decoder.with_suffix(".json")
    return Path(str(decoder) + ".json")


def _load_input(args, outdir: Path) -> tuple[np.ndarray, np.ndarray]:
    if args.slat and args.coords:
        slat = np.load(args.slat).astype(np.float32)
        coords = np.load(args.coords).astype(np.int32)
    elif args.slat or args.coords:
        raise SystemExit("--slat and --coords must be provided together")
    else:
        slat = np.linspace(-0.75, 0.75, 32, dtype=np.float32).reshape(1, 32)
        coords = np.array([[0, 0, 0, 0]], dtype=np.int32)
    np.save(outdir / f"{args.prefix}_input_slat.npy", slat)
    np.save(outdir / f"{args.prefix}_input_coords.npy", coords)
    return slat, coords


def _dense_guides(coords_np: np.ndarray, n_stages: int, device: torch.device):
    from trellis2.modules.sparse.basic import SparseTensor

    guides = []
    coords = coords_np.astype(np.int32, copy=True)
    for _ in range(n_stages):
        feats = torch.ones((coords.shape[0], 8), dtype=torch.bool, device=device)
        coords_t = torch.from_numpy(coords).to(device=device, dtype=torch.int32)
        guides.append(SparseTensor(feats=feats, coords=coords_t))

        children = []
        for row in coords:
            b, z, y, x = [int(v) for v in row]
            for s in range(8):
                # Match upstream SparseChannel2Spatial: coord dimension i gets
                # bit i of subidx, i.e. coords are treated as (b, z, y, x).
                dz = s & 1
                dy = (s >> 1) & 1
                dx = (s >> 2) & 1
                children.append((b, z * 2 + dz, y * 2 + dy, x * 2 + dx))
        coords = np.asarray(children, dtype=np.int32)
    return guides


def _save_sparse(outdir: Path, prefix: str, name: str, x) -> None:
    feats = x.feats.detach().float().cpu().numpy()
    coords = x.coords.detach().cpu().numpy().astype(np.int32)
    np.save(outdir / f"{prefix}_{name}_feats.npy", feats)
    np.save(outdir / f"{prefix}_{name}_coords.npy", coords)


def _forward_scvae_with_dumps(model, x, guides, outdir: Path, prefix: str):
    h = model.from_latent(x)
    _save_sparse(outdir, prefix, "from_latent", h)
    h = h.type(model.dtype)
    for stage, res in enumerate(model.blocks):
        for j, block in enumerate(res):
            is_c2s = stage < len(model.blocks) - 1 and j == len(res) - 1
            if is_c2s:
                tag = f"stage{stage}_c2s"
                x_in = h
                if model.pred_subdiv:
                    subdiv = block.to_subdiv(x_in)
                    _save_sparse(outdir, prefix, f"{tag}_subdiv_logits", subdiv)
                else:
                    subdiv = guides[stage] if guides is not None else None

                hh = x_in.replace(block.norm1(x_in.feats))
                hh = hh.replace(F.silu(hh.feats))
                _save_sparse(outdir, prefix, f"{tag}_pre_conv1", hh)
                hh = block.conv1(hh)
                _save_sparse(outdir, prefix, f"{tag}_post_conv1", hh)

                subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
                hh = block.updown(hh, subdiv_binarized)
                _save_sparse(outdir, prefix, f"{tag}_post_updown_h", hh)
                xx = block.updown(x_in, subdiv_binarized)
                _save_sparse(outdir, prefix, f"{tag}_post_updown_x", xx)

                hh = hh.replace(block.norm2(hh.feats))
                hh = hh.replace(F.silu(hh.feats))
                _save_sparse(outdir, prefix, f"{tag}_pre_conv2", hh)
                hh = block.conv2(hh)
                _save_sparse(outdir, prefix, f"{tag}_post_conv2", hh)
                skip = block.skip_connection(xx)
                _save_sparse(outdir, prefix, f"{tag}_skip", skip)
                h = hh + skip
                _save_sparse(outdir, prefix, f"{tag}_final", h)
            else:
                if ((stage == 0 and j == 0) or
                    (stage == 2 and j == 7) or
                    (stage == 3 and j == 3)):
                    hc = block.conv(h)
                    tag = f"stage{stage}_b{j}"
                    _save_sparse(outdir, prefix, f"{tag}_post_conv", hc)
                    hn = hc.replace(block.norm(hc.feats))
                    _save_sparse(outdir, prefix, f"{tag}_post_ln", hn)
                    h_fc1 = block.mlp[0](hn.feats)
                    _save_sparse(outdir, prefix, f"{tag}_mlp_fc1", hn.replace(h_fc1))
                    h_silu = block.mlp[1](h_fc1)
                    _save_sparse(outdir, prefix, f"{tag}_mlp_silu", hn.replace(h_silu))
                    hm = hn.replace(block.mlp[2](h_silu))
                    _save_sparse(outdir, prefix, f"{tag}_post_mlp", hm)
                h = block(h)
                _save_sparse(outdir, prefix, f"stage{stage}_block{j}", h)
        _save_sparse(outdir, prefix, f"stage{stage}", h)
    h = h.type(x.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    _save_sparse(outdir, prefix, "pre_output", h)
    h = model.output_layer(h)
    _save_sparse(outdir, prefix, "output", h)
    return h


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True, help="SC-VAE decoder .safetensors")
    ap.add_argument("--config", help="Decoder config JSON. Defaults beside --decoder.")
    ap.add_argument("--slat", help="Input denormalized latent [N,32] .npy")
    ap.add_argument("--coords", help="Input sparse coords [N,4] .npy")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "ref" / "trellis2" / "dumps" / "scvae_ref"))
    ap.add_argument("--prefix", default="ref")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "config"],
                    help="fp32 overrides config use_fp16=false; config uses the JSON dtype.")
    ap.add_argument("--dump-stages", action="store_true",
                    help="Dump from_latent, post-stage, pre_output, and output sparse tensors.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    decoder = Path(args.decoder)
    config_path = Path(args.config) if args.config else _default_config_path(decoder)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if args.dtype == "fp32" and "use_fp16" in cfg.get("args", {}):
        cfg["args"]["use_fp16"] = False

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")

    _install_pytorch_sparse_conv_backend()
    from trellis2.modules.sparse.basic import SparseTensor
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder

    if cfg["name"] == "FlexiDualGridVaeDecoder":
        # The upstream FDG wrapper imports mesh/cumesh code even though raw
        # decoder logits do not need it. Instantiate the base SC-VAE decoder
        # directly with the same weights and architecture.
        dec_args = dict(cfg["args"])
        dec_args.pop("resolution", None)
        dec_args.pop("voxel_margin", None)
        model = SparseUnetVaeDecoder(out_channels=7, **dec_args)
    else:
        import trellis2.models as models
        model_cls = getattr(models, cfg["name"])
        model = model_cls(**cfg["args"])
    missing, unexpected = model.load_state_dict(load_file(str(decoder)), strict=False)
    if missing or unexpected:
        print(f"[load] missing={missing} unexpected={unexpected}", file=sys.stderr)
    model.eval().to(device)

    slat_np, coords_np = _load_input(args, outdir)
    feats = torch.from_numpy(slat_np).to(device=device, dtype=torch.float32)
    coords = torch.from_numpy(coords_np).to(device=device, dtype=torch.int32)
    x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([1, 32, 32, 32]))

    pred_subdiv = bool(getattr(model, "pred_subdiv", False))
    n_stages = max(0, len(getattr(model, "blocks", [])) - 1)
    guides = None if pred_subdiv else _dense_guides(coords_np, n_stages, device)

    t0 = time.time()
    if args.dump_stages:
        out = _forward_scvae_with_dumps(model, x, guides, outdir, args.prefix)
    else:
        out = model(x, guide_subs=guides)
        if isinstance(out, tuple):
            out = out[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000.0

    out_feats = out.feats.detach().float().cpu().numpy()
    out_coords = out.coords.detach().cpu().numpy().astype(np.int32)
    np.save(outdir / f"{args.prefix}_output_feats.npy", out_feats)
    np.save(outdir / f"{args.prefix}_output_coords.npy", out_coords)
    with open(outdir / f"{args.prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "decoder": str(decoder),
            "config": str(config_path),
            "model": cfg["name"],
            "dtype": args.dtype,
            "device": str(device),
            "input_shape": list(slat_np.shape),
            "output_shape": list(out_feats.shape),
            "elapsed_ms": dt,
            "pred_subdiv": pred_subdiv,
            "dense_guides": guides is not None,
        }, f, indent=2)

    print(f"[scvae-ref] input slat={slat_np.shape} coords={coords_np.shape}")
    print(f"[scvae-ref] output feats={out_feats.shape} coords={out_coords.shape} "
          f"device={device} dtype={args.dtype} time={dt:.1f} ms")
    print(f"[scvae-ref] wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
