#!/usr/bin/env python3
"""Ablation: for block N, compare SVDQuant reconstruction fidelity of weight-only vs activation-aware
smoothing (a few alphas) against the published nunchaku-r128 logical checkpoint. Probes use
activation-scaled inputs (x_j ~ N(0,1)*amax_j) so the cosine reflects real-output fidelity, where
smoothing pays off. Reuses svdquant_from_bf16.py."""
import argparse, importlib.util, numpy as np, torch
from safetensors import safe_open

spec = importlib.util.spec_from_file_location("sq", __file__.replace("ablate_smoothing.py", "svdquant_from_bf16.py"))
sq = importlib.util.module_from_spec(spec); spec.loader.exec_module(sq)


def recon_eff_weight(qint4, wscale, lora_up, lora_down, group=64):  # -> effective smoothed weight What_eff
    out, kh = qint4.shape; k = kh * 2
    u = (qint4 & 0xF).to(torch.int16); u -= 16 * (u >= 8)
    hi = (qint4 >> 4).to(torch.int16); hi -= 16 * (hi >= 8)
    nib = torch.empty(out, k, dtype=torch.float32); nib[:, 0::2] = u.float(); nib[:, 1::2] = hi.float()
    W = nib * wscale.float().repeat_interleave(group, dim=1)
    if lora_up is not None:
        W = W + lora_up.float() @ lora_down.float()
    return W


def cos_on(W_ref, lam, What_eff, x):
    y_ref = x @ W_ref.t()
    y_q = (x / lam) @ What_eff.t()
    return torch.nn.functional.cosine_similarity(y_q.flatten(), y_ref.flatten(), dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", default="/mnt/disk1/models/qwen-image/_bf16dl/split_files/diffusion_models/qwen_image_bf16.safetensors")
    ap.add_argument("--calib", nargs="+", required=True)
    ap.add_argument("--nunchaku", default="/mnt/disk1/models/qwen-image/nunchaku/svdq-int4_r128-qwen-image-logical-v3.safetensors")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--alphas", default="0.3,0.5")
    ap.add_argument("--clip", type=float, default=1e3)
    ap.add_argument("--probe", type=int, default=16)
    a = ap.parse_args()
    torch.manual_seed(0)
    alphas = [float(x) for x in a.alphas.split(",")]
    calib = sq.load_calib(a.calib)
    src = safe_open(a.bf16, "pt", "cpu")
    nun = safe_open(a.nunchaku, "pt", "cpu"); nun_keys = set(nun.keys())
    b = a.block
    print(f"block {b}: cosine (activation-scaled probe) — higher=better. target=nunchaku-r128")
    hdr = f"{'linear':20s} {'no-smooth':>10s} " + " ".join(f"a={al:<8.2f}" for al in alphas) + f"{'nunchaku':>10s}"
    print(hdr)
    agg = {k: [] for k in (["nosm"] + [f"a{al}" for al in alphas] + ["nun"])}
    for suf in sq.MAIN:
        key = f"transformer_blocks.{b}.{suf}"
        W = src.get_tensor(key + ".weight").float()
        amax = calib.get(key + ".amax")
        scale = amax.clamp(min=1e-6) if amax is not None else torch.ones(W.shape[1])
        x = torch.randn(a.probe, W.shape[1]) * scale.unsqueeze(0)
        row = f"{suf:20s} "
        # no-smooth
        lam = torch.ones(W.shape[1]); What = W * lam.unsqueeze(0)
        lu, ld, R, _ = sq.svd_lowrank_branch(What, a.rank); q, ws = sq.quant_int4_g64(R, a.group)
        c = cos_on(W, lam, recon_eff_weight(q, ws, lu, ld, a.group), x); agg["nosm"].append(c); row += f"{c:10.5f} "
        # smoothed alphas
        for al in alphas:
            lam = sq.smoothing_lambda(W, amax, al, a.clip); What = W * lam.unsqueeze(0)
            lu, ld, R, _ = sq.svd_lowrank_branch(What, a.rank); q, ws = sq.quant_int4_g64(R, a.group)
            c = cos_on(W, lam, recon_eff_weight(q, ws, lu, ld, a.group), x); agg[f"a{al}"].append(c); row += f"{c:9.5f} "
        # nunchaku-r128 baseline
        if key + ".qint4" in nun_keys:
            q = nun.get_tensor(key + ".qint4"); ws = nun.get_tensor(key + ".wscale")
            lu = nun.get_tensor(key + ".lora_up") if key + ".lora_up" in nun_keys else None
            ld = nun.get_tensor(key + ".lora_down") if key + ".lora_down" in nun_keys else None
            lam = nun.get_tensor(key + ".smooth").float() if key + ".smooth" in nun_keys else torch.ones(W.shape[1])
            c = cos_on(W, lam, recon_eff_weight(q, ws, lu, ld, a.group), x); agg["nun"].append(c); row += f"{c:10.5f}"
        else:
            row += f"{'n/a':>10s}"
        print(row)
    print("-" * len(hdr))
    mrow = f"{'MEAN':20s} "
    mrow += f"{np.mean(agg['nosm']):10.5f} "
    for al in alphas: mrow += f"{np.mean(agg[f'a{al}']):9.5f} "
    mrow += f"{np.mean(agg['nun']):10.5f}" if agg["nun"] else f"{'n/a':>10s}"
    print(mrow)


if __name__ == "__main__":
    main()
