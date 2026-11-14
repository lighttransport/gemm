"""Per-MoE-layer NVFP4 error attribution on REAL praxis activations (torch, GPU).

Inputs: a dump dir of the runner (PRAXIS_DUMP_MOE=1 ... test_cuda_praxis --out DIR) holding <enc|dec>_b<N>_moe_x / _top_idx / _top_gate,
and a checkpoint dir (bf16 shards). For every MoE layer the routed-expert output (sum_k gate * expert_k(x), unscaled) is computed with
bf16 operands / fp32 accumulate (= the runner's bf16 path) for variants of the quantisation (weights, activations, scale search), and the
rel-L2 against the bf16 variant is reported. Quantiser = praxis nvfp4.py recipe (global amax/2688, e4m3 block scale clamp [2^-6,448],
reciprocal multiplies, E2M1 RNE); also the runner's chunked global (4096 rows) vs praxis per-expert activation global.
Run:  ~/.venv/bin/python -I cuda/praxis/moe_attr.py --dump tmp/praxis_attr/bf16 --ckpt $PRAXIS_ROOT/praxis-1-droid [--layers enc2,dec1]
Scratch/outputs live under tmp/ (a symlink to a large disk) only.
"""
import argparse, json, os, sys
import numpy as np
import torch
from safetensors import safe_open

dev = "cuda"
BND_UP = (0.75, 1.75, 3.5)          # ties go up (to the even mantissa code)
BND_DN = (0.25, 1.25, 2.5, 5.0)     # ties go down
GRID = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=dev)


def e2m1_rne(a):
    """|a| (already clamped to 6) -> nearest grid value, RNE."""
    idx = torch.zeros(a.shape, dtype=torch.uint8, device=a.device)
    for b in BND_UP:
        idx += (a >= b).to(torch.uint8)
    for b in BND_DN:
        idx += (a > b).to(torch.uint8)
    return GRID[idx.long()]


def fq(v, g, mode="max6", kdim=-1):
    """fake-quantise v (fp32) with 16-blocks along dim kdim; g = per-matrix fp32 global (tensor or float). returns dequantised fp32."""
    v = v.movedim(kdim, -1)
    sh = v.shape
    b = v.reshape(*sh[:-1], sh[-1] // 16, 16)
    bm = b.abs().amax(-1)
    cands = {"max6": [6.0], "4o6": [6.0, 4.0], "mse": [6.0, 5.5, 5.0, 4.5, 4.0, 3.5]}[mode]
    best = None
    for d in cands:
        t = ((bm * (1.0 / d)) / g).clamp(2.0 ** -6, 448.0).to(torch.float8_e4m3fn).float()
        rcp = (1.0 / g) / t
        q = (b * rcp[..., None]).clamp(-6.0, 6.0)
        dq = torch.sign(q) * e2m1_rne(q.abs()) * t[..., None] * (g[..., None] if torch.is_tensor(g) and g.dim() else g)
        if best is None:
            best, berr = dq, None
            if len(cands) > 1:
                berr = ((dq - b) ** 2).sum(-1)
        else:
            e = ((dq - b) ** 2).sum(-1)
            sel = e < berr
            best = torch.where(sel[..., None], dq, best)
            berr = torch.where(sel, e, berr)
    return best.reshape(sh).movedim(-1, kdim)


def bf(x):
    return x.to(torch.bfloat16)


class Ckpt:
    def __init__(self, d):
        self.d = d
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.h = {}

    def get(self, name):
        f = self.idx[name]
        if f not in self.h:
            self.h[f] = safe_open(os.path.join(self.d, f), "pt", device="cpu")
        return self.h[f].get_tensor(name)


def npy(path):
    return np.load(path)


def moe_forward(x, topi, topg, gu, dn, E, Hm, cfg):
    """x fp32 [T,C]; gu [E,C,2Hm] bf16 (or dequantised bf16), dn [E,Hm,C]; cfg: dict(ax=bool, ah=bool, ag='chunk'|'expert', amode)."""
    T, C = x.shape
    k = topi.shape[1]
    y = torch.zeros(T, k, C, device=dev, dtype=torch.float32)
    flat = topi.reshape(-1)
    order = torch.argsort(flat, stable=True)
    cnt = torch.bincount(flat, minlength=E)
    seg = torch.cumsum(cnt, 0) - cnt
    tok = order // k
    chunk = 4096
    # activation globals
    xs = x[tok]                                    # rows sorted by expert (fp32)
    if cfg.get("ax"):
        if cfg["ag"] == "chunk":
            # runner: amax over the Tc rows of each 4096-token chunk (the unsorted x), applied to every expert's rows
            gx = torch.empty(tok.numel(), device=dev)
            for t0 in range(0, T, chunk):
                gx[(tok >= t0) & (tok < t0 + chunk)] = x[t0:t0 + chunk].abs().max().clamp_min(1e-12) / 2688.0
            xs = fq_rows(xs, gx[:, None], cfg["amode"])
        else:
            gx = torch.empty(tok.numel(), device=dev)
            for e in range(E):
                s, n = int(seg[e]), int(cnt[e])
                if n:
                    gx[s:s + n] = xs[s:s + n].abs().max().clamp_min(1e-12) / 2688.0
            xs = fq_rows(xs, gx[:, None], cfg["amode"])
    xb = bf(xs)
    hs = torch.empty(tok.numel(), Hm, device=dev, dtype=torch.bfloat16)
    gus = torch.empty(tok.numel(), 2 * Hm, device=dev, dtype=torch.bfloat16)
    for e in range(E):
        s, n = int(seg[e]), int(cnt[e])
        if n:
            gus[s:s + n] = (xb[s:s + n].float() @ gu[e].float()).to(torch.bfloat16) if False else torch.matmul(xb[s:s + n], gu[e])
    a, b = gus[:, :Hm].float(), gus[:, Hm:].float()
    hh = bf((a / (1.0 + torch.exp(-a))) * b)
    if cfg.get("ah"):
        hf = hh.float()
        gh = torch.empty(tok.numel(), device=dev)
        if cfg["ag"] == "chunk":
            # runner: ONE global over all hidden rows of the chunk (rows of that chunk's tokens)
            for t0 in range(0, T, chunk):
                m = (tok >= t0) & (tok < t0 + chunk)
                gh[m] = hf[m].abs().max().clamp_min(1e-12) / 2688.0
        else:
            for e in range(E):
                s, n = int(seg[e]), int(cnt[e])
                if n:
                    gh[s:s + n] = hf[s:s + n].abs().max().clamp_min(1e-12) / 2688.0
        hh = bf(fq_rows(hf, gh[:, None], cfg["amode"]))
    ys = torch.empty(tok.numel(), C, device=dev, dtype=torch.bfloat16)
    for e in range(E):
        s, n = int(seg[e]), int(cnt[e])
        if n:
            ys[s:s + n] = torch.matmul(hh[s:s + n], dn[e])
    # combine
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=dev)
    yk = ys[inv].float().reshape(T, k, C)
    return (yk * topg[:, :, None]).sum(1)


def fq_rows(v, g, mode):
    out = torch.empty_like(v)
    for r0 in range(0, v.shape[0], 2048):
        out[r0:r0 + 2048] = fq(v[r0:r0 + 2048], g[r0:r0 + 2048], mode, kdim=-1)
    return out


def rel(a, b):
    return float(((a - b) ** 2).sum().sqrt() / (b ** 2).sum().sqrt())


def quant_bank(w, mode, kdim):
    """w [E,K,N] bf16 -> dequantised bf16, per-expert-matrix global; kdim = index of the K axis in the per-expert matrix (0 for [K,N])."""
    out = torch.empty_like(w)
    for e in range(w.shape[0]):
        m = w[e].float()
        if mode == "fp8c":   # e4m3 per output channel (scale over K)
            sc = (m.abs().amax(0, keepdim=True) / 448.0).clamp_min(1e-30)
            out[e] = bf((m / sc).to(torch.float8_e4m3fn).float() * sc)
            continue
        g = m.abs().max().clamp_min(1e-12) * (1.0 / 2688.0)
        out[e] = bf(fq(m, g, mode, kdim=kdim))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--variants", default="base")
    ap.add_argument("--max_tokens", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ck = Ckpt(a.ckpt)
    layers = []
    for t, ids in (("enc", range(2, 14)), ("dec", range(1, 6))):
        for i in ids:
            layers.append((t, i))
    if a.layers:
        want = set(a.layers.split(","))
        layers = [(t, i) for t, i in layers if f"{t}{i}" in want]
    E, Hm = 128, 1280
    res = {}
    for t, i in layers:
        pre = f"{a.dump}/{t}_b{i}_"
        x = torch.from_numpy(npy(pre + "moe_x.npy")).to(dev)
        topi = torch.from_numpy(npy(pre + "top_idx.npy")).long().to(dev)
        topg = torch.from_numpy(npy(pre + "top_gate.npy")).to(dev)
        if a.max_tokens:
            x, topi, topg = x[:a.max_tokens], topi[:a.max_tokens], topg[:a.max_tokens]
        tower = {"enc": "encoder", "dec": "decoder"}[t]
        key = f"backbone.{tower}.core.blocks.{i}.ffn.routed."
        gu = ck.get(key + "gate_up_weight").to(dev)
        dn = ck.get(key + "down_weight").to(dev)
        ref = moe_forward(x, topi, topg, gu, dn, E, Hm, {})
        row = {}
        base = dict(ag="chunk", amode="max6")
        vs = {
            "W4A16": (("max6", "max6"), {}),
            "W4A16_4o6": (("4o6", "4o6"), {}),
            "W4A16_mse": (("mse", "mse"), {}),
            "Wgu_only": (("max6", None), {}),
            "Wdn_only": ((None, "max6"), {}),
            "W16A4_chunk": ((None, None), dict(ax=1, ah=1, ag="chunk", amode="max6")),
            "W16A4_exp": ((None, None), dict(ax=1, ah=1, ag="expert", amode="max6")),
            "Ax_only_exp": ((None, None), dict(ax=1, ag="expert", amode="max6")),
            "Ah_only_exp": ((None, None), dict(ah=1, ag="expert", amode="max6")),
            "W4A4_chunk": (("max6", "max6"), dict(ax=1, ah=1, ag="chunk", amode="max6")),
            "W4A4_exp": (("max6", "max6"), dict(ax=1, ah=1, ag="expert", amode="max6")),
            "W4A4_exp_4o6": (("4o6", "4o6"), dict(ax=1, ah=1, ag="expert", amode="4o6")),
            "W4A4_exp_mse": (("mse", "mse"), dict(ax=1, ah=1, ag="expert", amode="mse")),
            "W8A16": (("fp8c", "fp8c"), {}),
            "Wdn8_only": ((None, "fp8c"), {}),
            "Wgu4mse_Wdn8_A16": (("mse", "fp8c"), {}),
            "Wgu4mse_Wdn8_A4": (("mse", "fp8c"), dict(ax=1, ah=1, ag="expert", amode="max6")),
            "Wgu4_Wdn8_A16": (("max6", "fp8c"), {}),
            "Wgu8_Wdn4mse_A16": (("fp8c", "mse"), {}),
        }
        want = a.variants.split(",") if a.variants != "base" else ["W4A16", "Wgu_only", "Wdn_only", "W16A4_chunk", "W16A4_exp", "W4A4_chunk", "W4A4_exp"]
        if a.variants == "all":
            want = list(vs)[:13]
        cache = {}
        for name in want:
            (wg, wd), cfg = vs[name]
            kg, kd = (wg, "gu"), (wd, "dn")
            gq = gu if wg is None else cache.setdefault(("gu", wg), quant_bank(gu, wg, 0))
            dq = dn if wd is None else cache.setdefault(("dn", wd), quant_bank(dn, wd, 0))
            y = moe_forward(x, topi, topg, gq, dq, E, Hm, cfg)
            row[name] = rel(y, ref)
        res[f"{t}{i}"] = row
        print(f"{t}{i} T={x.shape[0]} " + " ".join(f"{k}={v:.4f}" for k, v in row.items()), flush=True)
        del gu, dn, cache
        torch.cuda.empty_cache()
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    print("MEAN " + " ".join(f"{k}={np.mean([r[k] for r in res.values()]):.4f}" for k in next(iter(res.values()))))


if __name__ == "__main__":
    main()
