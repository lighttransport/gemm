from __future__ import annotations

import hashlib
import numpy as np
from .native import unpack, pack, decode


def refine(codec, weight, qtype, inputs, validation, config, identity, device):
    """Optimize legal K-quant fields, selecting by hard packed reconstruction loss."""
    import torch
    import torch.nn.functional as F
    if not np.isfinite(weight).all() or not np.isfinite(inputs).all() or not np.isfinite(validation).all():
        raise ValueError("GSQ weights and activations must be finite")
    w = torch.as_tensor(weight, dtype=torch.float32, device=device)
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    val = torch.as_tensor(validation, dtype=torch.float32, device=device)
    if len(x) == 0 or len(val) == 0:
        raise ValueError("GSQ needs non-empty train and validation activations")
    initial = codec.encode(weight, qtype, np.mean(np.square(inputs), axis=0))
    q, scales, d, mins, dmin = unpack(initial, qtype)
    lo, hi = {"Q2_K": (0, 3), "Q3_K": (-4, 3), "Q4_K": (0, 15), "Q6_K": (-32, 31), "Q8_0": (-128, 127)}[qtype]
    seed = config["seed"] + int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if str(device).startswith("cuda") else []):
        torch.manual_seed(seed)
        qc = torch.as_tensor(q, dtype=torch.float32, device=device)
        if qtype == "Q2_K":
            grid = torch.arange(4, device=device, dtype=torch.float32).expand(*qc.shape, 4)
        else:
            grid = (qc[..., None] + torch.arange(-2, 3, device=device)).clamp(lo, hi)
        bias = config.get("gsq_initial_bias", 2*config["gsq_lr"])
        logits = torch.nn.Parameter(torch.where(grid == qc[..., None], bias, -bias).float())
        s = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=device))
        ds = torch.nn.Parameter(torch.tensor(d, dtype=torch.float32, device=device))
        ms = torch.nn.Parameter(torch.tensor(mins, dtype=torch.float32, device=device)) if mins is not None else None
        dm = torch.nn.Parameter(torch.tensor(dmin, dtype=torch.float32, device=device)) if dmin is not None else None
        optimizer = torch.optim.Adam([
            {"params": [logits], "lr": config["gsq_lr"]},
            {"params": [s]+([] if ms is None else [ms]), "lr": config.get("gsq_integer_lr", 0.1)},
            {"params": [ds]+([] if dm is None else [dm]), "lr": config["gsq_scale_lr"]},
        ])

        def rounded(t, low, high):
            bounded = t.clamp(low, high)
            return bounded + (bounded.round()-bounded).detach()

        def native_scales():
            lower, upper = {"Q2_K": (0, 15), "Q3_K": (-32, 31), "Q4_K": (0, 63), "Q6_K": (-128, 127), "Q8_0": (1, 1)}[qtype]
            sd = rounded(s, lower, upper)
            dd = ds.clamp(-65504, 65504)
            dd = dd + (dd.to(torch.float16).float()-dd).detach()
            if ms is None:
                return sd, dd, None, None
            mm = rounded(ms, 0, upper)
            md = dm.clamp(0, 65504)
            md = md + (md.to(torch.float16).float()-md).detach()
            return sd, dd, mm, md

        target = val @ w.T

        def hard_loss(raw):
            z = torch.as_tensor(decode(raw, qtype, weight.shape), device=device)
            return float(F.mse_loss(val @ z.T, target).detach().cpu())

        best, best_loss = initial, hard_loss(initial)
        initial_loss = best_loss
        for epoch in range(config["gsq_epochs"]):
            fraction = epoch/max(1, config["gsq_epochs"]-1)
            temp = config["gsq_temperature"][0]*(config["gsq_temperature"][1]/config["gsq_temperature"][0])**fraction
            order = torch.randperm(len(x), device=device)
            for idx in order.split(128):
                optimizer.zero_grad(set_to_none=True)
                probs = F.gumbel_softmax(logits.float(), tau=temp, hard=False, dim=-1)
                codes = (probs*grid).sum(-1)
                sd, dd, mm, md = native_scales()
                z = codes.reshape(len(q), s.shape[1], -1)*(sd*dd)[..., None]
                if mm is not None:
                    z = z - (mm*md)[..., None]
                z = z.reshape(w.shape)
                batch = x[idx]
                loss = F.mse_loss(batch @ z.T, batch @ w.T)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                sd, dd, mm, md = native_scales()
                codes = grid.gather(-1, logits.argmax(-1)[..., None]).squeeze(-1)
                cpu = lambda t: None if t is None else t.detach().cpu().numpy()
                raw = pack(qtype, cpu(codes), cpu(sd), cpu(dd), cpu(mm), cpu(md))
                score = hard_loss(raw)
                if np.isfinite(score) and score < best_loss:
                    best, best_loss = raw, score
    metrics = {"initial_mse": initial_loss, "selected_mse": best_loss, "improved": best_loss < initial_loss}
    parts = identity.rsplit(":", 2)
    if "work_dir" in config and len(parts) == 3 and parts[-1].isdigit() and int(parts[-1]) % 1024 == 0:
        from pathlib import Path
        from .common import atomic_json, log
        progress = {"tensor": parts[0], "qtype": parts[1], "start_row": int(parts[2]), **metrics}
        atomic_json(Path(config["work_dir"]) / "current-tile.json", progress)
        log("gsq_tile", **progress)
    return best, metrics
