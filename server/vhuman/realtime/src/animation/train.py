"""Small causal adapter training on timestamp-aligned, independently cleared takes."""
import json
from pathlib import Path
import numpy as np
import torch
from .causal import CausalMotion
from ..avatar.provenance import validate_receipts


def train(manifest, output, epochs=20, device="cpu", seed=7, threads=4):
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    manifest = Path(manifest)
    spec = json.loads(manifest.read_text())
    if spec.get("format") != "vhuman.motion_corpus.v1":
        raise ValueError("unsupported corpus")
    validate_receipts(spec["provenance"], "motion-training")
    # File checks apply to the corpus itself, not only to the declared model.
    root = manifest.parent.resolve()
    receipts = {r.get("path"): r for r in spec["provenance"]}
    from ..avatar.provenance import sha256
    takes = []
    for item in spec["takes"]:
        if item["path"] not in receipts:
            raise ValueError("take missing artifact receipt")
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or sha256(path) != receipts[item["path"]]["sha256"]:
            raise ValueError("take checksum/path mismatch")
        with np.load(path, allow_pickle=False) as z:
            hidden, codes, target = z["hidden"].copy(), z["codes"].copy(), z["controls"].copy()
        if hidden.ndim != 2 or codes.shape != (len(hidden), 16) or target.shape != (len(hidden), 8, len(spec["names"])):
            raise ValueError("take shapes must be hidden[T,H], codes[T,16], controls[T,8,C]")
        if codes.dtype.kind not in "iu" or (codes < 0).any() or (codes >= 2048).any() or not np.isfinite(hidden).all() or not np.isfinite(target).all():
            raise ValueError("invalid corpus features/targets")
        if not len(hidden) or (takes and hidden.shape[1] != takes[0][1].shape[1]):
            raise ValueError("empty take or inconsistent hidden size")
        if any(item["path"] == previous.get("path") for previous in spec["takes"][:len(takes)]):
            raise ValueError("same take cannot appear in multiple splits")
        takes.append((item["split"], hidden, codes, target))
    if not any(t[0] == "train" for t in takes) or not any(t[0] == "validation" for t in takes):
        raise ValueError("independent train and validation takes required")
    torch.manual_seed(seed)
    model = CausalMotion(takes[0][1].shape[1], spec["names"], spec["ranges"]).to(device)
    train_hidden = np.concatenate([t[1] for t in takes if t[0] == "train"])
    train_target = np.concatenate([t[3] for t in takes if t[0] == "train"])
    with torch.no_grad():
        model.hidden_mean.copy_(torch.tensor(train_hidden.mean(0), device=device))
        model.hidden_scale.copy_(torch.tensor(np.maximum(train_hidden.std(0), .1), device=device))
        low, high = model.bounds[:, 0], model.bounds[:, 1]
        mean = torch.tensor(train_target.mean(axis=(0, 1)), device=device)
        probability = ((mean-low)/(high-low).clamp_min(1e-6)).clamp(.01, .99)
        model.output.bias.copy_(torch.logit(probability).repeat(8))
    # Static controls must not dominate the loss simply because there are more
    # of them. Compute activity from training takes only; validation stays unseen.
    activity = np.max(np.stack([abs(t[3]).max(axis=(0, 1)) for t in takes if t[0] == "train"]), axis=0)
    weights = torch.tensor(np.where(activity > .05, 4., 1.), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    reports = []
    for epoch in range(epochs):
        losses = {"train": [], "validation": []}
        for split, hidden, codes, target in takes:
            if split not in losses: raise ValueError("invalid corpus split")
            state = None
            model.train(split == "train")
            # Truncated causal BPTT with persistent state; never mix utterances.
            for start in range(0, len(hidden), 32):
                h = torch.tensor(hidden[start:start+32], device=device)[None]
                c = torch.tensor(codes[start:start+32], device=device, dtype=torch.long)[None]
                y = torch.tensor(target[start:start+32], device=device)[None]
                with torch.set_grad_enabled(split == "train"):
                    pred, state = model(h, c, state)
                    loss = (torch.nn.functional.smooth_l1_loss(pred, y, beta=.1, reduction="none")*weights).mean()
                    temporal_pred, temporal_y = pred.flatten(1, 2), y.flatten(1, 2)
                    if temporal_pred.shape[1] > 1:
                        loss += .05 * ((temporal_pred[:, 1:]-temporal_pred[:, :-1] - temporal_y[:, 1:]+temporal_y[:, :-1]).abs()*weights).mean()
                    if split == "train":
                        optimizer.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                state = state.detach()
                losses[split].append(float(loss.detach().cpu()))
        reports.append({"epoch": epoch + 1, **{k: float(np.mean(v)) for k, v in losses.items()}})
    if not reports: raise ValueError("epochs must be positive")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    data = dict(format="vhuman.tts_motion.v1", trained=True, purpose=spec.get("purpose", "diagnostic"), hidden_size=model.hidden_size,
                text_feed=spec.get("text_feed", "incremental"),
                names=list(model.names), ranges=spec["ranges"], tts_revision=spec["tts_revision"],
                state_dict={k: v.cpu() for k, v in model.state_dict().items()}, reports=reports,
                provenance=spec["provenance"], corpus_sha256=sha256(manifest), hidden_normalization="training-only per-channel mean/std, floor0.1",
                objective="active-control weighted Huber plus temporal velocity", active_controls=[n for n, a in zip(spec["names"], activity) if a > .05])
    partial = output.with_name(output.name + ".partial")
    torch.save(data, partial); partial.replace(output)
    from .export_native import export
    export(output)
    return reports[-1]
