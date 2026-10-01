"""Held-out teacher agreement and neutral baseline; no perceptual-quality claim."""
import json
from pathlib import Path
import numpy as np


def evaluate(manifest, checkpoint, output, threads=1):
    import torch
    from ..animation.causal import MotionAdapter
    from ..avatar.provenance import verify_files, sha256
    torch.set_num_threads(threads)
    manifest = Path(manifest); spec = json.loads(manifest.read_text())
    if spec.get("format") != "vhuman.motion_corpus.v1": raise ValueError("unsupported corpus")
    verify_files(spec["provenance"], manifest.parent, "motion-training")
    adapter = MotionAdapter(checkpoint, spec["tts_revision"], allow_diagnostic=True)
    if list(adapter.model.names) != spec["names"]: raise ValueError("control order mismatch")
    errors, baselines, targets, details = [], [], [], []
    receipts = {r["path"] for r in spec["provenance"]}
    for item in spec["takes"]:
        if item["split"] != "validation": continue
        if item["path"] not in receipts: raise ValueError("missing validation receipt")
        path = (manifest.parent / item["path"]).resolve()
        if not path.is_relative_to(manifest.parent.resolve()): raise ValueError("take escapes corpus")
        with np.load(path, allow_pickle=False) as data:
            h, c, target = data["hidden"], data["codes"], data["controls"]
            state, output_steps = None, []
            with torch.inference_mode():
                for i in range(len(h)):
                    value, state = adapter.model(torch.tensor(h[i:i+1])[None], torch.tensor(c[i:i+1], dtype=torch.long)[None], state)
                    output_steps.append(value[0, 0].numpy())
            pred = np.stack(output_steps)
            with torch.inference_mode():
                full, _ = adapter.model(torch.tensor(h)[None], torch.tensor(c, dtype=torch.long)[None])
            parity = float(abs(pred-full[0].numpy()).max())
            if parity > 1e-5: raise RuntimeError("trained student step/full parity failed")
            errors.append(abs(pred-target).mean(axis=(0, 1)))
            baselines.append(abs(target).mean(axis=(0, 1)))
            targets.append(abs(target).max(axis=(0, 1)))
            detail = dict(path=item["path"], frames=len(h), mae=float(abs(pred-target).mean()), step_full_max_error=parity)
            if "jawOpen" in spec["names"]:
                jaw = spec["names"].index("jawOpen")
                p, y = pred[..., jaw].ravel(), target[..., jaw].ravel()
                correlations = []
                for lag in range(-20, 21):
                    left, right = (p[lag:], y[:-lag]) if lag > 0 else (p[:lag], y[-lag:]) if lag < 0 else (p, y)
                    correlations.append(float(np.corrcoef(left, right)[0, 1]) if min(left.std(),right.std()) > 1e-8 else -1.)
                best = int(np.argmax(correlations))
                detail.update(jaw_correlation=correlations[best], jaw_teacher_lag_ms=(best-20)*10)
            details.append(detail)
    if not errors: raise ValueError("no validation takes")
    # Equal-utterance weighting is deliberate; long takes must not hide failures.
    mae, neutral = np.mean(errors, 0), np.mean(baselines, 0)
    active = np.max(targets, 0) > .05
    result = dict(format="vhuman.motion_evaluation.v1", checkpoint_sha256=sha256(checkpoint), corpus_sha256=sha256(manifest),
        evaluation="sentence-held-out teacher agreement", purpose="diagnostic", takes=details,
        active_mae=float(mae[active].mean()) if active.any() else None,
        active_neutral_mae=float(neutral[active].mean()) if active.any() else None,
        controls={n: dict(mae=float(e), neutral_mae=float(b)) for n,e,b in zip(spec["names"],mae,neutral)})
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result
