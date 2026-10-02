"""Held-out teacher agreement and neutral baseline; no perceptual-quality claim."""
import json
from pathlib import Path
import numpy as np


def evaluate(manifest, checkpoint, output, threads=1, *, reference_parity=False):
    from contextlib import closing
    from ..animation.native_motion import MotionAdapter
    from ..avatar.provenance import verify_files
    manifest = Path(manifest); spec = json.loads(manifest.read_text())
    if spec.get("format") != "vhuman.motion_corpus.v1": raise ValueError("unsupported corpus")
    verify_files(spec["provenance"], manifest.parent, "motion-training")
    oracle = None
    if reference_parity:
        import torch
        from ..animation.causal_model import ReferenceMotionAdapter
        torch.set_num_threads(threads)
        oracle = ReferenceMotionAdapter(checkpoint, spec['tts_revision'], allow_diagnostic=True)
    with closing(MotionAdapter(checkpoint, spec['tts_revision'], allow_diagnostic=True)) as adapter:
        if list(adapter.model.names) != spec['names']:
            raise ValueError('control order mismatch')
        return _evaluate_takes(adapter, oracle, spec, manifest, checkpoint, output)


def _evaluate_takes(adapter, oracle, spec, manifest, checkpoint, output):
    from ..pipeline.protocol import TTSFeatureFrame
    from ..avatar.provenance import sha256
    if oracle is not None:
        import torch
    errors, baselines, targets, details = [], [], [], []
    receipts = {r["path"] for r in spec["provenance"]}
    for item in spec["takes"]:
        if item["split"] != "validation": continue
        if item["path"] not in receipts: raise ValueError("missing validation receipt")
        path = (manifest.parent / item["path"]).resolve()
        if not path.is_relative_to(manifest.parent.resolve()): raise ValueError("take escapes corpus")
        with np.load(path, allow_pickle=False) as data:
            h, c, target = data["hidden"], data["codes"], data["controls"]
            if not len(h) or len(c) != len(h) or target.shape != (len(h), 8, len(spec['names'])) or not np.isfinite(target).all():
                raise ValueError('invalid evaluation take')
            adapter.reset(0)
            output_steps = []
            for i in range(len(h)):
                frames = adapter.push(TTSFeatureFrame(0, i*1920, c[i], h[i], spec['tts_revision']))
                output_steps.append(np.stack([frame.controls for frame in frames]))
            pred = np.stack(output_steps)
            parity = None
            if oracle is not None:
                with torch.inference_mode():
                    full, _ = oracle.model(torch.tensor(h)[None], torch.tensor(c, dtype=torch.long)[None])
                parity = float(abs(pred-full[0].numpy()).max())
                if parity > 2e-5:
                    raise RuntimeError('native student/reference full-sequence parity failed')
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
    result = dict(format="vhuman.motion_evaluation.v1", checkpoint_sha256=sha256(Path(checkpoint)/'native.json' if Path(checkpoint).is_dir() else checkpoint), corpus_sha256=sha256(manifest),
        evaluation="sentence-held-out teacher agreement", backend="native_cpu", reference_parity_checked=oracle is not None, purpose="diagnostic", takes=details,
        active_mae=float(mae[active].mean()) if active.any() else None,
        active_neutral_mae=float(neutral[active].mean()) if active.any() else None,
        controls={n: dict(mae=float(e), neutral_mae=float(b)) for n,e,b in zip(spec["names"],mae,neutral)})
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result
