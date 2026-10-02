"""Distil offline LightGeom face motion into a small, seekable WebGL2 model.

Only the offline teacher needs a volume solver. Runtime uses a damped modal
recurrence and a pre-skinning vertex offset; the existing blendshapes, LBS and
post-skinning contacts remain responsible for the principal motion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
from pathlib import Path

import numpy as np

from . import rigdef, safetensors as st


def _inputs(rig: rigdef.Rig, controls: np.ndarray) -> np.ndarray:
    return np.stack([rig.input_vector(row) for row in controls]).astype(np.float32)


def _boundary_fade(rest: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Zero a patch displacement at its open edge so it joins the head."""
    from scipy.spatial import cKDTree

    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = np.unique(unique[counts == 1])
    if not len(boundary):
        return np.ones(len(rest), np.float32)
    distance, _ = cKDTree(rest[boundary]).query(rest)
    t = np.clip(distance / .006, 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def _contact_fade(rest: np.ndarray, ids: np.ndarray, viz: dict) -> np.ndarray:
    from scipy.spatial import cKDTree

    c = viz.get("contacts") or {}
    blocked = set(c.get("lip_ids", [])) | set(c.get("pairs_upper", [])) | set(c.get("pairs_lower", []))
    for eye in c.get("eye", []):
        blocked.update(eye.get("ids", []))
    if not blocked:
        return np.ones(len(ids), np.float32)
    contact = np.fromiter(sorted(blocked), np.int32)
    contact = contact[contact < len(rest)]
    if not len(contact):
        return np.ones(len(ids), np.float32)
    distance, _ = cKDTree(rest[contact]).query(rest[ids])
    t = np.clip(distance / .002, 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def _load_take(rig_dir: Path, take_dir: Path, rig: rigdef.Rig, package: dict,
               rest: np.ndarray, viz: dict, geometry_hash: str) -> tuple[np.ndarray, np.ndarray, float]:
    path = take_dir / "soft_tissue_samples.npz"
    with np.load(path) as sample:
        signature = str(sample["geometry_sha256"]) if "geometry_sha256" in sample else ""
        if signature and signature != geometry_hash:
            raise ValueError(f"{take_dir.name}: tissue samples use a different rig geometry")
        ids = sample["ids"].astype(np.int32)
        faces = sample["faces"].astype(np.int32)
        target = sample["target"].astype(np.float32)
        surface = sample["surface"].astype(np.float32)
        controls = sample["controls"].astype(np.float32)
        fps = float(sample["fps"])
    if not (0 < len(ids) <= len(rest) and np.unique(ids).size == len(ids)
            and (ids >= 0).all() and (ids < len(rest)).all() and
            target.shape == surface.shape == (len(controls), len(ids), 3)
            and faces.ndim == 2 and faces.shape[1] == 3 and
            (faces >= 0).all() and (faces < len(ids)).all() and
            math.isfinite(fps) and 10 <= fps <= 60):
        raise ValueError(f"{take_dir.name}: invalid tissue samples")
    fade = _boundary_fade(rest[ids], faces) * _contact_fade(rest, ids, viz)
    weights = package["skin.weights"][ids]
    joints = package["skin.joints"][ids].astype(np.int64)
    residual = np.zeros((len(controls), len(rest), 3), np.float32)
    for frame, values in enumerate(controls):
        posed = (surface[frame] - target[frame]) * fade[:, None]
        skin = rig.evaluate(values)["skin"]
        blend = (skin[joints, :3, :3] * weights[:, :, None, None]).sum(1)
        residual[frame, ids] = np.linalg.solve(blend, posed[..., None])[..., 0]
    if not np.isfinite(residual).all():
        raise ValueError(f"{take_dir.name}: non-finite inverse-skinned residual")
    return _inputs(rig, controls), residual.reshape(len(controls), -1), fps


def _oscillator(frequency: float, damping: float, fps: float) -> tuple[float, float, float]:
    dt = 1. / fps
    radius = math.exp(-2 * math.pi * frequency * damping * dt)
    angle = 2 * math.pi * frequency * math.sqrt(max(0., 1. - damping * damping)) * dt
    a, b = 2 * radius * math.cos(angle), -radius * radius
    return a, b, 1 - a - b


def predict_sequence(inputs: np.ndarray, weight: np.ndarray, recurrence: tuple[float, float, float]) -> np.ndarray:
    """Deterministic from frame zero: seeking never depends on browser history."""
    x = np.asarray(inputs, np.float32)
    w = np.asarray(weight, np.float32)
    a, b, k = recurrence
    out = np.zeros((len(x), len(w)), np.float32)
    previous = np.zeros(len(w), np.float32)
    previous2 = previous.copy()
    for i, row in enumerate(x):
        current = a * previous + b * previous2 + k * (w @ row)
        out[i] = current
        previous2, previous = previous, current
    return out


def _fit_weight(sequences: list[tuple[np.ndarray, np.ndarray]], recurrence,
                rank: int) -> np.ndarray:
    a, b, k = recurrence
    xs, ys = [], []
    for x, coeff in sequences:
        xs.append(x)
        prior = np.vstack((np.zeros((1, rank), np.float32), coeff[:-1]))
        prior2 = np.vstack((np.zeros((2, rank), np.float32), coeff[:-2]))[:len(coeff)]
        ys.append((coeff - a * prior - b * prior2) / k)
    X, Y = np.concatenate(xs).astype(np.float64), np.concatenate(ys).astype(np.float64)
    ridge = 1e-3 * len(X)
    weight = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)
    return weight.T.astype(np.float32)


def _remap_basis(basis: np.ndarray, rest: np.ndarray, low_rest: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    distance, nearest = cKDTree(rest).query(low_rest, k=4)
    weight = 1. / np.maximum(distance, 1e-5) ** 2
    weight /= weight.sum(1, keepdims=True)
    return np.einsum("vn,kvnc->kvc", weight, basis[:, nearest]).astype(np.float32)


def train(rig_dir: Path, take_dirs: list[Path], *, max_modes: int = 16) -> dict:
    """Fit spatial modes and a stable control-driven second-order recurrence."""
    rig_dir = Path(rig_dir)
    if not 8 <= max_modes <= 16:
        raise ValueError("max_modes must be in [8, 16]")
    rig_path = rig_dir / "rig.json"
    original = rig_path.read_bytes()
    old_rig_hash = hashlib.sha256(original).hexdigest()[:16]
    definition = json.loads(original)
    linear_definition = {key: value for key, value in definition.items() if key != "soft_deformer"}
    linear_rig_hash = hashlib.sha256(json.dumps(linear_definition, indent=1).encode()).hexdigest()[:16]
    rig = rigdef.Rig(definition)
    package_path = rig_dir / "rig_deformer.safetensors"
    geometry_hash = hashlib.sha256(package_path.read_bytes()).hexdigest()[:16]
    package, _ = st.load(package_path)
    rest = package["rest"].astype(np.float32)
    viz = json.loads((rig_dir / "viz.json").read_text())
    tracks = [_load_take(rig_dir, Path(t), rig, package, rest, viz, geometry_hash) for t in take_dirs]
    if not tracks:
        raise ValueError("simulate at least one take before training")
    fps = tracks[0][2]
    if any(abs(t[2] - fps) > 1e-3 for t in tracks):
        raise ValueError("tissue samples must have the same frame rate")
    if len(tracks) >= 2:
        learning, held = tracks[:-1], [tracks[-1]]
        split = "held-out take"
    else:
        x, y, _ = tracks[0]
        cut = max(3, int(.8 * len(x)))
        if cut >= len(x) - 2:
            raise ValueError("one tissue take needs at least six frames")
        learning, held = [(x[:cut], y[:cut], fps)], [(x[cut:], y[cut:], fps)]
        split = "held-out contiguous tail (single take)"
    train_matrix = np.concatenate([t[1] for t in learning])
    rank = min(max_modes, len(train_matrix) - 1)
    if rank < 4:
        raise ValueError("too few tissue frames for a modal model")
    from ..native_training import randomized_basis
    basis = randomized_basis(train_matrix, rank, seed=23)
    candidates = sorted(set(k for k in (8, 12, 16) if k <= rank) | {rank})
    scores = []
    for modes in candidates:
        train_seq = [(x, y @ basis[:modes].T) for x, y, _ in learning]
        for frequency in (3., 5., 8.):
            for damping in (.4, .7, 1.):
                recurrence = _oscillator(frequency, damping, fps)
                weight = _fit_weight(train_seq, recurrence, modes)
                errors = []
                for x, y, _ in held:
                    pred = predict_sequence(x, weight, recurrence) @ basis[:modes]
                    errors.append(np.mean((pred - y) ** 2))
                scores.append((float(np.mean(errors)), modes, frequency, damping, weight, recurrence))
    best_error = min(row[0] for row in scores)
    selected = min((row for row in scores if row[0] <= best_error * 1.1),
                   key=lambda row: (row[1], row[0]))
    error, modes, frequency, damping, weight, recurrence = selected
    selected_basis = basis[:modes].reshape(modes, len(rest), 3)
    baseline_mm = 1000 * math.sqrt(float(np.mean([np.mean(y * y) for _, y, _ in held])))
    model_mm = 1000 * math.sqrt(error)
    active = np.any(np.abs(train_matrix) > 1e-8, axis=0)
    if not active.any():
        raise ValueError("LightGeom teacher has no measurable facial residual")
    patch_baseline = []
    patch_model = []
    for x, y, _ in held:
        pred = predict_sequence(x, weight, recurrence) @ basis[:modes]
        patch_baseline.append(y[:, active])
        patch_model.append((pred - y)[:, active])
    patch_baseline_mm = 1000 * math.sqrt(float(np.mean(np.concatenate(patch_baseline) ** 2)))
    patch_model_mm = 1000 * math.sqrt(float(np.mean(np.concatenate(patch_model) ** 2)))
    if not np.isfinite(selected_basis).all() or not np.isfinite(weight).all():
        raise ValueError("non-finite trained deformer")
    model = {"weight": weight, "basis": selected_basis,
             "recurrence": np.asarray(recurrence, np.float32)}
    meta = {"format": "vhuman.soft_deformer.v1", "inputs": json.dumps(rig.inputs),
            "geometry_sha256": geometry_hash, "fps": str(fps)}
    st.save(rig_dir / "soft_deformer.safetensors", model, meta)
    lod_files = {}
    for level in (1, 2):
        path = rig_dir / f"rig_deformer_lod{level}.safetensors"
        if not path.is_file():
            continue
        low, _ = st.load(path)
        name = f"soft_deformer_lod{level}.safetensors"
        st.save(rig_dir / name, {"basis": _remap_basis(selected_basis, rest, low["rest"])}, meta)
        lod_files[str(level)] = name
    definition["soft_deformer"] = {"format": meta["format"], "model": "soft_deformer.safetensors",
                                   "lod_models": lod_files, "inputs": rig.inputs, "modes": modes,
                                   "fps": fps, "geometry_sha256": geometry_hash}
    staged_rig = rig_path.with_name("rig.json.soft.partial")
    staged_rig.write_text(json.dumps(definition, indent=1))
    staged_rig.replace(rig_path)
    # Adding the optional model does not alter the linear rig used to make
    # existing takes. Preserve their freshness without masking older stale
    # takes that referred to a genuinely different rig.
    new_rig_hash = hashlib.sha256(rig_path.read_bytes()).hexdigest()[:16]
    for manifest_path in (rig_dir / "takes").glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("rig_sha256") in (old_rig_hash, linear_rig_hash):
            manifest["rig_sha256"] = new_rig_hash
            staged_manifest = manifest_path.with_name("manifest.json.soft.partial")
            staged_manifest.write_text(json.dumps(manifest, indent=2))
            staged_manifest.replace(manifest_path)
    report = {"format": meta["format"], "backend": "native_cpu", "device": "cpu",
              "pca": "repository GEMM, NumPy thin QR/SVD",
              "takes": [Path(t).name for t in take_dirs],
              "split": split, "modes": modes, "frequency_hz": frequency, "damping": damping,
              "heldout_baseline_rmse_mm": baseline_mm, "heldout_model_rmse_mm": model_mm,
              "heldout_patch_baseline_rmse_mm": patch_baseline_mm,
              "heldout_patch_model_rmse_mm": patch_model_mm,
              "geometry_sha256": geometry_hash, "lod_models": lod_files}
    (rig_dir / "soft_deformer_report.json").write_text(json.dumps(report, indent=2))
    return report


def train_job(service, request: dict, progress, cancel, *, python=None) -> dict:
    from .job import DEFAULT_PYTHON

    head = request.get("head_id")
    rig_dir = service.rig_file(head, "rig.json").parent
    take_ids = request.get("take_ids") or [t["id"] for t in service.list_takes(head)
                                               if (rig_dir / "takes" / t["id"] / "soft_tissue_samples.npz").is_file()]
    if not isinstance(take_ids, list) or not 1 <= len(take_ids) <= 8:
        raise ValueError("provide 1–8 takes with completed soft-tissue samples")
    takes = []
    for take_id in take_ids:
        folder = service.take_file(head, take_id, "animation.json").parent
        if not (folder / "soft_tissue_samples.npz").is_file():
            raise ValueError(f"take {take_id} needs rig-soft-tissue first")
        takes.append(folder)
    py = Path(python) if python else DEFAULT_PYTHON
    if not py.is_file():
        raise ValueError(f"rig interpreter is missing: {py}")
    progress(.05, "fitting compact tissue modes")
    cmd = [str(py), "-m", "server.vhuman.rig.soft_deformer", str(rig_dir), *map(str, takes)]
    proc = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[3], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    try:
        while proc.poll() is None:
            if cancel.is_set():
                os.killpg(proc.pid, signal.SIGTERM)
                proc.communicate(timeout=5)
                raise RuntimeError("soft deformer training cancelled")
            import time
            time.sleep(.1)
        output = proc.communicate()[0]
        if proc.returncode:
            raise RuntimeError("soft deformer training failed: " + output[-1500:])
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    report = json.loads((rig_dir / "soft_deformer_report.json").read_text())
    progress(.99, "soft deformer ready")
    return {"head_id": head, "report": report,
            "model_url": f"/v1/heads/{head}/rig/soft_deformer.safetensors"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rig_dir", type=Path)
    ap.add_argument("take_dirs", type=Path, nargs="+")
    ap.add_argument("--max-modes", type=int, default=16)
    args = ap.parse_args(argv)
    print(json.dumps(train(args.rig_dir, args.take_dirs, max_modes=args.max_modes)))


if __name__ == "__main__":
    main()
