"""Portable body tracks. The animation runtime never imports the simulator."""
from __future__ import annotations

import numpy as np


def validate_tracks(names, rotations, times, root_translation=None):
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("joint names must be unique nonempty strings")
    q, t = np.asarray(rotations, dtype=np.float64), np.asarray(times, dtype=np.float64)
    if (t.ndim != 1 or not 1 <= len(t) <= 24001 or q.shape != (len(t), len(names), 4)
            or not np.isfinite(t).all() or not np.isfinite(q).all()
            or abs(t[0]) > 1e-6 or np.any(np.diff(t) <= 0)):
        raise ValueError("invalid motion dimensions, timestamps or nonfinite values")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(abs(norm - 1) > 1e-3):
        raise ValueError("motion quaternions must be normalized")
    q = q / norm
    for i in range(1, len(q)):
        q[i, np.sum(q[i - 1] * q[i], axis=-1) < 0] *= -1
    root = None if root_translation is None else np.asarray(root_translation, dtype=np.float64)
    if root is not None and (root.shape != (len(t), 3) or not np.isfinite(root).all()):
        raise ValueError("invalid root translations")
    return q, t, root


def document(names, rotations, times, root_translation, *, provenance=None, kind="musclemimic", terrain=None):
    q, t, root = validate_tracks(names, rotations, times, root_translation)
    if root is None or "mhr_root" not in names:
        raise ValueError("v2 requires an animated mhr_root")
    return {"version": 2, "kind": kind, **({"terrain": terrain} if terrain else {}), "fps": float(1 / np.median(np.diff(t))) if len(t) > 1 else 0,
            "duration": float(t[-1]), "joints": list(names), "space": "MHR local xyzw",
            "units": "m", "up_axis": "Y", "root_joint": "mhr_root", "root_translation": "animated",
            "provenance": provenance or {}, "frames": [
                {"t": float(s), "rotations": r.tolist(), "root_translation": p.tolist()}
                for s, r, p in zip(t, q, root)]}
