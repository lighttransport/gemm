"""Compatibility reader for vhuman.performance.v1 named dictionaries and arrays."""
import json
from pathlib import Path
import numpy as np


def load_performance(path):
    data = json.loads(Path(path).read_text())
    if data.get("format") != "vhuman.performance.v1": raise ValueError("unsupported performance")
    names = tuple(data["controls"])
    if not names or len(set(names)) != len(names): raise ValueError("invalid performance control names")
    samples, values = [], []
    for frame in data["frames"]:
        t = frame["t"]
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t) or t < 0:
            raise ValueError("invalid performance timestamp")
        row = frame["v"]
        if isinstance(row, dict):
            if set(row) - set(names): raise ValueError("unknown named performance controls")
            row = [row.get(name, 0) for name in names]
        row = np.asarray(row, np.float32)
        if row.shape != (len(names),) or not np.isfinite(row).all(): raise ValueError("invalid performance values")
        position = round(t * 24000)
        if position >= 2**63 or (samples and position <= samples[-1]): raise ValueError("nonmonotonic performance samples")
        samples.append(position); values.append(row)
    if not samples: raise ValueError("empty performance")
    return names, np.asarray(samples, np.int64), np.asarray(values, np.float32)
