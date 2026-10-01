"""Resample a cleared offline teacher take onto exact TTS codec sample positions."""
import json
from pathlib import Path
import numpy as np
from .timeline import retarget
from .performance import load_performance
from ..tts.features import read_features
from ..avatar.provenance import sha256


def capture(features, performance, output, revision, names, ranges):
    performance_path = Path(performance)
    source_names, times, source_values = load_performance(performance_path)
    with Path(features).open("rb") as stream: records = list(read_features(stream, revision))
    if not records: raise ValueError("empty features")
    values = retarget(source_values, source_names, names)
    if not np.isfinite(times).all() or len(times) < 2 or (np.diff(times) <= 0).any() or times[0] > 0:
        raise ValueError("invalid teacher timestamps")
    starts = np.array([r.sample_start for r in records], np.int64)
    positions = starts[:, None] + np.arange(8)[None] * 240
    if times[-1] < positions[-1, -1]: raise ValueError("teacher ends before the codec features")
    target = np.stack([np.interp(positions, times, values[:, i]) for i in range(len(names))], -1).astype(np.float32)
    bounds = np.asarray(ranges, np.float32)
    if bounds.shape != (len(names), 2): raise ValueError("invalid control ranges")
    target = np.clip(target, bounds[:, 0], bounds[:, 1])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, hidden=np.stack([r.hidden for r in records]), codes=np.stack([r.codes for r in records]),
                            controls=target, sample_positions=starts)
    return dict(path=output.name, sha256=sha256(output), frames=len(starts), tts_revision=revision,
                features_sha256=sha256(features), teacher_sha256=sha256(performance_path))
