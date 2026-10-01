"""Typed records. Sample positions are utterance-relative, never frame counts."""
from dataclasses import dataclass
import numpy as np


def integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    if value >= 2**63:
        raise ValueError(f"{name} exceeds int64")


@dataclass(frozen=True)
class AudioChunk:
    epoch: int
    sequence: int
    sample_start: int
    pcm: np.ndarray
    sample_rate: int = 24000

    def __post_init__(self):
        for name in ("epoch", "sequence", "sample_start"):
            integer(getattr(self, name), name)
        x = np.array(self.pcm, dtype=np.float32, copy=True)
        if self.sample_rate != 24000 or x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError("expected finite nonempty 24kHz mono PCM")
        if self.sample_start + len(x) >= 2**63:
            raise ValueError("audio range exceeds int64")
        x.flags.writeable = False
        object.__setattr__(self, "pcm", x)


@dataclass(frozen=True)
class MotionFrame:
    epoch: int
    sample_position: int
    controls: np.ndarray
    confidence: float = 1.0

    def __post_init__(self):
        integer(self.epoch, "epoch")
        integer(self.sample_position, "sample_position")
        x = np.array(self.controls, dtype=np.float32, copy=True)
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError("invalid motion controls")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid motion confidence")
        x.flags.writeable = False
        object.__setattr__(self, "controls", x)


@dataclass(frozen=True)
class TTSFeatureFrame:
    epoch: int
    sample_start: int
    codes: np.ndarray
    hidden: np.ndarray
    model_revision: str
    sample_count: int = 1920

    def __post_init__(self):
        integer(self.epoch, "epoch")
        integer(self.sample_start, "sample_start")
        codes = np.array(self.codes, copy=True)
        hidden = np.array(self.hidden, dtype=np.float32, copy=True)
        if codes.shape != (16,) or codes.dtype.kind not in "iu" or (codes < 0).any() or (codes >= 2048).any():
            raise ValueError("expected sixteen valid RVQ tokens")
        if hidden.ndim != 1 or not len(hidden) or not np.isfinite(hidden).all() or not self.model_revision:
            raise ValueError("invalid hidden state or missing revision")
        if self.sample_count != 1920 or self.sample_start % 1920:
            raise ValueError("codec interval must be aligned to 1920 samples")
        if self.sample_start + self.sample_count >= 2**63:
            raise ValueError("codec range exceeds int64")
        codes = codes.astype(np.int32)
        codes.flags.writeable = hidden.flags.writeable = False
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "hidden", hidden)
