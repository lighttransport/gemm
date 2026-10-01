"""Explicit runtime settings. Production paths do not use tuning env vars."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    sample_rate: int = 24000
    codec_samples: int = 1920
    delivery_samples: int = 480
    playback_capacity: int = 48000
    history_capacity: int = 96000
    startup_samples: int = 1920
    high_water_samples: int = 9600
    low_water_samples: int = 3840
    motion_capacity: int = 200
    fps: int = 60

    def __post_init__(self):
        for value in self.__dict__.values():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("runtime settings must be nonnegative integers")
        if self.sample_rate != 24000 or self.codec_samples != 1920:
            raise ValueError("Qwen 12Hz transport requires 24kHz / 1920 samples")
        if not 0 < self.startup_samples <= self.high_water_samples < self.playback_capacity:
            raise ValueError("invalid audio watermarks")
        if not 0 <= self.low_water_samples < self.high_water_samples:
            raise ValueError("invalid low watermark")
        if self.motion_capacity < 2 or self.fps <= 0 or self.delivery_samples <= 0:
            raise ValueError("invalid queue size or cadence")
