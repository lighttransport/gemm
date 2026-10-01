"""Distinguish first PCM from a heuristic speech-bearing chunk, without re-encoding."""
import numpy as np


class StartupMeter:
    def __init__(self):
        self.first_speech_ns = None
        self.leading_low_energy_samples = None

    def observe(self, chunk, received_ns):
        if self.first_speech_ns is not None: return
        # 20ms RMS threshold: a useful operational metric, not a phoneme detector.
        blocks = chunk.pcm.reshape(-1, 480)
        active = np.flatnonzero(np.sqrt(np.square(blocks.astype(np.float64)).mean(1)) > .003)
        if len(active):
            self.first_speech_ns = received_ns
            self.leading_low_energy_samples = chunk.sample_start + int(active[0])*480
