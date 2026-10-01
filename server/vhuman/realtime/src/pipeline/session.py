"""Bounded ingest and audio-master replay engine, independent of model backends."""
from ...config import RuntimeConfig
from ..audio.clock import SampleClock
from ..animation.timeline import MotionTimeline
from ..benchmark.metrics import Metrics


class Session:
    def __init__(self, ring, names, ranges, config=None, metrics=None):
        self.config = config or RuntimeConfig()
        self.ring = ring
        self.clock = SampleClock()
        self.motion = MotionTimeline(names, ranges, self.config.motion_capacity)
        self.metrics = metrics or Metrics()
        self.epoch = 0
        self.sequence = self.accepted = self.consumed = 0

    def cancel(self):
        """Invoke only after joining producer and stopping the device callback."""
        self.epoch += 1
        self.ring.reset()
        self.clock.reset(self.epoch)
        self.motion.reset(self.epoch)
        self.sequence = self.accepted = self.consumed = 0

    def ingest(self, chunk):
        if chunk.epoch != self.epoch:
            return False
        if chunk.sequence != self.sequence or chunk.sample_start != self.accepted:
            raise ValueError("out-of-order audio chunk")
        self.ring.write(chunk.pcm)
        self.sequence += 1
        self.accepted += len(chunk.pcm)
        self.metrics.add("audio_buffer_ms", self.ring.depth / 24)
        return True

    def pull(self, count, monotonic_ns):
        """Deterministic test/offline sink. Device bridge must report actual DAC time."""
        pcm, n = self.ring.read(count)
        self.clock.submit(n, self.consumed)
        self.consumed += n
        self.clock.submit(count - n)
        self.clock.observe(self.clock.submitted, 24000, monotonic_ns)
        self.metrics.add("underrun_samples", count - n)
        return pcm, self.motion.sample(self.clock.position())
