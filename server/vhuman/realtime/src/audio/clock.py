"""Audio-master clock including inserted-silence segments.

The device reports played samples in its own rate. Integer rational accounting
prevents cumulative rounding errors. Underruns advance playout, not speech.
"""
from collections import deque
from dataclasses import dataclass
from ..pipeline.protocol import integer


@dataclass(frozen=True)
class Segment:
    playout_start: int
    count: int
    speech_start: int | None
    held_speech: int


class SampleClock:
    def __init__(self, sample_rate=24000):
        integer(sample_rate, "sample_rate")
        if sample_rate <= 0:
            raise ValueError("invalid rate")
        self.rate = sample_rate
        self.reset(0)

    def reset(self, epoch):
        integer(epoch, "epoch")
        self.epoch = epoch
        self.segments = deque()
        self.submitted = self.speech_end = self.played = self.retired_speech = 0
        self.anchor_ns = None

    def submit(self, count, speech_start=None):
        integer(count, "count")
        if not count:
            return
        if speech_start is not None:
            integer(speech_start, "speech_start")
            if speech_start != self.speech_end:
                raise ValueError("noncontiguous speech samples")
        self.segments.append(Segment(self.submitted, count, speech_start, self.speech_end))
        self.submitted += count
        if speech_start is not None:
            self.speech_end += count

    def observe(self, device_played, device_rate, monotonic_ns):
        integer(device_rate, "device_rate")
        integer(device_played, "device_played")
        integer(monotonic_ns, "monotonic_ns")
        if device_rate <= 0:
            raise ValueError("invalid device rate")
        position = device_played * self.rate // device_rate
        if position < self.played or position > self.submitted:
            raise ValueError("device clock outside submitted monotonic timeline")
        if self.anchor_ns is not None and monotonic_ns < self.anchor_ns:
            raise ValueError("nonmonotonic device timestamp")
        self.played, self.anchor_ns = position, monotonic_ns
        while self.segments and self.segments[0].playout_start + self.segments[0].count <= position:
            s = self.segments.popleft()
            self.retired_speech = s.held_speech if s.speech_start is None else s.speech_start + s.count

    def position(self, monotonic_ns=None, presentation_ns=0):
        position = self.played
        if monotonic_ns is not None and self.anchor_ns is not None:
            elapsed = max(0, monotonic_ns + presentation_ns - self.anchor_ns)
            position = min(self.submitted, position + elapsed * self.rate // 1_000_000_000)
        for s in self.segments:
            if position < s.playout_start + s.count:
                offset = max(0, position - s.playout_start)
                return s.held_speech if s.speech_start is None else s.speech_start + offset
        return self.speech_end if position == self.submitted else self.retired_speech
