"""Headless playout driven by elapsed monotonic time, independent of rendering FPS."""
from ..pipeline.protocol import integer


class OfflinePlayout:
    def __init__(self, origin_ns, rate=24000):
        integer(origin_ns, "origin_ns")
        integer(rate, "rate")
        if not rate: raise ValueError("sample rate must be positive")
        self.origin_ns, self.rate, self.delivered, self.last_ns = origin_ns, rate, 0, origin_ns

    def due(self, now_ns):
        integer(now_ns, "now_ns")
        if now_ns < self.last_ns: raise ValueError("playout clock moved backwards")
        self.last_ns = now_ns
        target = (now_ns - self.origin_ns) * self.rate // 1_000_000_000
        count = target - self.delivered
        self.delivered = target
        return count
