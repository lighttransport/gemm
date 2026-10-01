"""Stage measurements, with wall and GPU clocks kept distinct."""
from collections import defaultdict, deque
from contextlib import contextmanager
import json
import time
import numpy as np


class Metrics:
    def __init__(self, trace=None, capacity=10000):
        self.samples = defaultdict(lambda: deque(maxlen=capacity))
        self.trace = trace

    def add(self, name, value, **fields):
        if not np.isfinite(value):
            raise ValueError("nonfinite measurement")
        self.samples[name].append(float(value))
        if self.trace:
            self.trace.write(json.dumps(dict(monotonic_ns=time.monotonic_ns(), metric=name,
                                            value=float(value), **fields)) + "\n")

    @contextmanager
    def wall(self, name):
        start = time.monotonic_ns()
        try:
            yield
        finally:
            self.add(name, (time.monotonic_ns() - start) / 1e6)

    def report(self):
        return {name: dict(count=len(values), p50=float(np.percentile(values, 50)),
                          p95=float(np.percentile(values, 95)), p99=float(np.percentile(values, 99)),
                          maximum=float(max(values)))
                for name, values in self.samples.items() if values}
