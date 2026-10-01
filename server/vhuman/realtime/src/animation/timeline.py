"""Bounded, sample-positioned motion interpolation and named retargeting."""
from collections import deque
import numpy as np
from ..pipeline.protocol import MotionFrame, integer


class MotionTimeline:
    def __init__(self, names, ranges, capacity=200):
        self.names = tuple(names)
        self.ranges = np.asarray(ranges, np.float32)
        if not self.names or len(set(self.names)) != len(self.names) or self.ranges.shape != (len(self.names), 2):
            raise ValueError("invalid control manifest")
        if not np.isfinite(self.ranges).all() or (self.ranges[:, 0] > self.ranges[:, 1]).any() or capacity < 2:
            raise ValueError("invalid control ranges/capacity")
        self.capacity = capacity
        self.reset(0)

    def reset(self, epoch):
        integer(epoch, "epoch")
        self.epoch = epoch
        self.frames = deque()

    def push(self, frame):
        if frame.epoch != self.epoch:
            return False
        if frame.controls.shape != (len(self.names),):
            raise ValueError("control count mismatch")
        if self.frames and frame.sample_position <= self.frames[-1].sample_position:
            raise ValueError("motion sample positions must increase")
        if len(self.frames) >= self.capacity:
            raise BufferError("motion queue full; apply producer backpressure")
        controls = np.clip(frame.controls, self.ranges[:, 0], self.ranges[:, 1])
        self.frames.append(MotionFrame(frame.epoch, frame.sample_position, controls, frame.confidence))
        return True

    def sample(self, position, hold_samples=2400, ease_samples=2400):
        integer(position, "position")
        integer(hold_samples, "hold_samples")
        integer(ease_samples, "ease_samples")
        while len(self.frames) >= 2 and self.frames[1].sample_position <= position:
            self.frames.popleft()
        neutral = np.clip(np.zeros(len(self.names), np.float32), self.ranges[:, 0], self.ranges[:, 1])
        if not self.frames or position < self.frames[0].sample_position:
            return neutral
        first = self.frames[0]
        if len(self.frames) >= 2:
            second = self.frames[1]
            alpha = (position - first.sample_position) / (second.sample_position - first.sample_position)
            return first.controls * (1 - alpha) + second.controls * alpha
        lag = max(0, position - first.sample_position - hold_samples)
        alpha = min(1.0, lag / max(1, ease_samples))
        return first.controls * (1 - alpha) + neutral * alpha


def retarget(values, source_names, target_names):
    """ARKit tongueOut belongs to our EXTRA controls, not the 51-face array."""
    source_names, target_names = tuple(source_names), tuple(target_names)
    x = np.asarray(values, np.float32)
    if x.ndim == 0 or x.shape[-1] != len(source_names) or len(set(source_names)) != len(source_names) or len(set(target_names)) != len(target_names) or not np.isfinite(x).all():
        raise ValueError("invalid source controls")
    indices = {name: i for i, name in enumerate(source_names)}
    out = np.zeros((*x.shape[:-1], len(target_names)), np.float32)
    for j, name in enumerate(target_names):
        if name in indices:
            out[..., j] = x[..., indices[name]]
    return out
