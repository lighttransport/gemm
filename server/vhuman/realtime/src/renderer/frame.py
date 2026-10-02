"""Owned native frame with an explicit completion event and presentation IO."""
from dataclasses import dataclass


@dataclass
class FrameHandle:
    rgba: object
    sample_position: int
    ready: object
    submitted_ns: int
    runtime: object = None

    def pixels(self, *, straight_alpha=False):
        self.ready.synchronize()
        if self.runtime is None or self.rgba is None:
            raise RuntimeError('frame is closed or lacks a presentation runtime')
        return self.runtime.pixels(self.rgba.data_ptr(), tuple(self.rgba.shape),
                                   straight_alpha=straight_alpha)

    def close(self):
        self.ready.close()
        if hasattr(self.rgba, 'close'):
            self.rgba.close()
        self.rgba = None

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
