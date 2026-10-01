"""Read incrementally emitted native Qwen features. Waveform decoding is separate."""
import struct
import sys
import numpy as np
from ..pipeline.protocol import TTSFeatureFrame
from .pcm import read_exact


def read_features(stream, revision, epoch=0, allow_end_marker=False):
    if sys.byteorder != "little":
        raise RuntimeError("native feature files currently require a little-endian host")
    header = read_exact(stream, 12)
    if len(header) != 12 or header[:8] != b"VHFEAT1\0":
        raise ValueError("invalid feature stream header")
    hidden_size, = struct.unpack("<i", header[8:])
    if not 1 <= hidden_size <= 16384:
        raise ValueError("invalid hidden size")
    size = 8 + 64 + hidden_size * 4
    expected = 0
    while True:
        record = read_exact(stream, size, allow_eof=True)
        if record is None:
            if allow_end_marker: raise ValueError("missing feature end marker")
            return
        if len(record) != size:
            raise ValueError("truncated feature frame")
        start, = struct.unpack("<q", record[:8])
        if allow_end_marker and start == -1:
            if any(record[8:]): raise ValueError("invalid feature end marker")
            return
        if start != expected:
            raise ValueError("noncontiguous feature clock")
        yield TTSFeatureFrame(epoch, start, np.frombuffer(record, "<i4", 16, 8),
                              np.frombuffer(record, "<f4", hidden_size, 72), revision)
        expected += 1920
