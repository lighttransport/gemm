"""Timestamped native PCM framing; pipes may split any field across reads."""
import struct
import numpy as np
from ..pipeline.protocol import AudioChunk


def read_exact(stream, count, allow_eof=False):
    pieces, received = [], 0
    while received < count:
        piece = stream.read(count - received)
        if not piece:
            if allow_eof and not received: return None
            raise ValueError("truncated native stream")
        pieces.append(piece); received += len(piece)
    return b"".join(pieces)


def read_pcm(stream, epoch=0, allow_end_marker=False):
    header = read_exact(stream, 12)
    if header[:8] != b"VHPCM1\0\0" or struct.unpack("<i", header[8:])[0] != 24000:
        raise ValueError("unsupported PCM stream")
    sequence = expected = 0
    while True:
        record = read_exact(stream, 12, allow_eof=True)
        if record is None:
            if allow_end_marker: raise ValueError("missing PCM end marker")
            return
        start, count = struct.unpack("<qi", record)
        if allow_end_marker and start == -1 and count == 0: return
        if start != expected or count != 1920: raise ValueError("invalid PCM interval")
        pcm = np.frombuffer(read_exact(stream, count * 4), "<f4")
        yield AudioChunk(epoch, sequence, start, pcm)
        sequence += 1; expected += count
