"""Versioned mobile wire records; audio samples are the only animation clock.

JSON envelopes use base64 little-endian float32 PCM. The first implementation
favours inspectability; a future binary transport must retain these invariants.
"""
import base64
import json
import numpy as np
from ..realtime.src.pipeline.protocol import AudioChunk, integer

SCHEMA = 'vhuman.mobile_stream.v1'
MAX_MESSAGE = 128 * 1024


def pose(epoch, sample_position, expression, rotations=None, translation=None):
    integer(epoch, 'epoch'); integer(sample_position, 'sample_position')
    result = dict(type='motion', epoch=int(epoch), sample_position=int(sample_position))
    for name, value, shape, bound in (
        ('expression', expression, (383,), 3),
        ('rotations', np.zeros((4, 3)) if rotations is None else rotations, (4, 3), 3.15),
        ('translation', np.zeros(3) if translation is None else translation, (3,), 10),
    ):
        a = np.asarray(value, dtype=np.float32)
        if a.shape != shape or not np.isfinite(a).all() or (np.abs(a) > bound).any():
            raise ValueError('invalid native ' + name)
        result[name] = a.reshape(-1).tolist()
    return result


def audio(chunk):
    if len(chunk.pcm) > 4800 or (np.abs(chunk.pcm) > 1).any():
        raise ValueError('audio must be at most 200ms, normalized mono PCM')
    return dict(type='audio', epoch=int(chunk.epoch), sequence=int(chunk.sequence),
        sample_start=int(chunk.sample_start), sample_rate=24000,
        pcm=base64.b64encode(np.asarray(chunk.pcm, dtype='<f4').tobytes()).decode('ascii'))


def encode(record):
    data = json.dumps(record, separators=(',', ':'), allow_nan=False)
    if len(data.encode()) > MAX_MESSAGE: raise ValueError('wire message exceeds limit')
    return data


def decode(data):
    if len(data) > MAX_MESSAGE: raise ValueError('wire message exceeds limit')
    record = json.loads(data)
    if not isinstance(record, dict): raise ValueError('expected wire object')
    kind = record.get('type')
    if kind == 'audio':
        pcm = base64.b64decode(record['pcm'], validate=True)
        if not pcm or len(pcm) % 4 or len(pcm) > 4800 * 4: raise ValueError('invalid PCM size')
        chunk = AudioChunk(record['epoch'], record['sequence'], record['sample_start'],
            np.frombuffer(pcm, dtype='<f4'), record['sample_rate'])
        audio(chunk)  # Apply normalized-amplitude and chunk-size gates.
        return chunk
    if kind == 'motion':
        return pose(record['epoch'], record['sample_position'], record['expression'],
            np.asarray(record['rotations']).reshape(4, 3), record['translation'])
    raise ValueError('unsupported data record')


class StreamOrder:
    """Receiver ordering gate; stale epochs are dropped, gaps fail closed."""
    def __init__(self):
        self.epoch = -1
        self.sequence = self.samples = 0
        self.motion_position = -1

    def begin(self, epoch):
        integer(epoch, 'epoch')
        if epoch <= self.epoch: raise ValueError('epoch must increase')
        self.epoch = epoch; self.sequence = self.samples = 0; self.motion_position = -1

    def accept(self, record):
        epoch = record.epoch if isinstance(record, AudioChunk) else record['epoch']
        if epoch < self.epoch: return False
        if epoch != self.epoch: raise ValueError('begin required before data')
        if isinstance(record, AudioChunk):
            if record.sequence != self.sequence or record.sample_start != self.samples:
                raise ValueError('audio discontinuity')
            self.sequence += 1; self.samples += len(record.pcm)
        else:
            if record['sample_position'] <= self.motion_position: raise ValueError('motion out of order')
            self.motion_position = record['sample_position']
        return True
