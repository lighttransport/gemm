"""Convert the pinned MediaPipe task to native graphs, without importing MediaPipe.

Only NumPy and the standard library are required. TFLite is an asset format here,
not an inference runtime. The small reader covers the pinned float16 graphs.
Schema: tensorflow/lite/schema/schema.fbs (Apache-2.0).
"""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import zipfile
import numpy as np

TASK_SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
MODELS = ('face_detector', 'face_landmarks_detector', 'face_blendshapes')
OPS = {0, 2, 3, 4, 6, 14, 17, 18, 19, 22, 34, 39, 40, 41, 42, 45, 54, 59, 74, 75, 76, 78, 92, 99}


class Table:
    def __init__(self, data, position):
        self.data, self.position = data, position
        self.vtable = position-self.read('i', position)
        self.size = self.read('H', self.vtable)

    def read(self, kind, offset):
        return struct.unpack_from('<'+kind, self.data, offset)[0]

    def field(self, index):
        offset = self.read('H', self.vtable+4+index*2) if 4+index*2 < self.size else 0
        return self.position+offset if offset else 0

    def scalar(self, index, kind='i', default=0):
        p = self.field(index)
        return self.read(kind, p) if p else default

    def table(self, index):
        p = self.field(index)
        return Table(self.data, p+self.read('I', p)) if p else None

    def vector(self, index, kind='i'):
        p = self.field(index)
        if not p:
            return []
        p += self.read('I', p)
        n = self.read('I', p)
        if kind == 'table':
            return [Table(self.data, q+self.read('I', q)) for q in range(p+4, p+4+n*4, 4)]
        return list(struct.unpack_from('<'+str(n)+kind, self.data, p+4))


def convert(data, output):
    if data[4:8] != b'TFL3':
        raise ValueError('expected TFL3 model')
    root = Table(data, struct.unpack_from('<I', data)[0])
    graphs = root.vector(2, 'table')
    if len(graphs) != 1:
        raise ValueError('expected one graph')
    graph = graphs[0]
    codes = [max(t.scalar(0, 'b'), t.scalar(3)) for t in root.vector(1, 'table')]
    buffers = root.vector(4, 'table')
    tensors = []
    for t in graph.vector(0, 'table'):
        shape = t.vector(0)
        if len(shape) > 4 or any(d < 1 for d in shape):
            raise ValueError('unsupported tensor shape')
        dtype = {0: '<f4', 1: '<f2', 2: '<i4'}.get(t.scalar(1, 'b'))
        if dtype is None:
            raise ValueError('unsupported tensor type')
        raw = bytes(buffers[t.scalar(2, 'I')].vector(0, 'B'))
        value = np.frombuffer(raw, dtype=dtype).astype('<f4') if raw else None
        if value is not None and len(value) != int(np.prod(shape)):
            raise ValueError('constant shape mismatch')
        tensors.append([shape, value])
    nodes = []
    for op in graph.vector(3, 'table'):
        code = codes[op.scalar(0, 'I')]
        inputs, outputs = op.vector(1), op.vector(2)
        if code not in OPS or len(outputs) != 1 or any(i < 0 for i in inputs):
            raise ValueError(f'unsupported operator {code}')
        if code == 6:  # Fold constant float16 -> float32 dequantization.
            if tensors[inputs[0]][1] is None:
                raise ValueError('dynamic dequantization unsupported')
            tensors[outputs[0]][1] = tensors[inputs[0]][1]
            continue
        opt, args = op.table(4), [0]*8
        if code in (3, 4, 17):
            args[:3] = [opt.scalar(0, 'b'), opt.scalar(1), opt.scalar(2)]
            if code == 3:
                args[3:6] = [opt.scalar(3, 'b'), opt.scalar(4, default=1), opt.scalar(5, default=1)]
            elif code == 4:
                args[3:7] = [opt.scalar(4, 'b'), opt.scalar(5, default=1), opt.scalar(6, default=1), opt.scalar(3)]
            else:
                args[3:6] = [opt.scalar(5, 'b'), opt.scalar(3), opt.scalar(4)]
        elif code == 2:
            args[:2] = [opt.scalar(0), opt.scalar(1, 'b')]
        elif code in (0, 18, 41, 42):
            args[0] = opt.scalar(0, 'b') if opt else 0
        elif code in (40, 74):
            args[0] = opt.scalar(0, 'b') if opt else 0
        elif code == 45:
            args[:5] = [opt.scalar(i) for i in range(5)]
        nodes.append((code, inputs, outputs[0], args))
    def ints(values):
        return struct.pack('<'+'i'*len(values), *values)
    with Path(output).open('wb') as f:
        f.write(b'VHFACE1\0')
        f.write(ints([len(tensors), len(nodes), len(graph.vector(1)), len(graph.vector(2))]))
        f.write(ints(graph.vector(1)+graph.vector(2)))
        for shape, value in tensors:
            f.write(ints([len(shape)]+shape+[1]*(4-len(shape))+[0 if value is None else len(value)]))
            if value is not None:
                f.write(value.tobytes())
        for code, inputs, output_id, args in nodes:
            f.write(ints([code, len(inputs), output_id]+args+inputs))
    return dict(tensors=len(tensors), operators=len(nodes), inputs=graph.vector(1), outputs=graph.vector(2),
                source_sha256=hashlib.sha256(data).hexdigest(),
                sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest())


def export(task, output):
    data = Path(task).read_bytes()
    if hashlib.sha256(data).hexdigest() != TASK_SHA256:
        raise ValueError('expected pinned MediaPipe float16 v1 task')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(format='vhuman.native_landmarks.v1', task_sha256=TASK_SHA256, models={})
    # Stage beside the destination: concurrent first-use exporters must never
    # expose a partially written graph. Pinned conversion is deterministic.
    with tempfile.TemporaryDirectory(prefix='.export-', dir=output) as staging:
        staging = Path(staging)
        with zipfile.ZipFile(task) as archive:
            for name in MODELS:
                receipt['models'][name] = convert(archive.read(name+'.tflite'), staging/(name+'.bin'))
        (staging/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
        for name in MODELS:
            (staging/(name+'.bin')).replace(output/(name+'.bin'))
        (staging/'manifest.json').replace(output/'manifest.json')
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True)
    print(json.dumps(export(**vars(parser.parse_args())), indent=2))
