#!/usr/bin/env python3
"""Compare recorded payload checksums of two completed v1 state snapshots.
The engine validates the actual payload checksums when importing a snapshot.
"""
import pathlib
import struct
import sys


def records(directory):
    result = {}
    for layer in range(65):
        p = pathlib.Path(directory) / (('meta' if layer == 64 else 'layer') + '%02d.bin' % layer)
        with p.open('rb') as f:
            raw = f.read(64)
            magic, version, model, prompt, pn, stored_layer, fmt, arith = struct.unpack('<8Q', raw)
            if magic != 0x5133385354415445 or version != 1 or stored_layer != layer:
                raise ValueError('bad state header: ' + str(p))
            result[(layer, 'header')] = raw
            if layer == 64:
                count, size = 1, 16 + 5120 * 4 + 8
            elif p.stat().st_size == 64 + 48 * (16 + (128 * 128 + 2 * 128 + 8 * 384) * 4):
                count, size = 48, 16 + (128 * 128 + 2 * 128 + 8 * 384) * 4
            else:
                count, size = 4, 16 + 2 * pn * 256 * 4
            if p.stat().st_size != 64 + count * size:
                raise ValueError('bad state length: ' + str(p))
            for head in range(count):
                f.seek(64 + head * size)
                result[(layer, head)] = f.read(16)
    return result


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('usage: compare_state.py SNAPSHOT_A SNAPSHOT_B')
    a, b = records(sys.argv[1]), records(sys.argv[2])
    different = [k for k in a if a[k] != b[k]]
    print('state checksums: compared=%d different=%d' % (len(a), len(different)))
    if different:
        print('first differences:', different[:16])
        sys.exit(1)
