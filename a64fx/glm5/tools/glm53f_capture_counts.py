"""Tie diagnostic captures to the requested prompt and delivered ID counts."""
import glob
import struct
from pathlib import Path


def validate_counts(prefix, prompt_tokens, output_tokens, phase):
    if prompt_tokens < 1 or output_tokens < 1:
        raise ValueError('prompt and output token counts must be positive')
    positions = prompt_tokens + output_tokens - 1
    expected = positions if phase == 'decode' else prompt_tokens
    capture = str(prefix) + ('.decode' if phase == 'decode' else '')
    records = glob.glob(capture + '.rank*.fields')
    if len(records) != 12:
        raise ValueError('exactly twelve rank manifests required')
    sparse_count = 0
    for record in records:
        for line in Path(record).read_text().splitlines()[1:]:
            parts = line.split()
            if parts[0] == 'HIDDEN' and int(parts[1]) != prompt_tokens:
                raise ValueError('prompt capture count differs from requested count')
            if parts[0] == 'SPARSE':
                path = record[:-len('.fields')] + '.layer{:02d}.sparse'.format(int(parts[1]))
                with open(path, 'rb') as stream:
                    length = struct.unpack('<i', stream.read(4))[0]
                if length != expected:
                    raise ValueError('state capture count differs from requested count')
                sparse_count += 1
    if not sparse_count:
        raise ValueError('missing sparse position metadata')
    # Route files span the whole completed run, including subsequent decode.
    for layer in range(3, 45):
        if Path('{}.layer{:02d}.routes'.format(prefix, layer)).stat().st_size != positions * 64:
            raise ValueError('route capture count differs from requested count')


def generated_ids_match(reference, candidate, expected):
    a = [int(x) for x in Path(reference).read_text().split()]
    b = [int(x) for x in Path(candidate).read_text().split()]
    return len(a) == expected and a == b and all(0 <= x < 154880 for x in a)
