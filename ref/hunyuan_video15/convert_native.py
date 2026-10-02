"""Convert diagnostic native F32 dumps into canonical NumPy axes."""
import argparse
import json
from pathlib import Path
import numpy as np

def convert(directory):
    directory = Path(directory)
    count = 0
    for raw in sorted(directory.glob('*.f32')):
        name = raw.stem
        legacy, canonical = directory / (name + '.shape.json'), directory / (name + '.json')
        if legacy.is_file():
            shape = json.loads(legacy.read_text())
            layout = None
        elif canonical.is_file():
            metadata = json.loads(canonical.read_text())
            shape, layout = metadata.get('shape'), metadata.get('layout')
            if metadata.get('dtype') != 'float32' or layout not in ('NTC', 'NCTHW'):
                raise ValueError('unsupported native tensor dtype/layout: ' + name)
        else:
            continue
        if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('invalid native tensor shape: ' + name)
        if layout is not None and len(shape) != {'NTC': 3, 'NCTHW': 5}[layout]:
            raise ValueError('native tensor rank/layout mismatch: ' + name)
        import math
        if math.prod(shape) * 4 != raw.stat().st_size:
            raise ValueError('native tensor byte count mismatch: ' + name)
        value = np.memmap(raw, dtype='<f4', mode='r', shape=tuple(shape))
        if layout is None:
            if name in ('qwen_hidden', 'byt5_hidden', 'siglip_hidden') and value.ndim == 2:
                value = value[None]
            if name in ('vae_encoded', 'vae_decoded') and value.ndim == 4:
                value = value[:, :, None, :, :]
        np.save(directory / (name + '.npy'), value, allow_pickle=False)
        count += 1
    if not count:
        raise ValueError('no native dump metadata found')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    convert(ap.parse_args().directory)

if __name__ == "__main__":
    main()
