"""Make isolated-eye EXR comparison sheets using a common display transform.

Inputs are captured/rendered independently and are not geometrically registered.
A shared exposure makes the display conversion consistent; it does not establish
that the engines' scene-linear units or lighting are calibrated.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .images import display_srgb, read_exr


def preview(path, size, ev):
    rgb = read_exr(path)[..., :3]
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all() or rgb.max() <= 0:
        raise ValueError(f'invalid or empty eye render: {path}')
    return Image.fromarray(display_srgb(rgb, ev)).resize((size, size), Image.Resampling.LANCZOS)


def sheet(reference, cycles_root, sid, side, variants, out, *, size=320, ev=-1.5, profile='fast'):
    paths = [('UE eyeball', Path(reference) / sid / f'eye_{side}.exr')]
    if profile not in ('fast', 'full'):
        raise ValueError('profile must be fast or full')
    suffix = '__fast' if profile == 'fast' else ''
    paths += [(v, Path(cycles_root) / sid / f'cycles_{v}{suffix}' / f'eye_{side}' / 'beauty.exr') for v in variants]
    cols = min(4, len(paths)); rows = (len(paths) + cols - 1) // cols
    canvas = Image.new('RGB', (size * cols, (size + 28) * rows), '#202020')
    draw = ImageDraw.Draw(canvas)
    for i, (label, path) in enumerate(paths):
        x, y = i % cols * size, i // cols * (size + 28)
        canvas.paste(preview(path, size, ev), (x, y + 28))
        draw.text((x + 8, y + 7), label, fill='white')
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True); canvas.save(out, quality=94)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True, help='directory containing per-identity UE EXRs')
    p.add_argument('--cycles-root', required=True, help='directory containing per-identity Cycles results')
    p.add_argument('--id', required=True, dest='sid')
    p.add_argument('--side', choices=('left', 'right'), required=True)
    p.add_argument('--variants', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--size', type=int, default=320)
    p.add_argument('--ev', type=float, default=-1.5)
    p.add_argument('--profile', choices=('fast', 'full'), default='fast')
    a = p.parse_args()
    print(sheet(a.reference, a.cycles_root, a.sid, a.side, a.variants, a.out, size=a.size, ev=a.ev, profile=a.profile))


if __name__ == '__main__':
    main()
