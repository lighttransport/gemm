#!/usr/bin/env python3
"""Compare last-token prefill dumps from one runner and llama.cpp layer.

Capture a single prefill chunk with --decode 1 so later steps do not overwrite
runner tensors. This compares matching stages, not raw tensor-file suffixes.
"""
import argparse
import json
from pathlib import Path
import numpy as np

# The reference stores features before its token axis (ggml dimension order).
STAGES = (
    ('attn_norm', 'attn-norm', 1),
    ('linear_attn_qkv_mixed', 'ssm-qkv', 1),
    ('z', 'batch-gate', 1),
    ('alpha', 'alpha-raw', 1),
    ('conv_output_silu', 'conv-silu', 1),
    ('final_output', 'ssm-norm', 1),
    ('linear_attn_out', 'linear-out', 1),
    ('attn_post_norm', 'ffn-norm', 1),
    ('l_out', 'layer-out', 1),
)


def compare(runner, reference, layer):
    results = []
    for ref_name, runner_name, token_axis in STAGES:
        ours = runner / f'runner-{runner_name}-{layer:02d}.bin'
        matches = sorted(reference.glob(f'*-{ref_name}-{layer}.bin'))
        if len(matches) != 1:
            raise ValueError(f'{ref_name}: expected one reference tensor, got {len(matches)}')
        path = matches[0]
        meta = json.loads(Path(str(path) + '.json').read_text())
        shape, strides = meta['shape'], meta['strides']
        if meta['type'] != 0 or len(shape) != 4 or any(n <= 0 for n in shape):
            raise ValueError(f'{path}: expected nonempty F32 tensor')
        expected_stride = 4
        for n, stride in zip(shape, strides):
            if n > 1 and stride != expected_stride:
                raise ValueError(f'{path}: noncontiguous tensor is not supported')
            expected_stride *= n
        if path.stat().st_size != expected_stride or len(strides) != 4:
            raise ValueError(f'{path}: tensor byte count/strides mismatch')
        if any(n != 1 for n in shape[token_axis+1:]):
            raise ValueError(f'{path}: multiple sequences are not supported')
        width = int(np.prod(shape[:token_axis]))
        if ours.stat().st_size != width * 4:
            raise ValueError(f'{ours}: expected one {width}-element token row')
        a = np.fromfile(ours, dtype='<f4')
        b = np.fromfile(path, dtype='<f4').reshape(shape[token_axis], width)[-1]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'{ref_name}: nonfinite tensor')
        d = a.astype('f8') - b
        results.append({
            'stage': ref_name, 'elements': width,
            'bit_identical': bool(np.array_equal(a.view('u4'), b.view('u4'))),
            'max_abs': float(np.abs(d).max()),
            'relative_l2': float(np.linalg.norm(d) / max(np.linalg.norm(b.astype('f8')), 1e-30)),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runner', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--require-exact', action='store_true')
    args = parser.parse_args()
    result = compare(args.runner, args.reference, args.layer)
    print(json.dumps(result, indent=2))
    return int(args.require_exact and not all(row['bit_identical'] for row in result))


if __name__ == '__main__':
    raise SystemExit(main())
