#!/usr/bin/env python3
"""Compare native RMBG2/MoGe camera inference with saved offline FP32 oracles."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from server.vhuman.native_models import recover_camera, run_image_model, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--runner', type=Path, default=ROOT / 'cpu/vhuman/vhuman_models')
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--existing-output', type=Path, help='Only compare this saved native output; do not run inference')
    args = parser.parse_args()
    fixture = args.fixture
    spec = json.loads((fixture / 'fixture.json').read_text())
    task, side = spec['task'], spec['side']
    channels = 1 if task == 'rmbg' else 4
    source = np.fromfile(fixture / 'input.f32', dtype='<f4').reshape(3, side, side)
    reference = np.fromfile(fixture / 'reference.f32', dtype='<f4').reshape(channels, side, side)
    extra = {}
    if task == 'moge':
        manifest = json.loads((fixture / 'native.json').read_text())
        assert manifest['format'] == 'vhuman.moge2_camera.v1'
        for name, digest in manifest['files'].items():
            assert sha256(fixture / name) == digest, name
        model = fixture / 'heads.safetensors'
        extra = dict(backbone=fixture / 'dinov2.safetensors', grid=(spec['grid'], spec['grid']))
    else:
        assert task == 'rmbg'
        model = Path(spec['model'])
    output = (np.fromfile(args.existing_output, dtype='<f4').reshape(reference.shape)
              if args.existing_output else run_image_model(
                  task, model, source, backend=args.backend, runner=args.runner,
                  device=args.device, threads=args.threads, **extra))
    assert np.isfinite(output).all() and np.isfinite(reference).all()
    error = abs(output.astype(np.float64) - reference)
    report = dict(task=task, reused_output=bool(args.existing_output),
                  max_abs=float(error.max()), mean_abs=float(error.mean()),
                  input_sha256=sha256(fixture / 'input.f32'),
                  reference_sha256=sha256(fixture / 'reference.f32'))
    if task == 'rmbg':
        def sigmoid(x):
            e = np.exp(-np.abs(x))
            return np.where(x >= 0, 1 / (1 + e), e / (1 + e))
        probability, oracle = sigmoid(output[0]), sigmoid(reference[0])
        alpha_error = abs(probability - oracle)
        mask_reference = Image.open(fixture / 'reference-mask.png')
        mask = Image.fromarray((probability * 255).astype(np.uint8)).resize(
            mask_reference.size, Image.Resampling.BICUBIC)
        mask_error = abs(np.asarray(mask, dtype=np.int16) - np.asarray(mask_reference, dtype=np.int16))
        report.update(alpha_max_abs=float(alpha_error.max()), alpha_mean_abs=float(alpha_error.mean()),
                      mask_max_u8=int(mask_error.max()))
        report['pass'] = bool(alpha_error.max() < 1e-3 and alpha_error.mean() < 1e-5 and mask_error.max() <= 1)
        mask.save(fixture / 'native-mask.png')
    else:
        camera = recover_camera(output[:3].transpose(1, 2, 0), output[3])
        fov_reference = 2 * math.atan(math.sqrt(2) / (2 * spec['focal']))
        report.update(focal=camera['focal'], shift=camera['shift'], fov=camera['fov'],
                      fov_abs_error=abs(camera['fov'] - fov_reference))
        report['pass'] = bool(error.max() < 1e-4 and error.mean() < 1e-5 and report['fov_abs_error'] < 1e-4)
    if not args.existing_output:
        output.astype('<f4').tofile(fixture / 'native.f32')
        report.update(backend=args.backend, runner=str(args.runner))
    (fixture / 'parity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
