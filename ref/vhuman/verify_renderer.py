"""Native rig + Gaussian CUDA acceptance; optional gsplat oracle is offline only."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from server.vhuman.realtime.src.avatar.bundle import GaussianAvatar
from server.vhuman.realtime.src.avatar.rig import RigAvatar
from server.vhuman.realtime.src.avatar.native_gpu import NativeSharedGPU
from server.vhuman.realtime.src.renderer.native import NativeGaussianRenderer
from server.vhuman.realtime.src.renderer.camera import for_avatar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rig', type=Path, required=True)
    parser.add_argument('--avatar', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--frames', type=int, default=60)
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError('frames must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    rig = RigAvatar(args.rig, args.out / 'build')
    renderer = shared = reference = handle = expected = vertices = None
    try:
        avatar = GaussianAvatar.load(args.avatar, rig.triangles)
        if tuple(avatar.metadata['control_names']) != rig.names:
            raise ValueError('avatar/rig controls differ')
        renderer = NativeGaussianRenderer(avatar, rig.triangles)
        shared = NativeSharedGPU(args.rig / 'rig_deformer.safetensors', args.out / 'build', runtime=renderer.runtime)
        if args.reference:
            from server.vhuman.realtime.src.renderer.gsplat_reference import GaussianRenderer
            reference = GaussianRenderer(avatar, rig.triangles)
        view, intrinsic = for_avatar(avatar, rig.rest)
        rows = []
        for pose in ({}, {'jawOpen': .35}, {'eyeBlinkLeft': .7, 'eyeBlinkRight': .7},
                     {'mouthSmileLeft': .6, 'mouthSmileRight': .6}):
            controls = np.zeros(len(rig.names), np.float32)
            for name, value in pose.items():
                controls[rig.names.index(name)] = value
            vertices = shared.submit(controls)
            handle = renderer.render(vertices, view, intrinsic, controls=controls)
            actual = handle.rgba.numpy()
            row = dict(pose=pose, finite=bool(np.isfinite(actual).all()))
            if reference:
                expected = reference.render(vertices.numpy(), view, intrinsic, controls=controls)
                expected.ready.synchronize()
                oracle = expected.rgba.cpu().numpy()
                error = np.abs(actual-oracle)
                row.update(max_error=float(error.max()), mean_error=float(error.mean()),
                           relative_l2=float(np.linalg.norm(actual-oracle)/max(np.linalg.norm(oracle),1e-20)))
                row['pass'] = row['finite'] and row['mean_error'] < 2e-5 and row['relative_l2'] < 2e-4
                expected.ready.close(); expected = None
            from PIL import Image
            Image.fromarray(handle.pixels()).save(args.out / f'pose-{len(rows)}.png')
            handle.ready.close(); handle.rgba.close(); handle = None
            rows.append(row)
        started = time.monotonic()
        for index in range(args.frames):
            controls[rig.names.index('jawOpen')] = .3 + .2*np.sin(index*.1)
            vertices = shared.submit(controls)
            handle = renderer.render(vertices, view, intrinsic, controls=controls)
            handle.ready.synchronize()
            handle.ready.close(); handle.rgba.close(); handle = None
        elapsed = time.monotonic()-started
        report = dict(backend='native_cuda', frames=args.frames, gaussians=renderer.n,
                      fps=args.frames/elapsed, milliseconds=elapsed*1000/args.frames,
                      memory=renderer.memory_stats(), reference_checked=args.reference, poses=rows,
                      visual_quality='not_certified')
        report['pass'] = all(r.get('pass', r['finite']) for r in rows)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report['pass'] else 1
    finally:
        vertices = None
        if handle:
            handle.ready.close(); handle.rgba.close()
        if expected:
            expected.ready.close(); expected = None
        from server.vhuman.realtime.src.pipeline.cleanup import close_resources
        close_resources(shared, renderer, reference, rig)


if __name__ == '__main__':
    raise SystemExit(main())
