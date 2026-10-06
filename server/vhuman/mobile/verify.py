"""Reproducible native mobile parity and host CPU timing (not device FPS)."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .export import validate_package
from .native import Native, build
from ..rig.gnm_model import GNMModel
from ..reconstruction.provenance import validate_candidate


def verify(candidate, package, work, poses=20):
    if not 2 <= poses <= 100: raise ValueError('pose count must be 2..100')
    manifest = validate_package(package); source = validate_candidate(candidate)
    if manifest['source_geometry_sha256'] != source['geometry_sha256']: raise ValueError('candidate mismatch')
    with np.load(Path(candidate)/'geometry.npz', allow_pickle=False) as z: g = dict(z)
    model = GNMModel(); rest, _ = model.evaluate(g['gnm_identity'])
    scale = float(g['scale']); rotation = g['rotation']
    origin = np.median(g['full_neutral']-scale*rest@rotation.T, axis=0)
    residual = (g['full_neutral']-origin)@rotation/scale-rest
    rng = np.random.default_rng(109); errors = []; timings = []
    with Native(build(work), Path(package)/'gnm.bin') as native:
        native.evaluate(g['gnm_expressions'][0])
        for index in range(poses):
            x = np.clip(g['gnm_expressions'][0]+rng.normal(0, .05, 383), -3, 3)
            r = rng.normal(0, .08, (4, 3)); r[0, 1] = -.6+1.2*index/(poses-1)
            t = rng.normal(0, .005, 3)
            started = time.perf_counter(); output = native.evaluate(x, r, t)
            timings.append((time.perf_counter()-started)*1000)
            reference, _ = model.evaluate(g['gnm_identity'], x, r, t, bind_residual=residual)
            reference = scale*reference@rotation.T+origin
            errors.append(np.linalg.norm(output-reference, axis=1)*1000)
    result = dict(poses=poses, p95_vertex_mm=float(np.percentile(errors, 95)), max_vertex_mm=float(np.max(errors)),
        cpu_median_ms=float(np.median(timings)), device='Linux host CPU; not iPhone',
        source_geometry_sha256=source['geometry_sha256'], threshold_p95_mm=.25)
    result['passed'] = result['p95_vertex_mm'] < .25
    (Path(work)/'parity.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'package', 'work'): p.add_argument('--'+name, required=True)
    p.add_argument('--poses', type=int, default=20)
    result = verify(**vars(p.parse_args())); print(json.dumps(result, indent=2))
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
