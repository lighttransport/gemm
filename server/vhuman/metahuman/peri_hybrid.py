"""Preserve native thin-part transport around an experimental face-lighting profile.

The anatomical mask comes from staged GNM geometry, not UE target pixels.
This diagnostic requires matching static native/scattered renders and a skin AOV.
"""
import argparse
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from .images import read_exr, write_exr
from .peri_scatter import load_passes
from ..reconstruction.periocular import native_preservation_profile
from ..reconstruction.skin_scatter import preserve_native_regions


@lru_cache(maxsize=1)
def excluded_native():
    from ..rig.gnm_model import GNMModel
    model = GNMModel()
    return model.group('ears') | model.group('upper_lip') | model.group('lower_lip')


def apply(native, scattered, capture, assets):
    baseline, treated = read_exr(native), read_exr(scattered)
    p = load_passes(capture)
    if baseline.shape != treated.shape or baseline.shape[:2] != p['Combined'].shape[:2]:
        raise ValueError('native, scattered and capture resolutions must match')
    with np.load(assets, allow_pickle=False) as data:
        profile = native_preservation_profile(data, excluded_native(), p['Position'][..., :3], p['skin_mask'][..., 0])
    result = preserve_native_regions(baseline, treated, profile)
    record = dict(method='face lighting profile with native ear/lip/non-skin preservation',
                  profile='four-neighbour interpolation of static anatomical skin vertices; ears/lips excluded',
                  native=str(native), scattered=str(scattered), capture=str(capture), assets=str(assets),
                  unchanged_pixels=int((profile == 0).sum()), treated_pixels=int((profile > .99).sum()),
                  resolution=list(result.shape[:2]))
    return result, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path); parser.add_argument('scattered', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    image, record = apply(args.native, args.scattered, args.capture, args.assets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_exr(args.out, image)
    args.out.with_suffix('.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
