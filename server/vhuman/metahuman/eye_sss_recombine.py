"""Reference for Unreal's linear-light subsurface recombination.

The operands correspond to PostProcessSubsurface.usf's
``SubsurfaceRecombinePS`` after lighting reconstruction and profile filtering.
This function does not calculate a subsurface profile, reconstruct lighting,
handle image alpha, or supply the required Cycles render passes.
"""
from __future__ import annotations

import numpy as np


def recombine(diffuse, scattered, base_color, specular, tint):
    """Combine matching RGB arrays, with a scalar or RGB profile tint in [0, 1].

    Inputs must be scene-linear and non-premultiplied, and represent the same
    pixels. ``tint`` includes any scattering fade (UE's ``FadedTint``). The
    caller is responsible for restricting this operation to the eye surface.
    """
    diffuse, scattered, base_color, specular = (
        np.asarray(x, dtype=np.float32) for x in (diffuse, scattered, base_color, specular)
    )
    if diffuse.ndim < 1 or diffuse.shape[-1] != 3 or any(
        x.shape != diffuse.shape for x in (scattered, base_color, specular)
    ):
        raise ValueError('lighting and base-color inputs must have matching RGB shapes')
    tint = np.asarray(tint, dtype=np.float32)
    if tint.shape not in ((), (3,), diffuse.shape):
        raise ValueError('tint must be a scalar, RGB triplet, or matching RGB array')
    if not all(np.isfinite(x).all() for x in (diffuse, scattered, base_color, specular, tint)):
        raise ValueError('inputs must be finite')
    if np.any((tint < 0.) | (tint > 1.)):
        raise ValueError('tint must be between 0 and 1')
    return (diffuse + (scattered - diffuse) * tint) * base_color + specular
