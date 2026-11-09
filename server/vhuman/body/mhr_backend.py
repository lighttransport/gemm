"""MHR implementation of the BodyModel backend interface (delegates to mhr.py)."""
from __future__ import annotations

from .mhr import ASSET_FILES, decode, resolve_assets  # noqa: F401  (re-exported)


def faces(head_assets):
    import numpy as np
    from .assemble import _safetensor
    return np.asarray(_safetensor(head_assets, "head_pose.faces"), np.int32)


def landmark_mapping(head_assets):
    if head_assets is None:
        raise ValueError("MHR landmarks need --head-assets")
    from .assemble import _safetensor
    return _safetensor(head_assets, "head_pose.keypoint_mapping")
