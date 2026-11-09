"""Body-model backends (MHR today; SOMA-X is added behind the same interface).

A backend decodes SAM 3D Body parameters into a posed mesh and a global joint
state. ``decode`` keeps the backend's native length unit; ``unit_to_m`` converts
it to metres so callers never hard-code a unit. Joint states are
``[tx, ty, tz, qx, qy, qz, qw, scale]`` per joint, parents precede children.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BodyModel:
    name: str            # CLI/API identifier
    label: str           # human-readable, written to avatar.json
    prefix: str          # joint-name prefix in exported rigs
    root_joint: str
    head_joint: str      # face rig is re-parented here
    neck_joint: str      # body/head cut height
    unit_to_m: float     # native length unit -> metres
    pose_dim: int        # SAM 3D Body model_params length
    identity_dim: int    # SAM 3D Body shape length
    bind_note: str

    def resolve_assets(self, assets=None, model=None) -> Path:
        return _impl(self.name).resolve_assets(assets, model)

    def decode(self, assets, params, identity, face=None, **kw):
        """-> (vertices[B,V,3] | None, state[B,J,8], rig, report) in native units."""
        return _impl(self.name).decode(assets, params, identity, face, **kw)

    def asset_files(self) -> tuple[str, ...]:
        return tuple(_impl(self.name).ASSET_FILES)

    def faces(self, head_assets: Path) -> np.ndarray:
        return _impl(self.name).faces(head_assets)

    def landmark_mapping(self, head_assets: Path | None = None) -> np.ndarray:
        """[>=5, V+J] weights for nose, left eye, right eye, left ear, right ear."""
        return _impl(self.name).landmark_mapping(head_assets)

    def check_params(self, params: np.ndarray, identity: np.ndarray) -> None:
        if params.shape != (self.pose_dim,) or identity.shape != (self.identity_dim,):
            raise ValueError(f"SAM 3D Body sidecar lacks decoded {self.label} pose or identity")


MHR = BodyModel(name="mhr", label="MHR", prefix="mhr_", root_joint="mhr_root",
                head_joint="mhr_c_head", neck_joint="mhr_c_neck", unit_to_m=.01,
                pose_dim=204, identity_dim=45, bind_note="SAM 3D Body prediction")

SOMA = BodyModel(name="soma", label="SOMA-X", prefix="soma_", root_joint="soma_Root",
                 head_joint="soma_Head", neck_joint="soma_Neck1", unit_to_m=1.0,
                 pose_dim=204, identity_dim=45,
                 bind_note="SAM 3D Body prediction (MHR identity, SOMA pose inversion)")

MODELS = {"mhr": MHR, "soma": SOMA}
DEFAULT = "mhr"


def get(name: str | None = None) -> BodyModel:
    name = name or DEFAULT
    if name not in MODELS:
        raise ValueError(f"unknown body model {name!r}; available: {', '.join(MODELS)}")
    return MODELS[name]


def from_avatar(definition: dict) -> BodyModel:
    """Body model that built an avatar.json (avatars predating --body-model are MHR)."""
    label = (definition.get("body") or {}).get("model", "MHR")
    for bm in MODELS.values():
        if bm.label == label:
            return bm
    raise ValueError(f"avatar was built with unknown body model {label!r}")


def _impl(name: str):
    if name == "mhr":
        from . import mhr_backend
        return mhr_backend
    if name == "soma":
        from . import soma_backend
        return soma_backend
    raise ValueError(f"no implementation for body model {name!r}")
