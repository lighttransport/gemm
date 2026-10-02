"""Framework-free streaming adapter; Torch is an optional legacy/reference model."""
from .native_motion import MotionAdapter


def __getattr__(name):
    if name == "CausalMotion":
        from .causal_model import CausalMotion
        return CausalMotion
    raise AttributeError(name)
