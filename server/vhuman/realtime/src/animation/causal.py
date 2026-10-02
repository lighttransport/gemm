"""Framework-free streaming adapter; the Torch model is imported only for training."""
from .native_motion import MotionAdapter


def __getattr__(name):
    if name == "CausalMotion":
        from .causal_model import CausalMotion
        return CausalMotion
    raise AttributeError(name)
