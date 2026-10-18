"""The native motion layer (common/vhuman_motion_layer.c, spec vhuman.motion_layer.v1)
for the live pipeline: seeded blinks, saccades, convergence, pupil, breathing and head
micro-motion merged into the sampled speech controls between and during utterances."""
from __future__ import annotations

import ctypes as C
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
CONTROLS = ("eyeLookOutLeft", "eyeLookInLeft", "eyeLookUpLeft", "eyeLookDownLeft", "eyeBlinkLeft", "eyeWideLeft",
            "eyeLookOutRight", "eyeLookInRight", "eyeLookUpRight", "eyeLookDownRight", "eyeBlinkRight", "eyeWideRight",
            "pupilNarrow", "pupilWide", "headYaw", "headPitch", "headRoll")


class _Input(C.Structure):
    _fields_ = [("head_yaw", C.c_double), ("head_pitch", C.c_double), ("focus_yaw", C.c_double), ("focus_pitch", C.c_double),
                ("focus_distance", C.c_double), ("speaking", C.c_int), ("light", C.c_double),
                ("pauses", C.POINTER(C.c_double)), ("pause_count", C.c_int), ("idle", C.c_int)]


class NativeMotionLayer:
    """ctypes binding; `library` defaults to cpu/vhuman/libvhuman_motion_layer.so."""

    def __init__(self, seed: int = 7, *, library=None):
        path = Path(library) if library else ROOT / "cpu/vhuman/libvhuman_motion_layer.so"
        if not path.is_file():
            raise RuntimeError("Build the native motion layer with make -C cpu/vhuman libvhuman_motion_layer.so")
        self.lib = C.CDLL(str(path.resolve()))
        self.lib.vhml_init.argtypes = [C.c_void_p, C.c_int32, C.c_void_p]
        self.lib.vhml_reset.argtypes = [C.c_void_p]
        self.lib.vhml_default_input.argtypes = [C.POINTER(_Input)]
        self.lib.vhml_sample.argtypes = [C.c_void_p, C.c_double, C.POINTER(_Input), C.POINTER(C.c_double), C.POINTER(C.c_double)]
        self.lib.vhml_sample.restype = C.c_int
        self.state = C.create_string_buffer(4096)          # larger than vhml_layer
        self.lib.vhml_init(self.state, int(seed), None)
        self.out = (C.c_double * len(CONTROLS))()
        self.gaze = (C.c_double * 2)()

    def reset(self):
        self.lib.vhml_reset(self.state)

    def sample(self, t: float, *, head_yaw=0.0, head_pitch=0.0, focus_yaw=0.0, focus_pitch=0.0, focus_distance=0.6,
               speaking=False, light=0.5, pauses=(), idle=True) -> dict:
        inp = _Input()
        self.lib.vhml_default_input(C.byref(inp))
        arr = (C.c_double * max(len(pauses), 1))(*pauses)
        inp.head_yaw, inp.head_pitch, inp.focus_yaw, inp.focus_pitch = head_yaw, head_pitch, focus_yaw, focus_pitch
        inp.focus_distance, inp.speaking, inp.light, inp.idle = focus_distance, int(bool(speaking)), light, int(bool(idle))
        inp.pauses, inp.pause_count = arr, len(pauses)
        if self.lib.vhml_sample(self.state, float(t), C.byref(inp), self.out, self.gaze):
            raise ValueError("motion layer queries must not go backwards")
        out = dict(zip(CONTROLS, self.out))
        out["_gaze_deg"] = (self.gaze[0], self.gaze[1])
        return out

    def apply(self, controls: np.ndarray, names, t: float, **kw) -> np.ndarray:
        """Merge into a control vector in `names` order: blinks by max, gaze/wide/pupil set, head added."""
        index = {n: i for i, n in enumerate(names)}
        values = self.sample(t, head_yaw=float(controls[index["headYaw"]]) if "headYaw" in index else 0.0,
                             head_pitch=float(controls[index["headPitch"]]) if "headPitch" in index else 0.0, **kw)
        out = np.array(controls, np.float32, copy=True)
        for name, value in values.items():
            i = index.get(name)
            if i is None:
                continue
            if name.startswith("eyeBlink"):
                out[i] = max(out[i], value)
            elif name.startswith("head"):
                out[i] += value
            else:
                out[i] = value
        return out
