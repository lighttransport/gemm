"""Native callback and DAC-clock bridge. Exactly one PortAudio sink per process."""
import ctypes as C
from pathlib import Path
import subprocess
import time


def build_device(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-fPIC", "-shared",
                    str(source / "pcm_ring.c"), str(source / "device.c"), "-lportaudio", "-o", str(path)], check=True)
    return path


class Event(C.Structure):
    _fields_ = [("count", C.c_uint64), ("speech_start", C.c_uint64), ("silence", C.c_int)]


class AudioDevice:
    def __init__(self, ring, library):
        self.ring = ring  # retain owner for the entire callback lifetime
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.vh_audio_open.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
        self.lib.vh_audio_open.restype = C.c_void_p
        self.lib.vh_audio_start.argtypes = [C.c_void_p]
        self.lib.vh_audio_close.argtypes = [C.c_void_p]
        self.lib.vh_audio_played.argtypes = [C.c_void_p]
        self.lib.vh_audio_played.restype = C.c_uint64
        self.lib.vh_audio_event_pop.argtypes = [C.c_void_p, C.POINTER(Event)]
        error = C.c_int()
        self.handle = self.lib.vh_audio_open(ring.handle, C.byref(error))
        if not self.handle:
            raise RuntimeError(f"PortAudio open failed ({error.value}); use --sink offline for headless replay")

    def start(self):
        code = self.lib.vh_audio_start(self.handle)
        if code:
            raise RuntimeError(f"PortAudio start failed ({code})")

    def update(self, clock):
        # Query played before draining events so it cannot exceed reported submits.
        played = self.lib.vh_audio_played(self.handle)
        timestamp = time.monotonic_ns()
        event = Event()
        while True:
            result = self.lib.vh_audio_event_pop(self.handle, C.byref(event))
            if result < 0: raise BufferError("DAC metadata queue overflow; stop session")
            if not result: break
            clock.submit(event.count, None if event.silence else event.speech_start)
        clock.observe(min(played, clock.submitted), 24000, timestamp)

    def close(self):
        if self.handle:
            self.lib.vh_audio_close(self.handle)
            self.handle = None
