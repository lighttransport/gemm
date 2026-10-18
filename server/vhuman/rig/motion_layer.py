"""Procedural eye and idle motion: fixations, saccades, microsaccades and
drift, blinks, vestibulo-ocular compensation, convergence, pupil response,
breathing and head micro-motion (spec `vhuman.motion_layer.v1`).

The same algorithm runs in web/vhuman_motion_layer.js (viewers) and
cpu/vhuman/vhuman_motion_layer.c (native); given a seed, all three produce
the same control tracks to floating-point precision because the model uses
only +, -, *, /, comparisons and a 32-bit integer PRNG/hash: no logarithms,
trigonometry or library random numbers. Durations and amplitudes are our own
choices in the range of published oculomotor figures (fixations of a few
hundred milliseconds, a saccade main sequence of about 20 ms + 2 ms/degree,
blinks around 200 ms every 3-4 s, more after large saccades and at pauses).

Queries must be monotonic in time (reset() to seek backwards). Head pose and
the focus point are inputs per query, so the layer composes with a speech
take, a tracked performance or a live viewer.
"""
from __future__ import annotations

SPEC = "vhuman.motion_layer.v1"
# the rig's gaze ranges (rigdef.joint_matrix_entries), degrees per unit control
DEG_UP, DEG_DOWN, DEG_IN, DEG_OUT = 22.0, 26.0, 28.0, 32.0
DEG_HEAD_YAW, DEG_HEAD_PITCH, DEG_HEAD_ROLL = 40.0, 25.0, 20.0   # neck + head per unit control
IPD = 0.063

DEFAULTS = dict(
    fixation_min=0.22, fixation_max=0.62, long_hold_p=0.15, long_hold_max=1.8,
    small_p=0.7, small_deg=4.0, large_yaw_deg=12.0, large_pitch_deg=6.0,
    saccade_base=0.021, saccade_per_deg=0.0022,
    micro_min=0.6, micro_max=1.5, micro_deg=0.6, micro_duration=0.025,
    drift_deg=0.25, drift_hz=0.8,
    blink_min=3.0, blink_max=5.0, blink_close=0.075, blink_open=0.13, blink_refractory=1.2,
    blink_after_saccade_deg=8.0, blink_after_saccade_p=0.3, blink_at_pause_p=0.7, blink_rollup=0.1,
    breath_hz=0.25, breath_deg=0.3, head_noise_hz=0.3, head_noise_deg=(0.6, 0.4, 0.3),
    hippus=0.05, hippus_hz=0.2, speaking_offset_gain=1.3, speaking_blink_gain=1.2,
)


def smoothstep(x: float) -> float:
    x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    return x * x * (3.0 - 2.0 * x)


def _hash(i: int, salt: int) -> int:
    """32-bit integer hash (Math.imul-compatible)."""
    x = (i * 0x9E3779B1 + salt * 0x85EBCA77) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 12
    x = (x * 0x297A2D39) & 0xFFFFFFFF
    x ^= x >> 15
    return x


def value_noise(x: float, salt: int) -> float:
    """Smooth lattice noise in [-1, 1] (Hermite blend of hashed lattice values)."""
    i = int(x // 1.0)
    f = x - i
    a = _hash(i, salt) / 4294967296.0 * 2.0 - 1.0
    b = _hash(i + 1, salt) / 4294967296.0 * 2.0 - 1.0
    return a + (b - a) * smoothstep(f)


class Rng:
    """xorshift32; uniform() in [0, 1)."""

    def __init__(self, seed: int):
        self.x = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF or 0x9E3779B9

    def uniform(self) -> float:
        x = self.x
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.x = x & 0xFFFFFFFF
        return self.x / 4294967296.0

    def between(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.uniform()


class MotionLayer:
    def __init__(self, seed: int = 7, **config):
        self.cfg = dict(DEFAULTS, **config)
        self.seed = int(seed)
        self.reset()

    def reset(self):
        c = self.cfg
        self.rng = Rng(self.seed)
        self.rng_blink = Rng(self.seed + 101)
        self.rng_micro = Rng(self.seed + 202)
        self.time = -1.0
        # fixation/saccade state: current fixation target (deg) and the saccade in flight
        self.fix = (0.0, 0.0)
        self.prev_fix = (0.0, 0.0)
        self.saccade_start = -1.0
        self.saccade_duration = 0.0
        self.next_fixation_end = 0.4 + self.rng.between(0.0, 0.3)
        # microsaccades: a small offset that jumps back to the fixation centre
        self.micro = (0.0, 0.0)
        self.micro_prev = (0.0, 0.0)
        self.micro_start = -1.0
        self.next_micro = self.rng_micro.between(c["micro_min"], c["micro_max"])
        # blinks
        self.next_blink = 0.8 + self.rng_blink.between(0.0, 0.45)
        self.blink_start = -10.0
        self.last_pause = -10.0

    # ---- schedules ------------------------------------------------------------------------
    def _advance(self, t: float, speaking: bool, pauses):
        c = self.cfg
        while self.next_fixation_end <= t:
            start = self.next_fixation_end
            gain = c["speaking_offset_gain"] if speaking else 1.0
            r = self.rng.uniform()
            if r < c["small_p"]:
                target = (self.rng.between(-c["small_deg"], c["small_deg"]) * gain,
                          self.rng.between(-c["small_deg"] * 0.6, c["small_deg"] * 0.6) * gain)
            else:
                target = (self.rng.between(-c["large_yaw_deg"], c["large_yaw_deg"]) * gain,
                          self.rng.between(-c["large_pitch_deg"], c["large_pitch_deg"]) * gain)
            amp = abs(target[0] - self.fix[0]) + abs(target[1] - self.fix[1])
            self.prev_fix, self.fix = self.fix, target
            self.saccade_start = start
            self.saccade_duration = c["saccade_base"] + c["saccade_per_deg"] * amp
            hold = self.rng.between(c["fixation_min"], c["fixation_max"])
            if self.rng.uniform() < c["long_hold_p"]:
                hold = self.rng.between(c["fixation_max"], c["long_hold_max"])
            self.next_fixation_end = start + self.saccade_duration + hold
            self.micro = (0.0, 0.0)
            self.micro_prev = (0.0, 0.0)
            # a large saccade often ends in a blink
            if (amp >= c["blink_after_saccade_deg"] and start - self.blink_start >= c["blink_refractory"]
                    and self.rng_blink.uniform() < c["blink_after_saccade_p"]):
                self.next_blink = min(self.next_blink, start + self.saccade_duration * 0.5)
        while self.next_micro <= t:
            start = self.next_micro
            self.micro_prev = self.micro
            if self.micro == (0.0, 0.0):
                self.micro = (self.rng_micro.between(-c["micro_deg"], c["micro_deg"]),
                              self.rng_micro.between(-c["micro_deg"] * 0.6, c["micro_deg"] * 0.6))
            else:
                self.micro = (0.0, 0.0)
            self.micro_start = start
            self.next_micro = start + self.rng_micro.between(c["micro_min"], c["micro_max"])
        for pause in pauses:
            if self.last_pause < pause <= t and pause - self.blink_start >= c["blink_refractory"]:
                self.last_pause = pause
                if self.rng_blink.uniform() < c["blink_at_pause_p"]:
                    self.next_blink = min(self.next_blink, pause + 0.05)
        while self.next_blink <= t:
            self.blink_start = self.next_blink
            gain = 1.0 / c["speaking_blink_gain"] if speaking else 1.0
            self.next_blink = self.blink_start + self.rng_blink.between(c["blink_min"], c["blink_max"]) * gain

    # ---- evaluation -----------------------------------------------------------------------
    def gaze_deg(self, t: float):
        """Gaze offset from the focus (yaw, pitch degrees): saccade + microsaccade + drift."""
        c = self.cfg
        s = smoothstep((t - self.saccade_start) / self.saccade_duration) if self.saccade_duration > 0 else 1.0
        gx = self.prev_fix[0] + (self.fix[0] - self.prev_fix[0]) * s
        gy = self.prev_fix[1] + (self.fix[1] - self.prev_fix[1]) * s
        m = smoothstep((t - self.micro_start) / c["micro_duration"]) if self.micro_start >= 0 else 1.0
        gx += self.micro_prev[0] + (self.micro[0] - self.micro_prev[0]) * m
        gy += self.micro_prev[1] + (self.micro[1] - self.micro_prev[1]) * m
        gx += c["drift_deg"] * value_noise(t * c["drift_hz"], self.seed * 7 + 1)
        gy += c["drift_deg"] * 0.6 * value_noise(t * c["drift_hz"], self.seed * 7 + 2)
        return gx, gy

    def blink(self, t: float) -> float:
        c = self.cfg
        phase = t - self.blink_start
        if phase < 0.0:
            return 0.0
        if phase < c["blink_close"]:
            return smoothstep(phase / c["blink_close"])
        if phase < c["blink_close"] + c["blink_open"]:
            return 1.0 - smoothstep((phase - c["blink_close"]) / c["blink_open"])
        return 0.0

    def sample(self, t: float, *, head_yaw: float = 0.0, head_pitch: float = 0.0, focus_yaw: float = 0.0,
               focus_pitch: float = 0.0, focus_distance: float = 0.6, speaking: bool = False,
               light: float = 0.5, pauses=(), idle: bool = True) -> dict:
        """Controls at time t. head_yaw/head_pitch: the rig's signed head
        controls at t (the gaze target is held in the world: VOR); focus_*:
        where the character looks (degrees from straight ahead, metres)."""
        if t < self.time:
            raise ValueError("motion layer queries must not go backwards; call reset()")
        self.time = t
        c = self.cfg
        self._advance(t, speaking, pauses)
        gx, gy = self.gaze_deg(t)
        yaw = focus_yaw + gx - head_yaw * DEG_HEAD_YAW
        pitch = focus_pitch + gy - head_pitch * DEG_HEAD_PITCH
        blink = self.blink(t)
        pitch += c["blink_rollup"] * DEG_UP * blink                  # Bell's phenomenon: a slight roll-up
        # convergence (small-angle): each eye turns inward by half the IPD over the distance
        verge = (IPD * 0.5 / max(focus_distance, 0.1)) * 57.29577951308232
        out = {}
        for side, sign in (("Left", 1.0), ("Right", -1.0)):
            # +yaw looks to the character's left: the left eye turns out, the right eye in
            y = yaw * sign - verge
            out[f"eyeLookOut{side}"] = min(1.0, max(0.0, y / DEG_OUT))
            out[f"eyeLookIn{side}"] = min(1.0, max(0.0, -y / DEG_IN))
            out[f"eyeLookUp{side}"] = min(1.0, max(0.0, pitch / DEG_UP))
            out[f"eyeLookDown{side}"] = min(1.0, max(0.0, -pitch / DEG_DOWN))
            out[f"eyeBlink{side}"] = blink
            out[f"eyeWide{side}"] = min(1.0, max(0.0, 0.25 * pitch / DEG_UP)) * (1.0 - blink)
        hippus = c["hippus"] * value_noise(t * c["hippus_hz"], self.seed * 7 + 3)
        narrow = 0.6 * light + 0.3 * min(1.0, verge / 8.0) + hippus - 0.3
        out["pupilNarrow"] = min(1.0, max(0.0, narrow))
        out["pupilWide"] = min(1.0, max(0.0, -narrow))
        if idle:
            breath = smoothstep((t * c["breath_hz"]) % 1.0 * 2.0)             # 0..1 ramp up
            breath = breath if (t * c["breath_hz"]) % 1.0 < 0.5 else smoothstep(2.0 - (t * c["breath_hz"]) % 1.0 * 2.0)
            n = c["head_noise_deg"]
            out["headYaw"] = n[0] * value_noise(t * c["head_noise_hz"], self.seed * 7 + 4) / DEG_HEAD_YAW
            out["headPitch"] = (n[1] * value_noise(t * c["head_noise_hz"], self.seed * 7 + 5)
                                + c["breath_deg"] * (breath - 0.5)) / DEG_HEAD_PITCH
            out["headRoll"] = n[2] * value_noise(t * c["head_noise_hz"], self.seed * 7 + 6) / DEG_HEAD_ROLL
        out["_gaze_deg"] = (yaw, pitch)
        return out


def track(duration: float, fps: float = 30.0, seed: int = 7, **kw) -> list[dict]:
    """Controls at frame times 0, 1/fps, ... <= duration (a convenience for takes)."""
    layer = MotionLayer(seed)
    n = int(duration * fps) + 1
    return [layer.sample(i / fps, **kw) for i in range(n)]
