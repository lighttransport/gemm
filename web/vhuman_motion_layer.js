// Procedural eye and idle motion: the JavaScript port of
// server/vhuman/rig/motion_layer.py (spec vhuman.motion_layer.v1). Same
// constants, same 32-bit PRNG/hash and polynomial-only math, so a seed gives
// the same control tracks as the Python and C implementations.
export const SPEC = 'vhuman.motion_layer.v1';
export const DEG_UP = 22.0, DEG_DOWN = 26.0, DEG_IN = 28.0, DEG_OUT = 32.0;
export const DEG_HEAD_YAW = 40.0, DEG_HEAD_PITCH = 25.0, DEG_HEAD_ROLL = 20.0;
export const IPD = 0.063;

export const DEFAULTS = {
  fixation_min: 0.22, fixation_max: 0.62, long_hold_p: 0.15, long_hold_max: 1.8,
  small_p: 0.7, small_deg: 4.0, large_yaw_deg: 12.0, large_pitch_deg: 6.0,
  saccade_base: 0.021, saccade_per_deg: 0.0022,
  micro_min: 0.6, micro_max: 1.5, micro_deg: 0.6, micro_duration: 0.025,
  drift_deg: 0.25, drift_hz: 0.8,
  blink_min: 3.0, blink_max: 5.0, blink_close: 0.075, blink_open: 0.13, blink_refractory: 1.2,
  blink_after_saccade_deg: 8.0, blink_after_saccade_p: 0.3, blink_at_pause_p: 0.7, blink_rollup: 0.1,
  breath_hz: 0.25, breath_deg: 0.3, head_noise_hz: 0.3, head_noise_deg: [0.6, 0.4, 0.3],
  hippus: 0.05, hippus_hz: 0.2, speaking_offset_gain: 1.3, speaking_blink_gain: 1.2,
};

export function smoothstep(x) {
  x = x < 0 ? 0 : x > 1 ? 1 : x;
  return x * x * (3.0 - 2.0 * x);
}

function hash(i, salt) {
  let x = (Math.imul(i, 0x9E3779B1) + Math.imul(salt, 0x85EBCA77)) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x2C1B3C6D) >>> 0;
  x = (x ^ (x >>> 12)) >>> 0;
  x = Math.imul(x, 0x297A2D39) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  return x;
}

export function valueNoise(x, salt) {
  const i = Math.floor(x), f = x - i;
  const a = hash(i | 0, salt | 0) / 4294967296.0 * 2.0 - 1.0;
  const b = hash((i + 1) | 0, salt | 0) / 4294967296.0 * 2.0 - 1.0;
  return a + (b - a) * smoothstep(f);
}

export class Rng {
  constructor(seed) {
    this.x = ((Math.imul(seed, 2654435761) + 1013904223) >>> 0) || 0x9E3779B9;
  }
  uniform() {
    let x = this.x;
    x = (x ^ (x << 13)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ (x << 5)) >>> 0;
    this.x = x;
    return x / 4294967296.0;
  }
  between(lo, hi) { return lo + (hi - lo) * this.uniform(); }
}

export class MotionLayer {
  constructor(seed = 7, config = {}) {
    this.cfg = Object.assign({}, DEFAULTS, config);
    this.seed = seed | 0;
    this.reset();
  }
  reset() {
    const c = this.cfg;
    this.rng = new Rng(this.seed);
    this.rngBlink = new Rng(this.seed + 101);
    this.rngMicro = new Rng(this.seed + 202);
    this.time = -1.0;
    this.fix = [0, 0]; this.prevFix = [0, 0];
    this.saccadeStart = -1.0; this.saccadeDuration = 0.0;
    this.nextFixationEnd = 0.4 + this.rng.between(0.0, 0.3);
    this.micro = [0, 0]; this.microPrev = [0, 0]; this.microStart = -1.0;
    this.nextMicro = this.rngMicro.between(c.micro_min, c.micro_max);
    this.nextBlink = 0.8 + this.rngBlink.between(0.0, 0.45);
    this.blinkStart = -10.0;
    this.lastPause = -10.0;
  }
  advance(t, speaking, pauses) {
    const c = this.cfg;
    while (this.nextFixationEnd <= t) {
      const start = this.nextFixationEnd;
      const gain = speaking ? c.speaking_offset_gain : 1.0;
      const r = this.rng.uniform();
      let target;
      if (r < c.small_p) {
        target = [this.rng.between(-c.small_deg, c.small_deg) * gain,
                  this.rng.between(-c.small_deg * 0.6, c.small_deg * 0.6) * gain];
      } else {
        target = [this.rng.between(-c.large_yaw_deg, c.large_yaw_deg) * gain,
                  this.rng.between(-c.large_pitch_deg, c.large_pitch_deg) * gain];
      }
      const amp = Math.abs(target[0] - this.fix[0]) + Math.abs(target[1] - this.fix[1]);
      this.prevFix = this.fix; this.fix = target;
      this.saccadeStart = start;
      this.saccadeDuration = c.saccade_base + c.saccade_per_deg * amp;
      let hold = this.rng.between(c.fixation_min, c.fixation_max);
      if (this.rng.uniform() < c.long_hold_p) hold = this.rng.between(c.fixation_max, c.long_hold_max);
      this.nextFixationEnd = start + this.saccadeDuration + hold;
      this.micro = [0, 0]; this.microPrev = [0, 0];
      if (amp >= c.blink_after_saccade_deg && start - this.blinkStart >= c.blink_refractory
          && this.rngBlink.uniform() < c.blink_after_saccade_p) {
        this.nextBlink = Math.min(this.nextBlink, start + this.saccadeDuration * 0.5);
      }
    }
    while (this.nextMicro <= t) {
      const start = this.nextMicro;
      this.microPrev = this.micro;
      if (this.micro[0] === 0 && this.micro[1] === 0) {
        this.micro = [this.rngMicro.between(-c.micro_deg, c.micro_deg),
                      this.rngMicro.between(-c.micro_deg * 0.6, c.micro_deg * 0.6)];
      } else this.micro = [0, 0];
      this.microStart = start;
      this.nextMicro = start + this.rngMicro.between(c.micro_min, c.micro_max);
    }
    for (const pause of pauses) {
      if (this.lastPause < pause && pause <= t && pause - this.blinkStart >= c.blink_refractory) {
        this.lastPause = pause;
        if (this.rngBlink.uniform() < c.blink_at_pause_p) this.nextBlink = Math.min(this.nextBlink, pause + 0.05);
      }
    }
    while (this.nextBlink <= t) {
      this.blinkStart = this.nextBlink;
      const gain = speaking ? 1.0 / c.speaking_blink_gain : 1.0;
      this.nextBlink = this.blinkStart + this.rngBlink.between(c.blink_min, c.blink_max) * gain;
    }
  }
  gazeDeg(t) {
    const c = this.cfg;
    const s = this.saccadeDuration > 0 ? smoothstep((t - this.saccadeStart) / this.saccadeDuration) : 1.0;
    let gx = this.prevFix[0] + (this.fix[0] - this.prevFix[0]) * s;
    let gy = this.prevFix[1] + (this.fix[1] - this.prevFix[1]) * s;
    const m = this.microStart >= 0 ? smoothstep((t - this.microStart) / c.micro_duration) : 1.0;
    gx += this.microPrev[0] + (this.micro[0] - this.microPrev[0]) * m;
    gy += this.microPrev[1] + (this.micro[1] - this.microPrev[1]) * m;
    gx += c.drift_deg * valueNoise(t * c.drift_hz, this.seed * 7 + 1);
    gy += c.drift_deg * 0.6 * valueNoise(t * c.drift_hz, this.seed * 7 + 2);
    return [gx, gy];
  }
  blink(t) {
    const c = this.cfg, phase = t - this.blinkStart;
    if (phase < 0) return 0.0;
    if (phase < c.blink_close) return smoothstep(phase / c.blink_close);
    if (phase < c.blink_close + c.blink_open) return 1.0 - smoothstep((phase - c.blink_close) / c.blink_open);
    return 0.0;
  }
  // Controls at time t (monotonic queries). Options mirror motion_layer.py sample().
  sample(t, { headYaw = 0, headPitch = 0, focusYaw = 0, focusPitch = 0, focusDistance = 0.6, speaking = false,
              light = 0.5, pauses = [], idle = true } = {}) {
    if (t < this.time) throw new Error('motion layer queries must not go backwards; call reset()');
    this.time = t;
    const c = this.cfg;
    this.advance(t, speaking, pauses);
    const [gx, gy] = this.gazeDeg(t);
    const yaw = focusYaw + gx - headYaw * DEG_HEAD_YAW;
    let pitch = focusPitch + gy - headPitch * DEG_HEAD_PITCH;
    const blink = this.blink(t);
    pitch += c.blink_rollup * DEG_UP * blink;
    const verge = (IPD * 0.5 / Math.max(focusDistance, 0.1)) * 57.29577951308232;
    const out = {};
    const clamp01 = (v) => Math.min(1.0, Math.max(0.0, v));
    for (const [side, sign] of [['Left', 1.0], ['Right', -1.0]]) {
      const y = yaw * sign - verge;
      out[`eyeLookOut${side}`] = clamp01(y / DEG_OUT);
      out[`eyeLookIn${side}`] = clamp01(-y / DEG_IN);
      out[`eyeLookUp${side}`] = clamp01(pitch / DEG_UP);
      out[`eyeLookDown${side}`] = clamp01(-pitch / DEG_DOWN);
      out[`eyeBlink${side}`] = blink;
      out[`eyeWide${side}`] = clamp01(0.25 * pitch / DEG_UP) * (1.0 - blink);
    }
    const hippus = c.hippus * valueNoise(t * c.hippus_hz, this.seed * 7 + 3);
    const narrow = 0.6 * light + 0.3 * Math.min(1.0, verge / 8.0) + hippus - 0.3;
    out.pupilNarrow = clamp01(narrow);
    out.pupilWide = clamp01(-narrow);
    if (idle) {
      const u = (t * c.breath_hz) % 1.0;
      const breath = u < 0.5 ? smoothstep(u * 2.0) : smoothstep(2.0 - u * 2.0);
      const n = c.head_noise_deg;
      out.headYaw = n[0] * valueNoise(t * c.head_noise_hz, this.seed * 7 + 4) / DEG_HEAD_YAW;
      out.headPitch = (n[1] * valueNoise(t * c.head_noise_hz, this.seed * 7 + 5) + c.breath_deg * (breath - 0.5)) / DEG_HEAD_PITCH;
      out.headRoll = n[2] * valueNoise(t * c.head_noise_hz, this.seed * 7 + 6) / DEG_HEAD_ROLL;
    }
    out._gaze_deg = [yaw, pitch];
    return out;
  }
}
