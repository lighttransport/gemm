/* See vhuman_motion_layer.h and server/vhuman/rig/motion_layer.py (the reference). */
#include "vhuman_motion_layer.h"
#include <math.h>
#include <string.h>

const char *const vhml_control_names[VHML_COUNT] = {
    "eyeLookOutLeft", "eyeLookInLeft", "eyeLookUpLeft", "eyeLookDownLeft", "eyeBlinkLeft", "eyeWideLeft",
    "eyeLookOutRight", "eyeLookInRight", "eyeLookUpRight", "eyeLookDownRight", "eyeBlinkRight", "eyeWideRight",
    "pupilNarrow", "pupilWide", "headYaw", "headPitch", "headRoll"};

#define DEG_UP 22.0
#define DEG_DOWN 26.0
#define DEG_IN 28.0
#define DEG_OUT 32.0
#define DEG_HEAD_YAW 40.0
#define DEG_HEAD_PITCH 25.0
#define DEG_HEAD_ROLL 20.0
#define IPD 0.063

void vhml_default_config(vhml_config *c) {
    c->fixation_min = 0.22; c->fixation_max = 0.62; c->long_hold_p = 0.15; c->long_hold_max = 1.8;
    c->small_p = 0.7; c->small_deg = 4.0; c->large_yaw_deg = 12.0; c->large_pitch_deg = 6.0;
    c->saccade_base = 0.021; c->saccade_per_deg = 0.0022;
    c->micro_min = 0.6; c->micro_max = 1.5; c->micro_deg = 0.6; c->micro_duration = 0.025;
    c->drift_deg = 0.25; c->drift_hz = 0.8;
    c->blink_min = 3.0; c->blink_max = 5.0; c->blink_close = 0.075; c->blink_open = 0.13; c->blink_refractory = 1.2;
    c->blink_after_saccade_deg = 8.0; c->blink_after_saccade_p = 0.3; c->blink_at_pause_p = 0.7; c->blink_rollup = 0.1;
    c->breath_hz = 0.25; c->breath_deg = 0.3; c->head_noise_hz = 0.3;
    c->head_noise_deg[0] = 0.6; c->head_noise_deg[1] = 0.4; c->head_noise_deg[2] = 0.3;
    c->hippus = 0.05; c->hippus_hz = 0.2; c->speaking_offset_gain = 1.3; c->speaking_blink_gain = 1.2;
}

static double smoothstep(double x) {
    x = x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x;
    return x * x * (3.0 - 2.0 * x);
}

static uint32_t hash32(int32_t i, int32_t salt) {
    uint32_t x = (uint32_t)i * 0x9E3779B1u + (uint32_t)salt * 0x85EBCA77u;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    return x;
}

double vhml_value_noise(double x, int32_t salt) {
    double fl = floor(x);
    int32_t i = (int32_t)fl;
    double f = x - fl;
    double a = hash32(i, salt) / 4294967296.0 * 2.0 - 1.0;
    double b = hash32(i + 1, salt) / 4294967296.0 * 2.0 - 1.0;
    return a + (b - a) * smoothstep(f);
}

static uint32_t rng_seed(int32_t seed) {
    uint32_t x = (uint32_t)seed * 2654435761u + 1013904223u;
    return x ? x : 0x9E3779B9u;
}
static double rng_uniform(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x / 4294967296.0;
}
static double rng_between(uint32_t *s, double lo, double hi) { return lo + (hi - lo) * rng_uniform(s); }

void vhml_reset(vhml_layer *l) {
    const vhml_config *c = &l->cfg;
    l->rng = rng_seed(l->seed); l->rng_blink = rng_seed(l->seed + 101); l->rng_micro = rng_seed(l->seed + 202);
    l->time = -1.0;
    l->fix[0] = l->fix[1] = l->prev_fix[0] = l->prev_fix[1] = 0.0;
    l->saccade_start = -1.0; l->saccade_duration = 0.0;
    l->next_fixation_end = 0.4 + rng_between(&l->rng, 0.0, 0.3);
    l->micro[0] = l->micro[1] = l->micro_prev[0] = l->micro_prev[1] = 0.0; l->micro_start = -1.0;
    l->next_micro = rng_between(&l->rng_micro, c->micro_min, c->micro_max);
    l->next_blink = 0.8 + rng_between(&l->rng_blink, 0.0, 0.45);
    l->blink_start = -10.0; l->last_pause = -10.0;
}

void vhml_init(vhml_layer *l, int32_t seed, const vhml_config *cfg) {
    memset(l, 0, sizeof *l);
    if (cfg) l->cfg = *cfg; else vhml_default_config(&l->cfg);
    l->seed = seed;
    vhml_reset(l);
}

void vhml_default_input(vhml_input *in) {
    memset(in, 0, sizeof *in);
    in->focus_distance = 0.6; in->light = 0.5; in->idle = 1;
}

static void advance(vhml_layer *l, double t, int speaking, const double *pauses, int npause) {
    const vhml_config *c = &l->cfg;
    while (l->next_fixation_end <= t) {
        double start = l->next_fixation_end, gain = speaking ? c->speaking_offset_gain : 1.0, target[2], amp, hold;
        double r = rng_uniform(&l->rng);
        if (r < c->small_p) {
            target[0] = rng_between(&l->rng, -c->small_deg, c->small_deg) * gain;
            target[1] = rng_between(&l->rng, -c->small_deg * 0.6, c->small_deg * 0.6) * gain;
        } else {
            target[0] = rng_between(&l->rng, -c->large_yaw_deg, c->large_yaw_deg) * gain;
            target[1] = rng_between(&l->rng, -c->large_pitch_deg, c->large_pitch_deg) * gain;
        }
        amp = fabs(target[0] - l->fix[0]) + fabs(target[1] - l->fix[1]);
        l->prev_fix[0] = l->fix[0]; l->prev_fix[1] = l->fix[1]; l->fix[0] = target[0]; l->fix[1] = target[1];
        l->saccade_start = start;
        l->saccade_duration = c->saccade_base + c->saccade_per_deg * amp;
        hold = rng_between(&l->rng, c->fixation_min, c->fixation_max);
        if (rng_uniform(&l->rng) < c->long_hold_p) hold = rng_between(&l->rng, c->fixation_max, c->long_hold_max);
        l->next_fixation_end = start + l->saccade_duration + hold;
        l->micro[0] = l->micro[1] = l->micro_prev[0] = l->micro_prev[1] = 0.0;
        if (amp >= c->blink_after_saccade_deg && start - l->blink_start >= c->blink_refractory
            && rng_uniform(&l->rng_blink) < c->blink_after_saccade_p) {
            double b = start + l->saccade_duration * 0.5;
            if (b < l->next_blink) l->next_blink = b;
        }
    }
    while (l->next_micro <= t) {
        double start = l->next_micro;
        l->micro_prev[0] = l->micro[0]; l->micro_prev[1] = l->micro[1];
        if (l->micro[0] == 0.0 && l->micro[1] == 0.0) {
            l->micro[0] = rng_between(&l->rng_micro, -c->micro_deg, c->micro_deg);
            l->micro[1] = rng_between(&l->rng_micro, -c->micro_deg * 0.6, c->micro_deg * 0.6);
        } else { l->micro[0] = l->micro[1] = 0.0; }
        l->micro_start = start;
        l->next_micro = start + rng_between(&l->rng_micro, c->micro_min, c->micro_max);
    }
    for (int i = 0; i < npause; i++) {
        double pause = pauses[i];
        if (l->last_pause < pause && pause <= t && pause - l->blink_start >= c->blink_refractory) {
            l->last_pause = pause;
            if (rng_uniform(&l->rng_blink) < c->blink_at_pause_p) {
                if (pause + 0.05 < l->next_blink) l->next_blink = pause + 0.05;
            }
        }
    }
    while (l->next_blink <= t) {
        double gain = speaking ? 1.0 / c->speaking_blink_gain : 1.0;
        l->blink_start = l->next_blink;
        l->next_blink = l->blink_start + rng_between(&l->rng_blink, c->blink_min, c->blink_max) * gain;
    }
}

static double blink_at(const vhml_layer *l, double t) {
    const vhml_config *c = &l->cfg;
    double phase = t - l->blink_start;
    if (phase < 0.0) return 0.0;
    if (phase < c->blink_close) return smoothstep(phase / c->blink_close);
    if (phase < c->blink_close + c->blink_open) return 1.0 - smoothstep((phase - c->blink_close) / c->blink_open);
    return 0.0;
}

static double clamp01(double v) { return v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v; }

int vhml_sample(vhml_layer *l, double t, const vhml_input *in, double out[VHML_COUNT], double gaze_deg[2]) {
    const vhml_config *c = &l->cfg;
    double s, m, gx, gy, yaw, pitch, blink, verge, hippus, narrow;
    if (t < l->time) return -1;
    l->time = t;
    advance(l, t, in->speaking, in->pauses, in->pause_count);
    s = l->saccade_duration > 0.0 ? smoothstep((t - l->saccade_start) / l->saccade_duration) : 1.0;
    gx = l->prev_fix[0] + (l->fix[0] - l->prev_fix[0]) * s;
    gy = l->prev_fix[1] + (l->fix[1] - l->prev_fix[1]) * s;
    m = l->micro_start >= 0.0 ? smoothstep((t - l->micro_start) / c->micro_duration) : 1.0;
    gx += l->micro_prev[0] + (l->micro[0] - l->micro_prev[0]) * m;
    gy += l->micro_prev[1] + (l->micro[1] - l->micro_prev[1]) * m;
    gx += c->drift_deg * vhml_value_noise(t * c->drift_hz, l->seed * 7 + 1);
    gy += c->drift_deg * 0.6 * vhml_value_noise(t * c->drift_hz, l->seed * 7 + 2);
    yaw = in->focus_yaw + gx - in->head_yaw * DEG_HEAD_YAW;
    pitch = in->focus_pitch + gy - in->head_pitch * DEG_HEAD_PITCH;
    blink = blink_at(l, t);
    pitch += c->blink_rollup * DEG_UP * blink;
    verge = (IPD * 0.5 / (in->focus_distance > 0.1 ? in->focus_distance : 0.1)) * 57.29577951308232;
    for (int side = 0; side < 2; side++) {
        double sign = side == 0 ? 1.0 : -1.0, y = yaw * sign - verge;
        int base = side == 0 ? VHML_LOOK_OUT_L : VHML_LOOK_OUT_R;
        out[base + 0] = clamp01(y / DEG_OUT);
        out[base + 1] = clamp01(-y / DEG_IN);
        out[base + 2] = clamp01(pitch / DEG_UP);
        out[base + 3] = clamp01(-pitch / DEG_DOWN);
        out[base + 4] = blink;
        out[base + 5] = clamp01(0.25 * pitch / DEG_UP) * (1.0 - blink);
    }
    hippus = c->hippus * vhml_value_noise(t * c->hippus_hz, l->seed * 7 + 3);
    narrow = 0.6 * in->light + 0.3 * (verge / 8.0 < 1.0 ? verge / 8.0 : 1.0) + hippus - 0.3;
    out[VHML_PUPIL_NARROW] = clamp01(narrow);
    out[VHML_PUPIL_WIDE] = clamp01(-narrow);
    out[VHML_HEAD_YAW] = out[VHML_HEAD_PITCH] = out[VHML_HEAD_ROLL] = 0.0;
    if (in->idle) {
        double u = fmod(t * c->breath_hz, 1.0), breath = u < 0.5 ? smoothstep(u * 2.0) : smoothstep(2.0 - u * 2.0);
        out[VHML_HEAD_YAW] = c->head_noise_deg[0] * vhml_value_noise(t * c->head_noise_hz, l->seed * 7 + 4) / DEG_HEAD_YAW;
        out[VHML_HEAD_PITCH] = (c->head_noise_deg[1] * vhml_value_noise(t * c->head_noise_hz, l->seed * 7 + 5)
                                + c->breath_deg * (breath - 0.5)) / DEG_HEAD_PITCH;
        out[VHML_HEAD_ROLL] = c->head_noise_deg[2] * vhml_value_noise(t * c->head_noise_hz, l->seed * 7 + 6) / DEG_HEAD_ROLL;
    }
    if (gaze_deg) { gaze_deg[0] = yaw; gaze_deg[1] = pitch; }
    return 0;
}
