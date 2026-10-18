/* Procedural eye and idle motion (spec vhuman.motion_layer.v1): the C port of
 * server/vhuman/rig/motion_layer.py, for the native realtime path. Same
 * constants, 32-bit PRNG/hash and polynomial-only math, so a seed gives the
 * same control tracks as the Python and JavaScript implementations. */
#ifndef VHUMAN_MOTION_LAYER_H
#define VHUMAN_MOTION_LAYER_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    VHML_LOOK_OUT_L, VHML_LOOK_IN_L, VHML_LOOK_UP_L, VHML_LOOK_DOWN_L, VHML_BLINK_L, VHML_WIDE_L,
    VHML_LOOK_OUT_R, VHML_LOOK_IN_R, VHML_LOOK_UP_R, VHML_LOOK_DOWN_R, VHML_BLINK_R, VHML_WIDE_R,
    VHML_PUPIL_NARROW, VHML_PUPIL_WIDE, VHML_HEAD_YAW, VHML_HEAD_PITCH, VHML_HEAD_ROLL, VHML_COUNT
};
extern const char *const vhml_control_names[VHML_COUNT];

typedef struct {
    double fixation_min, fixation_max, long_hold_p, long_hold_max;
    double small_p, small_deg, large_yaw_deg, large_pitch_deg;
    double saccade_base, saccade_per_deg;
    double micro_min, micro_max, micro_deg, micro_duration;
    double drift_deg, drift_hz;
    double blink_min, blink_max, blink_close, blink_open, blink_refractory;
    double blink_after_saccade_deg, blink_after_saccade_p, blink_at_pause_p, blink_rollup;
    double breath_hz, breath_deg, head_noise_hz, head_noise_deg[3];
    double hippus, hippus_hz, speaking_offset_gain, speaking_blink_gain;
} vhml_config;

typedef struct {
    vhml_config cfg;
    int32_t seed;
    uint32_t rng, rng_blink, rng_micro;
    double time;
    double fix[2], prev_fix[2], saccade_start, saccade_duration, next_fixation_end;
    double micro[2], micro_prev[2], micro_start, next_micro;
    double next_blink, blink_start, last_pause;
} vhml_layer;

typedef struct {
    double head_yaw, head_pitch;        /* the rig's signed head controls at t */
    double focus_yaw, focus_pitch;      /* degrees from straight ahead */
    double focus_distance;              /* metres */
    int speaking;
    double light;                       /* 0..1 */
    const double *pauses; int pause_count;   /* silence onsets (s), ascending */
    int idle;                           /* breathing and head micro-motion */
} vhml_input;

void vhml_default_config(vhml_config *cfg);
void vhml_init(vhml_layer *l, int32_t seed, const vhml_config *cfg);   /* cfg NULL: defaults */
void vhml_reset(vhml_layer *l);
void vhml_default_input(vhml_input *in);
/* Controls at time t (monotonic). Returns 0, or -1 when t goes backwards. gaze_deg (yaw, pitch) may be NULL. */
int vhml_sample(vhml_layer *l, double t, const vhml_input *in, double out[VHML_COUNT], double gaze_deg[2]);
double vhml_value_noise(double x, int32_t salt);

#ifdef __cplusplus
}
#endif
#endif
