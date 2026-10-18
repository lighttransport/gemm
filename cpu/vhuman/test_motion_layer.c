/* Prints the motion layer's control tracks for the parity fixture of
 * server/vhuman/test_motion_layer.py (seed 5, 301 frames at 30 fps). */
#include "../../common/vhuman_motion_layer.h"
#include <stdio.h>

int main(void) {
    vhml_layer layer; vhml_input in; double out[VHML_COUNT];
    const double pauses[2] = {3.0, 6.5};
    vhml_init(&layer, 5, NULL);
    vhml_default_input(&in);
    in.focus_distance = 0.5; in.light = 0.7; in.pauses = pauses; in.pause_count = 2; in.head_pitch = 0.1;
    printf("[");
    for (int i = 0; i <= 300; i++) {
        double t = i / 30.0;
        in.head_yaw = 0.2 * vhml_value_noise(t * 0.5, 99);
        in.speaking = i > 100;
        if (vhml_sample(&layer, t, &in, out, NULL)) return 1;
        printf("%s{", i ? "," : "");
        for (int k = 0; k < VHML_COUNT; k++) printf("%s\"%s\":%.17g", k ? "," : "", vhml_control_names[k], out[k]);
        printf("}");
    }
    printf("]\n");
    return 0;
}
