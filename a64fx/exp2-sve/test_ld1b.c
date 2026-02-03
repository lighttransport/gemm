#include <stdio.h>
#include <stdint.h>
#include <arm_neon.h>

int main() {
    int32_t arr[4] = {10, 20, 30, 40};
    
    int32_t* p0 = &arr[0];
    int32_t* p1 = &arr[1];
    int32_t* p2 = &arr[2];
    int32_t* p3 = &arr[3];
    
    int32x4_t v = vdupq_n_s32(0);
    
    // Load lanes one by one
    v = vld1q_lane_s32(p0, v, 0);
    v = vld1q_lane_s32(p1, v, 1);
    v = vld1q_lane_s32(p2, v, 2);
    v = vld1q_lane_s32(p3, v, 3);
    
    printf("Loaded: %d %d %d %d\n", 
           vgetq_lane_s32(v, 0),
           vgetq_lane_s32(v, 1),
           vgetq_lane_s32(v, 2),
           vgetq_lane_s32(v, 3));
    
    return 0;
}
