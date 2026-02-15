#include <stdio.h>
#include <stdint.h>
#include <arm_sve.h>
#include <arm_neon.h>

// Test the lane load in isolation
void test_lane_load(const int32_t* p0, const int32_t* p1, 
                    const int32_t* p2, const int32_t* p3) {
    int32x4_t v;
    asm volatile(
        "ldr s0, [%0]\n"
        "ld1 {v0.s}[1], [%1]\n"
        "ld1 {v0.s}[2], [%2]\n"
        "ld1 {v0.s}[3], [%3]\n"
        "str q0, [%4]\n"
        :
        : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(&v)
        : "v0", "memory"
    );
    
    printf("Lane load test: %d %d %d %d\n",
           vgetq_lane_s32(v, 0), vgetq_lane_s32(v, 1),
           vgetq_lane_s32(v, 2), vgetq_lane_s32(v, 3));
}

int main() {
    int32_t arr[4] = {10, 20, 30, 40};
    
    test_lane_load(&arr[0], &arr[1], &arr[2], &arr[3]);
    
    return 0;
}
