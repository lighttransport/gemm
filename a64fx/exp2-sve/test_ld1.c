#include <stdio.h>
#include <stdint.h>
#include <arm_neon.h>

int main() {
    int32_t arr[4] = {1, 2, 3, 4};
    int32x4_t v;
    
    int32_t* p0 = &arr[0];
    int32_t* p1 = &arr[1];
    int32_t* p2 = &arr[2];
    int32_t* p3 = &arr[3];
    
    // Test ldr s + ld1 lane
    float32x4_t fv;
    asm volatile(
        "ldr s0, [%0]\n"
        "ld1 {v0.s}[1], [%1]\n"
        "ld1 {v0.s}[2], [%2]\n"
        "ld1 {v0.s}[3], [%3]\n"
        "mov %0, v0.s[0]\n"
        "mov %1, v0.s[1]\n"
        "mov %2, v0.s[2]\n"
        "mov %3, v0.s[3]\n"
        : "+r"(p0), "+r"(p1), "+r"(p2), "+r"(p3)
        :
        : "v0", "memory"
    );
    
    printf("Loaded: %ld %ld %ld %ld\n", (long)p0, (long)p1, (long)p2, (long)p3);
    
    return 0;
}
