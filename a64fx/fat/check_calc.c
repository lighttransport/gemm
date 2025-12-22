#include <stdio.h>
#include <math.h>
#include <stdint.h>

int main() {
    float output = 8.9283770649e-05f;
    float ref = 8.8668952230e-05f;
    
    float err = fabsf(output - ref);
    float rel = (ref != 0) ? err / fabsf(ref) : err;
    
    printf("output = %.15e\n", output);
    printf("ref    = %.15e\n", ref);
    printf("err    = %.15e\n", err);
    printf("rel    = %.15e\n", rel);
    
    // Check bit patterns
    union { float f; uint32_t u; } uo, ur, ue, urr;
    uo.f = output;
    ur.f = ref;
    ue.f = err;
    urr.f = rel;
    printf("\nBit patterns:\n");
    printf("output = 0x%08x\n", uo.u);
    printf("ref    = 0x%08x\n", ur.u);
    printf("err    = 0x%08x\n", ue.u);
    printf("rel    = 0x%08x\n", urr.u);
    
    return 0;
}
