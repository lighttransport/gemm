#include <stdio.h>
#include <stdint.h>

// Debug version that just prints args
void debug_args(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
) {
    printf("S=%p V=%p O=%p P=%p\n", S, V, O, P);
    printf("Nc=%d scale=%.4f max_val=%.4f\n", Nc, scale, max_val);
    printf("ld_s=%d ld_v=%d ld_o=%d\n", ld_s, ld_v, ld_o);
    printf("S[0]=%d S[1]=%d S[2]=%d S[3]=%d\n", S[0], S[1], S[2], S[3]);
}

int main() {
    static int32_t S[16] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    static float V[256];
    static float O[256];
    static float P[16];
    
    debug_args(S, V, O, P, 4, 1.0f, 0.0f, 4, 256, 256);
    
    return 0;
}
