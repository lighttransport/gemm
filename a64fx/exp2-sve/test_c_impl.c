#include <stdio.h>
#include <stdint.h>
#include <math.h>

// C implementation of what the assembly should do
void exp2_c_impl(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
) {
    // Initialize accumulators
    float acc[4][4] = {0};  // 4 M rows, 4 D vectors (simplified for now)
    
    // Get row pointers like assembly does
    const int32_t* s_row[4];
    float* p_row[4];
    for (int m = 0; m < 4; m++) {
        s_row[m] = S + m * ld_s;
        p_row[m] = P + m * Nc;
    }
    
    // Main loop
    for (int k = 0; k < Nc; k++) {
        // Load S values
        int32_t s_val[4];
        for (int m = 0; m < 4; m++) {
            s_val[m] = s_row[m][k];
        }
        
        printf("k=%d: S values = %d, %d, %d, %d\n", k, s_val[0], s_val[1], s_val[2], s_val[3]);
        
        // Convert to float and compute x = S * scale - max
        float x[4];
        for (int m = 0; m < 4; m++) {
            x[m] = (float)s_val[m] * scale - max_val;
        }
        printf("       x values = %.4f, %.4f, %.4f, %.4f\n", x[0], x[1], x[2], x[3]);
        
        // Compute exp2 using FEXPA-style calculation
        float exp2_result[4];
        for (int m = 0; m < 4; m++) {
            exp2_result[m] = exp2f(x[m]);
        }
        printf("       exp2 = %.4f, %.4f, %.4f, %.4f\n", exp2_result[0], exp2_result[1], exp2_result[2], exp2_result[3]);
        
        // Store to P
        for (int m = 0; m < 4; m++) {
            p_row[m][k] = exp2_result[m];
        }
    }
}

int main() {
    static int32_t S[16] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    static float V[256];
    static float O[256];
    static float P[16];
    
    exp2_c_impl(S, V, O, P, 4, 1.0f, 0.0f, 4, 256, 256);
    
    printf("\nP output:\n");
    for (int m = 0; m < 4; m++) {
        printf("  Row %d:", m);
        for (int k = 0; k < 4; k++) {
            printf(" %.4f", P[m * 4 + k]);
        }
        printf("\n");
    }
    
    return 0;
}
