#include <stdio.h>
#include <stdint.h>
#include <arm_sve.h>

int main() {
    svbool_t p0 = svptrue_b32();
    
    // Test FEXPA with different inputs
    // FEXPA input = ((N+127) << 6) | m
    // where N is integer part, m is fractional table index (0-63)
    
    printf("Testing FEXPA encoding:\n");
    
    for (int n = -2; n <= 3; n++) {
        int fexpa_input = ((n + 127) << 6) | 0;  // m=0 (no fraction)
        
        svint32_t v_input = svdup_s32(fexpa_input);
        svfloat32_t v_result = svexpa_f32(svreinterpret_u32(v_input));
        
        float result = svlastb_f32(p0, v_result);
        float expected = exp2f((float)n);
        
        printf("  x=%d: input=%d (0x%x), FEXPA=%.6f, expected=%.6f %s\n",
               n, fexpa_input, fexpa_input, result, expected,
               fabsf(result - expected) < 0.01 ? "OK" : "FAIL");
    }
    
    printf("\nTesting with fractional parts:\n");
    for (int n = 0; n <= 2; n++) {
        for (int m = 0; m <= 32; m += 16) {
            int fexpa_input = ((n + 127) << 6) | m;
            
            svint32_t v_input = svdup_s32(fexpa_input);
            svfloat32_t v_result = svexpa_f32(svreinterpret_u32(v_input));
            
            float result = svlastb_f32(p0, v_result);
            float expected = exp2f((float)n + (float)m/64.0f);
            
            printf("  x=%.2f (n=%d, m=%d): input=%d, FEXPA=%.4f, expected=%.4f %s\n",
                   (float)n + (float)m/64.0f, n, m, fexpa_input, result, expected,
                   fabsf(result - expected)/expected < 0.02 ? "OK" : "FAIL");
        }
    }
    
    return 0;
}
