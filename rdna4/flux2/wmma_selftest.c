/*
 * wmma_selftest.c — empirical probe of gfx12 FP8 WMMA lane layout.
 *
 * Runs a single 16x16x16 v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 on known inputs
 * and dumps each lane's 8-element D fragment. We use A = row-index (small) and
 * B = column-index (small) so that C[m,n] = sum_k A[m,k]*B[k,n] has a
 * recognisable structure, letting us read back the (m,n) → lane mapping.
 *
 * Build: see Makefile target wmma_selftest.
 */

#include "../rocew.h"
#include "../hip_kernels_common.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char kernel_src[] =
"\n"
"static __device__ __forceinline__ unsigned char f32_to_fp8e4m3_rtne(float f) {\n"
"    if (f != f) return 0x7F;\n"
"    const float FP8_MAX = 448.0f;\n"
"    if (f >  FP8_MAX) f =  FP8_MAX;\n"
"    if (f < -FP8_MAX) f = -FP8_MAX;\n"
"    unsigned int bits; memcpy(&bits, &f, 4);\n"
"    unsigned int sign = (bits >> 31) & 1u;\n"
"    int exp = (int)((bits >> 23) & 0xFFu) - 127;\n"
"    unsigned int mant23 = bits & 0x7FFFFFu;\n"
"    if (exp < -9)   return (unsigned char)(sign << 7);\n"
"    if (exp <= -7) {\n"
"        unsigned int m_with_implicit = mant23 | 0x800000u;\n"
"        int shift = 20 + (-6 - exp);\n"
"        unsigned int mant3 = (m_with_implicit + (1u << (shift - 1))) >> shift;\n"
"        return (unsigned char)((sign << 7) | (mant3 & 0x7u));\n"
"    }\n"
"    int new_exp = exp + 7;\n"
"    if (new_exp >= 15) return (unsigned char)((sign << 7) | (15u << 3) | 0x6u);\n"
"    unsigned int round_bit = (mant23 >> 19) & 1u;\n"
"    unsigned int sticky = mant23 & 0x7FFFFu;\n"
"    unsigned int mant3 = (mant23 >> 20) & 0x7u;\n"
"    if (round_bit && (sticky || (mant3 & 1u))) {\n"
"        mant3 += 1u;\n"
"        if (mant3 == 8u) { mant3 = 0u; new_exp += 1; if (new_exp >= 15) return (unsigned char)((sign << 7) | (15u << 3) | 0x6u); }\n"
"    }\n"
"    return (unsigned char)((sign << 7) | ((unsigned int)new_exp << 3) | mant3);\n"
"}\n"
"__global__ void wmma_probe(float *out, const float *A_f32, const unsigned char *B_fp8) {\n"
"    int lane = threadIdx.x;\n"
"    int half = lane >> 4;\n"
"    int idx  = lane & 15;\n"
"    int k_off = half * 8;\n"
"    typedef float float8 __attribute__((ext_vector_type(8)));\n"
"    typedef int   int2_t __attribute__((ext_vector_type(2)));\n"
"    /* A load: F32 -> FP8 on the fly, lane L -> A[row=idx, k=k_off..k_off+7] */\n"
"    unsigned int a0 = 0, a1 = 0;\n"
"    int row = idx;\n"
"    for (int i = 0; i < 4; i++) a0 |= ((unsigned int)f32_to_fp8e4m3_rtne(A_f32[row*16 + k_off + i])) << (i*8);\n"
"    for (int i = 0; i < 4; i++) a1 |= ((unsigned int)f32_to_fp8e4m3_rtne(A_f32[row*16 + k_off + 4 + i])) << (i*8);\n"
"    /* Assumed layout guess B: lane L -> B[col=idx, k=k_off..k_off+7]; B stored as [N,K] */\n"
"    unsigned int b0 = 0, b1 = 0;\n"
"    int col = idx;\n"
"    for (int i = 0; i < 4; i++) b0 |= ((unsigned int)B_fp8[col*16 + k_off + i]) << (i*8);\n"
"    for (int i = 0; i < 4; i++) b1 |= ((unsigned int)B_fp8[col*16 + k_off + 4 + i]) << (i*8);\n"
"    int2_t a_vec, b_vec;\n"
"    a_vec.x = (int)a0; a_vec.y = (int)a1;\n"
"    b_vec.x = (int)b0; b_vec.y = (int)b1;\n"
"    float8 cv = {0,0,0,0,0,0,0,0};\n"
"    cv = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec, b_vec, cv);\n"
"    for (int i = 0; i < 8; i++) out[lane*8 + i] = cv[i];\n"
"}\n"
"} /* close extern \"C\" opened by hip_kernels_common_src */\n"
;

/* Reference fp8 e4m3 encode: integer values only (for N in [0..16]) */
static unsigned char enc_fp8(int n) {
    /* Represent small non-negative integer N as FP8 E4M3 */
    if (n == 0) return 0;
    int sign = 0;
    int a = n;
    int exp = 0;
    while (a >= 2) { a >>= 1; exp++; }
    /* a is now 1, exp gives the power-of-2 */
    int new_exp = exp + 7;
    /* We don't support non-power-of-2 here; use round-to-nearest */
    /* For small integers like 1,2,3,4..,16: encode exactly via FP32 → naive conversion */
    return 0;  /* placeholder */
}

static unsigned char f32_to_fp8e4m3(float f) {
    if (f == 0.0f) return 0;
    int sign = (f < 0) ? 1 : 0;
    float a = sign ? -f : f;
    if (a > 448.0f) a = 448.0f;
    int exp;
    float m = frexpf(a, &exp);
    /* m in [0.5, 1.0), so a = m * 2^exp. We want 1.xxx * 2^(exp-1). */
    m *= 2.0f; exp -= 1;
    int new_exp = exp + 7;
    if (new_exp <= 0) return (unsigned char)(sign << 7); /* subnormal → zero */
    if (new_exp >= 15) return (unsigned char)((sign<<7) | (15<<3) | 6); /* clamp 448 */
    /* mantissa in [1,2), extract top 3 bits of fractional part */
    int mant3 = (int)((m - 1.0f) * 8.0f + 0.5f);
    if (mant3 == 8) { mant3 = 0; new_exp++; if (new_exp >= 15) return (unsigned char)((sign<<7)|(15<<3)|6); }
    return (unsigned char)((sign << 7) | (new_exp << 3) | (mant3 & 7));
}

static float fp8e4m3_to_f32(unsigned char b) {
    int sign = (b >> 7) & 1;
    int exp = (b >> 3) & 0xF;
    int mant = b & 0x7;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float f;
    if (exp == 0) f = ldexpf((float)mant / 8.0f, -6);
    else          f = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
    return sign ? -f : f;
}

int main(void) {
    rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC);
    hipSetDevice(0);

    size_t len1 = strlen(hip_kernels_common_src);
    size_t len2 = strlen(kernel_src);
    char *full = (char *)malloc(len1 + len2 + 16);
    memcpy(full, hip_kernels_common_src, len1);
    memcpy(full + len1, kernel_src, len2 + 1);

    hipModule_t mod;
    if (hip_compile_kernels(&mod, 0, full, "wmma_probe.hip", 2, "wmma_selftest") < 0) {
        fprintf(stderr, "compile failed\n"); free(full); return 1;
    }
    free(full);

    hipFunction_t fn;
    hipModuleGetFunction(&fn, mod, "wmma_probe");

    /* Prepare A and B as uint8 FP8. A[row, k] = k+1 (small int). B[col, k] = col+1. */
    float Af32[16*16];
    unsigned char B[16*16];
    /* A[m,k] = random ~N(0, 1), passed as F32 to be quantized on GPU
     * B[n,k] = n + 1 (col-dep, k-indep)
     * C[m,n] = (n+1) * sum_k A[m,k]
     *
     * On CPU we quantize A the same way to get the reference. */
    srand(12345);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
            Af32[r*16 + c] = v;
            B[r*16 + c] = f32_to_fp8e4m3((float)(r + 1));
        }
    /* Compute reference using CPU FP8 quant of A */
    float C_ref[16*16];
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 16; n++) {
            float acc = 0;
            for (int k = 0; k < 16; k++) {
                float aq = fp8e4m3_to_f32(f32_to_fp8e4m3(Af32[m*16 + k]));
                float bq = fp8e4m3_to_f32(B[n*16 + k]);
                acc += aq * bq;
            }
            C_ref[m*16+n] = acc;
        }

    void *d_out, *d_A, *d_B;
    hipMalloc(&d_out, 32*8*sizeof(float));
    hipMalloc(&d_A, sizeof(Af32));
    hipMalloc(&d_B, 256);
    hipMemcpy(d_A, Af32, sizeof(Af32), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, 256, hipMemcpyHostToDevice);

    void *args[] = { &d_out, &d_A, &d_B };
    hipModuleLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL);
    hipDeviceSynchronize();

    float out[32*8];
    hipMemcpy(out, d_out, sizeof(out), hipMemcpyDeviceToHost);

    printf("Per-lane D fragment (8 floats each):\n");
    for (int L = 0; L < 32; L++) {
        printf("lane %2d:", L);
        for (int i = 0; i < 8; i++) printf(" %7.1f", out[L*8+i]);
        printf("\n");
    }

    /* Verified layout: lane L -> C[m = (L>>4)*8 + i, n = L & 15] for i=0..7 */
    int ok = 1;
    float max_err = 0;
    for (int L = 0; L < 32; L++) {
        int col = L & 15;
        int row_start = (L >> 4) * 8;
        for (int i = 0; i < 8; i++) {
            float exp = C_ref[(row_start + i)*16 + col];
            float err = fabsf(out[L*8+i] - exp);
            if (err > max_err) max_err = err;
            if (err > 1e-3f) {
                if (ok) fprintf(stderr, "first mismatch lane %d i %d: got %f expected %f\n",
                                L, i, out[L*8+i], exp);
                ok = 0;
            }
        }
    }
    printf("\nLayout lane L -> C[m=(L>>4)*8+i, n=L&15]: %s  max_err %f\n",
           ok ? "MATCH" : "NO MATCH", max_err);

    hipFree(d_out); hipFree(d_A); hipFree(d_B);
    return ok ? 0 : 1;
}
