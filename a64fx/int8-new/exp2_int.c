// exp2_int.c - Integer exp2() Approximation Implementation
// SVE 512-bit optimized for A64FX

#include "exp2_int.h"
#include <string.h>

// =============================================================================
// Pre-computed LUT: 2^(i/256) * 65536 for i in [0, 255]
// =============================================================================
// Entry i = 2^(i/256) * 65536
// i=0: 2^0 * 65536 = 65536
// i=255: 2^(255/256) * 65536 = 2^0.996 * 65536 ≈ 130855

const int32_t exp2_frac_lut[256] = {
    65536, 65714, 65892, 66071, 66250, 66430, 66610, 66791,
    66972, 67154, 67336, 67519, 67702, 67886, 68070, 68255,
    68440, 68626, 68812, 68999, 69186, 69374, 69562, 69751,
    69940, 70130, 70321, 70512, 70703, 70895, 71088, 71281,
    71475, 71669, 71864, 72059, 72255, 72451, 72648, 72846,
    73044, 73242, 73441, 73641, 73841, 74042, 74243, 74445,
    74647, 74850, 75053, 75257, 75462, 75667, 75873, 76079,
    76286, 76493, 76701, 76909, 77118, 77328, 77538, 77749,
    77960, 78172, 78384, 78597, 78811, 79025, 79240, 79455,
    79671, 79887, 80104, 80322, 80540, 80759, 80978, 81198,
    81419, 81640, 81862, 82084, 82307, 82531, 82755, 82980,
    83205, 83431, 83658, 83885, 84113, 84341, 84570, 84800,
    85030, 85261, 85493, 85725, 85958, 86191, 86425, 86660,
    86895, 87131, 87368, 87605, 87843, 88081, 88320, 88560,
    88800, 89041, 89283, 89525, 89768, 90012, 90256, 90501,
    90746, 90992, 91239, 91486, 91734, 91983, 92232, 92482,
    92733, 92984, 93236, 93489, 93742, 93996, 94251, 94506,
    94762, 95019, 95276, 95534, 95792, 96052, 96312, 96572,
    96834, 97096, 97358, 97622, 97886, 98150, 98416, 98682,
    98949, 99216, 99484, 99753, 100023, 100293, 100564, 100836,
    101108, 101381, 101655, 101930, 102205, 102481, 102758, 103035,
    103313, 103592, 103872, 104152, 104433, 104715, 104997, 105281,
    105565, 105849, 106135, 106421, 106708, 106996, 107284, 107573,
    107863, 108154, 108445, 108737, 109030, 109324, 109618, 109913,
    110209, 110506, 110803, 111101, 111400, 111700, 112000, 112302,
    112604, 112906, 113210, 113514, 113819, 114125, 114432, 114739,
    115047, 115356, 115666, 115976, 116287, 116599, 116912, 117226,
    117540, 117856, 118171, 118488, 118806, 119124, 119443, 119763,
    120084, 120405, 120728, 121051, 121375, 121700, 122025, 122352,
    122679, 123007, 123336, 123666, 123996, 124328, 124660, 124993,
    125327, 125661, 125997, 126333, 126670, 127008, 127347, 127687,
    128027, 128369, 128711, 129054, 129398, 129743, 130088, 130435,
};

// =============================================================================
// Scalar softmax exp computation
// =============================================================================

void softmax_exp_row(const int32_t* S, int32_t max_val, int32_t* out,
                     int32_t scale, int n)
{
    for (int i = 0; i < n; i++) {
        // x = (S[i] - max) * scale >> 16
        // This gives x in appropriate Q8.8 range
        int64_t diff = (int64_t)(S[i] - max_val) * scale;
        int32_t x = (int32_t)(diff >> 16);

        // exp(x) = 2^(x * log2(e))
        out[i] = exp_via_exp2(x);
    }
}

// =============================================================================
// SVE Vectorized exp2
// =============================================================================

// Process 16 INT32 values: compute 2^(x/256) for x in Q8.8
void exp2_int32_sve(const int32_t* x, int32_t* out, int n)
{
    // Process 16 elements per iteration (512-bit / 32-bit = 16)
    int i = 0;

    // Polynomial coefficients - need to load from memory for large values
    static const int32_t coeff_c1 = 45426;   // 0.693147 * 65536
    static const int32_t coeff_c2 = 15743;   // 0.240227 * 65536
    static const int32_t coeff_c3 = 3638;    // 0.0555 * 65536
    static const int32_t const_65536 = 65536;
    static const int32_t const_255 = 255;

    for (; i + 16 <= n; i += 16) {
        __asm__ volatile(
            "ptrue p0.s\n"

            // Load 16 x values
            "ld1w z0.s, p0/z, [%[x]]\n"

            // Clamp to [-2048, 0]
            "mov w9, #-2048\n"
            "mov z31.s, w9\n"
            "smax z0.s, p0/m, z0.s, z31.s\n"
            "mov z31.s, #0\n"
            "smin z0.s, p0/m, z0.s, z31.s\n"

            // n = x >> 8 (integer part, negative)
            "asr z1.s, z0.s, #8\n"

            // f = x & 0xFF (fractional part)
            // SVE AND immediate requires same src/dst, so copy first
            "mov z2.d, z0.d\n"
            "and z2.s, z2.s, #0xFF\n"

            // Compute 2^f using polynomial
            // z2 = f in [0, 255]

            // f² = f * f (predicated multiply)
            "movprfx z3, z2\n"
            "mul z3.s, p0/m, z3.s, z2.s\n"     // z3 = f²

            // f³ = f² * f
            "movprfx z4, z3\n"
            "mul z4.s, p0/m, z4.s, z2.s\n"     // z4 = f³

            // result = 65536 + (45426 * f >> 8) + (15743 * f² >> 16) + (3638 * f³ >> 24)
            // Load 65536 from register
            "ldr w9, [%[c65536]]\n"
            "mov z5.s, w9\n"                   // result = 65536

            // term1 = 45426 * f >> 8
            "ldr w9, [%[c1]]\n"
            "mov z6.s, w9\n"
            "movprfx z7, z6\n"
            "mul z7.s, p0/m, z7.s, z2.s\n"
            "asr z7.s, z7.s, #8\n"
            "add z5.s, z5.s, z7.s\n"

            // term2 = 15743 * f² >> 16
            "ldr w9, [%[c2]]\n"
            "mov z6.s, w9\n"
            "movprfx z7, z6\n"
            "mul z7.s, p0/m, z7.s, z3.s\n"
            "asr z7.s, z7.s, #16\n"
            "add z5.s, z5.s, z7.s\n"

            // term3 = 3638 * f³ >> 24
            // f³ can be up to 16581375, * 3638 = 60B (needs 64-bit)
            // Approximate: f³ >> 8 first, then * 3638 >> 16
            "ldr w9, [%[c3]]\n"
            "mov z6.s, w9\n"
            "asr z8.s, z4.s, #8\n"
            "movprfx z7, z6\n"
            "mul z7.s, p0/m, z7.s, z8.s\n"
            "asr z7.s, z7.s, #16\n"
            "add z5.s, z5.s, z7.s\n"

            // z5 now has 2^f in Q16.16
            // Apply 2^n by right shifting by |n|
            // n is in z1 (negative values)
            // shift = -n = neg(n)
            "neg z1.s, p0/m, z1.s\n"           // shift = |n|

            // Clamp shift to [0, 16]
            "mov z6.s, #16\n"
            "smin z1.s, p0/m, z1.s, z6.s\n"
            "mov z6.s, #0\n"
            "smax z1.s, p0/m, z1.s, z6.s\n"

            // Variable arithmetic right shift
            "asr z5.s, p0/m, z5.s, z1.s\n"

            // Store result
            "st1w z5.s, p0, [%[out]]\n"

            :
            : [x]"r"(x + i), [out]"r"(out + i),
              [c1]"r"(&coeff_c1), [c2]"r"(&coeff_c2), [c3]"r"(&coeff_c3),
              [c65536]"r"(&const_65536)
            : "memory", "p0", "w9", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z31"
        );
    }

    // Handle remaining elements with scalar
    for (; i < n; i++) {
        out[i] = exp2_int32(x[i]);
    }
}

// =============================================================================
// SVE Find Max
// =============================================================================

int32_t smax_int32_sve(const int32_t* x, int n)
{
    int32_t max_val = 0x80000000;  // INT_MIN

    if (n >= 16) {
        int processed = n & ~15;
        const int32_t* ptr = x;

        __asm__ volatile(
            "ptrue p0.s\n"
            "mov w9, #0x80000000\n"    // INT_MIN
            "mov z0.s, w9\n"           // Initialize max to INT_MIN

            "mov x0, %[count]\n"
            "mov x1, %[ptr]\n"

            "1:\n"
            "ld1w z1.s, p0/z, [x1]\n"
            "smax z0.s, p0/m, z0.s, z1.s\n"
            "add x1, x1, #64\n"
            "subs x0, x0, #16\n"
            "b.gt 1b\n"

            // Horizontal max
            "smaxv s0, p0, z0.s\n"
            "fmov %w[max_val], s0\n"

            : [max_val]"=r"(max_val)
            : [ptr]"r"(ptr), [count]"r"((long)processed)
            : "memory", "p0", "w9", "x0", "x1", "z0", "z1"
        );

        // Handle remainder
        for (int i = processed; i < n; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
    } else {
        for (int i = 0; i < n; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
    }

    return max_val;
}

// =============================================================================
// SVE Sum with 64-bit accumulator
// =============================================================================

int64_t sum_int32_sve(const int32_t* x, int n)
{
    int64_t sum = 0;

    // Use 64-bit accumulator to avoid overflow
    if (n >= 16) {
        int64_t partial_sum = 0;
        int processed = n & ~15;
        const int32_t* ptr = x;

        __asm__ volatile(
            "ptrue p0.s\n"
            "ptrue p1.d\n"
            "mov z0.d, #0\n"           // 64-bit accumulator

            "mov x0, %[count]\n"
            "mov x1, %[ptr]\n"

            "1:\n"
            "ld1w z1.s, p0/z, [x1]\n"
            // Widen to 64-bit and accumulate
            "sunpklo z2.d, z1.s\n"     // Lower 8 elements sign-extended to 64-bit
            "sunpkhi z3.d, z1.s\n"     // Upper 8 elements sign-extended to 64-bit
            "add z0.d, z0.d, z2.d\n"
            "add z0.d, z0.d, z3.d\n"
            "add x1, x1, #64\n"
            "subs x0, x0, #16\n"
            "b.gt 1b\n"

            // Horizontal sum of 64-bit values
            "uaddv d0, p1, z0.d\n"
            "fmov %[sum], d0\n"

            : [sum]"=r"(partial_sum)
            : [ptr]"r"(ptr), [count]"r"((long)processed)
            : "memory", "p0", "p1", "x0", "x1", "z0", "z1", "z2", "z3"
        );

        sum = partial_sum;

        // Handle remainder
        for (int i = processed; i < n; i++) {
            sum += x[i];
        }
    } else {
        for (int i = 0; i < n; i++) {
            sum += x[i];
        }
    }

    return sum;
}

// =============================================================================
// SVE Softmax Exp Row
// =============================================================================

void softmax_exp_row_sve(const int32_t* S, int32_t max_val, int32_t* out,
                         int32_t scale, int n)
{
    // Compute (S - max) * scale >> 16, then exp2

    int i = 0;

    // Pre-compute log2(e) * scale >> 8 for exp2 conversion
    // exp(x) = exp2(x * log2e)
    // We need: out = exp((S - max) * scale / 65536)
    //             = exp2((S - max) * scale / 65536 * log2e)
    // Let combined_scale = scale * log2e >> 8 (Q8.8)
    int32_t log2e_q16 = 94548;  // log2(e) in Q16.16
    int64_t combined_scale64 = ((int64_t)scale * log2e_q16) >> 16;
    int32_t combined_scale = (int32_t)combined_scale64;

    // Polynomial coefficients
    static const int32_t coeff_c1 = 45426;
    static const int32_t coeff_c2 = 15743;
    static const int32_t coeff_c3 = 3638;
    static const int32_t const_65536 = 65536;

    for (; i + 16 <= n; i += 16) {
        __asm__ volatile(
            "ptrue p0.s\n"
            "ptrue p1.d\n"

            // Load S
            "ld1w z0.s, p0/z, [%[S]]\n"

            // Subtract max
            "mov z1.s, %w[max_val]\n"
            "sub z0.s, z0.s, z1.s\n"          // z0 = S - max (<= 0)

            // Need 64-bit multiply for precision
            // Split into low/high halves
            "sunpklo z2.d, z0.s\n"            // Low 8 elements -> 64-bit
            "sunpkhi z3.d, z0.s\n"            // High 8 elements -> 64-bit

            "mov z4.d, %x[scale64]\n"         // scale in 64-bit

            "mul z2.d, p1/m, z2.d, z4.d\n"
            "mul z3.d, p1/m, z3.d, z4.d\n"

            // Shift right by 16
            "asr z2.d, z2.d, #16\n"
            "asr z3.d, z3.d, #16\n"

            // Pack back to 32-bit
            "uzp1 z0.s, z2.s, z3.s\n"         // x in Q8.8

            // Now compute exp2(x)
            // Clamp to [-2048, 0]
            "mov w9, #-2048\n"
            "mov z31.s, w9\n"
            "smax z0.s, p0/m, z0.s, z31.s\n"
            "mov z31.s, #0\n"
            "smin z0.s, p0/m, z0.s, z31.s\n"

            // n = x >> 8
            "asr z1.s, z0.s, #8\n"

            // f = x & 0xFF
            "mov z2.d, z0.d\n"
            "and z2.s, z2.s, #0xFF\n"

            // 2^f polynomial
            "movprfx z3, z2\n"
            "mul z3.s, p0/m, z3.s, z2.s\n"    // f²
            "movprfx z4, z3\n"
            "mul z4.s, p0/m, z4.s, z2.s\n"    // f³

            // result = 65536
            "ldr w9, [%[c65536]]\n"
            "mov z5.s, w9\n"

            // term1 = 45426 * f >> 8
            "ldr w9, [%[c1]]\n"
            "mov z6.s, w9\n"
            "movprfx z7, z6\n"
            "mul z7.s, p0/m, z7.s, z2.s\n"
            "asr z7.s, z7.s, #8\n"
            "add z5.s, z5.s, z7.s\n"

            // term2 = 15743 * f² >> 16
            "ldr w9, [%[c2]]\n"
            "mov z6.s, w9\n"
            "movprfx z7, z6\n"
            "mul z7.s, p0/m, z7.s, z3.s\n"
            "asr z7.s, z7.s, #16\n"
            "add z5.s, z5.s, z7.s\n"

            // term3 = 3638 * f³ >> 24 (approximate)
            "ldr w9, [%[c3]]\n"
            "mov z6.s, w9\n"
            "asr z8.s, z4.s, #8\n"
            "movprfx z7, z6\n"
            "mul z7.s, p0/m, z7.s, z8.s\n"
            "asr z7.s, z7.s, #16\n"
            "add z5.s, z5.s, z7.s\n"

            // Apply 2^n shift
            "neg z1.s, p0/m, z1.s\n"
            "mov z6.s, #16\n"
            "smin z1.s, p0/m, z1.s, z6.s\n"
            "mov z6.s, #0\n"
            "smax z1.s, p0/m, z1.s, z6.s\n"
            "asr z5.s, p0/m, z5.s, z1.s\n"

            // Store
            "st1w z5.s, p0, [%[out]]\n"

            :
            : [S]"r"(S + i), [out]"r"(out + i),
              [max_val]"r"(max_val), [scale64]"r"((int64_t)combined_scale),
              [c1]"r"(&coeff_c1), [c2]"r"(&coeff_c2), [c3]"r"(&coeff_c3),
              [c65536]"r"(&const_65536)
            : "memory", "p0", "p1", "w9", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z31"
        );
    }

    // Handle remainder
    for (; i < n; i++) {
        int64_t diff = (int64_t)(S[i] - max_val) * combined_scale;
        int32_t x = (int32_t)(diff >> 16);
        out[i] = exp2_int32(x);
    }
}
