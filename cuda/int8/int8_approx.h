/*
 * Integer Approximations for Neural Network Functions
 *
 * Provides fixed-point integer approximations for:
 * - exp(x)   : Exponential function (for softmax)
 * - tanh(x)  : Hyperbolic tangent (for GELU)
 * - sigmoid(x): Sigmoid function (for SiLU/Swish)
 * - gelu(x)  : Gaussian Error Linear Unit
 * - silu(x)  : Sigmoid Linear Unit (Swish)
 *
 * Fixed-point formats used:
 * - Q8.8:  8 bits integer, 8 bits fraction (range ~[-128, 127])
 * - Q16.16: 16 bits integer, 16 bits fraction
 * - Q7.24: 7 bits integer, 24 bits fraction (high precision)
 */

#ifndef INT8_APPROX_H
#define INT8_APPROX_H

#include <stdint.h>

/* Fixed-point format definitions */
#define Q8_FRAC_BITS   8
#define Q16_FRAC_BITS  16
#define Q24_FRAC_BITS  24

#define Q8_ONE         (1 << Q8_FRAC_BITS)      /* 256 */
#define Q16_ONE        (1 << Q16_FRAC_BITS)     /* 65536 */
#define Q24_ONE        (1 << Q24_FRAC_BITS)     /* 16777216 */

/* Convert float to Q8.8 fixed-point */
static inline int16_t float_to_q8(float x) {
    return (int16_t)(x * Q8_ONE);
}

/* Convert Q8.8 to float */
static inline float q8_to_float(int16_t x) {
    return (float)x / Q8_ONE;
}

/* Convert float to Q16.16 fixed-point */
static inline int32_t float_to_q16(float x) {
    return (int32_t)(x * Q16_ONE);
}

/* Convert Q16.16 to float */
static inline float q16_to_float(int32_t x) {
    return (float)x / Q16_ONE;
}

/*
 * ============================================================================
 * Integer Exponential Approximation
 * ============================================================================
 *
 * High-precision LUT with linear interpolation for exp(x).
 * For softmax, we only need relative values, so exp(x-max) works well.
 *
 * Input:  Q8.8 fixed-point
 * Output: Q8.8 fixed-point (or higher precision variants)
 */

/*
 * Lookup table for exp(x) where x is in [-8, 0] with 0.5 step (17 entries)
 * Values are in Q8.8 format: exp(x) * 256
 *
 * Computed as: round(exp(x) * 256)
 */
static const uint16_t exp_lut_q8[17] = {
    0,      /* exp(-8.0) = 0.000335 * 256 = 0.09 */
    0,      /* exp(-7.5) = 0.000553 * 256 = 0.14 */
    0,      /* exp(-7.0) = 0.000912 * 256 = 0.23 */
    0,      /* exp(-6.5) = 0.001503 * 256 = 0.38 */
    1,      /* exp(-6.0) = 0.002479 * 256 = 0.63 */
    1,      /* exp(-5.5) = 0.004087 * 256 = 1.05 */
    2,      /* exp(-5.0) = 0.006738 * 256 = 1.72 */
    3,      /* exp(-4.5) = 0.011109 * 256 = 2.84 */
    5,      /* exp(-4.0) = 0.018316 * 256 = 4.69 */
    8,      /* exp(-3.5) = 0.030197 * 256 = 7.73 */
    13,     /* exp(-3.0) = 0.049787 * 256 = 12.75 */
    21,     /* exp(-2.5) = 0.082085 * 256 = 21.01 */
    35,     /* exp(-2.0) = 0.135335 * 256 = 34.65 */
    57,     /* exp(-1.5) = 0.223130 * 256 = 57.12 */
    94,     /* exp(-1.0) = 0.367879 * 256 = 94.18 */
    155,    /* exp(-0.5) = 0.606531 * 256 = 155.27 */
    256     /* exp( 0.0) = 1.000000 * 256 = 256.00 */
};

/*
 * Higher precision LUT for exp(x) in Q16.16 format
 * For x in [-8, 0] with 0.25 step (33 entries)
 * Computed as: round(exp(x) * 65536)
 */
static const uint32_t exp_lut_q16[33] = {
    22,         /* exp(-8.00) = 0.000335 * 65536 */
    28,         /* exp(-7.75) = 0.000432 * 65536 */
    36,         /* exp(-7.50) = 0.000553 * 65536 */
    47,         /* exp(-7.25) = 0.000710 * 65536 */
    60,         /* exp(-7.00) = 0.000912 * 65536 */
    77,         /* exp(-6.75) = 0.001171 * 65536 */
    99,         /* exp(-6.50) = 0.001503 * 65536 */
    127,        /* exp(-6.25) = 0.001930 * 65536 */
    162,        /* exp(-6.00) = 0.002479 * 65536 */
    209,        /* exp(-5.75) = 0.003183 * 65536 */
    268,        /* exp(-5.50) = 0.004087 * 65536 */
    344,        /* exp(-5.25) = 0.005248 * 65536 */
    442,        /* exp(-5.00) = 0.006738 * 65536 */
    567,        /* exp(-4.75) = 0.008652 * 65536 */
    728,        /* exp(-4.50) = 0.011109 * 65536 */
    935,        /* exp(-4.25) = 0.014264 * 65536 */
    1201,       /* exp(-4.00) = 0.018316 * 65536 */
    1542,       /* exp(-3.75) = 0.023518 * 65536 */
    1979,       /* exp(-3.50) = 0.030197 * 65536 */
    2541,       /* exp(-3.25) = 0.038774 * 65536 */
    3263,       /* exp(-3.00) = 0.049787 * 65536 */
    4191,       /* exp(-2.75) = 0.063928 * 65536 */
    5381,       /* exp(-2.50) = 0.082085 * 65536 */
    6907,       /* exp(-2.25) = 0.105399 * 65536 */
    8871,       /* exp(-2.00) = 0.135335 * 65536 */
    11389,      /* exp(-1.75) = 0.173774 * 65536 */
    14625,      /* exp(-1.50) = 0.223130 * 65536 */
    18779,      /* exp(-1.25) = 0.286505 * 65536 */
    24109,      /* exp(-1.00) = 0.367879 * 65536 */
    30962,      /* exp(-0.75) = 0.472367 * 65536 */
    39760,      /* exp(-0.50) = 0.606531 * 65536 */
    51044,      /* exp(-0.25) = 0.778801 * 65536 */
    65536       /* exp( 0.00) = 1.000000 * 65536 */
};

/*
 * Integer exp approximation using LUT with linear interpolation
 *
 * Input: x in Q8.8 format (should be <= 0 for softmax stability)
 * Output: exp(x) in Q8.8 format
 */
static inline uint16_t int_exp_q8(int16_t x) {
    /* Clamp to valid range */
    if (x < -2048) return 0;  /* exp(-8) ≈ 0 */
    if (x > 512) return 0xFFFF; /* exp(2) overflow protection */

    /* For positive x, use 3rd order Taylor series */
    /* exp(x) ≈ 1 + x + x²/2 + x³/6 */
    if (x >= 0) {
        int32_t x32 = x;  /* Q8.8 */
        int32_t x2 = (x32 * x32) >> Q8_FRAC_BITS;  /* Q8.8 */
        int32_t x3 = (x2 * x32) >> Q8_FRAC_BITS;   /* Q8.8 */
        /* 1 + x + x²/2 + x³/6 in Q8.8 */
        int32_t result = Q8_ONE + x32 + (x2 >> 1) + (x3 / 6);
        if (result > 0xFFFF) return 0xFFFF;
        return (uint16_t)result;
    }

    /* For negative x, use LUT with linear interpolation */
    /* x is in Q8.8, range [-2048, 0] representing [-8.0, 0.0] */
    /* LUT step is 0.5 = 128 in Q8.8 */

    /* Convert x to positive offset from -8.0 */
    int32_t offset = (int32_t)x + 2048;  /* Now in [0, 2048] */

    /* Compute LUT index (each entry spans 128 Q8.8 units = 0.5) */
    int idx = offset >> 7;  /* Divide by 128 */
    if (idx < 0) idx = 0;
    if (idx > 15) idx = 15;

    /* Get LUT values */
    uint16_t y0 = exp_lut_q8[idx];
    uint16_t y1 = exp_lut_q8[idx + 1];

    /* Linear interpolation: y = y0 + (y1 - y0) * frac / 128 */
    int32_t frac = offset - (idx << 7);  /* Fractional part in [0, 127] */
    int32_t result = (int32_t)y0 + (((int32_t)(y1 - y0) * frac) >> 7);

    return (uint16_t)(result > 0 ? result : 0);
}

/*
 * Higher precision exp using Q16.16 LUT
 * Input: x in Q8.8 format
 * Output: exp(x) in Q16.16 format
 */
static inline uint32_t int_exp_q16_from_q8(int16_t x) {
    if (x < -2048) return 0;
    if (x > 512) return 0x7FFFFFFF;

    /* For positive x, use Taylor series */
    if (x >= 0) {
        int64_t x32 = (int64_t)x << 8;  /* Convert Q8.8 to Q16.16 */
        int64_t x2 = (x32 * x32) >> 16;
        int64_t x3 = (x2 * x32) >> 16;
        int64_t result = Q16_ONE + x32 + (x2 >> 1) + (x3 / 6);
        if (result > 0x7FFFFFFF) return 0x7FFFFFFF;
        return (uint32_t)result;
    }

    /* For negative x, use Q16 LUT with 0.25 step */
    /* x is in Q8.8, LUT step is 0.25 = 64 in Q8.8 */
    int32_t offset = (int32_t)x + 2048;  /* [0, 2048] */
    int idx = offset >> 6;  /* Divide by 64 */
    if (idx < 0) idx = 0;
    if (idx > 31) idx = 31;

    uint32_t y0 = exp_lut_q16[idx];
    uint32_t y1 = exp_lut_q16[idx + 1];

    int32_t frac = offset - (idx << 6);  /* [0, 63] */
    int64_t result = (int64_t)y0 + (((int64_t)(y1 - y0) * frac) >> 6);

    return (uint32_t)(result > 0 ? result : 0);
}

/*
 * Fast integer exp for softmax
 * Uses the identity: exp(x) = 2^(x/ln2) = 2^(x * 1.4427)
 * Then uses bit shifting for power of 2
 */
static inline uint32_t int_exp_fast_q16(int32_t x) {
    /* x is in Q16.16 format */
    /* exp(x) = 2^(x * log2(e)) where log2(e) ≈ 1.4427 */
    const int32_t LOG2_E_Q16 = 94548;  /* 1.4427 in Q16.16 */

    int64_t y = ((int64_t)x * LOG2_E_Q16) >> 16;  /* y = x * log2(e) in Q16.16 */

    /* Clamp to reasonable range */
    if (y < -32 * Q16_ONE) return 0;
    if (y > 15 * Q16_ONE) return 0x7FFFFFFF;

    /* Separate integer and fractional parts */
    int32_t int_part = (int32_t)(y >> 16);
    int32_t frac_part = (int32_t)(y & 0xFFFF);

    /* 2^frac using polynomial: 2^f ≈ 1 + 0.693*f + 0.240*f² */
    /* In Q16.16: 0.693 ≈ 45426, 0.240 ≈ 15729 */
    int32_t f2 = (frac_part * frac_part) >> 16;
    int32_t pow2_frac = Q16_ONE + ((45426 * frac_part) >> 16) + ((15729 * f2) >> 16);

    /* Apply integer part as bit shift */
    if (int_part >= 0) {
        if (int_part > 15) return 0x7FFFFFFF;
        return (uint32_t)(pow2_frac << int_part);
    } else {
        if (int_part < -31) return 0;
        return (uint32_t)(pow2_frac >> (-int_part));
    }
}

/*
 * ============================================================================
 * Integer Tanh Approximation
 * ============================================================================
 *
 * tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
 *
 * For efficiency, use piecewise linear or rational approximation:
 * tanh(x) ≈ x                           for |x| < 0.5
 * tanh(x) ≈ sign(x) * (1 - 2/(exp(2|x|)+1))  otherwise
 *
 * Or use polynomial: tanh(x) ≈ x - x³/3 + 2x⁵/15 for |x| < 1
 */

/* Tanh lookup table for x in [0, 4] with 0.25 step (17 entries) */
static const int16_t tanh_lut_q8[17] = {
    /* tanh(0) to tanh(4) in Q8.8 format (scaled by 256) */
    0,      /* tanh(0.00) = 0.000 */
    63,     /* tanh(0.25) = 0.245 */
    119,    /* tanh(0.50) = 0.462 */
    163,    /* tanh(0.75) = 0.635 */
    196,    /* tanh(1.00) = 0.762 */
    219,    /* tanh(1.25) = 0.848 */
    234,    /* tanh(1.50) = 0.905 */
    244,    /* tanh(1.75) = 0.942 */
    250,    /* tanh(2.00) = 0.964 */
    253,    /* tanh(2.25) = 0.978 */
    255,    /* tanh(2.50) = 0.987 */
    255,    /* tanh(2.75) = 0.992 */
    256,    /* tanh(3.00) = 0.995 */
    256,    /* tanh(3.25) = 0.997 */
    256,    /* tanh(3.50) = 0.998 */
    256,    /* tanh(3.75) = 0.999 */
    256     /* tanh(4.00) = 0.999 */
};

/*
 * Integer tanh approximation
 * Input: x in Q8.8 format
 * Output: tanh(x) in Q8.8 format (range [-256, 256] representing [-1, 1])
 */
static inline int16_t int_tanh_q8(int16_t x) {
    int16_t sign = 1;
    if (x < 0) {
        sign = -1;
        x = -x;
    }

    /* Saturate at |x| >= 4 (tanh ≈ ±1) */
    if (x >= 1024) {  /* 4.0 in Q8.8 */
        return sign * Q8_ONE;
    }

    /* Small x: use polynomial tanh(x) ≈ x - x³/3 */
    if (x < 128) {  /* |x| < 0.5 */
        int32_t x32 = x;
        int32_t x3 = (x32 * x32 * x32) >> (2 * Q8_FRAC_BITS);
        return sign * (int16_t)(x32 - x3 / 3);
    }

    /* Use LUT with linear interpolation */
    int idx = x >> 6;  /* Divide by 64 (0.25 in Q8.8) */
    if (idx > 15) idx = 15;

    int16_t y0 = tanh_lut_q8[idx];
    int16_t y1 = tanh_lut_q8[idx + 1];

    int32_t frac = x - (idx << 6);
    int16_t result = y0 + (int16_t)(((int32_t)(y1 - y0) * frac) >> 6);

    return sign * result;
}

/*
 * ============================================================================
 * Integer Sigmoid Approximation
 * ============================================================================
 *
 * sigmoid(x) = 1 / (1 + exp(-x)) = (1 + tanh(x/2)) / 2
 *
 * Input: x in Q8.8 format
 * Output: sigmoid(x) in Q8.8 format (range [0, 256] representing [0, 1])
 */
static inline uint16_t int_sigmoid_q8(int16_t x) {
    /* sigmoid(x) = (1 + tanh(x/2)) / 2 */
    int16_t tanh_val = int_tanh_q8(x >> 1);
    return (uint16_t)((Q8_ONE + tanh_val) >> 1);
}

/* Fast sigmoid using piecewise linear approximation */
static inline uint16_t int_sigmoid_fast_q8(int16_t x) {
    /* Piecewise linear approximation:
     * sigmoid(x) ≈ 0                  for x < -4
     * sigmoid(x) ≈ 0.5 + x/8          for -4 <= x <= 4
     * sigmoid(x) ≈ 1                  for x > 4
     */
    if (x <= -1024) return 0;           /* x <= -4 */
    if (x >= 1024) return Q8_ONE;       /* x >= 4 */

    /* Linear region: 0.5 + x/8 = 128 + x/8 in Q8.8 */
    return (uint16_t)(128 + (x >> 3));
}

/*
 * ============================================================================
 * Integer GELU Approximation
 * ============================================================================
 *
 * GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
 * GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 *
 * Simplified: GELU(x) ≈ x * sigmoid(1.702 * x)
 */

/* GELU using tanh formula (more accurate) */
static inline int16_t int_gelu_q8(int16_t x) {
    /* GELU(x) ≈ 0.5 * x * (1 + tanh(0.7978 * (x + 0.044715 * x³)))
     * Constants in Q8.8:
     * 0.7978 ≈ 204
     * 0.044715 ≈ 11
     */
    int32_t x32 = x;
    int32_t x3 = (x32 * x32 * x32) >> (2 * Q8_FRAC_BITS);

    /* inner = 0.7978 * (x + 0.044715 * x³) */
    int32_t x_plus_x3 = x32 + ((11 * x3) >> Q8_FRAC_BITS);
    int32_t inner = (204 * x_plus_x3) >> Q8_FRAC_BITS;

    /* Clamp inner to int16 range */
    if (inner > 32767) inner = 32767;
    if (inner < -32768) inner = -32768;

    int16_t tanh_val = int_tanh_q8((int16_t)inner);

    /* result = 0.5 * x * (1 + tanh_val) = x * (256 + tanh_val) / 512 */
    int32_t result = (x32 * (Q8_ONE + tanh_val)) >> (Q8_FRAC_BITS + 1);

    return (int16_t)result;
}

/* Fast GELU using sigmoid approximation */
static inline int16_t int_gelu_fast_q8(int16_t x) {
    /* GELU(x) ≈ x * sigmoid(1.702 * x)
     * 1.702 ≈ 436 in Q8.8
     */
    int32_t scaled_x = (436 * (int32_t)x) >> Q8_FRAC_BITS;
    if (scaled_x > 32767) scaled_x = 32767;
    if (scaled_x < -32768) scaled_x = -32768;

    uint16_t sig = int_sigmoid_q8((int16_t)scaled_x);
    int32_t result = ((int32_t)x * sig) >> Q8_FRAC_BITS;

    return (int16_t)result;
}

/*
 * ============================================================================
 * Integer SiLU (Swish) Approximation
 * ============================================================================
 *
 * SiLU(x) = x * sigmoid(x)
 */
static inline int16_t int_silu_q8(int16_t x) {
    uint16_t sig = int_sigmoid_q8(x);
    int32_t result = ((int32_t)x * sig) >> Q8_FRAC_BITS;
    return (int16_t)result;
}

static inline int16_t int_silu_fast_q8(int16_t x) {
    uint16_t sig = int_sigmoid_fast_q8(x);
    int32_t result = ((int32_t)x * sig) >> Q8_FRAC_BITS;
    return (int16_t)result;
}

/*
 * ============================================================================
 * Integer ReLU (for completeness)
 * ============================================================================
 */
static inline int16_t int_relu_q8(int16_t x) {
    return x > 0 ? x : 0;
}

static inline int16_t int_relu6_q8(int16_t x) {
    if (x < 0) return 0;
    if (x > 6 * Q8_ONE) return 6 * Q8_ONE;
    return x;
}

/*
 * ============================================================================
 * Softmax Utilities
 * ============================================================================
 */

/* Find max value in array (for softmax stability) */
static inline int32_t int_array_max(const int32_t* arr, int n) {
    int32_t max_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    return max_val;
}

/* Integer softmax in-place
 * Input: scores in Q16.16 format
 * Output: probabilities in Q16.16 format (sum to Q16_ONE)
 */
static inline void int_softmax_q16(int32_t* scores, int n) {
    /* Find max for numerical stability */
    int32_t max_score = int_array_max(scores, n);

    /* Compute exp(score - max) and sum */
    uint32_t sum = 0;
    uint32_t* exp_scores = (uint32_t*)scores;  /* Reuse buffer */

    for (int i = 0; i < n; i++) {
        int32_t shifted = scores[i] - max_score;
        exp_scores[i] = int_exp_fast_q16(shifted);
        sum += exp_scores[i];
    }

    /* Normalize: prob = exp_score / sum */
    if (sum == 0) sum = 1;  /* Prevent division by zero */

    for (int i = 0; i < n; i++) {
        /* Compute (exp_scores[i] * Q16_ONE) / sum */
        scores[i] = (int32_t)(((uint64_t)exp_scores[i] << 16) / sum);
    }
}

/*
 * ============================================================================
 * Integer Square Root (for LayerNorm, etc.)
 * ============================================================================
 */

/* Integer square root using Newton-Raphson */
static inline uint32_t int_sqrt(uint32_t x) {
    if (x == 0) return 0;
    if (x == 1) return 1;

    uint32_t guess = x >> 1;
    uint32_t prev;

    /* Newton-Raphson iteration */
    do {
        prev = guess;
        guess = (guess + x / guess) >> 1;
    } while (guess < prev);

    return prev;
}

/* Fixed-point square root (Q16.16 input, Q8.8 output) */
static inline uint16_t int_sqrt_q16_to_q8(uint32_t x) {
    /* sqrt(x_q16) in Q8.8 = sqrt(x * 2^16) / 2^8 = sqrt(x) * 2^4 */
    uint32_t s = int_sqrt(x);
    return (uint16_t)((s * 16) >> 8);  /* Adjust scaling */
}

/*
 * ============================================================================
 * Integer Reciprocal (for division)
 * ============================================================================
 */

/* Compute 1/x in Q16.16 format using Newton-Raphson */
static inline int32_t int_recip_q16(int32_t x) {
    if (x == 0) return 0x7FFFFFFF;  /* Return max for division by zero */

    int sign = 1;
    if (x < 0) {
        sign = -1;
        x = -x;
    }

    /* Initial guess based on leading zeros */
    int lz = __builtin_clz((uint32_t)x);
    uint32_t guess = (1U << (31 - lz + 16)) / (uint32_t)x;

    /* Newton-Raphson: x_{n+1} = x_n * (2 - a * x_n) */
    for (int i = 0; i < 2; i++) {
        uint64_t ax = ((uint64_t)x * guess) >> 16;
        uint64_t two_minus_ax = (2ULL << 16) - ax;
        guess = (uint32_t)((guess * two_minus_ax) >> 16);
    }

    return sign * (int32_t)guess;
}

#endif /* INT8_APPROX_H */
