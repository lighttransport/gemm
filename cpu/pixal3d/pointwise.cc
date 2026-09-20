/* Runtime-dispatched host vector operations; scalar paths preserve portability.
 * BF16/FP16 rounding and residual/modulation operation order are unchanged. */
#include "engine.hh"
#include <dlfcn.h>
#if defined(__x86_64__)
#include <immintrin.h>
#define PX_AVX2 __attribute__((target("avx2,f16c")))
#endif

namespace px {
static bool vector_available() {
#if defined(__x86_64__)
    static const bool available = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
    return available;
#else
    return false;
#endif
}
static float gelu_scalar(float x, bool approximate) {
    return approximate ? .5f * x * (1 + std::tanh(.7978845608028654f * (x + .044715f * x * x * x)))
                       : .5f * x * (1 + std::erf(x * .7071067811865475f));
}
#if defined(__x86_64__)
PX_AVX2 static __m256 round_vector(__m256 x, int precision) {
    if (precision == 2)
        return _mm256_cvtph_ps(_mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    if (!precision)
        return x;
    __m256i bits = _mm256_castps_si256(x), exponent = _mm256_set1_epi32(0x7f800000);
    __m256i special = _mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent), exponent);
    __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0x7fff),
                                    _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1)));
    bits = _mm256_add_epi32(bits, _mm256_andnot_si256(special, bias));
    return _mm256_castsi256_ps(_mm256_and_si256(bits, _mm256_set1_epi32(int(0xffff0000u))));
}
PX_AVX2 static void round_range(float *x, size_t n, int precision) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, round_vector(_mm256_loadu_ps(x + i), precision));
    for (; i < n; ++i)
        x[i] = rounded(x[i], precision);
}
PX_AVX2 static void bias_round_row(float *x, const float *bias, int columns,
                                   int precision, bool bias_is_rounded) {
    int i = 0;
    for (; i + 8 <= columns; i += 8) {
        __m256 value = _mm256_loadu_ps(x + i);
        if (bias) {
            __m256 add = _mm256_loadu_ps(bias + i);
            if (!bias_is_rounded)
                add = round_vector(add, precision);
            value = _mm256_add_ps(value, add);
        }
        _mm256_storeu_ps(x + i, round_vector(value, precision));
    }
    for (; i < columns; ++i) {
        float value = x[i] + (bias ? (bias_is_rounded ? bias[i] :
                                     rounded(bias[i], precision)) : 0);
        x[i] = rounded(value, precision);
    }
}
PX_AVX2 static void residual_row(float *x, const float *h, const float *gate, int c, bool bf) {
    int i = 0;
    for (; i + 8 <= c; i += 8) {
        __m256 v = _mm256_loadu_ps(h + i);
        if (gate)
            v = _mm256_mul_ps(v, _mm256_loadu_ps(gate + i));
        v = round_vector(v, bf);
        _mm256_storeu_ps(x + i, round_vector(_mm256_add_ps(_mm256_loadu_ps(x + i), v), bf));
    }
    for (; i < c; ++i)
        x[i] = rounded(x[i] + rounded(h[i] * (gate ? gate[i] : 1), bf), bf);
}
PX_AVX2 static void modulate_row(float *x, const float *shift, const float *scale, int c, bool bf) {
    int i = 0;
    for (; i + 8 <= c; i += 8) {
        __m256 a = round_vector(_mm256_add_ps(_mm256_set1_ps(1), _mm256_loadu_ps(scale + i)), bf);
        __m256 v = round_vector(_mm256_mul_ps(round_vector(_mm256_loadu_ps(x + i), bf), a), bf);
        _mm256_storeu_ps(x + i, round_vector(_mm256_add_ps(v, _mm256_loadu_ps(shift + i)), bf));
    }
    for (; i < c; ++i)
        x[i] = rounded(rounded(rounded(x[i], bf) * rounded(1 + scale[i], bf), bf) + shift[i], bf);
}
PX_AVX2 static void gelu_range(float *x, size_t n, bool approximate, int precision) {
    using VectorFunction = __m256 (*)(__m256);
    // libmvec is optional; no process-wide fast-math flags are used.
    static void *library = dlopen("libmvec.so.1", RTLD_LAZY | RTLD_LOCAL);
    static VectorFunction tanh_vector =
        library ? reinterpret_cast<VectorFunction>(dlsym(library, "_ZGVdN8v_tanhf")) : nullptr;
    static VectorFunction erf_vector =
        library ? reinterpret_cast<VectorFunction>(dlsym(library, "_ZGVdN8v_erff")) : nullptr;
    VectorFunction function = approximate ? tanh_vector : erf_vector;
    size_t i = 0;
    if (function)
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i), argument;
            if (approximate) {
                __m256 cube = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(.044715f), v), v), v);
                argument = _mm256_mul_ps(_mm256_set1_ps(.7978845608028654f), _mm256_add_ps(v, cube));
            } else
                argument = _mm256_mul_ps(v, _mm256_set1_ps(.7071067811865475f));
            __m256 y = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(.5f), v),
                                     _mm256_add_ps(_mm256_set1_ps(1), function(argument)));
            _mm256_storeu_ps(x + i, round_vector(y, precision));
        }
    for (; i < n; ++i)
        x[i] = rounded(gelu_scalar(x[i], approximate), precision);
}
#endif
void round_precision(Vec &x, int precision) {
    if (!precision)
        return;
    bool vector = vector_available();
#pragma omp parallel for schedule(static) if (x.size() > 16384)
    for (size_t start = 0; start < x.size(); start += 16384) {
        size_t n = std::min<size_t>(16384, x.size() - start);
#if defined(__x86_64__)
        if (vector) {
            round_range(x.data() + start, n, precision);
            continue;
        }
#else
        (void)vector;
#endif
        for (size_t i = start; i < start + n; ++i)
            x[i] = rounded(x[i], precision);
    }
}
void bias_round(float *x, const float *bias, int rows, int columns, int precision,
                bool bias_is_rounded) {
    require(x && rows > 0 && columns > 0, "Invalid bias/round dimensions");
    bool vector = vector_available();
#pragma omp parallel for schedule(static) if (size_t(rows) * columns >= 65536)
    for (int row = 0; row < rows; ++row) {
        float *values = x + size_t(row) * columns;
#if defined(__x86_64__)
        if (vector) {
            bias_round_row(values, bias, columns, precision, bias_is_rounded);
            continue;
        }
#else
        (void)vector;
#endif
        for (int i = 0; i < columns; ++i) {
            float value = values[i] + (bias ? (bias_is_rounded ? bias[i] :
                                               rounded(bias[i], precision)) : 0);
            values[i] = rounded(value, precision);
        }
    }
}
void round_bf16(Vec &x) { round_precision(x, 1); }
void add_residual(Vec &x, const Vec &h, const float *gate, int c, bool bf) {
    require(c > 0 && x.size() == h.size() && x.size() % c == 0, "Invalid residual dimensions");
    bool vector = vector_available();
#pragma omp parallel for schedule(static) if (x.size() > 16384)
    for (size_t row = 0; row < x.size() / c; ++row) {
        float *dst = x.data() + row * c;
        const float *src = h.data() + row * c;
#if defined(__x86_64__)
        if (vector) {
            residual_row(dst, src, gate, c, bf);
            continue;
        }
#else
        (void)vector;
#endif
        for (int i = 0; i < c; ++i)
            dst[i] = rounded(dst[i] + rounded(src[i] * (gate ? gate[i] : 1), bf), bf);
    }
}
void apply_modulation(Vec &x, const Vec &mod, int offset, int c, bool bf) {
    require(c > 0 && x.size() % c == 0 && offset >= 0 && mod.size() >= size_t(offset + 2 * c),
            "Invalid modulation dimensions");
    const float *shift = mod.data() + offset, *scale = shift + c;
    bool vector = vector_available();
#pragma omp parallel for schedule(static) if (x.size() > 16384)
    for (size_t row = 0; row < x.size() / c; ++row) {
        float *dst = x.data() + row * c;
#if defined(__x86_64__)
        if (vector) {
            modulate_row(dst, shift, scale, c, bf);
            continue;
        }
#else
        (void)vector;
#endif
        for (int i = 0; i < c; ++i)
            dst[i] = rounded(rounded(rounded(dst[i], bf) * rounded(1 + scale[i], bf), bf) + shift[i], bf);
    }
}
void gelu(Vec &x, bool approximate, int precision) {
    bool vector = vector_available();
#pragma omp parallel for schedule(static) if (x.size() > 16384)
    for (size_t start = 0; start < x.size(); start += 16384) {
        size_t n = std::min<size_t>(16384, x.size() - start);
#if defined(__x86_64__)
        if (vector) {
            gelu_range(x.data() + start, n, approximate, precision);
            continue;
        }
#else
        (void)vector;
#endif
        for (size_t i = start; i < start + n; ++i)
            x[i] = rounded(gelu_scalar(x[i], approximate), precision);
    }
}
} // namespace px
