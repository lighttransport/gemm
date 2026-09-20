#include "fused_opt.h"
#include <stdio.h>
#include <string.h>

extern void fused_i4_f16_m1_k4_super_sve(const uint8_t *, const _Float16 *, _Float16 *);
extern void fused_fp4_f16_m1_k4_super_sve(const uint8_t *, const _Float16 *, _Float16 *);

static int weight(unsigned code, int fp4)
{
    static const int table[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    code &= 15;
    return fp4 ? table[code] : (int)(code ^ 8) - 8;
}

void pack_i16x8_full(int8_t *dst, const int16_t *src, size_t blocks)
{
    for (size_t b = 0; b < blocks; ++b)
        for (size_t k = 0; k < 128; ++k) {
            int a = src[b * 128 + k];
            int low_unsigned = (uint16_t)a & 255;
            dst[b * 256 + k] = (int8_t)(low_unsigned - 128);
            dst[b * 256 + 128 + k] = (int8_t)((a - low_unsigned) / 256);
        }
}

static int packed_weight(const uint8_t *p, size_t b, size_t n, size_t k, int fp4)
{
    unsigned byte = p[(k / 4 * 2 + b) * 128 + n / 32 * 64 + n % 16 * 4 + k % 4];
    return weight(n & 16 ? byte >> 4 : byte, fp4);
}

void pack_weight_sums(uint8_t *record, int fp4)
{
    for (size_t b = 0; b < 2; ++b)
        for (size_t n = 0; n < 64; ++n) {
            int sum = 0;
            for (size_t k = 0; k < 128; ++k)
                sum += packed_weight(record, b, n, k, fp4);
            int16_t value = (int16_t)sum;
            memcpy(record + 8192 + (b * 64 + n) * 2, &value, 2);
        }
}

static _Float16 half_fma(_Float16 w, _Float16 a, _Float16 c)
{
    _Float16 r;
    __asm__("fmadd %h0, %h1, %h2, %h3" : "=w"(r) : "w"(w), "w"(a), "w"(c));
    return r;
}

int verify_fused_opt(void)
{
    uint8_t packed[16384] __attribute__((aligned(256)));
    int16_t act[256];
    int8_t digits[512];
    int32_t out[128];
    int64_t native[128];
    uint32_t rng = 1234567;
    for (int f = 0; f < 2; ++f) {
        for (size_t i = 0; i < 8192; ++i) {
            rng = rng * 1664525u + 1013904223u;
            packed[i] = (uint8_t)(rng >> 24);
        }
        pack_weight_sums(packed, f);
        /* Every INT16 value is consumed by each format, including all
         * previously unrepresentable positive endpoints. */
        for (int pass = 0; pass < 272; ++pass) {
            for (int i = 0; i < 256; ++i)
                act[i] = pass < 256 ? (int16_t)(pass * 256 + i - 32768) :
                         (i & 1) ? INT16_MIN : INT16_MAX;
            if (pass >= 256) {
                memset(packed, (pass - 256) * 17, 8192);
                pack_weight_sums(packed, f);
            }
            pack_i16x8_full(digits, act, 2);
            for (int i = 0; i < 256; ++i) {
                int b = i / 128, k = i % 128;
                if (digits[b*256+k] + 256*digits[b*256+128+k] + 128 != act[i])
                    return fprintf(stderr, "activation reconstruction failed\n"), -1;
            }
            if (f) fused_fp4_i16x8_full_sve(packed, digits, out);
            else fused_i4_i16x8_full_sve(packed, digits, out);
            if (f) fused_fp4_i16_opt_sve(packed, act, native);
            else fused_i4_i16_opt_sve(packed, act, native);
            for (size_t b = 0; b < 2; ++b) for (size_t n = 0; n < 64; ++n) {
                int64_t ref = 0;
                for (size_t k = 0; k < 128; ++k)
                    ref += (int64_t)packed_weight(packed, b, n, k, f) * act[b*128+k];
                if (out[b*64+n] != ref || native[b*64+n] != ref)
                    return fprintf(stderr, "full SDOT mismatch f=%d pass=%d b=%zu n=%zu got=%d ref=%ld\n",
                                   f, pass, b, n, out[b*64+n], (long)ref), -1;
            }
        }
    }
    _Float16 a[512], got[256], ref[256], legacy[256];
    for (int f = 0; f < 2; ++f) for (int pass = 0; pass < 20; ++pass) {
        for (size_t i = 0; i < sizeof(packed); ++i) {
            rng = rng * 1664525u + 1013904223u;
            packed[i] = (uint8_t)(rng >> 24);
        }
        for (int i = 0; i < 512; ++i) {
            rng = rng * 1664525u + 1013904223u;
            uint16_t bits = (uint16_t)((rng & 0x83ffu) | ((pass % 16) << 10));
            if (pass == 16) bits = (i & 1) ? 0xbc00 : 0x3c00;
            if (pass == 17) bits = 0x7bff; /* overflow, including inf cancellation */
            if (pass == 18) bits = (i & 1) ? 0x8001 : 1;
            if (pass == 19) bits = (uint16_t)(rng & 0x8000);
            memcpy(a+i, &bits, 2);
        }
        for (size_t b = 0; b < 4; ++b) for (size_t n = 0; n < 64; ++n) {
            _Float16 c = 0;
            for (size_t k = 0; k < 128; ++k) {
                unsigned byte = packed[(k * 4 + b) * 32 + n % 32];
                c = half_fma((_Float16)weight(n < 32 ? byte : byte >> 4, f), a[b*128+k], c);
            }
            ref[b*64+n] = c;
        }
        if (f) fused_fp4_f16_m1_k4_super_sve(packed, a, legacy);
        else fused_i4_f16_m1_k4_super_sve(packed, a, legacy);
        if (memcmp(legacy, ref, sizeof(ref)) != 0)
            return fprintf(stderr, "legacy FP16 reference mismatch f=%d pass=%d\n", f, pass), -1;
        for (int u = 1; u <= 2; ++u) {
            if (f) {
                if (u == 1) fused_fp4_f16_opt1_sve(packed, a, got);
                else fused_fp4_f16_opt2_sve(packed, a, got);
            } else {
                if (u == 1) fused_i4_f16_opt1_sve(packed, a, got);
                else fused_i4_f16_opt2_sve(packed, a, got);
            }
            for (int i = 0; i < 256; ++i) {
                uint16_t g, r;
                memcpy(&g, got+i, 2); memcpy(&r, ref+i, 2);
                if (g != r)
                    return fprintf(stderr, "FP16 mismatch f=%d pass=%d u=%d i=%d got=%04x ref=%04x\n",
                                   f, pass, u, i, g, r), -1;
            }
        }
    }
    puts("optimized correctness: PASS (full INT16 domain; sequential FP16 bit-exact)");
    return 0;
}

void report_fused_mapping(const void *address)
{
    FILE *file = fopen("/proc/self/smaps", "r");
    char line[512];
    uintptr_t start = 0, target = (uintptr_t)address;
    int found = 0;
    if (!file) return;
    while (fgets(line, sizeof(line), file)) {
        unsigned long first, last;
        if (sscanf(line, "%lx-%lx", &first, &last) == 2) {
            found = target >= first && target < last;
            if (found) start = first;
        } else if (found && (!strncmp(line, "KernelPageSize:", 15) ||
                            !strncmp(line, "MMUPageSize:", 12) ||
                            !strncmp(line, "AnonHugePages:", 14)))
            printf("# %s", line);
    }
    fclose(file);
    file = fopen("/proc/self/numa_maps", "r");
    if (!file) return;
    while (fgets(line, sizeof(line), file)) {
        unsigned long first;
        if (sscanf(line, "%lx", &first) == 1 && first == start) printf("# numa %s", line);
    }
    fclose(file);
}
