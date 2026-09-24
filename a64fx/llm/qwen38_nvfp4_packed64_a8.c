/* Experimental one-token W4A8 GEMV over load-time-repacked compact FP4.
 * Each 64-row x 16-column tile is 512 packed code bytes + 256 scale bytes,
 * exactly the 768 bytes occupied by eight source NVFP4 subblocks. */
#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

typedef struct { float d[8]; uint8_t qs[64]; } q38_p64_source_subblock;
typedef struct { q38_p64_source_subblock s[4]; } q38_p64_source_block;
typedef struct {
    uint8_t codes[2][4][64];
    float scales[64];
} q38_p64_tile;
_Static_assert(sizeof(q38_p64_tile) == 768, "packed64 tile size");

int q38_nvfp4_packed64_repack(void *dst, const void *source, int cols) {
    if (!dst || !source || cols <= 0 || cols % 64) return 0;
    q38_p64_tile *out = (q38_p64_tile *)dst;
    const q38_p64_source_block *in = (const q38_p64_source_block *)source;
    int nb = cols / 64;
    for (int tile = 0; tile < cols / 16; tile++) {
        int kbase = tile * 16;
        q38_p64_tile *t = &out[tile];
        for (int row = 0; row < 64; row++) {
            const q38_p64_source_subblock *p =
                &in[(row / 8) * nb + kbase / 64].s[(kbase % 64) / 16];
            t->scales[row] = p->d[row % 8];
            for (int pair = 0; pair < 2; pair++) {
                for (int j = 0; j < 4; j++) {
                    int v0 = pair * 8 + j;
                    int v1 = v0 + 4;
                    uint8_t z0 = p->qs[(row % 8) * 8 + v0 % 8];
                    uint8_t z1 = p->qs[(row % 8) * 8 + v1 % 8];
                    int c0 = v0 < 8 ? z0 & 15 : z0 >> 4;
                    int c1 = v1 < 8 ? z1 & 15 : z1 >> 4;
                    t->codes[pair][row / 16][(row % 16) * 4 + j] =
                        (uint8_t)(c0 | (c1 << 4));
                }
            }
        }
    }
    return 1;
}

int q38_nvfp4_packed64_a8_rows(float *y, const void *weights,
                                const int8_t *digits, float act_scale,
                                int rows, int cols) {
    if (svcntb() != 64 || rows <= 0 || rows % 64 ||
        cols <= 0 || cols % 64) return 0;
    static const int8_t code[64] = {
        0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
    };
    const svbool_t pb = svptrue_b8(), pf = svptrue_b32();
    const svint8_t lut = svld1_s8(pb, code);
    const q38_p64_tile *base = (const q38_p64_tile *)weights;
    int ntile = cols / 16;
    for (int row = 0; row < rows; row += 64) {
        svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0;
        const q38_p64_tile *tiles = base + (size_t)(row / 64) * ntile;
        for (int tile = 0; tile < ntile; tile++) {
            const q38_p64_tile *t = &tiles[tile];
            svint32_t c0=svdup_s32(0),c1=c0,c2=c0,c3=c0;
            for (int pair = 0; pair < 2; pair++) {
                uint32_t aq0, aq1;
                __builtin_memcpy(&aq0, digits + tile * 16 + pair * 8, 4);
                __builtin_memcpy(&aq1, digits + tile * 16 + pair * 8 + 4, 4);
                svint8_t av0 = svreinterpret_s8_u32(svdup_n_u32(aq0));
                svint8_t av1 = svreinterpret_s8_u32(svdup_n_u32(aq1));
#define Q38_P64_PAIR(G,C) do { \
                    svuint8_t z=svld1_u8(pb,t->codes[pair][G]); \
                    (C)=svdot_s32((C),svtbl_s8(lut,svand_n_u8_x(pb,z,15)),av0); \
                    (C)=svdot_s32((C),svtbl_s8(lut,svlsr_n_u8_x(pb,z,4)),av1); \
                } while (0)
                Q38_P64_PAIR(0,c0);
                Q38_P64_PAIR(1,c1);
                Q38_P64_PAIR(2,c2);
                Q38_P64_PAIR(3,c3);
#undef Q38_P64_PAIR
            }
            a0=svmla_f32_x(pf,a0,svcvt_f32_s32_x(pf,c0),
                           svmul_n_f32_x(pf,svld1_f32(pf,t->scales),act_scale));
            a1=svmla_f32_x(pf,a1,svcvt_f32_s32_x(pf,c1),
                           svmul_n_f32_x(pf,svld1_f32(pf,t->scales+16),act_scale));
            a2=svmla_f32_x(pf,a2,svcvt_f32_s32_x(pf,c2),
                           svmul_n_f32_x(pf,svld1_f32(pf,t->scales+32),act_scale));
            a3=svmla_f32_x(pf,a3,svcvt_f32_s32_x(pf,c3),
                           svmul_n_f32_x(pf,svld1_f32(pf,t->scales+48),act_scale));
        }
        svst1_f32(pf,y+row,a0);
        svst1_f32(pf,y+row+16,a1);
        svst1_f32(pf,y+row+32,a2);
        svst1_f32(pf,y+row+48,a3);
    }
    return 1;
}
