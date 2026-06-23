#include <stdint.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

void a64fx_sdot_4xvl(const int8_t *a, const int8_t *b, int32_t *c, int k)
{
    svbool_t pg_b = svptrue_b8();
    svbool_t pg_s = svptrue_b32();
    svint32_t acc0 = svdup_s32(0);
    svint32_t acc1 = svdup_s32(0);
    svint32_t acc2 = svdup_s32(0);
    svint32_t acc3 = svdup_s32(0);
    int vl = (int)svcntb();

    for (int p = 0; p < k; p += vl) {
        svint8_t bv = svld1_s8(pg_b, b + p);
        svint8_t a0 = svdup_n_s8(a[p + 0 * k]);
        svint8_t a1 = svdup_n_s8(a[p + 1 * k]);
        svint8_t a2 = svdup_n_s8(a[p + 2 * k]);
        svint8_t a3 = svdup_n_s8(a[p + 3 * k]);
        acc0 = svdot_s32(acc0, bv, a0);
        acc1 = svdot_s32(acc1, bv, a1);
        acc2 = svdot_s32(acc2, bv, a2);
        acc3 = svdot_s32(acc3, bv, a3);
    }

    svst1_s32(pg_s, c + 0 * svcntw(), acc0);
    svst1_s32(pg_s, c + 1 * svcntw(), acc1);
    svst1_s32(pg_s, c + 2 * svcntw(), acc2);
    svst1_s32(pg_s, c + 3 * svcntw(), acc3);
}

void a64fx_i16_smlal_4xvl(const int16_t *a, const int16_t *b, int32_t *c, int k)
{
    svbool_t pg_h = svptrue_b16();
    svbool_t pg_s = svptrue_b32();
    svint32_t acc0 = svdup_s32(0);
    svint32_t acc1 = svdup_s32(0);
    svint32_t acc2 = svdup_s32(0);
    svint32_t acc3 = svdup_s32(0);
    int vl = (int)svcnth();

    for (int p = 0; p < k; p += vl) {
        svint16_t bv = svld1_s16(pg_h, b + p);
        svint16_t a0 = svdup_n_s16(a[p + 0 * k]);
        svint16_t a1 = svdup_n_s16(a[p + 1 * k]);
        svint16_t a2 = svdup_n_s16(a[p + 2 * k]);
        svint16_t a3 = svdup_n_s16(a[p + 3 * k]);
        acc0 = svmlalb_s32(acc0, bv, a0);
        acc0 = svmlalt_s32(acc0, bv, a0);
        acc1 = svmlalb_s32(acc1, bv, a1);
        acc1 = svmlalt_s32(acc1, bv, a1);
        acc2 = svmlalb_s32(acc2, bv, a2);
        acc2 = svmlalt_s32(acc2, bv, a2);
        acc3 = svmlalb_s32(acc3, bv, a3);
        acc3 = svmlalt_s32(acc3, bv, a3);
    }

    svst1_s32(pg_s, c + 0 * svcntw(), acc0);
    svst1_s32(pg_s, c + 1 * svcntw(), acc1);
    svst1_s32(pg_s, c + 2 * svcntw(), acc2);
    svst1_s32(pg_s, c + 3 * svcntw(), acc3);
}
#else
void a64fx_sdot_4xvl(const int8_t *a, const int8_t *b, int32_t *c, int k)
{
    (void)a;
    (void)b;
    (void)c;
    (void)k;
}

void a64fx_i16_smlal_4xvl(const int16_t *a, const int16_t *b, int32_t *c, int k)
{
    (void)a;
    (void)b;
    (void)c;
    (void)k;
}
#endif
