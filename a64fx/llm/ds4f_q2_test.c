/* ds4f_q2_test.c — validate the Tier-B2 activation-quant / rotate kernels in
 * common/ds4f.h against the pure-Python reference (tools/ds4f_q2_ref.py):
 *   ds4f_round_scale_pow2  (kernel.py fast_round_scale, ue8m0 power-of-2 scale)
 *   ds4f_fp4_e2m1_snap     (RNE round to the float4_e2m1 grid)
 *   ds4f_fp4_act_quant_inplace (block-32 FP4 QAT quant->dequant, kernel.py fp4_quant)
 *   ds4f_rotate_activation (Sylvester FWHT * n^-0.5, model.py rotate_activation)
 * Writes q2_c.txt with the SAME line order as q2_py.txt so a paste/awk diff reads
 * column 2 vs column 4.
 *
 * Build:  fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
 *             -D_GNU_SOURCE -I../../common -o build/ds4f_q2_test \
 *             ds4f_q2_test.c -lm -lpthread
 */
#include "ds4f.h"

#define NFP4 64        /* 2 blocks of 32 */
#define BLK  32
#define NHAD 64        /* power of 2 */

int main(void) {
    FILE *fp = fopen("q2_c.txt", "w");

    /* --- round-scale bit trick (fp4 qmax=6, fp8 qmax=448) --- */
    float amaxes[6] = {0.3f, 1.0f, 5.7f, 12.3f, 448.0f, 449.0f};
    for (int i = 0; i < 6; i++) {
        fprintf(fp, "rs6_%-8d %.7e\n", i, ds4f_round_scale_pow2(amaxes[i], 1.0f/6.0f));
        fprintf(fp, "rs448_%-6d %.7e\n", i, ds4f_round_scale_pow2(amaxes[i], 1.0f/448.0f));
    }

    /* --- fp4 grid snap over a fine sweep (incl. near-midpoint values) --- */
    for (int i = 0; i < 49; i++) {
        float v = (float)(i - 24) * 0.25f;            /* -6 .. 6 step .25 -> hits midpoints */
        fprintf(fp, "snap%-8d %.7e\n", i, ds4f_fp4_e2m1_snap(v));
    }

    /* --- fp4 block-32 act-quant (two blocks, distinct scales) --- */
    float x[NFP4];
    for (int i = 0; i < NFP4; i++) x[i] = (float)((((i*37 + 11) % 97) - 48)) / 7.0f; /* ~-6.86..6.57 */
    ds4f_fp4_act_quant_inplace(x, NFP4, BLK);
    for (int i = 0; i < NFP4; i++) fprintf(fp, "fq%-9d %.7e\n", i, x[i]);

    /* --- hadamard rotate (FWHT * n^-0.5) --- */
    float h[NHAD];
    for (int i = 0; i < NHAD; i++) h[i] = (float)((((i*29 + 5) % 101) - 50)) / 25.0f; /* ~-2..2 */
    ds4f_rotate_activation(h, NHAD);
    for (int i = 0; i < NHAD; i++) fprintf(fp, "had%-8d %.7e\n", i, h[i]);

    fclose(fp);
    printf("wrote q2_c.txt\n");
    return 0;
}
