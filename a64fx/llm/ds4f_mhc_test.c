/* ds4f_mhc_test.c — validate the exact mHC math in common/ds4f.h against the
 * pure-Python reference (tools/ds4f_mhc_ref.py). Hand-builds a minimal model
 * (pool + cfg + hc tensors), fills the synthetic weights with the SAME
 * deterministic hashes as ds4f_alloc_synth, runs hc_pre/hc_post/hc_head on the
 * SAME test vectors, and writes mhc_c.txt. Tiny config: hc=4, d=8.
 *
 * Build:  fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
 *             -I../../common -o build/ds4f_mhc_test ds4f_mhc_test.c -lm -lpthread
 */
#include "ds4f.h"

#define HC 4
#define D  8
#define HD (HC*D)          /* 32 */
#define MIX ((2+HC)*HC)    /* 24 */

int main(void) {
    ds4f_model m; memset(&m, 0, sizeof(m));
    ds4f_config c = ds4f_default_config();
    c.hidden = D; c.hc_mult = HC; c.n_layers = 1;   /* hc_iters/hc_eps/norm_eps from default */
    m.cfg = c;
    m.pool = ds4f_pool_start(4, 1);

    /* allocate + fill the mHC weights exactly as ds4f_alloc_synth does */
    float *afn  = (float *)aligned_alloc(256, (size_t)MIX*HD*4);
    float *abase= (float *)aligned_alloc(64,  (size_t)MIX*4);
    float *ascale=(float *)aligned_alloc(64,  (size_t)3*4);
    m.hc_head_fn    = (float *)aligned_alloc(256, (size_t)HC*HD*4);
    m.hc_head_base  = (float *)aligned_alloc(64,  (size_t)HC*4);
    m.hc_head_scale = (float *)aligned_alloc(64,  (size_t)4);

    ds4f_tensor af = { afn, NULL, DS4F_F32, MIX, HD }; ds4f_fill(&m, af);
    ds4f_hc_fill_meta(abase, MIX, ascale, 3, 0);                 /* attn: layer-0 seed */
    ds4f_tensor hf = { m.hc_head_fn, NULL, DS4F_F32, HC, HD }; ds4f_fill(&m, hf);
    ds4f_hc_fill_meta(m.hc_head_base, HC, m.hc_head_scale, 1, 4096);

    /* test vectors (identical formulas to ds4f_mhc_ref.py) */
    float x4[HD];
    for (int k = 0; k < HC; k++) for (int d = 0; d < D; d++)
        x4[k*D+d] = (float)((((k*23 + d*5) % 31) - 15)) / 15.0f;
    float fblk[D];
    for (int d = 0; d < D; d++) fblk[d] = (float)((((d*7) % 13) - 6)) / 6.0f;

    float y_pre[D], post[HC], comb[HC*HC];
    ds4f_hc_pre(&m, x4, afn, ascale, abase, y_pre, post, comb);

    float y_post[HD];                       /* output streams */
    ds4f_hc_post(&m, y_post, x4, fblk, post, comb);   /* resid = x4, block out = fblk */

    float y_head[D];
    ds4f_hc_head(&m, x4, y_head);

    /* write in the SAME order as ds4f_mhc_ref.py so a line-wise paste diffs col 2 */
    FILE *fp = fopen("mhc_c.txt", "w");
    for (int d = 0; d < D; d++)  fprintf(fp, "y_pre%-5d %.7e\n", d, y_pre[d]);
    for (int k = 0; k < HC; k++) fprintf(fp, "post%-6d %.7e\n", k, post[k]);
    for (int j = 0; j < HC; j++) for (int k = 0; k < HC; k++)
        fprintf(fp, "comb%d_%-4d %.7e\n", j, k, comb[j*HC+k]);
    for (int k = 0; k < HC; k++) for (int d = 0; d < D; d++)
        fprintf(fp, "ypost%d_%-3d %.7e\n", k, d, y_post[k*D+d]);
    for (int d = 0; d < D; d++)  fprintf(fp, "yhead%-4d %.7e\n", d, y_head[d]);
    fclose(fp);

    ds4f_pool_stop(m.pool);
    printf("wrote mhc_c.txt\n");
    return 0;
}
