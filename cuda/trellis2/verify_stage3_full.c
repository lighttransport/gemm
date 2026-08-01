/* verify_stage3_full.c — Verify the CUDA Stage 3 FULL 12-step sampler vs PyTorch.
 *
 * Feeds PyTorch's EXACT Stage-3 (texture SLat) inputs through the same
 * FlowEulerGuidanceInterval loop the e2e harness runs, and compares the raw output
 * to 11_tex_slat_raw_feats. Stage 3 has guidance_strength=1.0 so CFG is fully
 * disabled (no neg-cond path). The DiT input each step is the [N,64] concat of the
 * current state [N,32] with the (already re-normalized) shape-SLat concat_cond [N,32].
 *
 * Inputs (model_root/pipeline.json tex_slat_sampler: steps=12, rescale_t=3.0,
 * guidance_strength=1.0):
 *   noise       = 09_tex_slat_noise_feats   [N,32]  (initial state)
 *   concat_cond = 10_tex_concat_cond_feats  [N,32]  (re-normalized shape SLat; used raw)
 *   coords      = 10b_tex_dit_step_coords   [N,4]
 *   cond        = 06b_slat_dit_step_cond    [1,n_cond,1024] (image cond)
 *   ref         = 11_tex_slat_raw_feats      [N,32]
 *
 * Usage: ./verify_stage3_full <stage3.st> <noise> <concat_cond> <coords> <cond> <ref>
 */
#include "cuda_trellis2_runner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static float *read_npy_f32(const char *p, int *nd, int *dd) {
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    fseek(f,8,SEEK_SET); uint16_t hl;
    if(fread(&hl,2,1,f)!=1){ fclose(f); return NULL; }
    char *h=malloc(hl+1);
    if(fread(h,1,hl,f)!=(size_t)hl){ free(h); fclose(f); return NULL; }
    h[hl]=0;
    *nd=0; char *sp=strstr(h,"shape"); if(sp){sp=strchr(sp,'('); if(sp){sp++;
        while(*sp&&*sp!=')'){
            while(*sp==' '||*sp==',')sp++;
            if(*sp==')')break;
            dd[*nd]=(int)strtol(sp,&sp,10);(*nd)++;
        }}}
    size_t n=1; for(int i=0;i<*nd;i++) n*=dd[i];
    float *d=malloc(n*sizeof(float));
    if(fread(d,sizeof(float),n,f)!=n){ free(d); free(h); fclose(f); return NULL; }
    fclose(f); free(h); return d;
}
static int32_t *read_npy_i32(const char *p, int *nd, int *dd) {
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    fseek(f,8,SEEK_SET); uint16_t hl;
    if(fread(&hl,2,1,f)!=1){ fclose(f); return NULL; }
    char *h=malloc(hl+1);
    if(fread(h,1,hl,f)!=(size_t)hl){ free(h); fclose(f); return NULL; }
    h[hl]=0;
    *nd=0; char *sp=strstr(h,"shape"); if(sp){sp=strchr(sp,'('); if(sp){sp++;
        while(*sp&&*sp!=')'){
            while(*sp==' '||*sp==',')sp++;
            if(*sp==')')break;
            dd[*nd]=(int)strtol(sp,&sp,10);(*nd)++;
        }}}
    size_t n=1; for(int i=0;i<*nd;i++) n*=dd[i];
    int32_t *d=malloc(n*sizeof(int32_t));
    if(fread(d,sizeof(int32_t),n,f)!=n){ free(d); free(h); fclose(f); return NULL; }
    fclose(f); free(h); return d;
}

static float rescale_t(float t, float rt){ return t*rt/(1.0f+(rt-1.0f)*t); }

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s <stage3.st> <noise> <concat_cond> <coords> <cond> <ref>\n", argv[0]);
        return 1;
    }
    const int   steps      = 12;
    const float rescale_tv = 3.0f;   /* pipeline.json tex_slat_sampler */

    cuda_trellis2_runner *r = cuda_trellis2_init(0, 1);
    if (!r) return 1;
    if (cuda_trellis2_load_stage3(r, argv[1]) != 0) return 1;

    int nd, dd[8];
    float   *state = read_npy_f32(argv[2], &nd, dd);   /* [N,32] initial noise */
    int N = dd[0], C = dd[1];
    float   *ccond = read_npy_f32(argv[3], &nd, dd);   /* [N,32] concat_cond (re-normalized shape) */
    int Ncc = dd[0];
    int32_t *coords = read_npy_i32(argv[4], &nd, dd);  /* [N,4] */
    int Nco = dd[0];
    float   *cond = read_npy_f32(argv[5], &nd, dd);    /* [1,n_cond,1024] */
    float   *ref  = read_npy_f32(argv[6], &nd, dd);    /* [N,32] */
    int Nr = dd[0];
    fprintf(stderr, "N=%d C=%d (concat_cond N=%d, coords N=%d, ref N=%d)\n", N, C, Ncc, Nco, Nr);
    if (C != 32 || N != Ncc || N != Nco || N != Nr) {
        fprintf(stderr, "ERROR: shape mismatch — inputs must be [N,32]/[N,4] sharing voxel set/order\n");
        return 2;
    }
    fprintf(stderr, "Sampler: steps=%d rescale_t=%.1f (no CFG, guidance_strength=1.0)\n", steps, rescale_tv);

    float *xt = (float *)malloc((size_t)N * 64 * sizeof(float));
    float *v  = (float *)malloc((size_t)N * 32 * sizeof(float));

    for (int step = 0; step < steps; step++) {
        float t_start = 1.0f - (float)step / (float)steps;
        float t_end   = 1.0f - (float)(step + 1) / (float)steps;
        float t_cur  = rescale_t(t_start, rescale_tv);
        float t_next = rescale_t(t_end,   rescale_tv);

        /* x_t = [state, concat_cond] -> [N,64] */
        for (int i = 0; i < N; i++) {
            memcpy(xt + i * 64,      state + i * 32, 32 * sizeof(float));
            memcpy(xt + i * 64 + 32, ccond + i * 32, 32 * sizeof(float));
        }
        cuda_trellis2_run_stage3_dit(r, xt, t_cur, cond, coords, N, v);
        for (int i = 0; i < N * 32; i++) state[i] -= (t_cur - t_next) * v[i];
        fprintf(stderr, "  step %2d/%d  t=%.4f->%.4f\n", step+1, steps, t_cur, t_next);
    }

    /* Compare state (raw output) vs ref (11_tex_slat_raw_feats) */
    double sr=0, sc=0, sr2=0, sc2=0, src=0, sd2=0;
    int total = N * 32;
    for (int i = 0; i < total; i++) {
        double a = state[i], b = ref[i];
        sr += b; sc += a; sr2 += b*b; sc2 += a*a; src += a*b; sd2 += (a-b)*(a-b);
    }
    double cos = src / (sqrt(sc2)*sqrt(sr2) + 1e-30);
    double rel = sqrt(sd2) / (sqrt(sr2) + 1e-30);
    double mr=sr/total, mc=sc/total;
    double corr = (src/total - mr*mc) / sqrt((sc2/total-mc*mc)*(sr2/total-mr*mr) + 1e-30);
    fprintf(stderr, "\n=== Stage-3 FULL sampler vs 11_tex_slat_raw_feats ===\n");
    fprintf(stderr, "  cosine=%.6f  relL2=%.4e  corr=%.6f\n", cos, rel, corr);
    fprintf(stderr, "  ref  [:4]=%.4f %.4f %.4f %.4f\n", ref[0],ref[1],ref[2],ref[3]);
    fprintf(stderr, "  ours [:4]=%.4f %.4f %.4f %.4f\n", state[0],state[1],state[2],state[3]);

    free(state); free(ccond); free(coords); free(cond); free(ref); free(xt); free(v);
    cuda_trellis2_free(r);
    return (cos > 0.99) ? 0 : 1;
}
