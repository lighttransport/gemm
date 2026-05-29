/* verify_stage2_full.c — Verify the CUDA Stage 2 FULL 12-step sampler vs PyTorch.
 *
 * Feeds PyTorch's EXACT Stage-2 inputs (initial noise 06_shape_slat_noise_feats,
 * coords 05_ss_coords/06b_slat_dit_step_coords, positive cond 06b_slat_dit_step_cond,
 * zero neg-cond) through the same FlowEulerGuidanceInterval loop the e2e harness
 * runs, and compares the raw output to 07_shape_slat_raw_feats. This isolates the
 * sampler from upstream Stage-1/coord divergence (single-step is verify_stage2).
 *
 * Params come from model_root/pipeline.json shape_slat_sampler:
 *   steps=12, rescale_t=3.0, guidance_strength=7.5, guidance_rescale=0.5,
 *   guidance_interval=[0.6,1.0], sigma_min=1e-5.
 *
 * Usage: ./verify_stage2_full <stage2.st> <noise.npy> <coords.npy> <cond.npy> <ref.npy> [cfg_rescale]
 *        (cfg_rescale defaults to 0.5; pass 0.7 to reproduce the old harness bug.)
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
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <stage2.st> <noise.npy> <coords.npy> <cond.npy> <ref.npy> [cfg_rescale]\n", argv[0]);
        return 1;
    }
    /* pipeline.json shape_slat_sampler */
    const int   steps      = 12;
    const float cfg        = 7.5f;
    const float rescale_tv = 3.0f;
    const float sigma_min  = 1e-5f;
    const float cfg_rescale = (argc > 6) ? (float)atof(argv[6]) : 0.5f;

    cuda_trellis2_runner *r = cuda_trellis2_init(0, 1);
    if (!r) return 1;
    if (cuda_trellis2_load_stage2(r, argv[1]) != 0) return 1;

    int nd, dd[8];
    float *x = read_npy_f32(argv[2], &nd, dd);          /* initial noise [N,32] */
    int N = dd[0], C = dd[1];
    int32_t *coords = read_npy_i32(argv[3], &nd, dd);   /* [N,4] */
    int Nc = dd[0];
    float *cond = read_npy_f32(argv[4], &nd, dd);       /* [1,n_cond,1024] */
    size_t cond_n = 1; for (int i=0;i<nd;i++) cond_n *= dd[i];
    int n_cond = (int)(cond_n / 1024);
    float *ref = read_npy_f32(argv[5], &nd, dd);        /* [N,32] */
    int Nr = dd[0];
    fprintf(stderr, "N=%d C=%d (coords N=%d, ref N=%d, n_cond=%d)\n", N, C, Nc, Nr, n_cond);
    if (N != Nc || N != Nr) {
        fprintf(stderr, "ERROR: N mismatch noise=%d coords=%d ref=%d — inputs must share voxel set/order\n", N, Nc, Nr);
        return 2;
    }
    fprintf(stderr, "Sampler: steps=%d cfg=%.1f rescale_t=%.1f cfg_rescale=%.2f interval=[0.6,1.0]\n",
            steps, cfg, rescale_tv, cfg_rescale);

    float *zeros_cond = (float *)calloc((size_t)n_cond * 1024, sizeof(float));
    float *v_cond   = (float *)malloc((size_t)N * C * sizeof(float));
    float *v_uncond = (float *)malloc((size_t)N * C * sizeof(float));

    for (int step = 0; step < steps; step++) {
        float t_start = 1.0f - (float)step / (float)steps;
        float t_end   = 1.0f - (float)(step + 1) / (float)steps;
        float t_cur  = rescale_t(t_start, rescale_tv);
        float t_next = rescale_t(t_end,   rescale_tv);
        int apply_cfg = (t_cur >= 0.6f && t_cur <= 1.0f && cfg != 1.0f);

        if (apply_cfg) {
            cuda_trellis2_run_stage2_dit(r, x, t_cur, cond,        coords, N, v_cond);
            cuda_trellis2_run_stage2_dit(r, x, t_cur, zeros_cond,  coords, N, v_uncond);
            float *pred_v = v_uncond;
            for (int i = 0; i < N * C; i++)
                pred_v[i] = cfg * v_cond[i] + (1.0f - cfg) * v_uncond[i];
            if (cfg_rescale > 0.0f) {
                float sm = sigma_min, tc = sm + (1.0f - sm) * t_cur, one_m_sm = 1.0f - sm;
                double sp=0, sc=0, sp2=0, sc2=0;
                for (int i = 0; i < N * C; i++) {
                    float x0p = one_m_sm * x[i] - tc * v_cond[i];
                    float x0c = one_m_sm * x[i] - tc * pred_v[i];
                    sp += x0p; sp2 += (double)x0p*x0p; sc += x0c; sc2 += (double)x0c*x0c;
                }
                double n_d = (double)(N * C);
                double std_pos = sqrt((sp2 - sp*sp/n_d) / (n_d - 1.0));
                double std_cfg = sqrt((sc2 - sc*sc/n_d) / (n_d - 1.0));
                float ratio = (std_cfg > 1e-8) ? (float)(std_pos/std_cfg) : 1.0f;
                float scf = cfg_rescale * ratio + (1.0f - cfg_rescale);
                for (int i = 0; i < N * C; i++) {
                    float x0c = one_m_sm * x[i] - tc * pred_v[i];
                    pred_v[i] = (one_m_sm * x[i] - scf * x0c) / tc;
                }
            }
            for (int i = 0; i < N * C; i++) x[i] -= (t_cur - t_next) * pred_v[i];
        } else {
            cuda_trellis2_run_stage2_dit(r, x, t_cur, cond, coords, N, v_cond);
            for (int i = 0; i < N * C; i++) x[i] -= (t_cur - t_next) * v_cond[i];
        }
        fprintf(stderr, "  step %2d/%d  t=%.4f->%.4f  %s\n",
                step+1, steps, t_cur, t_next, apply_cfg ? "CFG" : "noG");
    }

    /* Compare x (raw output) vs ref (07_shape_slat_raw_feats) */
    double sr=0, sc=0, sr2=0, sc2=0, src=0, sd2=0;
    int total = N * C;
    for (int i = 0; i < total; i++) {
        double a = x[i], b = ref[i];
        sr += b; sc += a; sr2 += b*b; sc2 += a*a; src += a*b; sd2 += (a-b)*(a-b);
    }
    double cos = src / (sqrt(sc2)*sqrt(sr2) + 1e-30);
    double rel = sqrt(sd2) / (sqrt(sr2) + 1e-30);
    double mr=sr/total, mc=sc/total;
    double corr = (src/total - mr*mc) / sqrt((sc2/total-mc*mc)*(sr2/total-mr*mr) + 1e-30);
    fprintf(stderr, "\n=== Stage-2 FULL sampler vs 07_shape_slat_raw_feats ===\n");
    fprintf(stderr, "  cosine=%.6f  relL2=%.4e  corr=%.6f\n", cos, rel, corr);
    fprintf(stderr, "  ref  [:4]=%.4f %.4f %.4f %.4f\n", ref[0],ref[1],ref[2],ref[3]);
    fprintf(stderr, "  ours [:4]=%.4f %.4f %.4f %.4f\n", x[0],x[1],x[2],x[3]);

    free(x); free(coords); free(cond); free(ref);
    free(zeros_cond); free(v_cond); free(v_uncond);
    cuda_trellis2_free(r);
    return (cos > 0.99) ? 0 : 1;
}
