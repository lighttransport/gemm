/* verify_stage1.c — Verify CUDA Stage 1 (sparse-structure) DiT single forward
 * step vs PyTorch.
 *
 * Oracle: dump_ground_truth.py (--dump-per-block) computes
 *     v_ss = flow_ss(noise_ss, t=0.5, cond)            # 02b_ss_dit_step_velocity
 * i.e. a single DIRECT model forward at raw t=0.5 (NOT through the sampler, so
 * no t*1000 scaling). cuda_trellis2_run_dit() multiplies its timestep by 1000
 * internally (cuda_trellis2_runner.c:1508), so we pass t_raw=0.0005 → the
 * embedder sees 0.5, matching the oracle.
 *
 * Layout: 02_ss_noise.npy and 02b_ss_dit_step_velocity.npy are [1,8,16,16,16]
 * CHANNEL-MAJOR (flat C-order = c*4096+pos = [C,N]), which is exactly what
 * run_dit consumes/produces. Feed and compare RAW — no token-major conversion.
 *
 * Usage: ./verify_stage1 <stage1.st> <noise.npy> <cond.npy> <ref_velocity.npy>
 */
#include "cuda_trellis2_runner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static float *read_npy_f32(const char *p, int *nd, int *dd) {
    FILE *f=fopen(p,"rb"); if(!f){ fprintf(stderr,"Cannot read %s\n",p); return NULL; }
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

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <stage1.st> <noise.npy> <cond.npy> <ref_velocity.npy>\n", argv[0]);
        return 1;
    }
    cuda_trellis2_runner *r = cuda_trellis2_init(0, 1);
    if (!r) return 1;
    /* Load ONLY the Stage-1 DiT (no DINOv3, no decoder). */
    if (cuda_trellis2_load_weights(r, NULL, argv[1], NULL) != 0) {
        fprintf(stderr, "Failed to load stage1 weights from %s\n", argv[1]);
        cuda_trellis2_free(r); return 1;
    }

    int nd, dd[8];
    float *noise = read_npy_f32(argv[2], &nd, dd);
    if (!noise) { cuda_trellis2_free(r); return 1; }
    int total = 1; for (int i = 0; i < nd; i++) total *= dd[i];
    fprintf(stderr, "Noise: ndim=%d total=%d (expect 8*4096=32768, channel-major)\n", nd, total);

    float *cond = read_npy_f32(argv[3], &nd, dd);   /* [1,1029,1024]; run_dit reads [1029,1024] */
    if (!cond) { free(noise); cuda_trellis2_free(r); return 1; }
    fprintf(stderr, "Cond:  ndim=%d dims=[%d,%d,%d]\n", nd, dd[0], nd>1?dd[1]:0, nd>2?dd[2]:0);

    float *ref = read_npy_f32(argv[4], &nd, dd);
    if (!ref) { free(noise); free(cond); cuda_trellis2_free(r); return 1; }

    float *output = (float *)malloc((size_t)total * sizeof(float));

    /* t_raw=0.0005 → run_dit's internal *1000 makes the embedder see 0.5,
     * matching the oracle's flow_ss(noise, t=0.5). */
    float t_raw = 0.0005f;
    fprintf(stderr, "Running Stage 1 DiT (t_raw=%.4f, model sees t=%.1f)...\n",
            t_raw, t_raw * 1000.0f);
    cuda_trellis2_run_dit(r, noise, t_raw, cond, output);

    /* Metrics: correlation, rel L2, max-abs. */
    double sr=0,sc=0,sr2=0,sc2=0,src2=0, num=0, den=0, maxabs=0;
    int maxi=0;
    for (int i=0;i<total;i++) {
        double rv=ref[i], cv=output[i], d=cv-rv;
        sr+=rv; sc+=cv; sr2+=rv*rv; sc2+=cv*cv; src2+=rv*cv;
        num += d*d; den += rv*rv;
        if (fabs(d) > maxabs) { maxabs = fabs(d); maxi = i; }
    }
    double mr=sr/total, mc=sc/total;
    double corr=(src2/total-mr*mc)/sqrt((sr2/total-mr*mr)*(sc2/total-mc*mc));
    double cosine = src2 / (sqrt(sr2) * sqrt(sc2));
    double relL2 = sqrt(num) / sqrt(den);
    fprintf(stderr, "Ref:  std=%.4f, [:4]=%.4f %.4f %.4f %.4f\n",
            sqrt(sr2/total-mr*mr), ref[0],ref[1],ref[2],ref[3]);
    fprintf(stderr, "CUDA: std=%.4f, [:4]=%.4f %.4f %.4f %.4f\n",
            sqrt(sc2/total-mc*mc), output[0],output[1],output[2],output[3]);
    fprintf(stderr, "Correlation: %.8f\n", corr);
    fprintf(stderr, "Cosine:      %.8f\n", cosine);
    fprintf(stderr, "rel L2:      %.8e\n", relL2);
    fprintf(stderr, "max abs:     %.8e  at idx=%d (ref=%.6f cuda=%.6f)\n",
            maxabs, maxi, ref[maxi], output[maxi]);

    free(noise); free(cond); free(ref); free(output);
    cuda_trellis2_free(r);
    return (corr > 0.99) ? 0 : 1;
}
