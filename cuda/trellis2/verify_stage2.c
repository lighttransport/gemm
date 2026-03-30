/* verify_stage2.c — Verify CUDA Stage 2 DiT single step vs PyTorch.
 * Usage: ./verify_stage2 <stage2.st> <noise.npy> <coords.npy> <cond.npy> <ref_output.npy> */
#include "cuda_trellis2_runner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static float *read_npy_f32(const char *p, int *nd, int *dd) {
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    fseek(f,8,SEEK_SET); uint16_t hl; fread(&hl,2,1,f);
    char *h=malloc(hl+1); fread(h,1,hl,f); h[hl]=0;
    *nd=0; char *sp=strstr(h,"shape"); if(sp){sp=strchr(sp,'('); if(sp){sp++;
        while(*sp&&*sp!=')'){while(*sp==' '||*sp==',')sp++;
            if(*sp==')')break; dd[*nd]=(int)strtol(sp,&sp,10);(*nd)++;}}}
    size_t n=1; for(int i=0;i<*nd;i++) n*=dd[i];
    float *d=malloc(n*sizeof(float)); fread(d,sizeof(float),n,f);
    fclose(f); free(h); return d;
}
static int32_t *read_npy_i32(const char *p, int *nd, int *dd) {
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    fseek(f,8,SEEK_SET); uint16_t hl; fread(&hl,2,1,f);
    char *h=malloc(hl+1); fread(h,1,hl,f); h[hl]=0;
    *nd=0; char *sp=strstr(h,"shape"); if(sp){sp=strchr(sp,'('); if(sp){sp++;
        while(*sp&&*sp!=')'){while(*sp==' '||*sp==',')sp++;
            if(*sp==')')break; dd[*nd]=(int)strtol(sp,&sp,10);(*nd)++;}}}
    size_t n=1; for(int i=0;i<*nd;i++) n*=dd[i];
    int32_t *d=malloc(n*sizeof(int32_t)); fread(d,sizeof(int32_t),n,f);
    fclose(f); free(h); return d;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <stage2.st> <noise.npy> <coords.npy> <cond.npy> <ref.npy>\n", argv[0]);
        return 1;
    }
    cuda_trellis2_runner *r = cuda_trellis2_init(0, 1);
    if (!r) return 1;
    if (cuda_trellis2_load_stage2(r, argv[1]) != 0) return 1;

    int nd, dd[8];
    float *noise = read_npy_f32(argv[2], &nd, dd);
    int N = dd[0]; int C = dd[1];
    fprintf(stderr, "Noise: [%d, %d]\n", N, C);

    int32_t *coords = read_npy_i32(argv[3], &nd, dd);
    fprintf(stderr, "Coords: [%d, %d]\n", dd[0], dd[1]);

    float *cond = read_npy_f32(argv[4], &nd, dd);
    float *ref = read_npy_f32(argv[5], &nd, dd);

    float *output = (float *)malloc((size_t)N * C * sizeof(float));
    fprintf(stderr, "Running Stage 2 DiT...\n");
    cuda_trellis2_run_stage2_dit(r, noise, 1000.0f, cond, coords, N, output);

    /* Compare */
    double sr=0,sc=0,sr2=0,sc2=0,src2=0;
    int total = N * C;
    for (int i=0;i<total;i++) {
        sr+=ref[i]; sc+=output[i];
        sr2+=(double)ref[i]*ref[i]; sc2+=(double)output[i]*output[i];
        src2+=(double)ref[i]*output[i];
    }
    double mr=sr/total, mc=sc/total;
    double corr=(src2/total-mr*mc)/sqrt((sr2/total-mr*mr)*(sc2/total-mc*mc));
    fprintf(stderr, "Ref:  std=%.4f, [:4]=%.4f %.4f %.4f %.4f\n",
            sqrt(sr2/total-mr*mr), ref[0],ref[1],ref[2],ref[3]);
    fprintf(stderr, "CUDA: std=%.4f, [:4]=%.4f %.4f %.4f %.4f\n",
            sqrt(sc2/total-mc*mc), output[0],output[1],output[2],output[3]);
    fprintf(stderr, "Correlation: %.8f\n", corr);

    free(noise); free(coords); free(cond); free(ref); free(output);
    cuda_trellis2_free(r);
    return (corr > 0.99) ? 0 : 1;
}
