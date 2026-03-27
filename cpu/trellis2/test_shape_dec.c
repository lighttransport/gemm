/* Quick test: load shape decoder and run forward on Stage 2 output. */
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define SPARSE3D_IMPLEMENTATION
#include "../../common/sparse3d.h"
#define T2_SHAPE_DEC_IMPLEMENTATION
#include "../../common/trellis2_shape_decoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* Minimal npy readers */
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
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <shape_dec.st> <slat.npy> <coords.npy> [-t threads]\n", argv[0]);
        return 1;
    }
    int n_threads = 4;
    for (int i = 4; i < argc; i++)
        if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);

    /* Load decoder */
    t2_shape_dec *dec = t2_shape_dec_load(argv[1]);
    if (!dec) return 1;

    /* Load structured latent */
    int nd, dd[8];
    float *slat_feats = read_npy_f32(argv[2], &nd, dd);
    int N = dd[0], C = nd >= 2 ? dd[1] : dd[0];
    fprintf(stderr, "Loaded slat: [%d, %d]\n", N, C);

    int32_t *coords = read_npy_i32(argv[3], &nd, dd);
    fprintf(stderr, "Loaded coords: [%d, %d]\n", dd[0], dd[1]);

    /* Create sparse tensor */
    sp3d_tensor *slat = sp3d_create(coords, slat_feats, N, C, 1);
    free(slat_feats); free(coords);

    /* Forward */
    fprintf(stderr, "\nRunning shape decoder (%d threads)...\n", n_threads);
    t2_shape_dec_result result = t2_shape_dec_forward(dec, slat, n_threads);

    fprintf(stderr, "\nResult: N=%d, feats[:4]=%.4f %.4f %.4f %.4f\n",
            result.N,
            result.feats[0], result.feats[1], result.feats[2], result.feats[3]);

    t2_shape_dec_result_free(&result);
    sp3d_free(slat);
    t2_shape_dec_free(dec);
    return 0;
}
