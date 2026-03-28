/* Quick test: load shape decoder and run forward on Stage 2 output. */
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define SPARSE3D_IMPLEMENTATION
#include "../../common/sparse3d.h"
#define T2_SHAPE_DEC_IMPLEMENTATION
#include "../../common/trellis2_shape_decoder.h"
#define T2_FDG_MESH_IMPLEMENTATION
#include "../../common/trellis2_fdg_mesh.h"

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

    fprintf(stderr, "\nResult: N=%d, raw feats[:7]=%.2f %.2f %.2f | %.2f %.2f %.2f | %.2f\n",
            result.N,
            result.feats[0], result.feats[1], result.feats[2],
            result.feats[3], result.feats[4], result.feats[5], result.feats[6]);

    /* Post-process: apply sigmoid to vertex offsets, threshold for intersected */
    float voxel_margin = 0.0f;  /* from official code default */
    for (int i = 0; i < result.N; i++) {
        float *f = result.feats + i * 7;
        /* vertex offsets: (1 + 2*margin) * sigmoid(x) - margin */
        for (int j = 0; j < 3; j++)
            f[j] = (1.0f + 2.0f * voxel_margin) / (1.0f + expf(-f[j])) - voxel_margin;
        /* intersected: f[3..5] > 0 (already raw logits, keep as-is for mesh extraction) */
        /* split_weight: softplus(f[6]) = log(1 + exp(x)) */
        f[6] = logf(1.0f + expf(f[6]));
    }

    /* Extract coords without batch dim: [N, 3] from result.coords [N, 4] */
    int32_t *coords3 = (int32_t *)malloc((size_t)result.N * 3 * sizeof(int32_t));
    for (int i = 0; i < result.N; i++) {
        coords3[i*3+0] = result.coords[i*4+1];  /* z */
        coords3[i*3+1] = result.coords[i*4+2];  /* y */
        coords3[i*3+2] = result.coords[i*4+3];  /* x */
    }

    /* Mesh extraction */
    float aabb[6] = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
    /* Compute voxel_size from grid resolution and aabb */
    int max_coord = 0;
    for (int i = 0; i < result.N * 3; i++)
        if (coords3[i] > max_coord) max_coord = coords3[i];
    float vs = (aabb[3] - aabb[0]) / (float)(max_coord + 1);

    fprintf(stderr, "\nExtracting mesh (voxel_size=%.4f, max_coord=%d)...\n", vs, max_coord);
    t2_fdg_mesh mesh = t2_fdg_to_mesh(coords3, result.feats, result.N, vs, aabb);
    free(coords3);

    if (mesh.n_tris > 0) {
        const char *obj_path = "shape_output.obj";
        t2_fdg_write_obj(obj_path, &mesh);
    }

    t2_fdg_mesh_free(&mesh);
    t2_shape_dec_result_free(&result);
    sp3d_free(slat);
    t2_shape_dec_free(dec);
    return 0;
}
