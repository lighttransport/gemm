/*
 * test_paint_view_maps_stage.c - smoke harness for paint_stage_view_maps.
 *
 * Drives only the opaque API in paint_stages.h: create -> set_mesh ->
 * render -> destroy, then writes per-view PPMs and .npy dumps for visual
 * + bit-level comparison against test_view_maps.c output.
 *
 * Usage:
 *   ./test_paint_view_maps_stage <mesh.obj> [out_prefix] [resolution]
 */

#include "../cuew.h"
#include "paint_stages.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float *pos; int *tri; int n_verts, n_tris; } obj_mesh;

static int read_obj(const char *path, obj_mesh *m) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    int cap_v = 1<<14, cap_t = 1<<14;
    m->pos = (float *)malloc((size_t)cap_v * 3 * sizeof(float));
    m->tri = (int *)  malloc((size_t)cap_t * 3 * sizeof(int));
    m->n_verts = 0; m->n_tris = 0;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        if (line[0]=='v' && line[1]==' ') {
            float x,y,z;
            if (sscanf(line+2, "%f %f %f", &x,&y,&z)==3) {
                if (m->n_verts >= cap_v) { cap_v *= 2; m->pos = (float *)realloc(m->pos, (size_t)cap_v*3*sizeof(float)); }
                m->pos[m->n_verts*3+0]=x; m->pos[m->n_verts*3+1]=y; m->pos[m->n_verts*3+2]=z; m->n_verts++;
            }
        } else if (line[0]=='f' && line[1]==' ') {
            int idx[3]={0,0,0}; const char *p = line+2; int k=0;
            while (*p && k<3) {
                while (*p==' '||*p=='\t') p++;
                if (!*p||*p=='\n') break;
                idx[k++]=atoi(p);
                while (*p && *p!=' ' && *p!='\t' && *p!='\n') p++;
            }
            if (k==3) {
                for (int i=0;i<3;i++) idx[i] = idx[i] < 0 ? m->n_verts + idx[i] : idx[i] - 1;
                if (m->n_tris >= cap_t) { cap_t *= 2; m->tri = (int *)realloc(m->tri, (size_t)cap_t*3*sizeof(int)); }
                m->tri[m->n_tris*3+0]=idx[0]; m->tri[m->n_tris*3+1]=idx[1]; m->tri[m->n_tris*3+2]=idx[2]; m->n_tris++;
            }
        }
    }
    fclose(f); return 0;
}

static void write_ppm(const char *path, const float *rgb, int W, int H) {
    FILE *f = fopen(path, "wb"); if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < W*H; i++)
        for (int c = 0; c < 3; c++) {
            float v = rgb[i*3+c]; if (v<0) v=0; if (v>1) v=1;
            fputc((uint8_t)(v*255.f+0.5f), f);
        }
    fclose(f);
}

static void write_npy_f32(const char *path, const float *data, int *sh, int nd) {
    FILE *f = fopen(path, "wb"); if (!f) return;
    fwrite("\x93NUMPY", 1, 6, f);
    uint8_t ver[2]={1,0}; fwrite(ver, 1, 2, f);
    char hdr[256], shape_s[128] = ""; size_t total = 1;
    for (int i = 0; i < nd; i++) { char tmp[32]; snprintf(tmp,sizeof(tmp),"%d, ",sh[i]); strcat(shape_s,tmp); total*=(size_t)sh[i]; }
    int hl = snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", shape_s);
    int tot = 10 + hl + 1; int pad = ((tot+63)/64)*64 - tot;
    uint16_t header_len = (uint16_t)(hl + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(hdr, 1, (size_t)hl, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), total, f);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mesh.obj> [out_prefix] [resolution]\n", argv[0]);
        return 1;
    }
    const char *obj_path = argv[1];
    const char *prefix   = argc >= 3 ? argv[2] : "views_stage";
    int res              = argc >= 4 ? atoi(argv[3]) : 512;

    obj_mesh m = {0};
    if (read_obj(obj_path, &m) != 0) return 1;
    fprintf(stderr, "Loaded %s: %d v, %d t\n", obj_path, m.n_verts, m.n_tris);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    paint_stage_view_maps *vm = paint_stage_view_maps_create(dev, res);
    if (!vm) return 1;
    paint_stage_view_maps_set_mesh(vm, m.pos, m.n_verts, m.tri, m.n_tris);

    const int N_VIEWS = 6;
    size_t per_view = (size_t)res * res * 3;
    CUdeviceptr d_nrm, d_pos;
    cuMemAlloc(&d_nrm, N_VIEWS * per_view * sizeof(float));
    cuMemAlloc(&d_pos, N_VIEWS * per_view * sizeof(float));
    paint_stage_view_maps_render(vm, d_nrm, d_pos, 0, 0, 0, NULL, NULL);

    float *h_nrm = (float *)malloc(N_VIEWS * per_view * sizeof(float));
    float *h_pos = (float *)malloc(N_VIEWS * per_view * sizeof(float));
    cuMemcpyDtoH(h_nrm, d_nrm, N_VIEWS * per_view * sizeof(float));
    cuMemcpyDtoH(h_pos, d_pos, N_VIEWS * per_view * sizeof(float));

    for (int v = 0; v < N_VIEWS; v++) {
        char path[1024]; int sh[3] = {res, res, 3};
        snprintf(path, sizeof(path), "%s_view%d_normal.ppm",   prefix, v);
        write_ppm(path, h_nrm + v * per_view, res, res);
        snprintf(path, sizeof(path), "%s_view%d_normal.npy",   prefix, v);
        write_npy_f32(path, h_nrm + v * per_view, sh, 3);
        snprintf(path, sizeof(path), "%s_view%d_position.ppm", prefix, v);
        write_ppm(path, h_pos + v * per_view, res, res);
        snprintf(path, sizeof(path), "%s_view%d_position.npy", prefix, v);
        write_npy_f32(path, h_pos + v * per_view, sh, 3);
    }
    fprintf(stderr, "wrote %d views to %s_view*.{ppm,npy}\n", N_VIEWS, prefix);

    free(h_nrm); free(h_pos);
    cuMemFree(d_nrm); cuMemFree(d_pos);
    paint_stage_view_maps_destroy(vm);
    cuCtxDestroy(ctx);
    free(m.pos); free(m.tri);
    return 0;
}
