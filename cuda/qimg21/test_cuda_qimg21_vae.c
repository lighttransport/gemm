/*
 * Native Qwen-Image 2.1 VAE decoder.
 *
 * The transformer runner deliberately lives in a separate process from this
 * executable.  The decoder therefore owns only the 1.35 GB qimg-21 VAE
 * safetensors file and can run at 256..1024 resolution without competing for
 * the transformer's resident blocks on a 5060 Ti.
 *
 * The low-level CUDA VAE kernels are shared with cuda/qimg; this file only
 * supplies the qimg-21 tensor names and its z_dim=64 / 5-stage decoder graph.
 */

#define SAFETENSORS_IMPLEMENTATION
#define CUDA_QIMG_RUNNER_IMPLEMENTATION
#include "../../common/safetensors.h"
#include "../qimg/cuda_qimg_runner.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct {
    float *data;
    size_t n;
    int ndim;
    size_t shape[8];
} q21_npy;

static void q21_npy_free(q21_npy *a) {
    free(a->data);
    memset(a, 0, sizeof(*a));
}

static int q21_npy_read_f32(const char *path, q21_npy *out) {
    FILE *fp = fopen(path, "rb");
    char magic[6], header[65536];
    uint8_t ver[2];
    uint16_t h16 = 0;
    uint32_t h32 = 0;
    size_t hlen, pos, n = 1;
    memset(out, 0, sizeof(*out));
    if (!fp) {
        fprintf(stderr, "qimg21-vae: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (fread(magic, 1, 6, fp) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0 ||
        fread(ver, 1, 2, fp) != 2) {
        fclose(fp);
        return -1;
    }
    if (ver[0] == 1) {
        if (fread(&h16, 2, 1, fp) != 1) { fclose(fp); return -1; }
        hlen = h16;
    } else if (ver[0] == 2 || ver[0] == 3) {
        if (fread(&h32, 4, 1, fp) != 1) { fclose(fp); return -1; }
        hlen = h32;
    } else {
        fclose(fp);
        return -1;
    }
    if (hlen == 0 || hlen >= sizeof(header) || fread(header, 1, hlen, fp) != hlen) {
        fclose(fp);
        return -1;
    }
    header[hlen] = 0;
    if (!strstr(header, "'descr': '<f4'") && !strstr(header, "\"descr\": \"<f4\"")) {
        fprintf(stderr, "qimg21-vae: %s is not little-endian F32\n", path);
        fclose(fp);
        return -1;
    }
    if (strstr(header, "fortran_order': True") || strstr(header, "fortran_order\": True")) {
        fprintf(stderr, "qimg21-vae: Fortran-order input is unsupported: %s\n", path);
        fclose(fp);
        return -1;
    }
    char *shape = strstr(header, "shape");
    if (!shape || !(shape = strchr(shape, '('))) { fclose(fp); return -1; }
    pos = (size_t)(shape - header) + 1;
    while (pos < hlen && header[pos] != ')') {
        char *end;
        unsigned long long v;
        while (pos < hlen && (header[pos] == ' ' || header[pos] == ',')) pos++;
        if (header[pos] == ')') break;
        v = strtoull(header + pos, &end, 10);
        if (end == header + pos || out->ndim >= 8 || v == 0) { fclose(fp); return -1; }
        out->shape[out->ndim++] = (size_t)v;
        if ((size_t)v > SIZE_MAX / n) { fclose(fp); return -1; }
        n *= (size_t)v;
        pos = (size_t)(end - header);
        while (pos < hlen && header[pos] != ',' && header[pos] != ')') pos++;
    }
    if (n > SIZE_MAX / sizeof(float)) { fclose(fp); return -1; }
    out->data = (float *)malloc(n * sizeof(float));
    if (!out->data || fread(out->data, sizeof(float), n, fp) != n) {
        fclose(fp);
        q21_npy_free(out);
        return -1;
    }
    fclose(fp);
    out->n = n;
    return 0;
}

static int q21_npy_write_shape(const char *path, const float *x, size_t n, const char *shape) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    char hdr[256], body[256];
    int len = snprintf(hdr, sizeof(hdr),
                       "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shape);
    int padded = ((len + 10 + 63) / 64) * 64 - 10;
    if (padded >= (int)sizeof(body)) { fclose(fp); return -1; }
    memset(body, ' ', (size_t)padded);
    memcpy(body, hdr, (size_t)len);
    body[padded - 1] = '\n';
    uint16_t hlen = (uint16_t)padded;
    int ok = fwrite("\x93NUMPY\x01\x00", 1, 8, fp) == 8 &&
             fwrite(&hlen, 2, 1, fp) == 1 &&
             fwrite(body, 1, (size_t)padded, fp) == (size_t)padded &&
             fwrite(x, sizeof(float), n, fp) == n ? 0 : -1;
    if (fclose(fp) != 0) ok = -1;
    return ok;
}

static int q21_npy_write_chw(const char *path, const float *x, size_t n, int c, int h, int w) {
    char shape[96];snprintf(shape,sizeof(shape),"(%d, %d, %d)",c,h,w);
    return q21_npy_write_shape(path,x,n,shape);
}

static const char *q21_vae_dump_dir;
static void q21_dump(cuda_qimg_runner *r, CUdeviceptr x, int c, int h, int w,
                     const char *label) {
    if (!q21_vae_dump_dir) return;
    if (getenv("QIMG21_VAE_DUMP_STAGES")) {
        int keep = 0;
        for (int stage = 0; stage < 5; stage++) {
            char expected[32];
            snprintf(expected, sizeof(expected), "encoder_down_%d", stage);
            if (!strcmp(label, expected)) keep = 1;
        }
        if (!keep) return;
    }
    const char *match = getenv("QIMG21_VAE_DUMP_MATCH");
    if (match && (getenv("QIMG21_VAE_DUMP_EXACT") ? strcmp(label, match) != 0
                                                   : !strstr(label, match))) return;
    size_t n = (size_t)c * h * w;
    float *host = (float *)malloc(n * sizeof(float));
    if (!host) return;
    /* All VAE kernels run on the runner's non-default stream.  A synchronous
     * driver copy from a non-default stream is not a dependency fence, so
     * make the stage capture explicit and deterministic. */
    cuStreamSynchronize(r->stream);
    if (cuMemcpyDtoH(host, x, n * sizeof(float)) == CUDA_SUCCESS) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.npy", q21_vae_dump_dir, label);
        q21_npy_write_chw(path, host, n, c, h, w);
    }
    free(host);
    (void)r;
}

static float q21_bf16(uint16_t x) {
    uint32_t u = (uint32_t)x << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static int q21_vae_bf16_mode;
typedef int (*q21_cutlass_vae_attention_fn)(float*,const float*,const float*,const float*,int,int,void*);
static q21_cutlass_vae_attention_fn q21_cutlass_vae_attention;

static float q21_round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Upload a qimg-21 F32/BF16 tensor and return its 4-D conv shape. */
static CUdeviceptr q21_upload(const st_context *st, const char *name,
                              int *co, int *ci, int *kh, int *kw) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        fprintf(stderr, "qimg21-vae: missing tensor %s\n", name);
        return 0;
    }
    int nd = safetensors_ndims(st, idx);
    const uint64_t *shape = safetensors_shape(st, idx);
    size_t n = 1;
    for (int i = 0; i < nd; i++) n *= (size_t)shape[i];
    if (co) *co = nd >= 1 ? (int)shape[0] : 1;
    if (ci) *ci = nd >= 2 ? (int)shape[1] : 1;
    if (kh) *kh = nd >= 3 ? (int)shape[2] : 1;
    if (kw) *kw = nd >= 4 ? (int)shape[3] : 1;
    float *host = (float *)malloc(n * sizeof(float));
    if (!host) return 0;
    const char *dtype = safetensors_dtype(st, idx);
    const void *src = safetensors_data(st, idx);
    if (!strcmp(dtype, "F32")) memcpy(host, src, n * sizeof(float));
    else if (!strcmp(dtype, "BF16")) {
        const uint16_t *p = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) host[i] = q21_bf16(p[i]);
    } else {
        fprintf(stderr, "qimg21-vae: unsupported %s dtype %s\n", name, dtype);
        free(host);
        return 0;
    }
    if (q21_vae_bf16_mode)
        for (size_t i = 0; i < n; i++) host[i] = q21_round_bf16(host[i]);
    CUdeviceptr d = checked_cuMemAlloc(n * sizeof(float));
    if (d) cuMemcpyHtoD(d, host, n * sizeof(float));
    free(host);
    return d;
}

static void q21_free(CUdeviceptr *p) {
    if (*p) cuMemFree(*p);
    *p = 0;
}

static CUdeviceptr q21_load_weight(const st_context *st, const char *name) {
    return q21_upload(st, name, NULL, NULL, NULL, NULL);
}

static int q21_load_resblock(const st_context *st, const char *prefix,
                             CUdeviceptr *n1, CUdeviceptr *c1, CUdeviceptr *b1,
                             CUdeviceptr *n2, CUdeviceptr *c2, CUdeviceptr *b2,
                             CUdeviceptr *scw, CUdeviceptr *scb) {
    char name[256];
    snprintf(name, sizeof(name), "%s.norm1.gamma", prefix); *n1 = q21_load_weight(st, name);
    snprintf(name, sizeof(name), "%s.conv1.weight", prefix); *c1 = q21_load_weight(st, name);
    snprintf(name, sizeof(name), "%s.conv1.bias", prefix); *b1 = q21_load_weight(st, name);
    snprintf(name, sizeof(name), "%s.norm2.gamma", prefix); *n2 = q21_load_weight(st, name);
    snprintf(name, sizeof(name), "%s.conv2.weight", prefix); *c2 = q21_load_weight(st, name);
    snprintf(name, sizeof(name), "%s.conv2.bias", prefix); *b2 = q21_load_weight(st, name);
    snprintf(name, sizeof(name), "%s.conv_shortcut.weight", prefix);
    *scw = safetensors_find(st, name) >= 0 ? q21_load_weight(st, name) : 0;
    snprintf(name, sizeof(name), "%s.conv_shortcut.bias", prefix);
    *scb = safetensors_find(st, name) >= 0 ? q21_load_weight(st, name) : 0;
    return *n1 && *c1 && *b1 && *n2 && *c2 && *b2;
}

static void q21_free_resblock(CUdeviceptr *n1, CUdeviceptr *c1, CUdeviceptr *b1,
                              CUdeviceptr *n2, CUdeviceptr *c2, CUdeviceptr *b2,
                              CUdeviceptr *scw, CUdeviceptr *scb) {
    q21_free(n1); q21_free(c1); q21_free(b1); q21_free(n2);
    q21_free(c2); q21_free(b2); q21_free(scw); q21_free(scb);
}

static CUdeviceptr q21_resblock(cuda_qimg_runner *r, CUdeviceptr x,
                                const st_context *st, const char *prefix,
                                int ci, int co, int h, int w) {
    CUdeviceptr n1=0,c1=0,b1=0,n2=0,c2=0,b2=0,scw=0,scb=0;
    if (!q21_load_resblock(st, prefix, &n1,&c1,&b1,&n2,&c2,&b2,&scw,&scb)) {
        q21_free_resblock(&n1,&c1,&b1,&n2,&c2,&b2,&scw,&scb);
        return 0;
    }
    CUdeviceptr y = vae_resblock_gpu(r, x, n1,c1,b1,n2,c2,b2,scw,scb,ci,co,h,w);
    q21_free_resblock(&n1,&c1,&b1,&n2,&c2,&b2,&scw,&scb);
    return y;
}

/* Optional stage trace used while bringing up the native decoder.  Keeping
 * this separate from vae_resblock_gpu leaves the shared qimg implementation
 * untouched and makes the trace opt-in via QIMG21_VAE_TRACE. */
static CUdeviceptr q21_resblock_trace(cuda_qimg_runner *r, CUdeviceptr x,
                                      const st_context *st, const char *prefix,
                                      int ci, int co, int h, int w) {
    CUdeviceptr n1=0,c1=0,b1=0,n2=0,c2=0,b2=0,scw=0,scb=0;
    int sp=h*w; CUdeviceptr tmp=0,c1o=0,c2o=0,out=0;
    q21_dump(r,x,ci,h,w,"trace_input_before");
    if (!q21_load_resblock(st,prefix,&n1,&c1,&b1,&n2,&c2,&b2,&scw,&scb)) goto fail;
    q21_dump(r,x,ci,h,w,"trace_input");
    q21_dump(r,n1,ci,1,1,"trace_gamma1");
    tmp=checked_cuMemAlloc((size_t)ci*sp*sizeof(float)); if(!tmp) goto fail;
    vae_op_gn(r,tmp,x,n1,ci,sp); vae_bf16(r,tmp,ci*sp); q21_dump(r,tmp,ci,h,w,"trace_norm1");
    vae_op_silu(r,tmp,ci*sp); vae_bf16(r,tmp,ci*sp); q21_dump(r,tmp,ci,h,w,"trace_silu1");
    c1o=checked_cuMemAlloc((size_t)co*sp*sizeof(float)); if(!c1o) goto fail;
    vae_op_conv2d(r,c1o,tmp,c1,b1,ci,h,w,co,3,3,0); vae_bf16(r,c1o,co*sp); q21_dump(r,c1o,co,h,w,"trace_conv1");
    q21_free(&tmp); tmp=checked_cuMemAlloc((size_t)co*sp*sizeof(float)); if(!tmp) goto fail;
    vae_op_gn(r,tmp,c1o,n2,co,sp); vae_bf16(r,tmp,co*sp); q21_dump(r,tmp,co,h,w,"trace_norm2");
    vae_op_silu(r,tmp,co*sp); vae_bf16(r,tmp,co*sp); q21_dump(r,tmp,co,h,w,"trace_silu2");
    c2o=checked_cuMemAlloc((size_t)co*sp*sizeof(float)); if(!c2o) goto fail;
    vae_op_conv2d(r,c2o,tmp,c2,b2,co,h,w,co,3,3,0); vae_bf16(r,c2o,co*sp); q21_dump(r,c2o,co,h,w,"trace_conv2");
    out=checked_cuMemAlloc((size_t)co*sp*sizeof(float)); if(!out) goto fail;
    if(scw) vae_op_conv2d(r,out,x,scw,scb,ci,h,w,co,1,1,0); else cuMemcpyDtoD(out,x,(size_t)co*sp*sizeof(float));
    q21_dump(r,out,co,h,w,"trace_shortcut");
    { int n=co*sp; float one=1.0f; void *a[]={&out,&c2o,&one,&n}; cuLaunchKernel(r->euler_step,(n+255)/256,1,1,256,1,1,0,r->stream,a,NULL); }
    vae_bf16(r,out,co*sp);
    q21_dump(r,out,co,h,w,"trace_out");
    q21_free_resblock(&n1,&c1,&b1,&n2,&c2,&b2,&scw,&scb); q21_free(&tmp);q21_free(&c1o);q21_free(&c2o); return out;
fail:
    q21_free_resblock(&n1,&c1,&b1,&n2,&c2,&b2,&scw,&scb); q21_free(&tmp);q21_free(&c1o);q21_free(&c2o);q21_free(&out); return 0;
}

static CUdeviceptr q21_conv(cuda_qimg_runner *r, const st_context *st,
                            CUdeviceptr x, int ci, int h, int w, int co,
                            const char *weight, const char *bias) {
    CUdeviceptr dw = q21_load_weight(st, weight);
    CUdeviceptr db = q21_load_weight(st, bias);
    if (!dw || !db) { q21_free(&dw); q21_free(&db); return 0; }
    CUdeviceptr y = checked_cuMemAlloc((size_t)co * h * w * sizeof(float));
    if (y) vae_op_conv2d(r, y, x, dw, db, ci, h, w, co, 3, 3, 0);
    q21_free(&dw); q21_free(&db);
    return y;
}

static CUdeviceptr q21_conv1(cuda_qimg_runner *r, const st_context *st,
                             CUdeviceptr x, int ci, int h, int w, int co,
                             const char *weight, const char *bias) {
    CUdeviceptr dw = q21_load_weight(st, weight);
    CUdeviceptr db = q21_load_weight(st, bias);
    if (!dw || !db) { q21_free(&dw); q21_free(&db); return 0; }
    CUdeviceptr y = checked_cuMemAlloc((size_t)co * h * w * sizeof(float));
    if (y) vae_op_conv2d(r, y, x, dw, db, ci, h, w, co, 1, 1, 0);
    q21_free(&dw); q21_free(&db);
    return y;
}

/* First temporal frame of Qwen-Image's DupUp3D shortcut.  The decoder only
 * receives one latent frame, but the residual decoder still performs the
 * temporal shuffle.  Keeping the first frame is exactly what the pipeline
 * does when it later selects [:, :, 0]. */
static CUdeviceptr q21_dup_first(cuda_qimg_runner *r, CUdeviceptr x,
                                 int in_c, int out_c, int h, int w,
                                 int factor_t, int factor_s) {
    int factor = factor_t * factor_s * factor_s;
    int repeats = out_c * factor / in_c;
    size_t in_n = (size_t)in_c * h * w;
    size_t out_n = (size_t)out_c * (h * factor_s) * (w * factor_s);
    float *hin = (float *)malloc(in_n * sizeof(float));
    float *hout = (float *)malloc(out_n * sizeof(float));
    if (!hin || !hout) { free(hin); free(hout); return 0; }
    cuStreamSynchronize(r->stream);
    if (cuMemcpyDtoH(hin, x, in_n * sizeof(float)) != CUDA_SUCCESS) {
        free(hin); free(hout); return 0;
    }
    int oh = h * factor_s, ow = w * factor_s;
    for (int oc = 0; oc < out_c; oc++) for (int oy = 0; oy < oh; oy++) for (int ox = 0; ox < ow; ox++) {
        int sy = oy % factor_s, sx = ox % factor_s;
        int iy = oy / factor_s, ix = ox / factor_s;
        /* first_chunk=True keeps temporal slice factor_t-1 after the
         * channel repeat/reshape.  The spatial subpixel is interleaved
         * inside that selected temporal group. */
        size_t rep = (((size_t)oc * factor_t + (size_t)(factor_t - 1)) *
                      factor_s * factor_s + (size_t)sy * factor_s + sx);
        int ic = (int)(rep / (size_t)repeats);
        hout[(size_t)oc * oh * ow + (size_t)oy * ow + ox] = hin[(size_t)ic * h * w + (size_t)iy * w + ix];
    }
    CUdeviceptr out = checked_cuMemAlloc(out_n * sizeof(float));
    if (out) cuMemcpyHtoD(out, hout, out_n * sizeof(float));
    free(hin); free(hout);
    return out;
}

static CUdeviceptr q21_mid_attention_named(cuda_qimg_runner *r, const st_context *st,
                                     CUdeviceptr x, int c, int h, int w, const char *prefix) {
    int spatial = h * w;
    CUdeviceptr gn=0,qkvw=0,qkvb=0,pw=0,pb=0,norm=0,qkv=0,qs=0,ks=0,vs=0,as=0,ach=0,po=0;
    char name[256];
    snprintf(name,sizeof(name),"%s.norm.gamma",prefix);gn=q21_load_weight(st,name);
    snprintf(name,sizeof(name),"%s.to_qkv.weight",prefix);qkvw=q21_load_weight(st,name);
    snprintf(name,sizeof(name),"%s.to_qkv.bias",prefix);qkvb=q21_load_weight(st,name);
    snprintf(name,sizeof(name),"%s.proj.weight",prefix);pw=q21_load_weight(st,name);
    snprintf(name,sizeof(name),"%s.proj.bias",prefix);pb=q21_load_weight(st,name);
    if(!gn||!qkvw||!qkvb||!pw||!pb) goto fail;
    norm=checked_cuMemAlloc((size_t)c*spatial*sizeof(float));
    qkv=checked_cuMemAlloc((size_t)3*c*spatial*sizeof(float));
    if(!norm||!qkv) goto fail;
    vae_op_gn(r,norm,x,gn,c,spatial);
    vae_bf16(r,norm,c*spatial);
    vae_op_conv2d(r,qkv,norm,qkvw,qkvb,c,h,w,3*c,1,1,0);
    vae_bf16(r,qkv,3*c*spatial);
    q21_dump(r,qkv,3*c,h,w,"mid_qkv");
    q21_free(&gn); q21_free(&qkvw); q21_free(&qkvb); q21_free(&norm);
    qs=checked_cuMemAlloc((size_t)spatial*c*sizeof(float));
    ks=checked_cuMemAlloc((size_t)spatial*c*sizeof(float));
    vs=checked_cuMemAlloc((size_t)spatial*c*sizeof(float));
    if(!qs||!ks||!vs) goto fail;
    { unsigned bx=16,by=16,gx=(unsigned)((spatial+15)/16),gy=(unsigned)((c+15)/16);
      CUdeviceptr q=qkv,k=qkv+(size_t)c*spatial*sizeof(float),v=k+(size_t)c*spatial*sizeof(float);
      int sp=spatial; void *a[]={&qs,&q,&c,&sp}; cuLaunchKernel(r->vae_transpose_chw_to_sc,gx,gy,1,bx,by,1,0,r->stream,a,NULL);
      void *b[]={&ks,&k,&c,&sp}; cuLaunchKernel(r->vae_transpose_chw_to_sc,gx,gy,1,bx,by,1,0,r->stream,b,NULL);
      void *d[]={&vs,&v,&c,&sp}; cuLaunchKernel(r->vae_transpose_chw_to_sc,gx,gy,1,bx,by,1,0,r->stream,d,NULL); }
    q21_free(&qkv);
    as=checked_cuMemAlloc((size_t)spatial*c*sizeof(float));
    if(!as) goto fail;
    if(q21_cutlass_vae_attention) {
      if(q21_cutlass_vae_attention((float*)(uintptr_t)as,(float*)(uintptr_t)qs,
          (float*)(uintptr_t)ks,(float*)(uintptr_t)vs,spatial,c,(void*)(uintptr_t)r->stream))goto fail;
    } else { float scale=1.0f/sqrtf((float)c); int sp=spatial; size_t smem=(size_t)2*c*sizeof(float);
      void *a[]={&as,&qs,&ks,&vs,&sp,&c,&scale}; cuLaunchKernel(r->vae_attn_sc,(unsigned)spatial,1,1,32,1,1,smem,r->stream,a,NULL); }
    vae_bf16(r,as,c*spatial);
    q21_dump(r,as,c,1,spatial,"mid_attn_raw");
    q21_free(&qs); q21_free(&ks); q21_free(&vs);
    ach=checked_cuMemAlloc((size_t)c*spatial*sizeof(float));
    if(!ach) goto fail;
    { unsigned bx=16,by=16,gx=(unsigned)((spatial+15)/16),gy=(unsigned)((c+15)/16); int sp=spatial;
      void *a[]={&ach,&as,&c,&sp}; cuLaunchKernel(r->vae_transpose_sc_to_chw,gx,gy,1,bx,by,1,0,r->stream,a,NULL); }
    q21_free(&as);
    po=checked_cuMemAlloc((size_t)c*spatial*sizeof(float));
    if(!po) goto fail;
    vae_op_conv2d(r,po,ach,pw,pb,c,h,w,c,1,1,0);
    vae_bf16(r,po,c*spatial);
    q21_dump(r,po,c,h,w,"mid_proj");
    q21_free(&ach); q21_free(&pw); q21_free(&pb);
    { int n=c*spatial; float one=1.0f; void *a[]={&x,&po,&one,&n}; cuLaunchKernel(r->euler_step,(n+255)/256,1,1,256,1,1,0,r->stream,a,NULL); }
    vae_bf16(r,x,c*spatial);
    q21_free(&po);
    return x;
fail:
    q21_free(&gn);q21_free(&qkvw);q21_free(&qkvb);q21_free(&pw);q21_free(&pb);q21_free(&norm);q21_free(&qkv);
    q21_free(&qs);q21_free(&ks);q21_free(&vs);q21_free(&as);q21_free(&ach);q21_free(&po);
    return 0;
}

static CUdeviceptr q21_mid_attention(cuda_qimg_runner *r, const st_context *st,
                                     CUdeviceptr x, int c, int h, int w) {
    return q21_mid_attention_named(r,st,x,c,h,w,"decoder.mid_block.attentions.0");
}

static void q21_latent_stats(float *out_mean, float *out_std) {
    /* qimg-21 stores normalized latent channels; decode expects z*std+mean. */
    static const float mean[64] = {
      0.5126f,0.7721f,-0.0631f,1.3506f,-0.7855f,-2.1025f,-0.3458f,1.3722f,
      1.8873f,-1.7177f,-0.651f,0.2732f,0.7562f,-0.6163f,-1.0277f,3.8363f,
      2.021f,0.0472f,0.932f,2.0087f,2.4954f,-0.1391f,-1.4249f,1.8464f,
      -0.5236f,1.2826f,3.7046f,-1.3035f,2.7286f,-1.4518f,-1.9036f,-1.9955f,
      -0.0342f,-1.0265f,-0.7636f,3.0555f,0.0746f,-3.0751f,-0.1076f,1.7376f,
      -1.0914f,-1.9435f,-0.2784f,-1.368f,0.4809f,-0.4433f,0.3764f,0.5729f,
      -2.0595f,1.096f,-1.326f,-2.0211f,-5.0179f,0.5275f,4.0162f,1.8505f,
      0.3026f,1.9373f,1.4937f,0.2632f,0.5547f,-1.7121f,-0.1562f,0.0304f };
    static const float std[64] = {
      3.2001f,3.2936f,3.4321f,3.0091f,3.1061f,4.0379f,4.0705f,3.791f,
      3.0785f,3.65f,3.9308f,3.0904f,2.8778f,3.7675f,3.732f,5.0756f,
      3.2864f,4.0397f,3.1317f,4.0443f,2.9249f,3.9454f,3.0988f,4.2489f,
      3.4896f,3.8513f,3.9323f,3.4719f,3.7498f,4.283f,3.5694f,4.2467f,
      3.9037f,3.2947f,5.077f,3.5075f,3.27f,3.4767f,2.8063f,5.1125f,
      3.5327f,4.7833f,3.1286f,4.1819f,3.8527f,3.8312f,3.5605f,4.3875f,
      3.9624f,4.0168f,3.5643f,4.055f,5.5614f,4.2963f,4.408f,3.4959f,
      3.8747f,3.7608f,3.5735f,3.149f,3.7662f,3.6746f,3.4563f,3.8161f };
    memcpy(out_mean,mean,sizeof(mean));memcpy(out_std,std,sizeof(std));
}

static int qimg21_vae_decode(cuda_qimg_runner *r, const st_context *st,
                             const float *latent, int h, int w, float *out) {
    const int c0=64;
    float mean[64],std[64];q21_latent_stats(mean,std);
    CUdeviceptr x=checked_cuMemAlloc((size_t)c0*h*w*sizeof(float));
    if(!x)return -1;
    /* A tiny host transform avoids another model-specific CUDA kernel. */
    float *z=(float *)malloc((size_t)c0*h*w*sizeof(float));
    if(!z){q21_free(&x);return -1;}
    /* Native denoising stores [token, channel].  The VAE consumes the
     * channel-first [channel, height, width] layout used by Conv2d. */
    for(int cc=0;cc<c0;cc++) for(int i=0;i<h*w;i++)
        z[(size_t)cc*h*w+i]=latent[(size_t)i*c0+cc]*std[cc]+mean[cc];
    cuMemcpyHtoD(x,z,(size_t)c0*h*w*sizeof(float)); free(z);
    q21_dump(r,x,c0,h,w,"latent_denorm");
    /* AutoencoderKLQwenImage21 applies post_quant_conv before entering the
     * decoder.  It is a learned 1x1 Conv3d whose single temporal slice is
     * exactly this per-frame Conv2d. */
    CUdeviceptr post = q21_conv1(r,st,x,c0,h,w,c0,
                                 "post_quant_conv.weight","post_quant_conv.bias");
    q21_free(&x); if(!post)return -1; x=post;
    q21_dump(r,x,c0,h,w,"post_quant");
    CUdeviceptr y=q21_conv(r,st,x,c0,h,w,1152,"decoder.conv_in.weight","decoder.conv_in.bias");
    q21_free(&x); if(!y)return -1; x=y; int c=1152;
    q21_dump(r,x,c,h,w,"conv_in");
    y=q21_resblock(r,x,st,"decoder.mid_block.resnets.0",c,c,h,w);
    q21_free(&x); if(!y)return -1; x=y; q21_dump(r,x,c,h,w,"mid_res0");
    /* QwenImage21MidBlock order is ResNet -> attention -> ResNet. */
    y=q21_mid_attention(r,st,x,c,h,w); if(!y){q21_free(&x);return -1;} x=y;
    q21_dump(r,x,c,h,w,"mid_attn");
    y=(q21_vae_dump_dir && getenv("QIMG21_VAE_TRACE"))
        ? q21_resblock_trace(r,x,st,"decoder.mid_block.resnets.1",c,c,h,w)
        : q21_resblock(r,x,st,"decoder.mid_block.resnets.1",c,c,h,w);
    q21_free(&x); if(!y)return -1; x=y; q21_dump(r,x,c,h,w,"mid_res1");
    const int up_in[5]={1152,1152,1152,576,288};
    const int up_out[5]={1152,1152,576,288,144};
    for(int u=0;u<5;u++){
        int cin=up_in[u], cout=up_out[u];
        CUdeviceptr skip=0;
        if(u<4){
            size_t sn=(size_t)cin*h*w;
            skip=checked_cuMemAlloc(sn*sizeof(float));
            if(!skip){q21_free(&x);return -1;}
            cuMemcpyDtoD(skip,x,sn*sizeof(float));
        }
        for(int b=0;b<3;b++){
            char p[128]; snprintf(p,sizeof(p),"decoder.up_blocks.%d.resnets.%d",u,b);
            y=(u==0 && b==2 && q21_vae_dump_dir && getenv("QIMG21_VAE_TRACE_RES2"))
                ? q21_resblock_trace(r,x,st,p,cin,cout,h,w)
                : q21_resblock(r,x,st,p,cin,cout,h,w);
            q21_free(&x); if(!y)return -1; x=y; cin=cout;
            if (q21_vae_dump_dir && getenv("QIMG21_VAE_TRACE") && u==0) {
                char label[64]; snprintf(label,sizeof(label),"trace_up0_res%d",b);
                q21_dump(r,x,cin,h,w,label);
            }
        }
        c=cout;
        if(u<4){
            CUdeviceptr up=vae_op_upsample(r,x,c,h,w); q21_free(&x); if(!up)return -1;
            h*=2; w*=2;
            char wn[128],bn[128]; snprintf(wn,sizeof(wn),"decoder.up_blocks.%d.upsampler.resample.1.weight",u); snprintf(bn,sizeof(bn),"decoder.up_blocks.%d.upsampler.resample.1.bias",u);
            CUdeviceptr dw=q21_load_weight(st,wn),db=q21_load_weight(st,bn),co_buf=checked_cuMemAlloc((size_t)c*h*w*sizeof(float));
            if(!dw||!db||!co_buf){q21_free(&up);q21_free(&dw);q21_free(&db);q21_free(&co_buf);q21_free(&skip);return -1;}
            vae_op_conv2d(r,co_buf,up,dw,db,c,h,w,c,3,3,0);
            q21_free(&up); q21_free(&dw); q21_free(&db);
            x=co_buf;
            /* With first_chunk=True the decoder's cached upsample3d path
             * only primes its temporal cache on this first frame; it does
             * not apply time_conv yet.  The spatial resample above is the
             * complete main-path operation for this single-frame decode. */
            CUdeviceptr sc=q21_dup_first(r,skip,up_in[u],cout,h/2,w/2,u<3?2:1,2);
            q21_free(&skip); if(!sc){q21_free(&x);return -1;}
            if (q21_vae_dump_dir && getenv("QIMG21_VAE_TRACE") && u==0) {
                q21_dump(r,x,c,h,w,"trace_up0_main");
                q21_dump(r,sc,cout,h,w,"trace_up0_shortcut");
            }
            { int n=cout*h*w; float one=1.0f; void *a[]={&x,&sc,&one,&n}; cuLaunchKernel(r->euler_step,(n+255)/256,1,1,256,1,1,0,r->stream,a,NULL); }
            q21_free(&sc);
        }
        q21_dump(r,x,c,h,w,u==0?"up0":u==1?"up1":u==2?"up2":u==3?"up3":"up4");
    }
    {
        CUdeviceptr gn=q21_load_weight(st,"decoder.norm_out.gamma");
        CUdeviceptr norm=checked_cuMemAlloc((size_t)c*h*w*sizeof(float));
        if(!gn||!norm){q21_free(&gn);q21_free(&norm);q21_free(&x);return -1;}
        vae_op_gn(r,norm,x,gn,c,h*w); q21_free(&gn); q21_free(&x);
        vae_op_silu(r,norm,c*h*w);
        CUdeviceptr ow=q21_load_weight(st,"decoder.conv_out.weight");
        CUdeviceptr ob=q21_load_weight(st,"decoder.conv_out.bias");
        CUdeviceptr outd=checked_cuMemAlloc((size_t)4*h*w*sizeof(float));
        if(!ow||!ob||!outd){q21_free(&norm);q21_free(&ow);q21_free(&ob);q21_free(&outd);return -1;}
        vae_op_conv2d(r,outd,norm,ow,ob,c,h,w,4,3,3,0);
        CUresult copy=cuStreamSynchronize(r->stream);
        if (copy == CUDA_SUCCESS)
            copy=cuMemcpyDtoH(out,outd,(size_t)4*h*w*sizeof(float));
        q21_dump(r,outd,4,h,w,"conv_out");
        if (copy == CUDA_SUCCESS) {
            for (size_t i=0; i<(size_t)4*h*w; i++) {
                if (!isfinite(out[i])) { copy=CUDA_ERROR_INVALID_VALUE; break; }
                out[i]=fmaxf(-1.0f,fminf(1.0f,out[i]));
            }
        }
        q21_free(&norm);q21_free(&ow);q21_free(&ob);q21_free(&outd);
        return copy==CUDA_SUCCESS ? 0 : -1;
    }
}

int main(int argc, char **argv) {
    const char *model=NULL,*latent_path=NULL,*out_path=NULL; int h=0,w=0,verbose=1;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--model")&&i+1<argc)model=argv[++i];
        else if(!strcmp(argv[i],"--latents")&&i+1<argc)latent_path=argv[++i];
        else if(!strcmp(argv[i],"--height-tokens")&&i+1<argc)h=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--width-tokens")&&i+1<argc)w=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--out")&&i+1<argc)out_path=argv[++i];
        else if(!strcmp(argv[i],"--quiet"))verbose=0;
        else {fprintf(stderr,"usage: %s --model VAE_DIR --latents L.npy --height-tokens H --width-tokens W --out OUT.npy\n",argv[0]);return 2;}
    }
    if(!model||!latent_path||!out_path||h<=0||w<=0)return 2;
    /* All shared kernel element indices are signed 32-bit. The largest
     * stage has 288 channels at the final spatial resolution. */
    if ((uint64_t)h*w > INT_MAX / (288u*256u)) {
        fprintf(stderr,"qimg21-vae: dimensions exceed kernel indexing limits\n");
        return 2;
    }
    q21_vae_dump_dir = getenv("QIMG21_VAE_DUMP_DIR");
    if(q21_vae_dump_dir) mkdir(q21_vae_dump_dir,0755);
    q21_npy a; if(q21_npy_read_f32(latent_path,&a)!=0)return 1;
    size_t need=(size_t)h*w*64;
    if(a.n!=need || a.ndim!=2 || a.shape[0]!=(size_t)h*w || a.shape[1]!=64){
        fprintf(stderr,"qimg21-vae: expected latent shape [%d,64]\n",h*w);
        q21_npy_free(&a);return 1;
    }
    for(size_t i=0;i<a.n;i++) if(!isfinite(a.data[i])) {
        fprintf(stderr,"qimg21-vae: non-finite input latent\n");
        q21_npy_free(&a);return 1;
    }
    char st_path[1024]; snprintf(st_path,sizeof(st_path),"%s/diffusion_pytorch_model.safetensors",model);
    st_context *st=safetensors_open(st_path); if(!st){q21_npy_free(&a);return 1;}
    cuda_qimg_runner *r=cuda_qimg_init(0,verbose); if(!r){safetensors_close(st);q21_npy_free(&a);return 1;}
    r->use_fp8_pipe=0; r->use_fp8_pipe_perrow=0; /* retain F32 VAE quality */
    /* The shared VAE helpers use synchronous default-stream D2D copies for
     * residuals. Keep their kernels on that same stream: a nonblocking
     * stream otherwise races those copies and intermittently loses residuals. */
    CUstream saved_stream = r->stream;
    cuStreamSynchronize(saved_stream);
    r->stream = NULL;
    float *out=(float *)malloc((size_t)4*(h*16)*(w*16)*sizeof(float));
    int rc=out ? qimg21_vae_decode(r,st,a.data,h,w,out) : -1;
    cuStreamSynchronize(r->stream);
    r->stream = saved_stream;
    if(!rc) rc=q21_npy_write_chw(out_path,out,(size_t)4*(h*16)*(w*16),4,h*16,w*16);
    free(out); cuda_qimg_free(r); safetensors_close(st); q21_npy_free(&a); return rc?1:0;
}
