/* Lossless four-bit indices into the original signed IQ palette.
 * Requires mixed_iq_decode.h. Original per-16 FP32 scales are retained.
 * Low/high nibbles encode the first/last 32 values of each 64-value group.
 */
#ifndef A64FX_IQ4_DECODE_CACHE_H
#define A64FX_IQ4_DECODE_CACHE_H

typedef struct { uint8_t q[128]; float d[16]; } tf_iq4_cache_block;
typedef struct {
    int n_slices, n_cols;
    uint32_t type;
    int start[49];
    tf_iq4_cache_block *slice[48];
} tf_iq4_cache_view;

static const int8_t *tf_iq4_cache_palette(uint32_t type) {
    static const int8_t iq2[16] = {-43,-25,-8,8,25,43};
    static const int8_t iq3s[16] = {-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15};
    static const int8_t iq3xxs[16] = {-62,-52,-44,-36,-28,-20,-12,-4,4,12,20,28,36,44,52,62};
    if (type == GGML_TYPE_IQ3_S) return iq3s;
    if (type == GGML_TYPE_IQ3_XXS) return iq3xxs;
    return tf_mixed_iq_expandable(type) ? iq2 : NULL;
}

static int tf_iq4_cache_pack_row(tf_iq4_cache_block *dst, const void *src,
                                  uint32_t type, int n) {
    const int8_t *palette = tf_iq4_cache_palette(type);
    if (!palette || n<=0 || n%256) return -1;
    int8_t inverse[256];
    memset(inverse,-1,sizeof(inverse));
    for (int i=0;i<16;i++) inverse[(uint8_t)palette[i]]=(int8_t)i;
    size_t rb=tf_row_bytes(type,256);
    for (int b=0;b<n/256;b++) {
        const uint8_t *raw=(const uint8_t *)src+(size_t)b*rb;
        tf_mixed_weight_block expanded;
        if (type == GGML_TYPE_IQ3_XXS) {
            const block_iq3_xxs *w=(const block_iq3_xxs *)raw;
            float decoded[256];
            dequantize_row_iq3_xxs(raw,decoded,256);
            for (int g=0;g<16;g++) {
                uint32_t aux;
                memcpy(&aux,w->qs+64+(g/2)*4,sizeof(aux));
                float d=ggml_fp16_to_fp32(w->d)*(1+2*(aux>>28))*0.25f;
                expanded.d[g]=d;
                for (int k=0;k<16;k++) {
                    float q=d?nearbyintf(decoded[g*16+k]/d):0;
                    if (!isfinite(q) || q < -127 || q > 127 || q*d != decoded[g*16+k]) return -1;
                    expanded.q[g*16+k]=(int8_t)q;
                }
            }
        } else if (tf_mixed_expand_row(&expanded,raw,type,256)) return -1;
        memcpy(dst[b].d,expanded.d,sizeof(expanded.d));
        for (int g=0;g<4;g++) for (int k=0;k<32;k++) {
            int i0=g*64+k,i1=i0+32;
            int q0=expanded.d[i0/16]?inverse[(uint8_t)expanded.q[i0]]:0;
            int q1=expanded.d[i1/16]?inverse[(uint8_t)expanded.q[i1]]:0;
            if (q0<0 || q1<0) return -1;
            dst[b].q[g*32+k]=(uint8_t)(q0 | (q1<<4));
        }
    }
    return 0;
}

static inline __attribute__((always_inline)) float tf_iq4_cache_dot(
    const tf_iq4_cache_block *w, const float *x, const tf_mixed_q8_block *qx,
    int nb, const int8_t *palette) {
    const svbool_t pg=svptrue_b32(), p8=svptrue_b8();
    const svbool_t p16=svwhilelt_b8(0,16),p32=svwhilelt_b8(0,32);
    /* All palette integers are exactly representable in FP32. Convert once
     * per row instead of after each table lookup in the weight loop. */
    const svfloat32_t values=svcvt_f32_s32_x(pg,svld1sb_s32(pg,palette));
    const svint8_t bytes=svld1_s8(p16,palette);
    const svuint32_t scale_ix=svlsr_n_u32_x(pg,svindex_u32(0,1),2);
    svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0;
    for (int b=0;b<nb;b++) for (int g=0;g<4;g++) {
        const float *d=w[b].d+g*4;
        if (qx) {
            svuint8_t q=svld1_u8(p32,w[b].q+g*32);
            svint8_t lo=svtbl_s8(bytes,svand_n_u8_x(p8,q,15));
            svint8_t hi=svtbl_s8(bytes,svlsr_n_u8_x(p8,q,4));
            svint8_t v=svsplice_s8(p32,lo,hi);
            svint32_t dot=svdot_s32(svdup_s32(0),v,svld1_s8(p8,qx[b].q+g*64));
            svfloat32_t scale=svtbl_f32(svld1rq_f32(pg,d),scale_ix);
            a0=svmla_f32_x(pg,a0,svcvt_f32_s32_x(pg,dot),svmul_n_f32_x(pg,scale,qx[b].d));
        } else {
            svuint32_t q0=svld1ub_u32(pg,w[b].q+g*32);
            svuint32_t q1=svld1ub_u32(pg,w[b].q+g*32+16);
            svfloat32_t w0=svtbl_f32(values,svand_n_u32_x(pg,q0,15));
            svfloat32_t w1=svtbl_f32(values,svand_n_u32_x(pg,q1,15));
            svfloat32_t w2=svtbl_f32(values,svlsr_n_u32_x(pg,q0,4));
            svfloat32_t w3=svtbl_f32(values,svlsr_n_u32_x(pg,q1,4));
            const float *v=x+b*256+g*64;
            a0=svmla_f32_x(pg,a0,svmul_n_f32_x(pg,w0,d[0]),svld1_f32(pg,v));
            a1=svmla_f32_x(pg,a1,svmul_n_f32_x(pg,w1,d[1]),svld1_f32(pg,v+16));
            a2=svmla_f32_x(pg,a2,svmul_n_f32_x(pg,w2,d[2]),svld1_f32(pg,v+32));
            a3=svmla_f32_x(pg,a3,svmul_n_f32_x(pg,w3,d[3]),svld1_f32(pg,v+48));
        }
    }
    return svaddv_f32(pg,svadd_f32_x(pg,svadd_f32_x(pg,a0,a1),svadd_f32_x(pg,a2,a3)));
}

static void tf_iq4_cache_rows(float *dst, const tf_iq4_cache_block *w, const float *x,
                               int n, uint32_t type, int start, int end, int q8) {
    const int8_t *palette=tf_iq4_cache_palette(type);
    const int nb=n/256;
    if (q8 || type==GGML_TYPE_IQ3_XXS) {
        tf_mixed_q8_block *qx=alloca((size_t)nb*sizeof(*qx));
        tf_mixed_quant_q8(qx,x,n);
        for (int r=start;r<end;r++) dst[r]=tf_iq4_cache_dot(w+(size_t)r*nb,x,qx,nb,palette);
    } else {
        for (int r=start;r<end;r++) dst[r]=tf_iq4_cache_dot(w+(size_t)r*nb,x,NULL,nb,palette);
    }
}

/* Slices are allocated by their consuming pinned workers, not just touched
 * by them: Fugaku's hugepage allocator can establish NUMA placement at malloc. */
static void tf_iq4_cache_view_rows(float *dst, const void *opaque, const float *x,
                                   int start, int end) {
    const tf_iq4_cache_view *v=opaque;
    const int nb=v->n_cols/256;
    const int8_t *palette=tf_iq4_cache_palette(v->type);
    tf_mixed_q8_block *qx=NULL;
    if (tf_mixed_iq_q8_enabled || v->type==GGML_TYPE_IQ3_XXS) {
        qx=alloca((size_t)nb*sizeof(*qx));
        tf_mixed_quant_q8(qx,x,v->n_cols);
    }
    for (int s=0;s<v->n_slices && start<end;s++) {
        if (v->start[s+1]<=start) continue;
        int stop=end<v->start[s+1]?end:v->start[s+1];
        if (qx) {
            for (int r=start;r<stop;r++) dst[r]=tf_iq4_cache_dot(
                v->slice[s]+(size_t)(r-v->start[s])*nb,x,qx,nb,palette);
        } else {
            for (int r=start;r<stop;r++) dst[r]=tf_iq4_cache_dot(
                v->slice[s]+(size_t)(r-v->start[s])*nb,x,NULL,nb,palette);
        }
        start=stop;
    }
}
#endif
