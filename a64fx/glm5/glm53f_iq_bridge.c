/* Reuse the validated GLM-5.2 A64FX mixed-IQ kernels in the GLM-5.3 graph.
 * Keeping this bridge in one translation unit avoids duplicating the large IQ
 * lookup tables in every consumer of glm53f_expert_kern.h. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GLM5_IMPL
#include "../../common/glm5.h"
#include "glm53f_iq_bridge.h"
#include "glm53f_iq_metadata.h"

int glm53f_iq_type_supported(int type) {
    return type == GLM53F_GGML_Q4_K || type == GLM53F_GGML_Q5_K ||
           type == GLM53F_GGML_Q6_K ||
           type == GLM53F_GGML_IQ2_XS ||
           type == GLM53F_GGML_IQ3_XXS ||
           type == GLM53F_GGML_IQ4_XS;
}

static float q6_k_q8_row(const block_q6_K *w,
                         const glm5_iq_q8_block *x, int blocks) {
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const svbool_t p4 = svwhilelt_b32(0, 4), p32bytes = svwhilelt_b8(0, 32);
    svfloat32_t acc = svdup_f32(0.0f);
    for (int b = 0; b < blocks; ++b) {
        const float ds = ggml_fp16_to_fp32(w[b].d) * x[b].d;
        for (int half = 0; half < 2; ++half) {
            const uint8_t *ql = w[b].ql + half * 64;
            const svuint8_t qh = svld1_u8(p32bytes, w[b].qh + half * 32);
            const int8_t *sc = w[b].scales + half * 8;
            for (int g = 0; g < 4; ++g) {
                svuint8_t q = svld1_u8(p32bytes, ql + ((g & 1) ? 32 : 0));
                q = (g >= 2) ? svlsr_n_u8_x(p8, q, 4) : svand_n_u8_x(p8, q, 15);
                svuint8_t hi = svand_n_u8_x(p8, svlsr_n_u8_x(p8, qh, 2 * g), 3);
                svint8_t qs = svreinterpret_s8_u8(svsub_n_u8_x(
                    p8, svorr_u8_x(p8, q, svlsl_n_u8_x(p8, hi, 4)), 32));
                const svint8_t xv = svld1_s8(p32bytes,
                    x[b].q + half * 128 + g * 32);
                const svint32_t dot = svdot_s32(svdup_s32(0), qs, xv);
                const svint32_t scale = svsel_s32(p4, svdup_s32(sc[2 * g]),
                                                       svdup_s32(sc[2 * g + 1]));
                acc = svmla_n_f32_x(p32, acc,
                    svcvt_f32_s32_x(p32, svmul_s32_x(p32, dot, scale)), ds);
            }
        }
    }
    return svaddv_f32(p32, acc);
}

static float q5_k_q8_row(const block_q5_K *w,
                         const glm5_iq_q8_block *x, int blocks) {
    const svbool_t p8 = svptrue_b8();
    const svbool_t p32 = svptrue_b32();
    const svbool_t p32bytes = svwhilelt_b8(0, 32);
    const svbool_t lo8 = svwhilelt_b32(0, 8);
    const svint8_t ones = svdup_s8(1);
    svfloat32_t acc = svdup_f32(0.0f);
    for (int b = 0; b < blocks; ++b) {
        const float d = ggml_fp16_to_fp32(w[b].d);
        const float dm = ggml_fp16_to_fp32(w[b].dmin);
        svint32_t dots = svdup_s32(0), mins = svdup_s32(0);
        int is = 0;
        uint8_t bit0 = 1, bit1 = 2;
        for (int g = 0; g < 256; g += 64, is += 2, bit0 <<= 2, bit1 <<= 2) {
            uint8_t sc0, min0, sc1, min1;
            get_scale_min_k4(is, w[b].scales, &sc0, &min0);
            get_scale_min_k4(is + 1, w[b].scales, &sc1, &min1);
            const svuint8_t packed = svld1_u8(p32bytes, w[b].qs + (g >> 1));
            const svuint8_t high = svld1_u8(p32bytes, w[b].qh);
            svuint8_t uq0 = svand_n_u8_x(p8, packed, 15);
            svuint8_t uq1 = svlsr_n_u8_x(p8, packed, 4);
            uq0 = svadd_u8_x(p8, uq0, svsel_u8(svcmpne_n_u8(p32bytes,
                svand_n_u8_x(p8, high, bit0), 0), svdup_u8(16), svdup_u8(0)));
            uq1 = svadd_u8_x(p8, uq1, svsel_u8(svcmpne_n_u8(p32bytes,
                svand_n_u8_x(p8, high, bit1), 0), svdup_u8(16), svdup_u8(0)));
            const svint8_t qv = svreinterpret_s8_u8(svsplice_u8(p32bytes, uq0, uq1));
            const svint8_t xv = svld1_s8(p8, x[b].q + g);
            const svint32_t dot = svdot_s32(svdup_s32(0),qv,xv);
            const svint32_t sx = svdot_s32(svdup_s32(0),ones,xv);
            const svint32_t scv=svsel_s32(lo8,svdup_s32(sc0),svdup_s32(sc1));
            const svint32_t mnv=svsel_s32(lo8,svdup_s32(min0),svdup_s32(min1));
            dots = svmla_s32_x(p32, dots, dot, scv);
            mins = svmla_s32_x(p32, mins, sx, mnv);
        }
        acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, dots), x[b].d*d);
        acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, mins), -x[b].d*dm);
    }
    return svaddv_f32(p32, acc);
}

/* Q4_K is especially convenient with the existing Q8 activation blocks: the
 * affine correction only needs the sum of each 32-byte activation slice and
 * the main term maps directly to SVE SDOT. */
static float q4_k_q8_row(const block_q4_K *w,
                         const glm5_iq_q8_block *x, int blocks) {
    const svbool_t p8 = svptrue_b8();
    const svbool_t p32 = svptrue_b32();
    const svint8_t ones = svdup_s8(1);
    const svbool_t p32bytes = svwhilelt_b8(0, 32);
    const svbool_t lo8 = svwhilelt_b32(0, 8);
    svfloat32_t acc = svdup_f32(0.0f);
    for (int b = 0; b < blocks; ++b) {
        const float d = ggml_fp16_to_fp32(w[b].d);
        const float dm = ggml_fp16_to_fp32(w[b].dmin);
        svint32_t dots = svdup_s32(0), mins = svdup_s32(0);
        int is = 0;
        for (int g = 0; g < 256; g += 64, is += 2) {
            uint8_t sc0, min0, sc1, min1;
            get_scale_min_k4(is, w[b].scales, &sc0, &min0);
            get_scale_min_k4(is + 1, w[b].scales, &sc1, &min1);
            const svuint8_t packed = svld1_u8(p32bytes,
                                               w[b].qs + (g >> 1));
            const svint8_t q0 = svreinterpret_s8_u8(
                svand_n_u8_x(p8, packed, 15));
            const svint8_t q1 = svreinterpret_s8_u8(
                svlsr_n_u8_x(p8, packed, 4));
            const svint8_t qv = svsplice_s8(p32bytes, q0, q1);
            const svint8_t xv=svld1_s8(p8,x[b].q+g);
            const svint32_t dot=svdot_s32(svdup_s32(0),qv,xv);
            const svint32_t sx=svdot_s32(svdup_s32(0),ones,xv);
            const svint32_t scv=svsel_s32(lo8,svdup_s32(sc0),svdup_s32(sc1));
            const svint32_t mnv=svsel_s32(lo8,svdup_s32(min0),svdup_s32(min1));
            dots = svmla_s32_x(p32, dots, dot, scv);
            mins = svmla_s32_x(p32, mins, sx, mnv);
        }
        acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, dots), x[b].d*d);
        acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, mins), -x[b].d*dm);
    }
    return svaddv_f32(p32, acc);
}

static float iq_row(int type, const uint8_t *row,
                    const glm5_iq_q8_block *xq, int blocks) {
    if (type == GLM53F_GGML_Q4_K)
        return q4_k_q8_row((const block_q4_K *)row, xq, blocks);
    if (type == GLM53F_GGML_Q5_K)
        return q5_k_q8_row((const block_q5_K *)row, xq, blocks);
    if (type == GLM53F_GGML_Q6_K)
        return q6_k_q8_row((const block_q6_K *)row, xq, blocks);
    glm5_tensor tensor = {0};
    tensor.type = (glm5_qtype)type;
    return glm5_iq_q8_row(&tensor, row, xq, blocks);
}

/* Reuse unpacked Q4/Q5 weights across up to four activation rows. Each
 * accumulator follows precisely the scalar block and lane reduction order. */
static void iq_rows_multi_impl(float *out, int type, const uint8_t *row,
        const glm5_iq_q8_block **x, int tokens, int blocks,const iq_metadata *metadata,
        const int16_t **sums) {
    if (type != GLM53F_GGML_Q4_K && type != GLM53F_GGML_Q5_K) {
        for (int t=0;t<tokens;t++) out[t]=iq_row(type,row,x[t],blocks);
        return;
    }
    svbool_t p8=svptrue_b8(),p32=svptrue_b32();
    svbool_t pbytes=svwhilelt_b8(0,32),lo8=svwhilelt_b32(0,8);
    svint8_t ones=svdup_s8(1);
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
    for (int b=0;b<blocks;b++) {
        const uint8_t *qs,*scales,*qh=NULL;
        float d,dm;
        if (type==GLM53F_GGML_Q4_K) {
            const block_q4_K *w=(const block_q4_K *)row+b;
            qs=w->qs;scales=w->scales;d=metadata?metadata[b].d:ggml_fp16_to_fp32(w->d);dm=metadata?metadata[b].dm:ggml_fp16_to_fp32(w->dmin);
        } else {
            const block_q5_K *w=(const block_q5_K *)row+b;
            qs=w->qs;scales=w->scales;qh=w->qh;d=metadata?metadata[b].d:ggml_fp16_to_fp32(w->d);dm=metadata?metadata[b].dm:ggml_fp16_to_fp32(w->dmin);
        }
        svint32_t d0=svdup_s32(0),d1=svdup_s32(0),d2=svdup_s32(0),d3=svdup_s32(0);
        svint32_t m0=svdup_s32(0),m1=svdup_s32(0),m2=svdup_s32(0),m3=svdup_s32(0);
        for (int g=0;g<256;g+=64) {
            uint8_t sc0,min0,sc1,min1;
            if(metadata){
                sc0=metadata[b].scale[g/32];min0=metadata[b].minimum[g/32];
                sc1=metadata[b].scale[g/32+1];min1=metadata[b].minimum[g/32+1];
            }else{
                get_scale_min_k4(g/32,scales,&sc0,&min0);
                get_scale_min_k4(g/32+1,scales,&sc1,&min1);
            }
            svuint8_t packed=svld1_u8(pbytes,qs+g/2);
            svuint8_t q0=svand_n_u8_x(p8,packed,15),q1=svlsr_n_u8_x(p8,packed,4);
            if (qh) {
                svuint8_t hi=svld1_u8(pbytes,qh);
                q0=svadd_u8_x(p8,q0,svsel_u8(svcmpne_n_u8(pbytes,
                    svand_n_u8_x(p8,hi,1u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                q1=svadd_u8_x(p8,q1,svsel_u8(svcmpne_n_u8(pbytes,
                    svand_n_u8_x(p8,hi,2u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
            }
            svint8_t q=svreinterpret_s8_u8(svsplice_u8(pbytes,q0,q1));
            svint32_t sc=svsel_s32(lo8,svdup_s32(sc0),svdup_s32(sc1));
            svint32_t mn=svsel_s32(lo8,svdup_s32(min0),svdup_s32(min1));
#define MULTI_DOT(T) do { \
    svint8_t v=svld1_s8(p8,x[T][b].q+g); \
    d##T=svmla_s32_x(p32,d##T,svdot_s32(svdup_s32(0),q,v),sc); \
    svint32_t sx=sums?svld1sh_s32(p32,sums[T]+b*64+g/4):svdot_s32(svdup_s32(0),ones,v); \
    m##T=svmla_s32_x(p32,m##T,sx,mn); \
} while(0)
            MULTI_DOT(0);
            if(tokens>1) MULTI_DOT(1);
            if(tokens>2) MULTI_DOT(2);
            if(tokens>3) MULTI_DOT(3);
#undef MULTI_DOT
        }
#define MULTI_ACC(T) do { \
    a##T=svmla_n_f32_x(p32,a##T,svcvt_f32_s32_x(p32,d##T),x[T][b].d*d); \
    a##T=svmla_n_f32_x(p32,a##T,svcvt_f32_s32_x(p32,m##T),-x[T][b].d*dm); \
} while(0)
        MULTI_ACC(0);
        if(tokens>1) MULTI_ACC(1);
        if(tokens>2) MULTI_ACC(2);
        if(tokens>3) MULTI_ACC(3);
#undef MULTI_ACC
    }
    out[0]=svaddv_f32(p32,a0);
    if(tokens>1) out[1]=svaddv_f32(p32,a1);
    if(tokens>2) out[2]=svaddv_f32(p32,a2);
    if(tokens>3) out[3]=svaddv_f32(p32,a3);
}

static void iq_rows_multi(float *out,int type,const uint8_t *row,
        const glm5_iq_q8_block **x,int tokens,int blocks){
    iq_rows_multi_impl(out,type,row,x,tokens,blocks,NULL,NULL);
}
#include "glm53f_iq_tile.h"

static void iq_activation_sums(int16_t *out,const glm5_iq_q8_block *x,int blocks){
    for(int b=0;b<blocks;b++)for(int g=0;g<4;g++){
        svint32_t sum=svdot_s32(svdup_s32(0),svdup_s8(1),svld1_s8(svptrue_b8(),x[b].q+g*64));
        svst1h_s32(svptrue_b32(),out+b*64+g*16,sum);
    }
}

int glm53f_iq_expert_weighted_batch_cached(float *output,
        const glm53f_iq_part *parts, const float *weights, const int *counts,
        int tokens, const float *input, float *gate_up, float *activation,
        float *expert_output,glm53f_iq_cache *cache) {
    enum { H=4096, SLOTS=9, GU=1024, ACT=512 };
    int members[36][4],sizes[36]={0},starts[37]={0},groups=0;
    size_t gate_rb[36],down_rb[36];
    const iq_metadata *gate_meta[36],*down_meta[36];
    glm5_iq_q8_block input_q[4][H/256],act_q[36][ACT/256];
    int16_t input_sums[4][(H/256)*64],act_sums[36][(ACT/256)*64];
    if(!output||!parts||!weights||!counts||!input||!gate_up||!activation||
       !expert_output||tokens<1||tokens>4) return -1;
    int tile_mode=getenv("GLM53F_Q4_ROW_TILE")?atoi(getenv("GLM53F_Q4_ROW_TILE")):0;
    int tile=tile_mode?4:1;
    int precalc=cache&&tile==1&&getenv("GLM53F_Q4_ACT_SUMS")&&atoi(getenv("GLM53F_Q4_ACT_SUMS"));
    pthread_once(&glm5_iq_lut_once,glm5_iq_init_luts);
    for(int t=0;t<tokens;t++) {
        if(counts[t]<0||counts[t]>SLOTS) return -1;
        glm5_iq_quant_q8(input_q[t],input+(size_t)t*H,H);
        if(precalc)iq_activation_sums(input_sums[t],input_q[t],H/256);
        for(int k=0;k<counts[t];k++) {
            int slot=t*SLOTS+k,g;
            const glm53f_iq_part *p=parts+slot;
            if(!glm53f_iq_type_supported(p->gate_type)||!glm53f_iq_type_supported(p->down_type)||
               p->inter<256||p->inter>ACT||p->inter%256) return -1;
            for(g=0;g<groups;g++) {
                const glm53f_iq_part *q=parts+members[g][0];
                if(p->gate_up==q->gate_up&&p->down==q->down&&p->gate_type==q->gate_type&&
                   p->down_type==q->down_type&&p->inter==q->inter&&sizes[g]<4) break;
            }
            if(g==groups) groups++;
            members[g][sizes[g]++]=slot;
        }
    }
    for(int g=0;g<groups;g++) {
        const glm53f_iq_part *p=parts+members[g][0];
        starts[g+1]=starts[g]+2*p->inter;
        gate_rb[g]=dequant_row_size(p->gate_type,H);
        down_rb[g]=dequant_row_size(p->down_type,p->inter);
        gate_meta[g]=tile==1?iq_cache_get(cache,p->gate_up,p->gate_type,2*p->inter,H/256):NULL;
        down_meta[g]=tile==1?iq_cache_get(cache,p->down,p->down_type,H,p->inter/256):NULL;
    }
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for(int row=0;row<starts[groups];row+=tile) {
            int g=0;while(row>=starts[g+1])g++;
            int r=row-starts[g];const glm53f_iq_part *p=parts+members[g][0];
            const glm5_iq_q8_block *x[4];const int16_t *sx[4];float y[4][4];
            for(int j=0;j<sizes[g];j++){x[j]=input_q[members[g][j]/SLOTS];sx[j]=input_sums[members[g][j]/SLOTS];}
            if(tile_mode==2){
                iq_rows_tile2(y,p->gate_type,p->gate_up+(size_t)r*gate_rb[g],gate_rb[g],x,sizes[g],H/256);
                iq_rows_tile2(y+2,p->gate_type,p->gate_up+(size_t)(r+2)*gate_rb[g],gate_rb[g],x,sizes[g],H/256);
            }else if(tile==4)iq_rows_tile4(y,p->gate_type,p->gate_up+(size_t)r*gate_rb[g],gate_rb[g],x,sizes[g],H/256);
            else iq_rows_multi_impl(y[0],p->gate_type,p->gate_up+(size_t)r*gate_rb[g],x,sizes[g],H/256,
                                   gate_meta[g]?gate_meta[g]+(size_t)r*(H/256):NULL,precalc?sx:NULL);
            for(int z=0;z<tile;z++)for(int j=0;j<sizes[g];j++)gate_up[(size_t)members[g][j]*GU+r+z]=y[z][j];
        }
#pragma omp for schedule(static)
        for(int slot=0;slot<tokens*SLOTS;slot++) {
            if(slot%SLOTS>=counts[slot/SLOTS])continue;
            int inter=parts[slot].inter;
            for(int i=0;i<inter;i++) {
                float g=gate_up[(size_t)slot*GU+i],u=gate_up[(size_t)slot*GU+inter+i];
                if(g>10)g=10;if(g< -100)g=-100;
                if(u>10)u=10;if(u< -10)u=-10;
                activation[(size_t)slot*ACT+i]=(g/(1.0f+expf(-g)))*u;
            }
            glm5_iq_quant_q8(act_q[slot],activation+(size_t)slot*ACT,inter);
            if(precalc)iq_activation_sums(act_sums[slot],act_q[slot],inter/256);
        }
#pragma omp for schedule(static)
        for(int row=0;row<groups*H;row+=tile) {
            int g=row/H,r=row%H;const glm53f_iq_part *p=parts+members[g][0];
            const glm5_iq_q8_block *x[4];const int16_t *sx[4];float y[4][4];
            for(int j=0;j<sizes[g];j++){x[j]=act_q[members[g][j]];sx[j]=act_sums[members[g][j]];}
            if(tile_mode==2){
                iq_rows_tile2(y,p->down_type,p->down+(size_t)r*down_rb[g],down_rb[g],x,sizes[g],p->inter/256);
                iq_rows_tile2(y+2,p->down_type,p->down+(size_t)(r+2)*down_rb[g],down_rb[g],x,sizes[g],p->inter/256);
            }else if(tile==4)iq_rows_tile4(y,p->down_type,p->down+(size_t)r*down_rb[g],down_rb[g],x,sizes[g],p->inter/256);
            else iq_rows_multi_impl(y[0],p->down_type,p->down+(size_t)r*down_rb[g],x,sizes[g],p->inter/256,
                                   down_meta[g]?down_meta[g]+(size_t)r*(p->inter/256):NULL,precalc?sx:NULL);
            for(int z=0;z<tile;z++)for(int j=0;j<sizes[g];j++)expert_output[(size_t)members[g][j]*H+r+z]=y[z][j];
        }
#pragma omp for schedule(static)
        for(int row=0;row<tokens*H;row++) {
            int t=row/H,r=row%H;float sum=0;
            for(int k=0;k<counts[t];k++)sum=fmaf(weights[t*SLOTS+k],expert_output[((size_t)t*SLOTS+k)*H+r],sum);
            output[row]=sum;
        }
    }
    return 0;
}

int glm53f_iq_expert_weighted_batch(float *output,const glm53f_iq_part *parts,
        const float *weights,const int *counts,int tokens,const float *input,
        float *gate_up,float *activation,float *expert_output){
    return glm53f_iq_expert_weighted_batch_cached(output,parts,weights,counts,tokens,
                                               input,gate_up,activation,expert_output,NULL);
}

int glm53f_iq_expert_weighted(
        float *output, const glm53f_iq_part *parts, const float *weights,
        int count, const float *input, float *gate_up, float *activation) {
    enum { HIDDEN = 4096, GU_STRIDE = 1024, ACT_STRIDE = 512 };
    glm5_iq_q8_block input_q[HIDDEN / 256];
    glm5_iq_q8_block act_q[9][ACT_STRIDE / 256];
    size_t gate_rb[9], down_rb[9];
    int gate_blocks = HIDDEN / 256;
    if (!output || !parts || !weights || !input || !gate_up || !activation ||
        count < 1 || count > 9) return -1;
    pthread_once(&glm5_iq_lut_once, glm5_iq_init_luts);
    glm5_iq_quant_q8(input_q, input, HIDDEN);
    for (int k = 0; k < count; ++k) {
        if (!glm53f_iq_type_supported(parts[k].gate_type) ||
            !glm53f_iq_type_supported(parts[k].down_type) ||
            parts[k].inter < 256 || parts[k].inter > ACT_STRIDE ||
            parts[k].inter % 256) return -1;
        gate_rb[k] = dequant_row_size((uint32_t)parts[k].gate_type, HIDDEN);
        down_rb[k] = dequant_row_size((uint32_t)parts[k].down_type, parts[k].inter);
    }
#pragma omp parallel
    {
        int total_gate_rows = 0;
        for (int k = 0; k < count; ++k) total_gate_rows += 2 * parts[k].inter;
#pragma omp for schedule(static)
        for (int q = 0; q < total_gate_rows; ++q) {
            int k = 0, r = q;
            while (r >= 2 * parts[k].inter) r -= 2 * parts[k++].inter;
            gate_up[(size_t)k * GU_STRIDE + r] = iq_row(
                parts[k].gate_type, parts[k].gate_up + (size_t)r * gate_rb[k],
                input_q, gate_blocks);
        }
#pragma omp for schedule(static)
        for (int q = 0; q < count * ACT_STRIDE; ++q) {
            int k = q / ACT_STRIDE, i = q - k * ACT_STRIDE;
            if (i < parts[k].inter) {
                float g = gate_up[(size_t)k * GU_STRIDE + i];
                float u = gate_up[(size_t)k * GU_STRIDE + parts[k].inter + i];
                if (g > 10) g = 10; if (g < -100) g = -100;
                if (u > 10) u = 10; if (u < -10) u = -10;
                activation[(size_t)k * ACT_STRIDE + i] =
                    (g / (1.0f + expf(-g))) * u;
            }
        }
#pragma omp single
        for (int k = 0; k < count; ++k)
            glm5_iq_quant_q8(act_q[k], activation + (size_t)k * ACT_STRIDE,
                             parts[k].inter);
#pragma omp for schedule(static)
        for (int r = 0; r < HIDDEN; ++r) {
            float sum = 0.0f;
            for (int k = 0; k < count; ++k)
                sum += weights[k] * iq_row(
                    parts[k].down_type,
                    parts[k].down + (size_t)r * down_rb[k], act_q[k],
                    parts[k].inter / 256);
            output[r] = sum;
        }
    }
    return 0;
}
