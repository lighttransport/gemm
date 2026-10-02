/* 12-way head/tensor-parallel real-weight GLM-5.3F KDA decode layer. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"
#include "../../common/glm53f_arch.h"
#include "glm53f_kda_12n.h"
#include "glm53f_pp_manifest.h"
#include "glm53f_pp_core.h"
#include "glm53f_pf_plan.h"
#include "glm53f_collective_12n.h"
#include <arm_sve.h>
#include <mpi.h>
#include <omp.h>
#include "glm53f_expert_kern.h"
#include "glm53f_int8.h"
#include "glm53f_kda_prefill.h"
#include "glm53f_kda_columns_sve.h"
#include "glm53f_prefill_gemm.h"
#include "glm53f_iq_bridge.h"
#include "glm53f_team.h"
#include "glm53f_clock.h"
#include "glm53f_q80_panel64.h"
#include "glm53f_moe_grouped_native.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { H=4096, NH=64, D=128, QKV=8192, KERNEL=4 };
typedef struct { uint16_t *q,*k,*v,*qc,*kc,*vc,*fa,*fb,*b,*ga,*gb,*on,*op; float *al,*dt; } weights;

static double kd_acc[8]; static long kd_tokens; static int kd_atexit_set;
static double kd_dec[8]; static long kd_dec_calls;
static double kd_sub[8], kd_st; static int kd_sub_on;
#define KS(I) do { if (kd_sub_on) { _Pragma("omp barrier") _Pragma("omp master") { double n_ = glm53f_clock(); kd_sub[I] += n_ - kd_st; kd_st = n_; } } } while (0)
static void kd_report(void){if(kd_dec_calls&&getenv("GLM53F_KDA_DETAIL"))fprintf(stderr,"GLM53F_KDA_DECODE_DETAIL us per layer-call: proj=%.1f conv=%.1f fb+norm+decay=%.1f recurrence=%.1f gb+rmsnorm=%.1f oproj=%.1f allreduce=%.1f (calls=%ld)\n",kd_dec[0]*1e6/kd_dec_calls,kd_dec[1]*1e6/kd_dec_calls,kd_dec[2]*1e6/kd_dec_calls,kd_dec[3]*1e6/kd_dec_calls,kd_dec[4]*1e6/kd_dec_calls,kd_dec[5]*1e6/kd_dec_calls,kd_dec[6]*1e6/kd_dec_calls,kd_dec_calls);if(kd_tokens&&getenv("GLM53F_KDA_BATCH_DETAIL"))fprintf(stderr,"GLM53F_KDA_BATCH_DETAIL us_per_token_per_layer: proj=%.2f conv=%.2f prep=%.2f rec=%.2f norm=%.2f oproj=%.2f allreduce=%.2f (tokens=%ld)\n",kd_acc[0]*1e6/kd_tokens,kd_acc[1]*1e6/kd_tokens,kd_acc[2]*1e6/kd_tokens,kd_acc[3]*1e6/kd_tokens,kd_acc[4]*1e6/kd_tokens,kd_acc[5]*1e6/kd_tokens,kd_acc[6]*1e6/kd_tokens,kd_tokens);if(kd_tokens&&kd_sub[5]>0)fprintf(stderr,"GLM53F_KDA_GEMM_SUB us_per_token_per_layer: quant1=%.2f gemm1=%.2f copy1=%.2f quant2=%.2f gemm2=%.2f copy2=%.2f pre-region=%.2f\n",kd_sub[0]*1e6/kd_tokens,kd_sub[1]*1e6/kd_tokens,kd_sub[2]*1e6/kd_tokens,kd_sub[3]*1e6/kd_tokens,kd_sub[4]*1e6/kd_tokens,kd_sub[5]*1e6/kd_tokens,kd_sub[6]*1e6/kd_tokens);}
static void *a256(size_t n){void*p=NULL;if(posix_memalign(&p,256,n))p=NULL;if(!p)MPI_Abort(MPI_COMM_WORLD,2);return p;}
static inline void kda_l2norm(float *x,int n,float eps){double ss=0.0;for(int i=0;i<n;i++)ss+=(double)x[i]*x[i];float scale=1.0f/fmaxf(sqrtf((float)ss),eps);for(int i=0;i<n;i++)x[i]*=scale;}
static inline void dot8(float*y,const uint16_t*w,const float*x,int n){
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);int vl=(int)svcntw();
    for(int i=0;i<n;i+=vl){svbool_t p=svwhilelt_b32(i,n);svfloat32_t xv=svld1(p,x+i);
#define R(N,A) do{svuint32_t z=svlsl_n_u32_x(p,svld1uh_u32(p,w+(size_t)(N)*n+i),16);A=svmla_x(p,A,svreinterpret_f32_u32(z),xv);}while(0)
        R(0,a0);R(1,a1);R(2,a2);R(3,a3);R(4,a4);R(5,a5);R(6,a6);R(7,a7);
#undef R
    }svbool_t p=svptrue_b32();y[0]=svaddv_f32(p,a0);y[1]=svaddv_f32(p,a1);y[2]=svaddv_f32(p,a2);y[3]=svaddv_f32(p,a3);y[4]=svaddv_f32(p,a4);y[5]=svaddv_f32(p,a5);y[6]=svaddv_f32(p,a6);y[7]=svaddv_f32(p,a7);
}
static inline float dot1(const uint16_t*w,const float*x,int n){svfloat32_t a=svdup_f32(0);int vl=(int)svcntw();for(int i=0;i<n;i+=vl){svbool_t p=svwhilelt_b32(i,n);svuint32_t z=svlsl_n_u32_x(p,svld1uh_u32(p,w+i),16);a=svmla_x(p,a,svreinterpret_f32_u32(z),svld1(p,x+i));}return svaddv_f32(svptrue_b32(),a);}
#ifndef GLM53F_KDA_NO_MAIN
static void mv(float*y,const uint16_t*w,const float*x,int rows,int cols){int nb=rows/8;
#pragma omp parallel for schedule(static)
    for(int b=0;b<nb;b++)dot8(y+b*8,w+(size_t)b*8*cols,x,cols);
    if(rows&7){
#pragma omp parallel for schedule(static)
        for(int r=nb*8;r<rows;r++)y[r]=dot1(w+(size_t)r*cols,x,cols);
    }
}
#endif
/* Orphaned work-sharing variants let one team execute the complete scalar
 * decode graph.  Every loop has its implicit barrier because the following
 * projection consumes the preceding loop's output. Skip remainder worksharing
 * uniformly when rows are a multiple of eight: an empty omp for still pays
 * an implicit team barrier. */
static void mv_team(float*y,const uint16_t*w,const float*x,int rows,int cols){int nb=rows/8;
#pragma omp for schedule(static)
    for(int b=0;b<nb;b++)dot8(y+b*8,w+(size_t)b*8*cols,x,cols);
    if(rows&7){
#pragma omp for schedule(static)
    for(int r=nb*8;r<rows;r++)y[r]=dot1(w+(size_t)r*cols,x,cols);
    }
}
static void mv3_team(float*y0,float*y1,float*y2,const uint16_t*w0,
        const uint16_t*w1,const uint16_t*w2,const float*x,int rows,int cols){int nb=rows/8;
#pragma omp for schedule(static)
    for(int b=0;b<nb;b++){size_t off=(size_t)b*8*cols;dot8(y0+b*8,w0+off,x,cols);dot8(y1+b*8,w1+off,x,cols);dot8(y2+b*8,w2+off,x,cols);}
    if(rows&7){
#pragma omp for schedule(static)
    for(int r=nb*8;r<rows;r++){size_t off=(size_t)r*cols;y0[r]=dot1(w0+off,x,cols);y1[r]=dot1(w1+off,x,cols);y2[r]=dot1(w2+off,x,cols);}
    }
}
static void conv3_team(float*q,float*k,float*v,float*state,const uint16_t*qw,
        const uint16_t*kw,const uint16_t*vw,int channels){
#pragma omp for schedule(static)
    for(int j=0;j<3*channels;j++){int which=j/channels,c=j-which*channels;float*out=which==0?q:(which==1?k:v);const uint16_t*w=which==0?qw:(which==1?kw:vw);float*s=state+(size_t)j*KERNEL,y=0.0f;memmove(s,s+1,(KERNEL-1)*sizeof(*s));s[KERNEL-1]=out[c];for(int z=0;z<KERNEL;z++)y+=s[z]*glm53f_bf16_to_f32(w[(size_t)c*KERNEL+z]);out[c]=y/(1.0f+expf(-y));}
}
static void read_part(glm53f_st_context*st,const char*n,size_t off,void*p,size_t z,int rank){if(glm53f_st_read(st,n,off,p,z)){fprintf(stderr,"rank=%d read %s failed\n",rank,n);MPI_Abort(MPI_COMM_WORLD,2);}}
static void read_cols(glm53f_st_context*st,const char*n,uint16_t*p,int rows,int cols,int c0,int cn,int rank){(void)rows;if(glm53f_st_read_columns(st,n,(size_t)cols*sizeof(uint16_t),(size_t)c0*sizeof(uint16_t),(size_t)cn*sizeof(uint16_t),p)){fprintf(stderr,"rank=%d read columns %s failed\n",rank,n);MPI_Abort(MPI_COMM_WORLD,2);}}
static void name(char*out,int l,const char*s){snprintf(out,256,"model.language_model.layers.%d.self_attn.%s",l,s);}

struct glm53f_kda_context_12n {
    int rank, layer, h0, hn, qd, detail_profile;
    int ranks, image_rank;
    const glm53f_dist *dist;
    const char *native_stage;
    glm53f_prefill_config prefill;
    weights w;
    float *qkv,*small,*gate,*decay,*beta,*core,*normed,*work;
    float *conv,*state,*partial,*batch_partial,*decode_factor;
    float *bq,*bk,*bv,*bsmall_f,*bsmall_g,*bgate_f,*bgate_g,*bbeta,*bnormed;
    float *prefill_state, *prefill_decay, *prefill_core;
    double phase[3], detail[5];
    int8_t *int8_weight[4];
    float *int8_scale[4];
    int int8_enabled;
    uint8_t *q2_q,*q2_k,*q2_v,*q2_op;
    float *q2_normed;
    int q2_q_type,q2_k_type,q2_v_type,q2_op_type,q2_native;
    /* Native V3 stage: auxiliary gate/beta projections and a column-sliced
     * output (q2_op_cols == qd) or the replicated V2 output (== QKV). */
    uint8_t *q2_fa,*q2_fb,*q2_b,*q2_ga,*q2_gb;
    int q2_fa_type,q2_fb_type,q2_b_type,q2_ga_type,q2_gb_type;
    int q2_aux,q2_op_cols;
    float *small_g;
    void *act_x,*act_small,*act_out;
    unsigned char *batch_act;
    /* int8 panel64 GEMM path for the Q8_0 projections (GLM53F_KDA_GEMM: 1 on (default), 0 off, 2 verify vs native) */
    int kg_state, kg_mode, kg_beta;
    float *kg_cw; int kg_conv_state;
    uint8_t *kg_w1, *kg_w2f, *kg_w2g, *kg_w3;
    int8_t *kg_xq, *kg_xp, *kg_q2f, *kg_q2g, *kg_p2f, *kg_p2g, *kg_oq, *kg_op;
    float *kg_xs, *kg_xsp, *kg_bt, *kg_y1, *kg_y2f, *kg_y2g, *kg_y3, *kg_s2f, *kg_s2g, *kg_sp2f, *kg_sp2g, *kg_os, *kg_osp, *kg_cmp;
};

/* Look up one staged tensor, read it, and repack Q8_0 into the SVE layout.
 * Returns 1 when the tensor is absent (optional entries), -1 on error. */
static int kda_native_load_one(const char*blob,const char*manifest,const char*wanted,int rows,int cols,uint8_t**output,int*loaded_type,int require_hash){
    FILE*f=fopen(manifest,"r");char line[512],name[256],type_name[32];uint64_t offset=0;unsigned type=0;int r=0,c=0,found=0;uint64_t expected_hash=0;int have_hash=0;
    if(!f)return-1;
    while(fgets(line,sizeof line,f)) {
        unsigned long long o,bytes,hash;
        if (!found && line[0]!='#' && sscanf(line,"%llu %u %31s %d %d %255s",&o,&type,type_name,&r,&c,name)==6 && !strcmp(name,wanted)) {
            offset=(uint64_t)o;found=1;if(!require_hash)break;
        } else if (found && sscanf(line,"# PAYLOAD offset=%llu bytes=%llu fnv1a=%llx",&o,&bytes,&hash)==3 && o==offset) {
            size_t rb=glm53f_iq_row_size((int)type,cols);
            if(!rb || r!=rows || c!=cols || bytes!=(size_t)rows*rb)break;
            expected_hash=hash;have_hash=1;break;
        }
    }
    fclose(f);
    if(!found)return 1;
    if(require_hash&&!have_hash)return-1;
    if(r!=rows||c!=cols||!glm53f_native_type_supported((int)type))return-1;
    size_t rb=glm53f_iq_row_size((int)type,cols),bytes=(size_t)rows*rb,done=0;int fd=open(blob,O_RDONLY);
    uint8_t*p=a256(bytes);
    if(fd<0||!rb||!p){if(fd>=0)close(fd);free(p);return-1;}
    uint64_t hash=UINT64_C(1469598103934665603);
    while(done<bytes){size_t chunk=bytes-done;if(chunk>(1u<<20))chunk=1u<<20;
        ssize_t n=pread(fd,p+done,chunk,(off_t)(offset+done));if(n<0&&errno==EINTR)continue;
        if(n<=0){close(fd);free(p);return-1;}
        if(require_hash){for(ssize_t i=0;i<n;i++){hash^=p[done+i];hash*=UINT64_C(1099511628211);}
            (void)posix_fadvise(fd,(off_t)(offset+done),n,POSIX_FADV_DONTNEED);}
        done+=(size_t)n;
    }
    if(require_hash&&hash!=expected_hash){close(fd);free(p);return-1;}
    close(fd);
    uint8_t*packed=NULL;int packed_type=(int)type;
    /* TP12 beta has five/six rows and uses the rowwise layout, allowing it
     * to share the padded GEMM front. Keep that arithmetic for sixteen-head
     * PP slices instead of selecting a different beta projection kernel. */
    int rowwise = require_hash && strstr(wanted, ".ssm_beta.weight") != NULL;
    if((rowwise ? glm53f_native_repack_rowwise : glm53f_native_repack)(
            (int)type,p,rows,cols,&packed,&packed_type)){free(p);return-1;}
    if(packed){free(p);p=packed;}
    *output=p;*loaded_type=packed_type;return 0;}
static int kda_native_load(glm53f_kda_context_12n*c){
    const char*stage=c->dist?c->native_stage:getenv("GLM53F_Q2_KDA_STAGE");char blob[PATH_MAX],manifest[PATH_MAX],name[128];int rc;
    if(!stage||!*stage||c->layer<0||c->layer>=45||c->layer%4==3)return 0;
    snprintf(blob,sizeof blob,"%s/rank%02d.blob",stage,c->image_rank);snprintf(manifest,sizeof manifest,"%s/rank%02d.manifest",stage,c->image_rank);
    if(c->dist&&glm53f_pp_manifest_check(manifest,"KDA",c->dist,c->dist->map.first_layer,c->dist->map.end_layer))return-1;
#define LOAD(S,R,C,P,T) (snprintf(name,sizeof name,"blk.%d." S,c->layer),kda_native_load_one(blob,manifest,name,R,C,&c->P,&c->T,c->dist!=NULL))
    /* V2 images stage only layers 0--2: other layers keep the compact path. */
    rc=LOAD("attn_q.weight",c->qd,H,q2_q,q2_q_type);if(rc>0)return c->dist?-1:0;if(rc)return-1;
    if(LOAD("attn_k.weight",c->qd,H,q2_k,q2_k_type)||LOAD("attn_v.weight",c->qd,H,q2_v,q2_v_type))return-1;
    rc=LOAD("attn_output.weight",H,c->qd,q2_op,q2_op_type);
    if(rc==0)c->q2_op_cols=c->qd;
    else if(!c->dist&&rc>0&&!LOAD("attn_output.weight",H,QKV,q2_op,q2_op_type)){c->q2_op_cols=QKV;c->q2_normed=a256((size_t)QKV*sizeof(float));}
    else return-1;
    rc=LOAD("ssm_f_a.weight",D,H,q2_fa,q2_fa_type);
    if(rc==0){if(LOAD("ssm_f_b.weight",c->qd,D,q2_fb,q2_fb_type)||LOAD("ssm_beta.weight",c->hn,H,q2_b,q2_b_type)||
                 LOAD("ssm_g_a.weight",D,H,q2_ga,q2_ga_type)||LOAD("ssm_g_b.weight",c->qd,D,q2_gb,q2_gb_type))return-1;c->q2_aux=1;}
    else if(rc<0||c->dist)return-1;
#undef LOAD
    size_t ab=glm53f_native_act_bytes(QKV);
    c->act_x=a256(ab);c->act_small=a256(ab);c->act_out=a256(ab);c->small_g=a256(D*sizeof(float));
    c->q2_native=1;return 0;}

static int kda_is_q80(int type){return type==GLM53F_GGML_Q8_0||type==GLM53F_NATIVE_Q8_0R||type==GLM53F_NATIVE_Q8_0R16;}

void glm53f_kda_configure_prefill_12n(glm53f_kda_context_12n *c,
                                     const glm53f_prefill_config *config) {
    if (c && config) c->prefill = *config;
}

static void kda_dump(const char *suffix, const void *data, size_t bytes,
                     int rank, int layer) {
    const char *prefix = getenv("GLM53F_KDA_DUMP_PREFIX");
    char path[4096];
    if (!prefix || !*prefix) return;
    snprintf(path, sizeof(path), "%s.rank%02d.layer%02d.%s.bin",
             prefix, rank, layer, suffix);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(data, 1, bytes, f) != bytes || fclose(f))
        MPI_Abort(MPI_COMM_WORLD, 2);
}

int glm53f_kda_convert_int8_12n(glm53f_kda_context_12n *c) {
    if (!c || c->int8_enabled) return -1;
    uint16_t **source[4] = {&c->w.q, &c->w.k, &c->w.v, &c->w.op};
    for (int m = 0; m < 4; ++m) {
        int rows = m == 3 ? H : c->qd, cols = m == 3 ? c->qd : H, failed = 0;
        c->int8_weight[m] = a256((size_t)rows * cols);
        c->int8_scale[m] = a256((size_t)rows * sizeof(float));
#pragma omp parallel for schedule(static) reduction(|:failed)
        for (int r = 0; r < rows; r += 64)
            failed |= glm53f_i8_pack_bf16_tile(c->int8_weight[m] + (size_t)r * cols,
                c->int8_scale[m] + r, *source[m] + (size_t)r * cols, cols) != 0;
        if (failed) return -1;
        free(*source[m]);
        *source[m] = NULL;
    }
    c->int8_enabled = 1;
    return 0;
}

static glm53f_kda_context_12n *kda_create(const glm53f_dist *dist,const char *model,const char *native_stage,int layer){int rank,nr,h0,hn,qd;char n[256];glm53f_st_context*st;glm53f_kda_context_12n*c;
    if(dist){if(!dist->initialized||dist->config.layout!=GLM53F_PP3_TP4||layer<dist->map.first_layer||layer>=dist->map.end_layer||layer%4==3||!native_stage||!*native_stage)return NULL;rank=dist->map.tp_rank;nr=dist->map.tp_size;}
    else{MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(nr!=12)return NULL;}
    if(layer<0)return NULL;
    if(dist){char manifest[PATH_MAX];int n=snprintf(manifest,sizeof(manifest),"%s/rank%02d.manifest",native_stage,dist->map.world_rank);
        if(n<0||n>=(int)sizeof(manifest)||glm53f_pp_manifest_check(manifest,"KDA",dist,dist->map.first_layer,dist->map.end_layer))return NULL;
    }glm53f_balanced_slice(NH,rank,nr,&h0,&hn);qd=hn*D;st=dist?glm53f_pp_core_open(dist,model):glm53f_st_open(model);if(!st)return NULL;c=calloc(1,sizeof(*c));if(!c)MPI_Abort(MPI_COMM_WORLD,2);c->dist=dist;c->native_stage=native_stage;c->ranks=nr;c->image_rank=dist?dist->map.world_rank:rank;c->rank=rank;c->layer=layer;c->h0=h0;c->hn=hn;c->qd=qd;c->detail_profile=getenv("GLM53F_KDA_DETAIL")!=NULL;
#define PART(F,S,T,OFF,N) do{name(n,layer,S);c->w.F=(T*)a256((size_t)(N)*sizeof(T));read_part(st,n,(size_t)(OFF)*sizeof(T),c->w.F,(size_t)(N)*sizeof(T),rank);}while(0)
    PART(al,"A_log",float,h0,hn);PART(dt,"dt_bias",float,h0*D,qd);PART(q,"q_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(k,"k_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(v,"v_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(qc,"q_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(kc,"k_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(vc,"v_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(fb,"f_b_proj.weight",uint16_t,(size_t)h0*D*D,(size_t)qd*D);PART(b,"b_proj.weight",uint16_t,(size_t)h0*H,(size_t)hn*H);PART(gb,"g_b_proj.weight",uint16_t,(size_t)h0*D*D,(size_t)qd*D);PART(fa,"f_a_proj.weight",uint16_t,0,(size_t)D*H);PART(ga,"g_a_proj.weight",uint16_t,0,(size_t)D*H);PART(on,"o_norm.weight",uint16_t,0,D);name(n,layer,"o_proj.weight");c->w.op=a256((size_t)H*qd*sizeof(uint16_t));read_cols(st,n,c->w.op,H,QKV,h0*D,qd,rank);
#undef PART
    glm53f_st_close(st);if(kda_native_load(c)){fprintf(stderr,"rank=%d layer=%d failed to load GLM53F_Q2_KDA_STAGE\n",rank,layer);glm53f_kda_free_12n(c);return NULL;}c->qkv=a256((size_t)3*qd*4);c->small=a256(D*4);c->gate=a256(qd*4);c->decay=a256(qd*4);c->beta=a256(hn*4);c->core=a256(qd*4);c->normed=a256(qd*4);c->work=a256(qd*4);c->conv=a256((size_t)3*qd*KERNEL*4);c->state=a256((size_t)hn*D*D*4);c->partial=a256(H*4);c->batch_partial=a256((size_t)GLM53F_KDA_TILE_TOKENS*H*4);c->bq=a256((size_t)GLM53F_KDA_TILE_TOKENS*qd*4);c->bk=a256((size_t)GLM53F_KDA_TILE_TOKENS*qd*4);c->bv=a256((size_t)GLM53F_KDA_TILE_TOKENS*qd*4);c->bsmall_f=a256((size_t)GLM53F_KDA_TILE_TOKENS*D*4);c->bsmall_g=a256((size_t)GLM53F_KDA_TILE_TOKENS*D*4);c->bgate_f=a256((size_t)GLM53F_KDA_TILE_TOKENS*qd*4);c->bgate_g=a256((size_t)GLM53F_KDA_TILE_TOKENS*qd*4);c->bbeta=a256((size_t)GLM53F_KDA_TILE_TOKENS*hn*4);c->bnormed=a256((size_t)GLM53F_KDA_TILE_TOKENS*qd*4);glm53f_kda_reset_12n(c);return c;}

glm53f_kda_context_12n *glm53f_kda_create_12n(const char *model,int layer){return kda_create(NULL,model,NULL,layer);}
glm53f_kda_context_12n *glm53f_kda_create_dist(const glm53f_dist *dist,const char *model,const char *native_stage,int layer){return dist?kda_create(dist,model,native_stage,layer):NULL;}
static int kda_sum(const glm53f_kda_context_12n *c,const float *input,float *output,int count){
    if(!c->dist)return glm53f_sum_allreduce_12n(input,output,count);
    return MPI_Allreduce(input==output?MPI_IN_PLACE:input,output,count,MPI_FLOAT,MPI_SUM,c->dist->tp)==MPI_SUCCESS?0:-1;
}

void glm53f_kda_reset_12n(glm53f_kda_context_12n*c){if(!c)return;memset(c->conv,0,(size_t)3*c->qd*KERNEL*4);memset(c->state,0,(size_t)c->hn*D*D*4);}
/* Prefetch plan (glm53f_pf_plan.h) for the front projections of this layer's decode step. */
void glm53f_kda_prefetch_plan_12n(const glm53f_kda_context_12n *c) {
    if (!c || !c->q2_native) return;
    const int qd = c->qd, hn = c->hn;
    const glm53f_native_matrix mx[6] = {
        {NULL, c->q2_q, c->q2_q_type, qd, H}, {NULL, c->q2_k, c->q2_k_type, qd, H}, {NULL, c->q2_v, c->q2_v_type, qd, H},
        {NULL, c->q2_fa, c->q2_fa_type, D, H}, {NULL, c->q2_ga, c->q2_ga_type, D, H}, {NULL, c->q2_b, c->q2_b_type, hn, H}};
    glm53f_pf_add_matvec(mx, c->q2_aux ? 6 : 3);
}
/* Issue (non-blocking) L2 prefetches for this thread's static row slice of a native matrix that a LATER stage of the
 * layer will read.  The slice is the one native_matvec_team gives this thread, so the lines land in the CMG that uses
 * them.  Decode is latency bound, so the memory system is idle during the other stages of the layer. */
static inline void kda_prefetch_rows(const glm53f_native_matrix *m) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("GLM53F_KDA_PREFETCH"); on = !e || !*e || atoi(e); }
    if (!on || !m->weight) return;
    const int nt = omp_get_num_threads(), tid = omp_get_thread_num();
    const size_t rb = glm53f_native_row_size(m->type, m->columns);
    if (!rb) return;
    const size_t lo = (size_t)((long long)m->rows * tid / nt) * rb, hi = (size_t)((long long)m->rows * (tid + 1) / nt) * rb;
    for (size_t o = lo; o < hi; o += 256) __builtin_prefetch(m->weight + o, 0, 2);
}
struct kda_call {
    glm53f_kda_context_12n *c;
    const float *x;
    const int8_t *quant_x;
    float scale_x;
    double *td;
    int columns;
};
static void kda_worker(void *context) {
    struct kda_call *a = context;
    glm53f_kda_context_12n *c = a->c;
    weights *w = &c->w;
    int qd = c->qd, hn = c->hn;
    float *q = c->qkv, *k = q + qd, *v = k + qd;
    const float *x = a->x;
    const int8_t *quant_x = a->quant_x;
    const float scale_x = a->scale_x;

        /* detail_profile is uniform across this team. Keep timing singles
         * inside the condition so disabled instrumentation adds no barriers. */
        if (c->q2_native && c->q2_aux) {
            const glm53f_native_matrix pfb = {NULL, c->q2_fb, c->q2_fb_type, qd, D}, pgb = {NULL, c->q2_gb, c->q2_gb_type, qd, D};
            kda_prefetch_rows(&pfb);
            kda_prefetch_rows(&pgb);
        }
        if (c->q2_native) {
            /* One team: one thread quantizes x, then Q/K/V (and the x-input
             * auxiliary projections) share one work-shared row loop.  The
             * former nested glm53f_iq_matvec_2 ran on a single thread. */
            glm53f_native_matrix mx[6] = {
                {q, c->q2_q, c->q2_q_type, qd, H},
                {k, c->q2_k, c->q2_k_type, qd, H},
                {v, c->q2_v, c->q2_v_type, qd, H},
                {c->small, c->q2_fa, c->q2_fa_type, D, H},
                {c->small_g, c->q2_ga, c->q2_ga_type, D, H},
                {c->beta, c->q2_b, c->q2_b_type, hn, H}};
            const int nmx = c->q2_aux ? 6 : 3;
#pragma omp single
            {
                int need_q80 = 0, need_q8k = 0;
                for (int i = 0; i < nmx; ++i) {
                    if (kda_is_q80(mx[i].type)) need_q80 = 1; else need_q8k = 1;
                }
                if (glm53f_native_act_prepare(c->act_x, x, H, need_q8k, need_q80))
                    MPI_Abort(MPI_COMM_WORLD,2);
            }
            if (glm53f_native_matvec_team(mx, nmx, c->act_x))
                MPI_Abort(MPI_COMM_WORLD,2);
        } else if (c->int8_enabled) {
#pragma omp for collapse(2) schedule(static)
            for (int m = 0; m < 3; ++m)
                for (int r = 0; r < qd; r += 64)
                    glm53f_i8_dot64(c->qkv + m * qd + r,
                        c->int8_weight[m] + (size_t)r * H,
                        c->int8_scale[m] + r, quant_x, scale_x, H);
        } else mv3_team(q,k,v,w->q,w->k,w->v,x,qd,H);
#pragma omp single
        {
            kda_dump("q_proj", q, (size_t)qd * sizeof(float), c->rank, c->layer);
            kda_dump("k_proj", k, (size_t)qd * sizeof(float), c->rank, c->layer);
            kda_dump("v_proj", v, (size_t)qd * sizeof(float), c->rank, c->layer);
        }
        if(c->detail_profile){
#pragma omp single
            {double t=glm53f_clock();c->detail[0]=t-(*a->td);(*a->td)=t;}
        }
        conv3_team(q,k,v,c->conv,w->qc,w->kc,w->vc,qd);
        kda_dump("q_conv", q, (size_t)qd * sizeof(float), c->rank, c->layer);
        kda_dump("k_conv", k, (size_t)qd * sizeof(float), c->rank, c->layer);
        kda_dump("v_conv", v, (size_t)qd * sizeof(float), c->rank, c->layer);
        if(c->detail_profile){
#pragma omp single
            {double t=glm53f_clock();c->detail[1]=t-(*a->td);(*a->td)=t;}
        }
        if (c->q2_aux) {
            glm53f_native_matrix mf = {c->gate, c->q2_fb, c->q2_fb_type, qd, D};
#pragma omp single
            if (glm53f_native_act_prepare(c->act_small, c->small, D,
                    !kda_is_q80(c->q2_fb_type), kda_is_q80(c->q2_fb_type)))
                MPI_Abort(MPI_COMM_WORLD,2);
            if (glm53f_native_matvec_team(&mf, 1, c->act_small))
                MPI_Abort(MPI_COMM_WORLD,2);
        } else {
        mv_team(c->small,w->fa,x,D,H);
        mv_team(c->gate,w->fb,c->small,qd,D);
        mv_team(c->beta,w->b,x,hn,H);
        }
#pragma omp for schedule(static)
        for(int h=0;h<hn;h++){
            kda_l2norm(q+(size_t)h*D,D,1e-6f);
            kda_l2norm(k+(size_t)h*D,D,1e-6f);
            glm53f_kda_safe_log_decay(c->decay+(size_t)h*D,c->gate+(size_t)h*D,w->dt+(size_t)h*D,w->al[h],-5.0f,D);
            c->beta[h]=glm53f_sigmoid(c->beta[h]);
            if (a->columns)
                for (int d = 0; d < D; ++d)
                    c->decode_factor[h * D + d] = glm53f_kda_scalar_factor(c->decay[h * D + d]);
        }
        if(c->detail_profile){
#pragma omp single
            {double t=glm53f_clock();c->detail[2]=t-(*a->td);(*a->td)=t;}
        }
        if (c->q2_native && c->q2_op_cols == qd) {
            const glm53f_native_matrix pop = {NULL, c->q2_op, c->q2_op_type, H, qd};
            kda_prefetch_rows(&pop);
        }
        if (a->columns) {
#pragma omp for collapse(2) schedule(static)
            for (int h = 0; h < hn; ++h)
                for (int block = 0; block < 2; ++block)
                    glm53f_kda_columns64_sve(c->state + (size_t)h * D * D + block * 64,
                        c->core + h * D + block * 64, q + h * D, k + h * D,
                        v + h * D + block * 64, c->decode_factor + h * D, c->beta[h]);
        } else {
#pragma omp for schedule(static)
            for(int h=0;h<hn;h++)glm53f_kda_step_vec_streamed(c->state+(size_t)h*D*D,q+(size_t)h*D,k+(size_t)h*D,v+(size_t)h*D,c->decay+(size_t)h*D,c->beta[h],D,D,c->core+(size_t)h*D,c->work+(size_t)h*D);
        }
        if(c->detail_profile){
#pragma omp single
            {double t=glm53f_clock();c->detail[3]=t-(*a->td);(*a->td)=t;}
        }
        if (c->q2_aux) {
            glm53f_native_matrix mg = {c->gate, c->q2_gb, c->q2_gb_type, qd, D};
#pragma omp single
            if (glm53f_native_act_prepare(c->act_small, c->small_g, D,
                    !kda_is_q80(c->q2_gb_type), kda_is_q80(c->q2_gb_type)))
                MPI_Abort(MPI_COMM_WORLD,2);
            if (glm53f_native_matvec_team(&mg, 1, c->act_small))
                MPI_Abort(MPI_COMM_WORLD,2);
        } else {
        mv_team(c->small,w->ga,x,D,H);
        mv_team(c->gate,w->gb,c->small,qd,D);
        }
#pragma omp for schedule(static)
        for(int h=0;h<hn;h++)glm53f_rmsnorm_gated_bf16(c->normed+(size_t)h*D,c->core+(size_t)h*D,c->gate+(size_t)h*D,w->on,1,D,1e-5f);
        if(c->detail_profile){
#pragma omp single
            {c->detail[4]=glm53f_clock()-(*a->td);}
        }

}

static int kda_local(glm53f_kda_context_12n*c,float*out,const float*x){
    weights*w=&c->w;int qd=c->qd,hn=c->hn;double td=c->detail_profile?glm53f_clock():0;
    double t0=glm53f_clock();float*q=c->qkv,*k=q+qd,*v=k+qd;
    int8_t quant_x[H], quant_o[QKV]; float scale_x = 0, scale_o = 0;
    if (c->int8_enabled && glm53f_i8_quantize_x(quant_x, &scale_x, x, H)) return -1;
    const char *column_env = getenv("GLM53F_KDA_DECODE_COLUMNS");
    int columns = column_env && atoi(column_env);
    if (columns && !c->decode_factor) c->decode_factor = a256((size_t)qd * sizeof(float));
    struct kda_call call = {c, x, quant_x, scale_x, &td, columns};
    if (glm53f_team_available()) glm53f_team_dispatch(kda_worker, &call);
    else {
#pragma omp parallel
        { kda_worker(&call); }
    }
    kda_dump("q", q, (size_t)qd * sizeof(float), c->rank, c->layer);
    kda_dump("k", k, (size_t)qd * sizeof(float), c->rank, c->layer);
    kda_dump("v", v, (size_t)qd * sizeof(float), c->rank, c->layer);
    kda_dump("decay", c->decay, (size_t)qd * sizeof(float), c->rank, c->layer);
    kda_dump("beta", c->beta, (size_t)hn * sizeof(float), c->rank, c->layer);
    kda_dump("core", c->core, (size_t)qd * sizeof(float), c->rank, c->layer);
    kda_dump("normed", c->normed, (size_t)qd * sizeof(float), c->rank, c->layer);
    double t1=glm53f_clock();
    if (c->q2_native && c->q2_op_cols == qd) {
        /* Column-sliced output: Q8_0 blocks align to heads, so quantizing the
         * local slice equals llama.cpp's full-vector quantization.  The
         * partial product joins the existing sum all-reduce. */
        glm53f_native_matrix mo = {out, c->q2_op, c->q2_op_type, H, qd};
        int bad = 0;
        if (glm53f_native_act_prepare(c->act_out, c->normed, qd,
                !kda_is_q80(c->q2_op_type), kda_is_q80(c->q2_op_type))) return -1;
        bad = glm53f_native_matvec_prepared_n(&mo, 1, c->act_out);
        if (bad) return -1;
    } else if (c->q2_native) {
        int counts[12],displs[12];
        if(c->dist)return-1;
        for(int r=0;r<12;r++){int a=NH*r/12,b=NH*(r+1)/12;counts[r]=(b-a)*D;displs[r]=a*D;}
        if(MPI_Allgatherv(c->normed,qd,MPI_FLOAT,c->q2_normed,counts,displs,
                         MPI_FLOAT,MPI_COMM_WORLD)!=MPI_SUCCESS)return-1;
        int r0=H*c->rank/12,r1=H*(c->rank+1)/12;
        size_t rb=glm53f_native_row_size(c->q2_op_type,QKV);
        memset(out,0,(size_t)H*sizeof(*out));
        if(!rb||glm53f_iq_matvec(out+r0,c->q2_op+(size_t)r0*rb,
                c->q2_op_type,r1-r0,QKV,c->q2_normed))return-1;
    } else if (c->int8_enabled) {
        if (glm53f_i8_quantize_x(quant_o, &scale_o, c->normed, qd)) return -1;
#pragma omp parallel for schedule(static)
        for (int r = 0; r < H; r += 64)
            glm53f_i8_dot64(out + r, c->int8_weight[3] + (size_t)r * qd,
                c->int8_scale[3] + r, quant_o, scale_o, qd);
    } else {
#pragma omp parallel for schedule(static)
    for(int r=0;r<H;r++)out[r]=dot1(w->op+(size_t)r*qd,c->normed,qd);
    }
    double t2=glm53f_clock();c->phase[0]=t1-t0;c->phase[1]=t2-t1;c->phase[2]=0;return 0;
}
int glm53f_kda_sublayer_12n(void*context,float*out,const float*x){glm53f_kda_context_12n*c=context;if(!c||kda_local(c,c->partial,x))return-1;double t=glm53f_clock();int rc=kda_sum(c,c->partial,out,H);c->phase[2]=glm53f_clock()-t;if(c->detail_profile){for(int i=0;i<5;i++)kd_dec[i]+=c->detail[i];kd_dec[5]+=c->phase[1];kd_dec[6]+=c->phase[2];kd_dec_calls++;if(!kd_atexit_set){kd_atexit_set=1;atexit(kd_report);}}return rc;}
static int kda_batch_legacy(glm53f_kda_context_12n*c,float*out,const float*x,int tokens,void*states,size_t stride){size_t bytes=glm53f_kda_state_bytes_12n(c);if(!c||!out||!x||tokens<1||tokens>5||(states&&stride<bytes))return-1;if(tokens==5){if(glm53f_kda_sublayer_batch_capture_12n(c,out,x,4,states,stride))return-1;if(glm53f_kda_sublayer_12n(c,out+(size_t)4*H,x+(size_t)4*H))return-1;return !states||!glm53f_kda_save_state_12n(c,(unsigned char*)states+(size_t)4*stride,stride)?0:-1;}if(tokens==1){if(kda_local(c,c->batch_partial,x))return-1;if(states&&glm53f_kda_save_state_12n(c,states,stride))return-1;}else{weights*w=&c->w;int qd=c->qd,hn=c->hn;double t0=glm53f_clock();glm53f_mv_bf16_batch(c->bq,w->q,x,tokens,qd,H);glm53f_mv_bf16_batch(c->bk,w->k,x,tokens,qd,H);glm53f_mv_bf16_batch(c->bv,w->v,x,tokens,qd,H);glm53f_mv_bf16_batch(c->bsmall_f,w->fa,x,tokens,D,H);glm53f_mv_bf16_batch(c->bgate_f,w->fb,c->bsmall_f,tokens,qd,D);glm53f_mv_bf16_batch(c->bbeta,w->b,x,tokens,hn,H);glm53f_mv_bf16_batch(c->bsmall_g,w->ga,x,tokens,D,H);glm53f_mv_bf16_batch(c->bgate_g,w->gb,c->bsmall_g,tokens,qd,D);for(int t=0;t<tokens;t++){float*q=c->bq+(size_t)t*qd,*k=c->bk+(size_t)t*qd,*v=c->bv+(size_t)t*qd,*gate=c->bgate_f+(size_t)t*qd,*beta=c->bbeta+(size_t)t*hn;glm53f_causal_conv1d_silu_bf16(q,c->conv,q,w->qc,qd,KERNEL);glm53f_causal_conv1d_silu_bf16(k,c->conv+(size_t)qd*KERNEL,k,w->kc,qd,KERNEL);glm53f_causal_conv1d_silu_bf16(v,c->conv+(size_t)2*qd*KERNEL,v,w->vc,qd,KERNEL);for(int h=0;h<hn;h++){glm53f_l2norm(q+(size_t)h*D,D,1e-6f);glm53f_l2norm(k+(size_t)h*D,D,1e-6f);glm53f_kda_safe_log_decay(c->decay+(size_t)h*D,gate+(size_t)h*D,w->dt+(size_t)h*D,w->al[h],-5.0f,D);beta[h]=glm53f_sigmoid(beta[h]);}
#pragma omp parallel for schedule(static)
            for(int h=0;h<hn;h++)glm53f_kda_step_vec_streamed(c->state+(size_t)h*D*D,q+(size_t)h*D,k+(size_t)h*D,v+(size_t)h*D,c->decay+(size_t)h*D,beta[h],D,D,c->core+(size_t)h*D,c->work+(size_t)h*D);glm53f_rmsnorm_gated_bf16(c->bnormed+(size_t)t*qd,c->core,c->bgate_g+(size_t)t*qd,w->on,hn,D,1e-5f);if(states&&glm53f_kda_save_state_12n(c,(unsigned char*)states+(size_t)t*stride,stride))return-1;}double t1=glm53f_clock();glm53f_mv_bf16_batch(c->batch_partial,w->op,c->bnormed,tokens,H,qd);c->phase[0]=t1-t0;c->phase[1]=glm53f_clock()-t1;}double t=glm53f_clock();int rc=kda_sum(c,c->batch_partial,out,tokens*H);c->phase[2]=glm53f_clock()-t;return rc;}
static void mv_batch_team(float *y, const uint16_t *w, const float *x,
                          int tokens, int rows, int cols) {
    int n4 = rows / 4;
#pragma omp for schedule(static)
    for (int b = 0; b < n4; b++)
        glm53f_matvec_bf16_4x4(y + b * 4, rows,
            w + (size_t)b * 4 * cols, x, tokens, cols);
    if (rows % 4) {
#pragma omp for schedule(static)
        for (int t = 0; t < tokens; t++)
            for (int r = n4 * 4; r < rows; r++)
                y[(size_t)t * rows + r] = glm53f_dot_bf16_sve(
                    w + (size_t)r * cols, x + (size_t)t * cols, cols);
    }
}

static void mv_batch_wide_team(const glm53f_prefill_config *config,
                               float *y, const uint16_t *w, const float *x,
                               int tokens, int rows, int cols) {
    if (tokens > 5 && rows >= 48 && (config->features & GLM53F_PREFILL_GEMM)) {
        glm53f_prefill_gemm_team(y, w, NULL, x, tokens, rows, cols, 0, config->gemm_arena);
        return;
    }
    if (tokens > 5 && getenv("GLM53F_KDA_PREFILL") && atoi(getenv("GLM53F_KDA_PREFILL"))) {
        int n4 = rows / 4;
#pragma omp for collapse(2) schedule(static)
        for (int r = 0; r < n4; ++r)
            for (int base = 0; base < tokens; base += 4) {
                int n = tokens - base;
                if (n > 4) n = 4;
                glm53f_matvec_bf16_4x4(y + (size_t)base * rows + r * 4, rows,
                    w + (size_t)r * 4 * cols, x + (size_t)base * cols, n, cols);
            }
        if (rows % 4) {
#pragma omp for collapse(2) schedule(static)
            for (int t = 0; t < tokens; ++t)
                for (int r = n4 * 4; r < rows; ++r)
                    y[(size_t)t * rows + r] = glm53f_dot_bf16_sve(
                        w + (size_t)r * cols, x + (size_t)t * cols, cols);
        }
        return;
    }
    for (int base = 0; base < tokens; base += 4) {
        int n = tokens - base;
        if (n > 4) n = 4;
        mv_batch_team(y + (size_t)base * rows, w,
                      x + (size_t)base * cols, n, rows, cols);
    }
}

static void native_batch_team(glm53f_kda_context_12n *c,
        const glm53f_native_matrix *m, int count, const float *x, int tokens) {
    const int cols = m[0].columns;
    const size_t stride = glm53f_native_act_bytes(H);
    int need_q80 = 0, need_q8k = 0;
    for (int i = 0; i < count; ++i) {
        if (kda_is_q80(m[i].type)) need_q80 = 1; else need_q8k = 1;
    }
#pragma omp for schedule(static)
    for (int t = 0; t < tokens; ++t)
        if (glm53f_native_act_prepare(c->batch_act + (size_t)t * stride,
                x + (size_t)t * cols, cols, need_q8k, need_q80))
            MPI_Abort(MPI_COMM_WORLD, 2);
    if (glm53f_native_matvec_batch_team(m, count, c->batch_act, stride, tokens))
        MPI_Abort(MPI_COMM_WORLD, 2);
}



/* Causal depthwise conv (kernel 4) + SiLU, vectorized across channels: state and weights are [channel][4] fp32, so one
 * svld4 gives the four taps of 16 channels.  Same per-channel FMA order as the scalar loop; SiLU uses gmn_expf. */
static int kda_conv_setup(glm53f_kda_context_12n *c) {
    if (c->kg_conv_state) return c->kg_conv_state > 0;
    c->kg_conv_state = -1;
    const char *e = getenv("GLM53F_KDA_CONV_VEC");
    if ((e && *e && !atoi(e)) || c->qd % 16 || (int)svcntw() != 16) return 0;
    const int qd = c->qd;
    c->kg_cw = a256((size_t)3 * qd * KERNEL * sizeof(float));
    const uint16_t *cw[3] = {c->w.qc, c->w.kc, c->w.vc};
    for (int which = 0; which < 3; ++which)
        for (int ch = 0; ch < qd; ++ch)
            for (int z = 0; z < KERNEL; ++z)
                c->kg_cw[((size_t)which * qd + ch) * KERNEL + z] = glm53f_bf16_to_f32(cw[which][(size_t)ch * KERNEL + z]);
    c->kg_conv_state = 1;
    return 1;
}

static void kda_conv_vec(glm53f_kda_context_12n *c, int tokens) { /* orphaned worksharing */
    const int qd = c->qd, nb = qd / 16;
    const svbool_t pg = svptrue_b32();
#pragma omp for schedule(static)
    for (int task = 0; task < 3 * nb; ++task) {
        const int which = task / nb, ch0 = (task % nb) * 16, j0 = which * qd + ch0;
        float *proj = which == 0 ? c->bq : which == 1 ? c->bk : c->bv;
        svfloat32x4_t st = svld4_f32(pg, c->conv + (size_t)j0 * KERNEL), wt = svld4_f32(pg, c->kg_cw + (size_t)j0 * KERNEL);
        svfloat32_t s0 = svget4_f32(st, 0), s1 = svget4_f32(st, 1), s2 = svget4_f32(st, 2), s3 = svget4_f32(st, 3);
        const svfloat32_t w0 = svget4_f32(wt, 0), w1 = svget4_f32(wt, 1), w2 = svget4_f32(wt, 2), w3 = svget4_f32(wt, 3);
        for (int t = 0; t < tokens; ++t) {
            float *row = proj + (size_t)t * qd + ch0;
            s0 = s1; s1 = s2; s2 = s3; s3 = svld1_f32(pg, row);
            svfloat32_t y = svmul_f32_x(pg, s0, w0);
            y = svmla_f32_x(pg, y, s1, w1); y = svmla_f32_x(pg, y, s2, w2); y = svmla_f32_x(pg, y, s3, w3);
            svst1_f32(pg, row, svdiv_f32_x(pg, y, svadd_n_f32_x(pg, gmn_expf(pg, svneg_f32_x(pg, y)), 1.0f)));
        }
        st = svset4_f32(st, 0, s0); st = svset4_f32(st, 1, s1); st = svset4_f32(st, 2, s2); st = svset4_f32(st, 3, s3);
        svst4_f32(pg, c->conv + (size_t)j0 * KERNEL, st);
    }
}

/* ---- int8 panel64 GEMM path for the Q8_0 KDA projections (prefill micro-batches of 6..32 tokens) ------------ */
static double kg_worst[2]; static int kg_atexit_set;
static void kg_report(void) { if (getenv("GLM53F_KDA_GEMM") && atoi(getenv("GLM53F_KDA_GEMM")) == 2) fprintf(stderr, "GLM53F_KDA_GEMM_VERIFY worst rel_l2 vs native: front=%.3e oproj=%.3e\n", kg_worst[0], kg_worst[1]); }

static uint8_t *kg_concat(uint8_t *a, size_t ab, uint8_t *b, size_t bb) { /* takes ownership, returns a||b */
    uint8_t *out = NULL;
    if (posix_memalign((void **)&out, 256, ab + bb)) return NULL;
    memcpy(out, a, ab); memcpy(out + ab, b, bb);
    free(a); free(b);
    return out;
}

static int kda_gemm_setup(glm53f_kda_context_12n *c) {
    if (c->kg_state) return c->kg_state > 0;
    c->kg_state = -1;
    const char *e = getenv("GLM53F_KDA_GEMM");
    c->kg_mode = e && *e ? atoi(e) : 1;
    const int qd = c->qd;
    if (!c->kg_mode || !c->q2_native || !c->q2_aux || qd % 64 || c->q2_op_cols != qd) return 0;
    if (!glm53f_q80_family(c->q2_q_type) || !glm53f_q80_family(c->q2_k_type) || !glm53f_q80_family(c->q2_v_type) ||
        !glm53f_q80_family(c->q2_fa_type) || !glm53f_q80_family(c->q2_ga_type) || !glm53f_q80_family(c->q2_fb_type) ||
        !glm53f_q80_family(c->q2_gb_type) || !glm53f_q80_family(c->q2_op_type)) return 0;
    const size_t pb1 = gk_panel64_bytes(32, H), pb2 = gk_panel64_bytes(32, D), pb3 = gk_panel64_bytes(32, qd);
    uint8_t *q = glm53f_q80_to_panel64(c->q2_q_type, c->q2_q, qd, H), *k = glm53f_q80_to_panel64(c->q2_k_type, c->q2_k, qd, H);
    uint8_t *v = glm53f_q80_to_panel64(c->q2_v_type, c->q2_v, qd, H), *fa = glm53f_q80_to_panel64(c->q2_fa_type, c->q2_fa, D, H);
    uint8_t *ga = glm53f_q80_to_panel64(c->q2_ga_type, c->q2_ga, D, H);
    c->kg_w2f = glm53f_q80_to_panel64(c->q2_fb_type, c->q2_fb, qd, D);
    c->kg_w2g = glm53f_q80_to_panel64(c->q2_gb_type, c->q2_gb, qd, D);
    c->kg_w3 = glm53f_q80_to_panel64(c->q2_op_type, c->q2_op, H, qd);
    if (!q || !k || !v || !fa || !ga || !c->kg_w2f || !c->kg_w2g || !c->kg_w3) return 0;
    /* beta (hn rows) rides along as one zero-padded 64-row panel when its storage is row-major Q8_0 / Q8_0R */
    uint8_t *beta = NULL;
    c->kg_beta = 0;
    if (c->q2_b_type == GLM53F_GGML_Q8_0 || c->q2_b_type == GLM53F_NATIVE_Q8_0R) {
        const size_t rb = c->q2_b_type == GLM53F_GGML_Q8_0 ? (size_t)(H / 32) * 34 : (size_t)H + (size_t)(H / 32) * 4;
        uint8_t *pad = (uint8_t *)calloc(64, rb);
        if (pad) { memcpy(pad, c->q2_b, (size_t)c->hn * rb); beta = glm53f_q80_to_panel64(c->q2_b_type, pad, 64, H); free(pad); }
        c->kg_beta = beta != NULL;
    }
    uint8_t *w = kg_concat(q, (size_t)(qd / 64) * pb1, k, (size_t)(qd / 64) * pb1);
    w = w ? kg_concat(w, (size_t)(qd / 64) * pb1 * 2, v, (size_t)(qd / 64) * pb1) : NULL;
    w = w ? kg_concat(w, (size_t)(qd / 64) * pb1 * 3, fa, (size_t)(D / 64) * pb1) : NULL;
    w = w ? kg_concat(w, (size_t)(qd / 64) * pb1 * 3 + (size_t)(D / 64) * pb1, ga, (size_t)(D / 64) * pb1) : NULL;
    if (w && c->kg_beta) w = kg_concat(w, (size_t)(3 * (qd / 64) + 2 * (D / 64)) * pb1, beta, pb1);
    (void)pb2; (void)pb3;
    if (!w) return 0;
    c->kg_w1 = w;
    enum { T = GLM53F_KDA_TILE_TOKENS + 4 };
    const int R1 = 3 * qd + 2 * D + (c->kg_beta ? 64 : 0);
    c->kg_xq = a256((size_t)T * H); c->kg_xp = a256((size_t)T * H); c->kg_xs = a256((size_t)T * (H / 32) * 4);
    c->kg_xsp = a256((size_t)T * (H / 32) * 4); c->kg_bt = a256((size_t)T * (H / 32) * 4);
    c->kg_y1 = a256((size_t)T * R1 * 4);
    c->kg_q2f = a256((size_t)T * D); c->kg_q2g = a256((size_t)T * D); c->kg_p2f = a256((size_t)T * D); c->kg_p2g = a256((size_t)T * D);
    c->kg_s2f = a256((size_t)T * (D / 32) * 4); c->kg_s2g = a256((size_t)T * (D / 32) * 4);
    c->kg_sp2f = a256((size_t)T * (D / 32) * 4); c->kg_sp2g = a256((size_t)T * (D / 32) * 4);
    c->kg_y2f = a256((size_t)T * qd * 4); c->kg_y2g = a256((size_t)T * qd * 4);
    c->kg_oq = a256((size_t)T * qd); c->kg_op = a256((size_t)T * qd); c->kg_os = a256((size_t)T * (qd / 32) * 4); c->kg_osp = a256((size_t)T * (qd / 32) * 4);
    c->kg_y3 = a256((size_t)T * (H + 64) * 4);
    size_t cmpn = (size_t)5 * GLM53F_KDA_TILE_TOKENS * qd; if (cmpn < (size_t)GLM53F_KDA_TILE_TOKENS * H) cmpn = (size_t)GLM53F_KDA_TILE_TOKENS * H;
    c->kg_cmp = a256(cmpn * 4);
    memset(c->kg_xq, 0, (size_t)T * H); memset(c->kg_q2f, 0, (size_t)T * D); memset(c->kg_q2g, 0, (size_t)T * D); memset(c->kg_oq, 0, (size_t)T * qd);
    c->kg_state = 1;
    if (!kg_atexit_set) { kg_atexit_set = 1; atexit(kg_report); }
    return 1;
}

static inline void kg_pack_groups(int8_t *xp, float *xsp, const int8_t *xq, const float *xs, int K, int ng) {
    const int nb = K / 32;
#pragma omp for schedule(static)
    for (int g = 0; g < ng; ++g) {
        const int8_t *rows[6]; const float *xsr[6];
        for (int u = 0; u < 6; ++u) { rows[u] = xq + (size_t)(g * 6 + u) * K; xsr[u] = xs + (size_t)(g * 6 + u) * nb; }
        gmn_pack6(xp + (size_t)g * 6 * K, xsp + (size_t)g * 6 * nb, rows, xsr, K);
    }
}

/* Orphaned worksharing: every thread of the enclosing team calls this. */
static void kda_gemm_front(glm53f_kda_context_12n *c, const float *x, int tokens) {
    const int qd = c->qd, hn = c->hn, mpad = (tokens + 5) / 6 * 6, ng = mpad / 6, nb = H / 32, R1 = 3 * qd + 2 * D + (c->kg_beta ? 64 : 0);
    if (kd_sub_on) {
#pragma omp barrier
#pragma omp master
        kd_st = glm53f_clock();
    }
#pragma omp for schedule(static)
    for (int t = 0; t < mpad; ++t) {
        if (t < tokens) gmn_quant_row(x + (size_t)t * H, H, c->kg_xq + (size_t)t * H, c->kg_xs + (size_t)t * nb, c->kg_bt + (size_t)t * nb);
        else { memset(c->kg_xq + (size_t)t * H, 0, H); memset(c->kg_xs + (size_t)t * nb, 0, nb * 4); }
    }
    kg_pack_groups(c->kg_xp, c->kg_xsp, c->kg_xq, c->kg_xs, H, ng);
    KS(0);
    const int p1 = R1 / 64, nch = (ng + 2) / 3;
#pragma omp for schedule(dynamic, 1)
    for (int task = 0; task < p1 * nch; ++task) {
        const int pnl = task / nch, ch = task % nch, t0 = ch * 18, t1 = (ch + 1) * 18 < mpad ? (ch + 1) * 18 : mpad;
        gk_gemm_panel64(32, c->kg_w1, H, pnl * 64, pnl * 64 + 64, t0, t1, c->kg_xp, c->kg_xsp, c->kg_y1, (size_t)R1);
    }
    KS(1);
#pragma omp for schedule(static)
    for (int t = 0; t < tokens; ++t) {
        const float *y = c->kg_y1 + (size_t)t * R1;
        memcpy(c->bq + (size_t)t * qd, y, qd * 4); memcpy(c->bk + (size_t)t * qd, y + qd, qd * 4);
        memcpy(c->bv + (size_t)t * qd, y + 2 * qd, qd * 4);
        memcpy(c->bsmall_f + (size_t)t * D, y + 3 * qd, D * 4); memcpy(c->bsmall_g + (size_t)t * D, y + 3 * qd + D, D * 4);
        if (c->kg_beta) memcpy(c->bbeta + (size_t)t * hn, y + 3 * qd + 2 * D, hn * 4);
    }
    if (!c->kg_beta) {
        glm53f_native_matrix mb = {c->bbeta, c->q2_b, c->q2_b_type, hn, H};
        native_batch_team(c, &mb, 1, x, tokens);
    }
    KS(2);
    /* gate low-rank up-projections: f_b(bsmall_f), g_b(bsmall_g) */
    const int nd = D / 32;
#pragma omp for schedule(static)
    for (int t = 0; t < mpad; ++t) {
        if (t < tokens) {
            gmn_quant_row(c->bsmall_f + (size_t)t * D, D, c->kg_q2f + (size_t)t * D, c->kg_s2f + (size_t)t * nd, c->kg_bt + (size_t)t * nb);
            gmn_quant_row(c->bsmall_g + (size_t)t * D, D, c->kg_q2g + (size_t)t * D, c->kg_s2g + (size_t)t * nd, c->kg_bt + (size_t)t * nb);
        } else {
            memset(c->kg_q2f + (size_t)t * D, 0, D); memset(c->kg_q2g + (size_t)t * D, 0, D);
            memset(c->kg_s2f + (size_t)t * nd, 0, nd * 4); memset(c->kg_s2g + (size_t)t * nd, 0, nd * 4);
        }
    }
    kg_pack_groups(c->kg_p2f, c->kg_sp2f, c->kg_q2f, c->kg_s2f, D, ng);
    kg_pack_groups(c->kg_p2g, c->kg_sp2g, c->kg_q2g, c->kg_s2g, D, ng);
    KS(3);
    const int p2 = qd / 64;
#pragma omp for schedule(dynamic, 1)
    for (int task = 0; task < 2 * p2; ++task) {
        const int pnl = task % p2;
        if (task < p2) gk_gemm_panel64(32, c->kg_w2f, D, pnl * 64, pnl * 64 + 64, 0, mpad, c->kg_p2f, c->kg_sp2f, c->kg_y2f, (size_t)qd);
        else gk_gemm_panel64(32, c->kg_w2g, D, pnl * 64, pnl * 64 + 64, 0, mpad, c->kg_p2g, c->kg_sp2g, c->kg_y2g, (size_t)qd);
    }
    KS(4);
#pragma omp for schedule(static)
    for (int t = 0; t < tokens; ++t) {
        memcpy(c->bgate_f + (size_t)t * qd, c->kg_y2f + (size_t)t * qd, qd * 4);
        memcpy(c->bgate_g + (size_t)t * qd, c->kg_y2g + (size_t)t * qd, qd * 4);
    }
    KS(5);
}

static void kda_gemm_oproj(glm53f_kda_context_12n *c, int tokens) {
    const int qd = c->qd, mpad = (tokens + 5) / 6 * 6, ng = mpad / 6, nbq = qd / 32;
#pragma omp for schedule(static)
    for (int t = 0; t < mpad; ++t) {
        if (t < tokens) gmn_quant_row(c->bnormed + (size_t)t * qd, qd, c->kg_oq + (size_t)t * qd, c->kg_os + (size_t)t * nbq, c->kg_bt + (size_t)t * (H / 32));
        else { memset(c->kg_oq + (size_t)t * qd, 0, qd); memset(c->kg_os + (size_t)t * nbq, 0, nbq * 4); }
    }
    kg_pack_groups(c->kg_op, c->kg_osp, c->kg_oq, c->kg_os, qd, ng);
    const int nch = (ng + 2) / 3;
#pragma omp for schedule(dynamic, 1)
    for (int task = 0; task < (H / 64) * nch; ++task) {
        const int pnl = task / nch, ch = task % nch, t0 = ch * 18, t1 = (ch + 1) * 18 < mpad ? (ch + 1) * 18 : mpad;
        gk_gemm_panel64(32, c->kg_w3, qd, pnl * 64, pnl * 64 + 64, t0, t1, c->kg_op, c->kg_osp, c->kg_y3, (size_t)(H + 64));
    }
#pragma omp for schedule(static)
    for (int t = 0; t < tokens; ++t) memcpy(c->batch_partial + (size_t)t * H, c->kg_y3 + (size_t)t * (H + 64), H * 4);
}

static int kda_defer_reduce;
void glm53f_kda_set_defer_reduce_12n(int on) { kda_defer_reduce = on; }
/* Build the prefill panel copies / conv tables now (model load time) instead of inside the first prefill call. */
void glm53f_kda_prewarm_12n(glm53f_kda_context_12n *c) {
    if (c && c->q2_native) { (void)kda_gemm_setup(c); (void)kda_conv_setup(c); }
}
int glm53f_kda_sublayer_batch_capture_12n(glm53f_kda_context_12n *c,
        float *out, const float *x, int tokens, void *states, size_t stride) {
    size_t bytes = glm53f_kda_state_bytes_12n(c);
    int wide = getenv("GLM53F_KDA_WIDE_TILE") &&
               atoi(getenv("GLM53F_KDA_WIDE_TILE"));
    if (!c || !out || !x || tokens < 1 || tokens > (wide ? GLM53F_KDA_TILE_TOKENS : 5) ||
        (states && stride < bytes))
        return -1;
    /* Old V2 native stages lack auxiliary projections or use a replicated
     * output. Keep their scalar native path, including verifier snapshots;
     * never silently substitute the compact BF16 matrices in a batch. */
    if (c->int8_enabled || (c->q2_native &&
            (!c->q2_aux || c->q2_op_cols != c->qd ||
             !getenv("GLM53F_KDA_BATCH_TEAM") ||
             !atoi(getenv("GLM53F_KDA_BATCH_TEAM"))))) {
        for (int t = 0; t < tokens; ++t) {
            if (glm53f_kda_sublayer_12n(c, out + (size_t)t * H, x + (size_t)t * H)) return -1;
            if (states && glm53f_kda_save_state_12n(c,
                (unsigned char *)states + (size_t)t * stride, stride)) return -1;
        }
        return 0;
    }
    if (!getenv("GLM53F_KDA_BATCH_TEAM") ||
        !atoi(getenv("GLM53F_KDA_BATCH_TEAM")) || tokens == 1)
        return kda_batch_legacy(c, out, x, tokens, states, stride);
    if (tokens == 5) {
        if (glm53f_kda_sublayer_batch_capture_12n(c, out, x, 4, states, stride) ||
            glm53f_kda_sublayer_12n(c, out + (size_t)4 * H, x + (size_t)4 * H))
            return -1;
        return states ? glm53f_kda_save_state_12n(c,
            (unsigned char *)states + (size_t)4 * stride, stride) : 0;
    }
    weights *w = &c->w;
    int qd = c->qd, hn = c->hn;
    if (c->q2_native && !c->batch_act)
        c->batch_act = a256((size_t)GLM53F_KDA_TILE_TOKENS * glm53f_native_act_bytes(H));
    /* The column-tiled recurrence changes float operation order. Native
     * batches retain decode's recurrence and exact per-position state. */
    const char *native_col_env = getenv("GLM53F_KDA_NATIVE_COLUMN");
    int column_recurrence = (!c->q2_native || !native_col_env || atoi(native_col_env)) && !states && tokens > 5 &&
                           (c->prefill.features & GLM53F_PREFILL_RECURRENCE);
    const char *column_env = getenv("GLM53F_KDA_PREFILL_COLUMNS");
    int columns64 = column_recurrence && column_env && atoi(column_env);
    if (column_recurrence && !c->prefill_state) {
        c->prefill_state = a256((size_t)hn * D * D * sizeof(float));
        c->prefill_decay = a256((size_t)GLM53F_KDA_TILE_TOKENS * qd * sizeof(float));
        c->prefill_core = a256((size_t)GLM53F_KDA_TILE_TOKENS * qd * sizeof(float));
    }
    double start = glm53f_clock(), front_end = start;
    const int kg = c->q2_native && !states && tokens > 5 && tokens <= GLM53F_KDA_TILE_TOKENS && kda_gemm_setup(c);
    const int kconv = !states && tokens > 5 && kda_conv_setup(c);
    const int kd_detail = getenv("GLM53F_KDA_BATCH_DETAIL") != NULL;
    kd_sub_on = kd_detail;
    double kd_t = start; (void)kd_t;
    if (kd_detail) kd_sub[6] += glm53f_clock() - start; /* serial setup before the team starts */
#pragma omp parallel shared(front_end, kd_t)
    {
        if (c->q2_native) {
            glm53f_native_matrix mx[6] = {
                {c->bq,c->q2_q,c->q2_q_type,qd,H},
                {c->bk,c->q2_k,c->q2_k_type,qd,H},
                {c->bv,c->q2_v,c->q2_v_type,qd,H},
                {c->bsmall_f,c->q2_fa,c->q2_fa_type,D,H},
                {c->bsmall_g,c->q2_ga,c->q2_ga_type,D,H},
                {c->bbeta,c->q2_b,c->q2_b_type,hn,H}};
            glm53f_native_matrix mf = {c->bgate_f,c->q2_fb,c->q2_fb_type,qd,D};
            glm53f_native_matrix mg = {c->bgate_g,c->q2_gb,c->q2_gb_type,qd,D};
            if (kg) {
                kda_gemm_front(c, x, tokens);
                if (c->kg_mode == 2) { /* verify: snapshot the GEMM results, recompute natively, compare */
#pragma omp master
                    {
                        float *cm = c->kg_cmp; size_t n = (size_t)tokens * qd;
                        memcpy(cm, c->bq, n * 4); memcpy(cm + n, c->bk, n * 4); memcpy(cm + 2 * n, c->bv, n * 4);
                        memcpy(cm + 3 * n, c->bgate_f, n * 4); memcpy(cm + 4 * n, c->bgate_g, n * 4);
                    }
#pragma omp barrier
                    native_batch_team(c, mx, 6, x, tokens);
                    native_batch_team(c, &mf, 1, c->bsmall_f, tokens);
                    native_batch_team(c, &mg, 1, c->bsmall_g, tokens);
#pragma omp master
                    {
                        float *cm = c->kg_cmp; size_t n = (size_t)tokens * qd;
                        const float *ref[5] = {c->bq, c->bk, c->bv, c->bgate_f, c->bgate_g};
                        double se = 0, sr = 0;
                        for (int a = 0; a < 5; ++a) for (size_t i = 0; i < n; ++i) { double d = cm[a * n + i] - ref[a][i]; se += d * d; sr += (double)ref[a][i] * ref[a][i]; }
                        double rel = sqrt(se / (sr + 1e-30)); if (rel > kg_worst[0]) kg_worst[0] = rel;
                    }
#pragma omp barrier
                }
            } else {
            native_batch_team(c, mx, 6, x, tokens);
            native_batch_team(c, &mf, 1, c->bsmall_f, tokens);
            native_batch_team(c, &mg, 1, c->bsmall_g, tokens);
            }
        } else {
            mv_batch_wide_team(&c->prefill, c->bq, w->q, x, tokens, qd, H);
            mv_batch_wide_team(&c->prefill, c->bk, w->k, x, tokens, qd, H);
            mv_batch_wide_team(&c->prefill, c->bv, w->v, x, tokens, qd, H);
            mv_batch_wide_team(&c->prefill, c->bsmall_f, w->fa, x, tokens, D, H);
            mv_batch_wide_team(&c->prefill, c->bgate_f, w->fb, c->bsmall_f, tokens, qd, D);
            mv_batch_wide_team(&c->prefill, c->bbeta, w->b, x, tokens, hn, H);
            mv_batch_wide_team(&c->prefill, c->bsmall_g, w->ga, x, tokens, D, H);
            mv_batch_wide_team(&c->prefill, c->bgate_g, w->gb, c->bsmall_g, tokens, qd, D);
        }
#define KD_MARK(I) do { if (kd_detail) { _Pragma("omp barrier") _Pragma("omp master") { double n_ = glm53f_clock(); kd_acc[I] += n_ - kd_t; kd_t = n_; } } } while (0)
        KD_MARK(0);
        const int verify_owned = tokens <= 5 && getenv("GLM53F_VERIFY_GROUPED") &&
                                 atoi(getenv("GLM53F_VERIFY_GROUPED"));
        int prefill_recurrence = verify_owned || (!states && tokens > 5 &&
            getenv("GLM53F_KDA_PREFILL") && atoi(getenv("GLM53F_KDA_PREFILL")));
        if (prefill_recurrence) {
            /* Each convolution channel owns its chronological history. All
             * raw projections are available; normalization must not feed back
             * into convolution state. Then each head advances its recurrence
             * across the tile without a barrier between consecutive tokens. */
            if (kconv) kda_conv_vec(c, tokens);
            else
#pragma omp for schedule(static)
            for (int j = 0; j < 3 * qd; ++j) {
                int which = j / qd, channel = j % qd;
                float *projection = which == 0 ? c->bq : which == 1 ? c->bk : c->bv;
                const uint16_t *conv_weight = which == 0 ? w->qc : which == 1 ? w->kc : w->vc;
                float *state = c->conv + (size_t)j * KERNEL;
                for (int t = 0; t < tokens; ++t) {
                    float *value = projection + (size_t)t * qd + channel;
                    memmove(state, state + 1, (KERNEL - 1) * sizeof(float));
                    state[KERNEL - 1] = *value;
                    float y = 0;
                    for (int z = 0; z < KERNEL; ++z)
                        y += state[z] * glm53f_bf16_to_f32(conv_weight[(size_t)channel * KERNEL + z]);
                    *value = y / (1.0f + expf(-y));
                    if (states) {
                        unsigned char *dst = (unsigned char *)states + (size_t)t * stride +
                            (size_t)hn * D * D * sizeof(float) + (size_t)j * KERNEL * sizeof(float);
                        memcpy(dst, state, KERNEL * sizeof(float));
                    }
                }
            }
            KD_MARK(1);
            if (column_recurrence) {
#pragma omp for collapse(2) schedule(static)
                for (int h = 0; h < hn; ++h)
                    for (int t = 0; t < tokens; ++t) {
                        size_t off = (size_t)t * qd + h * D;
                        if (c->q2_native) {
                            kda_l2norm(c->bq + off, D, 1e-6f);
                            kda_l2norm(c->bk + off, D, 1e-6f);
                        } else {
                            glm53f_l2norm(c->bq + off, D, 1e-6f);
                            glm53f_l2norm(c->bk + off, D, 1e-6f);
                        }
                        glm53f_kda_safe_log_decay(c->prefill_decay + off,
                            c->bgate_f + off, w->dt + h * D, w->al[h], -5.0f, D);
                        for (int d = 0; d < D; ++d)
                            c->prefill_decay[off + d] = expf(c->prefill_decay[off + d]);
                        c->bbeta[(size_t)t * hn + h] = glm53f_sigmoid(c->bbeta[(size_t)t * hn + h]);
                    }
                KD_MARK(2);
                if (columns64) {
#pragma omp for collapse(2) schedule(static)
                    for (int h = 0; h < hn; ++h)
                        for (int block = 0; block < 2; ++block)
                            for (int t = 0; t < tokens; ++t) {
                                size_t off = (size_t)t * qd + h * D;
                                glm53f_kda_columns64_sve(c->state + (size_t)h * D * D + block * 64,
                                    c->prefill_core + off + block * 64,
                                    c->bq + off, c->bk + off, c->bv + off + block * 64,
                                    c->prefill_decay + off, c->bbeta[(size_t)t * hn + h]);
                            }
                } else {
#pragma omp for collapse(2) schedule(static)
                    for (int h = 0; h < hn; ++h)
                        for (int block = 0; block < 8; ++block) {
                            float *packed = c->prefill_state + ((size_t)h * 8 + block) * D * 16;
                            float *canonical = c->state + (size_t)h * D * D + block * 16;
                            for (int d = 0; d < D; ++d)
                                memcpy(packed + d * 16, canonical + d * D, 16 * sizeof(float));
                            glm53f_kda_column_tile(packed, c->prefill_core + h * D + block * 16,
                                c->bq + h * D, c->bk + h * D, c->bv + h * D + block * 16,
                                c->prefill_decay + h * D, c->bbeta + h, tokens, qd, hn);
                        }
                    /* Separate row-owned unpack: adjacent column tasks would
                     * false-share canonical state's 256-byte cache lines. */
#pragma omp for collapse(2) schedule(static)
                    for (int h = 0; h < hn; ++h)
                        for (int d = 0; d < D; ++d)
                            for (int block = 0; block < 8; ++block)
                                memcpy(c->state + ((size_t)h * D + d) * D + block * 16,
                                    c->prefill_state + ((size_t)h * 8 + block) * D * 16 + d * 16,
                                    16 * sizeof(float));
                }
                KD_MARK(3);
#pragma omp for collapse(2) schedule(static)
                for (int h = 0; h < hn; ++h)
                    for (int t = 0; t < tokens; ++t) {
                        size_t off = (size_t)t * qd + h * D;
                        glm53f_rmsnorm_gated_bf16(c->bnormed + off, c->prefill_core + off,
                            c->bgate_g + off, w->on, 1, D, 1e-5f);
                    }
                KD_MARK(4);
            } else {
#pragma omp for schedule(static)
            for (int h = 0; h < hn; ++h)
                for (int t = 0; t < tokens; ++t) {
                    float *q = c->bq + (size_t)t * qd + h * D;
                    float *k = c->bk + (size_t)t * qd + h * D;
                    float *v = c->bv + (size_t)t * qd + h * D;
                    float *beta = c->bbeta + (size_t)t * hn + h;
                    if (c->q2_native) {
                        kda_l2norm(q, D, 1e-6f);
                        kda_l2norm(k, D, 1e-6f);
                    } else {
                        glm53f_l2norm(q, D, 1e-6f);
                        glm53f_l2norm(k, D, 1e-6f);
                    }
                    glm53f_kda_safe_log_decay(c->decay + h * D,
                        c->bgate_f + (size_t)t * qd + h * D, w->dt + h * D, w->al[h], -5.0f, D);
                    *beta = glm53f_sigmoid(*beta);
                    glm53f_kda_step_vec_streamed(c->state + (size_t)h * D * D,
                        q, k, v, c->decay + h * D, *beta, D, D, c->core + h * D, c->work + h * D);
                    glm53f_rmsnorm_gated_bf16(c->bnormed + (size_t)t * qd + h * D,
                        c->core + h * D, c->bgate_g + (size_t)t * qd + h * D, w->on, 1, D, 1e-5f);
                    if (states) {
                        const size_t off = (size_t)h * D * D * sizeof(float);
                        memcpy((unsigned char *)states + (size_t)t * stride + off,
                               (unsigned char *)c->state + off, D * D * sizeof(float));
                    }
                }
            }
        } else
        /* Tokens remain causal; parallelize independent channels/heads within
         * each position and retain every snapshot before advancing state. */
        for (int t = 0; t < tokens; t++) {
            float *q = c->bq + (size_t)t * qd, *k = c->bk + (size_t)t * qd;
            float *v = c->bv + (size_t)t * qd, *gate = c->bgate_f + (size_t)t * qd;
            float *beta = c->bbeta + (size_t)t * hn;
            conv3_team(q, k, v, c->conv, w->qc, w->kc, w->vc, qd);
#pragma omp for schedule(static)
            for (int h = 0; h < hn; h++) {
                if (c->q2_native) {
                    kda_l2norm(q + (size_t)h * D, D, 1e-6f);
                    kda_l2norm(k + (size_t)h * D, D, 1e-6f);
                } else {
                    glm53f_l2norm(q + (size_t)h * D, D, 1e-6f);
                    glm53f_l2norm(k + (size_t)h * D, D, 1e-6f);
                }
                glm53f_kda_safe_log_decay(c->decay + (size_t)h * D,
                    gate + (size_t)h * D, w->dt + (size_t)h * D, w->al[h], -5.0f, D);
                beta[h] = glm53f_sigmoid(beta[h]);
                glm53f_kda_step_vec_streamed(c->state + (size_t)h * D * D,
                    q + (size_t)h * D, k + (size_t)h * D, v + (size_t)h * D,
                    c->decay + (size_t)h * D, beta[h], D, D,
                    c->core + (size_t)h * D, c->work + (size_t)h * D);
                glm53f_rmsnorm_gated_bf16(c->bnormed + (size_t)t * qd + h * D,
                    c->core + (size_t)h * D, c->bgate_g + (size_t)t * qd + h * D,
                    w->on, 1, D, 1e-5f);
            }
            if (states) {
                unsigned char *dst = (unsigned char *)states + (size_t)t * stride;
                size_t state_bytes = (size_t)hn * D * D * sizeof(float);
#pragma omp for schedule(static)
                for (size_t offset = 0; offset < bytes; offset += 256) {
                    size_t end = offset < state_bytes ? state_bytes : bytes;
                    size_t n = end - offset < 256 ? end - offset : 256;
                    const unsigned char *src = offset < state_bytes ?
                        (const unsigned char *)c->state + offset :
                        (const unsigned char *)c->conv + offset - state_bytes;
                    memcpy(dst + offset, src, n);
                }
            }
        }
#pragma omp master
        front_end = glm53f_clock();
        if (c->q2_native) {
            glm53f_native_matrix mo = {c->batch_partial,c->q2_op,c->q2_op_type,H,qd};
            if (kg) {
                kda_gemm_oproj(c, tokens);
                if (c->kg_mode == 2) {
#pragma omp master
                    memcpy(c->kg_cmp, c->batch_partial, (size_t)tokens * H * 4);
#pragma omp barrier
                    native_batch_team(c, &mo, 1, c->bnormed, tokens);
#pragma omp master
                    {
                        double se = 0, sr = 0;
                        for (size_t i = 0; i < (size_t)tokens * H; ++i) { double d = c->kg_cmp[i] - c->batch_partial[i]; se += d * d; sr += (double)c->batch_partial[i] * c->batch_partial[i]; }
                        double rel = sqrt(se / (sr + 1e-30)); if (rel > kg_worst[1]) kg_worst[1] = rel;
                    }
#pragma omp barrier
                }
            } else
            native_batch_team(c, &mo, 1, c->bnormed, tokens);
        } else mv_batch_wide_team(&c->prefill, c->batch_partial, w->op, c->bnormed, tokens, H, qd);
        KD_MARK(5);
    }
    double projection_end = glm53f_clock();
    c->phase[0] = front_end - start;
    c->phase[1] = projection_end - front_end;
    int rc = 0;
    int slab = !states && tokens > 5 && (c->prefill.features & GLM53F_PREFILL_COMM) ?
               c->prefill.slab_tokens : 4;
    if (!states && tokens > 5 && kda_defer_reduce) {
        /* The caller reduces the partials of the whole layer chunk with one collective. */
        memcpy(out, c->batch_partial, (size_t)tokens * H * sizeof(float));
    } else if (!c->dist && !states && tokens > 5 && (c->prefill.features & GLM53F_PREFILL_COMM))
        rc = glm53f_sum_allreduce_slabs_12n(c->batch_partial, out, tokens, H, slab);
    else for (int base = 0; base < tokens && !rc; base += slab) {
        int n = tokens - base;
        if (n > slab) n = slab;
        rc = kda_sum(c,c->batch_partial + (size_t)base * H,
                                     out + (size_t)base * H, n * H);
    }
    c->phase[2] = glm53f_clock() - projection_end;
    if (kd_detail && !states) {
        kd_acc[6] += c->phase[2]; kd_tokens += tokens;
        if (!kd_atexit_set) { kd_atexit_set = 1; atexit(kd_report); }
    }
    return rc;
}
int glm53f_kda_sublayer_batch_12n(glm53f_kda_context_12n*c,float*out,const float*x,int tokens){return glm53f_kda_sublayer_batch_capture_12n(c,out,x,tokens,NULL,0);}
void glm53f_kda_last_phase_12n(const glm53f_kda_context_12n*c,double p[3]){memcpy(p,c->phase,sizeof(c->phase));}
void glm53f_kda_last_detail_12n(const glm53f_kda_context_12n*c,double p[5]){memcpy(p,c->detail,sizeof(c->detail));}
int glm53f_kda_head_range_12n(const glm53f_kda_context_12n *c,int *first,int *count){if(!c||!first||!count)return-1;*first=c->h0;*count=c->hn;return 0;}
size_t glm53f_kda_state_bytes_12n(const glm53f_kda_context_12n*c){return c?((size_t)c->hn*D*D+(size_t)3*c->qd*KERNEL)*sizeof(float):0;}
int glm53f_kda_save_state_12n(const glm53f_kda_context_12n*c,void*dst,size_t bytes){size_t sb=c?(size_t)c->hn*D*D*sizeof(float):0,need=glm53f_kda_state_bytes_12n(c);if(!c||!dst||bytes<need)return-1;memcpy(dst,c->state,sb);memcpy((unsigned char*)dst+sb,c->conv,need-sb);return 0;}
int glm53f_kda_restore_state_12n(glm53f_kda_context_12n*c,const void*src,size_t bytes){size_t sb=c?(size_t)c->hn*D*D*sizeof(float):0,need=glm53f_kda_state_bytes_12n(c);if(!c||!src||bytes<need)return-1;memcpy(c->state,src,sb);memcpy(c->conv,(const unsigned char*)src+sb,need-sb);return 0;}
void glm53f_kda_free_12n(glm53f_kda_context_12n*c){if(!c)return;free(c->decode_factor);free(c->kg_cw);free(c->kg_w1);free(c->kg_w2f);free(c->kg_w2g);free(c->kg_w3);free(c->kg_xq);free(c->kg_xp);free(c->kg_q2f);free(c->kg_q2g);free(c->kg_p2f);free(c->kg_p2g);free(c->kg_oq);free(c->kg_op);free(c->kg_xs);free(c->kg_xsp);free(c->kg_bt);free(c->kg_y1);free(c->kg_y2f);free(c->kg_y2g);free(c->kg_y3);free(c->kg_s2f);free(c->kg_s2g);free(c->kg_sp2f);free(c->kg_sp2g);free(c->kg_os);free(c->kg_osp);free(c->kg_cmp);free(c->batch_act);free(c->act_out);free(c->act_small);free(c->act_x);free(c->small_g);free(c->q2_gb);free(c->q2_ga);free(c->q2_b);free(c->q2_fb);free(c->q2_fa);free(c->q2_normed);free(c->q2_op);free(c->q2_v);free(c->q2_k);free(c->q2_q);free(c->prefill_state);free(c->prefill_decay);free(c->prefill_core);for(int m=0;m<4;m++){free(c->int8_weight[m]);free(c->int8_scale[m]);}free(c->bnormed);free(c->bbeta);free(c->bgate_g);free(c->bgate_f);free(c->bsmall_g);free(c->bsmall_f);free(c->bv);free(c->bk);free(c->bq);free(c->batch_partial);free(c->partial);free(c->state);free(c->conv);free(c->work);free(c->normed);free(c->core);free(c->beta);free(c->decay);free(c->gate);free(c->small);free(c->qkv);free(c->w.op);free(c->w.on);free(c->w.ga);free(c->w.fa);free(c->w.gb);free(c->w.b);free(c->w.fb);free(c->w.vc);free(c->w.kc);free(c->w.qc);free(c->w.v);free(c->w.k);free(c->w.q);free(c->w.dt);free(c->w.al);free(c);}

#ifndef GLM53F_KDA_NO_MAIN
int main(int argc,char**argv){
    int rank,nr,layer=argc>2?atoi(argv[2]):44,h0,hn,qd;char n[256];glm53f_st_context*st;weights w={0};
    float *x,*qkv,*small,*gate,*decay,*beta,*core,*normed,*work,*conv,*state,*partial,*out[2];
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);
    if(argc<2||nr!=12){if(!rank)fprintf(stderr,"usage: mpiexec -np 12 %s MODEL_DIR [layer=44]\n",argv[0]);MPI_Finalize();return 2;}
    glm53f_balanced_slice(NH,rank,nr,&h0,&hn);qd=hn*D;st=glm53f_st_open(argv[1]);if(!st)MPI_Abort(MPI_COMM_WORLD,2);
#define PART(F,S,T,OFF,N) do{name(n,layer,S);w.F=(T*)a256((size_t)(N)*sizeof(T));read_part(st,n,(size_t)(OFF)*sizeof(T),w.F,(size_t)(N)*sizeof(T),rank);}while(0)
    PART(al,"A_log",float,h0,hn);PART(dt,"dt_bias",float,h0*D,qd);
    PART(q,"q_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(k,"k_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(v,"v_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);
    PART(qc,"q_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(kc,"k_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(vc,"v_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);
    PART(fb,"f_b_proj.weight",uint16_t,(size_t)h0*D*D,(size_t)qd*D);PART(b,"b_proj.weight",uint16_t,(size_t)h0*H,(size_t)hn*H);PART(gb,"g_b_proj.weight",uint16_t,(size_t)h0*D*D,(size_t)qd*D);
    PART(fa,"f_a_proj.weight",uint16_t,0,(size_t)D*H);PART(ga,"g_a_proj.weight",uint16_t,0,(size_t)D*H);PART(on,"o_norm.weight",uint16_t,0,D);name(n,layer,"o_proj.weight");w.op=a256((size_t)H*qd*sizeof(uint16_t));read_cols(st,n,w.op,H,QKV,h0*D,qd,rank);
#undef PART
    glm53f_st_close(st);
    x=a256(H*4);qkv=a256((size_t)3*qd*4);small=a256(D*4);gate=a256(qd*4);decay=a256(qd*4);beta=a256(hn*4);core=a256(qd*4);normed=a256(qd*4);work=a256(qd*4);conv=a256((size_t)3*qd*KERNEL*4);state=a256((size_t)hn*D*D*4);partial=a256(H*4);out[0]=a256(H*4);out[1]=a256(H*4);memset(conv,0,(size_t)3*qd*KERNEL*4);memset(state,0,(size_t)hn*D*D*4);
    for(int i=0;i<H;i++)x[i]=(float)(((i*17+3)%251)-125)/125.0f;
    float phase[2][3],elapsed[2];
    for(int pass=0;pass<2;pass++){
        double t0=glm53f_clock();float*q=qkv,*k=q+qd,*v=k+qd;mv(q,w.q,x,qd,H);mv(k,w.k,x,qd,H);mv(v,w.v,x,qd,H);glm53f_causal_conv1d_silu_bf16(q,conv,q,w.qc,qd,KERNEL);glm53f_causal_conv1d_silu_bf16(k,conv+(size_t)qd*KERNEL,k,w.kc,qd,KERNEL);glm53f_causal_conv1d_silu_bf16(v,conv+(size_t)2*qd*KERNEL,v,w.vc,qd,KERNEL);mv(small,w.fa,x,D,H);mv(gate,w.fb,small,qd,D);mv(beta,w.b,x,hn,H);
        for(int h=0;h<hn;h++){glm53f_l2norm(q+(size_t)h*D,D,1e-6f);glm53f_l2norm(k+(size_t)h*D,D,1e-6f);glm53f_kda_safe_log_decay(decay+(size_t)h*D,gate+(size_t)h*D,w.dt+(size_t)h*D,w.al[h],-5.0f,D);beta[h]=glm53f_sigmoid(beta[h]);}
#pragma omp parallel for schedule(static)
        for(int h=0;h<hn;h++)glm53f_kda_step_vec_streamed(state+(size_t)h*D*D,q+(size_t)h*D,k+(size_t)h*D,v+(size_t)h*D,decay+(size_t)h*D,beta[h],D,D,core+(size_t)h*D,work+(size_t)h*D);
        mv(small,w.ga,x,D,H);mv(gate,w.gb,small,qd,D);glm53f_rmsnorm_gated_bf16(normed,core,gate,w.on,hn,D,1e-5f);double t1=glm53f_clock();
#pragma omp parallel for schedule(static)
        for(int r=0;r<H;r++)partial[r]=dot1(w.op+(size_t)r*qd,normed,qd);
        double t2=glm53f_clock();MPI_Allreduce(partial,out[pass],H,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);double t3=glm53f_clock();phase[pass][0]=t1-t0;phase[pass][1]=t2-t1;phase[pass][2]=t3-t2;elapsed[pass]=t3-t0;
        for(int i=0;i<H;i++)x[i]=(float)(((i*29+7)%257)-128)/128.0f;
    }
    int stable=1,allstable;
    for(int i=0;i<H;i++)stable&=isfinite(out[0][i])&&isfinite(out[1][i]);
    MPI_Allreduce(&stable,&allstable,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    float maxe,maxp[3],rank_phase[12][3];MPI_Allreduce(&elapsed[1],&maxe,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);MPI_Allreduce(phase[1],maxp,3,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);MPI_Gather(phase[1],3,MPI_FLOAT,rank_phase,3,MPI_FLOAT,0,MPI_COMM_WORLD);
    double ss=0.0,sum=0.0;for(int i=0;i<H;i++){ss+=(double)out[1][i]*out[1][i];sum+=out[1][i];}
    if(!rank)printf("GLM53F_KDA_12N layer=%d heads=64 local_heads=%d..%d max_ms=%.3f local_graph_ms=%.3f oproj_ms=%.3f allreduce_ms=%.3f rms=%.9g sum=%.9g finite=%s %s\n",layer,h0,h0+hn,maxe*1e3f,maxp[0]*1e3f,maxp[1]*1e3f,maxp[2]*1e3f,sqrt(ss/H),sum,allstable?"YES":"NO",allstable?"PASS":"FAIL");
    if(!rank){printf("GLM53F_KDA_12N_RANK_MS");for(int r=0;r<12;r++)printf(" r%d=%.3f/%.3f/%.3f",r,rank_phase[r][0]*1e3f,rank_phase[r][1]*1e3f,rank_phase[r][2]*1e3f);putchar('\n');}
    if(!rank){const char*dump=getenv("GLM53F_KDA_OUTPUT");if(dump&&*dump){FILE*f=fopen(dump,"wb");if(!f||fwrite(out[1],sizeof(float),H,f)!=(size_t)H)MPI_Abort(MPI_COMM_WORLD,2);fclose(f);}}
    MPI_Finalize();return allstable?0:1;
}
#endif
