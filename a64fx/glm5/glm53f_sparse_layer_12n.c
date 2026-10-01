/* Complete decode-front + head-parallel sparse layer integration check. */
#define main glm53f_sparse_core_standalone_main
#include "glm53f_sparse_core_12n.c"
#undef main
#include "glm53f_sparse_12n.h"
#include "glm53f_pf_plan.h"
#include "glm53f_collective_12n.h"
#include "glm53f_index_score.h"
#include "glm53f_cache_bf16.h"
#include "glm53f_int8.h"
#include "glm53f_prefill.h"
#include "glm53f_prefill_gemm.h"
#include "glm53f_mla_prefill.h"
#include "glm53f_moe_grouped_native.h"
#include "glm53f_q80_panel64.h"
#include "glm53f_state_io.h"
#include "glm53f_iq_bridge.h"
#define GLM53F_CP_BF16_LATENT 1
#include <errno.h>
#include <inttypes.h>
#include <limits.h>

enum { QA=1536,IH=32,ID=128,KPOOL=4,TOPK=2048 };
static void b16dot8(float*y,const uint16_t*w,const float*x,int n){svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0),a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);int vl=(int)svcntw();for(int i=0;i<n;i+=vl){svbool_t p=svwhilelt_b32(i,n);svfloat32_t xv=svld1(p,x+i);
#define R(N,A) do{svuint32_t z=svlsl_n_u32_x(p,svld1uh_u32(p,w+(size_t)(N)*n+i),16);A=svmla_x(p,A,svreinterpret_f32_u32(z),xv);}while(0)
R(0,a0);R(1,a1);R(2,a2);R(3,a3);R(4,a4);R(5,a5);R(6,a6);R(7,a7);
#undef R
    }svbool_t p=svptrue_b32();y[0]=svaddv_f32(p,a0);y[1]=svaddv_f32(p,a1);y[2]=svaddv_f32(p,a2);y[3]=svaddv_f32(p,a3);y[4]=svaddv_f32(p,a4);y[5]=svaddv_f32(p,a5);y[6]=svaddv_f32(p,a6);y[7]=svaddv_f32(p,a7);}
static void mv_b16(float*y,const uint16_t*w,const float*x,int rows,int cols){
    if(glm53f_sparse_scalar_reference){
#pragma omp parallel for schedule(static)
        for(int r=0;r<rows;r++)y[r]=glm53f_dot_bf16(w+(size_t)r*cols,x,cols);
        return;
    }
    int nb=rows/8;
#pragma omp parallel for schedule(static)
    for(int b=0;b<nb;b++)b16dot8(y+b*8,w+(size_t)b*8*cols,x,cols);
#pragma omp parallel for schedule(static)
    for(int r=nb*8;r<rows;r++)y[r]=bf16dot(w+(size_t)r*cols,x,cols);
}
static void f8dot8(float*y,const uint8_t*w,const float*s,const float*x,int n){svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0),a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);int vl=(int)svcntw();for(int b=0;b<n;b+=128){int e=b+128<n?b+128:n;for(int i=b;i<e;i+=vl){svbool_t p=svwhilelt_b32(i,e);svfloat32_t xv=svmul_n_f32_x(p,svld1(p,x+i),s[b/128]);
#define R(N,A) A=svmla_x(p,A,glm53f_fp8_e4m3_bits(p,w+(size_t)(N)*n,i),xv)
R(0,a0);R(1,a1);R(2,a2);R(3,a3);R(4,a4);R(5,a5);R(6,a6);R(7,a7);
#undef R
    }}svbool_t p=svptrue_b32();y[0]=svaddv_f32(p,a0);y[1]=svaddv_f32(p,a1);y[2]=svaddv_f32(p,a2);y[3]=svaddv_f32(p,a3);y[4]=svaddv_f32(p,a4);y[5]=svaddv_f32(p,a5);y[6]=svaddv_f32(p,a6);y[7]=svaddv_f32(p,a7);}
static void mv_f8(float*y,const uint8_t*w,const float*s,const float*x,int rows,int cols){int sb=cols/128,nb=rows/8;
    if(glm53f_sparse_scalar_reference){
#pragma omp parallel for schedule(static)
        for(int r=0;r<rows;r++)y[r]=glm53f_dot_fp8_block128(w+(size_t)r*cols,s+(size_t)(r/128)*sb,x,cols);
        return;
    }
#pragma omp parallel for schedule(static)
    for(int b=0;b<nb;b++)f8dot8(y+b*8,w+(size_t)b*8*cols,s+(size_t)(b*8/128)*sb,x,cols);
#pragma omp parallel for schedule(static)
    for(int r=nb*8;r<rows;r++)y[r]=fp8dot(w+(size_t)r*cols,s+(size_t)(r/128)*sb,x,cols);
}
static void *tensor(glm53f_st_context*st,const char*n,int rank){const st_tensor_info*t=glm53f_st_find(st,n,NULL);if(!t){fprintf(stderr,"missing %s\n",n);MPI_Abort(MPI_COMM_WORLD,2);}void*p=a256(t->nbytes);readp(st,n,0,p,t->nbytes,rank);return p;}
typedef struct{float score;int id;} pool_score;
static int pool_cmp(const void*a,const void*b){const pool_score*x=a,*y=b;if(x->score>y->score)return -1;if(x->score<y->score)return 1;return x->id<y->id?-1:x->id>y->id;}
static void pool_heap_down(pool_score*h,int n,int p){for(;;){int w=p,l=2*p+1,r=l+1;if(l<n&&pool_cmp(h+l,h+w)>0)w=l;if(r<n&&pool_cmp(h+r,h+w)>0)w=r;if(w==p)return;pool_score t=h[p];h[p]=h[w];h[w]=t;p=w;}}
static void pool_top_exact(pool_score*h,const float*score,int np,int nc){for(int p=0;p<nc;p++)h[p]=(pool_score){score[p],p};for(int p=nc/2;p-->0;)pool_heap_down(h,nc,p);for(int p=nc;p<np;p++){pool_score x={score[p],p};if(pool_cmp(&x,h)<0){h[0]=x;pool_heap_down(h,nc,0);}}qsort(h,nc,sizeof(*h),pool_cmp);}
#ifndef GLM53F_SPARSE_NO_MAIN
static int select_cached(const float*pool,int*selected,const float*q,const float*hw,
                         int tokens){int np=tokens/KPOOL,nc=TOPK/KPOOL;if(nc>np)nc=np;pool_score*score=a256((size_t)(np?np:1)*sizeof(*score));int out=0;
#pragma omp parallel for schedule(static)
    for(int p=0;p<np;p++){score[p].score=0;score[p].id=p;for(int h=0;h<IH;h++){float z=f32dot(q+(size_t)h*ID,pool+(size_t)p*ID,ID)/sqrtf((float)ID);if(z>0)score[p].score+=hw[h]*z/sqrtf((float)IH);}}qsort(score,np,sizeof(*score),pool_cmp);for(int i=0;i<nc;i++)for(int z=0;z<KPOOL;z++)selected[out++]=score[i].id*KPOOL+z;for(int z=np*KPOOL;z<tokens;z++)selected[out++]=z;free(score);return out;}
#endif

typedef struct{float score;int id;} cp_candidate;

struct glm53f_sparse_context_12n {
    glm53f_prefill_config prefill;
    int rank,ranks,layer,h0,hn,qd,capacity,length,cp,local_capacity,pool_capacity,latent_bf16,prefix_replicated;
    uint8_t *qa,*qb,*kva,*op;
    uint8_t *q2_qa,*q2_qb,*q2_kva,*q2_vb,*q2_op;
    int q2_native,q2_qa_type,q2_op_type,q2_qb_type,q2_kva_type,q2_vb_type;
    float *q8v_ql,*q8v_log,*q8v_va,*q8v_sum;
    float *mlb_qs,*mlb_ql,*mlb_va,*mlb_out,*mlb_ref,*mlb_lg; unsigned char *mlb_act; size_t mlb_act_bytes; int mlb_threads;
    unsigned char *q8v_act;
    uint8_t *sg_w1, *sg_wqb, *sg_wop; int sg_state; /* int8 panel64 GEMM copies of q_a|kv_a and q_b (prefill front) */
    int8_t *sg_xq, *sg_xp, *sg_qq, *sg_qp, *sg_oq, *sg_op; float *sg_xs, *sg_xsp, *sg_bt, *sg_qs, *sg_qsp, *sg_y1, *sg_yq, *sg_os, *sg_osp, *sg_yo;
    unsigned char *front_act_x, *front_act_q; /* fused decode front: native activations of x and of qres */
    size_t q8v_act_bytes;
    float *qas,*qbs,*kvas,*ops;
    uint16_t *qan,*kvan,*kvb,*wk,*knw,*knb,*gatew,*ape,*wqb,*wp;
    float *qres,*query,*latent,*key,*gcache,*iq,*iw,*pool,*attn,*partial,*apef,*packed;
    float *pool_score_local,*pool_score_global;
    float *mla_ql,*mla_log,*mla_part,*mla_va;
    float *cp_latent,*cp_key,*cp_gate,*cp_pool,*cp_pack,*cp_exchange,*cp_gather;
    uint16_t *cp_bf16_local, *cp_bf16_gather;
    uint16_t *hot_latent;
    int hot_prefix;
    float *cp_cur_latent,*cp_cur_key,*cp_cur_gate;
    int *cp_pack_index;
    cp_candidate *cp_candidate_local,*cp_candidate_gather;
    pool_score *pool_score_cache;
    int *selected,*packed_index;
    float *batch_attn, *batch_partial;
    int8_t *int8_weight[4];
    float *int8_scale[4];
    int int8_enabled;
    int profile;
    double profile_phase[GLM53F_SPARSE_PROFILE_PHASES];
};

static int sparse_native_load_one(const char *blob, const char *manifest,
        const char *wanted, int expected_type, int expected_rows,
        int expected_cols, uint8_t **output, int *loaded_type,
        int allow_panel) {
    FILE *m = fopen(manifest, "r");
    char line[512], type_name[32], name[256];
    uint64_t offset;
    unsigned type;
    int rows, cols, found = 0;
    if (!m) {
        fprintf(stderr, "sparse native open manifest failed path=%s error=%s\n",
                manifest, strerror(errno));
        return -1;
    }
    while (fgets(line, sizeof(line), m)) {
        if (line[0] == '#') continue;
        if (sscanf(line, "%" SCNu64 " %u %31s %d %d %255s",
                   &offset, &type, type_name, &rows, &cols, name) == 6 &&
            !strcmp(name, wanted)) {
            found = (expected_type < 0 || type == (unsigned)expected_type) && rows == expected_rows &&
                    cols == expected_cols;
            break;
        }
    }
    fclose(m);
    if (!found) {
        fprintf(stderr, "sparse native manifest mismatch tensor=%s expected_type=%d rows=%d cols=%d path=%s\n",
                wanted, expected_type, expected_rows, expected_cols, manifest);
        return -1;
    }
    size_t bytes = glm53f_native_type_supported((int)type) ?
        (size_t)rows * glm53f_iq_row_size((int)type, cols) : 0;
    if (!bytes) {
        fprintf(stderr, "sparse native unsupported type tensor=%s type=%u\n", wanted, type);
        return -1;
    }
    int fd = open(blob, O_RDONLY);
    uint8_t *p = a256(bytes);
    size_t done = 0;
    if (fd < 0 || !p) {
        fprintf(stderr, "sparse native blob allocation/open failed tensor=%s path=%s bytes=%zu error=%s\n",
                wanted, blob, bytes, strerror(errno));
        if (fd >= 0) close(fd); free(p); return -1;
    }
    while (done < bytes) {
        ssize_t n = pread(fd, p + done, bytes - done, (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            fprintf(stderr, "sparse native blob read failed tensor=%s offset=%" PRIu64 " done=%zu bytes=%zu error=%s\n",
                    wanted, offset, done, bytes, n < 0 ? strerror(errno) : "short file");
            close(fd); free(p); return -1;
        }
        done += (size_t)n;
    }
    close(fd);
    /* Q8_0 rows are repacked into the bit-identical SVE layout. */
    uint8_t *packed = NULL;
    int packed_type = (int)type;
    int repack_error = allow_panel ?
        glm53f_native_repack((int)type, p, rows, cols, &packed, &packed_type) :
        glm53f_native_repack_rowwise((int)type, p, rows, cols, &packed, &packed_type);
    if (repack_error) {
        free(p);
        return -1;
    }
    if (packed) { free(p); p = packed; }
    *output = p;
    if (loaded_type) *loaded_type = packed_type;
    return 0;
}

static int sparse_native_load(glm53f_sparse_context_12n *c) {
    const char *stage = getenv("GLM53F_Q2_SPARSE_STAGE");
    char blob[PATH_MAX], manifest[PATH_MAX], name[256];
    if (!stage || !*stage) return 0;
    if (c->layer >= 45) return 0; /* MTP draft layer: not in the target native image, uses the FP8 checkpoint path */
    snprintf(blob, sizeof(blob), "%s/rank%02d.blob", stage, c->rank);
    snprintf(manifest, sizeof(manifest), "%s/rank%02d.manifest", stage, c->rank);
#define LOAD(S, T, R, C, P, TP) do { \
    snprintf(name, sizeof(name), "blk.%d." S, c->layer); \
    if (sparse_native_load_one(blob, manifest, name, T, R, C, &c->P, TP, 1)) return -1; \
} while (0)
    LOAD("attn_q_a.weight", -1, QA, H, q2_qa, &c->q2_qa_type);
    LOAD("attn_q_b.weight", GLM53F_GGML_Q8_0, c->qd, QA, q2_qb, &c->q2_qb_type);
    LOAD("attn_kv_a_mqa.weight", GLM53F_GGML_Q8_0, LAT, H, q2_kva, &c->q2_kva_type);
    snprintf(name, sizeof(name), "blk.%d.attn_v_b.weight", c->layer);
    if (sparse_native_load_one(blob, manifest, name, GLM53F_GGML_Q8_0,
            c->hn * VD, LAT, &c->q2_vb, &c->q2_vb_type, 0)) return -1;
    LOAD("attn_output.weight", -1, H, c->hn * VD, q2_op, &c->q2_op_type);
#undef LOAD
    c->q2_native = 1;
    return 0;
}

int glm53f_sparse_native_stage_probe_12n(int layer) {
    glm53f_sparse_context_12n c = {0};
    MPI_Comm_rank(MPI_COMM_WORLD, &c.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &c.ranks);
    if (c.ranks != 12 || layer < 3 || layer >= 45 || layer % 4 != 3) return -1;
    c.layer = layer;
    glm53f_balanced_slice(NH, c.rank, c.ranks, &c.h0, &c.hn);
    c.qd = c.hn * KD;
    int rc = sparse_native_load(&c);
    free(c.q2_op); free(c.q2_vb); free(c.q2_kva); free(c.q2_qb); free(c.q2_qa);
    return rc || !c.q2_native ? -1 : 0;
}

static void sparse_dump(const glm53f_sparse_context_12n *c, const char *name,
                        const void *data, size_t bytes) {
    const char *prefix = getenv("GLM53F_SPARSE_DUMP_PREFIX");
    const char *layer_env = getenv("GLM53F_SPARSE_DUMP_LAYER");
    int layer = layer_env && *layer_env ? atoi(layer_env) : -1;
    char path[4096];
    if (!prefix || !*prefix || c->layer != layer) return;
    snprintf(path, sizeof(path), "%s.rank%02d.layer%02d.%s.bin",
             prefix, c->rank, c->layer, name);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(data, 1, bytes, f) != bytes || fclose(f))
        MPI_Abort(MPI_COMM_WORLD, 2);
}

void glm53f_sparse_configure_prefill_12n(glm53f_sparse_context_12n *c,
                                        const glm53f_prefill_config *config) {
    if (c && config) c->prefill = *config;
}

static double sparse_clock(const glm53f_sparse_context_12n *c) {
    return c->profile ? MPI_Wtime() : 0.0;
}
void glm53f_sparse_profile_add_12n(const glm53f_sparse_context_12n *c,
                                  double *seconds) {
    if (c && seconds)
        for (int i = 0; i < GLM53F_SPARSE_PROFILE_PHASES; ++i)
            seconds[i] += c->profile_phase[i];
}
void glm53f_sparse_profile_reset_12n(glm53f_sparse_context_12n *c) {
    if (c) memset(c->profile_phase, 0, sizeof(c->profile_phase));
}

static int sparse_mv_int8(float *out, const int8_t *weight,
        const float *row_scale, const float *x, int rows, int cols) {
    int8_t qx[H];
    float xs;
    if (cols > H || glm53f_i8_quantize_x(qx, &xs, x, cols)) return -1;
#pragma omp parallel for schedule(static)
    for (int q = 0; q < rows / 16; ++q) {
        int group = q / 4, quarter = q % 4;
        glm53f_i8_dot16(out + q * 16,
            weight + (size_t)group * 64 * cols + quarter * 64,
            row_scale + q * 16, qx, xs, cols);
    }
    return 0;
}

int glm53f_sparse_state_io_12n(const glm53f_sparse_context_12n *c, glm53f_state_io *io) {
    if (!c || !io) return -1;
    int meta[] = {c->length, c->cp, c->latent_bf16, c->prefix_replicated,
                  c->hot_prefix};
    if (glm53f_state_io_bytes(io, meta, sizeof(meta), "sparse_meta")) return -1;
    int rows = c->cp ? (c->length + c->ranks - 1 - c->rank) / c->ranks : c->length;
    int pools = c->cp ? (c->length / KPOOL + c->ranks - 1 - c->rank) / c->ranks : c->length / KPOOL;
    if ((c->cp && c->latent_bf16 ? glm53f_state_io_bytes : glm53f_state_io_floats)(io, c->cp ? c->cp_latent : c->latent,
            (size_t)rows * LAT * (c->cp && c->latent_bf16 ? 2 : 4), "sparse_latent") ||
        glm53f_state_io_floats(io, c->cp ? c->cp_key : c->key,
            (size_t)rows * ID * sizeof(float), "sparse_key") ||
        glm53f_state_io_floats(io, c->cp ? c->cp_gate : c->gcache,
            (size_t)rows * ID * sizeof(float), "sparse_gate") ||
        glm53f_state_io_floats(io, c->cp ? c->cp_pool : c->pool,
            (size_t)pools * ID * sizeof(float), "sparse_pool")) return -1;
    if (c->cp && c->prefix_replicated && c->local_capacity >= 1024) {
        int prefix = c->length < 512 ? c->length : 512;
        size_t element = c->latent_bf16 ? 2 : 4;
        if ((c->latent_bf16 ? glm53f_state_io_bytes : glm53f_state_io_floats)(io, (const unsigned char *)c->cp_latent +
                (size_t)(c->local_capacity - 512) * LAT * element,
                (size_t)prefix * LAT * element, "sparse_prefix_latent") ||
            glm53f_state_io_floats(io, c->cp_pool + (size_t)(c->pool_capacity - 128) * ID,
                (size_t)(prefix / KPOOL) * ID * sizeof(float), "sparse_prefix_pool")) return -1;
    }
    if (c->cp && c->hot_prefix && glm53f_state_io_bytes(io, c->hot_latent,
            (size_t)(c->length < c->hot_prefix ? c->length : c->hot_prefix) * LAT * sizeof(uint16_t),
            "sparse_hot_latent")) return -1;
    int n = c->length / KPOOL;
    if (n > TOPK / KPOOL) n = TOPK / KPOOL;
    n = n * KPOOL + c->length % KPOOL;
    return glm53f_state_io_indices(io, c->selected, (size_t)n, "sparse_selected");
}

int glm53f_sparse_convert_int8_12n(glm53f_sparse_context_12n *c) {
    if (!c || c->int8_enabled) return -1;
    uint8_t **source[4] = {&c->qa, &c->qb, &c->kva, &c->op};
    float **source_scale[4] = {&c->qas, &c->qbs, &c->kvas, &c->ops};
    int rows[4] = {QA, c->qd, LAT, H};
    int cols[4] = {H, QA, H, c->hn * VD};
    for (int m = 0; m < 4; ++m) {
        int failed = 0;
        c->int8_weight[m] = a256((size_t)rows[m] * cols[m]);
        c->int8_scale[m] = a256((size_t)rows[m] * sizeof(float));
#pragma omp parallel reduction(|:failed)
        {
            int8_t *scratch = malloc((size_t)64 * cols[m]);
            if (!scratch) failed = 1;
#pragma omp for schedule(static)
            for (int r = 0; r < rows[m]; r += 64)
                if (scratch) {
                    memcpy(c->int8_weight[m] + (size_t)r * cols[m],
                           *source[m] + (size_t)r * cols[m],
                           (size_t)64 * cols[m]);
                    failed |= glm53f_i8_pack_fp8_tile(
                        (uint8_t *)c->int8_weight[m] + (size_t)r * cols[m],
                        c->int8_scale[m] + r, *source_scale[m], r, cols[m],
                        scratch) != 0;
                }
            free(scratch);
        }
        if (failed) return -1;
        free(*source[m]); *source[m] = NULL;
        free(*source_scale[m]); *source_scale[m] = NULL;
    }
    c->int8_enabled = 1;
    return 0;
}

int glm53f_sparse_set_hot_prefix_12n(glm53f_sparse_context_12n *c, int hot_prefix) {
    if (!c || hot_prefix < 0 || hot_prefix > c->capacity || c->length ||
        !c->cp || !c->latent_bf16) return hot_prefix == 0 ? 0 : -1;
    if (hot_prefix == c->hot_prefix) return 0;
    free(c->hot_latent);
    c->hot_latent = NULL;
    c->hot_prefix = hot_prefix;
    if (!hot_prefix) return 0;
    c->hot_latent = a256((size_t)hot_prefix * LAT * sizeof(uint16_t));
    if (!c->hot_latent) { c->hot_prefix = 0; return -1; }
    return 0;
}

glm53f_sparse_context_12n*glm53f_sparse_create_format_12n(const char*model,int layer,int capacity,int latent_bf16){int rank,nr,h0,hn,qd;char n[256];glm53f_st_context*st;glm53f_sparse_context_12n*c;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(nr!=12||capacity<1)return NULL;glm53f_balanced_slice(NH,rank,nr,&h0,&hn);qd=hn*KD;st=glm53f_st_open(model);if(!st)return NULL;c=calloc(1,sizeof(*c));if(!c)MPI_Abort(MPI_COMM_WORLD,2);c->rank=rank;c->ranks=nr;c->layer=layer;c->h0=h0;c->hn=hn;c->qd=qd;c->capacity=capacity;c->cp=getenv("GLM53F_SPARSE_CP")?atoi(getenv("GLM53F_SPARSE_CP"))!=0:capacity>=65536;c->latent_bf16=latent_bf16!=0;c->prefix_replicated=getenv("GLM53F_CP_PREFIX_REPLICATED")&&atoi(getenv("GLM53F_CP_PREFIX_REPLICATED"));if(c->latent_bf16)c->cp=1;c->local_capacity=(capacity+nr-1)/nr;c->pool_capacity=(capacity/KPOOL+nr-1)/nr;
#define N(S) snprintf(n,sizeof n,"model.language_model.layers.%d.self_attn.%s",layer,S)
    N("q_a_proj.weight");c->qa=tensor(st,n,rank);N("q_a_proj.weight_scale_inv");c->qas=tensor(st,n,rank);N("q_a_layernorm.weight");c->qan=tensor(st,n,rank);N("q_b_proj.weight");c->qb=a256((size_t)qd*QA);readp(st,n,(size_t)h0*KD*QA,c->qb,(size_t)qd*QA,rank);N("q_b_proj.weight_scale_inv");{int nb=QA/128,tiles=qd/128;c->qbs=a256((size_t)tiles*nb*4);readp(st,n,(size_t)(h0*KD/128)*nb*4,c->qbs,(size_t)tiles*nb*4,rank);}N("kv_a_proj_with_mqa.weight");c->kva=tensor(st,n,rank);N("kv_a_proj_with_mqa.weight_scale_inv");c->kvas=tensor(st,n,rank);N("kv_a_layernorm.weight");c->kvan=tensor(st,n,rank);N("kv_b_proj.weight");c->kvb=a256((size_t)hn*(KD+VD)*LAT*2);readp(st,n,(size_t)h0*(KD+VD)*LAT*2,c->kvb,(size_t)hn*(KD+VD)*LAT*2,rank);{int oc0=h0*VD,ocn=hn*VD,ob=QKV/128,ob0=oc0/128,obn=ocn/128;N("o_proj.weight");c->op=a256((size_t)H*ocn);if(glm53f_st_read_columns(st,n,QKV,oc0,ocn,c->op))MPI_Abort(MPI_COMM_WORLD,2);N("o_proj.weight_scale_inv");c->ops=a256((size_t)(H/128)*obn*sizeof(float));if(glm53f_st_read_columns(st,n,(size_t)ob*sizeof(float),(size_t)ob0*sizeof(float),(size_t)obn*sizeof(float),c->ops))MPI_Abort(MPI_COMM_WORLD,2);}N("indexer.wk.weight");c->wk=tensor(st,n,rank);N("indexer.k_norm.weight");c->knw=tensor(st,n,rank);N("indexer.k_norm.bias");c->knb=tensor(st,n,rank);N("indexer.index_kpool_compress_gate");c->gatew=tensor(st,n,rank);N("indexer.index_kpool_compress_ape");c->ape=tensor(st,n,rank);N("indexer.wq_b.weight");c->wqb=tensor(st,n,rank);N("indexer.weights_proj.weight");c->wp=tensor(st,n,rank);
#undef N
    c->profile = getenv("GLM53F_PROFILE") != NULL;
    if (sparse_native_load(c)) {
        fprintf(stderr, "rank=%d layer=%d failed to load GLM53F_Q2_SPARSE_STAGE\n",
                rank, layer);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    glm53f_st_close(st);c->qres=a256(QA*4);c->query=a256((size_t)qd*4);if(c->cp){c->cp_latent=a256((size_t)c->local_capacity*LAT*(c->latent_bf16?2:4));c->cp_key=a256((size_t)c->local_capacity*ID*4);c->cp_gate=a256((size_t)c->local_capacity*ID*4);c->cp_pool=a256((size_t)(c->pool_capacity+1)*ID*4);c->cp_pack=a256((size_t)(TOPK+KPOOL)*LAT*4);c->cp_exchange=a256((size_t)2*KPOOL*ID*4);c->cp_cur_latent=a256(LAT*4);c->cp_cur_key=a256(ID*4);c->cp_cur_gate=a256(ID*4);c->cp_pack_index=a256((size_t)(TOPK+KPOOL)*sizeof(int));int nc=c->pool_capacity+1>TOPK/KPOOL?c->pool_capacity+1:TOPK/KPOOL;c->cp_candidate_local=a256((size_t)nc*sizeof(cp_candidate));c->cp_candidate_gather=a256((size_t)nr*(TOPK/KPOOL)*sizeof(cp_candidate));}else{c->latent=a256((size_t)capacity*LAT*4);c->key=a256((size_t)capacity*ID*4);c->gcache=a256((size_t)capacity*ID*4);c->pool=a256((size_t)(capacity/KPOOL+1)*ID*4);c->pool_score_cache=a256((size_t)(capacity/KPOOL+1)*sizeof(pool_score));c->pool_score_local=a256((size_t)(capacity/KPOOL+1)*sizeof(float));c->pool_score_global=a256((size_t)(capacity/KPOOL+1)*sizeof(float));c->packed=a256((size_t)(TOPK+KPOOL)*LAT*4);c->packed_index=a256((size_t)(TOPK+KPOOL)*sizeof(int));for(int i=0;i<TOPK+KPOOL;i++)c->packed_index[i]=i;}c->iq=a256((size_t)IH*ID*4);c->iw=a256(IH*4);c->attn=a256((size_t)hn*VD*4);c->partial=a256(H*4);c->selected=a256((size_t)(TOPK+KPOOL)*sizeof(int));c->apef=a256(KPOOL*ID*4);for(int i=0;i<KPOOL*ID;i++)c->apef[i]=glm53f_bf16_to_f32(c->ape[i]);if(getenv("GLM53F_TOUCH_CACHE")){if(c->cp){memset(c->cp_latent,0,(size_t)c->local_capacity*LAT*(c->latent_bf16?2:4));memset(c->cp_key,0,(size_t)c->local_capacity*ID*4);memset(c->cp_gate,0,(size_t)c->local_capacity*ID*4);memset(c->cp_pool,0,(size_t)(c->pool_capacity+1)*ID*4);}else{memset(c->latent,0,(size_t)capacity*LAT*4);memset(c->key,0,(size_t)capacity*ID*4);memset(c->gcache,0,(size_t)capacity*ID*4);memset(c->pool,0,(size_t)(capacity/KPOOL+1)*ID*4);}}return c;}
glm53f_sparse_context_12n *glm53f_sparse_create_12n(const char *model, int layer, int capacity) {
    return glm53f_sparse_create_format_12n(model, layer, capacity, 0);
}
static void cp_store_latent(glm53f_sparse_context_12n *c, int slot, const float *source) {
    if (c->latent_bf16) {
        uint16_t *out = (uint16_t *)c->cp_latent + (size_t)slot * LAT;
        for (int d = 0; d < LAT; ++d) out[d] = glm53f_cache_bf16_round(source[d]);
    } else memcpy(c->cp_latent + (size_t)slot * LAT, source, LAT * sizeof(float));
}
static void cp_read_latent(float *out, const glm53f_sparse_context_12n *c, int slot) {
    if (c->latent_bf16) {
        const uint16_t *in = (const uint16_t *)c->cp_latent + (size_t)slot * LAT;
        for (int d = 0; d < LAT; d += (int)svcntw()) {
            svbool_t p = svwhilelt_b32(d, LAT);
            svst1(p, out + d, svreinterpret_f32_u32(svlsl_n_u32_x(p, svld1uh_u32(p, in + d), 16)));
        }
    } else memcpy(out, c->cp_latent + (size_t)slot * LAT, LAT * sizeof(float));
}
static void hot_store_latent(glm53f_sparse_context_12n *c, int pos,
                             const float *source) {
    uint16_t *out = c->hot_latent + (size_t)pos * LAT;
    for (int d = 0; d < LAT; ++d) out[d] = glm53f_cache_bf16_round(source[d]);
}
static void hot_read_latent(float *out, const glm53f_sparse_context_12n *c,
                            int pos) {
    const uint16_t *in = c->hot_latent + (size_t)pos * LAT;
    for (int d = 0; d < LAT; d += (int)svcntw()) {
        svbool_t p = svwhilelt_b32(d, LAT);
        svst1(p, out + d, svreinterpret_f32_u32(
            svlsl_n_u32_x(p, svld1uh_u32(p, in + d), 16)));
    }
}
static void read_bf16_row(float *out, const uint16_t *in) {
    for (int d = 0; d < LAT; d += (int)svcntw()) {
        svbool_t p = svwhilelt_b32(d, LAT);
        svst1(p, out + d, svreinterpret_f32_u32(
            svlsl_n_u32_x(p, svld1uh_u32(p, in + d), 16)));
    }
}
int glm53f_sparse_touch_cache_12n(glm53f_sparse_context_12n *c) {
    if (!c || c->length) return -1;
    if (c->cp) {
        memset(c->cp_latent, 0, (size_t)c->local_capacity * LAT * (c->latent_bf16 ? 2 : 4));
        memset(c->cp_key, 0, (size_t)c->local_capacity * ID * 4);
        memset(c->cp_gate, 0, (size_t)c->local_capacity * ID * 4);
        memset(c->cp_pool, 0, (size_t)(c->pool_capacity + 1) * ID * 4);
        if (c->hot_latent)
            memset(c->hot_latent, 0, (size_t)c->hot_prefix * LAT * sizeof(uint16_t));
    } else {
        memset(c->latent, 0, (size_t)c->capacity * LAT * 4);
        memset(c->key, 0, (size_t)c->capacity * ID * 4);
        memset(c->gcache, 0, (size_t)c->capacity * ID * 4);
        memset(c->pool, 0, (size_t)(c->capacity / KPOOL + 1) * ID * 4);
    }
    return 0;
}
void glm53f_sparse_reset_12n(glm53f_sparse_context_12n*c){if(c)c->length=0;}
int glm53f_sparse_length_12n(const glm53f_sparse_context_12n*c){return c?c->length:-1;}
int glm53f_sparse_restore_length_12n(glm53f_sparse_context_12n*c,int length){if(!c||length<0||length>c->length)return-1;c->length=length;return 0;}
int glm53f_sparse_is_context_parallel_12n(const glm53f_sparse_context_12n*c){return c?c->cp:0;}
size_t glm53f_sparse_cache_bytes_12n(const glm53f_sparse_context_12n *c) {
    if (!c) return 0;
    if (!c->cp) return ((size_t)c->capacity * (LAT + 2 * ID) + (size_t)(c->capacity / KPOOL + 1) * ID) * 4;
    size_t n = (size_t)c->local_capacity * LAT * (c->latent_bf16 ? 2 : 4);
    n += ((size_t)c->local_capacity * 2 * ID + (size_t)(c->pool_capacity + 1) * ID + (size_t)(TOPK + KPOOL) * LAT + 2 * KPOOL * ID) * 4;
    if (c->cp_gather) n += (size_t)(TOPK + KPOOL) * LAT * 4;
    if (c->cp_bf16_local)
        n += (size_t)2 * (TOPK + KPOOL) * LAT * sizeof(uint16_t);
    if (c->mla_ql) n += (size_t)c->hn * (2 * LAT + TOPK + KPOOL) * 4;
    n += (size_t)c->hot_prefix * LAT * sizeof(uint16_t);
    return n;
}
static void update_completed_pool(glm53f_sparse_context_12n*c,int pool){float*pk=c->pool+(size_t)pool*ID;for(int d=0;d<ID;d++){float mx=-INFINITY,den=0,val=0;for(int z=0;z<KPOOL;z++){float a=c->gcache[(size_t)(pool*KPOOL+z)*ID+d]+c->apef[(size_t)z*ID+d];if(a>mx)mx=a;}for(int z=0;z<KPOOL;z++){float a=expf(c->gcache[(size_t)(pool*KPOOL+z)*ID+d]+c->apef[(size_t)z*ID+d]-mx);den+=a;val+=a*c->key[(size_t)(pool*KPOOL+z)*ID+d];}pk[d]=val/den;}}
static int sp_ar(const float *in, float *out, int n);
static int select_incremental(glm53f_sparse_context_12n*c,int tokens){int np=tokens/KPOOL,nc=TOPK/KPOOL,out=0;if(nc>np)nc=np;
    if(tokens<=TOPK+KPOOL-1){
#pragma omp parallel for schedule(static)
        for(int p=0;p<np;p++){float s=0;const float*pk=c->pool+(size_t)p*ID;for(int h=0;h<IH;h++){double dot=0;for(int d=0;d<ID;d++)dot+=(double)c->iq[(size_t)h*ID+d]*pk[d];if(dot>0)s+=c->iw[h]*(float)(dot/sqrt((double)ID))/sqrtf((float)IH);}c->pool_score_cache[p]=(pool_score){s,p};}
        qsort(c->pool_score_cache,np,sizeof(pool_score),pool_cmp);goto expand;
    }
#pragma omp parallel for schedule(static)
    for(int p=0;p<np;p++){float s=0;if(p%c->ranks==c->rank){const float*pk=c->pool+(size_t)p*ID;for(int h=0;h<IH;h++){float dot=f32dot(c->iq+(size_t)h*ID,pk,ID);if(dot>0)s+=c->iw[h]*dot/sqrtf((float)(ID*IH));}}c->pool_score_local[p]=s;}
    if(sp_ar(c->pool_score_local,c->pool_score_global,np))return-1;
    pool_top_exact(c->pool_score_cache,c->pool_score_global,np,nc);
expand: for(int i=0;i<nc;i++)for(int z=0;z<KPOOL;z++)c->selected[out++]=c->pool_score_cache[i].id*KPOOL+z;for(int z=np*KPOOL;z<tokens;z++)c->selected[out++]=z;return out;}
static void ensure_mla_shards(glm53f_sparse_context_12n*c){if(c->mla_ql)return;size_t base=(size_t)(TOPK+KPOOL)*LAT,n=base+(size_t)c->hn*(2*LAT+TOPK+KPOOL+8*LAT);float*p=realloc(c->packed,n*4);if(!p)MPI_Abort(MPI_COMM_WORLD,2);c->packed=p;c->mla_ql=p+base;c->mla_log=c->mla_ql+(size_t)c->hn*LAT;c->mla_part=c->mla_log+(size_t)c->hn*(TOPK+KPOOL);c->mla_va=c->mla_part+(size_t)c->hn*8*LAT;}
/* Native Q8 value path: absorbed-query logits over the FP16-rounded latent
 * rows, softmax, value accumulation, then the per-head GGUF v_b matvec.  All
 * phases are work-shared by one team; every output element keeps the serial
 * accumulation order (j for ql, t for va), so results match the former
 * single-threaded loop. */
static int mla_heads_q8_value(glm53f_sparse_context_12n*c,float*out,
        const float*q,const float*z,const int*selected,int nt){
    enum { DC = 64, NDC = LAT / DC, SLOTS = TOPK + KPOOL };
    const size_t row_bytes=glm53f_native_row_size(c->q2_vb_type,LAT);
    const int hn=c->hn,vl=(int)svcntw();
    if(nt<1||nt>SLOTS||hn>8)return-1;
    if(!c->q8v_ql){
        c->q8v_act_bytes=(glm53f_native_act_bytes(LAT)+255)&~(size_t)255;
        c->q8v_ql=a256((size_t)hn*LAT*sizeof(float));
        c->q8v_va=a256((size_t)hn*LAT*sizeof(float));
        c->q8v_log=a256((size_t)hn*SLOTS*sizeof(float));
        c->q8v_sum=a256((size_t)hn*sizeof(float));
        c->q8v_act=a256((size_t)hn*c->q8v_act_bytes);
    }
    float*ql=c->q8v_ql,*va=c->q8v_va,*lg=c->q8v_log,*hsum=c->q8v_sum;
    const int q80=c->q2_vb_type==GLM53F_GGML_Q8_0||c->q2_vb_type==GLM53F_NATIVE_Q8_0R;
    int bad=0;
#pragma omp parallel reduction(|:bad)
    {
#pragma omp for schedule(static)
        for(int w=0;w<hn*NDC;w++){
            const int h=w/NDC,d0=(w%NDC)*DC;
            const uint16_t*wk=c->kvb+(size_t)h*(KD+VD)*LAT;
            float*o=ql+(size_t)h*LAT;
            for(int d=d0;d<d0+DC;d++)o[d]=0.0f;
            for(int j=0;j<KD;j++){float x=q[(size_t)h*KD+j]/sqrtf((float)KD);
                for(int d=d0;d<d0+DC;d+=vl){svbool_t p=svwhilelt_b32(d,LAT);
                    svuint32_t b=svlsl_n_u32_x(p,svld1uh_u32(p,wk+(size_t)j*LAT+d),16);
                    svst1(p,o+d,svmla_n_f32_x(p,svld1(p,o+d),svreinterpret_f32_u32(b),x));}}
        }
#pragma omp for schedule(static)
        for(int w=0;w<hn*nt;w++){
            const int h=w/nt,t=w%nt,r=selected?selected[t]:t;
            lg[(size_t)h*SLOTS+t]=f32dot(ql+(size_t)h*LAT,z+(size_t)r*LAT,LAT);
        }
#pragma omp for schedule(static)
        for(int h=0;h<hn;h++){
            float*l=lg+(size_t)h*SLOTS,mx=-INFINITY,sum=0;
            for(int t=0;t<nt;t++)if(l[t]>mx)mx=l[t];
            for(int t=0;t<nt;t++){l[t]=expf(l[t]-mx);sum+=l[t];}
            hsum[h]=sum;
        }
#pragma omp for schedule(static)
        for(int w=0;w<hn*NDC;w++){
            const int h=w/NDC,d0=(w%NDC)*DC;
            const float*l=lg+(size_t)h*SLOTS;
            float*o=va+(size_t)h*LAT;
            for(int d=d0;d<d0+DC;d++)o[d]=0.0f;
            for(int t=0;t<nt;t++){int r=selected?selected[t]:t;float x=l[t]/hsum[h];
                for(int d=d0;d<d0+DC;d+=vl){svbool_t p=svwhilelt_b32(d,LAT);
                    svst1(p,o+d,svmla_n_f32_x(p,svld1(p,o+d),svld1(p,z+(size_t)r*LAT+d),x));}}
        }
#pragma omp for schedule(static)
        for(int h=0;h<hn;h++)
            if(glm53f_native_act_prepare(c->q8v_act+(size_t)h*c->q8v_act_bytes,
                    va+(size_t)h*LAT,LAT,!q80,q80))bad=1;
        for(int h=0;h<hn;h++){
            glm53f_native_matrix m={out+(size_t)h*VD,c->q2_vb+(size_t)h*VD*row_bytes,
                c->q2_vb_type,VD,LAT};
            if(glm53f_native_matvec_team(&m,1,c->q8v_act+(size_t)h*c->q8v_act_bytes))bad=1;
        }
    }
    return bad?-1:0;
}
/* Prefetch plan (glm53f_pf_plan.h): the native q_a / kv_a / q_b matvecs of this layer's decode front. */
void glm53f_sparse_prefetch_plan_12n(const glm53f_sparse_context_12n *c) {
    if (!c || !c->q2_native || c->cp) return;
    const glm53f_native_matrix ax[2] = {{NULL, c->q2_qa, c->q2_qa_type, QA, H}, {NULL, c->q2_kva, c->q2_kva_type, LAT, H}};
    const glm53f_native_matrix aq = {NULL, c->q2_qb, c->q2_qb_type, c->qd, QA};
    glm53f_pf_add_matvec(ax, 2);
    glm53f_pf_add_matvec(&aq, 1);
}
/* ---- int8 panel64 GEMM for the native Q8_0 prefill projections (q_a|kv_a and q_b), as in the KDA prefill ------------- */
enum { SG_T = GLM53F_PREFILL_ATTN_TOKENS + 4 };
static inline void sg_pack(int8_t *xp, float *xsp, const int8_t *xq, const float *xs, int K, int ng) {
    const int nb = K / 32;
#pragma omp for schedule(static)
    for (int g = 0; g < ng; ++g) {
        const int8_t *rows[6]; const float *xsr[6];
        for (int u = 0; u < 6; ++u) { rows[u] = xq + (size_t)(g * 6 + u) * K; xsr[u] = xs + (size_t)(g * 6 + u) * nb; }
        gmn_pack6(xp + (size_t)g * 6 * K, xsp + (size_t)g * 6 * nb, rows, xsr, K);
    }
}
static uint8_t *sg_concat(uint8_t *a, size_t ab, uint8_t *b, size_t bb) {
    uint8_t *out = NULL;
    if (posix_memalign((void **)&out, 256, ab + bb)) return NULL;
    memcpy(out, a, ab); memcpy(out + ab, b, bb);
    free(a); free(b);
    return out;
}
static int sparse_gemm_setup(glm53f_sparse_context_12n *c) {
    if (c->sg_state) return c->sg_state > 0;
    c->sg_state = -1;
    const char *e = getenv("GLM53F_SPARSE_GEMM");
    if (e && *e && !atoi(e)) return 0;
    if (!c->q2_native || c->cp || c->qd % 64 || !glm53f_q80_family(c->q2_qa_type) || !glm53f_q80_family(c->q2_kva_type) ||
        !glm53f_q80_family(c->q2_qb_type)) return 0;
    uint8_t *qa = glm53f_q80_to_panel64(c->q2_qa_type, c->q2_qa, QA, H), *kv = glm53f_q80_to_panel64(c->q2_kva_type, c->q2_kva, LAT, H);
    uint8_t *qb = glm53f_q80_to_panel64(c->q2_qb_type, c->q2_qb, c->qd, QA);
    const int ocols = c->hn * VD;
    uint8_t *opw = glm53f_q80_family(c->q2_op_type) ? glm53f_q80_to_panel64(c->q2_op_type, c->q2_op, H, ocols) : NULL;
    if (!qa || !kv || !qb) { free(qa); free(kv); free(qb); free(opw); return 0; }
    const size_t pb = gk_panel64_bytes(32, H);
    c->sg_w1 = sg_concat(qa, (size_t)(QA / 64) * pb, kv, (size_t)(LAT / 64) * pb);
    c->sg_wqb = qb;
    c->sg_wop = opw; /* may be NULL: o_proj then keeps the native path */
    if (!c->sg_w1) return 0;
    c->sg_xq = a256((size_t)SG_T * H); c->sg_xp = a256((size_t)SG_T * H); c->sg_xs = a256((size_t)SG_T * (H / 32) * 4);
    c->sg_xsp = a256((size_t)SG_T * (H / 32) * 4); c->sg_bt = a256((size_t)SG_T * (H / 32) * 4);
    c->sg_qq = a256((size_t)SG_T * QA); c->sg_qp = a256((size_t)SG_T * QA); c->sg_qs = a256((size_t)SG_T * (QA / 32) * 4);
    c->sg_qsp = a256((size_t)SG_T * (QA / 32) * 4);
    c->sg_y1 = a256((size_t)SG_T * (QA + LAT) * 4); c->sg_yq = a256((size_t)SG_T * c->qd * 4);
    if (opw) {
        c->sg_oq = a256((size_t)SG_T * ocols); c->sg_op = a256((size_t)SG_T * ocols); c->sg_os = a256((size_t)SG_T * (ocols / 32) * 4);
        c->sg_osp = a256((size_t)SG_T * (ocols / 32) * 4); c->sg_yo = a256((size_t)SG_T * H * 4);
        memset(c->sg_oq, 0, (size_t)SG_T * ocols);
    }
    memset(c->sg_xq, 0, (size_t)SG_T * H); memset(c->sg_qq, 0, (size_t)SG_T * QA);
    c->sg_state = 1;
    return 1;
}
/* y[t][r] = W x[t] for tokens <= GLM53F_PREFILL_ATTN_TOKENS; one parallel region */
static void sparse_gemm_run(const uint8_t *wp, int K, int R, const float *x, int tokens, int8_t *xq, float *xs, int8_t *xp,
                            float *xsp, float *bt, float *y) {
    const int mpad = (tokens + 5) / 6 * 6, ng = mpad / 6, nb = K / 32, panels = R / 64, nch = (ng + 2) / 3;
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int t = 0; t < mpad; ++t) {
            if (t < tokens) gmn_quant_row(x + (size_t)t * K, K, xq + (size_t)t * K, xs + (size_t)t * nb, bt + (size_t)t * nb);
            else { memset(xq + (size_t)t * K, 0, K); memset(xs + (size_t)t * nb, 0, nb * 4); }
        }
        sg_pack(xp, xsp, xq, xs, K, ng);
#pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < panels * nch; ++task) {
            const int pnl = task / nch, ch = task % nch, t0 = ch * 18, t1 = (ch + 1) * 18 < mpad ? (ch + 1) * 18 : mpad;
            gk_gemm_panel64(32, wp, K, pnl * 64, pnl * 64 + 64, t0, t1, xp, xsp, y, (size_t)R);
        }
    }
}
/* Prefill async reduction (target decode): the o_proj partials of a tile are reduced by a helper thread on the spare core, so
 * every OTHER collective of the tile (index scores, pool scores) goes through MPI to avoid sharing the multi-TNI state. */
static int sp_defer_reduce;
void glm53f_sparse_set_defer_reduce_12n(int on) { sp_defer_reduce = on; }
static inline int sp_ar(const float *in, float *out, int n) {
    return sp_defer_reduce ? glm53f_sum_allreduce_mpi_12n(in, out, n) : glm53f_sum_allreduce_12n(in, out, n);
}
static inline int sp_ar_prefill(const float *in, float *out, int n) {
    return sp_defer_reduce ? glm53f_sum_allreduce_mpi_12n(in, out, n) : glm53f_sum_allreduce_prefill_12n(in, out, n);
}
void glm53f_sparse_prewarm_12n(glm53f_sparse_context_12n *c) {
    if (c && c->q2_native && !c->cp) (void)sparse_gemm_setup(c);
}
static inline int sp_is_q80(int type) {
    return type == GLM53F_GGML_Q8_0 || type == GLM53F_NATIVE_Q8_0R || type == GLM53F_NATIVE_Q8_0R16;
}
/* Decode front of the replicated sparse layer in ONE parallel region (GLM53F_SPARSE_FUSE_FRONT=0 disables):
 * q_a / kv_a (native Q8_0) and the three bf16 index projections of x, then q_b and the bf16 index query projection
 * of the normalised q_a.  Same kernels and per-row arithmetic as the separate calls, so results are bit-identical. */
static int sparse_front_fused(glm53f_sparse_context_12n *c, int pos, const float *x, float *raw) {
    if (!c->q2_native || glm53f_sparse_scalar_reference) return 0;
    static int enabled = -1;
    if (enabled < 0) { const char *e = getenv("GLM53F_SPARSE_FUSE_FRONT"); enabled = !e || !*e || atoi(e); }
    if (!enabled || !sp_is_q80(c->q2_qa_type) || !sp_is_q80(c->q2_kva_type) || !sp_is_q80(c->q2_qb_type)) return 0;
    if (!c->front_act_x) {
        c->front_act_x = a256((glm53f_native_act_bytes(H) + 255) & ~(size_t)255);
        c->front_act_q = a256((glm53f_native_act_bytes(QA) + 255) & ~(size_t)255);
    }
    float *latent = c->latent + (size_t)pos * LAT;
    const glm53f_native_matrix ax[2] = {{c->qres, c->q2_qa, c->q2_qa_type, QA, H},
                                        {latent, c->q2_kva, c->q2_kva_type, LAT, H}};
    const glm53f_native_matrix aq = {c->query, c->q2_qb, c->q2_qb_type, c->qd, QA};
    float *gout = c->gcache + (size_t)pos * ID;
    int bad = 0;
#pragma omp parallel reduction(|:bad)
    {
        bad |= glm53f_native_act_prepare_team(c->front_act_x, x, H, 0, 1) != 0;
        bad |= glm53f_native_matvec_team(ax, 2, c->front_act_x) != 0;      /* barrier */
        /* bf16 projections of x: wk (ID rows), gatew (ID rows), wp (IH rows) in 8-row blocks */
#pragma omp for schedule(static)
        for (int t = 0; t < ID / 8 * 2 + IH / 8; ++t) {
            if (t < ID / 8) b16dot8(raw + t * 8, c->wk + (size_t)t * 8 * H, x, H);
            else if (t < 2 * ID / 8) b16dot8(gout + (t - ID / 8) * 8, c->gatew + (size_t)(t - ID / 8) * 8 * H, x, H);
            else b16dot8(c->iw + (t - 2 * ID / 8) * 8, c->wp + (size_t)(t - 2 * ID / 8) * 8 * H, x, H);
        }
#pragma omp single
        {
            glm53f_rmsnorm_bf16(c->qres, c->qres, c->qan, QA, 1e-5f);
            glm53f_rmsnorm_bf16(latent, latent, c->kvan, LAT, 1e-5f);
            glm53f_layernorm_bf16(c->key + (size_t)pos * ID, raw, c->knw, c->knb, ID, 1e-6f);
        }
        bad |= glm53f_native_act_prepare_team(c->front_act_q, c->qres, QA, 0, 1) != 0;
        bad |= glm53f_native_matvec_team(&aq, 1, c->front_act_q) != 0;     /* barrier */
#pragma omp for schedule(static)
        for (int b = 0; b < IH * ID / 8; ++b) b16dot8(c->iq + b * 8, c->wqb + (size_t)b * 8 * QA, c->qres, QA);
    }
    return bad ? -1 : 1;
}
static int sparse_attention_local_replicated(glm53f_sparse_context_12n *c,
                                             float *attn, const float *x) {
    if (c->length >= c->capacity) return -1;
    int pos = c->length, tokens = pos + 1, ns;
    double begin = sparse_clock(c);
    float raw[ID];
    const int fused = sparse_front_fused(c, pos, x, raw);
    if (fused < 0) return -1;
    if (fused) {
        sparse_dump(c, "q_a_norm", c->qres, QA * sizeof(float));
        sparse_dump(c, "q_b", c->query, (size_t)c->qd * sizeof(float));
        sparse_dump(c, "kv_a_norm", c->latent + (size_t)pos * LAT, LAT * sizeof(float));
    } else {
    if (c->q2_native) {
        if (glm53f_iq_matvec(c->qres, c->q2_qa, c->q2_qa_type,
                             QA, H, x)) return -1;
    } else if (c->int8_enabled) {
        if (sparse_mv_int8(c->qres, c->int8_weight[0], c->int8_scale[0], x, QA, H)) return -1;
    } else mv_f8(c->qres, c->qa, c->qas, x, QA, H);
    glm53f_rmsnorm_bf16(c->qres, c->qres, c->qan, QA, 1e-5f);
    sparse_dump(c, "q_a_norm", c->qres, QA * sizeof(float));
    if (c->q2_native) {
        if (glm53f_iq_matvec(c->query, c->q2_qb, c->q2_qb_type,
                             c->qd, QA, c->qres) ||
            glm53f_iq_matvec(c->latent + (size_t)pos * LAT, c->q2_kva,
                             c->q2_kva_type, LAT, H, x)) return -1;
    } else if (c->int8_enabled) {
        if (sparse_mv_int8(c->query, c->int8_weight[1], c->int8_scale[1], c->qres, c->qd, QA) ||
            sparse_mv_int8(c->latent + (size_t)pos * LAT, c->int8_weight[2], c->int8_scale[2], x, LAT, H)) return -1;
    } else {
        mv_f8(c->query, c->qb, c->qbs, c->qres, c->qd, QA);
        mv_f8(c->latent + (size_t)pos * LAT, c->kva, c->kvas, x, LAT, H);
    }
    glm53f_rmsnorm_bf16(c->latent + (size_t)pos * LAT,
        c->latent + (size_t)pos * LAT, c->kvan, LAT, 1e-5f);
    sparse_dump(c, "q_b", c->query, (size_t)c->qd * sizeof(float));
    sparse_dump(c, "kv_a_norm", c->latent + (size_t)pos * LAT,
                LAT * sizeof(float));
    mv_b16(raw, c->wk, x, ID, H);
    glm53f_layernorm_bf16(c->key + (size_t)pos * ID, raw,
                          c->knw, c->knb, ID, 1e-6f);
    mv_b16(c->gcache + (size_t)pos * ID, c->gatew, x, ID, H);
    mv_b16(c->iq, c->wqb, c->qres, IH * ID, QA);
    mv_b16(c->iw, c->wp, x, IH, H);
    }
    c->profile_phase[0] += sparse_clock(c) - begin;
    begin = sparse_clock(c);
    if (getenv("GLM53F_SPARSE_REFERENCE"))
        ns = glm53f_index_select_decode(c->pool, c->selected, c->iq, c->iw,
            c->key, c->gcache, c->apef, tokens, KPOOL, TOPK, IH, ID);
    else {
        if (tokens % KPOOL == 0) update_completed_pool(c, tokens / KPOOL - 1);
        ns = select_incremental(c, tokens);
    }
    c->profile_phase[1] += sparse_clock(c) - begin;
    if (ns < 1) return -1;
    begin = sparse_clock(c);
    if (c->q2_native) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < ns; ++i)
            for (int d = 0; d < LAT; ++d)
                c->packed[(size_t)i * LAT + d] =
                    (float)(_Float16)c->latent[(size_t)c->selected[i] * LAT + d];
        c->profile_phase[2] += sparse_clock(c) - begin;
        begin = sparse_clock(c);
        if (mla_heads_q8_value(c,attn,c->query,c->packed,NULL,ns)) return -1;
    } else if (tokens > TOPK + KPOOL - 1 && !getenv("GLM53F_SPARSE_NO_PACK")) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < ns; ++i)
            memcpy(c->packed + (size_t)i * LAT,
                   c->latent + (size_t)c->selected[i] * LAT, LAT * sizeof(float));
        ensure_mla_shards(c);
        c->profile_phase[2] += sparse_clock(c) - begin;
        begin = sparse_clock(c);
        if (mla_heads_sharded(attn, c->query, c->packed, c->kvb, ns, c->hn,
                              c->mla_ql, c->mla_log, c->mla_part, c->mla_va))
            return -1;
    } else if (mla_heads(attn, c->query, c->latent, c->kvb,
                         c->selected, ns, c->hn)) return -1;
    sparse_dump(c, "attn_local", attn, (size_t)c->hn * VD * sizeof(float));
    c->profile_phase[3] += sparse_clock(c) - begin;
    c->length++;
    return 0;
}

static int cp_candidate_cmp(const void*a,const void*b){const cp_candidate*x=a,*y=b;if(x->score>y->score)return-1;if(x->score<y->score)return 1;return x->id<y->id?-1:x->id>y->id;}
static void cp_heap_down(cp_candidate *h, int n, int p) {
    for (;;) {
        int worst = p, left = 2 * p + 1, right = left + 1;
        if (left < n && cp_candidate_cmp(h + left, h + worst) > 0) worst = left;
        if (right < n && cp_candidate_cmp(h + right, h + worst) > 0) worst = right;
        if (worst == p) return;
        cp_candidate tmp = h[p]; h[p] = h[worst]; h[worst] = tmp; p = worst;
    }
}
/* Same total ordering as qsort, including ties. Only the best k are consumed. */
static void cp_top_exact(cp_candidate *v, int n, int k) {
    if (k > n) k = n;
    if (!k) return;
    for (int p = k / 2; p-- > 0;) cp_heap_down(v, k, p);
    for (int p = k; p < n; ++p) if (cp_candidate_cmp(v + p, v) < 0) {
        v[0] = v[p]; cp_heap_down(v, k, 0);
    }
    qsort(v, k, sizeof(*v), cp_candidate_cmp);
}
static int cp_mla(glm53f_sparse_context_12n *c, float *attn, int ns) {
    int parallel_min = getenv("GLM53F_CP_MLA_PAR_MIN") ?
                       atoi(getenv("GLM53F_CP_MLA_PAR_MIN")) : 512;
    if (ns < parallel_min || getenv("GLM53F_CP_MLA_REFERENCE"))
        return mla_heads(attn, c->query, c->cp_pack, c->kvb, c->cp_pack_index, ns, c->hn);
    if (!c->mla_ql) {
        c->packed = a256((size_t)c->hn * (2 * LAT + TOPK + KPOOL) * sizeof(float));
        c->mla_ql = c->packed;
        c->mla_log = c->mla_ql + (size_t)c->hn * LAT;
        c->mla_va = c->mla_log + (size_t)c->hn * (TOPK + KPOOL);
    }
    return mla_heads_exact_parallel(attn, c->query, c->cp_pack, c->kvb,
        ns, c->hn, c->mla_ql, c->mla_log, c->mla_va);
}
static int cp_exchange_selected(glm53f_sparse_context_12n *c, int ns) {
    if (getenv("GLM53F_CP_EXCHANGE_REFERENCE")) {
        memset(c->cp_pack, 0, (size_t)ns * LAT * sizeof(float));
        for (int i = 0; i < ns; ++i) {
            int p = c->selected[i];
            if (p % c->ranks == c->rank)
                cp_read_latent(c->cp_pack + (size_t)i * LAT, c, p / c->ranks);
            c->cp_pack_index[i] = i;
        }
        return MPI_Allreduce(MPI_IN_PLACE, c->cp_pack, ns * LAT, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS ? 0 : -1;
    }
    /* The hot prefix is computed identically on every rank. Avoid the
     * selected-row collective while the complete selected set is resident. */
    int all_hot = c->hot_latent != NULL;
    for (int i = 0; i < ns && all_hot; ++i)
        if (c->selected[i] >= c->hot_prefix) all_hot = 0;
    if (all_hot) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < ns; ++i) {
            hot_read_latent(c->cp_pack + (size_t)i * LAT, c, c->selected[i]);
            c->cp_pack_index[i] = i;
        }
        return 0;
    }
    /* Each selected latent has exactly one owner. Gather only owned rows,
     * instead of reducing a mostly-zero 4 MiB vector from every rank. */
    int counts[12] = {0}, offsets[12], cursor[12], local_rows = 0;
    if (!c->cp_gather) c->cp_gather = a256((size_t)(TOPK + KPOOL) * LAT * sizeof(float));
    if (c->latent_bf16 && !c->cp_bf16_local) {
        c->cp_bf16_local = a256((size_t)(TOPK + KPOOL) * LAT * sizeof(uint16_t));
        c->cp_bf16_gather = a256((size_t)(TOPK + KPOOL) * LAT * sizeof(uint16_t));
        if (!c->cp_bf16_local || !c->cp_bf16_gather) return -1;
    }
    for (int i = 0; i < ns; ++i)
        if (!c->hot_latent || c->selected[i] >= c->hot_prefix)
            counts[c->selected[i] % c->ranks] += LAT;
    int total = 0;
    for (int r = 0; r < c->ranks; ++r) { cursor[r] = offsets[r] = total; total += counts[r]; }
    for (int i = 0; i < ns; ++i) {
        int p = c->selected[i], owner = p % c->ranks;
        if (c->hot_latent && p < c->hot_prefix) {
            c->cp_pack_index[i] = -1;
        } else {
            c->cp_pack_index[i] = cursor[owner]; cursor[owner] += LAT;
        }
        if ((!c->hot_latent || p >= c->hot_prefix) && owner == c->rank) {
            if (c->latent_bf16)
                memcpy(c->cp_bf16_local + (size_t)local_rows++ * LAT,
                       (const uint16_t *)c->cp_latent + (size_t)(p / c->ranks) * LAT,
                       LAT * sizeof(uint16_t));
            else {
                cp_read_latent(c->cp_pack + (size_t)local_rows * LAT, c, p / c->ranks);
                local_rows++;
            }
        }
    }
    if (c->latent_bf16) {
        if (MPI_Allgatherv(c->cp_bf16_local, counts[c->rank], MPI_UINT16_T,
                c->cp_bf16_gather, counts, offsets, MPI_UINT16_T,
                MPI_COMM_WORLD) != MPI_SUCCESS) return -1;
    } else if (MPI_Allgatherv(c->cp_pack, counts[c->rank], MPI_FLOAT, c->cp_gather,
            counts, offsets, MPI_FLOAT, MPI_COMM_WORLD) != MPI_SUCCESS) return -1;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < ns; ++i) {
        float *destination = c->cp_pack + (size_t)i * LAT;
        if (c->cp_pack_index[i] < 0) {
            hot_read_latent(destination, c, c->selected[i]);
            c->cp_pack_index[i] = i;
            continue;
        }
        if (c->latent_bf16) {
            read_bf16_row(destination, c->cp_bf16_gather + c->cp_pack_index[i]);
        } else {
            const float *source = c->cp_gather + c->cp_pack_index[i];
            /* Canonicalize negative zero as the old reduction path did. */
            for (int d = 0; d < LAT; ++d) destination[d] = source[d] == 0 ? 0.0f : source[d];
        }
        c->cp_pack_index[i] = i;
    }
    return 0;
}
static float cp_pool_score(const float*q,const float*hw,const float*pk){float score=0;for(int h=0;h<IH;h++){double dot=0;for(int d=0;d<ID;d++)dot+=(double)q[(size_t)h*ID+d]*pk[d];if(dot>0)score+=hw[h]*(float)(dot/sqrt((double)ID))/sqrtf((float)IH);}return score;}
static int sparse_attention_local_cp(glm53f_sparse_context_12n *c, float *attn, const float *x) {
    if (c->length >= c->capacity)
        return -1;
    int pos = c->length, tokens = pos + 1, owner = pos % c->ranks, slot = pos / c->ranks;
    if (c->q2_native) glm53f_iq_matvec(c->qres, c->q2_qa, c->q2_qa_type, QA, H, x);
    else if (c->int8_enabled) sparse_mv_int8(c->qres, c->int8_weight[0], c->int8_scale[0], x, QA, H);
    else mv_f8(c->qres, c->qa, c->qas, x, QA, H);
    glm53f_rmsnorm_bf16(c->qres, c->qres, c->qan, QA, 1e-5f);
    if (c->q2_native) {
        glm53f_iq_matvec(c->query, c->q2_qb, c->q2_qb_type, c->qd, QA, c->qres);
        glm53f_iq_matvec(c->cp_cur_latent, c->q2_kva, c->q2_kva_type, LAT, H, x);
    } else if (c->int8_enabled) {
        sparse_mv_int8(c->query, c->int8_weight[1], c->int8_scale[1], c->qres, c->qd, QA);
        sparse_mv_int8(c->cp_cur_latent, c->int8_weight[2], c->int8_scale[2], x, LAT, H);
    } else {
        mv_f8(c->query, c->qb, c->qbs, c->qres, c->qd, QA);
        mv_f8(c->cp_cur_latent, c->kva, c->kvas, x, LAT, H);
    }
    glm53f_rmsnorm_bf16(c->cp_cur_latent, c->cp_cur_latent, c->kvan, LAT, 1e-5f);
    float raw[ID];
    mv_b16(raw, c->wk, x, ID, H);
    glm53f_layernorm_bf16(c->cp_cur_key, raw, c->knw, c->knb, ID, 1e-6f);
    mv_b16(c->cp_cur_gate, c->gatew, x, ID, H);
    if (owner == c->rank) {
        cp_store_latent(c, slot, c->cp_cur_latent);
        memcpy(c->cp_key + (size_t)slot * ID, c->cp_cur_key, ID * 4);
        memcpy(c->cp_gate + (size_t)slot * ID, c->cp_cur_gate, ID * 4);
    }
    if (c->hot_latent && pos < c->hot_prefix)
        hot_store_latent(c, pos, c->cp_cur_latent);
    int prefix = c->prefix_replicated && tokens <= 512 && c->local_capacity >= 1024;
    if (prefix)
        cp_store_latent(c, c->local_capacity - 512 + pos, c->cp_cur_latent);
    mv_b16(c->iq, c->wqb, c->qres, IH * ID, QA);
    mv_b16(c->iw, c->wp, x, IH, H);
    int pools = tokens / KPOOL;
    if (tokens % KPOOL == 0) {
        int pool = pools - 1, powner = pool % c->ranks;
        memset(c->cp_exchange, 0, (size_t)2 * KPOOL * ID * 4);
        for (int z = 0; z < KPOOL; z++) {
            int p = pool * KPOOL + z;
            if (p % c->ranks == c->rank) {
                int s = p / c->ranks;
                memcpy(c->cp_exchange + (size_t)z * ID, c->cp_key + (size_t)s * ID, ID * 4);
                memcpy(c->cp_exchange + (size_t)(KPOOL + z) * ID, c->cp_gate + (size_t)s * ID, ID * 4);
            }
        }
        if (glm53f_sum_allreduce_12n(c->cp_exchange, c->cp_exchange,
                                    2 * KPOOL * ID))
            return -1;
        if (powner == c->rank || prefix) {
            float *pk = prefix ? c->cp_pool + (size_t)(c->pool_capacity - 128 + pool) * ID
                               : c->cp_pool + (size_t)(pool / c->ranks) * ID;
            for (int d = 0; d < ID; d++) {
                float mx = -INFINITY, den = 0, val = 0;
                for (int z = 0; z < KPOOL; z++) {
                    float a = c->cp_exchange[(size_t)(KPOOL + z) * ID + d] + c->apef[(size_t)z * ID + d];
                    if (a > mx)
                        mx = a;
                }
                for (int z = 0; z < KPOOL; z++) {
                    float a = expf(c->cp_exchange[(size_t)(KPOOL + z) * ID + d] + c->apef[(size_t)z * ID + d] - mx);
                    den += a;
                    val += a * c->cp_exchange[(size_t)z * ID + d];
                }
                pk[d] = val / den;
            }
            if (prefix && powner == c->rank)
                memcpy(c->cp_pool + (size_t)(pool / c->ranks) * ID, pk, ID * sizeof(float));
        }
    }
    if (prefix) {
        cp_candidate *candidate = c->cp_candidate_local;
        double qt[IH * ID];
        glm53f_index_query_transpose(qt, c->iq);
        int score_reference = getenv("GLM53F_INDEX_REFERENCE") != NULL;
#pragma omp parallel for schedule(static)
        for (int pool = 0; pool < pools; ++pool)
            candidate[pool] = (cp_candidate){
                score_reference ? cp_pool_score(c->iq, c->iw,
                    c->cp_pool + (size_t)(c->pool_capacity - 128 + pool) * ID)
                : glm53f_index_score_transposed(qt, c->iw,
                    c->cp_pool + (size_t)(c->pool_capacity - 128 + pool) * ID), pool};
        cp_top_exact(candidate, pools, pools);
        int ns = 0;
        for (int pool = 0; pool < pools; ++pool)
            for (int z = 0; z < KPOOL; ++z) c->selected[ns++] = candidate[pool].id * KPOOL + z;
        for (int p = pools * KPOOL; p < tokens; ++p) c->selected[ns++] = p;
        for (int i = 0; i < ns; i++) {
            int selected = c->selected[i];
            cp_read_latent(c->cp_pack + (size_t)i * LAT, c, c->local_capacity - 512 + selected);
            for (int d = 0; d < LAT; d++)
                if (c->cp_pack[(size_t)i * LAT + d] == 0)
                    c->cp_pack[(size_t)i * LAT + d] = 0.0f;
            c->cp_pack_index[i] = i;
        }
        if (cp_mla(c, attn, ns))
            return -1;
        c->length++;
        return 0;
    }
    int choose = TOPK / KPOOL;
    if (choose > pools)
        choose = pools;
    int local_pools = (pools + c->ranks - 1 - c->rank) / c->ranks;
    cp_candidate *local = c->cp_candidate_local, *gather = c->cp_candidate_gather;
    double qt[IH * ID];
    glm53f_index_query_transpose(qt, c->iq);
    int score_reference = getenv("GLM53F_INDEX_REFERENCE") != NULL;
/* Each pool score is independent; keep the scalar double accumulation
 * inside a score unchanged while distributing pools over all cores. */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < local_pools; i++) {
        int p = c->rank + i * c->ranks;
        local[i] =
            (cp_candidate){score_reference ? cp_pool_score(c->iq, c->iw, c->cp_pool + (size_t)i * ID)
                                           : glm53f_index_score_transposed(qt, c->iw, c->cp_pool + (size_t)i * ID),
                           p};
    }
    if (getenv("GLM53F_CP_TOP_REFERENCE"))
        qsort(local, local_pools, sizeof(*local), cp_candidate_cmp);
    else
        cp_top_exact(local, local_pools, choose);
    for (int i = local_pools; i < choose; i++)
        local[i] = (cp_candidate){-INFINITY, INT_MAX};
    if (choose && MPI_Allgather(local, choose * (int)sizeof(*local), MPI_BYTE, gather, choose * (int)sizeof(*local),
                                MPI_BYTE, MPI_COMM_WORLD) != MPI_SUCCESS)
        return -1;
    if (choose) {
        if (getenv("GLM53F_CP_TOP_REFERENCE"))
            qsort(gather, (size_t)c->ranks * choose, sizeof(*gather), cp_candidate_cmp);
        else
            cp_top_exact(gather, c->ranks * choose, choose);
    }
    int ns = 0;
    for (int i = 0; i < choose; i++)
        for (int z = 0; z < KPOOL; z++)
            c->selected[ns++] = gather[i].id * KPOOL + z;
    for (int p = pools * KPOOL; p < tokens; p++)
        c->selected[ns++] = p;
    if (ns < 1)
        return -1;
    if (cp_exchange_selected(c, ns) || cp_mla(c, attn, ns))
        return -1;
    c->length++;
    return 0;
}
static int sparse_attention_local(glm53f_sparse_context_12n *c,
                                   float *attn, const float *x) {
    if (!c->cp) return sparse_attention_local_replicated(c, attn, x);
    double begin = sparse_clock(c);
    int rc = sparse_attention_local_cp(c, attn, x);
    c->profile_phase[6] += sparse_clock(c) - begin;
    return rc;
}
int glm53f_sparse_cache_append_12n(glm53f_sparse_context_12n *c, const float *x) {
    if (!c || !x || c->length >= c->capacity) return -1;
    /* Keep the context-parallel collectives unchanged. The replicated cache
     * used by 8K decode needs only these three projections and pool updates. */
    if (c->cp) return sparse_attention_local(c, c->attn, x);
    int pos = c->length;
    float raw[ID];
    float *latent = c->latent + (size_t)pos * LAT;
    if (c->q2_native) glm53f_iq_matvec(latent, c->q2_kva, c->q2_kva_type, LAT, H, x);
    else mv_f8(latent, c->kva, c->kvas, x, LAT, H);
    glm53f_rmsnorm_bf16(latent, latent, c->kvan, LAT, 1e-5f);
    mv_b16(raw, c->wk, x, ID, H);
    glm53f_layernorm_bf16(c->key + (size_t)pos * ID, raw,
                         c->knw, c->knb, ID, 1e-5f);
    mv_b16(c->gcache + (size_t)pos * ID, c->gatew, x, ID, H);
    if ((pos + 1) % KPOOL == 0) update_completed_pool(c, (pos + 1) / KPOOL - 1);
    c->length++;
    return 0;
}
int glm53f_sparse_sublayer_12n(void*context,float*out,const float*x){glm53f_sparse_context_12n*c=context;if(!c||sparse_attention_local(c,c->attn,x))return-1;int local_cols=c->hn*VD;
    double begin = sparse_clock(c);
    /* The output projection is the largest sparse-layer matvec.  Use the
     * eight-row SVE kernel so each thread reuses a decoded weight vector
     * across rows, while preserving the per-row accumulation order. */
    if (c->q2_native) glm53f_iq_matvec(c->partial,c->q2_op,c->q2_op_type,H,local_cols,c->attn);
    else if (c->int8_enabled) sparse_mv_int8(c->partial,c->int8_weight[3],c->int8_scale[3],c->attn,H,local_cols);
    else glm53f_mv_fp8_block128_bits(c->partial,c->op,c->ops,c->attn,H,local_cols);
    c->profile_phase[4] += sparse_clock(c) - begin;
    begin = sparse_clock(c);
    int rc = glm53f_sum_allreduce_12n(c->partial,out,H);
    if (!rc) sparse_dump(c, "dsa_out", out, H * sizeof(float));
    c->profile_phase[5] += sparse_clock(c) - begin;
    return rc;}
int glm53f_sparse_output_reference_12n(glm53f_sparse_context_12n*c,float*out,
        const float*attention_heads){
    if(!c||!out||!attention_heads)return-1;
    int cols=c->hn*VD;const float*local=attention_heads+(size_t)c->h0*VD;
    if(c->q2_native){if(glm53f_iq_matvec(c->partial,c->q2_op,c->q2_op_type,H,cols,local))return-1;}
    else if(c->int8_enabled)sparse_mv_int8(c->partial,c->int8_weight[3],c->int8_scale[3],local,H,cols);
    else glm53f_mv_fp8_block128_bits(c->partial,c->op,c->ops,local,H,cols);
    return glm53f_sum_allreduce_12n(c->partial,out,H);
}
int glm53f_sparse_value_reference_12n(glm53f_sparse_context_12n*c,float*out,
        const float*kv_latent){
    if(!c||!out||!kv_latent||!c->q2_native)return-1;
    const size_t row_bytes=glm53f_native_row_size(c->q2_vb_type,LAT);
    float cached[LAT];
    for(int d=0;d<LAT;d++)cached[d]=(float)(_Float16)kv_latent[d];
    for(int h=0;h<c->hn;h++)
        if(glm53f_iq_matvec(out+(size_t)h*VD,
                c->q2_vb+(size_t)h*VD*row_bytes,
                c->q2_vb_type,VD,LAT,cached))return-1;
    return 0;
}
int glm53f_sparse_sublayer_batch_12n(glm53f_sparse_context_12n *c,
        float *out, const float *x, int tokens) {
    if (!c || !out || !x || tokens < 1 || tokens > 5 || c->length + tokens > c->capacity)
        return -1;
    int batch_mode = getenv("GLM53F_SPARSE_BATCH_OP") ?
                     atoi(getenv("GLM53F_SPARSE_BATCH_OP")) : 0;
    if (!batch_mode || c->int8_enabled) {
        for (int t = 0; t < tokens; t++)
            if (glm53f_sparse_sublayer_12n(c, out + (size_t)t * H, x + (size_t)t * H))
                return -1;
        return 0;
    }
    if (tokens == 1) return glm53f_sparse_sublayer_12n(c, out, x);
    if (tokens == 5) {
        if (glm53f_sparse_sublayer_batch_12n(c, out, x, 4)) return -1;
        return glm53f_sparse_sublayer_12n(c, out + (size_t)4 * H, x + (size_t)4 * H);
    }
    int cols = c->hn * VD;
    if (!c->batch_attn) {
        c->batch_attn = a256((size_t)4 * cols * sizeof(float));
        c->batch_partial = a256((size_t)4 * H * sizeof(float));
    }
    /* Attention and KV updates remain causal. Reuse output-projection weights
     * across verified positions, then combine all positions in one collective. */
    for (int t = 0; t < tokens; t++)
        if (sparse_attention_local(c, c->batch_attn + (size_t)t * cols,
                                   x + (size_t)t * H)) return -1;
    double begin = sparse_clock(c);
    if (c->q2_native) {
        glm53f_native_matrix op = {c->batch_partial,c->q2_op,c->q2_op_type,H,cols};
        if (glm53f_native_matvec_batch(&op, 1, c->batch_attn, tokens)) return -1;
    } else glm53f_mv_fp8_block128_bits_batch(c->batch_partial, c->op, c->ops,
                                            c->batch_attn, tokens, H, cols);
    c->profile_phase[4] += sparse_clock(c) - begin;
    begin = sparse_clock(c);
    if (batch_mode == 2) {
        for (int t = 0; t < tokens; ++t)
            if (glm53f_sum_allreduce_12n(c->batch_partial + (size_t)t * H,
                                         out + (size_t)t * H, H)) return -1;
        c->profile_phase[5] += sparse_clock(c) - begin;
        return 0;
    }
    int rc = glm53f_sum_allreduce_12n(c->batch_partial, out, tokens * H);
    c->profile_phase[5] += sparse_clock(c) - begin;
    return rc;
}
struct glm53f_sparse_prefill_workspace_12n {
    float qres[GLM53F_PREFILL_ATTN_TOKENS * QA];
    float query[GLM53F_PREFILL_ATTN_TOKENS * 6 * KD];
    float iq[GLM53F_PREFILL_ATTN_TOKENS * IH * ID];
    float iw[GLM53F_PREFILL_ATTN_TOKENS * IH];
    float raw[GLM53F_PREFILL_ATTN_TOKENS * ID];
    float attn[GLM53F_PREFILL_ATTN_TOKENS * 6 * VD];
    float partial[GLM53F_PREFILL_ATTN_TOKENS * H];
    int selected[GLM53F_PREFILL_ATTN_TOKENS][TOPK + KPOOL];
    int count[GLM53F_PREFILL_ATTN_TOKENS];
    double qt[GLM53F_PREFILL_ATTN_TOKENS][IH * ID];
    pool_score top[GLM53F_PREFILL_ATTN_TOKENS][TOPK / KPOOL];
    float *score_local, *score_global;
    float *score_packed_local, *score_packed_global;
    int *score_offsets;
    int score_stride;
};
glm53f_sparse_prefill_workspace_12n *glm53f_sparse_prefill_workspace_create_12n(void) {
    glm53f_sparse_prefill_workspace_12n *w = a256(sizeof(*w));
    memset(w, 0, sizeof(*w));
    return w;
}
void glm53f_sparse_prefill_workspace_free_12n(glm53f_sparse_prefill_workspace_12n *w) {
    if (!w) return;
    free(w->score_offsets); free(w->score_packed_global); free(w->score_packed_local);
    free(w->score_global); free(w->score_local);
    free(w);
}
static int sparse_select_prefill(glm53f_sparse_context_12n *c,
        glm53f_sparse_prefill_workspace_12n *w, int base, int tokens) {
    int pools = (base + tokens) / KPOOL;
    if (w->score_stride < pools || !w->score_stride) {
        int cap = 512;
        while (cap < pools) cap *= 2;
        size_t bytes = (size_t)GLM53F_PREFILL_ATTN_TOKENS * cap * sizeof(float);
        float *local = malloc(bytes), *global = malloc(bytes);
        if (!local || !global) { free(local); free(global); return -1; }
        free(w->score_local); free(w->score_global);
        w->score_local = local; w->score_global = global; w->score_stride = cap;
        size_t packed_bytes = (size_t)GLM53F_PREFILL_ATTN_TOKENS * cap * sizeof(float);
        float *packed_local = malloc(packed_bytes), *packed_global = malloc(packed_bytes);
        int *offsets = malloc((size_t)(GLM53F_PREFILL_ATTN_TOKENS + 1) * sizeof(int));
        if (!packed_local || !packed_global || !offsets) {
            free(packed_local); free(packed_global); free(offsets); return -1;
        }
        free(w->score_packed_local); free(w->score_packed_global); free(w->score_offsets);
        w->score_packed_local = packed_local; w->score_packed_global = packed_global;
        w->score_offsets = offsets;
    }
    int score_columns = pools;
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int p = base / KPOOL; p < pools; ++p) update_completed_pool(c, p);
#pragma omp for schedule(static)
        for (int t = 0; t < tokens; ++t)
            if (base + t + 1 <= TOPK + KPOOL - 1)
                glm53f_index_query_transpose(w->qt[t], w->iq + (size_t)t * IH * ID);
#pragma omp for collapse(2) schedule(static)
        for (int t = 0; t < tokens; ++t)
            for (int p = 0; p < score_columns; ++p) {
                int length = base + t + 1;
                float score = 0;
                if (p < length / KPOOL) {
                    const float *key = c->pool + (size_t)p * ID;
                    const float *hw = w->iw + (size_t)t * IH;
                    if (length <= TOPK + KPOOL - 1)
                        score = glm53f_index_score_transposed(w->qt[t], hw, key);
                    else if (p % c->ranks == c->rank) {
                        const float *q = w->iq + (size_t)t * IH * ID;
                        /* Long-context baseline uses FP32 SVE dots, not the
                         * short-context FP64 index arithmetic. Keep both. */
                        for (int h = 0; h < IH; ++h) {
                            float dot = f32dot(q + (size_t)h * ID, key, ID);
                            if (dot > 0) score += hw[h] * dot / sqrtf((float)(ID * IH));
                        }
                    }
                }
                w->score_local[(size_t)t * w->score_stride + p] = score;
            }
    }
    if (c->prefill.features & GLM53F_PREFILL_COMM) {
        int first = TOPK + KPOOL - 1 - base;
        if (first < 0) first = 0;
        if (first < tokens) {
            int total = 0;
            for (int t = first; t < tokens; ++t) {
                w->score_offsets[t] = total;
                total += (base + t + 1) / KPOOL;
                memcpy(w->score_packed_local + w->score_offsets[t],
                       w->score_local + (size_t)t * w->score_stride,
                       (size_t)((base + t + 1) / KPOOL) * sizeof(float));
            }
            w->score_offsets[tokens] = total;
            if (sp_ar_prefill(w->score_packed_local,
                    w->score_packed_global, total)) return -1;
        }
    } else for (int t = 0; t < tokens; ++t) {
        int length = base + t + 1;
        if (length > TOPK + KPOOL - 1 &&
            sp_ar(w->score_local + (size_t)t * w->score_stride,
                w->score_global + (size_t)t * w->score_stride, length / KPOOL)) return -1;
    }
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t) {
        int length = base + t + 1, np = length / KPOOL, nc = np;
        if (nc > TOPK / KPOOL) nc = TOPK / KPOOL;
        const float *scores;
        if (length <= TOPK + KPOOL - 1) scores = w->score_local + (size_t)t * w->score_stride;
        else if (c->prefill.features & GLM53F_PREFILL_COMM)
            scores = w->score_packed_global + w->score_offsets[t];
        else scores = w->score_global + (size_t)t * w->score_stride;
        pool_top_exact(w->top[t], scores, np, nc);
        int count = 0;
        for (int p = 0; p < nc; ++p)
            for (int z = 0; z < KPOOL; ++z) w->selected[t][count++] = w->top[t][p].id * KPOOL + z;
        for (int p = np * KPOOL; p < length; ++p) w->selected[t][count++] = p;
        w->count[t] = count;
    }
    memcpy(c->iq, w->iq + (size_t)(tokens - 1) * IH * ID, IH * ID * sizeof(float));
    memcpy(c->iw, w->iw + (size_t)(tokens - 1) * IH, IH * sizeof(float));
    memcpy(c->selected, w->selected[tokens - 1], (size_t)w->count[tokens - 1] * sizeof(int));
    return 0;
}
static void sparse_mv_fp8_wide(const glm53f_prefill_config *config,
        float *out, const uint8_t *weight,
        const float *scale, const float *x, int tokens, int rows, int cols) {
    if (tokens > 5 && (config->features & GLM53F_PREFILL_GEMM)) {
#pragma omp parallel
        glm53f_prefill_gemm_team(out, weight, scale, x, tokens, rows, cols, 1, config->gemm_arena);
        return;
    }
#pragma omp parallel for collapse(2) schedule(static)
    for (int r = 0; r < rows; r += 4)
        for (int t = 0; t < tokens; t += 4) {
            int n = tokens - t;
            if (n > 4) n = 4;
            glm53f_matvec_fp8_bits_4x4(out + (size_t)t * rows + r, rows,
                weight + (size_t)r * cols, scale + (size_t)(r / 128) * (cols / 128),
                x + (size_t)t * cols, n, cols);
        }
}
static void sparse_mv_bf16_wide(const glm53f_prefill_config *config,
        float *out, const uint16_t *weight,
        const float *x, int tokens, int rows, int cols) {
    if (tokens > 5 && rows >= 48 && (config->features & GLM53F_PREFILL_GEMM)) {
#pragma omp parallel
        glm53f_prefill_gemm_team(out, weight, NULL, x, tokens, rows, cols, 0, config->gemm_arena);
        return;
    }
#pragma omp parallel for collapse(2) schedule(static)
    for (int r = 0; r < rows; r += 4)
        for (int t = 0; t < tokens; t += 4) {
            int n = tokens - t;
            if (n > 4) n = 4;
            glm53f_matvec_bf16_4x4(out + (size_t)t * rows + r, rows,
                weight + (size_t)r * cols, x + (size_t)t * cols, n, cols);
        }
}

/* ---- batched native MLA for prefill ---------------------------------------------------------------
 * Bit-identical to mla_heads_q8_value (same per-dot lane-wise FMA chains, same F16 rounding of the latent rows,
 * same softmax and v_b path), but register-blocked so that independent chains hide FMA latency: the logits use
 * 4 keys x heads accumulators sharing every loaded latent vector, the values use heads x 4 vectors, and the
 * absorb step reuses each bf16 weight vector for up to six tokens.  All heads of a token are handled by one
 * thread; the 32-token attention micro-batch is spread over the team. */
static inline svfloat32_t mlb_r16(svbool_t p, svfloat32_t v) {
    return svcvt_f32_f16_x(p, svcvt_f16_f32_x(p, v));
}
#define MLB_SLOTS (TOPK + KPOOL)

static inline __attribute__((always_inline)) void mlb_absorb(float *ql, const float *qs, const uint16_t *wk,
        int hn, int h, int d0, int g, const int n, int tokens) {
    (void)tokens;
    const svbool_t pg = svptrue_b32();
#define MLB_A_DECL(U) svfloat32_t a##U##0 = svdup_f32(0), a##U##1 = a##U##0, a##U##2 = a##U##0, a##U##3 = a##U##0
    MLB_A_DECL(0); MLB_A_DECL(1); MLB_A_DECL(2); MLB_A_DECL(3); MLB_A_DECL(4); MLB_A_DECL(5);
#undef MLB_A_DECL
    for (int j = 0; j < KD; ++j) {
        const uint16_t *wp = wk + (size_t)j * LAT + d0;
        const svfloat32_t w0 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wp), 16));
        const svfloat32_t w1 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wp + 16), 16));
        const svfloat32_t w2 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wp + 32), 16));
        const svfloat32_t w3 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wp + 48), 16));
#define MLB_A_STEP(U) if ((U) < n) { const float x = qs[((size_t)(g + (U)) * hn + h) * KD + j]; \
        a##U##0 = svmla_n_f32_x(pg, a##U##0, w0, x); a##U##1 = svmla_n_f32_x(pg, a##U##1, w1, x); \
        a##U##2 = svmla_n_f32_x(pg, a##U##2, w2, x); a##U##3 = svmla_n_f32_x(pg, a##U##3, w3, x); }
        MLB_A_STEP(0) MLB_A_STEP(1) MLB_A_STEP(2) MLB_A_STEP(3) MLB_A_STEP(4) MLB_A_STEP(5)
#undef MLB_A_STEP
    }
#define MLB_A_ST(U) if ((U) < n) { float *o = ql + ((size_t)(g + (U)) * hn + h) * LAT + d0; \
        svst1(pg, o, a##U##0); svst1(pg, o + 16, a##U##1); svst1(pg, o + 32, a##U##2); svst1(pg, o + 48, a##U##3); }
    MLB_A_ST(0) MLB_A_ST(1) MLB_A_ST(2) MLB_A_ST(3) MLB_A_ST(4) MLB_A_ST(5)
#undef MLB_A_ST
}

/* logits[h][t] = f32dot(ql[h], round16(cache[sel[t]])) for NH heads, 4 keys at a time. */
static inline __attribute__((always_inline)) void mlb_logits(float *lg, const float *ql, const float *cache,
        const int *sel, int nt, const int NH) {
    const svbool_t pg = svptrue_b32();
    int t = 0;
    for (; t + 4 <= nt; t += 4) {
        const float *z0 = cache + (size_t)sel[t] * LAT, *z1 = cache + (size_t)sel[t + 1] * LAT;
        const float *z2 = cache + (size_t)sel[t + 2] * LAT, *z3 = cache + (size_t)sel[t + 3] * LAT;
#define MLB_L_DECL(H) svfloat32_t l0_##H = svdup_f32(0), l1_##H = l0_##H, l2_##H = l0_##H, l3_##H = l0_##H
        MLB_L_DECL(0); MLB_L_DECL(1); MLB_L_DECL(2); MLB_L_DECL(3); MLB_L_DECL(4); MLB_L_DECL(5);
#undef MLB_L_DECL
        for (int d = 0; d < LAT; d += 16) {
            const svfloat32_t v0 = mlb_r16(pg, svld1(pg, z0 + d)), v1 = mlb_r16(pg, svld1(pg, z1 + d));
            const svfloat32_t v2 = mlb_r16(pg, svld1(pg, z2 + d)), v3 = mlb_r16(pg, svld1(pg, z3 + d));
#define MLB_L_STEP(H) if ((H) < NH) { const svfloat32_t q = svld1(pg, ql + (size_t)(H) * LAT + d); \
            l0_##H = svmla_f32_x(pg, l0_##H, q, v0); l1_##H = svmla_f32_x(pg, l1_##H, q, v1); \
            l2_##H = svmla_f32_x(pg, l2_##H, q, v2); l3_##H = svmla_f32_x(pg, l3_##H, q, v3); }
            MLB_L_STEP(0) MLB_L_STEP(1) MLB_L_STEP(2) MLB_L_STEP(3) MLB_L_STEP(4) MLB_L_STEP(5)
#undef MLB_L_STEP
        }
#define MLB_L_ST(H) if ((H) < NH) { float *o = lg + (size_t)(H) * MLB_SLOTS + t; \
        o[0] = svaddv_f32(pg, l0_##H); o[1] = svaddv_f32(pg, l1_##H); o[2] = svaddv_f32(pg, l2_##H); o[3] = svaddv_f32(pg, l3_##H); }
        MLB_L_ST(0) MLB_L_ST(1) MLB_L_ST(2) MLB_L_ST(3) MLB_L_ST(4) MLB_L_ST(5)
#undef MLB_L_ST
    }
    for (; t < nt; ++t) {
        const float *z0 = cache + (size_t)sel[t] * LAT;
#define MLB_T_DECL(H) svfloat32_t s##H = svdup_f32(0)
        MLB_T_DECL(0); MLB_T_DECL(1); MLB_T_DECL(2); MLB_T_DECL(3); MLB_T_DECL(4); MLB_T_DECL(5);
#undef MLB_T_DECL
        for (int d = 0; d < LAT; d += 16) {
            const svfloat32_t v0 = mlb_r16(pg, svld1(pg, z0 + d));
#define MLB_T_STEP(H) if ((H) < NH) s##H = svmla_f32_x(pg, s##H, svld1(pg, ql + (size_t)(H) * LAT + d), v0);
            MLB_T_STEP(0) MLB_T_STEP(1) MLB_T_STEP(2) MLB_T_STEP(3) MLB_T_STEP(4) MLB_T_STEP(5)
#undef MLB_T_STEP
        }
#define MLB_T_ST(H) if ((H) < NH) lg[(size_t)(H) * MLB_SLOTS + t] = svaddv_f32(pg, s##H);
        MLB_T_ST(0) MLB_T_ST(1) MLB_T_ST(2) MLB_T_ST(3) MLB_T_ST(4) MLB_T_ST(5)
#undef MLB_T_ST
    }
}

/* va[h][d] = sum_t p[h][t] * round16(cache[sel[t]][d]) in key order, 64 columns at a time. va rows: va + h*va_stride. */
static inline __attribute__((always_inline)) void mlb_values(float *va, size_t va_stride, const float *lg,
        const float *cache, const int *sel, int nt, const int NH) {
    const svbool_t pg = svptrue_b32();
    for (int db = 0; db < LAT; db += 64) {
#define MLB_V_DECL(H) svfloat32_t v##H##0 = svdup_f32(0), v##H##1 = v##H##0, v##H##2 = v##H##0, v##H##3 = v##H##0
        MLB_V_DECL(0); MLB_V_DECL(1); MLB_V_DECL(2); MLB_V_DECL(3); MLB_V_DECL(4); MLB_V_DECL(5);
#undef MLB_V_DECL
        for (int t = 0; t < nt; ++t) {
            const float *z = cache + (size_t)sel[t] * LAT + db;
            const svfloat32_t z0 = mlb_r16(pg, svld1(pg, z)), z1 = mlb_r16(pg, svld1(pg, z + 16));
            const svfloat32_t z2 = mlb_r16(pg, svld1(pg, z + 32)), z3 = mlb_r16(pg, svld1(pg, z + 48));
#define MLB_V_STEP(H) if ((H) < NH) { const float x = lg[(size_t)(H) * MLB_SLOTS + t]; \
            v##H##0 = svmla_n_f32_x(pg, v##H##0, z0, x); v##H##1 = svmla_n_f32_x(pg, v##H##1, z1, x); \
            v##H##2 = svmla_n_f32_x(pg, v##H##2, z2, x); v##H##3 = svmla_n_f32_x(pg, v##H##3, z3, x); }
            MLB_V_STEP(0) MLB_V_STEP(1) MLB_V_STEP(2) MLB_V_STEP(3) MLB_V_STEP(4) MLB_V_STEP(5)
#undef MLB_V_STEP
        }
#define MLB_V_ST(H) if ((H) < NH) { float *o = va + (size_t)(H) * va_stride + db; \
        svst1(pg, o, v##H##0); svst1(pg, o + 16, v##H##1); svst1(pg, o + 32, v##H##2); svst1(pg, o + 48, v##H##3); }
        MLB_V_ST(0) MLB_V_ST(1) MLB_V_ST(2) MLB_V_ST(3) MLB_V_ST(4) MLB_V_ST(5)
#undef MLB_V_ST
    }
}

static void mlb_token(float *va, size_t va_stride, float *lg, const float *ql, const float *cache,
        const int *sel, int nt, int NH) {
    switch (NH) {
#define MLB_CASE(N) case N: mlb_logits(lg, ql, cache, sel, nt, N); break;
    MLB_CASE(1) MLB_CASE(2) MLB_CASE(3) MLB_CASE(4) MLB_CASE(5) MLB_CASE(6)
#undef MLB_CASE
    }
    for (int h = 0; h < NH; ++h) {
        float *l = lg + (size_t)h * MLB_SLOTS;
        const svbool_t pt = svptrue_b32();
        svfloat32_t vmx = svdup_f32(-INFINITY);
        int t = 0;
        for (; t + 16 <= nt; t += 16) vmx = svmax_f32_x(pt, vmx, svld1_f32(pt, l + t));
        float mx = svmaxv_f32(pt, vmx);
        for (; t < nt; ++t) if (l[t] > mx) mx = l[t];
        /* vector exp (rel. error ~1e-7) replaces 12k scalar expf calls per token */
        svfloat32_t vsum = svdup_f32(0);
        for (t = 0; t < nt; t += 16) {
            const svbool_t p = svwhilelt_b32(t, nt);
            svfloat32_t e = gmn_expf(p, svsub_n_f32_x(p, svld1_f32(p, l + t), mx));
            svst1_f32(p, l + t, e);
            vsum = svadd_f32_m(p, vsum, e);
        }
        const float sum = svaddv_f32(pt, vsum);
        for (t = 0; t < nt; t += 16) {
            const svbool_t p = svwhilelt_b32(t, nt);
            svst1_f32(p, l + t, svdiv_n_f32_x(p, svld1_f32(p, l + t), sum));
        }
    }
    switch (NH) {
#define MLB_CASE(N) case N: mlb_values(va, va_stride, lg, cache, sel, nt, N); break;
    MLB_CASE(1) MLB_CASE(2) MLB_CASE(3) MLB_CASE(4) MLB_CASE(5) MLB_CASE(6)
#undef MLB_CASE
    }
}

/* Returns 0 on success, -2 when unsupported (caller falls back), -1 on error. */
static int mla_native_batch(glm53f_sparse_context_12n *c, glm53f_sparse_prefill_workspace_12n *w, int tokens) {
    enum { TCAP = GLM53F_PREFILL_ATTN_TOKENS };
    const int hn = c->hn, cols = hn * VD;
    if ((int)svcntw() != 16 || hn < 1 || hn > 6 || tokens < 1 || tokens > TCAP || !c->q2_native) return -2;
    if (!c->mlb_qs) {
        c->mlb_threads = omp_get_max_threads();
        c->mlb_qs = a256((size_t)TCAP * hn * KD * sizeof(float));
        c->mlb_ql = a256((size_t)TCAP * hn * LAT * sizeof(float));
        c->mlb_va = a256((size_t)hn * TCAP * LAT * sizeof(float));
        c->mlb_out = a256((size_t)hn * TCAP * VD * sizeof(float));
        c->mlb_lg = a256((size_t)c->mlb_threads * 8 * MLB_SLOTS * sizeof(float));
        c->mlb_act_bytes = (glm53f_native_act_bytes(LAT) + 255) & ~(size_t)255;
        c->mlb_act = a256((size_t)hn * TCAP * c->mlb_act_bytes);
    }
    const size_t row_bytes = glm53f_native_row_size(c->q2_vb_type, LAT);
    float *qs = c->mlb_qs, *ql = c->mlb_ql, *va = c->mlb_va, *out = c->mlb_out;
    const size_t va_stride = (size_t)tokens * LAT;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < tokens * hn * KD; ++i)
        qs[i] = w->query[(size_t)(i / (hn * KD)) * c->qd + (i % (hn * KD))] / sqrtf((float)KD);
    const int ng = (tokens + 5) / 6;
#pragma omp parallel for schedule(static)
    for (int item = 0; item < hn * 8 * ng; ++item) {
        const int h = item / (8 * ng), d0 = ((item / ng) % 8) * 64, g = (item % ng) * 6;
        const int n = tokens - g < 6 ? tokens - g : 6;
        const uint16_t *wk = c->kvb + (size_t)h * (KD + VD) * LAT;
        switch (n) {
#define MLB_CASE(N) case N: mlb_absorb(ql, qs, wk, hn, h, d0, g, N, tokens); break;
        MLB_CASE(1) MLB_CASE(2) MLB_CASE(3) MLB_CASE(4) MLB_CASE(5) MLB_CASE(6)
#undef MLB_CASE
        }
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < tokens; ++t) {
        float *lg = c->mlb_lg + (size_t)omp_get_thread_num() * 8 * MLB_SLOTS;
        /* va is head-major [h][t][LAT]: hand mlb_values the token's row of head 0 and a head stride of tokens*LAT. */
        mlb_token(va + (size_t)t * LAT, va_stride, lg, ql + (size_t)t * hn * LAT, c->latent, w->selected[t], w->count[t], hn);
    }
    {
        /* Prepare the activations with the exact per-head function of the reference path (tie-breaking in the
         * Q8 quantizer differs in glm53f_native_matvec_batch), then run the pre-prepared batch matvec. */
        const int q80 = c->q2_vb_type == GLM53F_GGML_Q8_0 || c->q2_vb_type == GLM53F_NATIVE_Q8_0R;
        int bad = 0;
#pragma omp parallel reduction(|:bad)
        {
            for (int h = 0; h < hn; ++h) {
#pragma omp for schedule(static)
                for (int t = 0; t < tokens; ++t)
                    if (glm53f_native_act_prepare(c->mlb_act + ((size_t)h * tokens + t) * c->mlb_act_bytes,
                            va + (size_t)h * va_stride + (size_t)t * LAT, LAT, !q80, q80)) bad = 1;
                glm53f_native_matrix m = {out + (size_t)h * tokens * VD, c->q2_vb + (size_t)h * VD * row_bytes, c->q2_vb_type, VD, LAT};
                if (glm53f_native_matvec_batch_team(&m, 1, c->mlb_act + (size_t)h * tokens * c->mlb_act_bytes, c->mlb_act_bytes, tokens)) bad = 1;
            }
        }
        if (bad) return -1;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < tokens * cols; ++i) {
        const int t = i / cols, r = i % cols, h = r / VD, j = r % VD;
        w->attn[(size_t)t * cols + r] = out[((size_t)h * tokens + t) * VD + j];
    }
    return 0;
}

static int mla_native_reference(glm53f_sparse_context_12n *c, glm53f_sparse_prefill_workspace_12n *w, int tokens) {
    const int cols = c->hn * VD;
    for (int t = 0; t < tokens; ++t) {
            const int ns = w->count[t];
#pragma omp parallel for schedule(static)
            for (int i = 0; i < ns; ++i)
                for (int d = 0; d < LAT; ++d)
                    c->packed[(size_t)i * LAT + d] =
                        (float)(_Float16)c->latent[(size_t)w->selected[t][i] * LAT + d];
            if (mla_heads_q8_value(c, w->attn + (size_t)t * cols,
                    w->query + (size_t)t * c->qd, c->packed, NULL, ns)) return -1;
        }
    return 0;
}

static double sp_front_acc[10]; static long sp_front_tokens; static double sp_front_t; static int sp_front_on = -1, sp_front_atexit;
static void sp_front_report(void) { if (sp_front_tokens) fprintf(stderr, "GLM53F_SPARSE_FRONT_DETAIL us/token/layer: qa+kva=%.2f rms_qa=%.2f qb=%.2f rms_kva=%.2f wk=%.2f keyln=%.2f gate=%.2f wqb=%.2f wp=%.2f (tokens=%ld)\n", sp_front_acc[0]*1e6/sp_front_tokens, sp_front_acc[1]*1e6/sp_front_tokens, sp_front_acc[2]*1e6/sp_front_tokens, sp_front_acc[3]*1e6/sp_front_tokens, sp_front_acc[4]*1e6/sp_front_tokens, sp_front_acc[5]*1e6/sp_front_tokens, sp_front_acc[6]*1e6/sp_front_tokens, sp_front_acc[7]*1e6/sp_front_tokens, sp_front_acc[8]*1e6/sp_front_tokens, sp_front_tokens); }
#define SPF(I) do { if (sp_front_on) { double n_ = sparse_clock(c); sp_front_acc[I] += n_ - sp_front_t; sp_front_t = n_; } } while (0)
int glm53f_sparse_prefill_12n(glm53f_sparse_context_12n *c,
        glm53f_sparse_prefill_workspace_12n *w, float *out,
        const float *x, int tokens) {
    if (!c || !w || !out || !x || tokens < 1 || tokens > GLM53F_PREFILL_ATTN_TOKENS ||
        c->length > c->capacity - tokens) return -1;
    /* CP owns distributed caches and its own exact selector/exchange. Retain
     * that path at long capacities; never allocate replicated wide caches. */
    if (c->cp || c->int8_enabled || getenv("GLM53F_SPARSE_REFERENCE") ||
        getenv("GLM53F_SPARSE_SCALAR") || glm53f_sparse_scalar_reference) {
        for (int t = 0; t < tokens; t += 4) {
            int n = tokens - t;
            if (n > 4) n = 4;
            if (glm53f_sparse_sublayer_batch_12n(c, out + (size_t)t * H, x + (size_t)t * H, n))
                return -1;
        }
        return 0;
    }
    int base = c->length, cols = c->hn * VD, failed = 0;
    double begin = sparse_clock(c);
    if (sp_front_on < 0) sp_front_on = getenv("GLM53F_SPARSE_FRONT_DETAIL") != NULL;
    if (sp_front_on) { if (!sp_front_atexit) { sp_front_atexit = 1; atexit(sp_front_report); } sp_front_tokens += tokens; sp_front_t = sparse_clock(c); }
    glm53f_prefill_config projection_config = c->prefill;
    /* Native prefill retains decode's F32 accumulation for the compact
     * indexer too; its BF16 GEMM recipe rounds activations differently. */
    if (c->q2_native) projection_config.features &= ~GLM53F_PREFILL_GEMM;
    const int sg = c->q2_native && sparse_gemm_setup(c);
    if (sg) {
        sparse_gemm_run(c->sg_w1, H, QA + LAT, x, tokens, c->sg_xq, c->sg_xs, c->sg_xp, c->sg_xsp, c->sg_bt, c->sg_y1);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < tokens; ++t) {
            memcpy(w->qres + (size_t)t * QA, c->sg_y1 + (size_t)t * (QA + LAT), QA * sizeof(float));
            memcpy(c->latent + (size_t)(base + t) * LAT, c->sg_y1 + (size_t)t * (QA + LAT) + QA, LAT * sizeof(float));
        }
    } else if (c->q2_native) {
        glm53f_native_matrix front[2] = {
            {w->qres,c->q2_qa,c->q2_qa_type,QA,H},
            {c->latent+(size_t)base*LAT,c->q2_kva,c->q2_kva_type,LAT,H}};
        if (glm53f_native_matvec_batch(front, 2, x, tokens)) return -1;
    } else sparse_mv_fp8_wide(&c->prefill, w->qres, c->qa, c->qas, x, tokens, QA, H);
    SPF(0); /* q_a + kv_a */
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t)
        glm53f_rmsnorm_bf16(w->qres + (size_t)t * QA, w->qres + (size_t)t * QA, c->qan, QA, 1e-5f);
    SPF(1); /* rmsnorm q_a */
    if (sg) {
        sparse_gemm_run(c->sg_wqb, QA, c->qd, w->qres, tokens, c->sg_qq, c->sg_qs, c->sg_qp, c->sg_qsp, c->sg_bt, c->sg_yq);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < tokens; ++t) memcpy(w->query + (size_t)t * c->qd, c->sg_yq + (size_t)t * c->qd, c->qd * sizeof(float));
    } else if (c->q2_native) {
        glm53f_native_matrix qb = {w->query,c->q2_qb,c->q2_qb_type,c->qd,QA};
        if (glm53f_native_matvec_batch(&qb, 1, w->qres, tokens)) return -1;
    } else {
        sparse_mv_fp8_wide(&c->prefill, w->query, c->qb, c->qbs, w->qres, tokens, c->qd, QA);
        sparse_mv_fp8_wide(&c->prefill, c->latent + (size_t)base * LAT, c->kva, c->kvas, x, tokens, LAT, H);
    }
    SPF(2); /* q_b */
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t) {
        float *latent = c->latent + (size_t)(base + t) * LAT;
        glm53f_rmsnorm_bf16(latent, latent, c->kvan, LAT, 1e-5f);
    }
    SPF(3); /* rmsnorm kv_a */
    sparse_mv_bf16_wide(&projection_config, w->raw, c->wk, x, tokens, ID, H);
    SPF(4); /* wk */
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t)
        glm53f_layernorm_bf16(c->key + (size_t)(base + t) * ID,
            w->raw + (size_t)t * ID, c->knw, c->knb, ID, 1e-6f);
    SPF(5); /* key layernorm */
    sparse_mv_bf16_wide(&projection_config, c->gcache + (size_t)base * ID, c->gatew, x, tokens, ID, H);
    SPF(6); /* gate */
    sparse_mv_bf16_wide(&projection_config, w->iq, c->wqb, w->qres, tokens, IH * ID, QA);
    SPF(7); /* wqb */
    sparse_mv_bf16_wide(&projection_config, w->iw, c->wp, x, tokens, IH, H);
    SPF(8); /* wp */
    c->profile_phase[0] += sparse_clock(c) - begin;
    begin = sparse_clock(c);
    /* Future cache rows may be materialized, but select_incremental bounds
     * both completed pools and the raw tail by this query's own position. */
    if (getenv("GLM53F_SPARSE_INDEX_BATCH") && atoi(getenv("GLM53F_SPARSE_INDEX_BATCH"))) {
        if (sparse_select_prefill(c, w, base, tokens)) return -1;
    } else for (int t = 0; t < tokens; ++t) {
        int length = base + t + 1;
        memcpy(c->iq, w->iq + (size_t)t * IH * ID, IH * ID * sizeof(float));
        memcpy(c->iw, w->iw + (size_t)t * IH, IH * sizeof(float));
        if (length % KPOOL == 0) update_completed_pool(c, length / KPOOL - 1);
        w->count[t] = select_incremental(c, length);
        if (w->count[t] < 1) return -1;
        memcpy(w->selected[t], c->selected, (size_t)w->count[t] * sizeof(int));
    }
    c->profile_phase[1] += sparse_clock(c) - begin;
    begin = sparse_clock(c);
    int sharded = !getenv("GLM53F_SPARSE_NO_PACK");
    int mlb_mode = getenv("GLM53F_SPARSE_MLA_BATCH") ? atoi(getenv("GLM53F_SPARSE_MLA_BATCH")) : 1, mlb_done = 0;
    if (c->q2_native && mlb_mode && tokens >= 2) {
        int rc = mla_native_batch(c, w, tokens);
        if (rc == -1) return -1;
        mlb_done = rc == 0;
        if (mlb_done && mlb_mode == 2) {
            if (!c->mlb_ref) c->mlb_ref = a256((size_t)GLM53F_PREFILL_ATTN_TOKENS * cols * sizeof(float));
            memcpy(c->mlb_ref, w->attn, (size_t)tokens * cols * sizeof(float));
            if (mla_native_batch(c, w, tokens) == 0) { /* self-consistency: a second run must be bitwise equal */
                long bad2 = 0, total2 = (long)tokens * cols;
                for (long i = 0; i < total2; ++i) bad2 += memcmp(&c->mlb_ref[i], &w->attn[i], 4) != 0;
                if (!c->rank && bad2) fprintf(stderr, "GLM53F_SPARSE_MLA_SELF layer=%d nondeterministic_floats=%ld\n", c->layer, bad2);
            }
            mlb_done = 0; /* recompute with the reference path, then compare bitwise */
        }
    }
    if (c->q2_native) {
      if (!mlb_done) {
        /* Selection and the F16 cache view are bounded by each query's
         * position even though projections materialized future cache rows. */
        if (mla_native_reference(c, w, tokens)) return -1;
        if (mlb_mode == 2) { /* determinism of the reference path itself */
            static float *first = NULL; static size_t cap = 0;
            const size_t need = (size_t)tokens * cols;
            if (cap < need) { free(first); first = malloc(need * sizeof(float)); cap = need; }
            memcpy(first, w->attn, need * sizeof(float));
            if (mla_native_reference(c, w, tokens)) return -1;
            long bad3 = 0;
            for (size_t i = 0; i < need; ++i) bad3 += memcmp(&first[i], &w->attn[i], 4) != 0;
            if (!c->rank && bad3) fprintf(stderr, "GLM53F_SPARSE_MLA_REFSELF layer=%d nondeterministic_floats=%ld\n", c->layer, bad3);
        }
        if (mlb_mode == 2 && c->mlb_ref) {
            long bad = 0, total = (long)tokens * cols;
            for (long i = 0; i < total; ++i) bad += memcmp(&c->mlb_ref[i], &w->attn[i], 4) != 0;
            if (!c->rank) fprintf(stderr, "GLM53F_SPARSE_MLA_VERIFY layer=%d tokens=%d mismatching_floats=%ld of %ld\n", c->layer, tokens, bad, total);
            if (!c->rank && bad) {
                int shown = 0; const int hn_ = c->hn;
                for (long i = 0; i < total && shown < 3; i += VD) {
                    long cnt = 0;
                    for (int j = 0; j < VD; ++j) cnt += memcmp(&c->mlb_ref[i + j], &w->attn[i + j], 4) != 0;
                    if (cnt) {
                        int t = (int)(i / cols), h = (int)((i % cols) / VD);
                        fprintf(stderr, "GLM53F_SPARSE_MLA_VERIFY_DETAIL t=%d h=%d nt=%d base=%d differing=%ld new0=%.9g old0=%.9g new1=%.9g old1=%.9g\n",
                                t, h, w->count[t], base, cnt, c->mlb_ref[i], w->attn[i], c->mlb_ref[i + 1], w->attn[i + 1]);
                        if (shown == 0) { /* double-precision reference of the batched path's intermediates */
                            const int nt = w->count[t];
                            const uint16_t *wk = c->kvb + (size_t)h * (KD + VD) * LAT;
                            double *qlr = malloc(LAT * sizeof(double)), *pr = malloc((size_t)nt * sizeof(double)), *var = calloc(LAT, sizeof(double));
                            const float *qq = w->query + (size_t)t * c->qd + h * KD;
                            for (int d = 0; d < LAT; ++d) {
                                double a = 0;
                                for (int j = 0; j < KD; ++j) {
                                    uint32_t bits = (uint32_t)wk[(size_t)j * LAT + d] << 16; float wf; memcpy(&wf, &bits, 4);
                                    a += (double)(qq[j] / sqrtf((float)KD)) * wf;
                                }
                                qlr[d] = a;
                            }
                            double mx = -1e300, sm = 0;
                            for (int i = 0; i < nt; ++i) {
                                double a = 0;
                                for (int d = 0; d < LAT; ++d) a += qlr[d] * (double)(float)(_Float16)c->latent[(size_t)w->selected[t][i] * LAT + d];
                                pr[i] = a; if (a > mx) mx = a;
                            }
                            for (int i = 0; i < nt; ++i) { pr[i] = exp(pr[i] - mx); sm += pr[i]; }
                            for (int i = 0; i < nt; ++i)
                                for (int d = 0; d < LAT; ++d) var[d] += pr[i] / sm * (double)(float)(_Float16)c->latent[(size_t)w->selected[t][i] * LAT + d];
                            double eq = 0, sq = 0, ev = 0, sv = 0;
                            const float *qln = c->mlb_ql + ((size_t)t * hn_ + h) * LAT, *van = c->mlb_va + ((size_t)h * tokens + t) * LAT;
                            for (int d = 0; d < LAT; ++d) { eq += (qln[d] - qlr[d]) * (qln[d] - qlr[d]); sq += qlr[d] * qlr[d]; ev += (van[d] - var[d]) * (van[d] - var[d]); sv += var[d] * var[d]; }
                            /* the reference path's own intermediate for this token (c->q8v_va after the call) */
                            double eo = 0;
                            {
                                const int ns = nt;
                                for (int i = 0; i < ns; ++i)
                                    for (int d = 0; d < LAT; ++d)
                                        c->packed[(size_t)i * LAT + d] = (float)(_Float16)c->latent[(size_t)w->selected[t][i] * LAT + d];
                                float tmp[8 * VD];
                                mla_heads_q8_value(c, tmp, w->query + (size_t)t * c->qd, c->packed, NULL, ns);
                                const float *vao = c->q8v_va + (size_t)h * LAT, *qlo = c->q8v_ql + (size_t)h * LAT;
                                double eql = 0;
                                for (int d = 0; d < LAT; ++d) { eo += (vao[d] - var[d]) * (vao[d] - var[d]); eql += (qlo[d] - qlr[d]) * (qlo[d] - qlr[d]); }
                                fprintf(stderr, "GLM53F_SPARSE_MLA_REF_OLD t=%d h=%d old_ql_rel=%.3e old_va_rel=%.3e\n", t, h, sqrt(eql / (sq + 1e-300)), sqrt(eo / (sv + 1e-300)));
                            }
                            fprintf(stderr, "GLM53F_SPARSE_MLA_REF t=%d h=%d nt=%d new_ql_rel=%.3e new_va_rel=%.3e\n", t, h, nt, sqrt(eq / (sq + 1e-300)), sqrt(ev / (sv + 1e-300)));
                            free(qlr); free(pr); free(var);
                        }
                        ++shown;
                    }
                }
            }
        }
      }
    } else {
#pragma omp parallel for collapse(2) schedule(static) reduction(|:failed)
        for (int t = 0; t < tokens; ++t)
            for (int h = 0; h < c->hn; ++h) {
                const uint16_t *weight = c->kvb + (size_t)h * (KD + VD) * LAT;
                const float *q = w->query + (size_t)t * c->qd + h * KD;
                float *a = w->attn + (size_t)t * cols + h * VD;
                if (tokens > 5 && (c->prefill.features & GLM53F_PREFILL_MLA_REG) && svcntw() == 16)
                    failed |= glm53f_mla_prefill_one(a, q, c->latent, weight,
                        w->selected[t], w->count[t], base + t + 1 > TOPK + KPOOL - 1 && sharded ? 8 : 1) != 0;
                else if (base + t + 1 > TOPK + KPOOL - 1 && sharded)
                    failed |= mla_one_sharded_indexed(a, q, c->latent, weight, w->selected[t], w->count[t]) != 0;
                else failed |= mla_one(a, q, c->latent, weight, w->selected[t], w->count[t]) != 0;
            }
    }
    c->profile_phase[3] += sparse_clock(c) - begin;
    if (failed) return -1;
    c->length += tokens;
    begin = sparse_clock(c);
    if (sg && c->sg_wop) {
        sparse_gemm_run(c->sg_wop, cols, H, w->attn, tokens, c->sg_oq, c->sg_os, c->sg_op, c->sg_osp, c->sg_bt, c->sg_yo);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < tokens; ++t) memcpy(w->partial + (size_t)t * H, c->sg_yo + (size_t)t * H, H * sizeof(float));
    } else if (c->q2_native) {
        glm53f_native_matrix op = {w->partial,c->q2_op,c->q2_op_type,H,cols};
        if (glm53f_native_matvec_batch(&op, 1, w->attn, tokens)) return -1;
    } else sparse_mv_fp8_wide(&c->prefill, w->partial, c->op, c->ops, w->attn, tokens, H, cols);
    c->profile_phase[4] += sparse_clock(c) - begin;
    if (sp_defer_reduce) { memcpy(out, w->partial, (size_t)tokens * H * sizeof(float)); return 0; } /* reduced by the caller's helper */
    begin = sparse_clock(c);
    if (glm53f_sum_allreduce_slabs_12n(w->partial, out, tokens, H,
            c->prefill.features & GLM53F_PREFILL_COMM ? c->prefill.slab_tokens : 1)) return -1;
    c->profile_phase[5] += sparse_clock(c) - begin;
    return 0;
}
void glm53f_sparse_free_12n(glm53f_sparse_context_12n*c){if(!c)return;free(c->mlb_qs);free(c->mlb_ql);free(c->mlb_va);free(c->mlb_out);free(c->mlb_ref);free(c->mlb_lg);free(c->mlb_act);free(c->q8v_ql);free(c->q8v_va);free(c->q8v_log);free(c->q8v_sum);free(c->q8v_act);for(int i=0;i<4;i++){free(c->int8_scale[i]);free(c->int8_weight[i]);}free(c->q2_op);free(c->q2_vb);free(c->q2_kva);free(c->q2_qb);free(c->q2_qa);free(c->cp_bf16_gather);free(c->cp_bf16_local);free(c->hot_latent);free(c->batch_partial);free(c->batch_attn);free(c->packed_index);free(c->packed);free(c->pool_score_global);free(c->pool_score_local);free(c->pool_score_cache);free(c->cp_candidate_gather);free(c->cp_candidate_local);free(c->cp_pack_index);free(c->cp_cur_gate);free(c->cp_cur_key);free(c->cp_cur_latent);free(c->cp_exchange);free(c->cp_gather);free(c->cp_pack);free(c->cp_pool);free(c->cp_gate);free(c->cp_key);free(c->cp_latent);free(c->apef);free(c->selected);free(c->partial);free(c->attn);free(c->pool);free(c->iw);free(c->iq);free(c->gcache);free(c->key);free(c->latent);free(c->query);free(c->qres);free(c->wp);free(c->wqb);free(c->ape);free(c->gatew);free(c->knb);free(c->knw);free(c->wk);free(c->ops);free(c->op);free(c->kvb);free(c->kvan);free(c->kvas);free(c->kva);free(c->qbs);free(c->qb);free(c->qan);free(c->qas);free(c->qa);free(c);}
#ifndef GLM53F_SPARSE_NO_MAIN
int main(int argc,char**argv){
    int rank,nr,layer=argc>3?atoi(argv[3]):43,tokens=argc>2?atoi(argv[2]):512,h0,hn,qd;char n[256];glm53f_st_context*st;
    uint8_t *qa,*qb,*kva,*op;float *qas,*qbs,*kvas,*ops;uint16_t *qan,*kvan,*kvb,*wk,*knw,*knb,*gatew,*ape,*wqb,*wp;
    float *x,*qres,*query,*latent,*key,*gcache,*iq,*iw,*pool,*attn,*partial,*out[2];int*sel;MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);
    if(argc<2||nr!=12||tokens<1||tokens>16384){if(!rank)fprintf(stderr,"usage: mpiexec -np 12 %s MODEL_DIR [tokens=512] [layer=43]\n",argv[0]);MPI_Finalize();return 2;}glm53f_balanced_slice(NH,rank,nr,&h0,&hn);qd=hn*KD;st=glm53f_st_open(argv[1]);if(!st)MPI_Abort(MPI_COMM_WORLD,2);
#define N(S) snprintf(n,sizeof n,"model.language_model.layers.%d.self_attn.%s",layer,S)
    N("q_a_proj.weight");qa=tensor(st,n,rank);N("q_a_proj.weight_scale_inv");qas=tensor(st,n,rank);N("q_a_layernorm.weight");qan=tensor(st,n,rank);
    N("q_b_proj.weight");qb=a256((size_t)qd*QA);readp(st,n,(size_t)h0*KD*QA,qb,(size_t)qd*QA,rank);N("q_b_proj.weight_scale_inv");{int nb=QA/128,tiles=qd/128;qbs=a256((size_t)tiles*nb*4);readp(st,n,(size_t)(h0*KD/128)*nb*4,qbs,(size_t)tiles*nb*4,rank);}
    N("kv_a_proj_with_mqa.weight");kva=tensor(st,n,rank);N("kv_a_proj_with_mqa.weight_scale_inv");kvas=tensor(st,n,rank);N("kv_a_layernorm.weight");kvan=tensor(st,n,rank);
    N("kv_b_proj.weight");kvb=a256((size_t)hn*(KD+VD)*LAT*2);readp(st,n,(size_t)h0*(KD+VD)*LAT*2,kvb,(size_t)hn*(KD+VD)*LAT*2,rank);
    {int oc0=h0*VD,ocn=hn*VD,ob=QKV/128,ob0=oc0/128,obn=ocn/128;
     N("o_proj.weight");op=a256((size_t)H*ocn);if(glm53f_st_read_columns(st,n,QKV,oc0,ocn,op))MPI_Abort(MPI_COMM_WORLD,2);
     N("o_proj.weight_scale_inv");ops=a256((size_t)(H/128)*obn*sizeof(float));if(glm53f_st_read_columns(st,n,(size_t)ob*sizeof(float),(size_t)ob0*sizeof(float),(size_t)obn*sizeof(float),ops))MPI_Abort(MPI_COMM_WORLD,2);}
    N("indexer.wk.weight");wk=tensor(st,n,rank);N("indexer.k_norm.weight");knw=tensor(st,n,rank);N("indexer.k_norm.bias");knb=tensor(st,n,rank);N("indexer.index_kpool_compress_gate");gatew=tensor(st,n,rank);N("indexer.index_kpool_compress_ape");ape=tensor(st,n,rank);N("indexer.wq_b.weight");wqb=tensor(st,n,rank);N("indexer.weights_proj.weight");wp=tensor(st,n,rank);
#undef N
    glm53f_st_close(st);x=a256(H*4);qres=a256(QA*4);query=a256((size_t)qd*4);latent=a256((size_t)tokens*LAT*4);key=a256((size_t)tokens*ID*4);gcache=a256((size_t)tokens*ID*4);iq=a256((size_t)IH*ID*4);iw=a256(IH*4);pool=a256((size_t)(tokens/KPOOL+1)*ID*4);attn=a256((size_t)hn*VD*4);partial=a256(H*4);out[0]=a256(H*4);out[1]=a256(H*4);sel=a256((size_t)(TOPK+KPOOL)*4);
    for(int t=0;t<tokens;t++){for(int d=0;d<LAT;d++)latent[(size_t)t*LAT+d]=(float)(((t*29+d*11+3)%257)-128)/128.0f;for(int d=0;d<ID;d++){key[(size_t)t*ID+d]=(float)(((t*13+d*17+5)%251)-125)/125.0f;gcache[(size_t)t*ID+d]=(float)(((t*7+d*19+1)%127)-63)/63.0f;}}
    for(int i=0;i<H;i++)x[i]=(float)(((i*17+tokens*3+3)%251)-125)/125.0f;float apef[KPOOL*ID];for(int i=0;i<KPOOL*ID;i++)apef[i]=glm53f_bf16_to_f32(ape[i]);float ph[2][4],el[2];int ns[2];int local_cols=hn*VD,local_blocks=local_cols/128;
    /* Build the persistent cache state once. Decode only refreshes the current
     * token and scores already-compressed completed pools. */
    mv_f8(qres,qa,qas,x,QA,H);glm53f_rmsnorm_bf16(qres,qres,qan,QA,1e-5f);mv_f8(latent+(size_t)(tokens-1)*LAT,kva,kvas,x,LAT,H);glm53f_rmsnorm_bf16(latent+(size_t)(tokens-1)*LAT,latent+(size_t)(tokens-1)*LAT,kvan,LAT,1e-5f);{float raw[ID];mv_b16(raw,wk,x,ID,H);glm53f_layernorm_bf16(key+(size_t)(tokens-1)*ID,raw,knw,knb,ID,1e-6f);}mv_b16(gcache+(size_t)(tokens-1)*ID,gatew,x,ID,H);mv_b16(iq,wqb,qres,IH*ID,QA);mv_b16(iw,wp,x,IH,H);glm53f_index_select_decode(pool,sel,iq,iw,key,gcache,apef,tokens,KPOOL,TOPK,IH,ID);
    for(int pass=0;pass<2;pass++){glm53f_sparse_scalar_reference=getenv("GLM53F_SPARSE_LAYER_SCALAR_CHECK")&&pass==1;double t0=MPI_Wtime();mv_f8(qres,qa,qas,x,QA,H);glm53f_rmsnorm_bf16(qres,qres,qan,QA,1e-5f);mv_f8(query,qb,qbs,qres,qd,QA);mv_f8(latent+(size_t)(tokens-1)*LAT,kva,kvas,x,LAT,H);glm53f_rmsnorm_bf16(latent+(size_t)(tokens-1)*LAT,latent+(size_t)(tokens-1)*LAT,kvan,LAT,1e-5f);float raw[ID];mv_b16(raw,wk,x,ID,H);glm53f_layernorm_bf16(key+(size_t)(tokens-1)*ID,raw,knw,knb,ID,1e-6f);mv_b16(gcache+(size_t)(tokens-1)*ID,gatew,x,ID,H);mv_b16(iq,wqb,qres,IH*ID,QA);mv_b16(iw,wp,x,IH,H);ns[pass]=select_cached(pool,sel,iq,iw,tokens);double t1=MPI_Wtime();if(mla_heads(attn,query,latent,kvb,sel,ns[pass],hn))MPI_Abort(MPI_COMM_WORLD,2);double t2=MPI_Wtime();
#pragma omp parallel for schedule(static)
        for(int r=0;r<H;r++)partial[r]=fp8dot(op+(size_t)r*local_cols,ops+(size_t)(r/128)*local_blocks,attn,local_cols);double t3=MPI_Wtime();MPI_Allreduce(partial,out[pass],H,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);double t4=MPI_Wtime();ph[pass][0]=t1-t0;ph[pass][1]=t2-t1;ph[pass][2]=t3-t2;ph[pass][3]=t4-t3;el[pass]=t4-t0;}
    double se=0,sr=0,ma=0;for(int i=0;i<H;i++){double de=(double)out[1][i]-out[0][i];se+=de*de;sr+=(double)out[0][i]*out[0][i];if(fabs(de)>ma)ma=fabs(de);}double rel=sqrt(se/(sr+1e-30));int scalar_check=getenv("GLM53F_SPARSE_LAYER_SCALAR_CHECK")!=NULL;int ok=ns[0]==ns[1]&&(scalar_check?rel<2e-5:!memcmp(out[0],out[1],H*4)),all;MPI_Allreduce(&ok,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);float me,mp[4];MPI_Allreduce(&el[1],&me,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);MPI_Allreduce(ph[1],mp,4,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);double ss=0;for(int i=0;i<H;i++)ss+=(double)out[0][i]*out[0][i];if(!rank)printf("GLM53F_SPARSE_LAYER_12N layer=%d tokens=%d selected=%d max_ms=%.3f front_ms=%.3f mla_ms=%.3f oproj_ms=%.3f ar_ms=%.3f rms=%.9g scalar_rel_l2=%.9g scalar_max_abs=%.9g repeat=%s %s\n",layer,tokens,ns[0],me*1e3f,mp[0]*1e3f,mp[1]*1e3f,mp[2]*1e3f,mp[3]*1e3f,sqrt(ss/H),rel,ma,all?(scalar_check?"SCALAR_PASS":"BIT_EXACT"):"FAIL",all?"PASS":"FAIL");MPI_Finalize();return all?0:1;
}
#endif
