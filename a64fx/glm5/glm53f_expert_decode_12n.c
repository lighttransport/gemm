/* Distributed real-weight GLM-5.3F routed-expert decode benchmark. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef GLM53F_EXTERNAL_ST_IMPLEMENTATION
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#endif
#include <arm_sve.h>
#include <mpi.h>
#include <omp.h>
#ifndef __ARM_FEATURE_SVE
#define __ARM_FEATURE_SVE 1
#endif
#include "glm53f_expert_kern.h"
#include "glm53f_pf_plan.h"
#include "glm53f_moe_12n.h"
#include "glm53f_moe_stage_12n.h"
#include "glm53f_team.h"
#include "glm53f_clock.h"
#include "glm53f_moe_combine.h"
#include "glm53f_collective_12n.h"
#include "glm53f_int8.h"
#include "glm53f_prefill.h"
#include "glm53f_moe_grouped_native.h"
#include "glm53f_bf16_rows.h"
#include "glm53f_cmg_place.h"
#include "kern/glm53f_kern.h"
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"

#include "glm53f_pp_native.h"
#include "glm53f_pp_core.h"
#include "glm53f_pp_routed_manifest.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

enum { FIRST_LAYER = 3, LAST_LAYER = 46, NLAYERS = 43, NEXPERTS = 288 };

typedef struct {
    uint64_t gate_up, gate_up_scale, down, down_scale;
    int inter, gate_type, down_type;
    float *i8_gate_scale, *i8_down_scale;
} expert_offset;
typedef expert_offset shared_offset;

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static long mem_available(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    char key[64], unit[16];
    long kb, result = -1;
    if (!f) return -1;
    while (fscanf(f, "%63s %ld %15s", key, &kb, unit) == 3) {
        if (!strcmp(key, "MemAvailable:")) { result = kb * 1024L; break; }
    }
    fclose(f);
    return result;
}

static int load_manifest(const char *path, expert_offset *table) {
    FILE *f = fopen(path, "r");
    char line[2048], dtype[16], name[1024], suffix[128];
    unsigned long long off;
    int nd, rows, cols, layer, expert, found = 0;
    if (!f) return -1;
    for (int i = 0; i < NLAYERS * NEXPERTS; ++i) {
        table[i].gate_up = table[i].gate_up_scale = UINT64_MAX;
        table[i].down = table[i].down_scale = UINT64_MAX;
        table[i].gate_type = table[i].down_type = 0;
    }
    while (fgets(line, sizeof(line), f)) {
        char *last;
        if (line[0] == '#') continue;
        if (sscanf(line, "%llu %15s %d %d %d", &off, dtype, &nd, &rows, &cols) != 5) continue;
        last = strrchr(line, ' ');
        if (!last) continue;
        snprintf(name, sizeof(name), "%s", last + 1);
        name[strcspn(name, "\r\n")] = 0;
        if (sscanf(name, "model.language_model.layers.%d.mlp.experts.%d.%127s",
                   &layer, &expert, suffix) != 3) continue;
        if (layer < FIRST_LAYER || layer >= LAST_LAYER || expert < 0 || expert >= NEXPERTS) continue;
        expert_offset *p = &table[(layer - FIRST_LAYER) * NEXPERTS + expert];
        int qtype = 0;
        if (!strcmp(dtype, "Q4_K")) qtype = GLM53F_GGML_Q4_K;
        else if (!strcmp(dtype, "Q5_K")) qtype = GLM53F_GGML_Q5_K;
        else if (!strcmp(dtype, "Q6_K")) qtype = GLM53F_GGML_Q6_K;
        else if (!strcmp(dtype, "IQ2_XS")) qtype = GLM53F_GGML_IQ2_XS;
        else if (!strcmp(dtype, "IQ3_XXS")) qtype = GLM53F_GGML_IQ3_XXS;
        else if (!strcmp(dtype, "IQ4_XS")) qtype = GLM53F_GGML_IQ4_XS;
        if (!strcmp(suffix, "gate_up_fused.weight")) { p->gate_up = off; p->gate_type = qtype; }
        else if (!strcmp(suffix, "gate_up_fused.weight_scale_inv")) p->gate_up_scale = off;
        else if (!strcmp(suffix, "down_proj.weight")) { p->down = off; p->inter = cols; p->down_type = qtype; }
        else if (!strcmp(suffix, "down_proj.weight_scale_inv")) p->down_scale = off;
        else continue;
        found++;
    }
    fclose(f);
    return found;
}

static int load_shared_manifest(const char *path, shared_offset *table) {
    FILE *f = fopen(path, "r");
    char line[2048], dtype[16], name[1024], suffix[128];
    unsigned long long off;
    int nd, rows, cols, layer, found = 0;
    if (!f) return -1;
    for (int i = 0; i < NLAYERS; ++i) {
        table[i].gate_up = table[i].gate_up_scale = UINT64_MAX;
        table[i].down = table[i].down_scale = UINT64_MAX;
    }
    while (fgets(line, sizeof(line), f)) {
        char *last;
        if (line[0] == '#' ||
            sscanf(line, "%llu %15s %d %d %d", &off, dtype, &nd, &rows, &cols) != 5) continue;
        last = strrchr(line, ' '); if (!last) continue;
        snprintf(name, sizeof(name), "%s", last + 1); name[strcspn(name, "\r\n")] = 0;
        if (sscanf(name, "model.language_model.layers.%d.mlp.shared_experts.%127s",
                   &layer, suffix) != 2 || layer < FIRST_LAYER || layer >= LAST_LAYER) continue;
        shared_offset *p = &table[layer - FIRST_LAYER];
        if (!strcmp(suffix, "gate_up_fused.weight")) p->gate_up = off;
        else if (!strcmp(suffix, "gate_up_fused.weight_scale_inv")) p->gate_up_scale = off;
        else if (!strcmp(suffix, "down_proj.weight")) { p->down = off; p->inter = cols; }
        else if (!strcmp(suffix, "down_proj.weight_scale_inv")) p->down_scale = off;
        else continue;
        found++;
    }
    fclose(f);
    return found;
}

static unsigned char *load_anon_guard(const char *path, size_t *bytes, int rank, long minimum) {
    const size_t chunk = 64u << 20;
    struct stat st;
    int fd = open(path, O_RDONLY);
    unsigned char *data = NULL;
    if (fd < 0) return NULL;
    if (fstat(fd, &st) || st.st_size <= 0) { close(fd); return NULL; }
    long available = mem_available();
    if (available < minimum || st.st_size > available - minimum) {
        fprintf(stderr, "GLM53F_LOAD_HEADROOM rank=%d need_GiB=%.3f available_GiB=%.3f reject\n",
            rank, st.st_size / 1073741824.0 + minimum / 1073741824.0, available / 1073741824.0);
        close(fd); return NULL;
    }
    if (posix_memalign((void **)&data, 256, (size_t)st.st_size)) { close(fd); return NULL; }
    double t0 = now_sec();
    for (size_t off = 0; off < (size_t)st.st_size; off += chunk) {
        size_t n = (size_t)st.st_size - off;
        if (n > chunk) n = chunk;
        ssize_t got = pread(fd, data + off, n, (off_t)off);
        if (got != (ssize_t)n) { free(data); close(fd); return NULL; }
        posix_fadvise(fd, (off_t)off, (off_t)n, POSIX_FADV_DONTNEED);
        if (mem_available() < minimum) {
            fprintf(stderr, "GLM53F_LOAD_HEADROOM rank=%d offset=%zu reject\n", rank, off);
            free(data); close(fd); return NULL;
        }
    }
    close(fd);
    *bytes = (size_t)st.st_size;
    fprintf(stderr, "rank=%d loaded=%.3fGiB seconds=%.2f MemAvailable=%.3fGiB\n",
            rank, *bytes / 1073741824.0, now_sec() - t0,
            mem_available() / 1073741824.0);
    return data;
}

#ifndef GLM53F_EXPERT_NO_MAIN
static uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void route8(int token, int layer, int expert[8]) {
    uint64_t state = mix64((uint64_t)token * 47 + (uint64_t)layer * 0x10001u);
    int n = 0;
    while (n < 8) {
        state = mix64(state);
        int e = (int)(state % NEXPERTS), duplicate = 0;
        for (int j = 0; j < n; ++j) duplicate |= expert[j] == e;
        if (!duplicate) expert[n++] = e;
    }
}
#endif

static unsigned char *load_anon(const char *path,size_t *bytes,int rank){return load_anon_guard(path,bytes,rank,2L<<30);}
struct glm53f_moe_stage_context_12n {
    glm53f_prefill_config prefill;
    int rank, first_layer, layer_count, active_layer;
    const glm53f_dist *dist;
    int image_rank;
    FILE *route_export[NLAYERS];
    const char *pp_shared;
    expert_offset *table;
    shared_offset shared[NLAYERS];
    unsigned char *blob, *shared_blob;
    int fused_ready; /* fused MoE layer: local_output holds this layer's experts */
    int router_in_sublayer; /* router dispatched by the sublayer itself: never fuse */
    uint16_t *router_w;
    int8_t *router_i8;
    float *router_i8_scale;
    float *router_bias, *router_logits;
    glm53f_moe_scratch_12n *scratch;
    float *batch_up, *batch_activation, *batch_shared, *batch_local;
    float *batch_router, *batch_group_x, *batch_routes;
    glm53f_iq_batch_scratch *verify_scratch;
    float *task_up, *task_activation, *task_output;
    float *int8_scales[NLAYERS];
    int int8_enabled, router_ready;
    int profile;
    double profile_phase[4];
    double gn_phase[8]; /* native: 0 total, 1 x-quant, 2 old-expert loop, 3 shared, 4 combine, 5 task wall, 6 avg busy, 7 tasks */
    unsigned long long occupancy[5]; /* 0, 1, 2-3, 4-7, >=8 tokens/expert. */
    void *i8_prefill_storage;
    /* Grouped prefill straight from native GGUF Q4_K gate/up + Q5_K down (default on; GLM53F_MOE_NATIVE_GROUPED=0 disables;
     * =2 also recomputes with the per-token decode kernels and reports the relative difference). */
    int gn_mode, rg_mode, sg_mode, ar_slab;
    uint16_t *sg_gu[NLAYERS], *sg_dn[NLAYERS];
    uint8_t *s8_gu[NLAYERS], *s8_dn[NLAYERS];
    int8_t *s8_xq, *s8_xp, *s8_a8, *s8_xp2; float *s8_xs, *s8_bt, *s8_xsp, *s8_ygu, *s8_as, *s8_asp, *s8_bd, *s8_y;
    float *sg_up, *sg_act, *sg_out;
    uint16_t *router_wt; /* tiled router weights for gmn_router_run */
    int8_t *gn_xq;
    float *gn_xs, *gn_bt, *gn_verify;
    void *gn_threads;
    int gn_nthreads;
    /* Native GGUF shared expert (GLM53F_Q2_SHEXP_STAGE): rank-local rows of
     * gate/up and columns of down, used by single-token decode. */
    int nsh_native, nsh_in;
    uint8_t *nsh_g[NLAYERS], *nsh_u[NLAYERS], *nsh_d[NLAYERS];
    int nsh_gt[NLAYERS], nsh_ut[NLAYERS], nsh_dt[NLAYERS];
    float *nsh_gv, *nsh_uv, *nsh_act, *nsh_out;
    void *nsh_act_x, *nsh_act_h;
};

static int moe_sum(const glm53f_moe_stage_context_12n *c,const float *input,float *output,int count){
    if(!c->dist)return glm53f_sum_allreduce_12n(input,output,count);
    return MPI_Allreduce(input==output?MPI_IN_PLACE:input,output,count,MPI_FLOAT,MPI_SUM,c->dist->tp)==MPI_SUCCESS?0:-1;
}
static int moe_sum_mpi(const glm53f_moe_stage_context_12n *c,const float *input,float *output,int count){
    return c->dist?moe_sum(c,input,output,count):glm53f_sum_allreduce_mpi_12n(input,output,count);
}
static int moe_sum_slabs(const glm53f_moe_stage_context_12n *c,const float *input,float *output,int tokens,int columns,int slab){
    if(!c->dist)return glm53f_sum_allreduce_slabs_12n(input,output,tokens,columns,slab);
    if(slab<1)return-1;for(int base=0;base<tokens;base+=slab){int n=tokens-base;if(n>slab)n=slab;
        if(moe_sum(c,input+(size_t)base*columns,output+(size_t)base*columns,n*columns))return-1;}
    return 0;
}
static int nsh_load_one(int fd, const char *manifest, const char *wanted,
                        int rows, int cols, uint8_t **out, int *out_type) {
    FILE *f = fopen(manifest, "r");
    char line[512], tn[32], name[256];
    unsigned long long offset = 0;
    unsigned type = 0;
    int r = 0, cc = 0, found = 0;
    if (!f) return -1;
    while (fgets(line, sizeof(line), f))
        if (line[0] != '#' && sscanf(line, "%llu %u %31s %d %d %255s", &offset,
                &type, tn, &r, &cc, name) == 6 && !strcmp(name, wanted)) {
            found = 1;
            break;
        }
    fclose(f);
    if (!found || r != rows || cc != cols ||
        !glm53f_native_type_supported((int)type)) return -1;
    size_t bytes = (size_t)rows * glm53f_iq_row_size((int)type, cols), done = 0;
    uint8_t *p = NULL;
    if (!bytes || posix_memalign((void **)&p, 256, bytes)) return -1;
    while (done < bytes) {
        ssize_t n = pread(fd, p + done, bytes - done, (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { free(p); return -1; }
        done += (size_t)n;
    }
    uint8_t *packed = NULL;
    int packed_type = (int)type;
    if (glm53f_native_repack((int)type, p, rows, cols, &packed, &packed_type)) {
        free(p);
        return -1;
    }
    if (packed) { free(p); p = packed; }
    *out = p;
    *out_type = packed_type;
    return 0;
}

static int nsh_load(glm53f_moe_stage_context_12n *c, const char *stage) {
    char blob[PATH_MAX], manifest[PATH_MAX], name[128], line[8192];
    int in = 0, i0 = 0, fd;
    snprintf(blob, sizeof(blob), "%s/rank%02d.blob", stage, c->image_rank);
    snprintf(manifest, sizeof(manifest), "%s/rank%02d.manifest", stage, c->image_rank);
    if(c->dist){if(glm53f_pp_manifest_check(manifest,"SHARED",c->dist,c->dist->map.first_layer,c->dist->map.end_layer))return-1;
        in=512;i0=c->dist->map.tp_rank*512;
    }else{
    FILE *f = fopen(manifest, "r");
    if (!f) return -1;
    while (fgets(line, sizeof(line), f)) {
        const char *p = strstr(line, " slice=");
        if (line[0] == '#' && p && sscanf(p, " slice=%d+%d", &i0, &in) == 2) break;
    }
    fclose(f);
    }
    if (in < 64 || in % 64 || (fd = open(blob, O_RDONLY)) < 0) return -1;
    int rc = 0;
    for (int li = 0; li < c->layer_count && !rc; ++li) {
        int layer = c->first_layer + li, t = layer - FIRST_LAYER;
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", layer);
        rc |= c->dist?glm53f_pp_native_load(fd,manifest,name,-1,in,4096,1,&c->nsh_g[t],&c->nsh_gt[t]):nsh_load_one(fd,manifest,name,in,4096,&c->nsh_g[t],&c->nsh_gt[t]);
        snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", layer);
        rc |= c->dist?glm53f_pp_native_load(fd,manifest,name,-1,in,4096,1,&c->nsh_u[t],&c->nsh_ut[t]):nsh_load_one(fd,manifest,name,in,4096,&c->nsh_u[t],&c->nsh_ut[t]);
        snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", layer);
        rc |= c->dist?glm53f_pp_native_load(fd,manifest,name,-1,4096,in,1,&c->nsh_d[t],&c->nsh_dt[t]):nsh_load_one(fd,manifest,name,4096,in,&c->nsh_d[t],&c->nsh_dt[t]);
    }
    close(fd);
    if (rc) return -1;
    size_t ab = glm53f_native_act_bytes(4096);
    if (posix_memalign((void **)&c->nsh_gv, 256, (size_t)in * 4) ||
        posix_memalign((void **)&c->nsh_uv, 256, (size_t)in * 4) ||
        posix_memalign((void **)&c->nsh_act, 256, (size_t)in * 4) ||
        posix_memalign((void **)&c->nsh_out, 256, 4096 * 4) ||
        posix_memalign(&c->nsh_act_x, 256, ab) ||
        posix_memalign(&c->nsh_act_h, 256, ab)) return -1;
    c->nsh_in = in;
    c->nsh_native = 1;
    if (!c->rank)
        fprintf(stderr, "GLM53F_MOE_LOAD phase=native_shexp slice=%d+%d type=%d/%d/%d\n",
                i0, in, c->nsh_gt[c->first_layer - FIRST_LAYER],
                c->nsh_ut[c->first_layer - FIRST_LAYER],
                c->nsh_dt[c->first_layer - FIRST_LAYER]);
    return 0;
}

static int nsh_is_q80(int type) {
    return type == GLM53F_GGML_Q8_0 || type == GLM53F_NATIVE_Q8_0R ||
           type == GLM53F_NATIVE_Q8_0R16;
}

/* output += shared_expert(x) for this rank's intermediate slice.  One team:
 * quantize x, gate/up rows, SwiGLU (the MoE clamp), quantize the activation,
 * then down rows.  Every stage follows llama.cpp's Q8_0 vec_dot contract. */
struct shared_call { glm53f_moe_stage_context_12n *c; int layer, bad; const float *x; float *output; };
static void shared_worker(void *context) {
    struct shared_call *a = context;
    glm53f_moe_stage_context_12n *c = a->c;
    const int in = c->nsh_in, t = a->layer;
    const float *x = a->x;
    float *output = a->output;
    glm53f_native_matrix gu[2] = {{c->nsh_gv, c->nsh_g[t], c->nsh_gt[t], in, 4096},
        {c->nsh_uv, c->nsh_u[t], c->nsh_ut[t], in, 4096}};
    glm53f_native_matrix dn = {c->nsh_out, c->nsh_d[t], c->nsh_dt[t], 4096, in};
    int bad = 0;

#pragma omp single
        bad |= glm53f_native_act_prepare(c->nsh_act_x, x, 4096,
                   !nsh_is_q80(gu[0].type) || !nsh_is_q80(gu[1].type),
                   nsh_is_q80(gu[0].type) || nsh_is_q80(gu[1].type)) != 0;
        bad |= glm53f_native_matvec_team(gu, 2, c->nsh_act_x) != 0;
#pragma omp single
        {
            for (int i = 0; i < in; ++i) {
                float g = c->nsh_gv[i], u = c->nsh_uv[i];
                if (g > 10) g = 10;
                if (g < -100) g = -100;
                if (u > 10) u = 10;
                if (u < -10) u = -10;
                c->nsh_act[i] = (g / (1 + expf(-g))) * u;
            }
            bad |= glm53f_native_act_prepare(c->nsh_act_h, c->nsh_act, in,
                       !nsh_is_q80(dn.type), nsh_is_q80(dn.type)) != 0;
        }
        bad |= glm53f_native_matvec_team(&dn, 1, c->nsh_act_h) != 0;
#pragma omp for schedule(static)
        for (int i = 0; i < 4096; ++i) output[i] += c->nsh_out[i];

    if (bad) {
#pragma omp atomic write
        a->bad = 1;
    }
}
static int nsh_accumulate(glm53f_moe_stage_context_12n *c, int t,
                          const float *x, float *output) {
    int bad = 0;
    struct shared_call call = {c, t, 0, x, output};
    if (glm53f_team_available()) glm53f_team_dispatch(shared_worker, &call);
    else {
#pragma omp parallel
        { shared_worker(&call); }
    }
    bad = call.bad;
    return bad ? -1 : 0;
}

/* Up to four positions, using the same native weights and SwiGLU clamp as
 * decode. Existing batch scratch is sized for 512 intermediate channels. */
static int nsh_batch(glm53f_moe_stage_context_12n *c, int layer,
                     const float *x, int tokens) {
    const int in = c->nsh_in;
    float *gate = c->batch_up, *up = gate + (size_t)tokens * in;
    glm53f_native_matrix gu[2] = {
        {gate,c->nsh_g[layer],c->nsh_gt[layer],in,4096},
        {up,c->nsh_u[layer],c->nsh_ut[layer],in,4096}};
    glm53f_native_matrix dn = {
        c->batch_shared,c->nsh_d[layer],c->nsh_dt[layer],4096,in};
    if (tokens < 1 || tokens > 4 || in > 512 ||
        glm53f_native_matvec_batch(gu, 2, x, tokens)) return -1;
#pragma omp parallel for schedule(static)
    for (int q = 0; q < tokens * in; ++q) {
        float g = gate[q], u = up[q];
        if (g > 10) g = 10;
        if (g < -100) g = -100;
        if (u > 10) u = 10;
        if (u < -10) u = -10;
        c->batch_activation[q] = (g / (1 + expf(-g))) * u;
    }
    return glm53f_native_matvec_batch(&dn, 1, c->batch_activation, tokens);
}

void glm53f_moe_configure_prefill_12n(glm53f_moe_stage_context_12n *c,
                                     const glm53f_prefill_config *config) {
    if (c && config) c->prefill = *config;
}

static void moe_profile_occupancy(glm53f_moe_stage_context_12n *c,
                                  const int *selected, int tokens) {
    if (!c->profile || c->rank) return;
    int count[NEXPERTS] = {0};
    for (int i = 0; i < tokens * 8; ++i) ++count[selected[i]];
    for (int e = 0; e < NEXPERTS; ++e) {
        int n = count[e];
        ++c->occupancy[n < 2 ? n : n < 4 ? 2 : n < 8 ? 3 : 4];
    }
}

int glm53f_moe_stage_convert_int8_12n(glm53f_moe_stage_context_12n *c) {
    if (!c || c->int8_enabled || !c->shared_blob) return -1;
    size_t count = 0;
    size_t layer_counts[NLAYERS] = {0};
    int first = c->first_layer - FIRST_LAYER;
    for (int l = first; l < first + c->layer_count; ++l) {
        for (int e = 0; e < NEXPERTS; ++e) {
            expert_offset *p = c->table + l * NEXPERTS + e;
            if (p->gate_up == UINT64_MAX) continue;
            if (p->gate_type || p->down_type || p->gate_up_scale == UINT64_MAX ||
                p->down_scale == UINT64_MAX || p->inter < 128 || p->inter > 512 || p->inter % 128) return -1;
            layer_counts[l] += 2 * p->inter + 4096;
        }
        shared_offset *s = c->shared + l;
        if (s->gate_up == UINT64_MAX || s->down == UINT64_MAX ||
            s->gate_type || s->down_type || s->gate_up_scale == UINT64_MAX ||
            s->down_scale == UINT64_MAX || s->inter < 128 || s->inter > 512 || s->inter % 128) return -1;
        layer_counts[l] += 2 * s->inter + 4096;
        count += layer_counts[l];
    }
    /* ~5 MiB/layer fits reclaimed BF16 projection chunks; one 205 MiB scale
     * allocation cannot reuse those holes in Fugaku's retained heap arena. */
    for (int l = first; l < first + c->layer_count; ++l)
        if (posix_memalign((void **)&c->int8_scales[l], 256, layer_counts[l] * sizeof(float))) return -1;
    for (int l = first; l < first + c->layer_count; ++l) {
        size_t off = 0;
        for (int e = 0; e <= NEXPERTS; ++e) {
            expert_offset *p = e == NEXPERTS ? c->shared + l : c->table + l * NEXPERTS + e;
            if (p->gate_up == UINT64_MAX) continue;
            p->i8_gate_scale = c->int8_scales[l] + off; off += 2 * p->inter;
            p->i8_down_scale = c->int8_scales[l] + off; off += 4096;
        }
    }
    double begin = now_sec();
    int failed = 0;
    /* One persistent team, one bounded tile per worker. The byte count of
     * the anonymous weight blob is unchanged throughout conversion. */
#pragma omp parallel reduction(|:failed)
    {
        int8_t *scratch = malloc(64u * 4096u);
        if (!scratch) failed = 1;
#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < c->layer_count * (NEXPERTS + 1); ++i) {
            int l = first + i / (NEXPERTS + 1), e = i % (NEXPERTS + 1);
            expert_offset *p = e == NEXPERTS ? c->shared + l : c->table + l * NEXPERTS + e;
            uint8_t *blob = e == NEXPERTS ? c->shared_blob : c->blob;
            if (!scratch || p->gate_up == UINT64_MAX) continue;
            for (int r = 0; r < 2 * p->inter; r += 64)
                failed |= glm53f_i8_pack_fp8_tile(blob + p->gate_up + (size_t)r * 4096,
                    p->i8_gate_scale + r, (float *)(blob + p->gate_up_scale), r, 4096, scratch) != 0;
            for (int r = 0; r < 4096; r += 64)
                failed |= glm53f_i8_pack_fp8_tile(blob + p->down + (size_t)r * p->inter,
                    p->i8_down_scale + r, (float *)(blob + p->down_scale), r, p->inter, scratch) != 0;
        }
        free(scratch);
    }
    if (failed) return -1;
    if (getenv("GLM53F_INT8_ROUTER")) {
        size_t rows = (size_t)c->layer_count * NEXPERTS;
        if (posix_memalign((void **)&c->router_i8, 256, rows * 4096) ||
            posix_memalign((void **)&c->router_i8_scale, 256, rows * sizeof(float)))
            return -1;
#pragma omp parallel for schedule(dynamic, 1) reduction(|:failed)
        for (size_t g = 0; g < rows / 64; ++g)
            failed |= glm53f_i8_pack_bf16_tile(
                c->router_i8 + g * 64 * 4096,
                c->router_i8_scale + g * 64,
                c->router_w + g * 64 * 4096, 4096) != 0;
        if (failed) return -1;
        free(c->router_w);
        c->router_w = NULL;
    }
    c->int8_enabled = 1;
    fprintf(stderr, "GLM53F_INT8_LOAD rank=%d scale_MiB=%.3f scratch_MiB_max=%.3f seconds=%.3f MemAvailable_GiB=%.3f\n",
        c->rank, count * sizeof(float) / 1048576.0, omp_get_max_threads() * 0.25,
        now_sec() - begin, mem_available() / 1073741824.0);
    return 0;
}

static int moe_int8_local(glm53f_moe_stage_context_12n *c, float *out,
        const float *x, const int8_t *input_qx, float input_xs,
        const int *selected, const float *route_weight, int layer) {
    const expert_offset *part[9];
    const uint8_t *blob[9];
    float weights[9], xs, as[9] = {0};
    int8_t qx[4096], qa[9 * 512] = {0};
    int count = 0, prefix[10] = {0}, failed = 0;
    if (input_qx) {
        memcpy(qx, input_qx, sizeof(qx));
        xs = input_xs;
    } else if (glm53f_i8_quantize_x(qx, &xs, x, 4096)) return -1;
    for (int k = 0; k < 8; ++k) {
        const expert_offset *p = c->table + layer * NEXPERTS + selected[k];
        if (p->gate_up == UINT64_MAX) continue;
        part[count] = p; blob[count] = c->blob; weights[count++] = route_weight[k];
    }
    part[count] = c->shared + layer; blob[count] = c->shared_blob; weights[count++] = 1;
    int row_tile = getenv("GLM53F_MOE_I8_16ROW") ? 16 : 64;
    for (int k = 0; k < count; ++k) prefix[k + 1] = prefix[k] + 2 * part[k]->inter / row_tile;
#pragma omp parallel reduction(|:failed)
    {
#pragma omp for schedule(static)
        for (int t = 0; t < prefix[count]; ++t) {
            int k = 0;
            while (t >= prefix[k + 1]) ++k;
            int r = (t - prefix[k]) * row_tile;
            if (row_tile == 16) {
                int group = r / 64, quarter = (r % 64) / 16;
                glm53f_i8_dot16(c->scratch->up + k * 1024 + r,
                    (const int8_t *)(blob[k] + part[k]->gate_up +
                    (size_t)group * 64 * 4096 + quarter * 64),
                    part[k]->i8_gate_scale + r, qx, xs, 4096);
            } else
                glm53f_i8_dot64(c->scratch->up + k * 1024 + r,
                    (const int8_t *)(blob[k] + part[k]->gate_up + (size_t)r * 4096),
                    part[k]->i8_gate_scale + r, qx, xs, 4096);
        }
#pragma omp for schedule(static)
        for (int k = 0; k < count; ++k) {
            int n = part[k]->inter;
            float *act = c->scratch->activation + k * 512;
            for (int i = 0; i < n; ++i) {
                float g = c->scratch->up[k * 1024 + i], u = c->scratch->up[k * 1024 + n + i];
                if (g > 10) g = 10;
                if (g < -100) g = -100;
                if (u > 10) u = 10;
                if (u < -10) u = -10;
                act[i] = (g / (1 + expf(-g))) * u;
            }
            failed |= glm53f_i8_quantize_x(qa + k * 512, as + k, act, n) != 0;
        }
#pragma omp for schedule(static)
        for (int r = 0; r < 4096; r += row_tile) {
            float sum[64] = {0}, value[64];
            for (int k = 0; k < count; ++k) {
                if (row_tile == 16) {
                    int group = r / 64, quarter = (r % 64) / 16;
                    glm53f_i8_dot16(value,
                        (const int8_t *)(blob[k] + part[k]->down +
                        (size_t)group * 64 * part[k]->inter + quarter * 64),
                        part[k]->i8_down_scale + r, qa + k * 512, as[k], part[k]->inter);
                } else
                    glm53f_i8_dot64(value,
                        (const int8_t *)(blob[k] + part[k]->down + (size_t)r * part[k]->inter),
                        part[k]->i8_down_scale + r, qa + k * 512, as[k], part[k]->inter);
                for (int j = 0; j < row_tile; ++j) sum[j] += weights[k] * value[j];
            }
            memcpy(out + r, sum, (size_t)row_tile * sizeof(float));
        }
    }
    return failed ? -1 : 0;
}

static glm53f_moe_stage_context_12n *moe_create(
        const glm53f_dist *dist,const char*routed_stage,const char*shared_stage,const char*pp_shared,const char*model_dir,
        int first_layer, int layer_count) {
    int rank, nr;
    char path[512];
    size_t bytes;
    glm53f_st_context *st = NULL;
    glm53f_moe_stage_context_12n *c;
    if(dist){if(!dist->initialized||dist->config.layout!=GLM53F_PP3_TP4||first_layer<dist->map.first_layer||first_layer+layer_count>dist->map.end_layer||!routed_stage||!pp_shared||shared_stage)return NULL;
        rank=dist->map.tp_rank;nr=dist->map.tp_size;
    }else{MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(nr!=12)return NULL;}
    if(first_layer<FIRST_LAYER||layer_count<1||first_layer+layer_count>(dist?45:LAST_LAYER))return NULL;
    c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->dist=dist;c->image_rank=dist?dist->map.world_rank:rank;c->pp_shared=pp_shared;
    c->rank = rank; c->first_layer = first_layer;
    c->layer_count = layer_count; c->active_layer = first_layer;
    c->profile = getenv("GLM53F_PROFILE") != NULL;
    c->table = malloc((size_t)NLAYERS * NEXPERTS * sizeof(*c->table));
    if (!c->table) goto fail;
    snprintf(path, sizeof(path), "%s/rank%02d.manifest", routed_stage, c->image_rank);
    if(dist){char blob[512];struct stat st;
        int z=snprintf(blob,sizeof(blob),"%s/rank%02d.blob",routed_stage,c->image_rank);
        if(z<0||z>=(int)sizeof(blob)||stat(blob,&st)||st.st_size<1||glm53f_pp_routed_validate(path,dist,first_layer,first_layer+layer_count,(uint64_t)st.st_size,NULL))goto fail;
    }
    if (load_manifest(path, c->table) < layer_count * (dist?288*2:96*4)) goto fail;
    if (shared_stage) {
        snprintf(path, sizeof(path), "%s/rank%02d.manifest", shared_stage, c->image_rank);
        if (load_shared_manifest(path, c->shared) != layer_count * 4) goto fail;
    }
    /* The checkpoint index/parser has a substantial transient allocation.
     * Finish all router reads and release parser pages BEFORE loading 23.6 GiB
     * of routed experts. Opening metadata after that load can OOM at 512K. */
    if (!rank) fprintf(stderr, "GLM53F_MOE_LOAD phase=router_open\n");
    st = dist?glm53f_pp_core_open(dist,model_dir):glm53f_st_open(model_dir);
    if (!st) goto fail;
    size_t wn = (size_t)layer_count * NEXPERTS * 4096 * sizeof(uint16_t);
    size_t bn = (size_t)layer_count * NEXPERTS * sizeof(float);
    if (posix_memalign((void **)&c->router_w, 256, wn) ||
        posix_memalign((void **)&c->router_bias, 256, bn) ||
        posix_memalign((void **)&c->router_logits, 256, NEXPERTS * sizeof(float)) ||
        posix_memalign((void **)&c->scratch, 256, sizeof(*c->scratch)) ||
        posix_memalign((void **)&c->batch_up, 256, (size_t)4 * 1024 * 4) ||
        posix_memalign((void **)&c->batch_activation, 256, (size_t)4 * 512 * 4) ||
        posix_memalign((void **)&c->batch_shared, 256, (size_t)4 * 4096 * 4) ||
        posix_memalign((void **)&c->batch_local, 256, (size_t)GLM53F_PREFILL_MAX_TOKENS * 4096 * 4) ||
        posix_memalign((void **)&c->batch_router, 256, (size_t)GLM53F_PREFILL_MAX_TOKENS * NEXPERTS * 4) ||
        posix_memalign((void **)&c->batch_group_x, 256, (size_t)4 * 4096 * 4) ||
        posix_memalign((void **)&c->batch_routes, 256,
                      (size_t)GLM53F_PREFILL_MAX_TOKENS * 8 * 4096 * 4)) goto fail;
    if (getenv("GLM53F_VERIFY_GROUPED") && atoi(getenv("GLM53F_VERIFY_GROUPED"))) {
        c->verify_scratch = glm53f_iq_batch_scratch_create();
        if (!c->verify_scratch) goto fail;
    }
    if (!rank) fprintf(stderr, "GLM53F_MOE_LOAD phase=router_read bytes=%zu\n", wn);
    for (int li = 0; li < layer_count; ++li) {
        char n[256];
        snprintf(n, sizeof(n), "model.language_model.layers.%d.mlp.gate.weight", first_layer + li);
        if (glm53f_st_read(st, n, 0, c->router_w + (size_t)li * NEXPERTS * 4096,
                (size_t)NEXPERTS * 4096 * sizeof(uint16_t))) goto fail;
        snprintf(n, sizeof(n), "model.language_model.layers.%d.mlp.gate.e_score_correction_bias", first_layer + li);
        if (glm53f_st_read(st, n, 0, c->router_bias + (size_t)li * NEXPERTS,
                NEXPERTS * sizeof(float))) goto fail;
    }
    glm53f_st_close(st); st = NULL;
    if (posix_memalign((void **)&c->router_wt, 256, (size_t)layer_count * NEXPERTS * 4096 * sizeof(uint16_t))) goto fail;
    for (int li = 0; li < layer_count; ++li)
        gmn_router_pack(c->router_wt + (size_t)li * NEXPERTS * 4096, c->router_w + (size_t)li * NEXPERTS * 4096, 4096);
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
    if (!rank) fprintf(stderr, "GLM53F_MOE_LOAD phase=resident_weights\n");
    snprintf(path, sizeof(path), "%s/rank%02d.blob", routed_stage, c->image_rank);
    c->blob = load_anon_guard(path,&bytes,rank,dist?6L<<30:2L<<30);
    if(c->blob&&dist){char manifest[512];snprintf(manifest,sizeof(manifest),"%s/rank%02d.manifest",routed_stage,c->image_rank);
        if(glm53f_pp_routed_validate(manifest,dist,first_layer,first_layer+layer_count,bytes,c->blob))goto fail;}
    if (!c->blob) goto fail;
    if (shared_stage) {
        snprintf(path, sizeof(path), "%s/rank%02d.blob", shared_stage, c->image_rank);
        c->shared_blob = load_anon(path, &bytes, rank);
        if (!c->shared_blob) goto fail;
    }
    {
        const char *nsh = dist?pp_shared:getenv("GLM53F_Q2_SHEXP_STAGE");
        if (nsh && *nsh && c->first_layer < 45 && nsh_load(c, nsh)) {
            fprintf(stderr, "rank=%d failed to load GLM53F_Q2_SHEXP_STAGE=%s\n", rank, nsh);
            goto fail;
        }
    }
    {
        const char *gn = getenv("GLM53F_MOE_NATIVE_GROUPED");
        c->gn_mode = gn && *gn ? atoi(gn) : 1;
        const char *rg = getenv("GLM53F_MOE_ROUTER_GEMM");
        c->rg_mode = rg && *rg ? atoi(rg) : 1;
        const char *sg = getenv("GLM53F_MOE_SHARED_GEMM");
        c->sg_mode = sg && *sg ? atoi(sg) : 1;
        const char *as = getenv("GLM53F_MOE_AR_SLAB");
        c->ar_slab = as && *as ? atoi(as) : 512;
    }
    return c;
fail:
    if (st) glm53f_st_close(st);
    glm53f_moe_stage_free_12n(c);
    return NULL;
}
glm53f_moe_stage_context_12n *glm53f_moe_stage_create_12n(const char *routed,const char *shared,const char *model,int first,int count){return moe_create(NULL,routed,shared,NULL,model,first,count);}
glm53f_moe_stage_context_12n *glm53f_moe_stage_create_dist(const glm53f_dist *dist,const char *routed,const char *native_shared,const char *model,int first,int count){return dist?moe_create(dist,routed,NULL,native_shared,model,first,count):NULL;}
/* Routed experts + native shared expert in a single parallel region (glm53f_iq_expert_weighted_shared).
 * Returns 1 when the step was computed into c->scratch->local_output, 0 to fall back to the separate paths. */
static int moe_fused_shared(glm53f_moe_stage_context_12n *c, int t, const float *x, const glm53f_expert_part *part,
                            const float *part_weight, int npart) {
    static int enabled = -1;
    if (enabled < 0) { const char *e = getenv("GLM53F_MOE_FUSE_SHARED"); enabled = !e || !*e || atoi(e); }
    if (!enabled || !c->nsh_native || npart < 1 || npart > 9) return 0;
    glm53f_iq_part iq[9];
    for (int k = 0; k < npart; ++k) {
        if (!part[k].gate_type && !part[k].down_type) return 0;
        iq[k] = (glm53f_iq_part){part[k].gate_up, part[k].down, part[k].gate_type, part[k].down_type, part[k].inter};
    }
    const int in = c->nsh_in;
    glm53f_iq_shared sh = {
        .gu = {{c->nsh_gv, c->nsh_g[t], c->nsh_gt[t], in, 4096}, {c->nsh_uv, c->nsh_u[t], c->nsh_ut[t], in, 4096}},
        .dn = {c->nsh_out, c->nsh_d[t], c->nsh_dt[t], 4096, in},
        .act_x = c->nsh_act_x, .act_h = c->nsh_act_h, .act = c->nsh_act, .rows = in,
        .x_q8k = !nsh_is_q80(c->nsh_gt[t]) || !nsh_is_q80(c->nsh_ut[t]),
        .x_q80 = nsh_is_q80(c->nsh_gt[t]) || nsh_is_q80(c->nsh_ut[t]),
        .h_q8k = !nsh_is_q80(c->nsh_dt[t]), .h_q80 = nsh_is_q80(c->nsh_dt[t])};
    double tn = c->profile ? glm53f_clock() : 0.0;
    if (glm53f_iq_expert_weighted_shared(c->scratch->local_output, iq, part_weight, npart, x, &sh)) return 0;
    if (c->profile) c->profile_phase[3] += glm53f_clock() - tn;
    return 1;
}
/* Eight bf16 weight rows against one vector: eight independent accumulator chains (same per-row accumulation order as
 * glm53f_dot_bf16_sve, so the logits are bit-identical) instead of one latency-bound chain per row. */
static void router_dot8(float *y, const uint16_t *w, const float *x, int n) {
    svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const int vl = (int)svcntw();
    for (int i = 0; i < n; i += vl) {
        const svbool_t p = svwhilelt_b32(i, n);
        const svfloat32_t xv = svld1(p, x + i);
#define RD8(N, A) do { svuint32_t z = svlsl_n_u32_x(p, svld1uh_u32(p, w + (size_t)(N) * n + i), 16); \
                       A = svmla_x(p, A, svreinterpret_f32_u32(z), xv); } while (0)
        RD8(0, a0); RD8(1, a1); RD8(2, a2); RD8(3, a3); RD8(4, a4); RD8(5, a5); RD8(6, a6); RD8(7, a7);
#undef RD8
    }
    const svbool_t p = svptrue_b32();
    y[0] = svaddv_f32(p, a0); y[1] = svaddv_f32(p, a1); y[2] = svaddv_f32(p, a2); y[3] = svaddv_f32(p, a3);
    y[4] = svaddv_f32(p, a4); y[5] = svaddv_f32(p, a5); y[6] = svaddv_f32(p, a6); y[7] = svaddv_f32(p, a7);
}
/* Prefetch plan (glm53f_pf_plan.h) for MoE layer `layer`: router weights and the native shared expert. */
void glm53f_moe_stage_prefetch_plan_12n(glm53f_moe_stage_context_12n *c, int layer) {
    const int li = layer - c->first_layer, t = layer - FIRST_LAYER;
    if (!c || li < 0 || li >= c->layer_count) return;
    if (c->router_w && !c->router_i8) glm53f_pf_add_tasks(c->router_w + (size_t)li * NEXPERTS * 4096, (size_t)8 * 4096 * sizeof(uint16_t), NEXPERTS / 8);
    if (c->nsh_native) {
        const int in = c->nsh_in;
        const glm53f_native_matrix gu[2] = {{NULL, c->nsh_g[t], c->nsh_gt[t], in, 4096}, {NULL, c->nsh_u[t], c->nsh_ut[t], in, 4096}};
        const glm53f_native_matrix dn = {NULL, c->nsh_d[t], c->nsh_dt[t], 4096, in};
        glm53f_pf_add_matvec(gu, 2);
        glm53f_pf_add_matvec(&dn, 1);
    }
}
/* CMG-local placement of every routed-expert part this rank owns, matching the
 * affine decode split (GLM53F_IQ_AFFINE=1): CMG c streams rows
 * [bound[c], bound[c+1]) of each part's gate/up and down matrices. */
void glm53f_moe_stage_place_experts_12n(glm53f_moe_stage_context_12n *c, glm53f_cmg_batch *b,
                                        int nt, const int *cmg_node) {
    if (!c || !c->blob || !c->table) return;
    for (int li = 0; li < c->layer_count; ++li) {
        const int t = c->first_layer + li - FIRST_LAYER;
        if (t < 0 || t >= NLAYERS) continue;
        for (int e = 0; e < NEXPERTS; ++e) {
            const expert_offset *p = &c->table[(size_t)t * NEXPERTS + e];
            if (p->gate_up == UINT64_MAX || !p->gate_type || !p->down_type) continue;
            glm53f_cmg_place_rows(b, c->blob + p->gate_up, glm53f_iq_row_size(p->gate_type, 4096),
                                  2 * p->inter, nt, cmg_node);
            glm53f_cmg_place_rows(b, c->blob + p->down, glm53f_iq_row_size(p->down_type, p->inter),
                                  4096, nt, cmg_node);
        }
    }
}
void glm53f_moe_router_team_12n(void *context, const float *x) {
    glm53f_moe_stage_context_12n *c = context;
    const int li = c->active_layer - c->first_layer;
    if (c->router_i8 || li < 0 || li >= c->layer_count) return;
#pragma omp for schedule(static)
    for (int b = 0; b < NEXPERTS / 8; ++b)
        router_dot8(c->router_logits + b * 8,
            c->router_w + ((size_t)li * NEXPERTS + b * 8) * 4096, x, 4096);
#pragma omp single
    c->router_ready = 1;
    /* Opt-in fused MoE layer (GLM53F_MOE_FUSED_LAYER=1, --moe-layer-kernel fused):
     * the same team continues with top-k (computed identically by every
     * thread) and the routed+shared expert step, removing the separate expert
     * dispatch and the serial routing section. Every condition below is
     * identical across threads, so all of them take the same path. */
    const char *fused_env = getenv("GLM53F_MOE_FUSED_LAYER");
    const char *shared_env = getenv("GLM53F_MOE_FUSE_SHARED");
    const int t = c->active_layer - FIRST_LAYER;
    if (!fused_env || !atoi(fused_env) || c->router_in_sublayer || !c->nsh_native || c->int8_enabled || t < 0 || t >= NLAYERS ||
        c->route_export[t] || (shared_env && *shared_env && !atoi(shared_env))) return;
    int selected[8], npart = 0;
    float route_weight[8], part_weight[9];
    glm53f_iq_part iq[9];
    glm53f_router_topk(c->router_logits, c->router_bias + (size_t)li * NEXPERTS, NEXPERTS, 8, 2.5f,
                       selected, route_weight);
    for (int k = 0; k < 8; ++k) {
        const expert_offset *p = &c->table[(size_t)t * NEXPERTS + selected[k]];
        if (p->gate_up == UINT64_MAX) continue;
        if (!p->gate_type && !p->down_type) { npart = 0; break; }
        iq[npart] = (glm53f_iq_part){c->blob + p->gate_up, c->blob + p->down, p->gate_type, p->down_type, p->inter};
        part_weight[npart++] = route_weight[k];
    }
    const int in = c->nsh_in;
    const glm53f_iq_shared sh = {
        .gu = {{c->nsh_gv, c->nsh_g[t], c->nsh_gt[t], in, 4096}, {c->nsh_uv, c->nsh_u[t], c->nsh_ut[t], in, 4096}},
        .dn = {c->nsh_out, c->nsh_d[t], c->nsh_dt[t], 4096, in},
        .act_x = c->nsh_act_x, .act_h = c->nsh_act_h, .act = c->nsh_act, .rows = in,
        .x_q8k = !nsh_is_q80(c->nsh_gt[t]) || !nsh_is_q80(c->nsh_ut[t]),
        .x_q80 = nsh_is_q80(c->nsh_gt[t]) || nsh_is_q80(c->nsh_ut[t]),
        .h_q8k = !nsh_is_q80(c->nsh_dt[t]), .h_q80 = nsh_is_q80(c->nsh_dt[t])};
    const int ok = npart >= 1 &&
        glm53f_iq_expert_weighted_shared_team(c->scratch->local_output, iq, part_weight, npart, x, &sh) == 0;
#pragma omp single
    c->fused_ready = ok;
}
struct router_call { glm53f_moe_stage_context_12n *c; const float *x; };
static void router_worker(void *context) {
    struct router_call *a = context;
    glm53f_moe_router_team_12n(a->c, a->x);
}
void glm53f_moe_stage_set_layer_12n(glm53f_moe_stage_context_12n*c,int layer){if(c)c->active_layer=layer;}
int glm53f_moe_set_route_export_12n(glm53f_moe_stage_context_12n *c,const char *prefix) {
    if(!c||!prefix||!*prefix)return-1;
    if(c->rank)return 0;
    for(int layer=c->first_layer;layer<c->first_layer+c->layer_count;layer++){
        int local=layer-FIRST_LAYER;char path[4096];
        int n=snprintf(path,sizeof(path),"%s.layer%02d.routes",prefix,layer);
        if(n<0||n>=(int)sizeof(path)||c->route_export[local]||!(c->route_export[local]=fopen(path,"wbx")))return-1;
    }
    return 0;
}
static int moe_trace_routes(glm53f_moe_stage_context_12n *c,const int *selected,const float *weight,int tokens) {
    int layer=c->active_layer-FIRST_LAYER;
    if(layer<0||layer>=NLAYERS)return-1;
    FILE *f=c->route_export[layer];if(!f)return 0;
    for(int t=0;t<tokens;t++){
        int32_t ids[8];for(int k=0;k<8;k++)ids[k]=selected[t*8+k];
        if(fwrite(ids,sizeof(ids),1,f)!=1||fwrite(weight+t*8,8*sizeof(float),1,f)!=1)return-1;
    }
    /* Diagnostics are excluded from speed runs. Flush before capture retrieval. */
    if(fflush(f)||fsync(fileno(f)))return-1;
    return 0;
}
int glm53f_moe_stage_sublayer_12n(void*context,float*out,const float*x){glm53f_moe_stage_context_12n*c=context;int li=c->active_layer-c->first_layer,selected[8],npart=0;float route_weight[8],part_weight[9];glm53f_expert_part part[9];int8_t router_qx[4096];float router_xs=0;if(li<0||li>=c->layer_count)return-1;double t=c->profile?glm53f_clock():0.0;
if(c->fused_ready){ /* routed+shared experts already ran in the router team (fused layer) */
    c->fused_ready=0;c->router_ready=0;
    int rc=moe_sum(c,c->scratch->local_output,out,4096);
    if(c->profile)c->profile_phase[2]+=glm53f_clock()-t;
    return rc;
}
if(!c->router_ready&&!c->router_i8){
    if (glm53f_team_available()) {
        struct router_call call = {c, x};
        c->router_in_sublayer = 1;
        glm53f_team_dispatch(router_worker, &call);
        c->router_in_sublayer = 0;
    } else {
#pragma omp parallel for schedule(static)
        for(int b=0;b<NEXPERTS/8;b++)router_dot8(c->router_logits+b*8,c->router_w+((size_t)li*NEXPERTS+b*8)*4096,x,4096);
    }
}if(c->router_i8){if(glm53f_i8_quantize_x(router_qx,&router_xs,x,4096))return-1;
#pragma omp parallel for schedule(static)
    for(int q=0;q<NEXPERTS/16;q++){int g=q/4,j=q%4;glm53f_i8_dot16(c->router_logits+q*16,c->router_i8+((size_t)li*NEXPERTS+g*64)*4096+j*64,c->router_i8_scale+(size_t)li*NEXPERTS+q*16,router_qx,router_xs,4096);}}c->router_ready=0;glm53f_router_topk(c->router_logits,c->router_bias+(size_t)li*NEXPERTS,NEXPERTS,8,2.5f,selected,route_weight);if(moe_trace_routes(c,selected,route_weight,1))return-1;if(c->profile){c->profile_phase[0]+=glm53f_clock()-t;t=glm53f_clock();}int table_layer=c->active_layer-FIRST_LAYER;if(c->int8_enabled){if(moe_int8_local(c,c->scratch->local_output,x,c->router_i8?router_qx:NULL,router_xs,selected,route_weight,table_layer))return-1;if(c->profile){c->profile_phase[1]+=glm53f_clock()-t;t=glm53f_clock();}int rc=moe_sum(c,c->scratch->local_output,out,4096);if(c->profile)c->profile_phase[2]+=glm53f_clock()-t;return rc;}for(int k=0;k<8;k++){expert_offset*p=&c->table[table_layer*NEXPERTS+selected[k]];if(p->gate_up==UINT64_MAX)continue;part[npart]=(glm53f_expert_part){c->blob+p->gate_up,p->gate_up_scale==UINT64_MAX?NULL:(const float*)(c->blob+p->gate_up_scale),c->blob+p->down,p->down_scale==UINT64_MAX?NULL:(const float*)(c->blob+p->down_scale),p->inter,p->gate_type,p->down_type};part_weight[npart++]=route_weight[k];}if(c->shared_blob&&!c->nsh_native){shared_offset*p=&c->shared[table_layer];part[npart]=(glm53f_expert_part){c->shared_blob+p->gate_up,(const float*)(c->shared_blob+p->gate_up_scale),c->shared_blob+p->down,(const float*)(c->shared_blob+p->down_scale),p->inter,0,0};part_weight[npart++]=1.0f;}if(!moe_fused_shared(c,table_layer,x,part,part_weight,npart)){glm53f_moe_local_12n(c->scratch->local_output,part,part_weight,npart,x,c->scratch);{double tn=c->profile?glm53f_clock():0.0;if(c->nsh_native&&nsh_accumulate(c,table_layer,x,c->scratch->local_output))return-1;if(c->profile)c->profile_phase[3]+=glm53f_clock()-tn;}}if(c->profile){c->profile_phase[1]+=glm53f_clock()-t;t=glm53f_clock();}int rc=moe_sum(c,c->scratch->local_output,out,4096);if(c->profile)c->profile_phase[2]+=glm53f_clock()-t;return rc;}

void glm53f_moe_stage_profile_reset_12n(glm53f_moe_stage_context_12n *c) {
    if (c) {
        memset(c->profile_phase, 0, sizeof(c->profile_phase));
        memset(c->gn_phase, 0, sizeof(c->gn_phase));
        memset(c->occupancy, 0, sizeof(c->occupancy));
    }
}
void glm53f_moe_stage_profile_report_12n(const glm53f_moe_stage_context_12n *c,
                                        long positions, const char *label) {
    if (!c || !c->profile) return;
    double p[3], minimum[3];
    MPI_Reduce(c->profile_phase,p,3,MPI_DOUBLE,MPI_MAX,0,c->dist?c->dist->tp:MPI_COMM_WORLD);
    MPI_Reduce(c->profile_phase,minimum,3,MPI_DOUBLE,MPI_MIN,0,c->dist?c->dist->tp:MPI_COMM_WORLD);
    if (getenv("GLM53F_PROFILE_RANKS")) {
        double d = positions ? positions : 1;
        printf("GLM53F_MOE_RANK rank=%d router_ms=%.6f local_ms=%.6f allreduce_ms=%.6f\n",
            c->rank,c->profile_phase[0]*1e3/d,c->profile_phase[1]*1e3/d,c->profile_phase[2]*1e3/d);
    }
    if (!c->rank) {
        double d = positions ? positions : 1;
        printf("GLM53F_MOE_PROFILE label=%s router=%.3f local=%.3f allreduce=%.3f nsh_in_local=%.3f ms_pos\n",
               label?label:"target", p[0]*1e3/d, p[1]*1e3/d, p[2]*1e3/d,c->profile_phase[3]*1e3/d);
        printf("GLM53F_MOE_RANK_MIN label=%s router=%.3f local=%.3f allreduce=%.3f ms_pos\n",
               label?label:"target", minimum[0]*1e3/d, minimum[1]*1e3/d, minimum[2]*1e3/d);
        printf("GLM53F_MOE_PROFILE_DETAIL label=%s native_total=%.3f xquant=%.3f native_tasks_wall=%.3f native_avg_busy=%.3f old_expert_loop=%.3f shared=%.3f combine=%.3f tasks_per_call=%.1f ms_pos\n",
               label?label:"target", c->gn_phase[0]*1e3/d, c->gn_phase[1]*1e3/d, c->gn_phase[5]*1e3/d, c->gn_phase[6]*1e3/d,
               c->gn_phase[2]*1e3/d, c->gn_phase[3]*1e3/d, c->gn_phase[4]*1e3/d, c->gn_phase[7]);
        printf("GLM53F_MOE_OCCUPANCY empty=%llu one=%llu two_three=%llu four_seven=%llu eight_plus=%llu\n",
               c->occupancy[0],c->occupancy[1],c->occupancy[2],c->occupancy[3],c->occupancy[4]);
    }
}
static inline int moe_prefill_activation(int8_t *ga, float *scale,
                                          const float *up, int n) {
    float act[512];
    for (int i = 0; i < n; ++i) {
        float g = up[i], u = up[n + i];
        if (g > 10) g = 10;
        if (g < -100) g = -100;
        if (u > 10) u = 10;
        if (u < -10) u = -10;
        act[i] = (g / (1 + expf(-g))) * u;
    }
    if (glm53f_i8_quantize_x(ga, scale, act, n)) {
        memset(ga, 0, n); *scale = 0; return 1;
    }
    return 0;
}

/* One shared workspace for every routed layer. Group all expert panels before
 * entering the team: unlike the previous grouped experiment, no expert or
 * token pays a separate OpenMP launch. The shared expert is the final group. */
static int moe_prefill_int8_local(glm53f_moe_stage_context_12n *c, float *out,
        const float *x, int tokens, int table_layer, const int selected[][8],
        const float route_weight[][8]) {
    enum { H = 4096, MAX_SLOTS = GLM53F_PREFILL_MAX_TOKENS * 9 };
    const size_t input_bytes = (size_t)GLM53F_PREFILL_MAX_TOKENS * H;
    const size_t group_bytes = (size_t)MAX_SLOTS * H;
    const size_t act_bytes = (size_t)MAX_SLOTS * 512;
    const size_t up_bytes = (size_t)MAX_SLOTS * 1024 * sizeof(float);
    const size_t bytes = input_bytes + group_bytes + act_bytes + up_bytes +
        (GLM53F_PREFILL_MAX_TOKENS + 2 * MAX_SLOTS) * sizeof(float);
    if (!c->i8_prefill_storage &&
        posix_memalign(&c->i8_prefill_storage, 256, bytes)) return -1;
    int8_t *qx = c->i8_prefill_storage, *gx = qx + input_bytes;
    int8_t *ga = gx + group_bytes;
    float *up = (float *)(ga + act_bytes);
    float *xs = (float *)((unsigned char *)up + up_bytes);
    float *gs = xs + GLM53F_PREFILL_MAX_TOKENS, *as = gs + MAX_SLOTS;
    int counts[NEXPERTS + 1] = {0}, start[NEXPERTS + 2], cursor[NEXPERTS + 1];
    int pos[MAX_SLOTS], slot[MAX_SLOTS], inter[MAX_SLOTS], slots = 0, panels = 0, failed = 0;
    int panel_size = tokens > 5 && (c->prefill.features & GLM53F_PREFILL_EXPERT16) ? 16 : 8;
    const expert_offset *table = c->table + table_layer * NEXPERTS;
    typedef struct {
        const expert_offset *weight;
        const unsigned char *blob;
        int begin, count;
    } panel_task;
    panel_task task[MAX_SLOTS];
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < 8; ++k)
            if (table[selected[t][k]].gate_up != UINT64_MAX) ++counts[selected[t][k]];
    counts[NEXPERTS] = tokens;
    for (int e = 0; e <= NEXPERTS; ++e) {
        cursor[e] = start[e] = slots;
        slots += counts[e];
    }
    start[NEXPERTS + 1] = slots;
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < 8; ++k) {
            int e = selected[t][k];
            if (table[e].gate_up == UINT64_MAX) continue;
            int s = cursor[e]++;
            pos[s] = t; slot[s] = k;
        }
        int s = cursor[NEXPERTS]++;
        pos[s] = t; slot[s] = 8;
    }
    for (int e = 0; e <= NEXPERTS; ++e)
        for (int s = start[e]; s < start[e + 1]; s += panel_size) {
            int n = start[e + 1] - s;
            if (n > panel_size) n = panel_size;
            task[panels++] = (panel_task){e == NEXPERTS ? c->shared + table_layer : table + e,
                e == NEXPERTS ? c->shared_blob : c->blob, s, n};
            for (int t = 0; t < n; ++t) inter[s + t] = task[panels - 1].weight->inter;
        }
    /* Every owned slot is overwritten across all H down-projection rows.
     * Combine skips nonowned slots, so fast prefill need not clear them. */
    if (panel_size != 16)
        memset(c->batch_routes, 0, (size_t)tokens * 8 * H * sizeof(float));
#pragma omp parallel reduction(|:failed)
    {
#pragma omp for schedule(static)
        for (int t = 0; t < tokens; ++t)
            if (glm53f_i8_quantize_x(qx + (size_t)t * H, xs + t,
                                     x + (size_t)t * H, H)) {
                memset(qx + (size_t)t * H, 0, H); xs[t] = 0; failed = 1;
            }
#pragma omp for schedule(static)
        for (int s = 0; s < slots; ++s) {
            memcpy(gx + (size_t)s * H, qx + (size_t)pos[s] * H, H);
            gs[s] = xs[pos[s]];
        }
#pragma omp for collapse(2) schedule(static)
        for (int p = 0; p < panels; ++p)
            for (int r = 0; r < 1024; r += 16) {
                const panel_task *q = task + p;
                const expert_offset *w = q->weight;
                if (r >= 2 * w->inter) continue;
                glm53f_i8_dot16_batch16(up + (size_t)q->begin * 1024 + r, 1024,
                    (const int8_t *)(q->blob + w->gate_up +
                        (size_t)(r / 64) * 64 * H + (r % 64) * 4),
                    w->i8_gate_scale + r, gx + (size_t)q->begin * H, H,
                    gs + q->begin, q->count, H);
            }
        if (panel_size == 16) {
#pragma omp for schedule(static)
            for (int s = 0; s < slots; ++s)
                failed |= moe_prefill_activation(ga + (size_t)s * 512, as + s,
                                                  up + (size_t)s * 1024, inter[s]);
        } else {
#pragma omp for schedule(static)
            for (int p = 0; p < panels; ++p) {
                const panel_task *q = task + p;
                for (int t = 0; t < q->count; ++t) {
                    int s = q->begin + t;
                    failed |= moe_prefill_activation(ga + (size_t)s * 512, as + s,
                                                      up + (size_t)s * 1024, inter[s]);
                }
            }
        }
#pragma omp for collapse(2) schedule(static)
        for (int p = 0; p < panels; ++p)
            for (int r = 0; r < H; r += 16) {
                const panel_task *q = task + p;
                const expert_offset *w = q->weight;
                float value[16 * 16];
                glm53f_i8_dot16_batch16(value, 16,
                    (const int8_t *)(q->blob + w->down +
                        (size_t)(r / 64) * 64 * w->inter + (r % 64) * 4),
                    w->i8_down_scale + r, ga + (size_t)q->begin * 512, 512,
                    as + q->begin, q->count, w->inter);
                for (int t = 0; t < q->count; ++t) {
                    int s = q->begin + t;
                    float *dest = slot[s] == 8 ? c->batch_local + (size_t)pos[s] * H :
                        c->batch_routes + ((size_t)pos[s] * 8 + slot[s]) * H;
                    memcpy(dest + r, value + t * 16, 16 * sizeof(float));
                }
            }
#pragma omp for schedule(static)
        for (int q = 0; q < tokens * H; ++q) {
            int t = q / H, i = q % H;
            float sum = 0;
            for (int k = 0; k < 8; ++k)
                if (table[selected[t][k]].gate_up != UINT64_MAX)
                    sum += route_weight[t][k] * c->batch_routes[((size_t)t * 8 + k) * H + i];
            out[q] = sum + c->batch_local[q];
        }
    }
    return failed ? -1 : 0;
}


/* ---- grouped prefill directly from native GGUF Q4_K (gate/up) + Q5_K (down) ---------------------------- */
enum { GN_MAXM = 96, GN_LDY_DN = 4096 + 64, GN_MAXINTER = 512 };
typedef struct {
    int8_t *xp, *xp2, *a8;
    float *xsp, *Bg, *ygu, *as, *asp, *mina, *yd;
    uint8_t *cbuf;
    double busy;
} gn_tbuf;
typedef struct { int e, off, n; } gn_task;

void dequantize_row_q4_K(const void *src, float *dst, int n);
void dequantize_row_q5_K(const void *src, float *dst, int n);
void dequantize_row_q6_K(const void *src, float *dst, int n);

static void *gn_alloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 256, n ? n : 256)) return NULL;
    memset(p, 0, n);
    return p;
}

static int gn_thread_init(gn_tbuf *b) {
    b->xp = gn_alloc((size_t)GN_MAXM * 4096);
    b->xsp = gn_alloc((size_t)GN_MAXM * 128 * 4);
    b->Bg = gn_alloc((size_t)GN_MAXM * 128 * 4);
    b->xp2 = gn_alloc((size_t)GN_MAXM * GN_MAXINTER);
    b->a8 = gn_alloc((size_t)GN_MAXM * GN_MAXINTER);
    b->as = gn_alloc((size_t)GN_MAXM * 16 * 4);
    b->asp = gn_alloc((size_t)GN_MAXM * 16 * 4);
    b->ygu = gn_alloc((size_t)GN_MAXM * (2 * GN_MAXINTER + 64) * 4);
    b->mina = gn_alloc((size_t)(4096 / 32) * 64 * 4);
    b->yd = gn_alloc((size_t)GN_MAXM * GN_LDY_DN * 4);
    b->cbuf = gn_alloc((size_t)(GMN_KC / 32) * GMN_BLK);
    return !(b->xp && b->xsp && b->Bg && b->xp2 && b->a8 && b->as && b->asp && b->ygu && b->mina && b->yd && b->cbuf);
}

static int gn_expert_eligible(const glm53f_moe_stage_context_12n *c, const expert_offset *ep) {
    const int q6 = ep->down_type == GLM53F_GGML_Q6_K;
    return ep->gate_up != UINT64_MAX && (ep->gate_type == GLM53F_GGML_Q4_K || ep->gate_type == GLM53F_GGML_Q5_K) &&
           (ep->down_type == GLM53F_GGML_Q5_K || (q6 && ep->inter == 256)) &&
           ep->inter >= 256 && ep->inter % 256 == 0 && ep->inter <= GN_MAXINTER &&
           !(((uintptr_t)(c->blob + ep->gate_up)) & 3) && !(((uintptr_t)(c->blob + ep->down)) & (q6 ? 1 : 3));
}

/* Computes batch_routes rows for every eligible expert and sets handled[e].  x is [tokens][4096] float. */
static int moe_native_grouped(glm53f_moe_stage_context_12n *c, const float *x, int tokens, int table_layer,
                              int (*selected)[8], unsigned char *handled) {
    enum { H = 4096 };
    /* Padding breaks gate/up output-stride aliases in the 256-byte L1 sets.
     * Capture the startup selector once, outside expert tasks. */
    const char *pad_env = getenv("GLM53F_MOE_GU_PAD");
    const int gu_padding = pad_env && atoi(pad_env) ? 64 : 0;
    static const int8_t zrow[H];
    static const float zxs[H / 32];
    if (!c->gn_xq) {
        c->gn_xq = gn_alloc((size_t)GLM53F_PREFILL_MAX_TOKENS * H);
        c->gn_xs = gn_alloc((size_t)GLM53F_PREFILL_MAX_TOKENS * (H / 32) * 4);
        c->gn_bt = gn_alloc((size_t)GLM53F_PREFILL_MAX_TOKENS * (H / 32) * 4);
        c->gn_nthreads = omp_get_max_threads();
        c->gn_threads = calloc((size_t)c->gn_nthreads, sizeof(gn_tbuf));
        if (!c->gn_xq || !c->gn_xs || !c->gn_bt || !c->gn_threads) return -1;
    }
    gn_tbuf *tbs = (gn_tbuf *)c->gn_threads;
    int cnt[NEXPERTS], base[NEXPERTS + 1];
    memset(cnt, 0, sizeof(cnt));
    unsigned char elig[NEXPERTS];
    for (int e = 0; e < NEXPERTS; ++e)
        elig[e] = (unsigned char)gn_expert_eligible(c, &c->table[table_layer * NEXPERTS + e]);
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < 8; ++k)
            if (elig[selected[t][k]]) cnt[selected[t][k]]++;
    base[0] = 0;
    for (int e = 0; e < NEXPERTS; ++e) base[e + 1] = base[e] + cnt[e];
    const int total = base[NEXPERTS];
    if (!total) return 0;
    int *plist = (int *)malloc((size_t)total * 2 * sizeof(int));
    gn_task *tasks = (gn_task *)malloc((size_t)(NEXPERTS + total / GN_MAXM + 1) * sizeof(gn_task));
    if (!plist || !tasks) { free(plist); free(tasks); return -1; }
    {
        int fill[NEXPERTS];
        memset(fill, 0, sizeof(fill));
        for (int t = 0; t < tokens; ++t)
            for (int k = 0; k < 8; ++k) {
                int e = selected[t][k];
                if (!elig[e]) continue;
                int i = base[e] + fill[e]++;
                plist[2 * i] = t;
                plist[2 * i + 1] = k;
            }
    }
    int ntasks = 0;
    for (int e = 0; e < NEXPERTS; ++e)
        for (int off = 0; off < cnt[e]; off += GN_MAXM) {
            tasks[ntasks++] = (gn_task){e, base[e] + off, cnt[e] - off < GN_MAXM ? cnt[e] - off : GN_MAXM};
            handled[e] = 1;
        }
    for (int i = 1; i < ntasks; ++i) { /* largest groups first */
        gn_task v = tasks[i];
        int j = i - 1;
        while (j >= 0 && tasks[j].n < v.n) { tasks[j + 1] = tasks[j]; --j; }
        tasks[j + 1] = v;
    }
    int failed = 0, next = 0;
    double gt0 = c->profile ? glm53f_clock() : 0.0;
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t)
        gmn_quant_row(x + (size_t)t * H, H, c->gn_xq + (size_t)t * H, c->gn_xs + (size_t)t * (H / 32),
                      c->gn_bt + (size_t)t * (H / 32));
    double gt1 = c->profile ? glm53f_clock() : 0.0;
    for (int i = 0; i < c->gn_nthreads; ++i) tbs[i].busy = 0.0;
#pragma omp parallel
    {
        gn_tbuf *B = &tbs[omp_get_thread_num()];
        if (!B->xp && gn_thread_init(B)) {
#pragma omp atomic write
            failed = 1;
        }
        for (; !failed;) {
            int ti = __atomic_fetch_add(&next, 1, __ATOMIC_RELAXED);
            if (ti >= ntasks) break;
            const gn_task tk = tasks[ti];
            double tb0 = c->profile ? glm53f_clock() : 0.0;
            const expert_offset *ep = &c->table[table_layer * NEXPERTS + tk.e];
            const int m = tk.n, mpad = (m + 5) / 6 * 6, inter = ep->inter, nbi = inter / 32;
            for (int g = 0; g < mpad / 6; ++g) {
                const int8_t *rows[6];
                const float *xsr[6];
                for (int u = 0; u < 6; ++u) {
                    int sl = g * 6 + u;
                    if (sl < m) {
                        int t = plist[2 * (tk.off + sl)];
                        rows[u] = c->gn_xq + (size_t)t * H;
                        xsr[u] = c->gn_xs + (size_t)t * (H / 32);
                    } else { rows[u] = zrow; xsr[u] = zxs; }
                }
                gmn_pack6(B->xp + (size_t)g * 6 * H, B->xsp + (size_t)g * 6 * (H / 32), rows, xsr, H);
            }
            for (int sl = 0; sl < mpad; ++sl) {
                if (sl < m) memcpy(B->Bg + (size_t)sl * (H / 32), c->gn_bt + (size_t)plist[2 * (tk.off + sl)] * (H / 32), (H / 32) * 4);
                else memset(B->Bg + (size_t)sl * (H / 32), 0, (H / 32) * 4);
            }
            const size_t ldg = (size_t)2 * inter + gu_padding;
            const int gtype = ep->gate_type == GLM53F_GGML_Q5_K ? GMN_TYPE_Q5K : GMN_TYPE_Q4K;
            gmn_gemm(gtype, c->blob + ep->gate_up, gmn_row_bytes(gtype, H), H, 2 * inter, mpad,
                     B->xp, B->xsp, B->Bg, B->ygu, ldg, B->cbuf, B->mina, 1);
            float *bd = B->Bg; /* gate/up min-correction input is no longer needed */
            if (ep->down_type == GLM53F_GGML_Q6_K) { /* 16-column scale blocks, no min term */
                const int nb16 = inter / 16;
                gmn_swiglu_quant16(B->ygu, ldg, inter, m, mpad, B->a8, B->as);
                for (int g = 0; g < mpad / 6; ++g) {
                    const int8_t *rows[6];
                    const float *xsr[6];
                    for (int u = 0; u < 6; ++u) {
                        rows[u] = B->a8 + (size_t)(g * 6 + u) * inter;
                        xsr[u] = B->as + (size_t)(g * 6 + u) * nb16;
                    }
                    gmn_pack6_sb(B->xp2 + (size_t)g * 6 * inter, B->asp + (size_t)g * 6 * nb16, rows, xsr, inter, 16);
                }
                gmn_gemm_q6(c->blob + ep->down, GMN_Q6K_BYTES * (size_t)(inter / 256), inter, H, mpad,
                            B->xp2, B->asp, B->yd, GN_LDY_DN, B->cbuf);
            } else {
            gmn_swiglu_quant(B->ygu, ldg, inter, m, mpad, B->a8, B->as, bd);
            for (int g = 0; g < mpad / 6; ++g) {
                const int8_t *rows[6];
                const float *xsr[6];
                for (int u = 0; u < 6; ++u) {
                    rows[u] = B->a8 + (size_t)(g * 6 + u) * inter;
                    xsr[u] = B->as + (size_t)(g * 6 + u) * nbi;
                }
                gmn_pack6(B->xp2 + (size_t)g * 6 * inter, B->asp + (size_t)g * 6 * nbi, rows, xsr, inter);
            }
            gmn_gemm(GMN_TYPE_Q5K, c->blob + ep->down, gmn_row_bytes(GMN_TYPE_Q5K, inter), inter, H, mpad,
                     B->xp2, B->asp, bd, B->yd, GN_LDY_DN, B->cbuf, B->mina, 1);
            }
            for (int sl = 0; sl < m; ++sl) {
                const int t = plist[2 * (tk.off + sl)], k = plist[2 * (tk.off + sl) + 1];
                memcpy(c->batch_routes + ((size_t)t * 8 + k) * H, B->yd + (size_t)sl * GN_LDY_DN, H * sizeof(float));
            }
            if (c->profile) B->busy += glm53f_clock() - tb0;
        }
    }
    if (c->profile) {
        double busy = 0;
        for (int i = 0; i < c->gn_nthreads; ++i) busy += tbs[i].busy;
        c->gn_phase[1] += gt1 - gt0;
        c->gn_phase[5] += glm53f_clock() - gt1;
        c->gn_phase[6] += busy / c->gn_nthreads;
        c->gn_phase[7] += ntasks;
    }
    free(plist);
    free(tasks);
    return failed ? -1 : 0;
}



/* ---- shared expert (native Q8_0 GGUF) through the int8 panel64 tile GEMM ------------------------------------ */
static float s8_q80_scale(const uint8_t *p) { _Float16 h; memcpy(&h, p, 2); return (float)h; }

/* Convert `rows` x `cols` Q8_0-family weights to panel64 (sb=32); returns malloc'd buffer of rows/64 panels. */
static uint8_t *s8_convert(int type, const uint8_t *src, int rows, int cols) {
    const size_t pbytes = gk_panel64_bytes(32, cols);
    uint8_t *dst = (uint8_t *)gn_alloc((size_t)(rows / 64) * pbytes);
    if (!dst) return NULL;
    const int nb = cols / 32;
    const size_t rb_r = (size_t)cols + (size_t)nb * 4, rb_q = (size_t)nb * 34;
    int8_t *q = (int8_t *)malloc((size_t)64 * cols);
    float *sc = (float *)malloc((size_t)64 * nb * sizeof(float));
    if (!q || !sc) { free(q); free(sc); free(dst); return NULL; }
    for (int panel = 0; panel < rows / 64; ++panel) {
        for (int r = 0; r < 64; ++r) {
            const int row = panel * 64 + r;
            for (int b = 0; b < nb; ++b) {
                if (type == GLM53F_NATIVE_Q8_0R) {
                    memcpy(q + (size_t)r * cols + b * 32, src + (size_t)row * rb_r + b * 32, 32);
                    memcpy(&sc[(size_t)r * nb + b], src + (size_t)row * rb_r + cols + (size_t)b * 4, 4);
                } else if (type == GLM53F_GGML_Q8_0) {
                    const uint8_t *blk = src + (size_t)row * rb_q + (size_t)b * 34;
                    sc[(size_t)r * nb + b] = s8_q80_scale(blk);
                    memcpy(q + (size_t)r * cols + b * 32, blk + 2, 32);
                } else { /* GLM53F_NATIVE_Q8_0R16: 16-row panels, 576 B per 32 columns */
                    const size_t pb16 = (size_t)nb * 576;
                    const uint8_t *blk = src + (size_t)(row / 16) * pb16 + (size_t)b * 576;
                    const int rr = row % 16;
                    for (int j = 0; j < 8; ++j) memcpy(q + (size_t)r * cols + b * 32 + 4 * j, blk + j * 64 + rr * 4, 4);
                    memcpy(&sc[(size_t)r * nb + b], blk + 512 + rr * 4, 4);
                }
            }
        }
        gk_pack_panel64(32, dst + (size_t)panel * pbytes, q, sc, 64, cols);
    }
    free(q); free(sc);
    return dst;
}

static int moe_shared_q8(glm53f_moe_stage_context_12n *c, float *dst, const float *x, int tokens, int layer) {
    enum { H = 4096, LDY_DN = 4096 + 64 };
    const int in = c->nsh_in;
    if (!nsh_is_q80(c->nsh_gt[layer]) || !nsh_is_q80(c->nsh_ut[layer]) || !nsh_is_q80(c->nsh_dt[layer]) ||
        in < 64 || in % 64 || in > (c->dist?512:256) || c->nsh_gt[layer] != c->nsh_ut[layer]) return -2;
    if (!c->s8_gu[layer]) {
        /* gate rows first, then up rows, in one panel64 matrix of 2*in rows */
        uint8_t *g = s8_convert(c->nsh_gt[layer], c->nsh_g[layer], in, H);
        uint8_t *u = s8_convert(c->nsh_ut[layer], c->nsh_u[layer], in, H);
        uint8_t *d = s8_convert(c->nsh_dt[layer], c->nsh_d[layer], H, in);
        if (!g || !u || !d) { free(g); free(u); free(d); return -1; }
        const size_t gb = (size_t)(in / 64) * gk_panel64_bytes(32, H);
        uint8_t *gu = (uint8_t *)gn_alloc(2 * gb);
        if (!gu) { free(g); free(u); free(d); return -1; }
        memcpy(gu, g, gb); memcpy(gu + gb, u, gb);
        free(g); free(u);
        c->s8_gu[layer] = gu; c->s8_dn[layer] = d;
    }
    const int T = GLM53F_PREFILL_MAX_TOKENS + 6;
    if (!c->s8_xq) {
        const size_t width=c->dist?512u:256u, blocks=width/32;
        c->s8_xq = (int8_t *)gn_alloc((size_t)T * H); c->s8_xs = (float *)gn_alloc((size_t)T * (H / 32) * 4);
        c->s8_bt = (float *)gn_alloc((size_t)T * (H / 32) * 4);
        c->s8_xp = (int8_t *)gn_alloc((size_t)T * H); c->s8_xsp = (float *)gn_alloc((size_t)T * (H / 32) * 4);
        c->s8_ygu = (float *)gn_alloc((size_t)T * 2 * width * 4);
        c->s8_a8 = (int8_t *)gn_alloc((size_t)T * width); c->s8_as = (float *)gn_alloc((size_t)T * blocks * 4); c->s8_bd = (float *)gn_alloc((size_t)T * blocks * 4);
        c->s8_xp2 = (int8_t *)gn_alloc((size_t)T * width); c->s8_asp = (float *)gn_alloc((size_t)T * blocks * 4);
        c->s8_y = (float *)gn_alloc((size_t)T * LDY_DN * 4);
        if (!c->s8_xq || !c->s8_xs || !c->s8_bt || !c->s8_xp || !c->s8_xsp || !c->s8_ygu || !c->s8_a8 || !c->s8_as ||
            !c->s8_bd || !c->s8_xp2 || !c->s8_asp || !c->s8_y) return -1;
    }
    const int mpad = (tokens + 5) / 6 * 6, ngroups = mpad / 6, gu_rows = 2 * in, nb = H / 32;
    int8_t *zrow = c->s8_xq + (size_t)tokens * H; /* rows tokens..mpad-1 stay zero */
    (void)zrow;
#pragma omp parallel for schedule(static)
    for (int tk = 0; tk < mpad; ++tk) {
        if (tk < tokens) gmn_quant_row(x + (size_t)tk * H, H, c->s8_xq + (size_t)tk * H, c->s8_xs + (size_t)tk * nb, c->s8_bt + (size_t)tk * nb);
        else { memset(c->s8_xq + (size_t)tk * H, 0, H); memset(c->s8_xs + (size_t)tk * nb, 0, nb * 4); }
    }
#pragma omp parallel for schedule(static)
    for (int g = 0; g < ngroups; ++g) {
        const int8_t *rows[6]; const float *xsr[6];
        for (int u = 0; u < 6; ++u) { rows[u] = c->s8_xq + (size_t)(g * 6 + u) * H; xsr[u] = c->s8_xs + (size_t)(g * 6 + u) * nb; }
        gmn_pack6(c->s8_xp + (size_t)g * 6 * H, c->s8_xsp + (size_t)g * 6 * nb, rows, xsr, H);
    }
    const int gch = 8, nch = (ngroups + gch - 1) / gch, gpanels = gu_rows / 64;
#pragma omp parallel for schedule(dynamic, 1)
    for (int task = 0; task < gpanels * nch; ++task) {
        const int pnl = task / nch, ch = task % nch;
        const int t0 = ch * gch * 6, t1 = ((ch + 1) * gch * 6 < mpad) ? (ch + 1) * gch * 6 : mpad;
        gk_gemm_panel64(32, c->s8_gu[layer], H, pnl * 64, pnl * 64 + 64, t0, t1, c->s8_xp, c->s8_xsp, c->s8_ygu, (size_t)gu_rows);
    }
    gmn_swiglu_quant(c->s8_ygu, (size_t)gu_rows, in, tokens, mpad, c->s8_a8, c->s8_as, c->s8_bd);
    const int nbi = in / 32;
#pragma omp parallel for schedule(static)
    for (int g = 0; g < ngroups; ++g) {
        const int8_t *rows[6]; const float *xsr[6];
        for (int u = 0; u < 6; ++u) { rows[u] = c->s8_a8 + (size_t)(g * 6 + u) * in; xsr[u] = c->s8_as + (size_t)(g * 6 + u) * nbi; }
        gmn_pack6(c->s8_xp2 + (size_t)g * 6 * in, c->s8_asp + (size_t)g * 6 * nbi, rows, xsr, in);
    }
    const int dch = 4, dnch = (ngroups + dch - 1) / dch;
#pragma omp parallel for schedule(dynamic, 1)
    for (int task = 0; task < (H / 64) * dnch; ++task) {
        const int pnl = task / dnch, ch = task % dnch;
        const int t0 = ch * dch * 6, t1 = ((ch + 1) * dch * 6 < mpad) ? (ch + 1) * dch * 6 : mpad;
        gk_gemm_panel64(32, c->s8_dn[layer], in, pnl * 64, pnl * 64 + 64, t0, t1, c->s8_xp2, c->s8_asp, c->s8_y, (size_t)LDY_DN);
    }
#pragma omp parallel for schedule(static)
    for (int tk = 0; tk < tokens; ++tk) memcpy(dst + (size_t)tk * H, c->s8_y + (size_t)tk * LDY_DN, H * sizeof(float));
    return 0;
}

/* Shared expert for a token batch through the bf16 tile GEMM. dst is [tokens][4096]. Returns 0, or -2 if unsupported. */
static int moe_shared_gemm(glm53f_moe_stage_context_12n *c, float *dst, const float *x, int tokens, int table_layer) {
    enum { H = 4096 };
    if (c->nsh_native) return moe_shared_q8(c, dst, x, tokens, table_layer);
    const shared_offset *sp = &c->shared[table_layer];
    const int inter = sp->inter, gu_rows = 2 * inter;
    if (!c->shared_blob || c->nsh_native || inter < 16 || gu_rows % 32 || inter > 512) {
        static int said;
        if (!said++ && !c->rank) fprintf(stderr, "GLM53F_MOE_SHARED_GEMM unsupported: shared_blob=%d nsh_native=%d inter=%d\n", c->shared_blob != NULL, c->nsh_native, inter);
        return -2;
    }
    if (!c->sg_gu[table_layer]) {
        uint16_t *gu = gn_alloc((size_t)gu_rows * H * sizeof(uint16_t)), *dn = gn_alloc((size_t)H * inter * sizeof(uint16_t));
        if (!gu || !dn) { free(gu); free(dn); return -1; }
        gmn_fp8_pack_tiles(gu, c->shared_blob + sp->gate_up, gu_rows, H);
        gmn_fp8_pack_tiles(dn, c->shared_blob + sp->down, H, inter);
        c->sg_gu[table_layer] = gu; c->sg_dn[table_layer] = dn;
    }
    if (!c->sg_up) {
        c->sg_up = gn_alloc((size_t)GLM53F_PREFILL_MAX_TOKENS * 2 * 512 * sizeof(float));
        c->sg_act = gn_alloc((size_t)GLM53F_PREFILL_MAX_TOKENS * 512 * sizeof(float));
        if (!c->sg_up || !c->sg_act) return -1;
    }
    const uint16_t *gu = c->sg_gu[table_layer], *dn = c->sg_dn[table_layer];
    const float *gscale = (const float *)(c->shared_blob + sp->gate_up_scale), *dscale = (const float *)(c->shared_blob + sp->down_scale);
    const int gblocks = (H + 127) / 128, dblocks = (inter + 127) / 128;
    const int ngroups = (tokens + 5) / 6, gch = 4, nch = (ngroups + gch - 1) / gch;
    const int gtiles = gu_rows / 32, dtiles = H / 32;
#pragma omp parallel for schedule(dynamic, 1)
    for (int task = 0; task < gtiles * nch; ++task) {
        const int tile = task / nch, ch = task % nch;
        for (int g = ch * gch; g < ngroups && g < (ch + 1) * gch; ++g) {
            const int t0 = g * 6, n = tokens - t0 < 6 ? tokens - t0 : 6;
            gmn_fp8g_run(c->sg_up + (size_t)t0 * gu_rows + tile * 32, gu_rows, gu + (size_t)tile * H * 32,
                         gscale + (size_t)((tile * 32) / 128) * gblocks, x + (size_t)t0 * H, H, H, n);
        }
    }
    {
        const svbool_t pg = svptrue_b32();
#pragma omp parallel for schedule(static)
        for (int t = 0; t < tokens; ++t) {
            const float *g = c->sg_up + (size_t)t * gu_rows, *u = g + inter;
            for (int i = 0; i < inter; i += 16) {
                const svbool_t p = svwhilelt_b32(i, inter);
                svfloat32_t gv = svmax_n_f32_x(p, svmin_n_f32_x(p, svld1_f32(p, g + i), 10.f), -100.f);
                svfloat32_t uv = svmax_n_f32_x(p, svmin_n_f32_x(p, svld1_f32(p, u + i), 10.f), -10.f);
                svst1_f32(p, c->sg_act + (size_t)t * inter + i,
                          svmul_f32_x(p, svdiv_f32_x(p, gv, svadd_n_f32_x(p, gmn_expf(p, svneg_f32_x(p, gv)), 1.f)), uv));
            }
        }
        (void)pg;
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (int task = 0; task < dtiles * nch; ++task) {
        const int tile = task / nch, ch = task % nch;
        for (int g = ch * gch; g < ngroups && g < (ch + 1) * gch; ++g) {
            const int t0 = g * 6, n = tokens - t0 < 6 ? tokens - t0 : 6;
            gmn_fp8g_run(dst + (size_t)t0 * H + tile * 32, H, dn + (size_t)tile * inter * 32,
                         dscale + (size_t)((tile * 32) / 128) * dblocks, c->sg_act + (size_t)t0 * inter, inter, inter, n);
        }
    }
    return 0;
}

/* Returns 0 for the legacy combine, 1 for vector combine, 2 when output has
 * also been reduced. One bounded slab becomes visible after every worker has
 * completed it; input remains live until the owner reports all slabs done. */
static int moe_prefill_combine(glm53f_moe_stage_context_12n *c, float *out,
        int tokens, int table_layer, const int selected[][8], const float weight[][8]) {
    const char *setting = getenv("GLM53F_MOE_COMBINE");
    int mode = setting ? atoi(setting) : 0;
    if (!mode) return 0;
    const float *route[GLM53F_PREFILL_MAX_TOKENS][8];
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < 8; ++k)
            route[t][k] = c->table[table_layer * NEXPERTS + selected[t][k]].gate_up == UINT64_MAX ?
                NULL : c->batch_routes + ((size_t)t * 8 + k) * 4096;
    int overlap = !c->dist && mode == 2 && glm53f_async_available_12n();
    int slab = overlap ? c->prefill.slab_tokens : tokens;
    if (overlap && glm53f_async_begin_12n(c->batch_local, out, tokens, 4096, slab)) return -1;
#pragma omp parallel
    {
        for (int base = 0; base < tokens; base += slab) {
            int end = base + slab < tokens ? base + slab : tokens;
#pragma omp for collapse(2) schedule(static)
            for (int t = base; t < end; ++t)
                for (int r = 0; r < 4096; r += 64)
                    glm53f_moe_combine_rows(c->batch_local + (size_t)t * 4096,
                        route[t], weight[t], r, r + 64);
#pragma omp single
            { if (overlap) glm53f_async_ready_12n(end); }
        }
    }
    if (overlap && glm53f_async_finish_12n()) return -1;
    return overlap ? 2 : 1;
}

static int moe_prefill_grouped(glm53f_moe_stage_context_12n *c, float *out,
        const float *x, int tokens, int li, int table_layer) {
    enum { H = 4096, PANEL = 4 };
    int selected[GLM53F_PREFILL_MAX_TOKENS][8];
    float route_weight[GLM53F_PREFILL_MAX_TOKENS][8];
    double begin = c->profile ? glm53f_clock() : 0.0;
    if (c->rg_mode && c->router_wt) {
        const char *tiles = getenv("GLM53F_MOE_ROUTER_TILES12");
        const int mode = tiles ? atoi(tiles) : 0;
        const int tile_mode = (mode >= 1 && mode <= 3) && svcntw() == 16 &&
            tokens > (mode == 2 ? 8 : 6) && c->rg_mode == 1 ? mode : 0;
        const uint16_t *wl = c->router_wt + (size_t)li * NEXPERTS * H;
        gmn_router_prefill(c->batch_router, wl, x, tokens, H, tile_mode);
    }
    if (!c->rg_mode || c->rg_mode == 2 || !c->router_wt) {
        float *ref = c->rg_mode == 2 ? (float *)malloc((size_t)tokens * NEXPERTS * sizeof(float)) : c->batch_router;
        for (int base = 0; base < tokens; base += PANEL) {
            int n = tokens - base;
            if (n > PANEL) n = PANEL;
            glm53f_mv_bf16_batch(ref + (size_t)base * NEXPERTS,
                c->router_w + (size_t)li * NEXPERTS * H,
                x + (size_t)base * H, n, NEXPERTS, H);
        }
        if (c->rg_mode == 2 && ref) { /* verify: compare against the GEMM logits and their top-8 sets */
            double se = 0, sr = 0; int flips = 0;
            for (int t = 0; t < tokens; ++t) {
                int sa[8], sb[8]; float wa[8], wb[8];
                glm53f_router_topk(c->batch_router + (size_t)t * NEXPERTS, c->router_bias + (size_t)li * NEXPERTS, NEXPERTS, 8, 2.5f, sa, wa);
                glm53f_router_topk(ref + (size_t)t * NEXPERTS, c->router_bias + (size_t)li * NEXPERTS, NEXPERTS, 8, 2.5f, sb, wb);
                for (int a = 0; a < 8; ++a) { int f = 0; for (int b = 0; b < 8; ++b) f |= sa[a] == sb[b]; flips += !f; }
                for (int e = 0; e < NEXPERTS; ++e) { double d = (double)c->batch_router[(size_t)t * NEXPERTS + e] - ref[(size_t)t * NEXPERTS + e]; se += d * d; sr += (double)ref[(size_t)t * NEXPERTS + e] * ref[(size_t)t * NEXPERTS + e]; }
            }
            if (!c->rank) fprintf(stderr, "GLM53F_MOE_ROUTER_VERIFY layer=%d tokens=%d rel_l2=%.3e topk_slots_changed=%d of %d\n", c->active_layer, tokens, sqrt(se / (sr + 1e-30)), flips, tokens * 8);
            free(ref);
        }
    }
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t)
        glm53f_router_topk(c->batch_router + (size_t)t * NEXPERTS,
            c->router_bias + (size_t)li * NEXPERTS, NEXPERTS, 8, 2.5f,
            selected[t], route_weight[t]);
    if (c->profile) {
        c->profile_phase[0] += glm53f_clock() - begin;
        begin = glm53f_clock();
    }
    if(moe_trace_routes(c,&selected[0][0],&route_weight[0][0],tokens))return-1;
    moe_profile_occupancy(c, &selected[0][0], tokens);
    /* batch_routes rows of experts without a part on this rank are never read (see the combine below). */
    unsigned char native_done[NEXPERTS];
    memset(native_done, 0, sizeof(native_done));
    double gp0 = c->profile ? glm53f_clock() : 0.0;
    if (c->gn_mode && moe_native_grouped(c, x, tokens, table_layer, selected, native_done)) return -1;
    double gp1 = c->profile ? glm53f_clock() : 0.0;
    if (c->gn_mode) c->gn_phase[0] += gp1 - gp0;
    if (c->gn_mode == 2) { /* verify: keep the native rows, recompute everything with the decode kernels */
        if (!c->gn_verify) c->gn_verify = gn_alloc((size_t)GLM53F_PREFILL_MAX_TOKENS * 8 * H * sizeof(float));
        if (!c->gn_verify) return -1;
        memcpy(c->gn_verify, c->batch_routes, (size_t)tokens * 8 * H * sizeof(float));
    }
    unsigned char verify_mask[NEXPERTS];
    memcpy(verify_mask, native_done, sizeof(verify_mask));
    if (c->gn_mode == 2) memset(native_done, 0, sizeof(native_done));
    for (int e = 0; e < NEXPERTS; ++e) {
        expert_offset *ep = &c->table[table_layer * NEXPERTS + e];
        if (ep->gate_up == UINT64_MAX || native_done[e]) continue;
        glm53f_expert_part part = {c->blob + ep->gate_up,
            ep->gate_up_scale == UINT64_MAX ? NULL :
                (const float *)(c->blob + ep->gate_up_scale),
            c->blob + ep->down,
            ep->down_scale == UINT64_MAX ? NULL :
                (const float *)(c->blob + ep->down_scale),
            ep->inter, ep->gate_type, ep->down_type};
        if (part.gate_type || part.down_type) {
            /* GGUF K-quant/IQ routed parts: the FP8 panel kernel cannot read
             * them.  Use the decode kernel per selected token (the part is
             * L2-resident across its tokens), matching decode arithmetic. */
            const glm53f_iq_part iq = {part.gate_up, part.down,
                part.gate_type, part.down_type, part.inter};
            const float one = 1.0f;
            for (int t = 0; t < tokens; ++t)
                for (int k = 0; k < 8; ++k)
                    if (selected[t][k] == e &&
                        glm53f_iq_expert_weighted(
                            c->batch_routes + ((size_t)t * 8 + k) * H, &iq, &one, 1,
                            x + (size_t)t * H, c->batch_up, c->batch_activation))
                        return -1;
            continue;
        }
        int pos[PANEL], slot[PANEL], count = 0;
        for (int t = 0; t < tokens; ++t)
            for (int k = 0; k < 8; ++k)
                if (selected[t][k] == e) {
                    pos[count] = t;
                    slot[count] = k;
                    memcpy(c->batch_group_x + (size_t)count * H,
                           x + (size_t)t * H, H * sizeof(float));
                    if (++count == PANEL) {
                        glm53f_expert_tokens_bits(&part, count, c->batch_group_x,
                            c->batch_up, c->batch_activation, c->batch_shared);
#pragma omp parallel for schedule(static)
                        for (int q = 0; q < count * H; ++q) {
                            int g = q / H, i = q - g * H;
                            c->batch_routes[((size_t)pos[g] * 8 + slot[g]) * H + i] =
                                c->batch_shared[(size_t)g * H + i];
                        }
                        count = 0;
                    }
                }
        if (count) {
            glm53f_expert_tokens_bits(&part, count, c->batch_group_x,
                c->batch_up, c->batch_activation, c->batch_shared);
#pragma omp parallel for schedule(static)
            for (int q = 0; q < count * H; ++q) {
                int g = q / H, i = q - g * H;
                c->batch_routes[((size_t)pos[g] * 8 + slot[g]) * H + i] =
                    c->batch_shared[(size_t)g * H + i];
            }
        }
    }
    double gp2 = c->profile ? glm53f_clock() : 0.0;
    c->gn_phase[2] += c->profile ? gp2 - gp1 : 0.0;
    if (c->gn_mode == 2) {
        double se = 0, sr = 0;
        long rows_checked = 0;
        for (int t = 0; t < tokens; ++t)
            for (int k = 0; k < 8; ++k) {
                if (!verify_mask[selected[t][k]]) continue;
                const float *a = c->gn_verify + ((size_t)t * 8 + k) * H, *b = c->batch_routes + ((size_t)t * 8 + k) * H;
                for (int i = 0; i < H; ++i) { double d = (double)a[i] - b[i]; se += d * d; sr += (double)b[i] * b[i]; }
                ++rows_checked;
            }
        if (!c->rank)
            fprintf(stderr, "GLM53F_MOE_NATIVE_VERIFY layer=%d tokens=%d rows=%ld rel_l2=%.4e\n",
                    c->active_layer, tokens, rows_checked, sqrt(se / (sr + 1e-30)));
        /* exact fp32 reference (dequantized weights, fp32 activations) for up to 3 tokens of the first native expert */
        int e0 = -1;
        for (int e = 0; e < NEXPERTS && e0 < 0; ++e) if (verify_mask[e]) e0 = e;
        for (int e = 0; e < NEXPERTS; ++e) /* prefer an expert with Q6_K down when the layer has one */
            if (verify_mask[e] && c->table[table_layer * NEXPERTS + e].down_type == GLM53F_GGML_Q6_K) { e0 = e; break; }
        if (e0 >= 0 && !c->rank) {
            const expert_offset *ep = &c->table[table_layer * NEXPERTS + e0];
            const int inter = ep->inter;
            float *wg = (float *)malloc((size_t)2 * inter * H * sizeof(float)), *wd = (float *)malloc((size_t)H * inter * sizeof(float));
            const int dq6 = ep->down_type == GLM53F_GGML_Q6_K;
            const int gq5 = ep->gate_type == GLM53F_GGML_Q5_K;
            const size_t grb = (gq5 ? 176 : 144) * (H / 256), drb = (dq6 ? 210 : 176) * (size_t)(inter / 256);
            if (wg && wd) {
                for (int r = 0; r < 2 * inter; ++r) {
                    if (gq5) dequantize_row_q5_K(c->blob + ep->gate_up + (size_t)r * grb, wg + (size_t)r * H, H);
                    else dequantize_row_q4_K(c->blob + ep->gate_up + (size_t)r * grb, wg + (size_t)r * H, H);
                }
                for (int r = 0; r < H; ++r) {
                    if (dq6) dequantize_row_q6_K(c->blob + ep->down + (size_t)r * drb, wd + (size_t)r * inter, inter);
                    else dequantize_row_q5_K(c->blob + ep->down + (size_t)r * drb, wd + (size_t)r * inter, inter);
                }
                double en = 0, eo = 0, sy = 0; int used = 0;
                for (int t = 0; t < tokens && used < 3; ++t)
                    for (int k = 0; k < 8 && used < 3; ++k) {
                        if (selected[t][k] != e0) continue;
                        float a[GN_MAXINTER];
                        for (int j = 0; j < inter; ++j) {
                            double g = 0, u = 0;
                            for (int i = 0; i < H; ++i) { g += (double)wg[(size_t)j * H + i] * x[(size_t)t * H + i]; u += (double)wg[(size_t)(inter + j) * H + i] * x[(size_t)t * H + i]; }
                            float gf = (float)g, uf = (float)u;
                            if (gf > 10) gf = 10; if (gf < -100) gf = -100; if (uf > 10) uf = 10; if (uf < -10) uf = -10;
                            a[j] = gf / (1.0f + expf(-gf)) * uf;
                        }
                        for (int r = 0; r < H; ++r) {
                            double y = 0;
                            for (int j = 0; j < inter; ++j) y += (double)wd[(size_t)r * inter + j] * a[j];
                            double dn = c->gn_verify[((size_t)t * 8 + k) * H + r] - y, dold = c->batch_routes[((size_t)t * 8 + k) * H + r] - y;
                            en += dn * dn; eo += dold * dold; sy += y * y;
                        }
                        ++used;
                    }
                fprintf(stderr, "GLM53F_MOE_NATIVE_EXACT layer=%d expert=%d rows=%d err_native=%.4e err_decode_kernel=%.4e\n",
                        c->active_layer, e0, used, sqrt(en / (sy + 1e-30)), sqrt(eo / (sy + 1e-30)));
            }
            free(wg); free(wd);
        }
    }
    shared_offset *sp = &c->shared[table_layer];
    glm53f_expert_part shared = {0};
    if (!c->nsh_native) shared = (glm53f_expert_part){c->shared_blob + sp->gate_up,
        (const float *)(c->shared_blob + sp->gate_up_scale),
        c->shared_blob + sp->down,
        (const float *)(c->shared_blob + sp->down_scale), sp->inter, 0, 0};
    int shared_done = 0;
    if (c->sg_mode) {
        if (!c->sg_out && c->sg_mode == 2) c->sg_out = gn_alloc((size_t)GLM53F_PREFILL_MAX_TOKENS * H * sizeof(float));
        int rc = moe_shared_gemm(c, c->sg_mode == 2 ? c->sg_out : c->batch_local, x, tokens, table_layer);
        if (rc == -1) return -1;
        shared_done = rc == 0 && c->sg_mode != 2;
    }
    if (!shared_done)
    for (int base = 0; base < tokens; base += PANEL) {
        int n = tokens - base;
        if (n > PANEL) n = PANEL;
        if (c->nsh_native) {
            if (nsh_batch(c, table_layer, x + (size_t)base * H, n)) return -1;
        } else glm53f_expert_tokens_bits(&shared, n, x + (size_t)base * H,
            c->batch_up, c->batch_activation, c->batch_shared);
#pragma omp parallel for schedule(static)
        for (int q = 0; q < n * H; ++q)
            c->batch_local[(size_t)base * H + q] = c->batch_shared[q];
    }
    if (c->sg_mode == 2 && c->sg_out) {
        double se = 0, sr = 0;
        for (size_t i = 0; i < (size_t)tokens * H; ++i) { double d = (double)c->sg_out[i] - c->batch_local[i]; se += d * d; sr += (double)c->batch_local[i] * c->batch_local[i]; }
        if (!c->rank) fprintf(stderr, "GLM53F_MOE_SHARED_VERIFY layer=%d tokens=%d rel_l2=%.3e\n", c->active_layer, tokens, sqrt(se / (sr + 1e-30)));
    }
    double gp3 = c->profile ? glm53f_clock() : 0.0;
    c->gn_phase[3] += c->profile ? gp3 - gp2 : 0.0;
    int combined = moe_prefill_combine(c, out, tokens, table_layer, selected, route_weight);
    if (combined < 0) return -1;
    if (!combined) {
#pragma omp parallel for schedule(static)
    for (int q = 0; q < tokens * H; ++q) {
        int t = q / H, i = q - t * H;
        float routed = 0.0f;
        for (int k = 0; k < 8; ++k)
            if (c->table[table_layer * NEXPERTS + selected[t][k]].gate_up != UINT64_MAX)
                routed += route_weight[t][k] *
                    c->batch_routes[((size_t)t * 8 + k) * H + i];
        c->batch_local[q] += routed;
    }
    }
    if (c->profile) c->gn_phase[4] += glm53f_clock() - gp3;
    if (c->profile) {
        c->profile_phase[1] += glm53f_clock() - begin;
        begin = glm53f_clock();
    }
    if (combined == 2) return 0;
    if (c->ar_slab > 32) { /* whole-chunk MPI_Allreduce calls: 3.4 ms per 512 tokens vs 17 ms for 4-token panels */
        for (int base = 0; base < tokens; base += c->ar_slab) {
            int n = tokens - base < c->ar_slab ? tokens - base : c->ar_slab;
            if (moe_sum_mpi(c,c->batch_local + (size_t)base * H, out + (size_t)base * H, n * H)) return -1;
        }
    } else if (c->ar_slab > 0) { /* uTofu-wrapper slabs of at most 32 tokens */
        if (moe_sum_slabs(c,c->batch_local, out, tokens, H, c->ar_slab)) return -1;
    } else for (int base = 0; base < tokens; base += PANEL) {
        int n = tokens - base;
        if (n > PANEL) n = PANEL;
        if (moe_sum(c,c->batch_local + (size_t)base * H,
                                    out + (size_t)base * H, n * H)) return -1;
    }
    if (c->profile) c->profile_phase[2] += glm53f_clock() - begin;
    return 0;
}

int glm53f_moe_stage_sublayer_batch_12n(glm53f_moe_stage_context_12n*c,float*out,const float*x,int tokens){
    int li=c?c->active_layer-c->first_layer:-1,table_layer=c?c->active_layer-FIRST_LAYER:-1;
    if(!c||!out||!x||tokens<1||tokens>GLM53F_PREFILL_MAX_TOKENS||li<0||li>=c->layer_count||(!c->shared_blob&&!c->nsh_native))return-1;
    if (tokens > 4 && !c->int8_enabled)
        return moe_prefill_grouped(c, out, x, tokens, li, table_layer);
    if (c->int8_enabled) {
        int grouped = getenv("GLM53F_MOE_I8_GROUPED") &&
                      atoi(getenv("GLM53F_MOE_I8_GROUPED"));
        int router_prefill = tokens > 5 && getenv("GLM53F_MOE_ROUTER_PREFILL") &&
                             atoi(getenv("GLM53F_MOE_ROUTER_PREFILL"));
        int batch_router = tokens > 1 && !c->router_i8 &&
            (grouped || router_prefill || (getenv("GLM53F_MOE_I8_BATCH_ROUTER") &&
            atoi(getenv("GLM53F_MOE_I8_BATCH_ROUTER"))));
        if (batch_router) {
            double begin = c->profile ? glm53f_clock() : 0.0;
            int selected[GLM53F_PREFILL_MAX_TOKENS][8];
            float route_weight[GLM53F_PREFILL_MAX_TOKENS][8];
            const uint16_t *router = c->router_w + (size_t)li * NEXPERTS * 4096;
            if (router_prefill) {
#pragma omp parallel
                {
#pragma omp for collapse(2) schedule(static)
                    for (int r = 0; r < NEXPERTS; r += 4)
                        for (int base = 0; base < tokens; base += 4) {
                            int n = tokens - base;
                            if (n > 4) n = 4;
                            glm53f_matvec_bf16_4x4(
                                c->batch_router + (size_t)base * NEXPERTS + r,
                                NEXPERTS, router + (size_t)r * 4096,
                                x + (size_t)base * 4096, n, 4096);
                        }
#pragma omp for schedule(static)
                    for (int t = 0; t < tokens; ++t)
                        glm53f_router_topk(c->batch_router + (size_t)t * NEXPERTS,
                            c->router_bias + (size_t)li * NEXPERTS, NEXPERTS, 8,
                            2.5f, selected[t], route_weight[t]);
                }
            } else {
                for (int base = 0; base < tokens; base += 4) {
                    int n = tokens - base;
                    if (n > 4) n = 4;
#pragma omp parallel for schedule(static)
                    for (int r = 0; r < NEXPERTS; r += 4)
                        glm53f_matvec_bf16_4x4(
                            c->batch_router + (size_t)base * NEXPERTS + r,
                            NEXPERTS, router + (size_t)r * 4096,
                            x + (size_t)base * 4096, n, 4096);
                }
                for (int t = 0; t < tokens; ++t)
                    glm53f_router_topk(c->batch_router + (size_t)t * NEXPERTS,
                        c->router_bias + (size_t)li * NEXPERTS, NEXPERTS, 8,
                        2.5f, selected[t], route_weight[t]);
            }
            if (c->profile) {
                c->profile_phase[0] += glm53f_clock() - begin;
                begin = glm53f_clock();
            }
            if(moe_trace_routes(c,&selected[0][0],&route_weight[0][0],tokens))return-1;
    moe_profile_occupancy(c, &selected[0][0], tokens);
            if (grouped) {
                if (moe_prefill_int8_local(c, c->batch_local, x, tokens,
                                           table_layer, selected, route_weight)) return -1;
                if (c->profile) {
                    c->profile_phase[1] += glm53f_clock() - begin;
                    begin = glm53f_clock();
                }
                int slab = tokens > 5 && (c->prefill.features & GLM53F_PREFILL_COMM) ?
                           c->prefill.slab_tokens : 1;
                if (moe_sum_slabs(c,c->batch_local, out,
                                                   tokens, 4096, slab)) return -1;
                if (c->profile) c->profile_phase[2] += glm53f_clock() - begin;
                return 0;
            }
            for (int t = 0; t < tokens; ++t) {
                if (moe_int8_local(c, c->scratch->local_output,
                        x + (size_t)t * 4096, NULL, 0.0f, selected[t],
                        route_weight[t], table_layer)) return -1;
                if (c->profile) {
                    c->profile_phase[1] += glm53f_clock() - begin;
                    begin = glm53f_clock();
                }
                if (moe_sum(c,c->scratch->local_output,
                        out + (size_t)t * 4096, 4096)) return -1;
                if (c->profile) {
                    c->profile_phase[2] += glm53f_clock() - begin;
                    begin = glm53f_clock();
                }
            }
            return 0;
        }
        for (int t = 0; t < tokens; ++t)
            if (glm53f_moe_stage_sublayer_12n(c, out + (size_t)t * 4096,
                                             x + (size_t)t * 4096)) return -1;
        return 0;
    }
    enum{MAXP=9,H=4096}; glm53f_expert_part parts[4*MAXP]; float weights[4*MAXP]; int counts[4];
    memset(parts,0,sizeof(parts)); memset(weights,0,sizeof(weights));
    /* Verify batches: one team streams the router weights once for all
     * positions; each logit keeps glm53f_dot_bf16_sve's chain (see router_dot8). */
    static float verify_logits[4][NEXPERTS];
    const char *verify_router_env = getenv("GLM53F_MOE_VERIFY_ROUTER");
    const int verify_router = tokens > 1 && tokens <= 4 && svcntw() == 16 &&
        verify_router_env && atoi(verify_router_env);
    if (verify_router) {
#pragma omp parallel for schedule(static)
        for (int b = 0; b < NEXPERTS / 4; ++b)
            glm53f_bf16_dot4x(&verify_logits[0][b * 4], NEXPERTS,
                c->router_w + ((size_t)li * NEXPERTS + b * 4) * H, x, H, H, tokens);
    }
    for(int t=0;t<tokens;t++){int selected[8],npart=0;float route_weight[8];const float*xt=x+(size_t)t*H;
        if (verify_router) memcpy(c->router_logits, verify_logits[t], sizeof(verify_logits[t]));
        else {
#pragma omp parallel for schedule(static)
        for(int e=0;e<NEXPERTS;e++)c->router_logits[e]=glm53f_dot_bf16_sve(c->router_w+((size_t)li*NEXPERTS+e)*H,xt,H);
        }
        glm53f_router_topk(c->router_logits,c->router_bias+(size_t)li*NEXPERTS,NEXPERTS,8,2.5f,selected,route_weight);if(moe_trace_routes(c,selected,route_weight,1))return-1;
        for(int k=0;k<8;k++){expert_offset*p=&c->table[table_layer*NEXPERTS+selected[k]];if(p->gate_up==UINT64_MAX)continue;parts[t*MAXP+npart]=(glm53f_expert_part){c->blob+p->gate_up,p->gate_up_scale==UINT64_MAX?NULL:(const float*)(c->blob+p->gate_up_scale),c->blob+p->down,p->down_scale==UINT64_MAX?NULL:(const float*)(c->blob+p->down_scale),p->inter,p->gate_type,p->down_type};weights[t*MAXP+npart++]=route_weight[k];}
        counts[t]=npart;
    }
    if (c->nsh_native) {
        int grouped = 0;
        if (c->verify_scratch) {
            glm53f_iq_part iq[4 * MAXP];
            for (int t = 0; t < tokens; ++t)
                for (int k = 0; k < counts[t]; ++k) {
                    const glm53f_expert_part *p = parts + t * MAXP + k;
                    iq[t * MAXP + k] = (glm53f_iq_part){p->gate_up, p->down,
                        p->gate_type, p->down_type, p->inter};
                }
            grouped = glm53f_iq_expert_weighted_batch(c->batch_local, iq,
                weights, counts, MAXP, x, tokens, c->verify_scratch) == 0;
        }
        for (int t = 0; t < tokens && !grouped; ++t) {
            if (!counts[t]) memset(c->batch_local + (size_t)t * H, 0, H * sizeof(float));
            else glm53f_moe_local_12n(c->batch_local + (size_t)t * H,
                parts + t * MAXP, weights + t * MAXP, counts[t],
                x + (size_t)t * H, c->scratch);
        }
        if (nsh_batch(c, table_layer, x, tokens)) return -1;
#pragma omp parallel for schedule(static)
        for (int q = 0; q < tokens * H; ++q) c->batch_local[q] += c->batch_shared[q];
        return moe_sum(c,c->batch_local, out, tokens * H);
    }
    int has_iq=0;
    for(int t=0;t<tokens&&!has_iq;t++)
        for(int k=0;k<counts[t];k++)
            has_iq|=parts[t*MAXP+k].gate_type||parts[t*MAXP+k].down_type;
    if(has_iq){
        shared_offset*sp=&c->shared[table_layer];
        int batch_shared = getenv("GLM53F_Q4_BATCH_SHARED") &&
                           atoi(getenv("GLM53F_Q4_BATCH_SHARED"));
        for (int t = 0; t < tokens && batch_shared; t++)
            for (int k = 0; k < counts[t]; k++)
                if (!glm53f_iq_type_supported(parts[t*MAXP+k].gate_type) ||
                    !glm53f_iq_type_supported(parts[t*MAXP+k].down_type))
                    batch_shared = 0;
        if (batch_shared) {
            /* Routed experts keep their validated SDOT accumulation order.
             * The FP8 shared expert is common to all positions: amortize its
             * weight reads and dequantization across the verification batch. */
            for (int t = 0; t < tokens; t++) {
                glm53f_iq_part iq[MAXP];
                for (int k = 0; k < counts[t]; k++) {
                    const glm53f_expert_part *p = &parts[t*MAXP+k];
                    iq[k] = (glm53f_iq_part){p->gate_up, p->down,
                        p->gate_type, p->down_type, p->inter};
                }
                if (!counts[t]) memset(c->batch_local + (size_t)t*H, 0, H*sizeof(float));
                else if (glm53f_iq_expert_weighted(c->batch_local + (size_t)t*H,
                    iq, weights + t*MAXP, counts[t], x + (size_t)t*H,
                    c->scratch->up, c->scratch->activation)) return -1;
            }
            glm53f_expert_part shared = {c->shared_blob + sp->gate_up,
                (const float *)(c->shared_blob + sp->gate_up_scale),
                c->shared_blob + sp->down,
                (const float *)(c->shared_blob + sp->down_scale), sp->inter, 0, 0};
            glm53f_expert_tokens_bits(&shared, tokens, x, c->batch_up,
                                      c->batch_activation, c->batch_shared);
#pragma omp parallel for schedule(static)
            for (int q = 0; q < tokens*H; q++) c->batch_local[q] += c->batch_shared[q];
            return moe_sum(c,c->batch_local, out, tokens*H);
        }
        for(int t=0;t<tokens;t++){
            int n=counts[t];
            parts[t*MAXP+n]=(glm53f_expert_part){c->shared_blob+sp->gate_up,
                (const float*)(c->shared_blob+sp->gate_up_scale),
                c->shared_blob+sp->down,(const float*)(c->shared_blob+sp->down_scale),
                sp->inter,0,0};
            weights[t*MAXP+n]=1.0f;
            glm53f_moe_local_12n(c->batch_local+(size_t)t*H,
                parts+t*MAXP,weights+t*MAXP,n+1,x+(size_t)t*H,c->scratch);
        }
        return moe_sum(c,c->batch_local,out,tokens*H);
    }
    if(!c->task_up){if(posix_memalign((void**)&c->task_up,256,4*9*1024*4)||posix_memalign((void**)&c->task_activation,256,4*9*512*4)||posix_memalign((void**)&c->task_output,256,4*9*H*4))return-1;}
    float*up=c->task_up; float*act=c->task_activation; float*y=c->task_output;
    /* Keep a full expert team per token: cross-token task partitioning leaves
     * too few lanes per matvec on A64FX and regresses decode throughput. */
    for(int t=0;t<tokens;t++)
        glm53f_expert_batch_bits(parts+t*MAXP,counts[t],x+(size_t)t*H,
                                 up+(size_t)t*MAXP*1024,
                                 act+(size_t)t*MAXP*512,
                                 y+(size_t)t*MAXP*H);
    shared_offset*sp=&c->shared[table_layer];glm53f_expert_part shared={c->shared_blob+sp->gate_up,(const float*)(c->shared_blob+sp->gate_up_scale),c->shared_blob+sp->down,(const float*)(c->shared_blob+sp->down_scale),sp->inter,0,0};
    glm53f_expert_tokens_bits(&shared,tokens,x,c->batch_up,c->batch_activation,c->batch_shared);
#pragma omp parallel for schedule(static)
    for(int q=0;q<tokens*H;q++){int t=q/H,i=q-t*H;float v=0.0f;for(int k=0;k<counts[t];k++)v+=weights[t*MAXP+k]*y[((size_t)t*MAXP+k)*H+i];c->batch_local[q]=v+c->batch_shared[q];}
    int rc=moe_sum(c,c->batch_local,out,tokens*H);return rc;
}
void glm53f_moe_stage_free_12n(glm53f_moe_stage_context_12n*c){if(!c)return;for(int l=0;l<NLAYERS;l++){if(c->route_export[l])fclose(c->route_export[l]);free(c->s8_gu[l]);free(c->s8_dn[l]);}free(c->s8_xq);free(c->s8_xs);free(c->s8_bt);free(c->s8_xp);free(c->s8_xsp);free(c->s8_ygu);free(c->s8_a8);free(c->s8_as);free(c->s8_bd);free(c->s8_xp2);free(c->s8_asp);free(c->s8_y);for(int l=0;l<NLAYERS;l++){free(c->sg_gu[l]);free(c->sg_dn[l]);}free(c->sg_up);free(c->sg_act);free(c->sg_out);if(c->gn_threads){gn_tbuf*tb=(gn_tbuf*)c->gn_threads;for(int i=0;i<c->gn_nthreads;i++){free(tb[i].xp);free(tb[i].xp2);free(tb[i].a8);free(tb[i].xsp);free(tb[i].Bg);free(tb[i].ygu);free(tb[i].as);free(tb[i].asp);free(tb[i].mina);free(tb[i].yd);free(tb[i].cbuf);}free(c->gn_threads);}free(c->gn_xq);free(c->gn_xs);free(c->gn_bt);free(c->gn_verify);for(int l=0;l<NLAYERS;l++){free(c->nsh_g[l]);free(c->nsh_u[l]);free(c->nsh_d[l]);}free(c->nsh_gv);free(c->nsh_uv);free(c->nsh_act);free(c->nsh_out);free(c->nsh_act_x);free(c->nsh_act_h);free(c->i8_prefill_storage);for(int l=0;l<NLAYERS;l++)free(c->int8_scales[l]);free(c->task_output);free(c->task_activation);free(c->task_up);glm53f_iq_batch_scratch_free(c->verify_scratch);free(c->batch_routes);free(c->batch_group_x);free(c->batch_router);free(c->batch_local);free(c->batch_shared);free(c->batch_activation);free(c->batch_up);free(c->scratch);free(c->router_logits);free(c->router_bias);free(c->router_i8_scale);free(c->router_i8);free(c->router_wt);free(c->router_w);free(c->shared_blob);free(c->blob);free(c->table);free(c);}

#ifndef GLM53F_EXPERT_NO_MAIN
int main(int argc, char **argv) {
    int rank, ranks, tokens = argc > 2 ? atoi(argv[2]) : 20;
    int first_layer = getenv("GLM53F_FIRST_LAYER") ?
        atoi(getenv("GLM53F_FIRST_LAYER")) : FIRST_LAYER;
    int run_layers = getenv("GLM53F_LAYER_COUNT") ?
        atoi(getenv("GLM53F_LAYER_COUNT")) : 42;
    int attention_combine = getenv("GLM53F_ATTENTION_COMBINE") ?
        atoi(getenv("GLM53F_ATTENTION_COMBINE")) : 0;
    const char *stage = argc > 1 ? argv[1] : getenv("GLM53F_STAGE_DIR");
    const char *shared_stage = getenv("GLM53F_SHARED_STAGE_DIR");
    const char *model_dir = getenv("GLM53F_MODEL_DIR");
    char blob_path[512], manifest_path[512];
    expert_offset *table;
    unsigned char *blob, *shared_blob = NULL;
    size_t blob_bytes = 0, shared_bytes = 0;
    shared_offset shared[NLAYERS];
    float *x, *sum;
    glm53f_moe_scratch_12n *moe_scratch;
    uint16_t *router_w = NULL;
    float *router_bias = NULL, *router_logits = NULL, route_weight[8];
    int router_check = 1;
    double compute = 0, combine = 0, attention = 0, wall0, wire_call;
    long local_tasks = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (!stage || ranks != 12 || tokens < 1 || first_layer < FIRST_LAYER ||
        run_layers < 1 || first_layer + run_layers > LAST_LAYER) {
        if (!rank) fprintf(stderr, "usage: %s STAGE_DIR [tokens=20] (requires 12 ranks)\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    snprintf(blob_path, sizeof(blob_path), "%s/rank%02d.blob", stage, rank);
    snprintf(manifest_path, sizeof(manifest_path), "%s/rank%02d.manifest", stage, rank);
    table = malloc((size_t)NLAYERS * NEXPERTS * sizeof(*table));
    int manifest_entries = table ? load_manifest(manifest_path, table) : -1;
    if (manifest_entries < run_layers * 96 * 4 || manifest_entries % (run_layers * 4)) {
        fprintf(stderr, "rank=%d manifest contract failed: %s\n", rank, manifest_path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    blob = load_anon(blob_path, &blob_bytes, rank);
    if (!blob) { fprintf(stderr, "rank=%d load failed: %s: %s\n", rank, blob_path, strerror(errno)); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (mem_available() < (2L << 30)) { fprintf(stderr, "rank=%d insufficient HBM headroom\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (shared_stage) {
        snprintf(blob_path, sizeof(blob_path), "%s/rank%02d.blob", shared_stage, rank);
        snprintf(manifest_path, sizeof(manifest_path), "%s/rank%02d.manifest", shared_stage, rank);
        if (load_shared_manifest(manifest_path, shared) != run_layers * 4 ||
            !(shared_blob = load_anon(blob_path, &shared_bytes, rank))) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    posix_memalign((void **)&x, 256, 4096 * sizeof(float));
    posix_memalign((void **)&moe_scratch, 256, sizeof(*moe_scratch));
    posix_memalign((void **)&sum, 256, 4096 * sizeof(float));
    if (!x || !moe_scratch || !sum) MPI_Abort(MPI_COMM_WORLD, 1);
    if (model_dir) {
        glm53f_st_context *st = glm53f_st_open(model_dir);
        char name[256];
        size_t wn = (size_t)run_layers * NEXPERTS * 4096 * sizeof(uint16_t);
        size_t bn = (size_t)run_layers * NEXPERTS * sizeof(float);
        if (!st || posix_memalign((void **)&router_w, 256, wn) ||
            posix_memalign((void **)&router_bias, 256, bn) ||
            posix_memalign((void **)&router_logits, 256, NEXPERTS * sizeof(float)))
            MPI_Abort(MPI_COMM_WORLD, 1);
        for (int li = 0; li < run_layers; ++li) {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.gate.weight", first_layer + li);
            if (glm53f_st_read(st, name, 0, router_w + (size_t)li * NEXPERTS * 4096,
                               (size_t)NEXPERTS * 4096 * sizeof(uint16_t))) MPI_Abort(MPI_COMM_WORLD, 1);
            snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.gate.e_score_correction_bias", first_layer + li);
            if (glm53f_st_read(st, name, 0, router_bias + (size_t)li * NEXPERTS,
                               NEXPERTS * sizeof(float))) MPI_Abort(MPI_COMM_WORLD, 1);
        }
        glm53f_st_close(st);
    }
    for (int i = 0; i < 4096; ++i) x[i] = (float)((i % 29) - 14) * .001f;
    if (router_w) {
        float ref_logits[NEXPERTS], ref_weight[8], sve_weight[8];
        int ref_id[8], sve_id[8];
#pragma omp parallel for schedule(static)
        for (int e = 0; e < NEXPERTS; ++e) {
            const uint16_t *w = router_w + (size_t)e * 4096;
            router_logits[e] = glm53f_dot_bf16_sve(w, x, 4096);
            ref_logits[e] = glm53f_dot_bf16(w, x, 4096);
        }
        glm53f_router_topk(router_logits, router_bias, NEXPERTS, 8, 2.5f,
                           sve_id, sve_weight);
        glm53f_router_topk(ref_logits, router_bias, NEXPERTS, 8, 2.5f,
                           ref_id, ref_weight);
        for (int k = 0; k < 8; ++k)
            router_check &= sve_id[k] == ref_id[k] &&
                            fabsf(sve_weight[k] - ref_weight[k]) < 2e-6f;
        int all_router_check;
        MPI_Allreduce(&router_check, &all_router_check, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        router_check = all_router_check;
        if (!router_check) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    memset(sum, 0, 4096 * sizeof(float));
    double wire0 = now_sec();
    for (int i = 0; i < 200; ++i)
        MPI_Allreduce(MPI_IN_PLACE, sum, 4096, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    double wire_local = (now_sec() - wire0) / 200.0;
    MPI_Allreduce(&wire_local, &wire_call, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    /* Warm OpenMP and touch one real route before measuring. */
    int total_tokens = tokens + 1;
    wall0 = now_sec();
    for (int tok = 0; tok < total_tokens; ++tok) {
        if (tok == 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            wall0 = now_sec();
            compute = combine = attention = 0;
            local_tasks = 0;
        }
        for (int li = 0; li < run_layers; ++li) {
            int layer_index = first_layer - FIRST_LAYER + li;
            int selected[8], n = 0;
            glm53f_expert_part part[9];
            float part_weight[9];
            if (attention_combine) {
                memset(sum, 0, 4096 * sizeof(float));
                double t0 = now_sec();
                MPI_Allreduce(MPI_IN_PLACE, sum, 4096, MPI_FLOAT, MPI_SUM,
                              MPI_COMM_WORLD);
                attention += now_sec() - t0;
            }
            double t0 = now_sec();
            if (router_w) {
#pragma omp parallel for schedule(static)
                for (int e = 0; e < NEXPERTS; ++e)
                    router_logits[e] = glm53f_dot_bf16_sve(
                        router_w + ((size_t)li * NEXPERTS + e) * 4096, x, 4096);
                glm53f_router_topk(router_logits,
                    router_bias + (size_t)li * NEXPERTS, NEXPERTS, 8, 2.5f,
                    selected, route_weight);
            } else {
                route8(tok, first_layer + li, selected);
                for (int k = 0; k < 8; ++k) route_weight[k] = 1.0f;
            }
            for (int k = 0; k < 8; ++k) {
                expert_offset *p = &table[layer_index * NEXPERTS + selected[k]];
                if (p->gate_up == UINT64_MAX) continue;
                part[n].gate_up = blob + p->gate_up;
                part[n].gate_up_scale = p->gate_up_scale==UINT64_MAX?NULL:(const float *)(blob + p->gate_up_scale);
                part[n].down = blob + p->down;
                part[n].down_scale = p->down_scale==UINT64_MAX?NULL:(const float *)(blob + p->down_scale);
                part[n].inter = p->inter;
                part[n].gate_type = p->gate_type;
                part[n].down_type = p->down_type;
                part_weight[n] = route_weight[k];
                n++;
            }
            if (shared_blob) {
                shared_offset *p = &shared[layer_index];
                part[n].gate_up = shared_blob + p->gate_up;
                part[n].gate_up_scale = (const float *)(shared_blob + p->gate_up_scale);
                part[n].down = shared_blob + p->down;
                part[n].down_scale = (const float *)(shared_blob + p->down_scale);
                part[n].inter = p->inter;
                part[n].gate_type = 0;
                part[n].down_type = 0;
                part_weight[n] = 1.0f;
                n++;
            }
            glm53f_moe_local_12n(sum, part, part_weight, n, x, moe_scratch);
            compute += now_sec() - t0;
            t0 = now_sec();
            MPI_Allreduce(MPI_IN_PLACE, sum, 4096, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            combine += now_sec() - t0;
            if (tok > 0) local_tasks += n;
            x[(li * 97 + tok) & 4095] += sum[(li * 131 + tok) & 4095] * 1e-5f;
        }
    }
    double wall = now_sec() - wall0, max_wall, max_compute, max_combine, max_attention;
    long max_tasks;
    MPI_Reduce(&wall, &max_wall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&combine, &max_combine, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&attention, &max_attention, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_tasks, &max_tasks, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        char line[2048];
        snprintf(line, sizeof(line), "GLM53F_EXPERT_DECODE_12N tokens=%d layers=%d attention_combine=%d real_router=%d router_check=%s expert_parts_rank=%d weight_GiB_rank=%.3f max_tasks=%ld wall_ms_tok=%.3f compute_ms_tok=%.3f combine_ms_tok=%.3f attention_ms_tok=%.3f wire_us_call=%.3f wire_ms_tok=%.3f arrival_ms_tok=%.3f tok_s=%.3f checksum=%.9g\n",
            tokens, run_layers, attention_combine, router_w != NULL,
            router_check ? "PASS" : "FAIL",
            manifest_entries / (run_layers * 4), (blob_bytes + shared_bytes) / 1073741824.0, max_tasks,
            max_wall * 1e3 / tokens, max_compute * 1e3 / tokens,
            max_combine * 1e3 / tokens, max_attention * 1e3 / tokens,
            wire_call * 1e6, wire_call * run_layers * (attention_combine + 1) * 1e3,
            (max_combine + max_attention) * 1e3 / tokens -
                wire_call * run_layers * (attention_combine + 1) * 1e3,
            tokens / max_wall, sum[0]);
        fputs(line, stdout); fflush(stdout);
        const char *status_path = getenv("GLM53F_BENCH_STATUS");
        if (status_path && *status_path) {
            FILE *status = fopen(status_path, "w");
            if (status) { fputs(line, status); fclose(status); }
        }
    }
    free(router_logits); free(router_bias); free(router_w);
    free(sum); free(moe_scratch); free(x); free(shared_blob); free(blob); free(table);
    MPI_Finalize();
    return 0;
}
#endif
