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
#include "glm53f_moe_12n.h"
#include "glm53f_moe_stage_12n.h"
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { FIRST_LAYER = 3, LAST_LAYER = 46, NLAYERS = 43, NEXPERTS = 288 };

typedef struct {
    uint64_t gate_up, gate_up_scale, down, down_scale;
    int inter;
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

static unsigned char *load_anon(const char *path, size_t *bytes, int rank) {
    const size_t chunk = 64u << 20;
    struct stat st;
    int fd = open(path, O_RDONLY);
    unsigned char *data = NULL;
    if (fd < 0 || fstat(fd, &st) || posix_memalign((void **)&data, 256, (size_t)st.st_size)) return NULL;
    double t0 = now_sec();
    for (size_t off = 0; off < (size_t)st.st_size; off += chunk) {
        size_t n = (size_t)st.st_size - off;
        if (n > chunk) n = chunk;
        ssize_t got = pread(fd, data + off, n, (off_t)off);
        if (got != (ssize_t)n) { free(data); close(fd); return NULL; }
        posix_fadvise(fd, (off_t)off, (off_t)n, POSIX_FADV_DONTNEED);
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

struct glm53f_moe_stage_context_12n {
    int rank, first_layer, layer_count, active_layer;
    expert_offset *table;
    shared_offset shared[NLAYERS];
    unsigned char *blob, *shared_blob;
    uint16_t *router_w;
    float *router_bias, *router_logits;
    glm53f_moe_scratch_12n *scratch;
};

glm53f_moe_stage_context_12n *glm53f_moe_stage_create_12n(
        const char*routed_stage,const char*shared_stage,const char*model_dir,
        int first_layer,int layer_count){int rank,nr;char path[512];size_t bytes;glm53f_moe_stage_context_12n*c;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(nr!=12||first_layer<FIRST_LAYER||layer_count<1||first_layer+layer_count>45)return NULL;c=calloc(1,sizeof(*c));if(!c)return NULL;c->rank=rank;c->first_layer=first_layer;c->layer_count=layer_count;c->active_layer=first_layer;c->table=malloc((size_t)NLAYERS*NEXPERTS*sizeof(*c->table));if(!c->table)goto fail;snprintf(path,sizeof(path),"%s/rank%02d.manifest",routed_stage,rank);if(load_manifest(path,c->table)<layer_count*NEXPERTS*4)goto fail;snprintf(path,sizeof(path),"%s/rank%02d.blob",routed_stage,rank);c->blob=load_anon(path,&bytes,rank);if(!c->blob)goto fail;if(shared_stage){snprintf(path,sizeof(path),"%s/rank%02d.manifest",shared_stage,rank);if(load_shared_manifest(path,c->shared)!=layer_count*4)goto fail;snprintf(path,sizeof(path),"%s/rank%02d.blob",shared_stage,rank);c->shared_blob=load_anon(path,&bytes,rank);if(!c->shared_blob)goto fail;}glm53f_st_context*st=glm53f_st_open(model_dir);if(!st)goto fail;size_t wn=(size_t)layer_count*NEXPERTS*4096*sizeof(uint16_t),bn=(size_t)layer_count*NEXPERTS*sizeof(float);if(posix_memalign((void**)&c->router_w,256,wn)||posix_memalign((void**)&c->router_bias,256,bn)||posix_memalign((void**)&c->router_logits,256,NEXPERTS*sizeof(float))||posix_memalign((void**)&c->scratch,256,sizeof(*c->scratch)))MPI_Abort(MPI_COMM_WORLD,2);for(int li=0;li<layer_count;li++){char n[256];snprintf(n,sizeof(n),"model.language_model.layers.%d.mlp.gate.weight",first_layer+li);if(glm53f_st_read(st,n,0,c->router_w+(size_t)li*NEXPERTS*4096,(size_t)NEXPERTS*4096*sizeof(uint16_t)))MPI_Abort(MPI_COMM_WORLD,2);snprintf(n,sizeof(n),"model.language_model.layers.%d.mlp.gate.e_score_correction_bias",first_layer+li);if(glm53f_st_read(st,n,0,c->router_bias+(size_t)li*NEXPERTS,NEXPERTS*sizeof(float)))MPI_Abort(MPI_COMM_WORLD,2);}glm53f_st_close(st);return c;fail:glm53f_moe_stage_free_12n(c);return NULL;}
void glm53f_moe_stage_set_layer_12n(glm53f_moe_stage_context_12n*c,int layer){if(c)c->active_layer=layer;}
int glm53f_moe_stage_sublayer_12n(void*context,float*out,const float*x){glm53f_moe_stage_context_12n*c=context;int li=c->active_layer-c->first_layer,selected[8],npart=0;float route_weight[8],part_weight[9];glm53f_expert_part part[9];if(li<0||li>=c->layer_count)return-1;
#pragma omp parallel for schedule(static)
    for(int e=0;e<NEXPERTS;e++)c->router_logits[e]=glm53f_dot_bf16_sve(c->router_w+((size_t)li*NEXPERTS+e)*4096,x,4096);glm53f_router_topk(c->router_logits,c->router_bias+(size_t)li*NEXPERTS,NEXPERTS,8,2.5f,selected,route_weight);int table_layer=c->active_layer-FIRST_LAYER;for(int k=0;k<8;k++){expert_offset*p=&c->table[table_layer*NEXPERTS+selected[k]];if(p->gate_up==UINT64_MAX)continue;part[npart]=(glm53f_expert_part){c->blob+p->gate_up,(const float*)(c->blob+p->gate_up_scale),c->blob+p->down,(const float*)(c->blob+p->down_scale),p->inter};part_weight[npart++]=route_weight[k];}if(c->shared_blob){shared_offset*p=&c->shared[table_layer];part[npart]=(glm53f_expert_part){c->shared_blob+p->gate_up,(const float*)(c->shared_blob+p->gate_up_scale),c->shared_blob+p->down,(const float*)(c->shared_blob+p->down_scale),p->inter};part_weight[npart++]=1.0f;}return glm53f_moe_forward_12n(out,part,part_weight,npart,x,c->scratch,MPI_COMM_WORLD);}
void glm53f_moe_stage_free_12n(glm53f_moe_stage_context_12n*c){if(!c)return;free(c->scratch);free(c->router_logits);free(c->router_bias);free(c->router_w);free(c->shared_blob);free(c->blob);free(c->table);free(c);}

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
                part[n].gate_up_scale = (const float *)(blob + p->gate_up_scale);
                part[n].down = blob + p->down;
                part[n].down_scale = (const float *)(blob + p->down_scale);
                part[n].inter = p->inter;
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
