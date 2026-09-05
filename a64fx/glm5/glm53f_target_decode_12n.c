/* Persistent real-weight 45-layer GLM-5.3F greedy target decode. */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../../common/glm53f_safetensors.h"
#include "glm53f_dense_ffn_12n.h"
#include "glm53f_embedding_12n.h"
#include "glm53f_kda_12n.h"
#include "glm53f_moe_stage_12n.h"
#include "glm53f_sparse_12n.h"
#include "glm53f_target_head_12n.h"
#include "glm53f_target_layer_12n.h"
#include "glm53f_target_model_12n.h"
#include "glm53f_collective_12n.h"

enum { LAYERS = 45, HIDDEN = 4096, STREAMS = 4, FLAT = 16384, MIX = 24 };

static void *a256(size_t bytes) {
    void *p = NULL;
    return posix_memalign(&p, 256, bytes) ? NULL : p;
}

static void target_memtrace(int rank, const char *phase) {
    const char *dir = getenv("GLM53F_MEMTRACE_DIR");
    char path[512], line[256];
    FILE *in, *out;
    long rss = -1, avail = -1;
    if (!dir || !*dir) return;
    in = fopen("/proc/self/status", "r");
    while (in && fgets(line, sizeof(line), in))
        if (sscanf(line, "VmRSS: %ld kB", &rss) == 1) break;
    if (in) fclose(in);
    in = fopen("/proc/meminfo", "r");
    while (in && fgets(line, sizeof(line), in))
        if (sscanf(line, "MemAvailable: %ld kB", &avail) == 1) break;
    if (in) fclose(in);
    snprintf(path, sizeof(path), "%s/rank%02d.mem", dir, rank);
    out = fopen(path, "a");
    if (out) { fprintf(out, "%s rss_kb=%ld mem_available_kb=%ld\n", phase, rss, avail); fclose(out); }
}

static void *load_exact(glm53f_st_context *st, const char *name, size_t bytes) {
    const st_tensor_info *tensor = glm53f_st_find(st, name, NULL);
    void *p = a256(bytes);
    if (!tensor || tensor->nbytes != bytes || !p ||
        glm53f_st_read(st, name, 0, p, bytes)) {
        fprintf(stderr, "target stack load failed: %s\n", name);
        free(p);
        return NULL;
    }
    return p;
}

static int load_layer(glm53f_st_context *st, int layer,
                      glm53f_target_layer_weights_12n *w) {
    char name[256];
    memset(w, 0, sizeof(*w));
#define SITE(FIELD, WHICH) do { \
    snprintf(name,sizeof(name),"model.language_model.layers.%d.hc_" WHICH "_fn",layer); \
    w->FIELD.fn=load_exact(st,name,(size_t)MIX*FLAT*sizeof(uint16_t)); \
    snprintf(name,sizeof(name),"model.language_model.layers.%d.hc_" WHICH "_base",layer); \
    w->FIELD.base=load_exact(st,name,MIX*sizeof(float)); \
    snprintf(name,sizeof(name),"model.language_model.layers.%d.hc_" WHICH "_scale",layer); \
    w->FIELD.scale=load_exact(st,name,3*sizeof(float)); \
} while(0)
    SITE(attention_mhc,"attn"); SITE(ffn_mhc,"ffn");
#undef SITE
    snprintf(name,sizeof(name),"model.language_model.layers.%d.input_layernorm.weight",layer);
    w->input_norm=load_exact(st,name,HIDDEN*sizeof(uint16_t));
    snprintf(name,sizeof(name),"model.language_model.layers.%d.post_attention_layernorm.weight",layer);
    w->post_attention_norm=load_exact(st,name,HIDDEN*sizeof(uint16_t));
    return w->attention_mhc.fn&&w->attention_mhc.base&&w->attention_mhc.scale&&
           w->ffn_mhc.fn&&w->ffn_mhc.base&&w->ffn_mhc.scale&&
           w->input_norm&&w->post_attention_norm?0:-1;
}

struct glm53f_target_model_12n {
    glm53f_target_layer_weights_12n layer_weight[LAYERS];
    glm53f_kda_context_12n *kda[LAYERS];
    glm53f_sparse_context_12n *sparse[LAYERS];
    glm53f_dense_ffn_context_12n *dense[3];
    glm53f_moe_stage_context_12n *moe;
    glm53f_embedding_context_12n *embedding;
    glm53f_target_head_context_12n *head;
    glm53f_target_layer_scratch_12n *scratch;
    float *streams;
    glm53f_target_layer_scratch_12n *batch_scratch;
    float *batch_streams, *batch_normalized, *batch_output;
    unsigned char *batch_state;
    size_t batch_state_stride;
    int profile;
    long scalar_steps, batch_calls, batch_positions;
    double scalar_phase[5], batch_phase[5];
};
struct glm53f_target_snapshot_12n {
    unsigned char *kda_state;
    size_t kda_bytes;
    int sparse_length[LAYERS];
};

glm53f_target_model_12n *glm53f_target_model_create_12n(
        const char *model_dir, const char *routed, const char *shared,
        int capacity) {
    int rank,ranks;
    glm53f_st_context *st;
    glm53f_target_model_12n *m = calloc(1, sizeof(*m));
    if (!m || capacity < 1) goto fail;
    m->profile = getenv("GLM53F_PROFILE") != NULL;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ranks != 12) goto fail;
    st = glm53f_st_open(model_dir);
    if (!st) goto fail;
    for (int l = 0; l < LAYERS; ++l)
        if (load_layer(st, l, &m->layer_weight[l])) {
            glm53f_st_close(st);
            goto fail;
        }
    glm53f_st_close(st);
    target_memtrace(rank, "layer_stack");
    m->embedding = glm53f_embedding_create_12n(model_dir);
    m->head = glm53f_target_head_create_12n(model_dir);
    target_memtrace(rank, "embedding_head");
    for (int l = 0; l < LAYERS; ++l) {
        if (l % 4 == 3) m->sparse[l] = glm53f_sparse_create_12n(model_dir, l, capacity);
        else m->kda[l] = glm53f_kda_create_12n(model_dir, l);
        if (!m->sparse[l] && !m->kda[l]) goto fail;
    }
    target_memtrace(rank, "attention");
    for (int l = 0; l < 3; ++l) {
        m->dense[l] = glm53f_dense_ffn_create_12n(model_dir, l);
        if (!m->dense[l]) goto fail;
    }
    target_memtrace(rank, "dense");
    target_memtrace(rank, "moe_before");
    m->moe = glm53f_moe_stage_create_12n(routed, shared, model_dir, 3, 42);
    target_memtrace(rank, "moe_after");
    m->scratch = a256(sizeof(*m->scratch));
    m->streams = a256((size_t)FLAT * sizeof(float));
    m->batch_scratch = a256((size_t)5 * sizeof(*m->batch_scratch));
    m->batch_streams = a256((size_t)5 * FLAT * sizeof(float));
    m->batch_normalized = a256((size_t)5 * HIDDEN * sizeof(float));
    m->batch_output = a256((size_t)5 * HIDDEN * sizeof(float));
    for (int l = 0; l < LAYERS; l++) {
        size_t n = glm53f_kda_state_bytes_12n(m->kda[l]);
        if (n > m->batch_state_stride) m->batch_state_stride = n;
    }
    m->batch_state = a256((size_t)5 * m->batch_state_stride);
    if (!m->embedding || !m->head || !m->moe || !m->scratch || !m->streams ||
        !m->batch_scratch || !m->batch_streams || !m->batch_normalized ||
        !m->batch_output || !m->batch_state) goto fail;
    if(!rank){size_t bytes=0;int cp=0;for(int l=0;l<LAYERS;l++)if(m->sparse[l]){bytes+=glm53f_sparse_cache_bytes_12n(m->sparse[l]);cp+=glm53f_sparse_is_context_parallel_12n(m->sparse[l]);}printf("GLM53F_TARGET_CACHE capacity=%d sparse_layers=11 cp_layers=%d bytes_rank=%zu GiB_rank=%.3f\n",capacity,cp,bytes,bytes/1073741824.0);}
    return m;
fail:
    glm53f_target_model_free_12n(m);
    return NULL;
}

int glm53f_target_model_step_12n(glm53f_target_model_12n *m, int token,
        int *next_token, float *next_logit, float *target_hidden) {
    double begin = m && m->profile ? MPI_Wtime() : 0.0;
    if (!m || !next_token || !next_logit ||
        glm53f_embedding_streams_12n(m->embedding, token, m->streams)) return -1;
    if (m->profile) m->scalar_phase[0] += MPI_Wtime() - begin;
    for (int l = 0; l < LAYERS; ++l) {
        const glm53f_target_layer_weights_12n *w = &m->layer_weight[l];
        begin = m->profile ? MPI_Wtime() : 0.0;
        glm53f_mhc_pre_sve(&m->scratch->mhc, m->streams, &w->attention_mhc,
                           w->input_norm);
        if (m->profile) m->scalar_phase[1] += MPI_Wtime() - begin;
        begin = m->profile ? MPI_Wtime() : 0.0;
        int rc = m->kda[l] ?
            glm53f_kda_sublayer_12n(m->kda[l], m->scratch->sublayer_output,
                                    m->scratch->mhc.normalized) :
            glm53f_sparse_sublayer_12n(m->sparse[l], m->scratch->sublayer_output,
                                       m->scratch->mhc.normalized);
        if (rc) return -1;
        if (m->profile) m->scalar_phase[2] += MPI_Wtime() - begin;
        begin = m->profile ? MPI_Wtime() : 0.0;
        glm53f_mhc_post_sve(m->streams, m->scratch->sublayer_output,
                            &m->scratch->mhc);
        glm53f_mhc_pre_sve(&m->scratch->mhc, m->streams, &w->ffn_mhc,
                           w->post_attention_norm);
        if (m->profile) m->scalar_phase[1] += MPI_Wtime() - begin;
        begin = m->profile ? MPI_Wtime() : 0.0;
        if (l < 3) {
            rc = glm53f_dense_ffn_sublayer_12n(
                m->dense[l], m->scratch->sublayer_output,
                m->scratch->mhc.normalized);
        } else {
            glm53f_moe_stage_set_layer_12n(m->moe, l);
            rc = glm53f_moe_stage_sublayer_12n(
                m->moe, m->scratch->sublayer_output,
                m->scratch->mhc.normalized);
        }
        if (rc) return -1;
        if (m->profile) m->scalar_phase[3] += MPI_Wtime() - begin;
        begin = m->profile ? MPI_Wtime() : 0.0;
        glm53f_mhc_post_sve(m->streams, m->scratch->sublayer_output,
                            &m->scratch->mhc);
        if (m->profile) m->scalar_phase[1] += MPI_Wtime() - begin;
    }
    begin = m->profile ? MPI_Wtime() : 0.0;
    if (target_hidden) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < HIDDEN; ++i) {
            float z = 0.0f;
            for (int s = 0; s < STREAMS; ++s) z += m->streams[(size_t)s * HIDDEN + i];
            target_hidden[i] = z / STREAMS;
        }
    }
    int rc = glm53f_target_head_argmax_12n(
        m->head, m->streams, next_token, next_logit);
    if (m->profile) {
        m->scalar_phase[4] += MPI_Wtime() - begin;
        m->scalar_steps++;
    }
    return rc;
}

int glm53f_target_model_step_batch_12n(glm53f_target_model_12n *m,
                                       const int *input, int tokens, int *next,
                                       float *logit, float *hidden,
                                       glm53f_target_snapshot_12n **after) {
    double begin = m && m->profile ? MPI_Wtime() : 0.0;
    if (!m || !input || (!next != !logit) || tokens < 1 || tokens > 5)
        return -1;
    for (int t = 0; t < tokens; t++)
        if (glm53f_embedding_streams_12n(m->embedding, input[t],
                                         m->batch_streams + (size_t)t * FLAT))
            return -1;
    if (m->profile) m->batch_phase[0] += MPI_Wtime() - begin;
    size_t state_off = 0;
    for (int l = 0; l < LAYERS; l++) {
        const glm53f_target_layer_weights_12n *w = &m->layer_weight[l];
        begin = m->profile ? MPI_Wtime() : 0.0;
        glm53f_mhc_pre_batch_sve(&m->batch_scratch[0].mhc, m->batch_streams,
                                 &w->attention_mhc, w->input_norm, tokens,
                                 sizeof(*m->batch_scratch),
                                 m->batch_normalized);
        if (m->profile) m->batch_phase[1] += MPI_Wtime() - begin;
        begin = m->profile ? MPI_Wtime() : 0.0;
        if (m->kda[l]) {
            size_t bytes = glm53f_kda_state_bytes_12n(m->kda[l]);
            if (glm53f_kda_sublayer_batch_capture_12n(
                    m->kda[l], m->batch_output, m->batch_normalized, tokens,
                    after ? m->batch_state : NULL, m->batch_state_stride))
                return -1;
            if (after)
                for (int t = 0; t < tokens; t++) {
                    if (!after[t] || after[t]->kda_bytes < state_off + bytes)
                        return -1;
                    memcpy(after[t]->kda_state + state_off,
                           m->batch_state + (size_t)t * m->batch_state_stride,
                           bytes);
                }
            state_off += bytes;
        } else {
            int base = glm53f_sparse_length_12n(m->sparse[l]);
            if (glm53f_sparse_sublayer_batch_12n(m->sparse[l], m->batch_output,
                                                 m->batch_normalized, tokens))
                return -1;
            if (after)
                for (int t = 0; t < tokens; t++) {
                    if (!after[t])
                        return -1;
                    after[t]->sparse_length[l] = base + t + 1;
                }
        }
        if (m->profile) m->batch_phase[2] += MPI_Wtime() - begin;
        begin = m->profile ? MPI_Wtime() : 0.0;
        glm53f_mhc_post_batch_sve(m->batch_streams, m->batch_output,
                                  &m->batch_scratch[0].mhc, tokens,
                                  sizeof(m->batch_scratch[0]));
        glm53f_mhc_pre_batch_sve(&m->batch_scratch[0].mhc, m->batch_streams,
                                 &w->ffn_mhc, w->post_attention_norm, tokens,
                                 sizeof(*m->batch_scratch),
                                 m->batch_normalized);
        if (m->profile) m->batch_phase[1] += MPI_Wtime() - begin;
        begin = m->profile ? MPI_Wtime() : 0.0;
        if (l < 3) {
            int n = tokens < 5 ? tokens : 4;
            if (glm53f_dense_ffn_sublayer_batch_12n(
                    m->dense[l], m->batch_output, m->batch_normalized, n))
                return -1;
            if (tokens == 5 &&
                glm53f_dense_ffn_sublayer_12n(
                    m->dense[l], m->batch_output + (size_t)4 * HIDDEN,
                    m->batch_normalized + (size_t)4 * HIDDEN))
                return -1;
        } else {
            glm53f_moe_stage_set_layer_12n(m->moe, l);
            int n = tokens < 5 ? tokens : 4;
            if (glm53f_moe_stage_sublayer_batch_12n(m->moe, m->batch_output,
                                                    m->batch_normalized, n))
                return -1;
            if (tokens == 5 && glm53f_moe_stage_sublayer_12n(
                                   m->moe, m->batch_output + (size_t)4 * HIDDEN,
                                   m->batch_normalized + (size_t)4 * HIDDEN))
                return -1;
        }
        if (m->profile) m->batch_phase[3] += MPI_Wtime() - begin;
        begin = m->profile ? MPI_Wtime() : 0.0;
        glm53f_mhc_post_batch_sve(m->batch_streams, m->batch_output,
                                  &m->batch_scratch[0].mhc, tokens,
                                  sizeof(m->batch_scratch[0]));
        if (m->profile) m->batch_phase[1] += MPI_Wtime() - begin;
    }
    if (after && state_off != after[0]->kda_bytes)
        return -1;
    begin = m->profile ? MPI_Wtime() : 0.0;
    /* Reduce all returned hidden states in one OpenMP region.  The previous
     * per-position parallel regions paid team wake-up/barrier overhead for
     * every verified token (and made the cost grow discontinuously with the
     * draft length); the flattened index preserves bit-exact scalar order. */
    if (hidden) {
#pragma omp parallel for schedule(static)
        for (int k = 0; k < tokens * HIDDEN; k++) {
            int t = k / HIDDEN, i = k % HIDDEN;
            const float *stream = m->batch_streams + (size_t)t * FLAT;
            float z = 0;
            for (int s = 0; s < STREAMS; s++)
                z += stream[(size_t)s * HIDDEN + i];
            hidden[(size_t)t * HIDDEN + i] = z / STREAMS;
        }
    }
    int rc = next ? glm53f_target_head_argmax_batch_12n(
        m->head, m->batch_streams, tokens, next, logit) : 0;
    if (m->profile) {
        m->batch_phase[4] += MPI_Wtime() - begin;
        m->batch_calls++;
        m->batch_positions += tokens;
    }
    return rc;
}

glm53f_target_snapshot_12n *glm53f_target_snapshot_create_12n(
        const glm53f_target_model_12n *m) {
    if (!m) return NULL;
    glm53f_target_snapshot_12n *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    for (int l = 0; l < LAYERS; ++l)
        s->kda_bytes += glm53f_kda_state_bytes_12n(m->kda[l]);
    s->kda_state = a256(s->kda_bytes ? s->kda_bytes : 1);
    if (!s->kda_state) { free(s); return NULL; }
    return s;
}
void glm53f_target_snapshot_free_12n(glm53f_target_snapshot_12n*s){if(s){free(s->kda_state);free(s);}}
int glm53f_target_snapshot_save_12n(const glm53f_target_model_12n*m,glm53f_target_snapshot_12n*s){if(!m||!s)return-1;size_t off=0;for(int l=0;l<LAYERS;l++){size_t n=glm53f_kda_state_bytes_12n(m->kda[l]);if(n&&glm53f_kda_save_state_12n(m->kda[l],s->kda_state+off,n))return-1;off+=n;s->sparse_length[l]=glm53f_sparse_length_12n(m->sparse[l]);}return off==s->kda_bytes?0:-1;}
int glm53f_target_snapshot_restore_12n(glm53f_target_model_12n*m,const glm53f_target_snapshot_12n*s){if(!m||!s)return-1;size_t off=0;for(int l=0;l<LAYERS;l++){size_t n=glm53f_kda_state_bytes_12n(m->kda[l]);if(n&&glm53f_kda_restore_state_12n(m->kda[l],s->kda_state+off,n))return-1;off+=n;if(m->sparse[l]&&glm53f_sparse_restore_length_12n(m->sparse[l],s->sparse_length[l]))return-1;}return off==s->kda_bytes?0:-1;}

void glm53f_target_profile_reset_12n(glm53f_target_model_12n *m) {
    if (!m) return;
    m->scalar_steps = m->batch_calls = m->batch_positions = 0;
    memset(m->scalar_phase, 0, sizeof(m->scalar_phase));
    memset(m->batch_phase, 0, sizeof(m->batch_phase));
}

void glm53f_target_profile_report_12n(
        const glm53f_target_model_12n *m, const char *label) {
    if (!m || !m->profile) return;
    int rank;
    double scalar[5], batch[5];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Reduce(m->scalar_phase, scalar, 5, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(m->batch_phase, batch, 5, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (!rank) {
        double scalar_denom = m->scalar_steps ? m->scalar_steps : 1;
        double batch_denom = m->batch_positions ? m->batch_positions : 1;
        printf("GLM53F_TARGET_PROFILE label=%s kind=scalar steps=%ld "
               "ms_pos=%.3f embed=%.3f mhc=%.3f attention=%.3f ffn=%.3f "
               "head=%.3f\n", label ? label : "target", m->scalar_steps,
               (scalar[0]+scalar[1]+scalar[2]+scalar[3]+scalar[4])*1e3/scalar_denom,
               scalar[0]*1e3/scalar_denom, scalar[1]*1e3/scalar_denom,
               scalar[2]*1e3/scalar_denom, scalar[3]*1e3/scalar_denom,
               scalar[4]*1e3/scalar_denom);
        printf("GLM53F_TARGET_PROFILE label=%s kind=batch calls=%ld positions=%ld "
               "ms_pos=%.3f embed=%.3f mhc=%.3f attention=%.3f ffn=%.3f "
               "head=%.3f\n", label ? label : "target", m->batch_calls,
               m->batch_positions,
               (batch[0]+batch[1]+batch[2]+batch[3]+batch[4])*1e3/batch_denom,
               batch[0]*1e3/batch_denom, batch[1]*1e3/batch_denom,
               batch[2]*1e3/batch_denom, batch[3]*1e3/batch_denom,
               batch[4]*1e3/batch_denom);
    }
}

void glm53f_target_model_free_12n(glm53f_target_model_12n *m) {
    if (!m) return;
    free(m->batch_state); free(m->batch_output); free(m->batch_normalized);
    free(m->batch_streams); free(m->batch_scratch);
    free(m->streams); free(m->scratch);
    glm53f_moe_stage_free_12n(m->moe);
    for (int l = 0; l < 3; ++l) glm53f_dense_ffn_free_12n(m->dense[l]);
    for (int l = 0; l < LAYERS; ++l) {
        glm53f_sparse_free_12n(m->sparse[l]);
        glm53f_kda_free_12n(m->kda[l]);
        free((void *)m->layer_weight[l].post_attention_norm);
        free((void *)m->layer_weight[l].input_norm);
        free((void *)m->layer_weight[l].ffn_mhc.scale);
        free((void *)m->layer_weight[l].ffn_mhc.base);
        free((void *)m->layer_weight[l].ffn_mhc.fn);
        free((void *)m->layer_weight[l].attention_mhc.scale);
        free((void *)m->layer_weight[l].attention_mhc.base);
        free((void *)m->layer_weight[l].attention_mhc.fn);
    }
    glm53f_target_head_free_12n(m->head);
    glm53f_embedding_free_12n(m->embedding);
    free(m);
}

#ifndef GLM53F_TARGET_MODEL_NO_MAIN
static int read_token_ids(const char *path, int **ids_out, int *count_out) {
    FILE *f = fopen(path, "r");
    int *ids = NULL, cap = 0, n = 0, id;
    if (!f) return -1;
    while (fscanf(f, "%d", &id) == 1) {
        if (id < 0 || id >= 154880) { free(ids); fclose(f); return -1; }
        if (n == cap) {
            int next_cap = cap ? cap * 2 : 4096;
            int *next = (int *)realloc(ids, (size_t)next_cap * sizeof(*ids));
            if (!next) { free(ids); fclose(f); return -1; }
            ids = next;
            cap = next_cap;
        }
        ids[n++] = id;
    }
    if (!feof(f)) { free(ids); fclose(f); return -1; }
    fclose(f);
    if (!n) { free(ids); return -1; }
    *ids_out = ids;
    *count_out = n;
    return 0;
}

int main(int argc, char **argv) {
    int rank, ranks, token, steps, generate = 0;
    int *prompt_ids = NULL, *generated_ids = NULL, prompt_count = 0, generated = 0;
    const char *output_ids = NULL;
    glm53f_target_model_12n *model;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 4 || ranks != 12) {
        if (!rank) fprintf(stderr,"usage: %s MODEL ROUTED_STAGE SHARED_STAGE [token=1] [steps=1]\n       %s MODEL ROUTED_STAGE SHARED_STAGE --generate PROMPT_IDS OUTPUT_IDS MAX_NEW\n",argv[0],argv[0]);
        MPI_Abort(MPI_COMM_WORLD,2);
    }
    generate = argc == 8 && !strcmp(argv[4], "--generate");
    if (generate) {
        if (read_token_ids(argv[5], &prompt_ids, &prompt_count)) {
            if (!rank) { fprintf(stderr, "cannot read prompt token IDs: %s\n", argv[5]); fflush(stderr); }
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        output_ids = argv[6];
        steps = atoi(argv[7]);
        if (steps < 1 || steps > 8192) MPI_Abort(MPI_COMM_WORLD, 2);
        if (!rank && !(generated_ids = malloc((size_t)steps * sizeof(*generated_ids))))
            MPI_Abort(MPI_COMM_WORLD, 2);
        token = prompt_ids[0];
    } else {
        token=argc>4?atoi(argv[4]):1;steps=argc>5?atoi(argv[5]):1;
        if(token<0||token>=154880||steps<1||steps>128)MPI_Abort(MPI_COMM_WORLD,2);
    }
    int capacity=getenv("GLM53F_CAPACITY")?atoi(getenv("GLM53F_CAPACITY")):steps;
    if (generate && capacity < prompt_count + steps) capacity = prompt_count + steps;
    if(capacity<steps)MPI_Abort(MPI_COMM_WORLD,2);
    if(getenv("GLM53F_UTOFU")){const char*topo=getenv("TOFU_TOPO_PATH");if(!topo)topo="../utofu-tests/tofu_topo.txt";if(glm53f_collective_init_12n(topo,5*HIDDEN))MPI_Abort(MPI_COMM_WORLD,2);}
    model=glm53f_target_model_create_12n(argv[1],argv[2],argv[3],capacity);
    if(!model)MPI_Abort(MPI_COMM_WORLD,2);
    glm53f_target_profile_reset_12n(model);
    MPI_Barrier(MPI_COMM_WORLD);double begin=MPI_Wtime(),prompt_elapsed=0.0,decode_elapsed=0.0,window_begin=begin;
    int total_steps = generate ? prompt_count + steps - 1 : steps;
    int completed_steps = 0;
    for(int step=0;step<total_steps;step++){
        double step_begin=MPI_Wtime();
        if(generate&&step==prompt_count-1){window_begin=step_begin;glm53f_target_profile_reset_12n(model);}
        float value;
        if(glm53f_target_model_step_12n(model,token,&token,&value,NULL))MPI_Abort(MPI_COMM_WORLD,2);
        double step_elapsed=MPI_Wtime()-step_begin;
        if(generate&&step<prompt_count-1)prompt_elapsed+=step_elapsed;else decode_elapsed+=step_elapsed;
        completed_steps++;
        if (generate && !rank && step < prompt_count - 1 && (step + 1) % 512 == 0) {
            printf("GLM53F_TARGET_PROMPT completed=%d total=%d tok_s=%.3f\n",
                   step + 1, prompt_count - 1, (step + 1) / prompt_elapsed);
            fflush(stdout);
        }
        if (generate && step >= prompt_count - 1) {
            if (!rank) {
                generated_ids[generated] = token;
            }
            generated++;
            if(!rank&&generated%64==0){double now=MPI_Wtime();printf("GLM53F_TARGET_DECODE_WINDOW end=%d tok_s=%.3f\n",generated,64.0/(now-window_begin));fflush(stdout);window_begin=now;}
        }
        if (!generate && !rank) printf("GLM53F_TARGET_TOKEN step=%d token=%d logit=%.9g\n",step,token,value);
        if(generate&&step>=prompt_count-1&&(token==154820||token==154827||token==154829))break;
        if (generate && step + 1 < prompt_count) token = prompt_ids[step + 1];
    }
    double elapsed=MPI_Wtime()-begin,max_elapsed;MPI_Reduce(&elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(!rank){
        if(generate){FILE*f=fopen(output_ids,"w");if(!f)MPI_Abort(MPI_COMM_WORLD,2);for(int i=0;i<generated;i++)fprintf(f,"%d%c",generated_ids[i],i+1==generated?'\n':' ');fclose(f);}
        printf("GLM53F_TARGET_%s_12N steps=%d generated=%d capacity=%d ms_tok=%.3f tok_s=%.3f final_token=%d PASS\n",generate ? "GENERATE" : "DECODE",completed_steps,generate?generated:steps,capacity,max_elapsed*1e3/completed_steps,completed_steps/max_elapsed,token);
        if(generate){printf("GLM53F_TARGET_TIMING prompt_tokens=%d prompt_tok_s=%.3f decode_tokens=%d decode_tok_s=%.3f\n",prompt_count-1,(prompt_count-1)/(prompt_elapsed?prompt_elapsed:1.0),generated,generated/(decode_elapsed?decode_elapsed:1.0));}
        const char *report=getenv("GLM53F_TARGET_REPORT");
        if(report&&*report){FILE *rf=fopen(report,"w");if(rf){fprintf(rf,"GLM53F_TARGET_DECODE_12N steps=%d capacity=%d ms_tok=%.3f tok_s=%.3f final_token=%d PASS\n",steps,capacity,max_elapsed*1e3/steps,steps/max_elapsed,token);fclose(rf);}}
    }
    glm53f_target_profile_report_12n(model,"decode");
    glm53f_target_model_free_12n(model);glm53f_collective_free_12n();free(generated_ids);free(prompt_ids);
    MPI_Finalize();return 0;
}
#endif
