/* Minimal C11 numerical runtime primitives for Q38FN safetensors weights. */
#ifndef Q38FN_RUNTIME_H
#define Q38FN_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "glm53f_safetensors.h"
#include "q38fn_arch.h"

float q38fn_bf16_to_f32(uint16_t value);
int q38fn_read_bf16_row(const glm53f_st_context *ctx, const char *name,
                        size_t row, size_t columns, float *output);
uint64_t q38fn_f32_checksum(const float *values, size_t count);
int q38fn_bf16_matvec(const glm53f_st_context *ctx, const char *name,
                      const float *input, size_t rows, size_t columns,
                      float *output);
int q38fn_gated_residual_mix(const glm53f_st_context *ctx, int layer,
                             const char *block, const float *hyper_input,
                             float *mixed_input, float *injection_weights);

typedef struct {
    float *conv;
    float *recurrent;
} q38fn_delta_state;

typedef struct {
    float *conv;
    uint64_t previous;
    uint64_t previous2;
} q38fn_ple_state;

typedef struct {
    float *keys;
    float *values;
    float *scores;
    size_t length;
    size_t capacity;
} q38fn_attention_state;

int q38fn_delta_state_init(q38fn_delta_state *state);
void q38fn_delta_state_destroy(q38fn_delta_state *state);
int q38fn_gated_delta_step(const glm53f_st_context *ctx, int layer,
                           q38fn_delta_state *state, const float *input,
                           float *output);
int q38fn_moe_step(const glm53f_st_context *ctx, int layer,
                   const float *input, float *output,
                   int selected[Q38FN_ACTIVE_EXPERTS]);
int q38fn_linear_layer_step(const glm53f_st_context *ctx, int layer,
                            q38fn_delta_state *state, float *hyper_input,
                            int selected[Q38FN_ACTIVE_EXPERTS]);
int q38fn_ple_state_init(q38fn_ple_state *state);
void q38fn_ple_state_destroy(q38fn_ple_state *state);
int q38fn_ple_step(const glm53f_st_context *ctx, int layer,
                   q38fn_ple_state *state, uint64_t token, float *hyper_input);
int q38fn_ple_apply_embedding(const glm53f_st_context *ctx, int layer,
                              q38fn_ple_state *state, uint64_t token,
                              const float *embedding, float *hyper_input);
int q38fn_attention_state_init(q38fn_attention_state *state, size_t capacity);
void q38fn_attention_state_destroy(q38fn_attention_state *state);
int q38fn_attention_step(const glm53f_st_context *ctx, int layer,
                         q38fn_attention_state *state, const float *input,
                         float *output);
int q38fn_final_mix(const glm53f_st_context *ctx, const float *hyper_input,
                    float *output);
int q38fn_lm_head_argmax(const glm53f_st_context *ctx, const float *input,
                        int *token, float *logit);
int q38fn_attention_layer_step(const glm53f_st_context *ctx, int layer,
                               q38fn_attention_state *state, float *hyper_input,
                               int selected[Q38FN_ACTIVE_EXPERTS]);
int q38fn_preload_layer(const glm53f_st_context *ctx, int layer);
int q38fn_preload_tensor(const glm53f_st_context *ctx, const char *name);
int q38fn_preload_ngram_owner(const glm53f_st_context *ctx, int rank, int ranks);
size_t q38fn_cached_weight_bytes(void);

#ifdef Q38FN_RUNTIME_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#if defined(__ARM_FEATURE_SVE)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include "ggml_dequant.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

typedef struct q38fn_weight_cache_entry {
    char *name;
    size_t offset, count;
    uint16_t *data;
    struct q38fn_weight_cache_entry *next;
} q38fn_weight_cache_entry;
static q38fn_weight_cache_entry *q38fn_weight_cache;
static size_t q38fn_weight_cache_bytes;
typedef struct q38fn_ngram_q8_entry {
    char *name;size_t rows,columns;int8_t *data;float *scale;
    struct q38fn_ngram_q8_entry *next;
} q38fn_ngram_q8_entry;
static q38fn_ngram_q8_entry *q38fn_ngram_q8_cache;

static void q38fn_debug_dump(const char *name, const float *values, size_t count)
{
    const char *dir = getenv("Q38FN_DUMP_DIR");
    char path[4096]; FILE *file;
    if (!dir || !*dir || snprintf(path,sizeof(path),"%s/serial-%s.f32",dir,name)>=(int)sizeof(path) ||
        !(file=fopen(path,"wb"))) return;
    fwrite(values,sizeof(*values),count,file); fclose(file);
}

static const uint16_t *q38fn_cached_bf16(const glm53f_st_context *ctx,
                                         const char *name, size_t offset,
                                         size_t count)
{
    for (q38fn_weight_cache_entry *entry=q38fn_weight_cache;entry;entry=entry->next)
        if(!strcmp(entry->name,name)&&offset>=entry->offset&&offset-entry->offset<=entry->count&&
           count<=entry->count-(offset-entry->offset))
            return entry->data+(offset-entry->offset);
    q38fn_weight_cache_entry *entry=(q38fn_weight_cache_entry*)calloc(1,sizeof(*entry));
    if(!entry)return NULL;
    entry->name=(char*)malloc(strlen(name)+1);if(entry->name)strcpy(entry->name,name);
    entry->offset=offset;entry->count=count;
    if(getenv("Q38FN_TRACE_PRELOAD"))
        fprintf(stderr,"Q38FN_CACHE_ALLOC tensor=%s offset=%zu bytes=%zu resident_before=%zu\n",
                name,offset,count*sizeof(*entry->data),q38fn_weight_cache_bytes);
    entry->data=(uint16_t*)malloc(count*sizeof(*entry->data));
    if(!entry->name||!entry->data){
        if(getenv("Q38FN_TRACE_PRELOAD"))
            fprintf(stderr,"Q38FN_CACHE_FAIL kind=allocation tensor=%s bytes=%zu\n",
                    name,count*sizeof(*entry->data));
        free(entry->name);free(entry->data);free(entry);return NULL;
    }
    if(glm53f_st_read(ctx,name,offset*sizeof(*entry->data),entry->data,count*sizeof(*entry->data))!=0){
        if(getenv("Q38FN_TRACE_PRELOAD"))
            fprintf(stderr,"Q38FN_CACHE_FAIL kind=read tensor=%s physical_shard=%d offset=%zu bytes=%zu\n",
                    name,glm53f_st_physical_shard(ctx,name),offset*sizeof(*entry->data),
                    count*sizeof(*entry->data));
        free(entry->name);free(entry->data);free(entry);return NULL;
    }
    entry->next=q38fn_weight_cache;q38fn_weight_cache=entry;
    q38fn_weight_cache_bytes+=count*sizeof(*entry->data);return entry->data;
}

size_t q38fn_cached_weight_bytes(void){return q38fn_weight_cache_bytes;}

float q38fn_bf16_to_f32(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline float q38fn_bf16_dot(const uint16_t *weights,
                                    const float *input, size_t count)
{
    float sum=0.0f;
#ifdef _OPENMP
#pragma omp simd reduction(+:sum)
#endif
    for(size_t i=0;i<count;++i)sum+=q38fn_bf16_to_f32(weights[i])*input[i];
    return sum;
}

int q38fn_read_bf16_row(const glm53f_st_context *ctx, const char *name,
                        size_t row, size_t columns, float *output)
{
    const st_tensor_info *tensor = glm53f_st_find(ctx, name, NULL);
    const uint16_t *source;
    size_t offset;

    if (!tensor || !output || strcmp(tensor->dtype_str, "BF16") != 0 ||
        tensor->n_dims != 2 || tensor->shape[1] != columns ||
        row >= tensor->shape[0] || columns > SIZE_MAX / sizeof(*source))
        return -1;
    if(strstr(name,"ngram_embedding.shard_"))for(q38fn_ngram_q8_entry*entry=q38fn_ngram_q8_cache;entry;entry=entry->next)
        if(!strcmp(entry->name,name)&&entry->columns==columns&&row<entry->rows){const int8_t*p=entry->data+row*columns;float scale=entry->scale[row];for(size_t i=0;i<columns;++i)output[i]=(float)p[i]*scale;return 0;}
    offset = row * columns * sizeof(*source);
    source = q38fn_cached_bf16(ctx,name,offset/sizeof(*source),columns);
    if (!source) return -1;
    for (size_t i = 0; i < columns; ++i)
        output[i] = q38fn_bf16_to_f32(source[i]);
    return 0;
}

uint64_t q38fn_f32_checksum(const float *values, size_t count)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, &values[i], sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= (bits >> (byte * 8)) & UINT32_C(0xff);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

int q38fn_bf16_matvec(const glm53f_st_context *ctx, const char *name,
                      const float *input, size_t rows, size_t columns,
                      float *output)
{
    const st_tensor_info *tensor = glm53f_st_find(ctx, name, NULL);
    const uint16_t *weights;
    size_t count;

    if (!tensor || !input || !output || strcmp(tensor->dtype_str, "BF16") != 0 ||
        tensor->n_dims != 2 || tensor->shape[0] != rows || tensor->shape[1] != columns ||
        rows > SIZE_MAX / columns || (count = rows * columns) > SIZE_MAX / sizeof(*weights))
        return -1;
    weights = q38fn_cached_bf16(ctx,name,0,count);
    if (!weights) return -1;
#if defined(__ARM_FEATURE_SVE)
    size_t groups=rows/8;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t group=0;group<groups;++group){
        const uint16_t *p=weights+group*8*columns;
        matvec_bf16_8row(output+group*8,p,p+columns,p+2*columns,p+3*columns,
                         p+4*columns,p+5*columns,p+6*columns,p+7*columns,
                         input,(int)columns);
    }
    for(size_t row=groups*8;row<rows;++row)
        output[row]=q38fn_bf16_dot(weights+row*columns,input,columns);
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t row = 0; row < rows; ++row) {
        const uint16_t *weight_row = weights + row * columns;
        output[row] = q38fn_bf16_dot(weight_row,input,columns);
    }
#endif
    return 0;
}

static float q38fn_sigmoid(float value)
{
    return 1.0f / (1.0f + expf(-value));
}

static float q38fn_silu(float value)
{
    return value * q38fn_sigmoid(value);
}

static float q38fn_softplus(float value)
{
    return value > 20.0f ? value : log1pf(expf(value));
}

static int q38fn_read_bf16_flat(const glm53f_st_context *ctx, const char *name,
                                size_t count, float *output)
{
    const st_tensor_info *tensor = glm53f_st_find(ctx, name, NULL);
    const uint16_t *raw;
    size_t actual = 1;

    if (!tensor || !output || strcmp(tensor->dtype_str, "BF16") != 0) return -1;
    for (int d = 0; d < tensor->n_dims; ++d) {
        if (tensor->shape[d] > SIZE_MAX / actual) return -1;
        actual *= (size_t)tensor->shape[d];
    }
    if (actual != count || count > SIZE_MAX / sizeof(*raw)) return -1;
    raw=q38fn_cached_bf16(ctx,name,0,count);if(!raw)return -1;
    for (size_t i = 0; i < count; ++i) output[i] = q38fn_bf16_to_f32(raw[i]);
    return 0;
}

int q38fn_gated_residual_mix(const glm53f_st_context *ctx, int layer,
                             const char *block, const float *hyper_input,
                             float *mixed_input, float *injection_weights)
{
    enum { HC_WIDTH = Q38FN_HC_COUNT * Q38FN_HIDDEN, HC_LOWRANK = 320 };
    char prefix[160], name[224];
    float normed[HC_WIDTH],mix_weight[HC_WIDTH];
    const uint16_t *norm_weight;
    float lowrank[HC_LOWRANK];
    int rc = -1;

    if (!ctx || !block || !hyper_input || !mixed_input || !injection_weights ||
        layer < 0 || layer >= Q38FN_LAYERS ||
        snprintf(prefix, sizeof(prefix), "model.language_model.layers.%d.%s_hyper_connection",
                 layer, block) >= (int)sizeof(prefix))
        return -1;
    snprintf(name, sizeof(name), "%s.hc_norm.weight", prefix);
    /* RMSNorm weights are stored as a vector, not a matrix. */
    {
        const st_tensor_info *tensor = glm53f_st_find(ctx, name, NULL);
        if (!tensor || strcmp(tensor->dtype_str, "BF16") != 0 ||
            tensor->n_dims != 1 || tensor->shape[0] != HC_WIDTH) goto out;
        norm_weight=q38fn_cached_bf16(ctx,name,0,HC_WIDTH);if(!norm_weight)goto out;
    }
    for (int stream = 0; stream < Q38FN_HC_COUNT; ++stream) {
        double square_sum = 0.0;
        int base = stream * Q38FN_HIDDEN;
        for (int i = 0; i < Q38FN_HIDDEN; ++i)
            square_sum += (double)hyper_input[base + i] * hyper_input[base + i];
        float scale = 1.0f / sqrtf((float)(square_sum / Q38FN_HIDDEN) + 1.0e-6f);
        for (int i = 0; i < Q38FN_HIDDEN; ++i)
            normed[base + i] = hyper_input[base + i] * scale *
                               (1.0f + q38fn_bf16_to_f32(norm_weight[base + i]));
    }
    snprintf(name, sizeof(name), "%s.input_mix_weight_down.weight", prefix);
    if (q38fn_bf16_matvec(ctx, name, normed, HC_LOWRANK, HC_WIDTH, lowrank) != 0) goto out;
    for (int i = 0; i < HC_LOWRANK; ++i) {
        float value = lowrank[i] / Q38FN_HC_COUNT;
        lowrank[i] = value * q38fn_sigmoid(value);
    }
    snprintf(name, sizeof(name), "%s.input_mix_weight_up.weight", prefix);
    if (q38fn_bf16_matvec(ctx, name, lowrank, HC_WIDTH, HC_LOWRANK, mix_weight) != 0) goto out;
    for (int i = 0; i < HC_WIDTH; ++i) mix_weight[i] = q38fn_sigmoid(mix_weight[i]);
    for (int i = 0; i < Q38FN_HIDDEN; ++i) {
        double sum = 0.0;
        for (int stream = 0; stream < Q38FN_HC_COUNT; ++stream) {
            int index = stream * Q38FN_HIDDEN + i;
            sum += (double)mix_weight[index] * normed[index];
        }
        mixed_input[i] = (float)(sum / Q38FN_HC_COUNT);
    }
    snprintf(name, sizeof(name), "%s.block_inject_weight.weight", prefix);
    if (q38fn_bf16_matvec(ctx, name, normed, Q38FN_HC_COUNT, HC_WIDTH,
                          injection_weights) != 0) goto out;
    for (int i = 0; i < Q38FN_HC_COUNT; ++i)
        injection_weights[i] = 2.0f * q38fn_sigmoid(injection_weights[i] / Q38FN_HC_COUNT);
    rc = 0;
out:
    return rc;
}

int q38fn_delta_state_init(q38fn_delta_state *state)
{
    size_t conv_count = (size_t)Q38FN_LINEAR_CONV_DIM * Q38FN_LINEAR_CONV_KERNEL;
    size_t recurrent_count = (size_t)Q38FN_LINEAR_VALUE_HEADS *
                             Q38FN_LINEAR_HEAD_DIM * Q38FN_LINEAR_HEAD_DIM;
    if (!state) return -1;
    state->conv = (float *)calloc(conv_count, sizeof(*state->conv));
    state->recurrent = (float *)calloc(recurrent_count, sizeof(*state->recurrent));
    if (!state->conv || !state->recurrent) {
        q38fn_delta_state_destroy(state); return -1;
    }
    return 0;
}

void q38fn_delta_state_destroy(q38fn_delta_state *state)
{
    if (!state) return;
    free(state->conv); free(state->recurrent);
    state->conv = NULL; state->recurrent = NULL;
}

int q38fn_gated_delta_step(const glm53f_st_context *ctx, int layer,
                           q38fn_delta_state *state, const float *input,
                           float *output)
{
    char name[192];
    static _Thread_local float qkv_storage[Q38FN_LINEAR_CONV_DIM];
    static _Thread_local float z_storage[Q38FN_LINEAR_VALUE_DIM];
    static _Thread_local float core_storage[Q38FN_LINEAR_VALUE_DIM];
    float *qkv=qkv_storage,*z=z_storage,*core=core_storage;
    const uint16_t *conv_weight;
    float a[Q38FN_LINEAR_VALUE_HEADS], b[Q38FN_LINEAR_VALUE_HEADS];
    float a_log[Q38FN_LINEAR_VALUE_HEADS], dt_bias[Q38FN_LINEAR_VALUE_HEADS];
    float norm_weight[Q38FN_LINEAR_HEAD_DIM];
    int rc = -1;

    if (!ctx || !state || !state->conv || !state->recurrent || !input || !output ||
        layer < 0 || layer >= Q38FN_LAYERS || q38fn_layer_is_full_attention((size_t)layer))
        return -1;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", layer);
    if (q38fn_bf16_matvec(ctx, name, input, Q38FN_LINEAR_CONV_DIM, Q38FN_HIDDEN, qkv) != 0) goto out;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_z.weight", layer);
    if (q38fn_bf16_matvec(ctx, name, input, Q38FN_LINEAR_VALUE_DIM, Q38FN_HIDDEN, z) != 0) goto out;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_a.weight", layer);
    if (q38fn_bf16_matvec(ctx, name, input, Q38FN_LINEAR_VALUE_HEADS, Q38FN_HIDDEN, a) != 0) goto out;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_b.weight", layer);
    if (q38fn_bf16_matvec(ctx, name, input, Q38FN_LINEAR_VALUE_HEADS, Q38FN_HIDDEN, b) != 0) goto out;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.conv1d.weight", layer);
    conv_weight=q38fn_cached_bf16(ctx,name,0,(size_t)Q38FN_LINEAR_CONV_DIM*
                                  Q38FN_LINEAR_CONV_KERNEL);
    if(!conv_weight)goto out;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.A_log", layer);
    if (q38fn_read_bf16_flat(ctx, name, Q38FN_LINEAR_VALUE_HEADS, a_log) != 0) goto out;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.dt_bias", layer);
    if (q38fn_read_bf16_flat(ctx, name, Q38FN_LINEAR_VALUE_HEADS, dt_bias) != 0) goto out;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.norm.weight", layer);
    if (q38fn_read_bf16_flat(ctx, name, Q38FN_LINEAR_HEAD_DIM, norm_weight) != 0) goto out;

    for (int channel = 0; channel < Q38FN_LINEAR_CONV_DIM; ++channel) {
        float *history = state->conv + (size_t)channel * Q38FN_LINEAR_CONV_KERNEL;
        const uint16_t *weight = conv_weight + (size_t)channel * Q38FN_LINEAR_CONV_KERNEL;
        float sum = 0.0f;
        for (int k = 0; k + 1 < Q38FN_LINEAR_CONV_KERNEL; ++k) history[k] = history[k + 1];
        history[Q38FN_LINEAR_CONV_KERNEL - 1] = qkv[channel];
        for (int k = 0; k < Q38FN_LINEAR_CONV_KERNEL; ++k)
            sum += history[k] * q38fn_bf16_to_f32(weight[k]);
        qkv[channel] = q38fn_silu(sum);
    }
    q38fn_debug_dump("delta-qkv", qkv, Q38FN_LINEAR_CONV_DIM);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int head = 0; head < Q38FN_LINEAR_VALUE_HEADS; ++head) {
        const int key_head = head / (Q38FN_LINEAR_VALUE_HEADS / Q38FN_LINEAR_KEY_HEADS);
        const float *query_source = qkv + key_head * Q38FN_LINEAR_HEAD_DIM;
        const float *key_source = qkv + Q38FN_LINEAR_KEY_DIM + key_head * Q38FN_LINEAR_HEAD_DIM;
        const float *value = qkv + 2 * Q38FN_LINEAR_KEY_DIM + head * Q38FN_LINEAR_HEAD_DIM;
        float *matrix = state->recurrent + (size_t)head * Q38FN_LINEAR_HEAD_DIM * Q38FN_LINEAR_HEAD_DIM;
        float query[Q38FN_LINEAR_HEAD_DIM], key[Q38FN_LINEAR_HEAD_DIM];
        float delta[Q38FN_LINEAR_HEAD_DIM];
        double q2 = 0.0, k2 = 0.0;
        for (int i = 0; i < Q38FN_LINEAR_HEAD_DIM; ++i) {
            q2 += (double)query_source[i] * query_source[i];
            k2 += (double)key_source[i] * key_source[i];
        }
        float q_scale = 1.0f / (sqrtf((float)q2 + 1.0e-6f) * sqrtf(Q38FN_LINEAR_HEAD_DIM));
        float k_scale = 1.0f / sqrtf((float)k2 + 1.0e-6f);
        for (int i = 0; i < Q38FN_LINEAR_HEAD_DIM; ++i) {
            query[i] = query_source[i] * q_scale;
            key[i] = key_source[i] * k_scale;
        }
        float decay = expf(-expf(a_log[head]) * q38fn_softplus(a[head] + dt_bias[head]));
        float beta = q38fn_sigmoid(b[head]);
        for (int j = 0; j < Q38FN_LINEAR_HEAD_DIM; ++j) {
            double memory_value = 0.0;
            for (int i = 0; i < Q38FN_LINEAR_HEAD_DIM; ++i) {
                size_t index = (size_t)i * Q38FN_LINEAR_HEAD_DIM + j;
                matrix[index] *= decay;
                memory_value += (double)matrix[index] * key[i];
            }
            delta[j] = (value[j] - (float)memory_value) * beta;
        }
        for (int i = 0; i < Q38FN_LINEAR_HEAD_DIM; ++i)
            for (int j = 0; j < Q38FN_LINEAR_HEAD_DIM; ++j)
                matrix[(size_t)i * Q38FN_LINEAR_HEAD_DIM + j] += key[i] * delta[j];
        double square_sum = 0.0;
        for (int j = 0; j < Q38FN_LINEAR_HEAD_DIM; ++j) {
            double value_out = 0.0;
            for (int i = 0; i < Q38FN_LINEAR_HEAD_DIM; ++i)
                value_out += (double)matrix[(size_t)i * Q38FN_LINEAR_HEAD_DIM + j] * query[i];
            core[head * Q38FN_LINEAR_HEAD_DIM + j] = (float)value_out;
            square_sum += value_out * value_out;
        }
        float norm_scale = 1.0f / sqrtf((float)(square_sum / Q38FN_LINEAR_HEAD_DIM) + 1.0e-6f);
        for (int j = 0; j < Q38FN_LINEAR_HEAD_DIM; ++j) {
            int index = head * Q38FN_LINEAR_HEAD_DIM + j;
            core[index] = core[index] * norm_scale * norm_weight[j] * q38fn_sigmoid(z[index]);
        }
    }
    q38fn_debug_dump("delta-core", core, Q38FN_LINEAR_VALUE_DIM);
    snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.out_proj.weight", layer);
    if (q38fn_bf16_matvec(ctx, name, core, Q38FN_HIDDEN, Q38FN_LINEAR_VALUE_DIM, output) != 0) goto out;
    rc = 0;
out:
    return rc;
}

int q38fn_moe_step(const glm53f_st_context *ctx, int layer,
                   const float *input, float *output,
                   int selected[Q38FN_ACTIVE_EXPERTS])
{
    enum { EXPERT_PACKED = 2 * Q38FN_EXPERT_INTERMEDIATE };
    char name[192];
    float logits[Q38FN_EXPERTS], probabilities[Q38FN_EXPERTS];
    float selected_weight[Q38FN_ACTIVE_EXPERTS];
    float *packed = NULL;
    static _Thread_local float packed_storage[Q38FN_ACTIVE_EXPERTS*EXPERT_PACKED];
    const uint16_t *all_gate_up = NULL, *all_down = NULL;
    const uint16_t *shared_gate_weights=NULL,*shared_up_weights=NULL;
    const uint16_t *shared_down_weights=NULL,*shared_scale_weights=NULL;
    float shared_gate[Q38FN_EXPERT_INTERMEDIATE],shared_hidden[Q38FN_EXPERT_INTERMEDIATE];
    float shared_scale[1];
    int local_selected[Q38FN_ACTIVE_EXPERTS];
    int *chosen = selected ? selected : local_selected;
    int rc = -1;

    if (!ctx || !input || !output || layer < 0 || layer >= Q38FN_LAYERS) return -1;
    memset(output, 0, Q38FN_HIDDEN * sizeof(*output));
    snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.gate.weight", layer);
    if (q38fn_bf16_matvec(ctx, name, input, Q38FN_EXPERTS, Q38FN_HIDDEN, logits) != 0) return -1;
    float maximum = logits[0], denominator = 0.0f;
    for (int i = 1; i < Q38FN_EXPERTS; ++i) if (logits[i] > maximum) maximum = logits[i];
    for (int i = 0; i < Q38FN_EXPERTS; ++i) {
        probabilities[i] = expf(logits[i] - maximum);
        denominator += probabilities[i];
    }
    for (int i = 0; i < Q38FN_EXPERTS; ++i) probabilities[i] /= denominator;
    float selected_sum = 0.0f;
    for (int slot = 0; slot < Q38FN_ACTIVE_EXPERTS; ++slot) {
        int best = -1;
        for (int expert = 0; expert < Q38FN_EXPERTS; ++expert) {
            int used = 0;
            for (int prior = 0; prior < slot; ++prior) if (chosen[prior] == expert) used = 1;
            if (!used && (best < 0 || probabilities[expert] > probabilities[best])) best = expert;
        }
        chosen[slot] = best;
        selected_weight[slot] = probabilities[best];
        selected_sum += selected_weight[slot];
    }
    for (int slot = 0; slot < Q38FN_ACTIVE_EXPERTS; ++slot) selected_weight[slot] /= selected_sum;

    packed=packed_storage;

    size_t gate_count=(size_t)EXPERT_PACKED*Q38FN_HIDDEN;
    size_t down_count=(size_t)Q38FN_HIDDEN*Q38FN_EXPERT_INTERMEDIATE;
    snprintf(name,sizeof(name),"model.language_model.layers.%d.mlp.experts.gate_up_proj",layer);
    all_gate_up=q38fn_cached_bf16(ctx,name,0,(size_t)Q38FN_EXPERTS*gate_count);
    snprintf(name,sizeof(name),"model.language_model.layers.%d.mlp.experts.down_proj",layer);
    all_down=q38fn_cached_bf16(ctx,name,0,(size_t)Q38FN_EXPERTS*down_count);
    snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.shared_expert.gate_proj.weight", layer);
    shared_gate_weights=q38fn_cached_bf16(ctx,name,0,(size_t)Q38FN_EXPERT_INTERMEDIATE*Q38FN_HIDDEN);
    snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.shared_expert.up_proj.weight", layer);
    shared_up_weights=q38fn_cached_bf16(ctx,name,0,(size_t)Q38FN_EXPERT_INTERMEDIATE*Q38FN_HIDDEN);
    snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.shared_expert.down_proj.weight", layer);
    shared_down_weights=q38fn_cached_bf16(ctx,name,0,(size_t)Q38FN_HIDDEN*Q38FN_EXPERT_INTERMEDIATE);
    if(!all_gate_up||!all_down||!shared_gate_weights||!shared_up_weights||!shared_down_weights)goto out;
#if defined(__ARM_FEATURE_SVE)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int task=0;task<(Q38FN_ACTIVE_EXPERTS+1)*(EXPERT_PACKED/8);++task){
        int slot=task/(EXPERT_PACKED/8),row=(task%(EXPERT_PACKED/8))*8;
        const uint16_t *w;float temporary[8],*destination;
        if(slot<Q38FN_ACTIVE_EXPERTS){w=all_gate_up+(size_t)chosen[slot]*gate_count+(size_t)row*Q38FN_HIDDEN;destination=packed+(size_t)slot*EXPERT_PACKED+row;}
        else {w=(row<Q38FN_EXPERT_INTERMEDIATE?shared_gate_weights:shared_up_weights)+(size_t)(row%Q38FN_EXPERT_INTERMEDIATE)*Q38FN_HIDDEN;destination=temporary;}
        matvec_bf16_8row(destination,w,w+Q38FN_HIDDEN,w+2*Q38FN_HIDDEN,w+3*Q38FN_HIDDEN,w+4*Q38FN_HIDDEN,w+5*Q38FN_HIDDEN,w+6*Q38FN_HIDDEN,w+7*Q38FN_HIDDEN,input,Q38FN_HIDDEN);
        if(slot==Q38FN_ACTIVE_EXPERTS)for(int lane=0;lane<8;++lane)(row<Q38FN_EXPERT_INTERMEDIATE?shared_gate:shared_hidden)[row%Q38FN_EXPERT_INTERMEDIATE+lane]=temporary[lane];
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int task=0;task<(Q38FN_ACTIVE_EXPERTS+1)*EXPERT_PACKED;++task){
        int slot=task/EXPERT_PACKED,row=task%EXPERT_PACKED;
        if(slot<Q38FN_ACTIVE_EXPERTS){const uint16_t*w=all_gate_up+(size_t)chosen[slot]*gate_count+(size_t)row*Q38FN_HIDDEN;packed[(size_t)slot*EXPERT_PACKED+row]=q38fn_bf16_dot(w,input,Q38FN_HIDDEN);}
        else {const uint16_t*w=(row<Q38FN_EXPERT_INTERMEDIATE?shared_gate_weights:shared_up_weights)+(size_t)(row%Q38FN_EXPERT_INTERMEDIATE)*Q38FN_HIDDEN;(row<Q38FN_EXPERT_INTERMEDIATE?shared_gate:shared_hidden)[row%Q38FN_EXPERT_INTERMEDIATE]=q38fn_bf16_dot(w,input,Q38FN_HIDDEN);}
    }
#endif
    for(int slot=0;slot<Q38FN_ACTIVE_EXPERTS;++slot)for(int i=0;i<Q38FN_EXPERT_INTERMEDIATE;++i){float*p=packed+(size_t)slot*EXPERT_PACKED;p[i]=q38fn_silu(p[i])*p[Q38FN_EXPERT_INTERMEDIATE+i];}
    for (int i = 0; i < Q38FN_EXPERT_INTERMEDIATE; ++i)
        shared_hidden[i] *= q38fn_silu(shared_gate[i]);
    snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.shared_expert_gate.weight", layer);
    shared_scale_weights=q38fn_cached_bf16(ctx,name,0,Q38FN_HIDDEN);if(!shared_scale_weights)goto out;
    shared_scale[0]=q38fn_bf16_dot(shared_scale_weights,input,Q38FN_HIDDEN);
    float scale = q38fn_sigmoid(shared_scale[0]);
#if defined(__ARM_FEATURE_SVE)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int row=0;row<Q38FN_HIDDEN;row+=8){float sum[8]={0},part[8];for(int slot=0;slot<Q38FN_ACTIVE_EXPERTS;++slot){const uint16_t*w=all_down+(size_t)chosen[slot]*down_count+(size_t)row*Q38FN_EXPERT_INTERMEDIATE;const float*x=packed+(size_t)slot*EXPERT_PACKED;matvec_bf16_8row(part,w,w+Q38FN_EXPERT_INTERMEDIATE,w+2*Q38FN_EXPERT_INTERMEDIATE,w+3*Q38FN_EXPERT_INTERMEDIATE,w+4*Q38FN_EXPERT_INTERMEDIATE,w+5*Q38FN_EXPERT_INTERMEDIATE,w+6*Q38FN_EXPERT_INTERMEDIATE,w+7*Q38FN_EXPERT_INTERMEDIATE,x,Q38FN_EXPERT_INTERMEDIATE);for(int lane=0;lane<8;++lane)sum[lane]+=selected_weight[slot]*part[lane];}const uint16_t*w=shared_down_weights+(size_t)row*Q38FN_EXPERT_INTERMEDIATE;matvec_bf16_8row(part,w,w+Q38FN_EXPERT_INTERMEDIATE,w+2*Q38FN_EXPERT_INTERMEDIATE,w+3*Q38FN_EXPERT_INTERMEDIATE,w+4*Q38FN_EXPERT_INTERMEDIATE,w+5*Q38FN_EXPERT_INTERMEDIATE,w+6*Q38FN_EXPERT_INTERMEDIATE,w+7*Q38FN_EXPERT_INTERMEDIATE,shared_hidden,Q38FN_EXPERT_INTERMEDIATE);for(int lane=0;lane<8;++lane)output[row+lane]=sum[lane]+scale*part[lane];}
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int row=0;row<Q38FN_HIDDEN;++row){float sum=0;for(int slot=0;slot<Q38FN_ACTIVE_EXPERTS;++slot){const uint16_t*w=all_down+(size_t)chosen[slot]*down_count+(size_t)row*Q38FN_EXPERT_INTERMEDIATE;const float*x=packed+(size_t)slot*EXPERT_PACKED;sum+=selected_weight[slot]*q38fn_bf16_dot(w,x,Q38FN_EXPERT_INTERMEDIATE);}output[row]=sum+scale*q38fn_bf16_dot(shared_down_weights+(size_t)row*Q38FN_EXPERT_INTERMEDIATE,shared_hidden,Q38FN_EXPERT_INTERMEDIATE);}
#endif
    rc = 0;
out:
    return rc;
}

int q38fn_linear_layer_step(const glm53f_st_context *ctx, int layer,
                            q38fn_delta_state *state, float *hyper_input,
                            int selected[Q38FN_ACTIVE_EXPERTS])
{
    float mixed[Q38FN_HIDDEN], block_output[Q38FN_HIDDEN];
    float injection[Q38FN_HC_COUNT];
    if (!ctx || !state || !hyper_input || q38fn_layer_is_full_attention((size_t)layer)) return -1;
    if (q38fn_gated_residual_mix(ctx, layer, "attn", hyper_input, mixed, injection) != 0 ||
        q38fn_gated_delta_step(ctx, layer, state, mixed, block_output) != 0) return -1;
    for (int stream = 0; stream < Q38FN_HC_COUNT; ++stream)
        for (int i = 0; i < Q38FN_HIDDEN; ++i)
            hyper_input[stream * Q38FN_HIDDEN + i] += injection[stream] * block_output[i];
    if (q38fn_gated_residual_mix(ctx, layer, "mlp", hyper_input, mixed, injection) != 0 ||
        q38fn_moe_step(ctx, layer, mixed, block_output, selected) != 0) return -1;
    for (int stream = 0; stream < Q38FN_HC_COUNT; ++stream)
        for (int i = 0; i < Q38FN_HIDDEN; ++i)
            hyper_input[stream * Q38FN_HIDDEN + i] += injection[stream] * block_output[i];
    return 0;
}

int q38fn_ple_state_init(q38fn_ple_state *state)
{
    enum { PLE_HISTORY = (4 - 1) * Q38FN_NGRAM_SIZE };
    if (!state) return -1;
    state->conv = (float *)calloc((size_t)Q38FN_HC_COUNT * Q38FN_HIDDEN * PLE_HISTORY,
                                  sizeof(*state->conv));
    state->previous = Q38FN_EOS;
    state->previous2 = Q38FN_EOS;
    return state->conv ? 0 : -1;
}

void q38fn_ple_state_destroy(q38fn_ple_state *state)
{
    if (!state) return;
    free(state->conv); state->conv = NULL;
}

int q38fn_ple_apply_embedding(const glm53f_st_context *ctx, int layer,
                              q38fn_ple_state *state, uint64_t token,
                              const float *provided_embedding, float *hyper_input)
{
    enum { HC_WIDTH = Q38FN_HC_COUNT * Q38FN_HIDDEN, PLE_HISTORY = 9 };
    char name[224];
    float value[Q38FN_HIDDEN];
    static _Thread_local float key_storage[HC_WIDTH],key_norm_storage[HC_WIDTH];
    static _Thread_local float query_norm_storage[HC_WIDTH],gated_storage[HC_WIDTH];
    float *key=key_storage,*key_norm=key_norm_storage;
    float *query_norm=query_norm_storage,*gated=gated_storage;
    const uint16_t *norm_weight,*conv_weight;
    int rc = -1;
    if (!ctx || !state || !state->conv || !hyper_input || !provided_embedding ||
        layer != Q38FN_PLE_LAYER) return -1;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.ple.key_proj.weight", layer);
    if (q38fn_bf16_matvec(ctx, name, provided_embedding, HC_WIDTH, Q38FN_HIDDEN, key) != 0) goto out;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.ple.value_proj.weight", layer);
    if (q38fn_bf16_matvec(ctx, name, provided_embedding, Q38FN_HIDDEN, Q38FN_HIDDEN, value) != 0) goto out;

    snprintf(name, sizeof(name), "model.language_model.layers.%d.ple.norm_key.weight", layer);
    norm_weight=q38fn_cached_bf16(ctx,name,0,HC_WIDTH);if(!norm_weight)goto out;
    for (int stream = 0; stream < Q38FN_HC_COUNT; ++stream) {
        double squares = 0.0;
        int base = stream * Q38FN_HIDDEN;
        for (int i = 0; i < Q38FN_HIDDEN; ++i) squares += (double)key[base+i] * key[base+i];
        float scale = 1.0f / sqrtf((float)(squares / Q38FN_HIDDEN) + 1.0e-6f);
        for (int i = 0; i < Q38FN_HIDDEN; ++i)
            key_norm[base+i] = key[base+i] * scale *
                               (1.0f + q38fn_bf16_to_f32(norm_weight[base+i]));
    }
    snprintf(name, sizeof(name), "model.language_model.layers.%d.ple.norm_query.weight", layer);
    norm_weight=q38fn_cached_bf16(ctx,name,0,HC_WIDTH);if(!norm_weight)goto out;
    for (int stream = 0; stream < Q38FN_HC_COUNT; ++stream) {
        double squares = 0.0;
        int base = stream * Q38FN_HIDDEN;
        for (int i = 0; i < Q38FN_HIDDEN; ++i) squares += (double)hyper_input[base+i] * hyper_input[base+i];
        float scale = 1.0f / sqrtf((float)(squares / Q38FN_HIDDEN) + 1.0e-6f);
        for (int i = 0; i < Q38FN_HIDDEN; ++i)
            query_norm[base+i] = hyper_input[base+i] * scale *
                                 (1.0f + q38fn_bf16_to_f32(norm_weight[base+i]));
        double dot = 0.0;
        for (int i = 0; i < Q38FN_HIDDEN; ++i) dot += (double)key_norm[base+i] * query_norm[base+i];
        float gate = (float)(dot / sqrtf(Q38FN_HIDDEN));
        gate = copysignf(sqrtf(fmaxf(fabsf(gate), 1.0e-6f)), gate);
        float scale_gate = q38fn_sigmoid(gate);
        for (int i = 0; i < Q38FN_HIDDEN; ++i) gated[base+i] = scale_gate * value[i];
    }
    snprintf(name, sizeof(name), "model.language_model.layers.%d.ple.norm_conv.weight", layer);
    norm_weight=q38fn_cached_bf16(ctx,name,0,HC_WIDTH);if(!norm_weight)goto out;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.ple.conv1d.weight", layer);
    conv_weight=q38fn_cached_bf16(ctx,name,0,(size_t)HC_WIDTH*4);if(!conv_weight)goto out;
    for (int stream = 0; stream < Q38FN_HC_COUNT; ++stream) {
        double squares = 0.0;
        int base = stream * Q38FN_HIDDEN;
        for (int i = 0; i < Q38FN_HIDDEN; ++i) squares += (double)gated[base+i] * gated[base+i];
        float norm_scale = 1.0f / sqrtf((float)(squares / Q38FN_HIDDEN) + 1.0e-6f);
        for (int i = 0; i < Q38FN_HIDDEN; ++i) {
            int channel = base + i;
            float normalized = gated[channel] * norm_scale *
                               (1.0f + q38fn_bf16_to_f32(norm_weight[channel]));
            float *history = state->conv + (size_t)channel * PLE_HISTORY;
            const uint16_t *weight = conv_weight + (size_t)channel * 4;
            float convolution = history[0]*q38fn_bf16_to_f32(weight[0]) +
                                history[3]*q38fn_bf16_to_f32(weight[1]) +
                                history[6]*q38fn_bf16_to_f32(weight[2]) +
                                normalized*q38fn_bf16_to_f32(weight[3]);
            for (int h = 0; h + 1 < PLE_HISTORY; ++h) history[h] = history[h+1];
            history[PLE_HISTORY-1] = normalized;
            hyper_input[channel] += gated[channel] + q38fn_silu(convolution);
        }
    }
    state->previous2 = state->previous;
    state->previous = token;
    rc = 0;
out:
    return rc;
}

int q38fn_ple_step(const glm53f_st_context *ctx, int layer,
                   q38fn_ple_state *state, uint64_t token, float *hyper_input)
{
    uint64_t rows[Q38FN_NGRAM_HEADS]; float embedding[Q38FN_HIDDEN];
    if(!state)return -1;
    q38fn_ngram_rows(token,state->previous,state->previous2,rows);
    for(int head=0;head<Q38FN_NGRAM_HEADS;++head){
        uint64_t shard=rows[head]/Q38FN_NGRAM_ROWS_PER_SHARD;
        uint64_t local=rows[head]%Q38FN_NGRAM_ROWS_PER_SHARD;char name[224];
        snprintf(name,sizeof(name),"model.language_model.layers.%d.ple.ple_embedding.ngram_embedding.shard_%llu.weight",layer,(unsigned long long)shard);
        if(q38fn_read_bf16_row(ctx,name,(size_t)local,Q38FN_NGRAM_HEAD_DIM,
                               embedding+head*Q38FN_NGRAM_HEAD_DIM)!=0)return -1;
    }
    return q38fn_ple_apply_embedding(ctx,layer,state,token,embedding,hyper_input);
}

int q38fn_attention_state_init(q38fn_attention_state *state, size_t capacity)
{
    size_t width = (size_t)Q38FN_KV_HEADS * Q38FN_HEAD_DIM;
    if (!state || !capacity || capacity > SIZE_MAX / width) return -1;
    state->keys = (float *)calloc(capacity * width, sizeof(*state->keys));
    state->values = (float *)calloc(capacity * width, sizeof(*state->values));
    state->scores = (float *)malloc(capacity * sizeof(*state->scores));
    state->length = 0; state->capacity = capacity;
    if (!state->keys || !state->values || !state->scores) {
        q38fn_attention_state_destroy(state); return -1;
    }
    return 0;
}

void q38fn_attention_state_destroy(q38fn_attention_state *state)
{
    if (!state) return;
    free(state->keys); free(state->values); free(state->scores);
    state->keys = state->values = state->scores = NULL;
    state->length = state->capacity = 0;
}

static int q38fn_norm_heads(const glm53f_st_context *ctx, const char *name,
                            float *values, int heads, int dim)
{
    const uint16_t *weight=q38fn_cached_bf16(ctx,name,0,(size_t)dim);
    if(!weight)return -1;
    for (int head = 0; head < heads; ++head) {
        double squares = 0.0;
        float *vector = values + head * dim;
        for (int i = 0; i < dim; ++i) squares += (double)vector[i] * vector[i];
        float scale = 1.0f / sqrtf((float)(squares / dim) + 1.0e-6f);
        for (int i = 0; i < dim; ++i)
            vector[i] *= scale * (1.0f + q38fn_bf16_to_f32(weight[i]));
    }
    return 0;
}

static void q38fn_rope_text(float *vector, size_t position)
{
    enum { ROTARY = Q38FN_HEAD_DIM / 4, HALF = ROTARY / 2 };
    float first[HALF], second[HALF];
    for (int i = 0; i < HALF; ++i) {
        first[i] = vector[i]; second[i] = vector[HALF+i];
    }
    for (int i = 0; i < HALF; ++i) {
        float angle = (float)position / powf(10000000.0f, (float)(2*i) / ROTARY);
        float cosine = cosf(angle), sine = sinf(angle);
        vector[i] = first[i] * cosine - second[i] * sine;
        vector[HALF+i] = second[i] * cosine + first[i] * sine;
    }
}

int q38fn_attention_step(const glm53f_st_context *ctx, int layer,
                         q38fn_attention_state *state, const float *input,
                         float *output)
{
    enum { QUERY_WIDTH = Q38FN_HEADS * Q38FN_HEAD_DIM, KV_WIDTH = Q38FN_KV_HEADS * Q38FN_HEAD_DIM };
    char name[192];
    static _Thread_local float qg_storage[2*QUERY_WIDTH],query_storage[QUERY_WIDTH];
    static _Thread_local float gate_storage[QUERY_WIDTH],attended_storage[QUERY_WIDTH];
    float *qg=qg_storage,*query=query_storage,*gate=gate_storage;
    float *attended=attended_storage,*scores=state?state->scores:NULL;
    float key[KV_WIDTH], value[KV_WIDTH];
    int rc = -1;
    if (!ctx || !state || !input || !output || !q38fn_layer_is_full_attention((size_t)layer) ||
        state->length >= state->capacity) return -1;
    memset(attended,0,QUERY_WIDTH*sizeof(*attended));
    snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_proj.weight", layer);
    if (q38fn_bf16_matvec(ctx, name, input, 2 * QUERY_WIDTH, Q38FN_HIDDEN, qg) != 0) goto out;
    for (int head = 0; head < Q38FN_HEADS; ++head) {
        memcpy(query + head * Q38FN_HEAD_DIM, qg + head * 2 * Q38FN_HEAD_DIM,
               Q38FN_HEAD_DIM * sizeof(float));
        memcpy(gate + head * Q38FN_HEAD_DIM, qg + head * 2 * Q38FN_HEAD_DIM + Q38FN_HEAD_DIM,
               Q38FN_HEAD_DIM * sizeof(float));
    }
    snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_proj.weight", layer);
    if (q38fn_bf16_matvec(ctx, name, input, KV_WIDTH, Q38FN_HIDDEN, key) != 0) goto out;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.v_proj.weight", layer);
    if (q38fn_bf16_matvec(ctx, name, input, KV_WIDTH, Q38FN_HIDDEN, value) != 0) goto out;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_norm.weight", layer);
    if (q38fn_norm_heads(ctx, name, query, Q38FN_HEADS, Q38FN_HEAD_DIM) != 0) goto out;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_norm.weight", layer);
    if (q38fn_norm_heads(ctx, name, key, Q38FN_KV_HEADS, Q38FN_HEAD_DIM) != 0) goto out;
    for (int h = 0; h < Q38FN_HEADS; ++h) q38fn_rope_text(query + h*Q38FN_HEAD_DIM, state->length);
    for (int h = 0; h < Q38FN_KV_HEADS; ++h) q38fn_rope_text(key + h*Q38FN_HEAD_DIM, state->length);
    memcpy(state->keys + state->length * KV_WIDTH, key, sizeof(key));
    memcpy(state->values + state->length * KV_WIDTH, value, sizeof(value));
    state->length++;
    for (int head = 0; head < Q38FN_HEADS; ++head) {
        int kv_head = head / (Q38FN_HEADS / Q38FN_KV_HEADS);
        float max_score = -FLT_MAX, sum = 0.0f;
        for (size_t position = 0; position < state->length; ++position) {
            const float *cached_key = state->keys + position*KV_WIDTH + kv_head*Q38FN_HEAD_DIM;
            double dot = 0.0;
            for (int i = 0; i < Q38FN_HEAD_DIM; ++i)
                dot += (double)query[head*Q38FN_HEAD_DIM+i] * cached_key[i];
            scores[position] = (float)(dot / sqrtf(Q38FN_HEAD_DIM));
            if (scores[position] > max_score) max_score = scores[position];
        }
        for (size_t position = 0; position < state->length; ++position) {
            scores[position] = expf(scores[position] - max_score); sum += scores[position];
        }
        for (size_t position = 0; position < state->length; ++position) {
            const float *cached_value = state->values + position*KV_WIDTH + kv_head*Q38FN_HEAD_DIM;
            float probability = scores[position] / sum;
            for (int i = 0; i < Q38FN_HEAD_DIM; ++i)
                attended[head*Q38FN_HEAD_DIM+i] += probability * cached_value[i];
        }
    }
    for (int i = 0; i < QUERY_WIDTH; ++i) attended[i] *= q38fn_sigmoid(gate[i]);
    snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.o_proj.weight", layer);
    if (q38fn_bf16_matvec(ctx, name, attended, Q38FN_HIDDEN, QUERY_WIDTH, output) != 0) goto out;
    rc = 0;
out:
    return rc;
}

int q38fn_final_mix(const glm53f_st_context *ctx, const float *hyper_input,
                    float *output)
{
    enum { HC_WIDTH = Q38FN_HC_COUNT * Q38FN_HIDDEN, LOWRANK = 320 };
    const char *prefix = "model.language_model.hyper_connection_mixer";
    char name[160];
    static _Thread_local float normed_storage[HC_WIDTH],mix_storage[HC_WIDTH];
    float *normed=normed_storage,*mix=mix_storage;
    const uint16_t *weight;
    float lowrank[LOWRANK];
    int rc = -1;
    if (!ctx || !hyper_input || !output) return -1;
    snprintf(name, sizeof(name), "%s.hc_norm.weight", prefix);
    weight=q38fn_cached_bf16(ctx,name,0,HC_WIDTH);if(!weight)goto out;
    for (int stream=0; stream<Q38FN_HC_COUNT; ++stream) {
        int base=stream*Q38FN_HIDDEN; double squares=0.0;
        for(int i=0;i<Q38FN_HIDDEN;++i) squares+=(double)hyper_input[base+i]*hyper_input[base+i];
        float scale=1.0f/sqrtf((float)(squares/Q38FN_HIDDEN)+1.0e-6f);
        for(int i=0;i<Q38FN_HIDDEN;++i)normed[base+i]=hyper_input[base+i]*scale*
            (1.0f+q38fn_bf16_to_f32(weight[base+i]));
    }
    snprintf(name,sizeof(name),"%s.input_mix_weight_down.weight",prefix);
    if(q38fn_bf16_matvec(ctx,name,normed,LOWRANK,HC_WIDTH,lowrank)!=0) goto out;
    for(int i=0;i<LOWRANK;++i) lowrank[i]=q38fn_silu(lowrank[i]/Q38FN_HC_COUNT);
    snprintf(name,sizeof(name),"%s.input_mix_weight_up.weight",prefix);
    if(q38fn_bf16_matvec(ctx,name,lowrank,HC_WIDTH,LOWRANK,mix)!=0) goto out;
    for(int i=0;i<Q38FN_HIDDEN;++i){double sum=0.0;for(int s=0;s<Q38FN_HC_COUNT;++s){int j=s*Q38FN_HIDDEN+i;sum+=(double)q38fn_sigmoid(mix[j])*normed[j];}output[i]=(float)(sum/Q38FN_HC_COUNT);}
    rc=0;
out:return rc;
}

int q38fn_lm_head_argmax(const glm53f_st_context *ctx, const float *input,
                        int *token, float *logit)
{
    const uint16_t *weights;
    int best=-1; float best_value=-FLT_MAX;
    if(!ctx||!input||!token) return -1;
    weights=q38fn_cached_bf16(ctx,"lm_head.weight",0,
                              (size_t)Q38FN_VOCAB*Q38FN_HIDDEN);
    if(!weights) return -1;
#if defined(__ARM_FEATURE_SVE)
#ifdef _OPENMP
#pragma omp parallel
    {
            int local_best=-1; float local_value=-FLT_MAX;
#pragma omp for schedule(static)
            for(int row=0;row<Q38FN_VOCAB;row+=8){const uint16_t*w=weights+(size_t)row*Q38FN_HIDDEN;float values[8];matvec_bf16_8row(values,w,w+Q38FN_HIDDEN,w+2*Q38FN_HIDDEN,w+3*Q38FN_HIDDEN,w+4*Q38FN_HIDDEN,w+5*Q38FN_HIDDEN,w+6*Q38FN_HIDDEN,w+7*Q38FN_HIDDEN,input,Q38FN_HIDDEN);for(int lane=0;lane<8;++lane)if(values[lane]>local_value){local_value=values[lane];local_best=row+lane;}}
#pragma omp critical
            if(local_value>best_value){best_value=local_value;best=local_best;}
    }
#else
    for(int row=0;row<Q38FN_VOCAB;row+=8){const uint16_t*w=weights+(size_t)row*Q38FN_HIDDEN;float values[8];matvec_bf16_8row(values,w,w+Q38FN_HIDDEN,w+2*Q38FN_HIDDEN,w+3*Q38FN_HIDDEN,w+4*Q38FN_HIDDEN,w+5*Q38FN_HIDDEN,w+6*Q38FN_HIDDEN,w+7*Q38FN_HIDDEN,input,Q38FN_HIDDEN);for(int lane=0;lane<8;++lane)if(values[lane]>best_value){best_value=values[lane];best=row+lane;}}
#endif
#else
#ifdef _OPENMP
#pragma omp parallel
    {int local_best=-1;float local_value=-FLT_MAX;
#pragma omp for schedule(static)
    for(int row=0;row<Q38FN_VOCAB;++row){const uint16_t*w=weights+(size_t)row*Q38FN_HIDDEN;float sum=q38fn_bf16_dot(w,input,Q38FN_HIDDEN);if(sum>local_value){local_value=sum;local_best=row;}}
#pragma omp critical
    if(local_value>best_value){best_value=local_value;best=local_best;}}
#else
    for(int row=0;row<Q38FN_VOCAB;++row){const uint16_t*w=weights+(size_t)row*Q38FN_HIDDEN;float sum=q38fn_bf16_dot(w,input,Q38FN_HIDDEN);if(sum>best_value){best_value=sum;best=row;}}
#endif
#endif
    *token=best;if(logit)*logit=best_value;return 0;
}

int q38fn_attention_layer_step(const glm53f_st_context *ctx, int layer,
                               q38fn_attention_state *state, float *hyper_input,
                               int selected[Q38FN_ACTIVE_EXPERTS])
{
    float mixed[Q38FN_HIDDEN], block[Q38FN_HIDDEN], injection[Q38FN_HC_COUNT];
    if(!ctx||!state||!hyper_input||!q38fn_layer_is_full_attention((size_t)layer))return -1;
    if(q38fn_gated_residual_mix(ctx,layer,"attn",hyper_input,mixed,injection)!=0||
       q38fn_attention_step(ctx,layer,state,mixed,block)!=0)return -1;
    for(int s=0;s<Q38FN_HC_COUNT;++s)for(int i=0;i<Q38FN_HIDDEN;++i)hyper_input[s*Q38FN_HIDDEN+i]+=injection[s]*block[i];
    if(q38fn_gated_residual_mix(ctx,layer,"mlp",hyper_input,mixed,injection)!=0||
       q38fn_moe_step(ctx,layer,mixed,block,selected)!=0)return -1;
    for(int s=0;s<Q38FN_HC_COUNT;++s)for(int i=0;i<Q38FN_HIDDEN;++i)hyper_input[s*Q38FN_HIDDEN+i]+=injection[s]*block[i];
    return 0;
}

int q38fn_preload_tensor(const glm53f_st_context *ctx, const char *name)
{
    const st_tensor_info*t=glm53f_st_find(ctx,name,NULL);size_t count;
    if(!t||strcmp(t->dtype_str,"BF16")||t->nbytes%2)return -1;
    count=t->nbytes/2;
    return q38fn_cached_bf16(ctx,name,0,count)?0:-1;
}

int q38fn_preload_layer(const glm53f_st_context *ctx, int layer)
{
    char prefix[96];int n=snprintf(prefix,sizeof(prefix),"model.language_model.layers.%d.",layer);
    if(!ctx||n<0||(size_t)n>=sizeof(prefix))return -1;
    for(int i=0;i<ctx->n_entries;++i){const char*name=ctx->entries[i].name;if(!strncmp(name,prefix,(size_t)n)&&!strstr(name,"ngram_embedding.shard_")){const st_context*owner=ctx->shards[ctx->entries[i].shard].st;const st_tensor_info*t=&owner->tensors[ctx->entries[i].tensor];if(!strcmp(t->dtype_str,"BF16")&&q38fn_preload_tensor(ctx,name)!=0)return -1;}}
    return 0;
}

int q38fn_preload_ngram_owner(const glm53f_st_context *ctx, int rank, int ranks)
{
    if(!ctx||ranks!=12||rank<0||rank>=ranks)return -1;
    for(int i=0;i<ctx->n_entries;++i){const char*name=ctx->entries[i].name;const char*p=strstr(name,"ngram_embedding.shard_");int shard=-1;if(p&&sscanf(p,"ngram_embedding.shard_%d.weight",&shard)==1&&q38fn_ngram_owner(shard)==rank){
        const st_tensor_info*t=glm53f_st_find(ctx,name,NULL);q38fn_ngram_q8_entry*entry;
        if(!t||strcmp(t->dtype_str,"BF16")||t->n_dims!=2||t->shape[1]!=Q38FN_NGRAM_HEAD_DIM)return -1;
        entry=(q38fn_ngram_q8_entry*)calloc(1,sizeof(*entry));if(!entry)return -1;
        entry->rows=(size_t)t->shape[0];entry->columns=(size_t)t->shape[1];entry->name=strdup(name);
        entry->data=(int8_t*)malloc(entry->rows*entry->columns);entry->scale=(float*)malloc(entry->rows*sizeof(*entry->scale));
        enum{CHUNK_ROWS=4096};uint16_t*temporary=(uint16_t*)malloc((size_t)CHUNK_ROWS*entry->columns*sizeof(*temporary));
        if(!entry->name||!entry->data||!entry->scale||!temporary){free(temporary);free(entry->scale);free(entry->data);free(entry->name);free(entry);return -1;}
        for(size_t first=0;first<entry->rows;first+=CHUNK_ROWS){size_t rows=entry->rows-first<CHUNK_ROWS?entry->rows-first:CHUNK_ROWS,count=rows*entry->columns;if(glm53f_st_read(ctx,name,first*entry->columns*2,temporary,count*2)){free(temporary);free(entry->scale);free(entry->data);free(entry->name);free(entry);return -1;}for(size_t r=0;r<rows;++r){float maximum=0.0f;for(size_t c=0;c<entry->columns;++c){float value=fabsf(q38fn_bf16_to_f32(temporary[r*entry->columns+c]));if(value>maximum)maximum=value;}float scale=maximum>0.0f?maximum/127.0f:1.0f;entry->scale[first+r]=scale;for(size_t c=0;c<entry->columns;++c){long q=lrintf(q38fn_bf16_to_f32(temporary[r*entry->columns+c])/scale);if(q>127)q=127;if(q< -127)q= -127;entry->data[(first+r)*entry->columns+c]=(int8_t)q;}}}
        free(temporary);entry->next=q38fn_ngram_q8_cache;q38fn_ngram_q8_cache=entry;q38fn_weight_cache_bytes+=entry->rows*(entry->columns+sizeof(*entry->scale));
    }}
    return 0;
}

#endif
#endif
