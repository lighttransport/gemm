#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../common/glm53f_safetensors.h"
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include "../a64fx/k3/k3_moe.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define Q38FN_LOWBIT_IMPLEMENTATION
#include "../common/q38fn_lowbit.h"

static double now_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1.0e-9;
}

static float bf16_value(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void reference_matvec(float *output, const uint16_t *weight,
                             const float *input, int rows, int columns)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row = 0; row < rows; ++row) {
        const uint16_t *source = weight + (size_t)row * columns;
        float sum = 0.0f;
#ifdef _OPENMP
#pragma omp simd reduction(+:sum)
#endif
        for (int column = 0; column < columns; ++column)
            sum += bf16_value(source[column]) * input[column];
        output[row] = sum;
    }
}

static void evict_cache(float *buffer, size_t count)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < count; ++i) buffer[i] += 1.0f;
}

int main(int argc, char **argv)
{
    const char *tensor_name = "model.language_model.embed_tokens.weight";
    int rows = 32768, repetitions = 12, threads = 48;
    glm53f_st_context *context = NULL;
    const st_tensor_info *tensor;
    uint16_t *weight = NULL;
    uint16_t *bf16_pv = NULL;
    uint8_t *packed = NULL, *scale = NULL;
    q38fn_q5_block *q5 = NULL;
    int8_t *q8 = NULL, *q8_input = NULL;
    float *q8_scale = NULL;
    float *input = NULL, *reference = NULL, *output = NULL, *eviction = NULL;
    int result = 1;

    if (argc < 3 || argc > 7) {
        fprintf(stderr, "usage: %s MODEL_DIR PAYLOAD_DIR [TENSOR [ROWS [REPS [THREADS]]]]\n", argv[0]);
        return 2;
    }
    if (argc > 3) tensor_name = argv[3];
    if (argc > 4) rows = atoi(argv[4]);
    if (argc > 5) repetitions = atoi(argv[5]);
    if (argc > 6) threads = atoi(argv[6]);
    if (rows < 8 || repetitions < 1 || threads < 1) return 2;
    rows -= rows % 16;
    setenv("GLM53F_ST_PAYLOAD_DIR", argv[2], 1);
    context = glm53f_st_open(argv[1]);
    tensor = context ? glm53f_st_find(context, tensor_name, NULL) : NULL;
    uint64_t available_rows = 1, columns64 = 0;
    if (tensor && (tensor->n_dims == 2 || tensor->n_dims == 3)) {
        columns64 = tensor->shape[tensor->n_dims - 1];
        for (int dimension = 0; dimension + 1 < tensor->n_dims; ++dimension)
            available_rows *= tensor->shape[dimension];
    }
    if (!tensor || strcmp(tensor->dtype_str, "BF16") ||
        (tensor->n_dims != 2 && tensor->n_dims != 3) ||
        available_rows < (uint64_t)rows || !columns64 ||
        columns64 > INT_MAX || columns64 % 32) {
        fprintf(stderr, "invalid BF16 matrix or row request: %s\n", tensor_name);
        goto done;
    }
    int columns = (int)columns64;
    size_t elements = (size_t)rows * columns;
    size_t packed_bytes = elements / 2, scale_bytes = elements / 32;
    weight = aligned_alloc(256, elements * sizeof(*weight));
    bf16_pv = aligned_alloc(256, elements * sizeof(*bf16_pv));
    packed = aligned_alloc(256, packed_bytes);
    scale = aligned_alloc(256, scale_bytes);
    size_t q5_bytes = q38fn_q5_bytes((size_t)rows, (size_t)columns);
    q5 = aligned_alloc(256, (q5_bytes + 255) & ~(size_t)255);
    size_t q8_bytes = k3_q8_matrix_bytes(rows, columns);
    size_t q8pv_bytes = columns % 64 ? 0 : k3_q8pv_matrix_bytes(rows, columns);
    size_t q8pv32_bytes = k3_q8pv32_matrix_bytes(rows, columns);
    size_t q8_alloc = q8pv_bytes > elements ? q8pv_bytes : elements;
    if (q8pv32_bytes > q8_alloc) q8_alloc = q8pv32_bytes;
    q8 = aligned_alloc(256, (q8_alloc + 255) & ~(size_t)255);
    q8_scale = aligned_alloc(256, ((size_t)rows * sizeof(*q8_scale) + 255) & ~(size_t)255);
    q8_input = aligned_alloc(256, ((size_t)columns + 255) & ~(size_t)255);
    input = aligned_alloc(256, (size_t)columns * sizeof(*input));
    reference = aligned_alloc(256, (size_t)rows * sizeof(*reference));
    output = aligned_alloc(256, (size_t)rows * sizeof(*output));
    size_t eviction_count = (size_t)256 << 20;
    eviction_count /= sizeof(*eviction);
    eviction = aligned_alloc(256, eviction_count * sizeof(*eviction));
    if (!weight || !bf16_pv || !packed || !scale || !q5 || !q8 || !q8_scale || !q8_input ||
        !input || !reference || !output ||
        !eviction) goto done;
    memset(eviction, 0, eviction_count * sizeof(*eviction));
    if (glm53f_st_read(context, tensor_name, 0, weight,
                       elements * sizeof(*weight))) goto done;
    for (int i = 0; i < columns; ++i)
        input[i] = (float)(((i * 1103515245u + 12345u) >> 16) & 1023) / 4096.0f - 0.125f;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    reference_matvec(reference, weight, input, rows, columns);
    k3_pack_bf16_pv(bf16_pv,weight,rows,columns);
    k3_matvec_bf16_pv(output,bf16_pv,rows,columns,input,threads);
    double bf_error2=0.0,bf_ref2=0.0;
    for(int row=0;row<rows;++row){double d=(double)output[row]-reference[row];bf_error2+=d*d;bf_ref2+=(double)reference[row]*reference[row];}
    double bf_elapsed=0.0;
    for(int repetition=0;repetition<repetitions;++repetition){evict_cache(eviction,eviction_count);double start=now_seconds();k3_matvec_bf16_pv(output,bf16_pv,rows,columns,input,threads);bf_elapsed+=now_seconds()-start;}
    double bf_mean=bf_elapsed/repetitions;
    printf("Q38FN_LOWBITS_BF16PV tensor=%s rows=%d cols=%d stored_bytes=%zu mean_ms=%.6f stored_GB_s=%.6f rel_l2=%.9g\n",
           tensor_name,rows,columns,elements*2,bf_mean*1e3,elements*2/bf_mean/1e9,
           sqrt(bf_error2/(bf_ref2+1e-30)));
    k3_mxfp4_quantize_bf16(packed, scale, weight, rows, columns);
    k3_mxfp4_matrix matrix = {packed, scale, rows, columns};
    k3_mxfp4_gemm_mode(output, &matrix, input, 1, threads, 0);
    double error2 = 0.0, reference2 = 0.0, output2 = 0.0, dot = 0.0;
    float maximum = 0.0f;
    for (int row = 0; row < rows; ++row) {
        double difference = (double)output[row] - reference[row];
        error2 += difference * difference;
        reference2 += (double)reference[row] * reference[row];
        output2 += (double)output[row] * output[row];
        dot += (double)output[row] * reference[row];
        if (fabsf((float)difference) > maximum) maximum = fabsf((float)difference);
    }
    double elapsed = 0.0;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        evict_cache(eviction, eviction_count);
        double start = now_seconds();
        k3_mxfp4_gemm_mode(output, &matrix, input, 1, threads, 0);
        elapsed += now_seconds() - start;
    }
    double mean = elapsed / repetitions;
    double relative_l2 = sqrt(error2 / (reference2 + 1.0e-30));
    double cosine = dot / sqrt((reference2 + 1.0e-30) * (output2 + 1.0e-30));
    printf("Q38FN_LOWBIT tensor=%s rows=%d cols=%d threads=%d stored_bytes=%zu ",
           tensor_name, rows, columns, threads, packed_bytes + scale_bytes);
    printf("mean_ms=%.6f stored_GB_s=%.6f bf16eq_GB_s=%.6f ",
           mean * 1.0e3, (packed_bytes + scale_bytes) / mean / 1.0e9,
           (2.0 * elements) / mean / 1.0e9);
    printf("rel_l2=%.9g cosine=%.9g max_abs=%.9g\n",
           relative_l2, cosine, maximum);
    q38fn_q5_quantize_bf16(q5, weight, (size_t)rows, (size_t)columns);
    q38fn_q5_matvec(output, q5, input, (size_t)rows, (size_t)columns);
    error2 = output2 = dot = 0.0; maximum = 0.0f;
    for (int row = 0; row < rows; ++row) {
        double difference = (double)output[row] - reference[row];
        error2 += difference * difference; output2 += (double)output[row] * output[row];
        dot += (double)output[row] * reference[row];
        if (fabsf((float)difference) > maximum) maximum = fabsf((float)difference);
    }
    relative_l2 = sqrt(error2 / (reference2 + 1.0e-30));
    cosine = dot / sqrt((reference2 + 1.0e-30) * (output2 + 1.0e-30));
    printf("Q38FN_LOWBIT_Q5 tensor=%s rows=%d cols=%d stored_bytes=%zu ",
           tensor_name, rows, columns, q5_bytes);
    elapsed = 0.0;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        evict_cache(eviction, eviction_count); double start = now_seconds();
        q38fn_q5_matvec(output, q5, input, (size_t)rows, (size_t)columns);
        elapsed += now_seconds() - start;
    }
    mean = elapsed / repetitions;
    printf("mean_ms=%.6f stored_GB_s=%.6f bf16eq_GB_s=%.6f ",
           mean * 1.0e3, q5_bytes / mean / 1.0e9,
           2.0 * elements / mean / 1.0e9);
    printf("rel_l2=%.9g cosine=%.9g max_abs=%.9g\n", relative_l2, cosine, maximum);
    if (columns % 64) goto skip_q8;
    k3_q8pv_quantize_bf16((uint8_t *)q8, weight, rows, columns);
    k3_q8pv_matrix q8pv_matrix = {(const uint8_t *)q8, rows, columns};
    k3_matvec_q8pv(output,&q8pv_matrix,input,q8_input,q8_scale,threads);
    error2 = output2 = dot = 0.0; maximum = 0.0f;
    for (int row = 0; row < rows; ++row) {
        double difference = (double)output[row] - reference[row];
        error2 += difference * difference; output2 += (double)output[row] * output[row];
        dot += (double)output[row] * reference[row];
        if (fabsf((float)difference) > maximum) maximum = fabsf((float)difference);
    }
    relative_l2 = sqrt(error2 / (reference2 + 1.0e-30));
    cosine = dot / sqrt((reference2 + 1.0e-30) * (output2 + 1.0e-30));
    elapsed = 0.0;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        evict_cache(eviction, eviction_count); double start = now_seconds();
        k3_matvec_q8pv(output,&q8pv_matrix,input,q8_input,q8_scale,threads);
        elapsed += now_seconds() - start;
    }
    mean = elapsed / repetitions;
    printf("Q38FN_LOWBIT_Q8PV tensor=%s rows=%d cols=%d stored_bytes=%zu ",
           tensor_name, rows, columns, q8pv_bytes);
    printf("mean_ms=%.6f stored_GB_s=%.6f bf16eq_GB_s=%.6f ",
           mean * 1.0e3, q8pv_bytes / mean / 1.0e9,
           2.0 * elements / mean / 1.0e9);
    printf("rel_l2=%.9g cosine=%.9g max_abs=%.9g\n",relative_l2,cosine,maximum);
    k3_q8pv32_quantize_bf16((uint8_t*)q8,weight,rows,columns);
    k3_q8pv_matrix q8pv32_matrix={(const uint8_t*)q8,rows,columns};
    k3_matvec_q8pv32(output,&q8pv32_matrix,input,q8_input,q8_scale,threads);
    error2=output2=dot=0.0;maximum=0.0f;
    for(int row=0;row<rows;++row){double difference=(double)output[row]-reference[row];error2+=difference*difference;output2+=(double)output[row]*output[row];dot+=(double)output[row]*reference[row];if(fabsf((float)difference)>maximum)maximum=fabsf((float)difference);}
    relative_l2=sqrt(error2/(reference2+1e-30));cosine=dot/sqrt((reference2+1e-30)*(output2+1e-30));elapsed=0.0;
    for(int repetition=0;repetition<repetitions;++repetition){evict_cache(eviction,eviction_count);double start=now_seconds();k3_matvec_q8pv32(output,&q8pv32_matrix,input,q8_input,q8_scale,threads);elapsed+=now_seconds()-start;}
    mean=elapsed/repetitions;
    printf("Q38FN_LOWBIT_Q8PV32 tensor=%s rows=%d cols=%d stored_bytes=%zu mean_ms=%.6f stored_GB_s=%.6f bf16eq_GB_s=%.6f rel_l2=%.9g cosine=%.9g max_abs=%.9g\n",tensor_name,rows,columns,q8pv32_bytes,mean*1e3,q8pv32_bytes/mean/1e9,2.0*elements/mean/1e9,relative_l2,cosine,maximum);
    k3_matvec_q8pv32_f32(output,&q8pv32_matrix,input,threads);
    error2=output2=dot=0.0;maximum=0.0f;
    for(int row=0;row<rows;++row){double difference=(double)output[row]-reference[row];error2+=difference*difference;output2+=(double)output[row]*output[row];dot+=(double)output[row]*reference[row];if(fabsf((float)difference)>maximum)maximum=fabsf((float)difference);}
    relative_l2=sqrt(error2/(reference2+1e-30));cosine=dot/sqrt((reference2+1e-30)*(output2+1e-30));elapsed=0.0;
    for(int repetition=0;repetition<repetitions;++repetition){evict_cache(eviction,eviction_count);double start=now_seconds();k3_matvec_q8pv32_f32(output,&q8pv32_matrix,input,threads);elapsed+=now_seconds()-start;}
    mean=elapsed/repetitions;
    printf("Q38FN_LOWBIT_Q8PV32_F32 tensor=%s rows=%d cols=%d stored_bytes=%zu mean_ms=%.6f stored_GB_s=%.6f bf16eq_GB_s=%.6f rel_l2=%.9g cosine=%.9g max_abs=%.9g\n",tensor_name,rows,columns,q8pv32_bytes,mean*1e3,q8pv32_bytes/mean/1e9,2.0*elements/mean/1e9,relative_l2,cosine,maximum);
    k3_q8p16_quantize_bf16(q8, q8_scale, weight, rows, columns);
    k3_matvec_q8p16(output, q8, q8_scale, rows, columns, input, q8_input, threads);
    error2 = output2 = dot = 0.0; maximum = 0.0f;
    for (int row = 0; row < rows; ++row) {
        double difference = (double)output[row] - reference[row];
        error2 += difference * difference; output2 += (double)output[row] * output[row];
        dot += (double)output[row] * reference[row];
        if (fabsf((float)difference) > maximum) maximum = fabsf((float)difference);
    }
    relative_l2 = sqrt(error2 / (reference2 + 1.0e-30));
    cosine = dot / sqrt((reference2 + 1.0e-30) * (output2 + 1.0e-30));
    elapsed = 0.0;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        evict_cache(eviction, eviction_count); double start = now_seconds();
        k3_matvec_q8p16(output,q8,q8_scale,rows,columns,input,q8_input,threads);
        elapsed += now_seconds() - start;
    }
    mean = elapsed / repetitions;
    printf("Q38FN_LOWBIT_Q8P16 tensor=%s rows=%d cols=%d stored_bytes=%zu ",
           tensor_name, rows, columns, q8_bytes);
    printf("mean_ms=%.6f stored_GB_s=%.6f bf16eq_GB_s=%.6f ",
           mean * 1.0e3, q8_bytes / mean / 1.0e9,
           2.0 * elements / mean / 1.0e9);
    printf("rel_l2=%.9g cosine=%.9g max_abs=%.9g\n",relative_l2,cosine,maximum);
    k3_q8_quantize_bf16_rows(q8,q8_scale,weight,rows,columns);
    k3_q8_matrix q8_matrix={q8,q8_scale,rows,columns};
    k3_matvec_q8_8(output,&q8_matrix,input,q8_input,threads);
    error2=output2=dot=0.0;maximum=0.0f;
    for(int row=0;row<rows;++row){double difference=(double)output[row]-reference[row];error2+=difference*difference;output2+=(double)output[row]*output[row];dot+=(double)output[row]*reference[row];if(fabsf((float)difference)>maximum)maximum=fabsf((float)difference);}
    relative_l2=sqrt(error2/(reference2+1e-30));cosine=dot/sqrt((reference2+1e-30)*(output2+1e-30));elapsed=0.0;
    for(int repetition=0;repetition<repetitions;++repetition){evict_cache(eviction,eviction_count);double start=now_seconds();k3_matvec_q8_8(output,&q8_matrix,input,q8_input,threads);elapsed+=now_seconds()-start;}
    mean=elapsed/repetitions;
    printf("Q38FN_LOWBIT_Q8ROW8 tensor=%s rows=%d cols=%d stored_bytes=%zu mean_ms=%.6f stored_GB_s=%.6f bf16eq_GB_s=%.6f rel_l2=%.9g cosine=%.9g max_abs=%.9g\n",tensor_name,rows,columns,q8_bytes,mean*1e3,q8_bytes/mean/1e9,2.0*elements/mean/1e9,relative_l2,cosine,maximum);
    k3_q8p8_quantize_bf16(q8,q8_scale,weight,rows,columns);
    k3_matvec_q8p8(output,q8,q8_scale,rows,columns,input,q8_input,threads);
    error2=output2=dot=0.0;maximum=0.0f;
    for(int row=0;row<rows;++row){double difference=(double)output[row]-reference[row];error2+=difference*difference;output2+=(double)output[row]*output[row];dot+=(double)output[row]*reference[row];if(fabsf((float)difference)>maximum)maximum=fabsf((float)difference);}
    relative_l2=sqrt(error2/(reference2+1e-30));cosine=dot/sqrt((reference2+1e-30)*(output2+1e-30));elapsed=0.0;
    for(int repetition=0;repetition<repetitions;++repetition){evict_cache(eviction,eviction_count);double start=now_seconds();k3_matvec_q8p8(output,q8,q8_scale,rows,columns,input,q8_input,threads);elapsed+=now_seconds()-start;}
    mean=elapsed/repetitions;
    printf("Q38FN_LOWBIT_Q8P8 tensor=%s rows=%d cols=%d stored_bytes=%zu mean_ms=%.6f stored_GB_s=%.6f bf16eq_GB_s=%.6f rel_l2=%.9g cosine=%.9g max_abs=%.9g\n",tensor_name,rows,columns,q8_bytes,mean*1e3,q8_bytes/mean/1e9,2.0*elements/mean/1e9,relative_l2,cosine,maximum);
    k3_q8_quantize_bf16_rows(q8,q8_scale,weight,rows,columns);
    k3_matvec_q8_4(output,&q8_matrix,input,q8_input,threads);
    elapsed=0.0;
    for(int repetition=0;repetition<repetitions;++repetition){evict_cache(eviction,eviction_count);double start=now_seconds();k3_matvec_q8_4(output,&q8_matrix,input,q8_input,threads);elapsed+=now_seconds()-start;}
    mean=elapsed/repetitions;
    printf("Q38FN_LOWBIT_Q8ROW4 tensor=%s rows=%d cols=%d stored_bytes=%zu mean_ms=%.6f stored_GB_s=%.6f bf16eq_GB_s=%.6f\n",tensor_name,rows,columns,q8_bytes,mean*1e3,q8_bytes/mean/1e9,2.0*elements/mean/1e9);
skip_q8:
    result = (!isfinite(relative_l2) || !isfinite(cosine) || cosine < 0.99) ? 1 : 0;
done:
    free(eviction); free(output); free(reference); free(input);
    free(q8_input); free(q8_scale); free(q8); free(q5); free(scale); free(packed);
    free(bf16_pv); free(weight); glm53f_st_close(context);
    return result;
}
