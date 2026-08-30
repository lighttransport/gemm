#define _GNU_SOURCE
#include <mpi.h>
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../common/glm53f_safetensors.h"
#include "../common/q38fn_tp_layout.h"
#define Q38FN_LOWBIT_IMPLEMENTATION
#include "../common/q38fn_lowbit.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { IO_BYTES = 64 << 20 };

static uint64_t fnv1a(uint64_t h, const void *data, size_t bytes)
{
    const unsigned char *p = (const unsigned char *)data;
    while (bytes--) { h ^= *p++; h *= UINT64_C(1099511628211); }
    return h;
}

static int write_all(int fd, const void *data, size_t bytes, uint64_t *hash)
{
    const unsigned char *p = (const unsigned char *)data;
    if (hash) *hash = fnv1a(*hash, data, bytes);
    while (bytes) {
        size_t part = bytes > (size_t)(1u << 30) ? (size_t)(1u << 30) : bytes;
        ssize_t n = write(fd, p, part);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += (size_t)n; bytes -= (size_t)n;
    }
    return 0;
}

static int copy_range(const glm53f_st_context *ctx, const char *name,
                      uint64_t element_offset, uint64_t elements, int out,
                      void *buffer, uint64_t *hash)
{
    uint64_t done = 0;
    while (done < elements) {
        uint64_t count = elements - done;
        if (count > IO_BYTES / sizeof(uint16_t)) count = IO_BYTES / sizeof(uint16_t);
        size_t bytes = (size_t)count * sizeof(uint16_t);
        if (glm53f_st_read(ctx, name, (size_t)(element_offset + done) * sizeof(uint16_t),
                           buffer, bytes) || write_all(out, buffer, bytes, hash)) return -1;
        done += count;
    }
    return 0;
}

static int copy_range_q5(const glm53f_st_context *ctx, const char *name,
                         uint64_t row_offset, uint64_t rows, uint64_t columns,
                         int out, uint16_t *buffer, q38fn_q5_block *packed,
                         uint64_t *written, uint64_t *hash)
{
    uint64_t rows_per_chunk = (IO_BYTES / sizeof(uint16_t)) / columns;
    if (!rows_per_chunk || columns % 32) return -1;
    for (uint64_t row = 0; row < rows; row += rows_per_chunk) {
        uint64_t count = rows - row;
        if (count > rows_per_chunk) count = rows_per_chunk;
        size_t input_bytes = (size_t)(count * columns) * sizeof(uint16_t);
        if (glm53f_st_read(ctx, name,
                          (size_t)((row_offset + row) * columns) * sizeof(uint16_t),
                          buffer, input_bytes) ||
            q38fn_q5_quantize_bf16(packed, buffer, (size_t)count, (size_t)columns))
            return -1;
        size_t output_bytes = q38fn_q5_bytes((size_t)count, (size_t)columns);
        if (write_all(out, packed, output_bytes, hash)) return -1;
        *written += output_bytes;
    }
    return 0;
}

static int copy_range_q8(const glm53f_st_context *ctx, const char *name,
                         uint64_t row_offset, uint64_t rows, uint64_t columns,
                         int out, uint16_t *source, uint64_t *written,
                         uint64_t *hash)
{
    size_t elements = (size_t)rows * (size_t)columns;
    if (!rows || !columns || elements / columns != rows) return -1;
    int8_t *weights = malloc(elements);
    float *scales = malloc((size_t)rows * sizeof(*scales));
    if (!weights || !scales) { free(weights); free(scales); return -1; }
    if (glm53f_st_read(ctx, name, (size_t)(row_offset * columns) * 2,
                       source, elements * 2) ||
        q38fn_q8_quantize_bf16(weights, scales, source, (size_t)rows,
                               (size_t)columns) || write_all(out, weights,
                               elements, hash) || write_all(out, scales,
                               (size_t)rows * sizeof(*scales), hash)) {
        free(weights); free(scales); return -1;
    }
    *written += q38fn_q8_bytes((size_t)rows, (size_t)columns);
    free(weights); free(scales); return 0;
}

static int copy_expert_q8(const glm53f_st_context *ctx, const char *name,
                          uint64_t expert, const q38fn_tp_plan *plan,
                          uint64_t source_rows, uint64_t columns, int out,
                          uint16_t *source, uint64_t *written, uint64_t *hash)
{
    uint64_t local_rows=0;
    for(int r=0;r<plan->n_ranges;r++) local_rows+=plan->range[r].count;
    size_t elements=(size_t)local_rows*(size_t)columns;
    int8_t *weights=malloc(elements);float *scales=malloc((size_t)local_rows*sizeof(*scales));
    if(!weights||!scales){free(weights);free(scales);return -1;}
    uint64_t dst_row=0;
    for(int r=0;r<plan->n_ranges;r++){
        uint64_t rows=plan->range[r].count;
        size_t bytes=(size_t)rows*(size_t)columns*2;
        uint64_t row_offset=expert*source_rows+plan->range[r].start;
        if(glm53f_st_read(ctx,name,(size_t)(row_offset*columns)*2,source,bytes)||
           q38fn_q8_quantize_bf16(weights+dst_row*columns,scales+dst_row,source,(size_t)rows,(size_t)columns)){
            free(weights);free(scales);return -1;
        }
        dst_row+=rows;
    }
    if(write_all(out,weights,elements,hash)||write_all(out,scales,(size_t)local_rows*4,hash)){
        free(weights);free(scales);return -1;
    }
    *written+=q38fn_q8_bytes((size_t)local_rows,(size_t)columns);
    free(weights);free(scales);return 0;
}

static int copy_all_experts_q8(const glm53f_st_context *ctx,const char*name,
                               uint64_t experts,const q38fn_tp_plan*plan,
                               uint64_t source_rows,uint64_t columns,int out,
                               uint64_t*written,uint64_t*hash)
{
    uint64_t local_rows=0;
    for(int r=0;r<plan->n_ranges;r++)local_rows+=plan->range[r].count;
    size_t weights_bytes=(size_t)local_rows*(size_t)columns;
    size_t block_bytes=weights_bytes+(size_t)local_rows*sizeof(float);
    if(experts&&block_bytes>SIZE_MAX/(size_t)experts)return-1;
    unsigned char*blocks=malloc(block_bytes*(size_t)experts);
    if(!blocks)return-1;
    int failed=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1) reduction(|:failed)
#endif
    for(uint64_t expert=0;expert<experts;expert++){
        size_t max_rows=0;
        for(int r=0;r<plan->n_ranges;r++)if(plan->range[r].count>max_rows)max_rows=(size_t)plan->range[r].count;
        uint16_t*source=malloc(max_rows*(size_t)columns*sizeof(*source));
        int8_t*weights=(int8_t*)(blocks+(size_t)expert*block_bytes);
        float*scales=(float*)(blocks+(size_t)expert*block_bytes+weights_bytes);
        uint64_t dst_row=0;
        if(!source){failed|=1;continue;}
        for(int r=0;r<plan->n_ranges&&!failed;r++){
            uint64_t rows=plan->range[r].count;
            size_t bytes=(size_t)rows*(size_t)columns*sizeof(*source);
            uint64_t row_offset=expert*source_rows+plan->range[r].start;
            if(glm53f_st_read(ctx,name,(size_t)(row_offset*columns)*sizeof(*source),source,bytes)||
               q38fn_q8_quantize_bf16(weights+dst_row*columns,scales+dst_row,source,(size_t)rows,(size_t)columns))failed|=1;
            dst_row+=rows;
        }
        free(source);
    }
    if(failed||write_all(out,blocks,block_bytes*(size_t)experts,hash)){free(blocks);return-1;}
    *written+=block_bytes*(size_t)experts;free(blocks);return 0;
}

static int copy_plan(const glm53f_st_context *ctx, const char *name,
                     const st_tensor_info *tensor, const q38fn_tp_plan *plan,
                     int out, void *buffer, q38fn_q5_block *packed,
                     int all_q5, int all_q8, uint64_t *written, uint64_t *hash)
{
    uint64_t total = 1, before = *written;
    for (int d = 0; d < tensor->n_dims; ++d) {
        if (tensor->shape[d] > UINT64_MAX / total) return -1;
        total *= tensor->shape[d];
    }
    if (plan->kind == Q38FN_TP_NGRAM_OWNER ||
        (all_q5 && tensor->n_dims >= 2 &&
         plan->kind == Q38FN_TP_FULL &&
         tensor->shape[tensor->n_dims - 1] % 32 == 0)) {
        uint64_t columns = tensor->shape[tensor->n_dims - 1];
        return copy_range_q5(ctx, name, 0, total / columns, columns, out,
                             (uint16_t *)buffer, packed, written, hash);
    }
    if (plan->kind == Q38FN_TP_FULL) {
        if (copy_range(ctx, name, 0, total, out, buffer, hash)) return -1;
        *written += total * sizeof(uint16_t); return 0;
    }
    if (plan->kind == Q38FN_TP_AXIS0) {
        uint64_t stride = 1;
        for (int d = 1; d < tensor->n_dims; ++d) stride *= tensor->shape[d];
        uint64_t count = plan->range[0].count * stride;
        if (all_q5 && tensor->n_dims >= 2 && stride % 32 == 0)
            return copy_range_q5(ctx, name, plan->range[0].start,
                                 plan->range[0].count, stride, out,
                                 (uint16_t *)buffer, packed, written, hash);
        if (copy_range(ctx, name, plan->range[0].start * stride, count,
                       out, buffer, hash)) return -1;
        *written += count * sizeof(uint16_t); return 0;
    }
    if (plan->kind == Q38FN_TP_EXPERT_GATE_UP) {
        uint64_t rows = tensor->shape[1], cols = tensor->shape[2];
        if(all_q8&&cols%64==0)
            return copy_all_experts_q8(ctx,name,tensor->shape[0],plan,rows,cols,
                                       out,written,hash);
        for (uint64_t expert = 0; expert < tensor->shape[0]; ++expert)
            for (int r = 0; r < plan->n_ranges; ++r) {
                uint64_t count = plan->range[r].count * cols;
                uint64_t offset = (expert * rows + plan->range[r].start) * cols;
                if (all_q8 && r == 0 && cols % 64 == 0) {
                    if (copy_expert_q8(ctx,name,expert,plan,rows,cols,out,
                                       (uint16_t*)buffer,written,hash)) return -1;
                    break;
                }
                if (all_q5 && cols % 32 == 0) {
                    if (copy_range_q5(ctx,name,expert*rows+plan->range[r].start,
                                      plan->range[r].count,cols,out,
                                      (uint16_t*)buffer,packed,written,hash))return -1;
                    continue;
                }
                if (copy_range(ctx, name, offset, count, out, buffer, hash)) return -1;
                *written += count * sizeof(uint16_t);
            }
        return 0;
    }
    if (plan->kind == Q38FN_TP_DELTA_QKV) {
        uint64_t cols = tensor->shape[1];
        for (int r = 0; r < plan->n_ranges; ++r) {
            uint64_t count = plan->range[r].count * cols;
            if (all_q5 && cols % 32 == 0) {
                if (copy_range_q5(ctx,name,plan->range[r].start,
                                  plan->range[r].count,cols,out,
                                  (uint16_t*)buffer,packed,written,hash))return -1;
                continue;
            }
            if (copy_range(ctx, name, plan->range[r].start * cols, count,
                           out, buffer, hash)) return -1;
            *written += count * sizeof(uint16_t);
        }
        return 0;
    }
    if (plan->kind == Q38FN_TP_AXIS1) {
        int axis = plan->axis;
        uint64_t outer = 1, width = tensor->shape[axis], rows_per_chunk;
        for (int d = 0; d < axis; ++d) outer *= tensor->shape[d];
        for (int d = axis + 1; d < tensor->n_dims; ++d)
            if (tensor->shape[d] != 1) return -1;
        rows_per_chunk = (IO_BYTES / sizeof(uint16_t)) / width;
        if (!rows_per_chunk) rows_per_chunk = 1;
        uint16_t *source = (uint16_t *)buffer;
        uint64_t local = plan->range[0].count;
        for (uint64_t row0 = 0; row0 < outer; row0 += rows_per_chunk) {
            uint64_t rows = outer - row0;
            if (rows > rows_per_chunk) rows = rows_per_chunk;
            size_t source_bytes = (size_t)(rows * width) * sizeof(uint16_t);
            if (glm53f_st_read(ctx, name, (size_t)(row0 * width) * sizeof(uint16_t),
                               source, source_bytes)) return -1;
            if (all_q5 && local % 32 == 0) {
                for (uint64_t row = 0; row < rows; ++row)
                    memmove(source + row * local,
                            source + row * width + plan->range[0].start,
                            (size_t)local * sizeof(uint16_t));
                if (q38fn_q5_quantize_bf16(packed,source,(size_t)rows,(size_t)local))return -1;
                size_t bytes=q38fn_q5_bytes((size_t)rows,(size_t)local);
                if(write_all(out,packed,bytes,hash))return -1;
                *written+=bytes;
                continue;
            }
            for (uint64_t row = 0; row < rows; ++row) {
                const uint16_t *slice = source + row * width + plan->range[0].start;
                size_t bytes = (size_t)local * sizeof(uint16_t);
                if (write_all(out, slice, bytes, hash)) return -1;
                *written += bytes;
            }
        }
        return 0;
    }
    return *written == before ? -1 : 0;
}

static int mkdir_p(const char *path)
{
    char *copy = strdup(path); if (!copy) return -1;
    for (char *p = copy + 1; *p; ++p) if (*p == '/') {
        *p = 0; if (mkdir(copy, 0700) && errno != EEXIST) { free(copy); return -1; } *p = '/';
    }
    int rc = (!mkdir(copy, 0700) || errno == EEXIST) ? 0 : -1; free(copy); return rc;
}

static int tensor_layer(const char *name)
{
    const char *p = strstr(name, "model.language_model.layers.");
    int layer = -1;
    return p && sscanf(p, "model.language_model.layers.%d.", &layer) == 1 ? layer : -1;
}

static int complete_blob(const char *blob, const char *manifest, int rank, int ranks,
                         int layer_limit)
{
    struct stat st; FILE *f; char line[512], last[512] = {0};
    int vr = -1, rr = -1, nn = -1, ll = -1;
    unsigned long long bytes = 0;
    if (stat(blob, &st) || !(f = fopen(manifest, "r"))) return 0;
    if (!fgets(line, sizeof(line), f) ||
        sscanf(line, "# Q38FNTP layout=%d rank=%d ranks=%d layers=%d", &vr, &rr, &nn, &ll) != 4) {
        fclose(f); return 0;
    }
    while (fgets(line, sizeof(line), f)) memcpy(last, line, sizeof(last));
    fclose(f);
    return vr == Q38FN_TP_LAYOUT_VERSION && rr == rank && nn == ranks && ll == layer_limit &&
           sscanf(last, "# COMPLETE blob_bytes=%llu", &bytes) == 1 &&
           bytes == (unsigned long long)st.st_size;
}

/* Recover a prefix published to the partial manifest.  Manifest records are
 * emitted in checkpoint order, so their count is enough to skip the same
 * selected tensors on restart.  Blob bytes beyond the last flushed record
 * belong to an interrupted tensor and are deliberately discarded. */
static int partial_prefix(const char *blob, const char *manifest, int rank, int ranks,
                          int layer_limit, uint64_t *bytes, int *tensors)
{
    FILE *f = fopen(manifest, "r"); char line[8192];
    int vr = -1, rr = -1, nn = -1, ll = -1, count = 0; uint64_t end = 0;
    if (!f || !fgets(line, sizeof(line), f) ||
        sscanf(line, "# Q38FNTP layout=%d rank=%d ranks=%d layers=%d",
               &vr, &rr, &nn, &ll) != 4 || vr != Q38FN_TP_LAYOUT_VERSION ||
        rr != rank || nn != ranks || ll != layer_limit) {
        if (f) fclose(f); return 0;
    }
    while (fgets(line, sizeof(line), f)) {
        unsigned long long off, size;
        if (line[0] == '#') continue;
        if (sscanf(line, "%llu %llu", &off, &size) != 2 ||
            off > UINT64_MAX - size) { fclose(f); return 0; }
        end = (uint64_t)off + (uint64_t)size; count++;
    }
    fclose(f);
    if (!count || truncate(blob, (off_t)end)) return 0;
    *bytes = end; *tensors = count; return 1;
}

int main(int argc, char **argv)
{
    int rank = 0, ranks = 0, world_rank = 0, world_ranks = 0;
    int rc = 1, out = -1, layer_start = 0, layer_limit = 0, layer_meta = 0;
#ifdef Q38FN_TP_STAGE_DEFAULT_Q5
    int all_q5 = Q38FN_TP_STAGE_DEFAULT_Q5;
#else
    int all_q5 = 0;
#endif
    int all_q8 = 0;
#ifdef Q38FN_TP_STAGE_DEFAULT_Q8
    all_q8 = Q38FN_TP_STAGE_DEFAULT_Q8;
#endif
    FILE *manifest = NULL;
    glm53f_st_context *ctx = NULL; void *buffer = NULL; q38fn_q5_block *packed = NULL;
    char dir[4096], blob[4096], mani[4096], blob_tmp[4096], mani_tmp[4096];
    MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); MPI_Comm_size(MPI_COMM_WORLD, &world_ranks);
    rank = world_rank; ranks = world_ranks;
    if (getenv("Q38FN_TP_PIPELINE") && world_ranks > Q38FN_TP_RANKS &&
        world_ranks % Q38FN_TP_RANKS == 0) {
        int stages = world_ranks / Q38FN_TP_RANKS;
        int stage = world_rank / Q38FN_TP_RANKS;
        rank = world_rank % Q38FN_TP_RANKS; ranks = Q38FN_TP_RANKS;
        layer_start = (Q38FN_LAYERS * stage) / stages;
        layer_limit = (Q38FN_LAYERS * (stage + 1)) / stages;
    }
    if (argc != 3 || ranks != Q38FN_TP_RANKS) {
        if (!world_rank) fprintf(stderr, "usage: %s MODEL_DIR LOCAL_BASE (%d ranks, or grouped pipeline)\n", argv[0], Q38FN_TP_RANKS);
        goto done;
    }
    if (snprintf(dir, sizeof(dir), "%s/rank-%02d", argv[2], rank) >= (int)sizeof(dir) ||
        snprintf(blob, sizeof(blob), "%s/tp%d-v%d.blob", dir, ranks, Q38FN_TP_LAYOUT_VERSION) >= (int)sizeof(blob) ||
        snprintf(mani, sizeof(mani), "%s/tp%d-v%d.manifest", dir, ranks, Q38FN_TP_LAYOUT_VERSION) >= (int)sizeof(mani) ||
        snprintf(blob_tmp, sizeof(blob_tmp), "%s.partial", blob) >= (int)sizeof(blob_tmp) ||
        snprintf(mani_tmp, sizeof(mani_tmp), "%s.partial", mani) >= (int)sizeof(mani_tmp) ||
        mkdir_p(dir)) goto done;
    {
        const char *value = getenv("Q38FN_TP_STAGE_LAYERS");
        if (value && *value) layer_limit = atoi(value);
        value = getenv("Q38FN_TP_STAGE_LAYER_START");
        if (value && *value) layer_start = atoi(value);
        value = getenv("Q38FN_TP_STAGE_ALL_Q5");
        if (value && *value) all_q5 = *value != '0';
        value = getenv("Q38FN_TP_STAGE_Q8");
        if (value && *value) all_q8 = *value != '0';
    }
    /* Partial-resume tensor counts are valid only for the same filtered
     * tensor sequence and storage format.  Encode staging options in the
     * manifest signature so changing n-gram/Q5/Q8 mode cannot silently
     * resume at the wrong tensor. */
    layer_meta = layer_start * 1000 + layer_limit;
    if (getenv("Q38FN_TP_STAGE_SKIP_NGRAM")) layer_meta += 1000000;
    if (all_q8) layer_meta += 2000000;
    if (all_q5) layer_meta += 4000000;
    if (complete_blob(blob, mani, rank, ranks, layer_meta)) { rc = 0; goto done; }
    ctx = glm53f_st_open(argv[1]); if (!ctx) goto done;
    buffer = malloc(IO_BYTES); if (!buffer) goto done;
    packed = malloc((size_t)IO_BYTES / 32 * sizeof(*packed) / sizeof(uint16_t));
    if (!packed) goto done;
    uint64_t offset = 0; int tensors = 0, resume_tensors = 0;
    int resumed = partial_prefix(blob_tmp, mani_tmp, rank, ranks, layer_meta,
                                 &offset, &resume_tensors);
    out = open(blob_tmp, O_WRONLY | O_CREAT | (resumed ? 0 : O_TRUNC), 0600);
    if (out < 0 || lseek(out, (off_t)offset, SEEK_SET) < 0) goto done;
    manifest = fopen(mani_tmp, resumed ? "a" : "w"); if (!manifest) goto done;
    if (!resumed) fprintf(manifest, "# Q38FNTP layout=%d rank=%d ranks=%d layers=%d\n",
                          Q38FN_TP_LAYOUT_VERSION, rank, ranks, layer_meta);
    for (int i = 0; i < ctx->n_entries; ++i) {
        const char *name = ctx->entries[i].name;
        const st_context *owner = ctx->shards[ctx->entries[i].shard].st;
        const st_tensor_info *tensor = &owner->tensors[ctx->entries[i].tensor];
        q38fn_tp_plan plan; uint64_t hash = UINT64_C(1469598103934665603), before;
        int layer = tensor_layer(name);
        if (getenv("Q38FN_TP_STAGE_SKIP_NGRAM") && strstr(name,"ngram_embedding.shard_")) continue;
        if (layer >= 0 && (layer < layer_start || (layer_limit > 0 && layer >= layer_limit))) continue;
        if (strcmp(tensor->dtype_str, "BF16") ||
            q38fn_tp_make_plan(name, tensor->shape, tensor->n_dims, rank, ranks, &plan)) continue;
        if (plan.kind == Q38FN_TP_SKIP) continue;
        if (tensors < resume_tensors) { tensors++; continue; }
        uint64_t aligned = (offset + 255u) & ~UINT64_C(255);
        if (aligned != offset && lseek(out, (off_t)aligned, SEEK_SET) < 0) goto done;
        offset = aligned; before = offset;
        if (copy_plan(ctx, name, tensor, &plan, out, buffer, packed, all_q5, all_q8,
                      &offset, &hash)) {
            fprintf(stderr, "q38fn_tp_stage rank=%d tensor=%s failed\n", rank, name); goto done;
        }
        fprintf(manifest, "%llu %llu %016llx %d %d %d",
                (unsigned long long)before, (unsigned long long)(offset - before),
                (unsigned long long)hash, (int)plan.kind, plan.axis, tensor->n_dims);
        for (int d = 0; d < tensor->n_dims; ++d)
            fprintf(manifest, " %llu", (unsigned long long)tensor->shape[d]);
        fprintf(manifest, " %d", plan.n_ranges);
        for (int r = 0; r < plan.n_ranges; ++r)
            fprintf(manifest, " %llu %llu", (unsigned long long)plan.range[r].start,
                    (unsigned long long)plan.range[r].count);
        fprintf(manifest, " %s\n", name); tensors++;
        /* A flushed manifest record is the restart commit point. */
        if (fflush(manifest)) goto done;
        if (!(tensors & 31) && fdatasync(out)) goto done;
        posix_fadvise(out, (off_t)before, (off_t)(offset-before), POSIX_FADV_DONTNEED);
    }
    fprintf(manifest, "# COMPLETE blob_bytes=%llu tensors=%d\n",
            (unsigned long long)offset, tensors);
    if (fflush(manifest) || fsync(fileno(manifest)) || fdatasync(out)) goto done;
    fclose(manifest); manifest = NULL; close(out); out = -1;
    if (rename(blob_tmp, blob) || rename(mani_tmp, mani)) goto done;
    fprintf(stderr, "Q38FN_TP_STAGE rank=%d tensors=%d bytes=%llu\n",
            rank, tensors, (unsigned long long)offset); rc = 0;
done:
    if (manifest) fclose(manifest);
    if (out >= 0) close(out);
    free(packed); free(buffer); glm53f_st_close(ctx);
    if (rc) MPI_Abort(MPI_COMM_WORLD, rc);
    MPI_Finalize(); return rc;
}
