/* Streamed Qwen3-VL text encoder, original BF16 checkpoint. Input may be a
 * native-tokenized prompt or unpadded text-only token IDs. No vision, KV
 * cache, or LM head.
 * Reuse the denoiser's checked weight upload and BF16 GEMM infrastructure. */
#define main qimg21_denoiser_main
#include "test_cuda_qimg21_native.c"
#undef main
#include "text_kernels.h"
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"
#include "qwen_tokenizer_json.h"

static int text_bf16_gemm_output = 1;

/* --prompts-file encodes several prompts in one pass: their rows are
 * concatenated, every streamed weight is used by each prompt in turn, and
 * everything that mixes rows (the GEMM's algorithm choice, attention, RoPE
 * positions) runs per prompt, so each prompt's embeddings are bitwise the
 * ones a single-prompt run gives. One segment covers all rows otherwise. */
#define TEXT_MAX_ROWS 32768
static int text_segments = 1, text_seg_start[256], text_seg_len[256];

/* Linear weights stream from the mapped checkpoint, about 14 GB per prompt,
 * while the GEMMs over a few dozen tokens take tens of milliseconds. So the
 * upload, not the math, is the encoder's cost, and it is pipelined:
 *
 * - A loader thread walks the matrices in the order the layers consume them
 *   and fills a ring of TEXT_SLOTS pinned + device buffers ahead of the GEMMs.
 * - Each matrix is read out of the page cache by TEXT_COPY_THREADS threads
 *   with pread() straight into the pinned slot. Copying from the checkpoint
 *   mapping instead faults in every page and leaves 14 GB of mappings for the
 *   kernel to tear down at exit, about a second on its own.
 * - The DMA of one matrix overlaps the host copy of the next.
 *
 * The bytes on the device are exactly the checkpoint's, as before. */
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#define TEXT_WEIGHT_BYTES ((size_t)12288 * 4096 * 2)
#define TEXT_SLOTS 2
#define TEXT_COPY_THREADS 8

static double text_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct { const uint8_t *src; size_t bytes; int fd; off_t offset; } text_matrix;
static struct {
    pthread_t thread; int started;
    pthread_mutex_t mu; pthread_cond_t cv;
    text_matrix *jobs; int count;
    int produced, consumed, failed, stop;
    void *pinned[TEXT_SLOTS]; CUdeviceptr dev[TEXT_SLOTS]; CUevent ready[TEXT_SLOTS];
    CUstream copy; CUcontext ctx;
    size_t bytes_total; double wait_s, read_s;
} tp = {.mu = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER};

typedef struct { uint8_t *dst; const uint8_t *src; size_t bytes; int fd; off_t offset; int failed; } text_copy_part;
static void *text_copy_part_run(void *arg) {
    text_copy_part *part = arg;
    if (part->fd < 0) { memcpy(part->dst, part->src, part->bytes); return NULL; }
    for (size_t done = 0; done < part->bytes;) {
        ssize_t got = pread(part->fd, part->dst + done, part->bytes - done, part->offset + (off_t)done);
        if (got <= 0) { part->failed = 1; return NULL; }
        done += (size_t)got;
    }
    return NULL;
}

static int text_parallel_copy(uint8_t *dst, const text_matrix *m) {
    const uint8_t *src = m->src;
    size_t bytes = m->bytes;
    pthread_t threads[TEXT_COPY_THREADS];
    text_copy_part parts[TEXT_COPY_THREADS];
    size_t per = (bytes / TEXT_COPY_THREADS + 4095) & ~(size_t)4095;
    int spawned = 0;
    for (int t = 0; t < TEXT_COPY_THREADS; t++) {
        size_t off = (size_t)t * per;
        if (off >= bytes) break;
        parts[t] = (text_copy_part){dst + off, src + off, bytes - off < per ? bytes - off : per,
                                    m->fd, m->offset + (off_t)off, 0};
        if (pthread_create(&threads[t], NULL, text_copy_part_run, &parts[t])) {
            text_copy_part_run(&parts[t]);
            continue;
        }
        spawned |= 1 << t;
    }
    int failed = 0;
    for (int t = 0; t < TEXT_COPY_THREADS; t++) {
        if (spawned & (1 << t)) pthread_join(threads[t], NULL);
        if ((size_t)t * per < bytes) failed |= parts[t].failed;
    }
    return failed ? -1 : 0;
}

static void *text_loader(void *arg) {
    (void)arg;
    int failed = cuCtxSetCurrent(tp.ctx) != 0;
    for (int i = 0; i < tp.count && !failed; i++) {
        int slot = i % TEXT_SLOTS;
        pthread_mutex_lock(&tp.mu);
        /* The slot is free once the consumer is done with the matrix that used
         * it last; its GEMM has completed, so so has that matrix's DMA. */
        while (!tp.stop && i - tp.consumed >= TEXT_SLOTS) pthread_cond_wait(&tp.cv, &tp.mu);
        int stop = tp.stop;
        pthread_mutex_unlock(&tp.mu);
        if (stop) break;
        double read_start = text_seconds();
        failed = text_parallel_copy(tp.pinned[slot], &tp.jobs[i]);
        tp.read_s += text_seconds() - read_start;
        failed = failed ||
                 cuMemcpyHtoDAsync(tp.dev[slot], tp.pinned[slot], tp.jobs[i].bytes, tp.copy) ||
                 cuEventRecord(tp.ready[slot], tp.copy);
        pthread_mutex_lock(&tp.mu);
        if (!failed) tp.produced = i + 1;
        pthread_cond_broadcast(&tp.cv);
        pthread_mutex_unlock(&tp.mu);
    }
    pthread_mutex_lock(&tp.mu);
    tp.failed |= failed;
    pthread_cond_broadcast(&tp.cv);
    pthread_mutex_unlock(&tp.mu);
    return NULL;
}

/* Start streaming `jobs` (owned by the caller until text_upload_free). */
static int text_loader_start(text_matrix *jobs, int count) {
    tp.jobs = jobs; tp.count = count;
    if (cuCtxGetCurrent(&tp.ctx) || cuStreamCreate(&tp.copy, CU_STREAM_NON_BLOCKING)) return -1;
    for (int s = 0; s < TEXT_SLOTS; s++) {
        if (cuMemHostAlloc(&tp.pinned[s], TEXT_WEIGHT_BYTES, 0)) { tp.pinned[s] = NULL; return -1; }
        if (!(tp.dev[s] = checked_cuMemAlloc(TEXT_WEIGHT_BYTES))) return -1;
        if (cuEventCreate(&tp.ready[s], CU_EVENT_DISABLE_TIMING)) return -1;
    }
    for (int i = 0; i < count; i++) tp.bytes_total += jobs[i].bytes;
    if (pthread_create(&tp.thread, NULL, text_loader, NULL)) return -1;
    tp.started = 1;
    return 0;
}

/* The device copy of matrix `i`, which must be the next one in stream order.
 * Valid until text_release(i). */
static CUdeviceptr text_acquire(int i, const uint8_t *src) {
    if (i >= tp.count || tp.jobs[i].src != src) {
        fprintf(stderr, "text: weight %d requested out of stream order\n", i);
        return 0;
    }
    double start = text_seconds();
    pthread_mutex_lock(&tp.mu);
    while (tp.produced <= i && !tp.failed) pthread_cond_wait(&tp.cv, &tp.mu);
    int ok = tp.produced > i;
    pthread_mutex_unlock(&tp.mu);
    if (!ok || cuEventSynchronize(tp.ready[i % TEXT_SLOTS])) return 0;
    tp.wait_s += text_seconds() - start;
    return tp.dev[i % TEXT_SLOTS];
}

static void text_release(int i) {
    pthread_mutex_lock(&tp.mu);
    tp.consumed = i + 1;
    pthread_cond_broadcast(&tp.cv);
    pthread_mutex_unlock(&tp.mu);
}

static void text_upload_free(void) {
    if (tp.started) {
        pthread_mutex_lock(&tp.mu);
        tp.stop = 1;
        pthread_cond_broadcast(&tp.cv);
        pthread_mutex_unlock(&tp.mu);
        pthread_join(tp.thread, NULL);
        tp.started = 0;
    }
    if (tp.copy) cuStreamSynchronize(tp.copy);
    for (int s = 0; s < TEXT_SLOTS; s++) {
        if (tp.ready[s]) cuEventDestroy(tp.ready[s]);
        if (tp.dev[s]) cuMemFree(tp.dev[s]);
        if (tp.pinned[s]) cuMemFreeHost(tp.pinned[s]);
        tp.ready[s] = NULL; tp.dev[s] = 0; tp.pinned[s] = NULL;
    }
    if (tp.copy) cuStreamDestroy(tp.copy);
    tp.copy = NULL;
}
static int text_matrix_index;
static int text_fds[4] = {-1, -1, -1, -1};
typedef int (*q21_cutlass_text_attention_fn)(float *, const void *, const void *,
                                             const void *, int, CUstream);

/* Stream-ordered variants of the denoiser's launch helpers, which end in a
 * context synchronize that would also wait on the weight loader's DMA. */
static int text_cast(cuda_qimg_runner *r, CUdeviceptr dst, CUdeviceptr src, int n) {
    void *a[] = {&src, &dst, &n};
    int rc = (int)cuLaunchKernel(r->cast_f32_to_bf16, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a, NULL);
    return rc ? rc : (int)cuStreamSynchronize(r->stream);
}
static int text_vec(CUfunction f, CUstream st, int n, CUdeviceptr x) {
    void *a[] = {&x, &n};
    int rc = (int)cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, st, a, NULL);
    return rc ? rc : (int)cuStreamSynchronize(st);
}

static int text_linear(cuda_qimg_runner *r, qimg21_kernels *k,
                       const qimg21_shards *s, const char *name,
                       CUdeviceptr out, CUdeviceptr in_bf, int n, int no, int ni) {
    int idx;
    st_context *st = find_tensor(s, name, &idx);
    if (!st || strcmp(safetensors_dtype(st, idx), "BF16") || safetensors_ndims(st, idx) != 2 ||
        safetensors_shape(st, idx)[0] != (uint64_t)no || safetensors_shape(st, idx)[1] != (uint64_t)ni ||
        safetensors_nbytes(st, idx) != (size_t)no * ni * 2) {
        fprintf(stderr, "text: unsupported/missing matrix %s\n", name);
        return -1;
    }
    int slot_index = text_matrix_index++;
    CUdeviceptr w = text_acquire(slot_index, (const uint8_t *)safetensors_data(st, idx));
    if (!w) { fprintf(stderr, "text: upload of %s failed\n", name); return -1; }
    /* Stream syncs only: a context synchronize would also wait for the next
     * matrix's DMA on the loader's stream and serialize the pipeline. The
     * GEMM result buffer is persistent for the same reason -- cuMemFree may
     * wait on the whole device. */
    static CUdeviceptr result;
    static size_t result_rows;
    int rc = 0;
    if (text_bf16_gemm_output) {
        /* Sized once, for the widest output, before the pipeline is busy. */
        size_t rows = n > 4096 ? (size_t)n : 4096;
        if (!result && (result = checked_cuMemAlloc(rows * 12288 * 2))) result_rows = rows;
        if (!result || (size_t)n > result_rows) rc = -1;
        CUdeviceptr bias = 0;
        for (int g = 0; g < text_segments && !rc; g++) {
            int rows_g = text_segments > 1 ? text_seg_len[g] : n;
            size_t at = text_segments > 1 ? (size_t)text_seg_start[g] : 0;
            rc = cublasew_gemm_bf16_bf16_bf16_rowmajor_nt(r->cublaslt_ctx, result + at * no * 2, w,
                                                          in_bf + at * ni * 2, rows_g, no, ni);
        }
        void *args[] = {&out, &result, &bias, &no, &n};
        if (!rc) rc = cuLaunchKernel(r->bf16_to_f32_add_bias, (n * no + 255) / 256, 1, 1,
                                     256, 1, 1, 0, r->stream, args, NULL);
    } else
        for (int g = 0; g < text_segments && !rc; g++) {
            int rows_g = text_segments > 1 ? text_seg_len[g] : n;
            size_t at = text_segments > 1 ? (size_t)text_seg_start[g] : 0;
            rc = gemm(r, out + at * no * 4, w, in_bf + at * ni * 2, rows_g, no, ni);
        }
    if (!rc) rc = text_vec(k->round_bf16, r->stream, n * no, out);
    /* The GEMM is done with the weight, so its slot can be refilled. */
    text_release(slot_index);
    return rc;
}

/* Norm weights are tiny (about 1 MB for all layers), so they are converted
 * and uploaded once, before the layers run, instead of an allocate, copy and
 * free per call -- a free may wait on the whole device, which would stall
 * the weight pipeline. text_norm_preload fills this table in call order. */
typedef struct { const void *src; CUdeviceptr dev; } text_norm_weight;
static text_norm_weight *text_norms;
static int text_norm_count;
static CUdeviceptr text_norm_arena;

static int text_norm_preload(const qimg21_shards *s, int start_layer, int layers) {
    static const char *order[] = {"input_layernorm.weight", "self_attn.q_norm.weight",
                                  "self_attn.k_norm.weight", "post_attention_layernorm.weight"};
    int count = (layers - start_layer) * 4;
    size_t floats = 0;
    char name[256];
    text_norms = calloc((size_t)count, sizeof(*text_norms));
    if (!text_norms) return -1;
    for (int pass = 0; pass < 2; pass++) {
        size_t at = 0;
        for (int l = start_layer, i = 0; l < layers; l++)
            for (int j = 0; j < 4; j++, i++) {
                int idx;
                snprintf(name, sizeof(name), "model.language_model.layers.%d.%s", l, order[j]);
                st_context *st = find_tensor(s, name, &idx);
                if (!st || strcmp(safetensors_dtype(st, idx), "BF16")) return -1;
                size_t d = safetensors_nbytes(st, idx) / 2;
                if (pass == 1) {
                    float *host = malloc(d * sizeof(float));
                    const uint16_t *bf = (const uint16_t *)safetensors_data(st, idx);
                    if (!host) return -1;
                    for (size_t k = 0; k < d; k++) { uint32_t u = (uint32_t)bf[k] << 16; memcpy(&host[k], &u, 4); }
                    text_norms[i] = (text_norm_weight){safetensors_data(st, idx), text_norm_arena + at * 4};
                    int rc = cuMemcpyHtoD(text_norms[i].dev, host, d * sizeof(float));
                    free(host);
                    if (rc) return -1;
                }
                at += (d + 63) & ~(size_t)63;
            }
        if (pass == 0 && !(text_norm_arena = checked_cuMemAlloc((floats = at) * sizeof(float)))) return -1;
    }
    (void)floats;
    text_norm_count = count;
    return 0;
}

static int text_norm(cuda_qimg_runner *r, CUfunction fn, const qimg21_shards *s,
                     const char *name, CUdeviceptr out, CUdeviceptr in, int rows, int d,
                     int aten_reduce) {
    int idx;
    st_context *st = find_tensor(s, name, &idx);
    if (!st || strcmp(safetensors_dtype(st, idx), "BF16") ||
        safetensors_nbytes(st, idx) != (size_t)d * 2) return -1;
    CUdeviceptr w = 0;
    for (int i = 0; i < text_norm_count && !w; i++)
        if (text_norms[i].src == safetensors_data(st, idx)) w = text_norms[i].dev;
    if (!w) { fprintf(stderr, "text: norm %s was not preloaded\n", name); return -1; }
    void *a[] = {&out, &in, &w, &d, &rows};
    int threads = d == 128 ? 32 : 256;
    int rc = cuCtxSynchronize();
    if (!rc) rc = aten_reduce
        ? cuLaunchKernel(fn, (rows + 15) / 16, 1, 1, 32, 16, 1, 0, r->stream, a, NULL)
        : cuLaunchKernel(fn, rows, 1, 1, threads, 1, 1, 0, r->stream, a, NULL);
    if (!rc) rc = cuStreamSynchronize(r->stream);
    return rc;
}

int main(int argc, char **argv) {
    const char *model = NULL, *tokens = NULL, *out = NULL, *dump_dir = NULL;
    const char *dump_tokens = NULL, *dump_rope_table = NULL;
    const char *vision_merged = NULL, *vision_deepstack_dir = NULL, *rope_table_path = NULL;
    const char *hidden_input = NULL;
    const char *prompt = NULL, *prompts_file = NULL, *out_dir = NULL;
    const char *attention_mode = "custom";
    const char *rms_mode = "auto";
    const char *post_rms_mode = "auto";
    int drop = 0, start_layer = 0, layers = 36, dump_layer = 0;
    int image_grid_h = 0, image_grid_w = 0, image_start = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--prompts-file") && i + 1 < argc) prompts_file = argv[++i];
        else if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--dump-tokens") && i + 1 < argc) dump_tokens = argv[++i];
        else if (!strcmp(argv[i], "--dump-rope-table") && i + 1 < argc) dump_rope_table = argv[++i];
        else if (!strcmp(argv[i], "--dump-dir") && i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--vision-merged") && i + 1 < argc) vision_merged = argv[++i];
        else if (!strcmp(argv[i], "--vision-deepstack-dir") && i + 1 < argc) vision_deepstack_dir = argv[++i];
        else if (!strcmp(argv[i], "--rope-table") && i + 1 < argc) rope_table_path = argv[++i];
        else if (!strcmp(argv[i], "--hidden") && i + 1 < argc) hidden_input = argv[++i];
        else if (!strcmp(argv[i], "--image-grid-height") && i + 1 < argc) image_grid_h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--image-grid-width") && i + 1 < argc) image_grid_w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--attention") && i + 1 < argc) attention_mode = argv[++i];
        else if (!strcmp(argv[i], "--rms") && i + 1 < argc) rms_mode = argv[++i];
        else if (!strcmp(argv[i], "--post-rms") && i + 1 < argc) post_rms_mode = argv[++i];
        else if (!strcmp(argv[i], "--bf16-gemm-output")) text_bf16_gemm_output = 1;
        else if (!strcmp(argv[i], "--f32-gemm-output")) text_bf16_gemm_output = 0;
        else if (!strcmp(argv[i], "--drop-prefix") && i + 1 < argc) drop = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-layers") && i + 1 < argc) layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--start-layer") && i + 1 < argc) start_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-layer") && i + 1 < argc) dump_layer = atoi(argv[++i]);
        else { fprintf(stderr, "text: unknown/incomplete option %s\n", argv[i]); return 2; }
    }
    if (prompts_file && (prompt || tokens || !out_dir || out || dump_tokens || dump_dir || dump_rope_table ||
                         hidden_input || rope_table_path || start_layer)) {
        fprintf(stderr, "text: --prompts-file takes --out-dir and none of --prompt/--tokens/--out/--dump-*/"
                        "--hidden/--rope-table/--start-layer\n");
        return 2;
    }
    if (prompts_file) prompt = "";   /* the checks below see one prompt source */
    if (!model || (!!tokens == !!prompt) || (!out && !dump_tokens && !out_dir) ||
        (vision_deepstack_dir && !vision_merged) ||
        (strcmp(attention_mode, "custom") && strcmp(attention_mode, "cutlass-efficient") &&
         strcmp(attention_mode, "flash-exact")) ||
        (strcmp(rms_mode,"auto") && strcmp(rms_mode,"scalar") &&
         strcmp(rms_mode,"vector2") && strcmp(rms_mode,"vector4") &&
         strcmp(rms_mode,"vector8") && strcmp(rms_mode,"vector16")) ||
        (strcmp(post_rms_mode,"auto") && strcmp(post_rms_mode,"scalar") &&
         strcmp(post_rms_mode,"vector2") && strcmp(post_rms_mode,"vector4") &&
         strcmp(post_rms_mode,"vector8") && strcmp(post_rms_mode,"vector16")) ||
        drop < 0 || layers < 1 || layers > 36 || start_layer < 0 || start_layer >= layers ||
        dump_layer < start_layer || dump_layer >= layers) {
        fprintf(stderr, "usage: %s --model DIR (--tokens ids.txt | --prompt TEXT) "
                        "[--out embeds.npy] [--dump-tokens ids.txt] "
                        "[--prompts-file NUL-separated.txt --out-dir DIR] "
                        "[--vision-merged FILE --vision-deepstack-dir DIR --rope-table FILE] "
                        "[--hidden FILE --start-layer N] "
                        "[--attention custom|cutlass-efficient|flash-exact --drop-prefix N "
                        "--max-layers 36 --dump-layer N]\n", argv[0]); return 2;
    }
    int *ids = malloc(TEXT_MAX_ROWS * sizeof(int)), n = 0, scanned = EOF;
    int seg_drop[256], seg_image_start[256];
    if (!ids) return 1;
    double t_start = text_seconds(), t_mark = t_start;
    /* One "timing:" line per setup phase, for the demo's breakdown. */
    #define TEXT_PHASE(label) do { double now_ = text_seconds(); \
        fprintf(stderr, "timing: %s %.3f s\n", label, now_ - t_mark); t_mark = now_; } while (0)
    if (prompts_file) {
        char tokenizer_path[2048];
        snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/processor/tokenizer.json", model);
        FILE *fp = fopen(prompts_file, "rb");
        if (!fp) { perror("text: open prompts file"); return 1; }
        size_t cap = 1 << 16, len = 0, got;
        char *text = malloc(cap + 1);
        while (text && (got = fread(text + len, 1, cap - len, fp)) > 0)
            if ((len += got) == cap) text = realloc(text, (cap *= 2) + 1);
        fclose(fp);
        if (!text) return 1;
        text[len] = 0;
        text_segments = 0;
        for (size_t at = 0; at < len; at += strlen(text + at) + 1) {
            if (!text[at]) continue;
            if (text_segments == 256) { fprintf(stderr, "text: more than 256 prompts\n"); return 1; }
            int g = text_segments, d = 0, first = -1;
            int m = vision_merged ? q21_build_multimodal_prompt_tokens(
                        tokenizer_path, text + at, image_grid_h * image_grid_w / 4, ids + n,
                        TEXT_MAX_ROWS - n, &d, &first) :
                    q21_build_prompt_tokens(tokenizer_path, text + at, ids + n, TEXT_MAX_ROWS - n, &d);
            if (m <= d) { fprintf(stderr, "text: tokenization of prompt %d failed (or over %d rows)\n",
                                  g, TEXT_MAX_ROWS); return 1; }
            text_seg_start[g] = n; text_seg_len[g] = m; seg_drop[g] = d; seg_image_start[g] = first;
            n += m; text_segments++;
        }
        free(text);
        if (!text_segments) { fprintf(stderr, "text: no prompts in %s\n", prompts_file); return 1; }
        fprintf(stderr, "text: %d prompts, %d rows\n", text_segments, n);
        if (mkdir(out_dir, 0755) && errno != EEXIST) { perror("text: out-dir"); return 1; }
    } else if (prompt) {
        char tokenizer_path[2048];
        int length = snprintf(tokenizer_path, sizeof(tokenizer_path),
                              "%s/processor/tokenizer.json", model);
        n = length < 0 || length >= (int)sizeof(tokenizer_path) ? -1 :
            vision_merged ? q21_build_multimodal_prompt_tokens(
                tokenizer_path, prompt, image_grid_h * image_grid_w / 4,
                ids, 4096, &drop, &image_start) :
            q21_build_prompt_tokens(tokenizer_path, prompt, ids, 4096, &drop);
        if (n < 0) { fprintf(stderr,"text: native tokenization failed\n"); return 1; }
        fprintf(stderr,"text: native tokenizer produced %d tokens, drop-prefix=%d\n",n,drop);
    } else {
        char word[64];
        FILE *fp = fopen(tokens, "r");
        if (!fp) return 1;
        while ((scanned = fscanf(fp, "%63s", word)) == 1) {
            char *end;
            errno = 0;
            long token = strtol(word, &end, 10);
            if (errno || *end || n == 4096 || token < 0 || token >= 151936 ||
                ((!vision_merged && !hidden_input) &&
                 (token == 151655 || token == 151656 || token == 151652 || token == 151653))) {
                fprintf(stderr, "text: invalid token/vision input or more than 4096 tokens\n");
                fclose(fp); return 1;
            }
            ids[n++] = token;
        }
        fclose(fp);
    }
    if (!prompts_file) {
        if (scanned != EOF || n <= drop) return 1;
        text_seg_start[0] = 0; text_seg_len[0] = n; seg_drop[0] = drop; seg_image_start[0] = image_start;
    }
    if (dump_tokens) {
        FILE *fp = fopen(dump_tokens, "w");
        if (!fp) { perror("text: open token dump"); return 1; }
        for (int i = 0; i < n; i++) fprintf(fp, "%d\n", ids[i]);
        if (fclose(fp)) { perror("text: close token dump"); return 1; }
    }
    if (!out && !out_dir) return 0;
    TEXT_PHASE("tokenize");

    int rc = 1, *visual_rows = NULL;
    text_matrix *jobs = NULL;
    qimg21_shards shards = {{0}, 0};
    cuda_qimg_runner *r = NULL;
    CUmodule module = NULL, base_module = NULL;
    CUdeviceptr x=0, norm=0, bf=0, q=0, key=0, v=0, att=0, tmp=0, gate=0, up=0;
    CUdeviceptr q_bf=0, key_bf=0, v_bf=0, rope_table=0;
    CUdeviceptr visual_rows_d=0, visual_embed_d=0;
    void *cutlass_plugin = NULL;
    q21_cutlass_text_attention_fn cutlass_attention = NULL;
    float *host = calloc((size_t)n * 4096, sizeof(float));
    if (!host) return 1;
    char path[2048], name[256];
    for (int i = 1; i <= 4; i++) {
        snprintf(path, sizeof(path), "%s/text_encoder/model-%05d-of-00004.safetensors", model, i);
        st_context *st = safetensors_open(path);
        if (!st) goto done;
        shards.st[shards.n++] = st;
    }
    int idx;
    st_context *st = find_tensor(&shards, "model.language_model.embed_tokens.weight", &idx);
    if (!st || strcmp(safetensors_dtype(st,idx),"BF16") ||
        safetensors_nbytes(st,idx) != (size_t)151936 * 4096 * 2) goto done;
    const uint16_t *embedding = safetensors_data(st,idx);
    for (int t = 0; t < n; t++) for (int j = 0; j < 4096; j++) {
        uint32_t bits = (uint32_t)embedding[(size_t)ids[t]*4096+j] << 16;
        memcpy(host+(size_t)t*4096+j, &bits, 4);
    }
    visual_rows = malloc((size_t)n * sizeof(int));
    int visual_count = 0, visual_per = 0;
    if (!visual_rows) goto done;
    for (int t = 0; t < n; t++) if (ids[t] == 151655) visual_rows[visual_count++] = t;
    for (int g = 0, first = 0; g < text_segments; g++) {
        int count = 0;
        for (int t = 0; t < text_seg_len[g]; t++) count += ids[text_seg_start[g] + t] == 151655;
        if (g && count != visual_per) {
            fprintf(stderr, "text: prompts disagree on the number of image tokens\n");
            goto done;
        }
        visual_per = count;
        if (count && seg_image_start[g] < 0) seg_image_start[g] = visual_rows[first] - text_seg_start[g];
        first += count;
    }
    if (visual_count && image_start < 0) image_start = seg_image_start[0];
    if (!hidden_input && (!!vision_merged != (visual_count > 0))) {
        fprintf(stderr, "text: image-pad tokens and --vision-merged must be supplied together\n");
        goto done;
    }
    if (visual_count && (!rope_table_path) &&
        (image_grid_h < 2 || image_grid_w < 2 || image_grid_h % 2 || image_grid_w % 2 ||
         image_grid_h / 2 * (image_grid_w / 2) != visual_per)) {
        fprintf(stderr, "text: native multimodal MRoPE requires matching even image grid dimensions\n");
        goto done;
    }
    if (vision_merged) {
        npy_f32 merged = {0};
        if (npy_read_f32(vision_merged, &merged) || merged.ndim != 2 ||
            merged.shape[0] != (size_t)visual_per || merged.shape[1] != 4096) {
            fprintf(stderr, "text: invalid merged vision embedding\n");
            npy_free(&merged); goto done;
        }
        for (int i = 0; i < visual_count; i++)
            memcpy(host + (size_t)visual_rows[i] * 4096,
                   merged.data + (size_t)(i % visual_per) * 4096, 4096 * sizeof(float));
        npy_free(&merged);
    }
    if (hidden_input) {
        npy_f32 hidden = {0};
        if (npy_read_f32(hidden_input, &hidden) ||
            !((hidden.ndim == 2 && hidden.shape[0] == (size_t)n && hidden.shape[1] == 4096) ||
              (hidden.ndim == 3 && hidden.shape[0] == 1 &&
               hidden.shape[1] == (size_t)n && hidden.shape[2] == 4096))) {
            fprintf(stderr, "text: invalid hidden-state replay input\n");
            npy_free(&hidden); goto done;
        }
        memcpy(host, hidden.data, (size_t)n * 4096 * sizeof(float));
        npy_free(&hidden);
    }
    TEXT_PHASE("open checkpoint + embed tokens");
    r = cuda_qimg_init(0, 1);
    if (!r) goto done;
    TEXT_PHASE("CUDA init");
    if (strcmp(attention_mode, "custom")) {
        const char *plugin_path = !strcmp(attention_mode, "flash-exact")
            ? "cuda/qimg21/libq21_flash_attention.so"
            : "cuda/qimg21/libq21_cutlass_attention.so";
        const char *symbol = !strcmp(attention_mode, "flash-exact")
            ? "q21_flash_text_attention" : "q21_cutlass_text_attention";
        cutlass_plugin = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
        if (!cutlass_plugin || !(cutlass_attention = (q21_cutlass_text_attention_fn)
              dlsym(cutlass_plugin, symbol))) {
            fprintf(stderr, "text: %s text attention plugin unavailable\n", attention_mode);
            goto done;
        }
        fprintf(stderr, "text: %s causal GQA attention enabled\n", attention_mode);
    }
    qimg21_kernels base;
    CUfunction rms, rms_aten, rms128, rms2, rms4, rms8, rms16, rope_lookup;
    CUfunction add, add_visual, attention, mul_silu;
    if (cu_compile_kernels(&module,r->device,q21_text_src,"qimg21_text.cu",0,"qimg21_text")<0 ||
        cu_compile_kernels(&base_module,r->device,qimg21_src,"qimg21_native.cu",1,"qimg21_native")<0 ||
        get_kernel(&base,base_module) || cuModuleGetFunction(&rms,module,"text_rms") ||
        cuModuleGetFunction(&rms_aten,module,"text_rms_aten") ||
        cuModuleGetFunction(&rms128,module,"text_rms128") ||
        cuModuleGetFunction(&rms2,module,"text_rms2") ||
        cuModuleGetFunction(&rms4,module,"text_rms4") ||
        cuModuleGetFunction(&rms8,module,"text_rms8") ||
        cuModuleGetFunction(&rms16,module,"text_rms16") ||
        cuModuleGetFunction(&add,module,"text_add") ||
        cuModuleGetFunction(&add_visual,module,"text_add_visual") ||
        cuModuleGetFunction(&rope_lookup,module,"text_rope_table") ||
        cuModuleGetFunction(&attention,module,"text_attn") ||
        cuModuleGetFunction(&mul_silu,module,"text_mul_silu")) goto done;
    TEXT_PHASE("load kernels + attention plugin");
    /* Every linear the layers will ask for, in the order they ask, so the
     * loader can run ahead of them. */
    {
        static const char *order[] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                      "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                                      "mlp.gate_proj.weight", "mlp.up_proj.weight",
                                      "mlp.down_proj.weight"};
        int count = (layers - start_layer) * 7, at = 0;
        for (int i = 0; i < shards.n; i++) {
            snprintf(path, sizeof(path), "%s/text_encoder/model-%05d-of-00004.safetensors", model, i + 1);
            text_fds[i] = open(path, O_RDONLY);
        }
        jobs = calloc((size_t)count, sizeof(*jobs));
        if (!jobs) goto done;
        for (int l = start_layer; l < layers; l++)
            for (int j = 0; j < 7; j++) {
                int tensor;
                snprintf(name, sizeof(name), "model.language_model.layers.%d.%s", l, order[j]);
                st_context *owner = find_tensor(&shards, name, &tensor);
                if (!owner || safetensors_nbytes(owner, tensor) > TEXT_WEIGHT_BYTES) {
                    fprintf(stderr, "text: unsupported/missing matrix %s\n", name);
                    goto done;
                }
                int shard = 0;
                while (shard < shards.n && shards.st[shard] != owner) shard++;
                const uint8_t *src = safetensors_data(owner, tensor);
                /* Fall back to copying from the mapping if the file will not open. */
                int fd = shard < shards.n ? text_fds[shard] : -1;
                jobs[at++] = (text_matrix){src, safetensors_nbytes(owner, tensor), fd,
                                           (off_t)(src - (const uint8_t *)owner->map_base)};
            }
        if (text_loader_start(jobs, count)) { fprintf(stderr, "text: weight loader failed to start\n"); goto done; }
    }
    if (text_norm_preload(&shards, start_layer, layers)) { fprintf(stderr, "text: norm weights failed to load\n"); goto done; }
    #define ALLOC(p,count,bytes) do { p=checked_cuMemAlloc((size_t)(count)*(bytes)); if(!p)goto done; } while(0)
    ALLOC(x,n*4096,4); ALLOC(norm,n*4096,4); ALLOC(bf,n*12288,2);
    ALLOC(q,n*4096,4); ALLOC(key,n*1024,4); ALLOC(v,n*1024,4);
    ALLOC(att,n*4096,4); ALLOC(tmp,n*4096,4); ALLOC(gate,n*12288,4); ALLOC(up,n*12288,4);
    if (visual_count && !rope_table_path) {
        npy_f32 base={0};
        float *composed=NULL;
        if(npy_read_f32("cuda/qimg21/qwen21_text_rope.npy",&base)||base.ndim!=3||
           base.shape[1]!=128||base.shape[2]!=2) {npy_free(&base);goto done;}
        composed=malloc((size_t)n*128*2*4);
        if(!composed){npy_free(&base);goto done;}
        /* Positions restart in every prompt. */
        int hh=image_grid_h/2,ww=image_grid_w/2;
        for(int g=0;g<text_segments;g++) {
          int image_start=seg_image_start[g],after=image_start+(hh>ww?hh:ww);
          for(int t=0;t<text_seg_len[g];t++)for(int j=0;j<128;j++) {
            int k=j&63,pos;
            size_t row=(size_t)text_seg_start[g]+t;
            if(t<image_start)pos=t;
            else if(t<image_start+visual_per) {
                int q=t-image_start;
                pos=k<60&&k%3==1?image_start+q/ww:
                    k<60&&k%3==2?image_start+q%ww:image_start;
            } else pos=after+t-image_start-visual_per;
            if(pos<0||pos>=(int)base.shape[0]){free(composed);npy_free(&base);goto done;}
            composed[(row*128+j)*2]=base.data[((size_t)pos*128+j)*2];
            composed[(row*128+j)*2+1]=base.data[((size_t)pos*128+j)*2+1];
          }
        }
        rope_table=checked_cuMemAlloc((size_t)n*128*2*4);
        if(!rope_table||cuMemcpyHtoD(rope_table,composed,(size_t)n*128*2*4)) {
            free(composed);npy_free(&base);goto done;
        }
        free(composed);npy_free(&base);
    } else {
        npy_f32 table={0};
        const char *table_path = rope_table_path ? rope_table_path : "cuda/qimg21/qwen21_text_rope.npy";
        int longest=0;
        for(int g=0;g<text_segments;g++) if(text_seg_len[g]>longest) longest=text_seg_len[g];
        if(npy_read_f32(table_path,&table) || table.ndim!=3 ||
           table.shape[0]<(size_t)longest || table.shape[1]!=128 || table.shape[2]!=2) {
            fprintf(stderr,"text: invalid/missing qwen21_text_rope.npy\n"); npy_free(&table); goto done;
        }
        rope_table=checked_cuMemAlloc((size_t)n*128*2*4);
        if(!rope_table) {npy_free(&table);goto done;}
        for(int g=0;g<text_segments;g++)
            if(cuMemcpyHtoD(rope_table+(size_t)text_seg_start[g]*128*2*4,table.data,
                            (size_t)text_seg_len[g]*128*2*4)) {npy_free(&table);goto done;}
        npy_free(&table);
    }
    if (dump_rope_table) {
        float *table_host = malloc((size_t)n * 128 * 2 * sizeof(float));
        if (!table_host || cuCtxSynchronize() ||
            cuMemcpyDtoH(table_host, rope_table, (size_t)n * 128 * 2 * sizeof(float)) ||
            npy_write_f32(dump_rope_table, table_host, (size_t)n * 128 * 2, n * 128, 2)) {
            free(table_host); goto done;
        }
        free(table_host);
    }
    if (cutlass_attention) {
        ALLOC(q_bf,n*4096,2); ALLOC(key_bf,n*1024,2); ALLOC(v_bf,n*1024,2);
    }
    #undef ALLOC
    if(cuMemcpyHtoD(x,host,(size_t)n*4096*4))goto done;
    if (visual_count) {
        visual_rows_d=checked_cuMemAlloc((size_t)visual_count*sizeof(int));
        visual_embed_d=checked_cuMemAlloc((size_t)visual_count*4096*4);
        if(!visual_rows_d||!visual_embed_d||
           cuMemcpyHtoD(visual_rows_d,visual_rows,(size_t)visual_count*sizeof(int)))goto done;
    }
    if(dump_dir && mkdir(dump_dir,0755) && errno!=EEXIST)goto done;
    #define CHECK(call) do { if((call)!=0)goto done; } while(0)
    TEXT_PHASE("buffers + RoPE table");
    double t_layers = text_seconds();
    #define NAME(suffix) snprintf(name,sizeof(name),"model.language_model.layers.%d.%s",l,suffix)
    #define LINEAR(suffix,dst,no,ni) do { NAME(suffix); CHECK(text_linear(r,&base,&shards,name,dst,bf,n,no,ni)); } while(0)
    #define DUMP(label,ptr,width) do { if(dump_dir && l==dump_layer) { CHECK(cuCtxSynchronize()); qimg21_stage_dir=dump_dir; dump_stage("stage_" label,ptr,(size_t)n*(width),n,width); } } while(0)
    for(int l=start_layer;l<layers;l++) {
        fprintf(stderr,"text: layer %d/%d (%d tokens)\n",l+1,layers,n);
        /* ATen's contiguous F32 mean reduction uses 16 independent warps per
         * CTA, four vector lanes per accumulator, and one output per warp. */
        CUfunction input_rms = !strcmp(rms_mode,"scalar") ? rms :
            !strcmp(rms_mode,"vector2") ? rms2 : !strcmp(rms_mode,"vector4") ? rms4 :
            !strcmp(rms_mode,"vector8") ? rms8 : !strcmp(rms_mode,"vector16") ? rms16 :
            rms_aten;
        NAME("input_layernorm.weight"); CHECK(text_norm(r,input_rms,&shards,name,norm,x,n,4096,
                                                         input_rms == rms_aten));
        DUMP("input_layernorm",norm,4096);
        CHECK(text_cast(r,bf,norm,n*4096));
        LINEAR("self_attn.q_proj.weight",q,4096,4096);
        LINEAR("self_attn.k_proj.weight",key,1024,4096);
        LINEAR("self_attn.v_proj.weight",v,1024,4096);
        DUMP("self_attn.q_proj",q,4096); DUMP("self_attn.k_proj",key,1024); DUMP("self_attn.v_proj",v,1024);
        NAME("self_attn.q_norm.weight"); CHECK(text_norm(r,rms128,&shards,name,q,q,n*32,128,0));
        NAME("self_attn.k_norm.weight"); CHECK(text_norm(r,rms128,&shards,name,key,key,n*8,128,0));
        DUMP("self_attn.q_norm",q,4096); DUMP("self_attn.k_norm",key,1024);
        int heads=32; void *qa[]={&q,&rope_table,&heads};
        CHECK(cuLaunchKernel(rope_lookup,n,32,1,64,1,1,0,r->stream,qa,NULL));
        heads=8; void *ka[]={&key,&rope_table,&heads};
        CHECK(cuLaunchKernel(rope_lookup,n,8,1,64,1,1,0,r->stream,ka,NULL));
        DUMP("rope_q",q,4096); DUMP("rope_k",key,1024);
        if (cutlass_attention) {
            CHECK(text_cast(r,q_bf,q,n*4096));
            CHECK(text_cast(r,key_bf,key,n*1024));
            CHECK(text_cast(r,v_bf,v,n*1024));
            CHECK(cuStreamSynchronize(r->stream));
            for(int g=0;g<text_segments;g++) {
                size_t o=text_seg_start[g];
                CHECK(cutlass_attention((float *)(uintptr_t)(att+o*4096*4),
                                        (const void *)(uintptr_t)(q_bf+o*4096*2),
                                        (const void *)(uintptr_t)(key_bf+o*1024*2),
                                        (const void *)(uintptr_t)(v_bf+o*1024*2),text_seg_len[g],r->stream));
            }
        } else {
            for(int g=0;g<text_segments;g++) {
                size_t o=text_seg_start[g];
                CUdeviceptr ag=att+o*4096*4,qg=q+o*4096*4,kg=key+o*1024*4,vg=v+o*1024*4;
                int len=text_seg_len[g];
                void *aa[]={&ag,&qg,&kg,&vg,&len};
                CHECK(cuLaunchKernel(attention,32,len,1,32,1,1,0,r->stream,aa,NULL));
            }
            CHECK(text_vec(base.round_bf16,r->stream,n*4096,att));
        }
        DUMP("self_attn.o_proj.input",att,4096);
        CHECK(text_cast(r,bf,att,n*4096));
        LINEAR("self_attn.o_proj.weight",tmp,4096,4096);
        DUMP("self_attn.o_proj",tmp,4096);
        int count=n*4096; void *ra[]={&x,&tmp,&count};
        CHECK(cuLaunchKernel(add,(count+255)/256,1,1,256,1,1,0,r->stream,ra,NULL));
        DUMP("post_attention_hidden",x,4096);
        CUfunction post_rms = !strcmp(post_rms_mode,"vector2") ? rms2 :
            !strcmp(post_rms_mode,"vector4") ? rms4 : !strcmp(post_rms_mode,"vector8") ? rms8 :
            !strcmp(post_rms_mode,"vector16") ? rms16 :
            rms_aten;
        NAME("post_attention_layernorm.weight"); CHECK(text_norm(r,post_rms,&shards,name,norm,x,n,4096,
                                                                  post_rms == rms_aten));
        DUMP("post_attention_layernorm",norm,4096);
        CHECK(text_cast(r,bf,norm,n*4096));
        LINEAR("mlp.gate_proj.weight",gate,12288,4096);
        LINEAR("mlp.up_proj.weight",up,12288,4096);
        DUMP("mlp.gate_proj",gate,12288); DUMP("mlp.up_proj",up,12288);
        int ffcount=n*12288; void *ma[]={&gate,&gate,&up,&ffcount};
        CHECK(cuLaunchKernel(mul_silu,(ffcount+255)/256,1,1,256,1,1,0,r->stream,ma,NULL));
        CHECK(text_cast(r,bf,gate,ffcount));
        LINEAR("mlp.down_proj.weight",tmp,4096,12288);
        DUMP("mlp.down_proj",tmp,4096);
        CHECK(cuLaunchKernel(add,(count+255)/256,1,1,256,1,1,0,r->stream,ra,NULL));
        if (vision_deepstack_dir && l < 3) {
            npy_f32 deep = {0};
            snprintf(path,sizeof(path),"%s/deepstack_%d.npy",vision_deepstack_dir,l);
            if (access(path,R_OK)) {
                snprintf(path,sizeof(path),"%s/vision_deepstack_%d.npy",vision_deepstack_dir,l);
            }
            if(npy_read_f32(path,&deep)||deep.ndim!=2||
               deep.shape[0]!=(size_t)visual_per||deep.shape[1]!=4096){
                npy_free(&deep);goto done;
            }
            for(int g=0;g<text_segments;g++)
                CHECK(cuMemcpyHtoD(visual_embed_d+(size_t)g*visual_per*4096*4,deep.data,
                                   (size_t)visual_per*4096*4));
            CHECK(cuCtxSynchronize());
            npy_free(&deep);
            int visual_values=visual_count*4096;
            void *va[]={&x,&visual_embed_d,&visual_rows_d,&visual_values,&(int){4096}};
            CHECK(cuLaunchKernel(add_visual,(visual_values+255)/256,1,1,256,1,1,0,
                                 r->stream,va,NULL));
        }
        if(dump_dir) {
            CHECK(cuStreamSynchronize(r->stream));
            CHECK(cuMemcpyDtoH(host,x,(size_t)n*4096*4));
            snprintf(path,sizeof(path),"%s/layer_%02d.npy",dump_dir,l);
            CHECK(npy_write_f32(path,host,(size_t)n*4096,n,4096));
        }
    }
    CHECK(cuStreamSynchronize(r->stream));
    {
        double seconds = text_seconds() - t_layers;
        fprintf(stderr, "timing: %d layers %.3f s (%.1f GB of weights streamed at %.1f GB/s, "
                        "%.3f s waiting on the upload, %.3f s reading the checkpoint)\n", layers - start_layer, seconds,
                tp.bytes_total / 1e9, tp.bytes_total / 1e9 / (seconds > 0 ? seconds : 1), tp.wait_s, tp.read_s);
        t_mark = text_seconds();
    }
    CHECK(cuMemcpyDtoH(host,x,(size_t)n*4096*4));
    for(size_t i=0;i<(size_t)n*4096;i++)if(!isfinite(host[i]))goto done;
    if (prompts_file) {
        rc = 0;
        for (int g = 0; g < text_segments && !rc; g++) {
            int rows = text_seg_len[g] - seg_drop[g];
            snprintf(path, sizeof(path), "%s/embeds_%03d.npy", out_dir, g);
            rc = npy_write_f32(path, host + (size_t)(text_seg_start[g] + seg_drop[g]) * 4096,
                               (size_t)rows * 4096, rows, 4096);
            snprintf(path, sizeof(path), "%s/tokens_%03d.txt", out_dir, g);
            FILE *fp = rc ? NULL : fopen(path, "w");
            if (!fp) { rc = 1; break; }
            for (int t = 0; t < text_seg_len[g]; t++) fprintf(fp, "%d\n", ids[text_seg_start[g] + t]);
            if (fclose(fp)) rc = 1;
        }
    } else
        rc=npy_write_f32(out,host+(size_t)drop*4096,(size_t)(n-drop)*4096,n-drop,4096);
    TEXT_PHASE("write embeddings");
    fprintf(stderr, "timing: text encoder total %.3f s\n", text_seconds() - t_start);
    /* Tearing down the context, the pinned ring and 17 GB of mappings costs
     * about a second and buys nothing: the process ends here either way. */
    if (!rc && !getenv("QIMG21_TEXT_CLEAN_EXIT")) { fflush(NULL); _exit(0); }
    #undef CHECK
    #undef NAME
    #undef LINEAR
    #undef DUMP
done:
    if(rc)fprintf(stderr,"text: encoder failed\n");
    free_d(&x);free_d(&norm);free_d(&bf);free_d(&q);free_d(&key);free_d(&v);
    free_d(&att);free_d(&tmp);free_d(&gate);free_d(&up);
    free_d(&q_bf);free_d(&key_bf);free_d(&v_bf);
    free_d(&rope_table);
    free_d(&visual_rows_d);free_d(&visual_embed_d);
    if(module)cuModuleUnload(module);
    if(base_module)cuModuleUnload(base_module);
    if(r)text_upload_free();
    if(text_norm_arena)cuMemFree(text_norm_arena);
    free(text_norms);
    free(jobs);
    for (int i = 0; i < 4; i++) if (text_fds[i] >= 0) close(text_fds[i]);
    if(r)cuda_qimg_free(r);
    if(cutlass_plugin)dlclose(cutlass_plugin);
    for(int i=0;i<shards.n;i++)safetensors_close(shards.st[i]);
    free(host);
    free(visual_rows);
    free(ids);
    return rc;
}
