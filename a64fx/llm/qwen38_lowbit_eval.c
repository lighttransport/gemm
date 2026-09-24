/* Untimed teacher-forced quality oracle. BF16/NVFP4 stream one layer at a
 * time on ONE node. Layer-major token order preserves each layer's causal
 * attention and recurrent state without rereading the model per token. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define BPE_TOKENIZER_IMPLEMENTATION
#include "bpe_tokenizer.h"
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct {
    char magic[8];
    uint64_t version, tokens, vocab, token_hash, format, arithmetic, reserved;
} logit_header;
_Static_assert(sizeof(logit_header) == 64, "logit header ABI");

typedef struct {
    qtensor *tensor;
    const void *original;
    size_t offset, bytes;
    int source;
} stream_entry;

static uint64_t token_hash(const int32_t *tokens, int count) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (int i = 0; i < count; i++) for (int b = 0; b < 4; b++) {
        h ^= ((uint32_t)tokens[i] >> (8 * b)) & 255;
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static int read_tensor(const gguf_context *g, int index, void *out, size_t bytes) {
    int fd = g->tensor_fds ? g->tensor_fds[index] : g->fd;
    uint64_t offset = g->tensor_file_offsets ? g->tensor_file_offsets[index] :
        g->data_offset + g->tensors[index].offset;
    if (offset > INT64_MAX || bytes > (uint64_t)INT64_MAX - offset) return 0;
    for (size_t done = 0; done < bytes; ) {
        size_t n = bytes - done < 8u*1024u*1024u ? bytes - done : 8u*1024u*1024u;
        ssize_t got = pread(fd, (char *)out + done, n, (off_t)(offset + done));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return 0;
        posix_fadvise(fd, (off_t)(offset + done), got, POSIX_FADV_DONTNEED);
        done += (size_t)got;
    }
    return 1;
}

static int stream_phase(transformer_model *m, const gguf_context *g, int phase,
    stream_entry *entries, size_t *bytes) {
    int count = 0;
    *bytes = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const char *name = g->tensors[i].name.str;
        int layer = -1, consumed = 0;
        int selected = phase == m->n_layers ?
            (!strcmp(name, "output.weight") || !strcmp(name, "output_norm.weight")) :
            (sscanf(name, "blk.%d.%n", &layer, &consumed) == 1 && consumed > 0 && layer == phase);
        if (!selected) continue;
        qtensor *t = tf_tp_stage_tensor(m, name);
        if (!t || !t->data) { fprintf(stderr, "oracle: unmapped tensor %s\n", name); return -1; }
        uint64_t n = gguf_tensor_size(g, (int)i);
        if (!n || n > SIZE_MAX - 255 || *bytes > SIZE_MAX - ((n + 255) & ~UINT64_C(255))) return -1;
        entries[count++] = (stream_entry){t, t->data, *bytes, (size_t)n, (int)i};
        *bytes += ((size_t)n + 255) & ~(size_t)255;
    }
    return count;
}

static double logsumexp(const float *x, int n) {
    float top = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (!isfinite(x[i])) return NAN;
        if (x[i] > top) top = x[i];
    }
    double total = 0;
    for (int i = 0; i < n; i++) total += exp((double)x[i] - top);
    return top + log(total);
}

static int usage(const char *program) {
    fprintf(stderr, "usage: %s MODEL --format bf16|nvfp4|fp4|fp6 "
        "(--tokens IDS.txt | --text CORPUS.txt) [--tokens-out IDS.txt] "
        "[--output LOGITS.bin] [--reference LOGITS.bin] [--image IMAGE] "
        "[--activation f32|a8|a16] [--max-tokens N] [--threads N] [--serial]\n", program);
    return 2;
}

int main(int argc, char **argv) {
    const char *token_path = NULL, *text_path = NULL, *token_out = NULL;
    const char *output = NULL, *reference = NULL, *image = NULL, *format_name = NULL;
    int format = 0, arithmetic = 0, maximum = 1025, threads = 48, serial = 0;
    if (argc < 4) return usage(argv[0]);
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--serial")) { serial = 1; continue; }
        if (i + 1 == argc) return usage(argv[0]);
        const char *key = argv[i++], *value = argv[i];
        if (!strcmp(key, "--format")) format_name = value;
        else if (!strcmp(key, "--tokens")) token_path = value;
        else if (!strcmp(key, "--text")) text_path = value;
        else if (!strcmp(key, "--tokens-out")) token_out = value;
        else if (!strcmp(key, "--output")) output = value;
        else if (!strcmp(key, "--reference")) reference = value;
        else if (!strcmp(key, "--image")) image = value;
        else if (!strcmp(key, "--max-tokens")) maximum = atoi(value);
        else if (!strcmp(key, "--threads")) threads = atoi(value);
        else if (!strcmp(key, "--activation")) {
            if (!strcmp(value, "a8")) arithmetic = 8;
            else if (!strcmp(value, "a16")) arithmetic = 16;
            else if (strcmp(value, "f32")) return usage(argv[0]);
        } else return usage(argv[0]);
    }
    if (!format_name || (!!token_path + !!text_path != 1) || maximum < 2 ||
        maximum > 16385 || threads < 1 || threads > 48) return usage(argv[0]);
    if (!strcmp(format_name, "fp4")) format = Q38_LB_NVFP4;
    else if (!strcmp(format_name, "fp6")) format = Q38_LB_FP6_E2M3;
    else if (strcmp(format_name, "bf16") && strcmp(format_name, "nvfp4")) return usage(argv[0]);
    if (!format && (arithmetic || image || serial)) return usage(argv[0]);
    if (output && reference && !strcmp(output, reference)) return usage(argv[0]);
    /* Never let the generic loader materialize a 52 GB BF16 model. */
    setenv("GGUF_LAZY_MMAP", "1", 1); setenv("TF_FORCE_MMAP", "1", 1);
    setenv("TF_KV_DTYPE", "f32", 1); setenv("NUMA_DISTRIBUTE", "1", 1);
    setenv("NUMA_N_CMGS", "4", 1);
    gguf_context *g = gguf_open_multi(argv[1], 1);
    if (!g) return 1;
    q38_lowbit_model *lowbit = NULL;
    transformer_model *m = NULL;
    bpe_vocab *v = NULL;
    int32_t *tokens = NULL;
    float *hidden = NULL, *gold = NULL;
    stream_entry *entries = NULL;
    void *buffer = MAP_FAILED;
    size_t capacity = 0;
    FILE *out = NULL, *ref = NULL;
    int rc = 1, count = 0, output_created = 0;
    v = bpe_vocab_load(g);
    tokens = malloc((size_t)maximum * sizeof(*tokens));
    if (!v || !tokens) goto done;
    if (token_path) {
        FILE *f = fopen(token_path, "r");
        if (!f) goto done;
        long long id;
        while (count < maximum && fscanf(f, "%lld", &id) == 1) {
            if (id < 0 || id >= v->n_tokens) { fclose(f); goto done; }
            tokens[count++] = (int32_t)id;
        }
        fclose(f);
    } else {
        FILE *f = fopen(text_path, "rb");
        if (!f) goto done;
        char *text = malloc(16u*1024u*1024u + 1);
        if (!text) { fclose(f); goto done; }
        size_t n = fread(text, 1, 16u*1024u*1024u, f);
        int ok = !ferror(f) && (feof(f) || fgetc(f) == EOF);
        fclose(f); text[n] = 0;
        if (ok) count = bpe_tokenize(v, text, (int)n, tokens, maximum);
        if (count > maximum) count = maximum; /* Tokenizer writes the bounded prefix. */
        free(text);
    }
    if (count < 2 || count > maximum) { fprintf(stderr, "oracle: need at least two tokens\n"); goto done; }
    if (token_out) {
        FILE *f = fopen(token_out, "w");
        if (!f) goto done;
        int ok = 1;
        for (int i = 0; i < count; i++) if (fprintf(f, "%d\n", tokens[i]) < 0) ok = 0;
        if (fclose(f)) ok = 0;
        if (!ok) goto done;
    }
    if (format) {
        lowbit = image ? q38_lowbit_model_load_image(g, format, arithmetic, threads == 48,
                    (size_t)6*1024*1024*1024, image) :
            q38_lowbit_model_load(g, format, arithmetic, threads == 48, (size_t)6*1024*1024*1024);
        if (!lowbit) goto done;
    } else {
        int expected = !strcmp(format_name, "bf16") ? GGML_TYPE_BF16 : GGML_TYPE_NVFP4;
        int found = 0;
        for (uint64_t i = 0; i < g->n_tensors; i++)
            if (!strcmp(g->tensors[i].name.str, "blk.0.ffn_gate.weight")) found = g->tensors[i].type == (uint32_t)expected;
        if (!found) { fprintf(stderr, "oracle: source format mismatch\n"); goto done; }
    }
    m = transformer_load(g, count);
    if (!m || !m->is_hybrid || m->use_moe || !m->has_lm_head) goto done;
    transformer_set_threads(m, threads); tf_numa_init(m);
    entries = calloc((size_t)g->n_tensors, sizeof(*entries));
    hidden = malloc((size_t)(count - 1) * m->n_embd * sizeof(*hidden));
    if (!entries || !hidden) goto done;
    if (!format) {
        for (int l = 0; l <= m->n_layers; l++) {
            size_t bytes;
            if (stream_phase(m, g, l, entries, &bytes) < 1) goto done;
            if (bytes > capacity) capacity = bytes;
        }
        if (capacity > (size_t)4*1024*1024*1024 ||
            tf_mem_available_gb() < 6.0 + capacity / (1024.0*1024.0*1024.0)) {
            fprintf(stderr, "oracle: layer/head exceeds 4 GiB bounded workspace\n"); goto done;
        }
        buffer = mmap(NULL, capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (buffer == MAP_FAILED) goto done;
        fprintf(stderr, "oracle: one-node layer workspace %.3f GB, hidden %.3f MB\n",
            capacity / 1e9, (double)(count - 1) * m->n_embd * sizeof(*hidden) / 1e6);
    }
    logit_header header = {{0}, 1, (uint64_t)count, (uint64_t)m->n_vocab,
        token_hash(tokens, count), (uint64_t)(format ? format : !strcmp(format_name, "bf16") ? 3 : 4),
        (uint64_t)arithmetic, 0};
    memcpy(header.magic, "Q38LOG1", 8);
    logit_header ref_header = {{0}, 0, 0, 0, 0, 0, 0, 0};
    uint64_t reference_hash = UINT64_C(14695981039346656037);
    if (reference) {
        logit_header h;
        ref = fopen(reference, "rb"); gold = malloc((size_t)m->n_vocab * sizeof(*gold));
        if (!ref || !gold || fread(&h, sizeof(h), 1, ref) != 1 ||
            memcmp(h.magic, header.magic, 8) || h.version != 1 || h.reserved ||
            h.vocab != header.vocab || h.tokens != header.tokens || h.token_hash != header.token_hash) {
            fprintf(stderr, "oracle: incompatible reference logits\n"); goto done;
        }
        ref_header = h;
    }
    if (output) {
        int fd = open(output, O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd < 0) goto done;
        output_created = 1;
        out = fdopen(fd, "wb");
        if (!out) close(fd);
        if (!out || fwrite(&header, sizeof(header), 1, out) != 1) goto done;
    }
    transformer_reset_runtime_state(m);
    if (!serial) {
        for (int t = 0; t < count - 1; t++) {
            transformer_embed_token(m, tokens[t]);
            memcpy(hidden + (size_t)t * m->n_embd, m->x, (size_t)m->n_embd * sizeof(float));
        }
        for (int l = 0; l < m->n_layers; l++) {
            if (tf_mem_available_gb() < 6.0) {
                fprintf(stderr, "oracle: memory reserve crossed\n"); goto done;
            }
            size_t bytes = 0;
            int n = format ? 0 : stream_phase(m, g, l, entries, &bytes);
            if (n < 0) goto done;
            for (int i = 0; i < n; i++) {
                if (!read_tensor(g, entries[i].source, (char *)buffer + entries[i].offset, entries[i].bytes)) goto done;
                entries[i].tensor->data = (uint8_t *)buffer + entries[i].offset;
            }
            for (int t = 0; t < count - 1; t++) {
                memcpy(m->x, hidden + (size_t)t * m->n_embd, (size_t)m->n_embd * sizeof(float));
                if (!transformer_forward_partial(m, t, l, l + 1)) goto done;
                memcpy(hidden + (size_t)t * m->n_embd, m->x, (size_t)m->n_embd * sizeof(float));
            }
            for (int i = 0; i < n; i++) entries[i].tensor->data = (uint8_t *)entries[i].original;
            fprintf(stderr, "oracle: layer %d/%d complete\n", l + 1, m->n_layers);
        }
    }
    size_t bytes = 0;
    int n = format ? 0 : stream_phase(m, g, m->n_layers, entries, &bytes);
    if (n < 0) goto done;
    for (int i = 0; i < n; i++) {
        if (!read_tensor(g, entries[i].source, (char *)buffer + entries[i].offset, entries[i].bytes)) goto done;
        entries[i].tensor->data = (uint8_t *)buffer + entries[i].offset;
    }
    double nll = 0, squared = 0, ref_squared = 0, max_error = 0, kl = 0;
    int agreement = 0, target_matches = 0, first_mismatch = -1;
    for (int t = 0; t < count - 1; t++) {
        float *logits;
        if (serial) logits = transformer_forward_logits(m, tokens[t], t);
        else {
            memcpy(m->x, hidden + (size_t)t * m->n_embd, (size_t)m->n_embd * sizeof(float));
            logits = transformer_compute_logits(m);
        }
        if (!logits) goto done;
        double z = logsumexp(logits, m->n_vocab);
        if (!isfinite(z)) goto done;
        nll += z - logits[tokens[t+1]];
        int best = 0;
        for (int i = 1; i < m->n_vocab; i++) if (logits[i] > logits[best]) best = i;
        target_matches += best == tokens[t+1];
        if (best != tokens[t+1] && first_mismatch < 0) first_mismatch = t + 1;
        if (out && fwrite(logits, sizeof(float), m->n_vocab, out) != (size_t)m->n_vocab) goto done;
        if (ref) {
            if (fread(gold, sizeof(float), m->n_vocab, ref) != (size_t)m->n_vocab) goto done;
            double rz = logsumexp(gold, m->n_vocab);
            if (!isfinite(rz)) goto done;
            int rb = 0;
            for (int i = 0; i < m->n_vocab; i++) {
                uint32_t bits;
                memcpy(&bits, gold + i, sizeof(bits));
                reference_hash ^= bits;
                reference_hash *= UINT64_C(1099511628211);
                if (gold[i] > gold[rb]) rb = i;
                double error = (double)logits[i] - gold[i];
                squared += error * error; ref_squared += (double)gold[i] * gold[i];
                if (fabs(error) > max_error) max_error = fabs(error);
                kl += exp((double)gold[i] - rz) * ((double)gold[i] - logits[i] + z - rz);
            }
            agreement += best == rb;
        }
        if ((t & 7) == 7 || t == count - 2) {
            if (out && (fflush(out) || fdatasync(fileno(out)))) goto done;
            if (out) posix_fadvise(fileno(out), 0, 0, POSIX_FADV_DONTNEED);
            if (ref) posix_fadvise(fileno(ref), 0, 0, POSIX_FADV_DONTNEED);
        }
    }
    for (int i = 0; i < n; i++) entries[i].tensor->data = (uint8_t *)entries[i].original;
    if (ref && fgetc(ref) != EOF) goto done;
    if (out && fflush(out)) goto done;
    printf("{\"format\":\"%s\",\"arithmetic\":%d,\"tokens\":%d,\"token_hash\":\"%016llx\","
           "\"nll\":%.17g,\"perplexity\":%.17g,\"has_reference\":%s,"
           "\"relative_l2\":%.17g,\"max_abs\":%.17g,\"kl\":%.17g,\"top1_matches\":%d,"
           "\"target_matches\":%d,\"first_target_mismatch\":%d,\"order\":\"%s\","
           "\"reference_format\":%llu,\"reference_arithmetic\":%llu,"
           "\"reference_token_hash\":\"%016llx\",\"reference_hash\":\"%016llx\"}\n",
        format_name, arithmetic, count, (unsigned long long)header.token_hash,
        nll / (count-1), exp(nll / (count-1)), ref ? "true" : "false",
        ref_squared ? sqrt(squared / ref_squared) : sqrt(squared), max_error,
        kl / (count-1), agreement, target_matches, first_mismatch, serial ? "serial" : "layer",
        (unsigned long long)ref_header.format, (unsigned long long)ref_header.arithmetic,
        (unsigned long long)ref_header.token_hash, (unsigned long long)reference_hash);
    rc = 0;
done:
    if (out && fclose(out)) rc = 1;
    if (ref) fclose(ref);
    if (m) transformer_free(m);
    q38_lowbit_model_free(lowbit); bpe_vocab_free(v); gguf_close(g);
    if (buffer != MAP_FAILED) munmap(buffer, capacity);
    free(entries); free(tokens); free(hidden); free(gold);
    if (rc && output_created) unlink(output); /* Never leave an incomplete reference. */
    return rc;
}
