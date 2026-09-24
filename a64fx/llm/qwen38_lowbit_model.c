#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "qwen38_lowbit_model.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <linux/mempolicy.h>

#define Q38_PAGE (2u * 1024u * 1024u)
struct q38_lowbit_model {
    gguf_context *gguf;
    q38_lowbit_matrix *matrix;
    void **raw, **original;
    size_t *raw_bytes;
    q38_lowbit_act *act;
    size_t act_capacity;
    uint64_t source_identity;
    const float *active_x;
    int active_cols;
    struct q38_lowbit_model *next;
};
static q38_lowbit_model *models;

static int prepare(q38_lowbit_act *a, size_t count, const float *x, int cols, int arithmetic) {
#ifdef __ARM_FEATURE_SVE
    return q38_lowbit_prepare_sve(a, count, x, cols, arithmetic);
#else
    return q38_lowbit_prepare(a, count, x, cols, arithmetic);
#endif
}

static size_t round_page(size_t n) { return (n + Q38_PAGE - 1) & ~(size_t)(Q38_PAGE - 1); }

static size_t available_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
    fclose(f);
    return kb <= SIZE_MAX / 1024 ? (size_t)kb * 1024 : 0;
}

static int read_source(const gguf_context *g, int i, void *out, size_t n, size_t off) {
    int fd = g->tensor_fds ? g->tensor_fds[i] : g->fd;
    uint64_t base = g->tensor_file_offsets ? g->tensor_file_offsets[i] :
        g->data_offset + g->tensors[i].offset;
    if (base > INT64_MAX || off > (uint64_t)INT64_MAX - base ||
        n > (uint64_t)INT64_MAX - base - off) return 0;
    size_t done = 0;
    while (done < n) {
        ssize_t got = pread(fd, (char *)out + done, n - done, (off_t)(base + off + done));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return 0;
        done += (size_t)got;
    }
    /* Advise whole consumed pages, never evict the next unread partial page. */
    uint64_t first = (base + off) & ~UINT64_C(4095);
    uint64_t last = (base + off + n) & ~UINT64_C(4095);
    if (last > first) posix_fadvise(fd, (off_t)first, (off_t)(last - first), POSIX_FADV_DONTNEED);
    return 1;
}

static int bind_part(void *p, size_t n, int cmg) {
    cpu_set_t set;
    CPU_ZERO(&set); CPU_SET(12 + cmg * 12, &set);
    unsigned long nodes = 1ul << (4 + cmg);
    /* Fugaku rejects maxnode=8 when bit 7 is set. Pass the complete mask
     * width, as in the native benchmark (verified on allocation 51893515). */
    if (sched_setaffinity(0, sizeof(set), &set) ||
        syscall(SYS_mbind, p, n, MPOL_BIND, &nodes, 64UL, 0UL)) {
        perror("lowbit placement"); return 0;
    }
    return 1;
}

static void *alloc_part(size_t bytes, int cmg, int numa) {
    size_t n = round_page(bytes);
    if (!n) return NULL;
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    if (numa && !bind_part(p, n, cmg)) { munmap(p, n); return NULL; }
    return p;
}

static int verify_part(void *p, size_t bytes, int cmg, int numa) {
    if (!numa) return 1;
    for (size_t off = 0; off < bytes; off += Q38_PAGE) {
        int node = -1;
        if (syscall(SYS_get_mempolicy, &node, NULL, 0, (char *)p + off,
                    MPOL_F_NODE | MPOL_F_ADDR) || node != cmg + 4) {
            fprintf(stderr, "lowbit placement: wanted node %d, got %d\n", cmg + 4, node);
            return 0;
        }
    }
    return 1;
}

static int eligible(const gguf_tensor_info *t, int format) {
    if (t->n_dims != 2 || t->dims[0] < 64 || t->dims[0] > 65536 ||
        !t->dims[1] || t->dims[1] > INT_MAX || strstr(t->name.str, "conv1d")) return 0;
    return format == Q38_LB_NVFP4 ? t->type == GGML_TYPE_NVFP4 && !(t->dims[0] % 64) :
        t->type == GGML_TYPE_BF16;
}

/* Unused auxiliary branches stay lazy; the serial text runner rejects MTP.
 * These GGUF pointers must not be used after their source is closed. */
static int auxiliary(const char *name) {
    return !strncmp(name, "nextn.", 6) || !strncmp(name, "v.", 2) ||
           !strncmp(name, "mm.", 3) || !strncmp(name, "mtp.", 4);
}

#include "qwen38_lowbit_image.inc"

static q38_lowbit_model *load_model(gguf_context *g, int format,
    int arithmetic, int numa, size_t reserve_bytes, const char *image_path) {
    if (!g || !g->use_mmap || !g->n_tensors || g->n_tensors > 100000 ||
        (format != Q38_LB_NVFP4 && format != Q38_LB_FP6_E2M3) ||
        (arithmetic != 0 && arithmetic != 8 && arithmetic != 16)) return NULL;
    q38_lowbit_model *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->gguf = g;
    int image_fd = -1;
    q38_image_record *records = NULL;
    if (!image_identity(g, &m->source_identity)) goto fail;
    size_t count = (size_t)g->n_tensors, total = 0;
    m->matrix = calloc(count, sizeof(*m->matrix));
    m->raw = calloc(count, sizeof(*m->raw));
    m->raw_bytes = calloc(count, sizeof(*m->raw_bytes));
    m->original = calloc(count, sizeof(*m->original));
    if (!m->matrix || !m->raw || !m->raw_bytes || !m->original) goto fail;
    for (size_t i = 0; i < count; i++) m->original[i] = gguf_tensor_data(g, (int)i);
    /* First compute the complete final footprint. Never start an unsafe load. */
    int converted = 0;
    for (size_t i = 0; i < count; i++) {
        const gguf_tensor_info *t = &g->tensors[i];
        if (auxiliary(t->name.str)) continue;
        size_t bytes = 0;
        if (eligible(t, format)) {
            q38_lowbit_matrix *mat = &m->matrix[i];
            mat->rows = (int)t->dims[1]; mat->cols = (int)t->dims[0];
            mat->format = format; mat->arithmetic = arithmetic; mat->owner = m;
            int groups = (mat->rows + 7) / 8;
            for (int c = 0; c <= 4; c++) {
                int row = (int)((int64_t)groups * c / 4) * 8;
                mat->first[c] = row < mat->rows ? row : mat->rows;
            }
            for (int c = 0; c < 4; c++)
                bytes += round_page(q38_lowbit_bytes(format, mat->first[c+1] - mat->first[c], mat->cols));
            converted++;
        } else {
            uint64_t n = gguf_tensor_size(g, (int)i);
            if (!n || n > SIZE_MAX - Q38_PAGE) goto fail;
            m->raw_bytes[i] = (size_t)n;
            bytes = round_page((size_t)n);
        }
        if (bytes > SIZE_MAX - total) goto fail;
        total += bytes;
    }
    if (image_path) {
        records = image_records(m, format);
        if (!records || (image_fd = image_open(g, format, image_path, records)) < 0) goto fail;
    }
    size_t available = available_bytes();
    if (!converted || available < reserve_bytes || total > available - reserve_bytes) {
        fprintf(stderr, "lowbit: refusing load: final=%.3fGB available=%.3fGB reserve=%.3fGB converted=%d\n",
            total / 1e9, available / 1e9, reserve_bytes / 1e9, converted);
        goto fail;
    }
    fprintf(stderr, "lowbit: final anonymous allocation %.3fGB, converted=%d\n", total / 1e9, converted);
    if (!g->tensor_data) {
        g->tensor_data = malloc(count * sizeof(*g->tensor_data));
        if (!g->tensor_data) goto fail;
        memcpy(g->tensor_data, m->original, count * sizeof(*g->tensor_data));
    }
    cpu_set_t saved;
    if (sched_getaffinity(0, sizeof(saved), &saved)) goto fail;
    int ok = 1;
    /* Amortize shared-filesystem latency without a second model allocation.
     * A source window is at most 8 MiB; BF16 conversion adds at most 16 MiB. */
    size_t source_window = 8u * 1024u * 1024u;
    uint16_t *input = malloc(source_window);
    float *floats = format == Q38_LB_FP6_E2M3 ? malloc(source_window * 2) : NULL;
    if (!input || (format == Q38_LB_FP6_E2M3 && !floats)) ok = 0;
    for (size_t i = 0; ok && i < count; i++) {
        q38_lowbit_matrix *mat = &m->matrix[i];
        if (mat->format) {
            size_t rb = format == Q38_LB_NVFP4 ? (size_t)mat->cols / 64 * 36 : (size_t)mat->cols * 2;
            int batch = (int)(source_window / rb) / 8 * 8;
            if (batch > 512) batch = 512;
            for (int c = 0; ok && c < 4; c++) {
                int rows = mat->first[c+1] - mat->first[c];
                if (!rows) continue;
                size_t bytes = q38_lowbit_bytes(format, rows, mat->cols);
                mat->part[c] = alloc_part(bytes, c, numa);
                if (!mat->part[c]) { ok = 0; break; }
                size_t group_bytes = q38_lowbit_bytes(format, 8, mat->cols);
                if (image_fd >= 0) ok = image_read_part(image_fd, &records[i], c, mat->part[c]);
                for (int row = 0; image_fd < 0 && ok && row < rows; row += batch) {
                    int n = rows - row < batch ? rows - row : batch;
                    ok = read_source(g, (int)i, input, (size_t)n * rb,
                                     (size_t)(mat->first[c] + row) * rb);
                    void *out = (char *)mat->part[c] + (size_t)(row / 8) * group_bytes;
                    size_t output_bytes = q38_lowbit_bytes(format, n, mat->cols);
                    if (ok && format == Q38_LB_NVFP4)
                        ok = q38_lowbit_pack_nvfp4(out, output_bytes, input, rb, n, mat->cols);
                    else if (ok) {
                        for (size_t j = 0; j < (size_t)n * mat->cols; j++) {
                            uint32_t bits = (uint32_t)input[j] << 16;
                            memcpy(floats + j, &bits, 4);
                        }
                        ok = q38_lowbit_pack_fp6(out, output_bytes, floats, mat->cols, n, mat->cols);
                    }
                }
                if (ok) ok = verify_part(mat->part[c], bytes, c, numa);
            }
        } else if (m->raw_bytes[i]) {
            size_t bytes = m->raw_bytes[i];
            m->raw[i] = alloc_part(bytes, 0, 0);
            if (!m->raw[i]) { ok = 0; break; }
            size_t pages = round_page(bytes) / Q38_PAGE;
            for (int c = 0; ok && c < 4; c++) {
                size_t first = pages * (size_t)c / 4 * Q38_PAGE;
                size_t last = pages * (size_t)(c + 1) / 4 * Q38_PAGE;
                if (first == last || first >= bytes) continue;
                char *part = (char *)m->raw[i] + first;
                if (numa && !bind_part(part, last - first, c)) { ok = 0; break; }
                if (last > bytes) last = bytes;
                for (size_t off = first; ok && off < last; ) {
                    size_t n = last - off < Q38_PAGE ? last - off : Q38_PAGE;
                    ok = image_fd >= 0 ? image_read(image_fd, (char *)m->raw[i] + off,
                        n, records[i].offset[0] + off) :
                        read_source(g, (int)i, (char *)m->raw[i] + off, n, off);
                    off += n;
                }
                if (ok) ok = verify_part(part, last - first, c, numa);
            }
            if (ok && image_fd >= 0)
                ok = image_hash(m->raw[i], bytes) == records[i].hash[0];
            if (ok) g->tensor_data[i] = m->raw[i];
        }
        if (available_bytes() < reserve_bytes) {
            fprintf(stderr, "lowbit: memory reserve crossed during load\n"); ok = 0;
        }
        if (!ok) fprintf(stderr, "lowbit: load failed at %s\n", g->tensors[i].name.str);
        else if (i % 64 == 63) fprintf(stderr, "lowbit: loaded %zu/%zu tensors\n", i+1, count);
    }
    free(input); free(floats);
    if (sched_setaffinity(0, sizeof(saved), &saved)) ok = 0;
    uint64_t final_identity;
    if (!image_identity(g, &final_identity) || final_identity != m->source_identity) ok = 0;
    if (!ok) goto fail;
    if (image_fd >= 0) close(image_fd);
    free(records);
    m->next = models; models = m;
    return m;
fail:
    if (image_fd >= 0) close(image_fd);
    free(records);
    q38_lowbit_model_free(m);
    return NULL;
}

q38_lowbit_model *q38_lowbit_model_load(gguf_context *g, int format,
    int arithmetic, int numa, size_t reserve_bytes) {
    return load_model(g, format, arithmetic, numa, reserve_bytes, NULL);
}

q38_lowbit_model *q38_lowbit_model_load_image(gguf_context *g, int format,
    int arithmetic, int numa, size_t reserve_bytes, const char *path) {
    if (!path || !*path) return NULL;
    return load_model(g, format, arithmetic, numa, reserve_bytes, path);
}

void q38_lowbit_model_free(q38_lowbit_model *m) {
    if (!m) return;
    q38_lowbit_model **p = &models;
    while (*p && *p != m) p = &(*p)->next;
    if (*p) *p = m->next;
    for (size_t i = 0; i < m->gguf->n_tensors; i++) {
        if (m->original && m->original[i] && m->gguf->tensor_data)
            m->gguf->tensor_data[i] = m->original[i];
        if (m->raw && m->raw[i]) munmap(m->raw[i], round_page(m->raw_bytes[i]));
        if (m->matrix) for (int c = 0; c < 4; c++) {
            q38_lowbit_matrix *mat = &m->matrix[i];
            if (mat->part[c]) munmap(mat->part[c], round_page(q38_lowbit_bytes(
                mat->format, mat->first[c+1] - mat->first[c], mat->cols)));
        }
    }
    free(m->raw); free(m->raw_bytes); free(m->original); free(m->matrix); free(m->act); free(m);
}

const q38_lowbit_matrix *q38_lowbit_model_tensor(const gguf_context *g, int index) {
    if (index < 0 || (uint64_t)index >= g->n_tensors) return NULL;
    for (q38_lowbit_model *m = models; m; m = m->next)
        if (m->gguf == g) return m->matrix[index].format ? &m->matrix[index] : NULL;
    return NULL;
}

int q38_lowbit_matrix_row(float *dst, const q38_lowbit_matrix *m, int row) {
    if (!m || row < 0 || row >= m->rows) return 0;
    for (int c = 0; c < 4; c++) if (row >= m->first[c] && row < m->first[c+1])
        return q38_lowbit_dequant_row(dst, m->part[c], m->format,
            m->first[c+1] - m->first[c], m->cols, row - m->first[c]);
    return 0;
}

int q38_lowbit_matrix_begin(const q38_lowbit_matrix *mat, const float *x) {
    if (!mat || !mat->arithmetic) return 1;
    q38_lowbit_model *m = mat->owner;
    if (m->active_x) return 0; /* Never reuse a previous or nested dispatch. */
    size_t count = ((size_t)mat->cols + 15) / 16;
    if (m->act_capacity < count) {
        q38_lowbit_act *p = realloc(m->act, count * sizeof(*p));
        if (!p) return 0;
        m->act = p; m->act_capacity = count;
    }
    if (!prepare(m->act, count, x, mat->cols, mat->arithmetic)) return 0;
    m->active_x = x; m->active_cols = mat->cols;
    return 1;
}

void q38_lowbit_matrix_end(const q38_lowbit_matrix *mat) {
    if (mat) { mat->owner->active_x = NULL; mat->owner->active_cols = 0; }
}

int q38_lowbit_matrix_rows(float *dst, const q38_lowbit_matrix *mat,
    const float *x, int first, int last) {
    if (!dst || !mat || !x || first < 0 || last < first || last > mat->rows) return 0;
    q38_lowbit_act *owned = NULL;
    const q38_lowbit_act *act = NULL;
    if (mat->arithmetic) {
        q38_lowbit_model *m = mat->owner;
        if (m->active_x == x && m->active_cols == mat->cols) act = m->act;
        else {
            size_t count = ((size_t)mat->cols + 15) / 16;
            owned = malloc(count * sizeof(*owned));
            if (!owned || !prepare(owned, count, x, mat->cols, mat->arithmetic)) {
                free(owned); return 0;
            }
            act = owned;
        }
    }
    int ok = 1;
    size_t gb = q38_lowbit_bytes(mat->format, 8, mat->cols);
    for (int c = 0; ok && c < 4; c++) {
        int lo = first > mat->first[c] ? first : mat->first[c];
        int hi = last < mat->first[c+1] ? last : mat->first[c+1];
        for (int row = lo; ok && row < hi; ) {
            int base = (row - mat->first[c]) / 8 * 8 + mat->first[c];
            /* Stream all complete groups in one kernel call; only the two
             * boundary groups need a temporary output for arbitrary slices. */
            int whole = (hi - row) / 8 * 8;
            if (row == base && whole) {
                const void *w = (const char *)mat->part[c] + (size_t)((base - mat->first[c]) / 8) * gb;
#ifdef __ARM_FEATURE_SVE
                if (!mat->arithmetic) ok = q38_lowbit_sve_f32(dst + row, w, mat->format, x, whole, mat->cols);
                else ok = q38_lowbit_sve(dst + row, w, mat->format, act, mat->arithmetic, whole, mat->cols);
#else
                if (!mat->arithmetic) ok = q38_lowbit_reference(dst + row, w, mat->format, x, whole, mat->cols);
                else ok = q38_lowbit_dot(dst + row, w, mat->format, act, mat->arithmetic, whole, mat->cols);
#endif
                row += whole;
                continue;
            }
            int n = mat->first[c+1] - base < 8 ? mat->first[c+1] - base : 8;
            float out[8];
            const void *w = (const char *)mat->part[c] + (size_t)((base - mat->first[c]) / 8) * gb;
#ifdef __ARM_FEATURE_SVE
            if (!mat->arithmetic) ok = q38_lowbit_sve_f32(out, w, mat->format, x, n, mat->cols);
            else ok = q38_lowbit_sve(out, w, mat->format, act, mat->arithmetic, n, mat->cols);
#else
            if (!mat->arithmetic) ok = q38_lowbit_reference(out, w, mat->format, x, n, mat->cols);
            else ok = q38_lowbit_dot(out, w, mat->format, act, mat->arithmetic, n, mat->cols);
#endif
            int end = base + n < hi ? base + n : hi;
            for (; row < end; row++) dst[row] = out[row - base];
        }
    }
    free(owned);
    return ok;
}
