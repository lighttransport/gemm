#define _GNU_SOURCE
#define GGML_DEQUANT_IMPLEMENTATION
#include "k3_quant.h"
#include "k3_gguf_graph_adapter.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifdef USE_FAPP
#include <fj_tool/fapp.h>
#endif

typedef struct {
    size_t off, nbytes;
    int type, ndims;
    size_t shape[3];
    char name[512];
} entry;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int dtype(const char *s) {
    return k3_quant_type_from_name(s);
}

static int load_manifest(const char *path, entry **out, int *count, size_t *blob_bytes) {
    FILE *f = fopen(path, "r");
    if (!f) return errno ? errno : EIO;
    int cap = 128, n = 0;
    entry *a = calloc((size_t)cap, sizeof(*a));
    char line[2048];
    *blob_bytes = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') {
            unsigned long long b = 0;
            if (sscanf(line, "# K3GGUFV1 %*[^b]blob_bytes=%llu", &b) == 1)
                *blob_bytes = (size_t)b;
            continue;
        }
        if (!line[0] || line[0] == '\n') continue;
        if (n == cap) {
            cap *= 2;
            entry *next = realloc(a, (size_t)cap * sizeof(*a));
            if (!next) { free(a); fclose(f); return ENOMEM; }
            a = next;
        }
        char typ[32];
        char *p = line;
        unsigned long long off, bytes;
        int nd;
        int used = 0;
        if (sscanf(p, "%llu %llu %31s %d %n", &off, &bytes, typ, &nd, &used) != 4 ||
            nd < 1 || nd > 3) { free(a); fclose(f); return EINVAL; }
        p += used;
        a[n].off = (size_t)off; a[n].nbytes = (size_t)bytes;
        a[n].type = dtype(typ); a[n].ndims = nd;
        if (!a[n].type) { free(a); fclose(f); return EINVAL; }
        for (int i = 0; i < nd; ++i) {
            char *end;
            unsigned long long d = strtoull(p, &end, 10);
            if (end == p) { free(a); fclose(f); return EINVAL; }
            a[n].shape[i] = (size_t)d; p = end;
        }
        while (*p == ' ' || *p == '\t') ++p;
        p[strcspn(p, "\r\n")] = 0;
        if (!*p || strlen(p) >= sizeof a[n].name) { free(a); fclose(f); return EINVAL; }
        strcpy(a[n].name, p);
        ++n;
    }
    fclose(f);
    *out = a; *count = n;
    return 0;
}

static int read_blob(const char *path, uint8_t **out, size_t bytes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return errno;
    struct stat st;
    if (fstat(fd, &st) || (size_t)st.st_size < bytes) { close(fd); return EINVAL; }
    uint8_t *p = malloc(bytes);
    if (!p) { close(fd); return ENOMEM; }
    size_t done = 0;
    while (done < bytes) {
        size_t want = bytes - done;
        if (want > 8 * 1024 * 1024) want = 8 * 1024 * 1024;
        ssize_t got = pread(fd, p + done, want, (off_t)done);
        if (got <= 0) { free(p); close(fd); return EIO; }
        done += (size_t)got;
    }
    (void)posix_fadvise(fd, 0, (off_t)bytes, POSIX_FADV_DONTNEED);
    close(fd); *out = p;
    return 0;
}

static int manifest_has(void *opaque, const char *name) {
    const entry *a = (const entry *)opaque;
    /* The small callback context is terminated by an empty name. */
    for (int i = 0; a[i].name[0]; ++i)
        if (!strcmp(a[i].name, name)) return 1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s MANIFEST BLOB [reps]\n", argv[0]);
        return 2;
    }
    int reps = argc == 4 ? atoi(argv[3]) : 3;
    if (reps < 1) reps = 1;
    entry *a = NULL; int n = 0; size_t blob_bytes = 0;
    int rc = load_manifest(argv[1], &a, &n, &blob_bytes);
    if (rc) { fprintf(stderr, "manifest: %s\n", strerror(rc)); return 2; }
    if (!blob_bytes) { fprintf(stderr, "manifest has no blob size\n"); free(a); return 2; }
    int layer_index = -1;
    {
        FILE *hf = fopen(argv[1], "r");
        char line[2048];
        if (hf && fgets(line, sizeof line, hf))
            (void)sscanf(line, "# K3GGUFV1 %*[^ ] %*[^ ] %*[^ ] layer_index=%d",
                         &layer_index);
        if (hf) fclose(hf);
    }
    if (layer_index >= 0 && layer_index < K3_GGUF_LAYERS) {
        /* Make the graph contract executable: every layer benchmark refuses a
         * manifest that cannot feed the corresponding native KDA/MLA graph. */
        entry *sentinel = realloc(a, (size_t)(n + 1) * sizeof(*a));
        if (!sentinel) { free(a); return 3; }
        a = sentinel;
        memset(&a[n], 0, sizeof a[n]);
        int missing = k3_gguf_validate_graph(a, manifest_has, layer_index,
                                              layer_index + 1, stderr);
        if (missing) {
            fprintf(stderr, "k3-gguf adapter: layer=%d missing=%d\n",
                    layer_index, missing);
            free(a);
            return 2;
        }
        printf("K3_GGUF_ADAPTER PASS layer=%d graph=%s\n", layer_index,
               k3_gguf_is_mla(layer_index) ? "MLA" : "KDA");
    }
    uint8_t *blob = NULL;
    rc = read_blob(argv[2], &blob, blob_bytes);
    if (rc) { fprintf(stderr, "blob: %s\n", strerror(rc)); free(a); return 2; }
    double layer_ms = 0.0;
    int measured = 0;
    for (int i = 0; i < n; ++i) {
        const char *only = getenv("K3_ONLY_TENSOR");
        if (only && *only && !strstr(a[i].name, only)) continue;
        if (a[i].ndims < 2) continue;
        int cols = (int)a[i].shape[0];
        size_t rows64 = 1;
        for (int d = 1; d < a[i].ndims; ++d) rows64 *= a[i].shape[d];
        size_t bench_rows64 = rows64;
        int expert_pool = a[i].ndims == 3 && strstr(a[i].name, "exps") != NULL;
        int active_experts = 1;
        int rows_per_expert = (int)rows64;
        int flat_experts = 0;
        if (expert_pool) {
            /* One decode token routes a small expert set, not all 896 experts.
             * Select actual planes; flattening the expert axis would measure
             * a fictitious matrix crossing unrelated expert weights. */
            active_experts = a[i].shape[2] < 8 ? (int)a[i].shape[2] : 8;
            const char *active_env = getenv("K3_ACTIVE_EXPERTS");
            if (active_env && atoi(active_env) > 0 &&
                atoi(active_env) < active_experts)
                active_experts = atoi(active_env);
            rows_per_expert = (int)a[i].shape[1];
            flat_experts = getenv("K3_EXPERT_FLAT") &&
                           atoi(getenv("K3_EXPERT_FLAT")) != 0;
            bench_rows64 = (size_t)rows_per_expert *
                           (size_t)(flat_experts ? active_experts : 1);
        }
        if (rows64 > 1000000 || bench_rows64 > 1000000 ||
            !k3_quant_valid_shape(a[i].type, 1, cols)) continue;
        int rows = (int)bench_rows64;
        int batch = 1;
        const char *batch_env = getenv("K3_PREFILL_BATCH");
        if (batch_env && atoi(batch_env) > 1) batch = atoi(batch_env);
        if (batch > 256) batch = 256;
        float *x = malloc((size_t)batch * cols * sizeof(*x));
        float *y = malloc((size_t)batch * rows * sizeof(*y));
        if (!x || !y) { free(x); free(y); free(blob); free(a); return 3; }
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < cols; ++c)
                x[(size_t)b * cols + c] = 0.001f * (float)(((c + 17 * b) % 127) - 63);
        k3_quant_matrix m = {blob + a[i].off, a[i].type, rows, cols,
                             a[i].nbytes / rows64};
        int threads = omp_get_max_threads();
        int mode = k3_quant_kernel_mode_env();
        int expert_parallel = expert_pool && getenv("K3_EXPERT_PARALLEL") &&
                              atoi(getenv("K3_EXPERT_PARALLEL")) != 0;
        k3_quant_packed packed = {0};
        k3_quant_workspace *pws = calloc((size_t)batch, sizeof(*pws));
        if (!pws) { free(x); free(y); free(blob); free(a); return 3; }
        int use_packed = getenv("K3_QUANT_PACKED") &&
            atoi(getenv("K3_QUANT_PACKED")) != 0 &&
            (a[i].type == K3_Q_IQ1_S || a[i].type == K3_Q_IQ2_XS) &&
            !((expert_pool ? rows_per_expert : rows) & 15);
        int use_nibble = use_packed && getenv("K3_QUANT_PACKED_NIBBLE") &&
                         atoi(getenv("K3_QUANT_PACKED_NIBBLE")) != 0 &&
                         a[i].type == K3_Q_IQ1_S;
        int use_pair = use_packed && getenv("K3_QUANT_PACKED_PAIR") &&
                       atoi(getenv("K3_QUANT_PACKED_PAIR")) != 0 &&
                       (a[i].type == K3_Q_IQ1_S ||
                        a[i].type == K3_Q_IQ2_XS) &&
                       !((expert_pool ? rows_per_expert : rows) & 31);
        k3_quant_packed expert_packed[8] = {{0}};
        k3_quant_matrix expert_m[8];
        int expert_packed_ok = 0;
        if (use_packed && expert_pool && !flat_experts) {
            expert_packed_ok = 1;
            size_t erb = a[i].nbytes / ((size_t)a[i].shape[1] * a[i].shape[2]);
            for (int e = 0; e < active_experts; ++e) {
                expert_m[e] = (k3_quant_matrix){
                    blob + a[i].off + (size_t)e * rows_per_expert * erb,
                    a[i].type, rows_per_expert, cols, erb};
                int prc = use_pair ?
                    (a[i].type == K3_Q_IQ1_S ?
                     k3_quant_pack_iq1_rows32_pair(&expert_packed[e], &expert_m[e]) :
                     k3_quant_pack_iq2_rows32_pair(&expert_packed[e], &expert_m[e])) :
                    (use_nibble && a[i].type == K3_Q_IQ1_S) ?
                    k3_quant_pack_iq_rows16_nibble(&expert_packed[e], &expert_m[e]) :
                    k3_quant_pack_iq_rows16(&expert_packed[e], &expert_m[e]);
                if (prc) { expert_packed_ok = 0; break; }
            }
            for (int b = 0; b < batch && expert_packed_ok; ++b) {
                if (k3_quant_workspace_prepare(&pws[b], cols, K3_QUANT_SVE_Q8))
                    expert_packed_ok = 0;
                else k3_quant_prepare_q8(&pws[b], x + (size_t)b * cols, cols);
            }
            if (!expert_packed_ok) {
                expert_packed_ok = 0;
                use_packed = 0;
            }
        } else if (use_packed) {
            use_packed = !(use_pair ?
                           (a[i].type == K3_Q_IQ1_S ?
                            k3_quant_pack_iq1_rows32_pair(&packed, &m) :
                            k3_quant_pack_iq2_rows32_pair(&packed, &m)) : use_nibble ?
                           k3_quant_pack_iq_rows16_nibble(&packed, &m) :
                           k3_quant_pack_iq_rows16(&packed, &m));
            for (int b = 0; b < batch && use_packed; ++b) {
                if (k3_quant_workspace_prepare(&pws[b], cols, K3_QUANT_SVE_Q8))
                    use_packed = 0;
                else k3_quant_prepare_q8(&pws[b], x + (size_t)b * cols, cols);
            }
        }
        int use_q8_batch = !use_packed && batch > 1 &&
                           a[i].type == K3_Q_Q8_0;
        for (int b = 0; b < batch && use_q8_batch; ++b) {
            if (k3_quant_workspace_prepare(&pws[b], cols, mode)) {
                use_q8_batch = 0;
            } else {
                k3_quant_prepare_a16(&pws[b], x + (size_t)b * cols, cols);
                pws[b].scale_a16 = pws[b].scale;
                pws[b].a16_ready = 1;
            }
        }
        if (use_packed && expert_pool && !flat_experts)
            for (int e = 0; e < active_experts; ++e)
                (void)k3_quant_matvec_packed_batch(y, rows, &expert_m[e],
                                                   &expert_packed[e], pws, batch);
        else if (use_packed)
            (void)k3_quant_matvec_packed_batch(y, rows, &m, &packed, pws, batch);
        else if (use_q8_batch)
            (void)k3_quant_q8_0_matvec_batch(y, rows, &m, pws, batch, threads,
                                              NULL);
        else if (batch == 1 && expert_pool && expert_parallel) {
#pragma omp parallel for schedule(static)
            for (int e = 0; e < active_experts; ++e)
                (void)k3_quant_matvec_expert3d(y, blob + a[i].off, a[i].type,
                                                cols, rows_per_expert,
                                                (int)a[i].shape[2], e, x,
                                                1, mode);
        } else if (expert_pool) {
            for (int b = 0; b < batch; ++b)
                for (int e = 0; e < active_experts; ++e)
                    (void)k3_quant_matvec_expert3d(y + (size_t)b * rows,
                        blob + a[i].off, a[i].type, cols, rows_per_expert,
                        (int)a[i].shape[2], e, x + (size_t)b * cols, threads, mode);
        } else for (int b = 0; b < batch; ++b)
            (void)k3_quant_matvec_mode(y + (size_t)b * rows, &m,
                                       x + (size_t)b * cols, threads, mode);
        double t0 = now_s();
#ifdef USE_FAPP
        fapp_start("k3_iq_decode", 1, 0);
#endif
        for (int r = 0; r < reps; ++r) {
            if (use_packed && expert_pool && !flat_experts)
                for (int e = 0; e < active_experts; ++e)
                    (void)k3_quant_matvec_packed_batch(y, rows, &expert_m[e],
                                                       &expert_packed[e], pws, batch);
            else if (use_packed)
                (void)k3_quant_matvec_packed_batch(y, rows, &m, &packed, pws, batch);
            else if (use_q8_batch)
                (void)k3_quant_q8_0_matvec_batch(y, rows, &m, pws, batch,
                                                  threads, NULL);
            else if (batch == 1 && expert_pool && expert_parallel) {
#pragma omp parallel for schedule(static)
                for (int e = 0; e < active_experts; ++e)
                    (void)k3_quant_matvec_expert3d(y, blob + a[i].off, a[i].type,
                                                    cols, rows_per_expert,
                                                    (int)a[i].shape[2], e, x,
                                                    1, mode);
            } else if (expert_pool) {
                for (int b = 0; b < batch; ++b)
                    for (int e = 0; e < active_experts; ++e)
                        (void)k3_quant_matvec_expert3d(y + (size_t)b * rows,
                            blob + a[i].off, a[i].type, cols, rows_per_expert,
                            (int)a[i].shape[2], e, x + (size_t)b * cols, threads, mode);
            } else for (int b = 0; b < batch; ++b)
                (void)k3_quant_matvec_mode(y + (size_t)b * rows, &m,
                                           x + (size_t)b * cols, threads, mode);
        }
#ifdef USE_FAPP
        fapp_stop("k3_iq_decode", 1, 0);
#endif
        double ms = (now_s() - t0) * 1000.0 / reps;
        layer_ms += ms; ++measured;
        size_t op_rows = (size_t)rows *
            (size_t)(expert_pool && !flat_experts ? active_experts : 1);
        double gflops = 2.0 * (double)op_rows * cols * batch /
                        (ms * 1.0e6);
        printf("K3_QBENCH mode=%s tensor=%s type=%s rows=%d cols=%d experts=%d batch=%d ms=%.3f gflops=%.1f tok/s=%.3f\n",
               use_packed ? (use_pair ? (a[i].type == K3_Q_IQ1_S ?
                                          "packed-iq4-pair-sve-q8" :
                                          "packed-iq2-semantic4-pair-sve-q8") :
                             use_nibble ? "packed-iq4-sve-q8" : "packed-sve-q8") : mode == K3_QUANT_REFERENCE ? "reference" :
               mode == K3_QUANT_SVE_Q8 ? "sve-q8" : "sve-a16",
               a[i].name, k3_quant_type_name(a[i].type), rows, cols,
               expert_pool ? active_experts : 1, batch, ms, gflops,
               ms > 0.0 ? batch * 1000.0 / ms : 0.0);
        free(x); free(y);
        for (int b = 0; b < batch; ++b) k3_quant_workspace_free(&pws[b]);
        free(pws);
        k3_quant_packed_free(&packed);
        for (int e = 0; e < active_experts; ++e)
            k3_quant_packed_free(&expert_packed[e]);
    }
    int report_batch = 1;
    const char *report_env = getenv("K3_PREFILL_BATCH");
    if (report_env && atoi(report_env) > 1) report_batch = atoi(report_env);
    if (report_batch > 256) report_batch = 256;
    printf("K3_QBENCH_LAYER measured=%d batch=%d ms=%.3f quant_projection_tok_s=%.3f blob_mib=%.1f\n",
           measured, report_batch, layer_ms,
           layer_ms > 0.0 ? report_batch * 1000.0 / layer_ms : 0.0,
           (double)blob_bytes / (1024.0 * 1024.0));
    free(blob); free(a);
    return measured ? 0 : 3;
}
