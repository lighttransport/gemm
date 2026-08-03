/* Stream a native K3 rank image into the mixed Q8W16 layout.
 *
 * The converter intentionally works one tensor at a time.  The source image
 * remains on the shared filesystem and only the current BF16 tensor plus its
 * packed destination are resident, which keeps a 12-node interactive job well
 * below the A64FX HBM2 limit.  MXFP4 expert tensors are copied unchanged.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "k3_dense.h"

#define ALIGNMENT 256
#define MAX_NAME 192
#define MAX_LINE 2048
#define MAX_ENTRIES 600000
#define COPY_CHUNK (8u * 1024u * 1024u)

typedef struct {
    uint64_t offset;
    uint64_t nbytes;
    char dtype[16];
    int ndims;
    size_t shape[3];
    char name[MAX_NAME];
} entry;

typedef struct {
    int version;
    int rank;
    int nodes;
    int layer_index;
    uint64_t blob_bytes;
    char mode[64];
} manifest;

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s --input-dir DIR --output-dir DIR --rank R [--nodes N] "
            "[--mode NAME] [--quality-gate] [--force]\n", program);
}

static int parse_header(const char *line, manifest *m) {
    int rank, nodes, layer;
    unsigned long long bytes;
    if (sscanf(line, "# K3FULLV3 mode=%63s rank=%d nodes=%d layer_index=%d "
               "tensors=%*d blob_bytes=%llu", m->mode, &rank, &nodes,
               &layer, &bytes) == 5) {
        m->version = 3;
        m->rank = rank;
        m->nodes = nodes;
        m->layer_index = layer;
        m->blob_bytes = bytes;
        return 1;
    }
    if (sscanf(line, "# K3FULLV2 mode=%63s rank=%d nodes=%d layer_index=%d "
               "tensors=%*d blob_bytes=%llu", m->mode, &rank, &nodes,
               &layer, &bytes) == 5) {
        m->version = 2;
        m->rank = rank;
        m->nodes = nodes;
        m->layer_index = layer;
        m->blob_bytes = bytes;
        return 1;
    }
    return 0;
}

static int load_manifest(const char *path, entry *entries, int *count,
                         manifest *meta) {
    FILE *f = fopen(path, "r");
    if (!f) return errno ? errno : EIO;
    char line[MAX_LINE];
    int n = 0;
    memset(meta, 0, sizeof *meta);
    meta->layer_index = -1;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') {
            parse_header(line, meta);
            continue;
        }
        if (n >= MAX_ENTRIES) {
            fclose(f);
            return E2BIG;
        }
        char *save = NULL;
        char *tok = strtok_r(line, " \t\r\n", &save);
        if (!tok) continue;
        char *end = NULL;
        entries[n].offset = strtoull(tok, &end, 10);
        if (!end || *end) { fclose(f); return EINVAL; }
        tok = strtok_r(NULL, " \t\r\n", &save);
        if (!tok) { fclose(f); return EINVAL; }
        entries[n].nbytes = strtoull(tok, &end, 10);
        if (!end || *end) { fclose(f); return EINVAL; }
        tok = strtok_r(NULL, " \t\r\n", &save);
        if (!tok || strlen(tok) >= sizeof entries[n].dtype) {
            fclose(f); return EINVAL;
        }
        strcpy(entries[n].dtype, tok);
        tok = strtok_r(NULL, " \t\r\n", &save);
        if (!tok) { fclose(f); return EINVAL; }
        entries[n].ndims = (int)strtol(tok, &end, 10);
        if (!end || *end || entries[n].ndims < 0 || entries[n].ndims > 3) {
            fclose(f); return EINVAL;
        }
        for (int d = 0; d < entries[n].ndims; ++d) {
            tok = strtok_r(NULL, " \t\r\n", &save);
            if (!tok) { fclose(f); return EINVAL; }
            entries[n].shape[d] = (size_t)strtoull(tok, &end, 10);
            if (!end || *end) { fclose(f); return EINVAL; }
        }
        tok = strtok_r(NULL, " \t\r\n", &save);
        if (!tok || strlen(tok) >= sizeof entries[n].name) {
            fclose(f); return EINVAL;
        }
        strcpy(entries[n].name, tok);
        ++n;
    }
    if (ferror(f)) {
        fclose(f);
        return EIO;
    }
    fclose(f);
    *count = n;
    return 0;
}

static int read_all_at(int fd, void *dst, size_t bytes, uint64_t offset) {
    unsigned char *p = (unsigned char *)dst;
    size_t done = 0;
    while (done < bytes) {
        size_t want = bytes - done;
        if (want > COPY_CHUNK) want = COPY_CHUNK;
        ssize_t got = pread(fd, p + done, want, (off_t)(offset + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            return errno ? errno : EIO;
        }
        if (got == 0) return EIO;
        done += (size_t)got;
    }
    return 0;
}

/* Keep the fadvise call isolated so this utility remains buildable with the
 * Fujitsu C front end as well as a normal glibc compiler. */
static void drop_source_cache(int fd, uint64_t offset, size_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    (void)posix_fadvise(fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)fd; (void)offset; (void)bytes;
#endif
}

static int copy_all_at(int src, int dst, uint64_t offset, size_t bytes) {
    unsigned char *buf = (unsigned char *)malloc(COPY_CHUNK);
    if (!buf) return ENOMEM;
    size_t done = 0;
    int rc = 0;
    while (done < bytes) {
        size_t want = bytes - done;
        if (want > COPY_CHUNK) want = COPY_CHUNK;
        ssize_t got = pread(src, buf, want, (off_t)(offset + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            rc = errno ? errno : EIO;
            break;
        }
        if ((size_t)got != want) { rc = EIO; break; }
        unsigned char *p = buf;
        size_t written = 0;
        while (written < want) {
            ssize_t n = write(dst, p + written, want - written);
            if (n < 0) {
                if (errno == EINTR) continue;
                rc = errno ? errno : EIO;
                break;
            }
            if (n == 0) { rc = EIO; break; }
            written += (size_t)n;
        }
        if (rc) break;
        drop_source_cache(src, offset + done, want);
        done += want;
    }
    free(buf);
    return rc;
}

static int write_all(int fd, const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = write(fd, p + done, bytes - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno ? errno : EIO;
        }
        if (n == 0) return EIO;
        done += (size_t)n;
    }
    return 0;
}

static int pad_to(int fd, uint64_t *offset) {
    uint64_t aligned = (*offset + ALIGNMENT - 1) & ~(uint64_t)(ALIGNMENT - 1);
    unsigned char zeros[ALIGNMENT] = {0};
    while (*offset < aligned) {
        size_t n = (size_t)(aligned - *offset);
        if (n > sizeof zeros) n = sizeof zeros;
        int rc = write_all(fd, zeros, n);
        if (rc) return rc;
        *offset += n;
    }
    return 0;
}

static int is_q8_projection(const char *name) {
    return strstr(name, "block_sparse_moe.routed_expert_down_proj.weight") ||
           strstr(name, "block_sparse_moe.routed_expert_up_proj.weight") ||
           strstr(name, "block_sparse_moe.shared_experts.gate_proj.weight") ||
           strstr(name, "block_sparse_moe.shared_experts.up_proj.weight") ||
           strstr(name, "block_sparse_moe.shared_experts.down_proj.weight");
}

static int is_q8_router(const char *name) {
    return strstr(name, "block_sparse_moe.gate.weight") != NULL;
}

static int quality_check(const uint16_t *src, const uint8_t *packed,
                         int rows, int cols, const char *name, int group8) {
    float *x = (float *)malloc((size_t)cols * sizeof *x);
    if (!x) return ENOMEM;
    double worst_rel = 0.0, worst_cos = 1.0;
    int blocks = cols / 16;
    for (int trial = 0; trial < 2; ++trial) {
        for (int c = 0; c < cols; ++c) {
            uint32_t z = (uint32_t)(c + 1) * 0x9e3779b9u +
                         (uint32_t)(trial + 11) * 0x85ebca6bu;
            z ^= z >> 16; z *= 0x7feb352du; z ^= z >> 15;
            x[c] = ((float)(z & 0xffffu) / 32768.0f - 1.0f) * 0.125f;
        }
        double se = 0.0, sr = 0.0, dot = 0.0, sq = 0.0;
        for (int r = 0; r < rows; ++r) {
            double ref = 0.0, got = 0.0;
            for (int c = 0; c < cols; ++c) {
                ref += (double)bf16_to_f32_scalar(src[(size_t)r * cols + c]) * x[c];
                size_t group = (size_t)(r / 8) * (size_t)blocks;
                size_t block_bytes = group8 ? 192 : 160;
                const uint8_t *blk = packed +
                    (group + (size_t)(c / 16)) * block_bytes;
                const float *scale = (const float *)blk;
                const int8_t *q = (const int8_t *)(blk + (group8 ? 64 : 32));
                int scale_index = group8 ?
                    (r % 8) * 2 + ((c % 16) >= 8) : r % 8;
                got += (double)q[(r % 8) * 16 + c % 16] *
                       scale[scale_index] * x[c];
            }
            double d = got - ref;
            se += d * d; sr += ref * ref; dot += got * ref; sq += got * got;
        }
        double rel = sqrt(se / (sr + 1e-30));
        double cos = dot / sqrt((sr + 1e-30) * (sq + 1e-30));
        if (rel > worst_rel) worst_rel = rel;
        if (cos < worst_cos) worst_cos = cos;
    }
    free(x);
    printf("CONVERT quality name=%s rel_l2=%.6e cosine=%.8f %s\n",
           name, worst_rel, worst_cos,
           worst_rel < 5e-3 && worst_cos >= .99995 ? "PASS" : "FAIL");
    return worst_rel < 5e-3 && worst_cos >= .99995 ? 0 : ERANGE;
}

static int convert(const char *input_dir, const char *output_dir, int rank,
                   int nodes, const char *mode, int quality_gate, int force) {
    char in_blob[1024], in_manifest[1024], out_blob[1024], out_manifest[1024];
    int n;
    n = snprintf(in_blob, sizeof in_blob, "%s/rank%03d.blob", input_dir, rank);
    if (n < 0 || (size_t)n >= sizeof in_blob) return ENAMETOOLONG;
    n = snprintf(in_manifest, sizeof in_manifest, "%s/rank%03d.manifest",
                 input_dir, rank);
    if (n < 0 || (size_t)n >= sizeof in_manifest) return ENAMETOOLONG;
    n = snprintf(out_blob, sizeof out_blob, "%s/rank%03d.blob", output_dir, rank);
    if (n < 0 || (size_t)n >= sizeof out_blob) return ENAMETOOLONG;
    n = snprintf(out_manifest, sizeof out_manifest, "%s/rank%03d.manifest",
                 output_dir, rank);
    if (n < 0 || (size_t)n >= sizeof out_manifest) return ENAMETOOLONG;

    if (!force && (access(out_blob, F_OK) == 0 || access(out_manifest, F_OK) == 0))
        return EEXIST;
    if (mkdir(output_dir, 0755) && errno != EEXIST) return errno;

    entry *entries = (entry *)calloc(MAX_ENTRIES, sizeof *entries);
    if (!entries) return ENOMEM;
    int count = 0;
    manifest meta;
    int rc = load_manifest(in_manifest, entries, &count, &meta);
    if (rc) { free(entries); return rc; }
    if (meta.version < 2 || meta.rank != rank || meta.nodes != nodes ||
        !strstr(meta.mode, "expert-tp")) {
        fprintf(stderr, "converter: source ownership/mode mismatch rank=%d/%d "
                "nodes=%d/%d mode=%s\n", meta.rank, rank, meta.nodes, nodes,
                meta.mode);
        free(entries);
        return EINVAL;
    }

    int src = open(in_blob, O_RDONLY);
    if (src < 0) { rc = errno ? errno : EIO; free(entries); return rc; }
    char blob_tmp[1100], manifest_tmp[1100];
    snprintf(blob_tmp, sizeof blob_tmp, "%s.tmp.%ld", out_blob, (long)getpid());
    snprintf(manifest_tmp, sizeof manifest_tmp, "%s.tmp.%ld", out_manifest,
             (long)getpid());
    int dst = open(blob_tmp, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (dst < 0) { rc = errno ? errno : EIO; close(src); free(entries); return rc; }

    uint16_t *src_buf = NULL;
    size_t src_cap = 0;
    uint8_t *q_buf = NULL;
    size_t q_cap = 0;
    uint64_t offset = 0;
    int quality_seen[5] = {0, 0, 0, 0, 0};
    entry *out_entries = (entry *)calloc((size_t)count, sizeof *out_entries);
    if (!out_entries) rc = ENOMEM;
    for (int i = 0; !rc && i < count; ++i) {
        entry out = entries[i];
        rc = pad_to(dst, &offset);
        if (rc) break;
        out.offset = offset;
        int router_q8 = is_q8_router(entries[i].name);
        int convert_q8 = (is_q8_projection(entries[i].name) || router_q8) &&
                         !strcmp(entries[i].dtype, "BF16") &&
                         entries[i].ndims == 2;
        if (convert_q8) {
            int rows = (int)entries[i].shape[0];
            int cols = (int)entries[i].shape[1];
            if (rows <= 0 || cols <= 0 || (rows & 7) || (cols & 15)) {
                fprintf(stderr, "converter: non-packable Q8 shape %s [%d,%d]\n",
                        entries[i].name, rows, cols);
                rc = EINVAL;
                break;
            }
            size_t source_bytes = (size_t)rows * (size_t)cols * sizeof(uint16_t);
            size_t q_bytes = router_q8 ? k3_q8pv8_matrix_bytes(rows, cols) :
                                        k3_q8pv16_matrix_bytes(rows, cols);
            if (entries[i].nbytes != source_bytes) {
                fprintf(stderr, "converter: BF16 byte mismatch %s manifest=%llu "
                        "shape=%zu\n", entries[i].name,
                        (unsigned long long)entries[i].nbytes, source_bytes);
                rc = EINVAL;
                break;
            }
            if (source_bytes > src_cap) {
                uint16_t *p = (uint16_t *)realloc(src_buf, source_bytes);
                if (!p) { rc = ENOMEM; break; }
                src_buf = p; src_cap = source_bytes;
            }
            if (q_bytes > q_cap) {
                uint8_t *p = (uint8_t *)realloc(q_buf, q_bytes);
                if (!p) { rc = ENOMEM; break; }
                q_buf = p; q_cap = q_bytes;
            }
            rc = read_all_at(src, src_buf, source_bytes, entries[i].offset);
            if (rc) break;
            drop_source_cache(src, entries[i].offset, source_bytes);
            if (router_q8)
                k3_q8pv8_quantize_bf16(q_buf, src_buf, rows, cols);
            else
                k3_q8pv16_quantize_bf16(q_buf, src_buf, rows, cols);
            if (quality_gate) {
                int kind = router_q8 ? -1 :
                           strstr(entries[i].name, "routed_expert_down") ? 0 :
                           strstr(entries[i].name, "routed_expert_up") ? 1 :
                           strstr(entries[i].name, "shared_experts.gate") ? 2 :
                           strstr(entries[i].name, "shared_experts.up") ? 3 : 4;
                /* Routed matrices are replicated and expensive to recheck;
                 * one sample per projection class gates them.  Shared
                 * matrices are rank-local shards, so gate every shard. */
                if (router_q8 || !quality_seen[kind] || kind >= 2) {
                    int qrc = quality_check(src_buf, q_buf, rows, cols,
                                            entries[i].name, router_q8);
                    if (!router_q8) quality_seen[kind] = 1;
                    if (qrc && qrc != ERANGE) { rc = qrc; break; }
                    if (qrc == ERANGE) {
                        /* Keep this tensor in its source BF16 form.  The
                         * runner dispatches by manifest dtype, so one weak
                         * shared shard does not disable Q8 for the rest. */
                        fprintf(stderr, "CONVERT fallback=BF16 name=%s\n",
                                entries[i].name);
                        rc = write_all(dst, src_buf, source_bytes);
                        out.nbytes = source_bytes;
                        if (!rc) drop_source_cache(dst, out.offset,
                                                   source_bytes);
                        if (rc) break;
                        out_entries[i] = out;
                        offset += out.nbytes;
                        continue;
                    }
                }
            }
            rc = write_all(dst, q_buf, q_bytes);
            out.nbytes = q_bytes;
            if (!rc) drop_source_cache(dst, out.offset, q_bytes);
            strcpy(out.dtype, router_q8 ? "Q8P8" : "Q8P16");
        } else {
            rc = copy_all_at(src, dst, entries[i].offset, (size_t)entries[i].nbytes);
            if (!rc) drop_source_cache(dst, out.offset, (size_t)out.nbytes);
        }
        if (!rc) {
            offset += out.nbytes;
            out_entries[i] = out;
        }
    }
    if (!rc && (!quality_gate || (quality_seen[0] && quality_seen[1] &&
                                  quality_seen[2] && quality_seen[3] &&
                                  quality_seen[4]))) {
        if (fsync(dst)) rc = errno ? errno : EIO;
    } else if (!rc) {
        fprintf(stderr, "converter: quality gate did not see all MoE projection classes\n");
        rc = ERANGE;
    }
    close(src);
    close(dst);
    free(src_buf); free(q_buf);
    if (!rc) {
        FILE *f = fopen(manifest_tmp, "w");
        if (!f) rc = errno ? errno : EIO;
        if (!rc) {
            fprintf(f, "# K3FULLV3 mode=%s rank=%d nodes=%d layer_index=%d "
                    "tensors=%d blob_bytes=%llu\n", mode, rank, nodes,
                    meta.layer_index, count, (unsigned long long)offset);
            for (int i = 0; i < count; ++i) {
                entry *e = &out_entries[i];
                fprintf(f, "%llu %llu %s %d", (unsigned long long)e->offset,
                        (unsigned long long)e->nbytes, e->dtype, e->ndims);
                for (int d = 0; d < e->ndims; ++d)
                    fprintf(f, " %zu", e->shape[d]);
                fprintf(f, " %s\n", e->name);
            }
            if (fflush(f) || fsync(fileno(f))) rc = errno ? errno : EIO;
            fclose(f);
        }
    }
    if (!rc) {
        if (rename(blob_tmp, out_blob) || rename(manifest_tmp, out_manifest))
            rc = errno ? errno : EIO;
    }
    if (rc) {
        unlink(blob_tmp);
        unlink(manifest_tmp);
    }
    free(out_entries);
    free(entries);
    if (!rc)
        printf("CONVERT PASS rank=%d/%d tensors=%d bytes=%llu output=%s\n",
               rank, nodes, count, (unsigned long long)offset, output_dir);
    return rc;
}

int main(int argc, char **argv) {
    const char *input_dir = NULL, *output_dir = NULL, *mode = "mixed-q8w16-expert-tp";
    int rank = -1, nodes = 96, quality_gate = 0, force = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--input-dir") && ++i < argc) input_dir = argv[i];
        else if (!strcmp(argv[i], "--output-dir") && ++i < argc) output_dir = argv[i];
        else if (!strcmp(argv[i], "--rank") && ++i < argc) rank = atoi(argv[i]);
        else if (!strcmp(argv[i], "--nodes") && ++i < argc) nodes = atoi(argv[i]);
        else if (!strcmp(argv[i], "--mode") && ++i < argc) mode = argv[i];
        else if (!strcmp(argv[i], "--quality-gate")) quality_gate = 1;
        else if (!strcmp(argv[i], "--force")) force = 1;
        else { usage(argv[0]); return 2; }
    }
    if (!input_dir || !output_dir || rank < 0 || rank >= nodes || nodes < 1) {
        usage(argv[0]);
        return 2;
    }
    int rc = convert(input_dir, output_dir, rank, nodes, mode, quality_gate, force);
    if (rc) fprintf(stderr, "k3_full_convert: failed rc=%d (%s) rank=%d\n",
                    rc, strerror(rc), rank);
    return rc ? 1 : 0;
}
