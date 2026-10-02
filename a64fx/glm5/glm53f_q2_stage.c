/* Stream the GLM-5.3-Flash mixed-IQ GGUF routed experts into one rank-local
 * image.  GGUF payloads are never mmap'ed or copied wholesale. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#include "../../common/ggml_dequant.h"
#include <mpi.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <pthread.h>
#ifdef GLM53F_PP_ROUTED_STAGE
#include "glm53f_parallel.h"
#include "glm53f_pp_blob.h"
static glm53f_parallel_config pp_config;
static glm53f_parallel_map pp_map;
static uint64_t pp_source_identity;
#endif

enum { NLAYERS = 45, NEXPERTS = 288, HIDDEN = 4096, INTER = 2048,
#ifdef GLM53F_PP_ROUTED_STAGE
       PARTS = 4, PART_INTER = 512 };
static const int owner_offset[PARTS] = {0,1,2,3};
#else
       PARTS = 8, PART_INTER = 256 };
static const int owner_offset[PARTS] = {0,1,3,4,6,7,9,10};
#endif

typedef struct {
    gguf_context *g;
    int fd;
    uint64_t base;
    const gguf_tensor_info *ti;
} tensor_ref;

static void die(int rank, const char *what) {
    fprintf(stderr, "rank=%d glm53f_q2_stage: %s: %s\n", rank, what,
            errno ? strerror(errno) : "contract failure");
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static int write_all(int fd, const void *ptr, size_t bytes, uint64_t *hash) {
    const uint8_t *p = ptr;
    while (bytes) {
        ssize_t n = write(fd, p, bytes);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (hash) for (ssize_t i = 0; i < n; ++i) {
            *hash ^= p[i]; *hash *= UINT64_C(1099511628211);
        }
        p += n; bytes -= (size_t)n;
    }
    return 0;
}

static int write_payload(int fd, const void *ptr, size_t bytes,
                         uint64_t *hash, uint64_t *entry_hash) {
#ifdef GLM53F_PP_ROUTED_STAGE
    const uint8_t *p = ptr;
    for (size_t i = 0; i < bytes; ++i) {
        *entry_hash ^= p[i]; *entry_hash *= UINT64_C(1099511628211);
    }
#else
    (void)entry_hash;
#endif
    return write_all(fd, ptr, bytes, hash);
}

static void stage_header(char *line, size_t cap, int first, int count, int rank) {
#ifdef GLM53F_PP_ROUTED_STAGE
    snprintf(line, cap, "# GLM53F_PP_ROUTED_V1 layout=pp3-tp4 world_rank=%d "
        "stage=%d tp_rank=%d tp_size=4 cuts=%d,%d layers=%d:%d "
        "expert_parts=4 part_inter=512 source_metadata_fnv1a=%016" PRIx64 "\n",
        rank, pp_map.stage, pp_map.tp_rank, pp_config.cuts[0], pp_config.cuts[1],
        first, first + count, pp_source_identity);
#else
    snprintf(line, cap, "# GLM53F_Q2_DECODE_V1 rank=%d ranks=12 expert_parts=8 "
        "offsets=0,1,3,4,6,7,9,10 layers=%d:%d\n", rank, first, first + count);
#endif
}

static size_t row_bytes(uint32_t type, int columns) {
    if (type >= GGML_TYPE_COUNT || ggml_type_info[type].block_size <= 0 ||
        columns % ggml_type_info[type].block_size) return 0;
    return (size_t)(columns / ggml_type_info[type].block_size) *
           ggml_type_info[type].type_size;
}

static tensor_ref find_tensor(gguf_context *g, const char *name) {
    tensor_ref r = {g, -1, 0, NULL};
    for (uint64_t i = 0; i < g->n_tensors; ++i) if
        (g->tensors[i].name.str && !strcmp(g->tensors[i].name.str, name)) {
        r.ti = &g->tensors[i];
        r.fd = g->tensor_fds ? g->tensor_fds[i] : g->fd;
        r.base = g->tensor_file_offsets ? g->tensor_file_offsets[i]
                                        : g->data_offset + g->tensors[i].offset;
        break;
    }
    return r;
}

static int supported_iq(uint32_t t) {
    return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K ||
           t == GGML_TYPE_IQ2_XS || t == GGML_TYPE_IQ3_XXS ||
           t == GGML_TYPE_IQ4_XS;
}

static int exact_read(const tensor_ref *r, uint64_t relative,
                      void *dst, size_t bytes) {
    uint8_t *p = dst;
    size_t left = bytes;
    while (left) {
        ssize_t n = pread(r->fd, p, left, (off_t)(r->base + relative));
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (!n) { errno = EIO; return -1; }
        p += n; left -= (size_t)n; relative += (uint64_t)n;
    }
    return 0;
}

#ifdef GLM53F_PP_ROUTED_STAGE
static int source_identity(const gguf_context *g, const char *path) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char *p = (const unsigned char *)path; *p; ++p) {
        hash ^= *p; hash *= UINT64_C(1099511628211);
    }
    for (int i = 0; i < g->n_shards; ++i) {
        struct stat st;
        if (fstat(g->shards[i]->fd, &st)) return -1;
        const uint64_t fields[] = {(uint64_t)st.st_size,
            (uint64_t)st.st_mtim.tv_sec, (uint64_t)st.st_mtim.tv_nsec,
            g->shards[i]->n_tensors, (uint64_t)g->shards[i]->data_offset};
        for (size_t j = 0; j < sizeof(fields) / sizeof(fields[0]); ++j)
            for (int byte = 0; byte < 8; ++byte) {
                hash ^= (fields[j] >> (8 * byte)) & 255;
                hash *= UINT64_C(1099511628211);
            }
    }
    if (!g->n_shards) return -1;
    pp_source_identity = hash;
    return 0;
}
#endif

static int owned_part(int expert, int rank) {
#ifdef GLM53F_PP_ROUTED_STAGE
    rank %= 4;
    const int nr = 4;
#else
    const int nr = 12;
#endif
    for (int p = 0; p < PARTS; ++p)
        if ((expert + owner_offset[p]) % nr == rank) return p;
    return -1;
}

static int stage_complete(const char *manifest_path, const char *blob_path,
                          int first_layer, int layer_count, int rank) {
    FILE *f = fopen(manifest_path, "r");
    struct stat st;
    char line[2048];
    uint64_t bytes = UINT64_MAX;
    int header = 0;
#ifdef GLM53F_PP_ROUTED_STAGE
    uint64_t expected_hash = 0; int has_hash = 0;
#endif
    if (!f || stat(blob_path, &st)) { if (f) fclose(f); return 0; }
    while (fgets(line, sizeof(line), f)) {
        unsigned long long got_bytes;
        char expected[512];
        stage_header(expected, sizeof(expected), first_layer, layer_count, rank);
        if (!strcmp(line, expected)) header = 1;
        if (sscanf(line, "# COMPLETE bytes=%llu", &got_bytes) == 1)
            bytes = (uint64_t)got_bytes;
#ifdef GLM53F_PP_ROUTED_STAGE
        unsigned long long got_hash;
        if (sscanf(line, "# COMPLETE bytes=%llu fnv1a=%llx", &got_bytes, &got_hash) == 2) {
            expected_hash = (uint64_t)got_hash; has_hash = 1;
        }
#endif
    }
    fclose(f);
#ifdef GLM53F_PP_ROUTED_STAGE
    return header && bytes == (uint64_t)st.st_size && has_hash &&
        !glm53f_pp_blob_verify(blob_path, bytes, expected_hash);
#else
    return header && bytes == (uint64_t)st.st_size;
#endif
}

static int put_gate_up(int out, FILE *manifest, uint64_t *offset,
                       uint64_t *hash, const tensor_ref *gate,
                       const tensor_ref *up, int layer, int expert, int part,
                       void *buffer, int dry) {
    size_t rb = row_bytes(gate->ti->type, HIDDEN);
    size_t bytes = PART_INTER * rb;
    uint64_t expert_base = (uint64_t)expert * INTER * rb;
    uint64_t begin = *offset, entry_hash = UINT64_C(1469598103934665603);
    if (!rb || gate->ti->type != up->ti->type) return -1;
    if (!dry) {
        if (exact_read(gate, expert_base + (uint64_t)part * PART_INTER * rb,
                       buffer, bytes) || write_payload(out, buffer, bytes, hash, &entry_hash) ||
            exact_read(up, expert_base + (uint64_t)part * PART_INTER * rb,
                       buffer, bytes) || write_payload(out, buffer, bytes, hash, &entry_hash)) return -1;
        posix_fadvise(gate->fd, (off_t)(gate->base + expert_base),
                      (off_t)(INTER * rb), POSIX_FADV_DONTNEED);
        posix_fadvise(up->fd, (off_t)(up->base + expert_base),
                      (off_t)(INTER * rb), POSIX_FADV_DONTNEED);
        fprintf(manifest, "%" PRIu64 " %s 2 %d %d part=%d source_expert=%d "
                "model.language_model.layers.%d.mlp.experts.%d.gate_up_fused.weight\n",
                begin, ggml_type_name(gate->ti->type), 2 * PART_INTER, HIDDEN,
                part, expert, layer, expert);
#ifdef GLM53F_PP_ROUTED_STAGE
        fprintf(manifest, "# PAYLOAD offset=%" PRIu64 " bytes=%zu fnv1a=%016" PRIx64
            " source_gate=%s source_up=%s rows=%d:%d columns=0:%d expert=%d\n",
            begin, 2 * bytes, entry_hash, gate->ti->name.str, up->ti->name.str,
            part * PART_INTER, (part + 1) * PART_INTER, HIDDEN, expert);
#endif
    }
    *offset += 2 * bytes;
    return 0;
}

static int put_down(int out, FILE *manifest, uint64_t *offset, uint64_t *hash,
                    const tensor_ref *down, int layer, int expert, int part,
                    uint8_t *input, uint8_t *output, int dry) {
    size_t full_rb = row_bytes(down->ti->type, INTER);
    size_t part_rb = row_bytes(down->ti->type, PART_INTER);
    const int chunk_rows = 128;
    uint64_t expert_base = (uint64_t)expert * HIDDEN * full_rb;
    uint64_t begin = *offset, entry_hash = UINT64_C(1469598103934665603);
    if (!full_rb || !part_rb || full_rb != PARTS * part_rb) return -1;
    if (!dry) {
        for (int r0 = 0; r0 < HIDDEN; r0 += chunk_rows) {
            int nr = HIDDEN - r0 < chunk_rows ? HIDDEN - r0 : chunk_rows;
            size_t in_bytes = (size_t)nr * full_rb;
            if (exact_read(down, expert_base + (uint64_t)r0 * full_rb,
                           input, in_bytes)) return -1;
            for (int r = 0; r < nr; ++r)
                memcpy(output + (size_t)r * part_rb,
                       input + (size_t)r * full_rb + (size_t)part * part_rb,
                       part_rb);
            if (write_payload(out, output, (size_t)nr * part_rb, hash, &entry_hash)) return -1;
        }
        posix_fadvise(down->fd, (off_t)(down->base + expert_base),
                      (off_t)(HIDDEN * full_rb), POSIX_FADV_DONTNEED);
        fprintf(manifest, "%" PRIu64 " %s 2 %d %d part=%d source_expert=%d "
                "model.language_model.layers.%d.mlp.experts.%d.down_proj.weight\n",
                begin, ggml_type_name(down->ti->type), HIDDEN, PART_INTER,
                part, expert, layer, expert);
#ifdef GLM53F_PP_ROUTED_STAGE
        fprintf(manifest, "# PAYLOAD offset=%" PRIu64 " bytes=%" PRIu64 " fnv1a=%016" PRIx64
            " source_down=%s rows=0:%d columns=%d:%d expert=%d\n",
            begin, (uint64_t)HIDDEN * part_rb, entry_hash, down->ti->name.str,
            HIDDEN, part * PART_INTER, (part + 1) * PART_INTER, expert);
#endif
    }
    *offset += (uint64_t)HIDDEN * part_rb;
    return 0;
}


/* ---- read-ahead pool: the shared filesystem is latency-bound for a single synchronous reader (about 5 MB/s per
 * node observed).  Worker threads read the exact ranges the main loop will consume, staying at most `lead` experts
 * ahead, so the client cache is warm when the main thread arrives.  Data are read and discarded. ---- */
typedef struct {
    const tensor_ref *gate, *up, *down;
    int n, lead, next, consumed;
    int expert[NEXPERTS], part[NEXPERTS];
} prefetch_t;

static void discard_read(const tensor_ref *r, uint64_t relative, size_t bytes, uint8_t *scratch, size_t cap) {
    while (bytes) {
        size_t n = bytes < cap ? bytes : cap;
        ssize_t got = pread(r->fd, scratch, n, (off_t)(r->base + relative));
        if (got <= 0) return;
        relative += (uint64_t)got; bytes -= (size_t)got;
    }
}

static void *prefetch_worker(void *arg) {
    prefetch_t *p = arg;
    const size_t cap = 4u << 20;
    uint8_t *scratch = malloc(cap);
    if (!scratch) return NULL;
    for (;;) {
        int i = __atomic_fetch_add(&p->next, 1, __ATOMIC_RELAXED);
        if (i >= p->n) break;
        while (i > __atomic_load_n(&p->consumed, __ATOMIC_ACQUIRE) + p->lead) usleep(300);
        const int e = p->expert[i], part = p->part[i];
        size_t rb = row_bytes(p->gate->ti->type, HIDDEN);
        uint64_t base = (uint64_t)e * INTER * rb + (uint64_t)part * PART_INTER * rb;
        discard_read(p->gate, base, PART_INTER * rb, scratch, cap);
        discard_read(p->up, base, PART_INTER * rb, scratch, cap);
        size_t full_rb = row_bytes(p->down->ti->type, INTER);
        discard_read(p->down, (uint64_t)e * HIDDEN * full_rb, (size_t)HIDDEN * full_rb, scratch, cap);
    }
    free(scratch);
    return NULL;
}

int main(int argc, char **argv) {
    (void)gguf_type_name; /* Metadata helper is unused by the payload-only loader. */
    int rank, ranks, dry = 0, out = -1, first_layer = 3, layer_count = NLAYERS - 3;
    char blob[4096], manifest_path[4096], blob_tmp[4096], manifest_tmp[4096];
    gguf_context *g = NULL;
    FILE *manifest = NULL;
    uint8_t *buf = NULL, *in = NULL, *packed = NULL;
    uint64_t offset = 0, hash = UINT64_C(1469598103934665603);
    MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
#ifdef GLM53F_PP_ROUTED_STAGE
    pp_config = glm53f_parallel_default(); pp_config.layout = GLM53F_PP3_TP4;
#endif
    for (int a = 3; a < argc; ++a) {
#ifdef GLM53F_PP_ROUTED_STAGE
        int option = glm53f_parallel_option(&pp_config, argc, argv, &a);
        if (option < 0) die(rank, "invalid pipeline option");
        if (option) continue;
#endif
        if (!strcmp(argv[a], "--dry-run")) dry = 1;
#ifndef GLM53F_PP_ROUTED_STAGE
        else if (!strcmp(argv[a], "--first-layer") && a + 1 < argc)
            first_layer = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--layers") && a + 1 < argc)
            layer_count = atoi(argv[++a]);
#endif
        else die(rank, "unknown option");
    }
#ifdef GLM53F_PP_ROUTED_STAGE
    if (pp_config.layout != GLM53F_PP3_TP4 ||
        glm53f_parallel_map_rank(&pp_config, rank, ranks, &pp_map)) die(rank, "PP layout");
    int fields[2] = {pp_config.cuts[0], pp_config.cuts[1]}, low[2], high[2];
    MPI_Allreduce(fields, low, 2, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(fields, high, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (memcmp(low, high, sizeof(low))) die(rank, "inconsistent cuts");
    first_layer = pp_map.first_layer < 3 ? 3 : pp_map.first_layer;
    layer_count = pp_map.end_layer > first_layer ? pp_map.end_layer - first_layer : 0;
#endif
    if (argc < 3 || ranks != 12 || first_layer < 3 || layer_count < 0 ||
#ifndef GLM53F_PP_ROUTED_STAGE
        layer_count == 0 ||
#endif
        first_layer + layer_count > NLAYERS)
        die(rank, "usage: MODEL-00001-of-00004.gguf STAGE_DIR [--dry-run] [--first-layer N] [--layers N]");
    if (mkdir(argv[2], 0755) && errno != EEXIST) die(rank, "mkdir stage");
    snprintf(blob, sizeof(blob), "%s/rank%02d.blob", argv[2], rank);
    snprintf(manifest_path, sizeof(manifest_path), "%s/rank%02d.manifest", argv[2], rank);
    snprintf(blob_tmp, sizeof(blob_tmp), "%s/.rank%02d.blob.tmp.%ld", argv[2], rank, (long)getpid());
    snprintf(manifest_tmp, sizeof(manifest_tmp), "%s/.rank%02d.manifest.tmp.%ld", argv[2], rank, (long)getpid());
#ifdef GLM53F_PP_ROUTED_STAGE
    g = gguf_open_multi(argv[1], 3);
    if (!g || g->n_tensors != 1412 || source_identity(g, argv[1]))
        die(rank, "GGUF metadata identity");
#endif
    if (!dry && stage_complete(manifest_path, blob, first_layer, layer_count, rank)) {
        printf("SENTINEL glm53f_q2_stage=REUSE rank=%d\n", rank);
        gguf_close(g); MPI_Finalize(); return 0;
    }
#ifndef GLM53F_PP_ROUTED_STAGE
    g = gguf_open_multi(argv[1], 3);
#endif
    if (!g || g->n_tensors != 1412) die(rank, "GGUF metadata contract");
    if (!dry) {
        out = open(blob_tmp, O_CREAT | O_EXCL | O_WRONLY, 0644);
        manifest = fopen(manifest_tmp, "wx");
        if (out < 0 || !manifest) die(rank, "create stage output");
        char header[512];
        stage_header(header, sizeof(header), first_layer, layer_count, rank);
        fputs(header, manifest);
    }
    buf = malloc((size_t)PART_INTER * row_bytes(GGML_TYPE_Q6_K, HIDDEN)); in = malloc(1u << 20); packed = malloc(1u << 20);
    if (!buf || !in || !packed) die(rank, "scratch allocation");
    for (int layer = first_layer; layer < first_layer + layer_count; ++layer) {
        char name[128];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer);
        tensor_ref gate = find_tensor(g, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer);
        tensor_ref up = find_tensor(g, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer);
        tensor_ref down = find_tensor(g, name);
        if (rank == 0) {
            fprintf(stderr, "GLM53F_STAGE_TYPES gate=%s/%u[%" PRIu64 ",%" PRIu64 ",%" PRIu64
                    "] up=%s/%u[%" PRIu64 ",%" PRIu64 ",%" PRIu64 "] down=%s/%u[%" PRIu64
                    ",%" PRIu64 ",%" PRIu64 "]\n",
                    gate.ti ? ggml_type_name(gate.ti->type) : "missing",
                    gate.ti ? gate.ti->n_dims : 0,
                    gate.ti ? gate.ti->dims[0] : 0, gate.ti ? gate.ti->dims[1] : 0,
                    gate.ti ? gate.ti->dims[2] : 0,
                    up.ti ? ggml_type_name(up.ti->type) : "missing",
                    up.ti ? up.ti->n_dims : 0,
                    up.ti ? up.ti->dims[0] : 0, up.ti ? up.ti->dims[1] : 0,
                    up.ti ? up.ti->dims[2] : 0,
                    down.ti ? ggml_type_name(down.ti->type) : "missing",
                    down.ti ? down.ti->n_dims : 0,
                    down.ti ? down.ti->dims[0] : 0, down.ti ? down.ti->dims[1] : 0,
                    down.ti ? down.ti->dims[2] : 0);
        }
        if (!gate.ti || !up.ti || !down.ti || !supported_iq(gate.ti->type) ||
            !supported_iq(up.ti->type) || !supported_iq(down.ti->type) ||
            gate.ti->n_dims != 3 || up.ti->n_dims != 3 || down.ti->n_dims != 3 ||
            up.ti->dims[0] != HIDDEN || up.ti->dims[1] != INTER ||
            up.ti->dims[2] != NEXPERTS ||
            gate.ti->dims[0] != HIDDEN || gate.ti->dims[1] != INTER ||
            gate.ti->dims[2] != NEXPERTS || down.ti->dims[0] != INTER ||
            down.ti->dims[1] != HIDDEN || down.ti->dims[2] != NEXPERTS)
            die(rank, "expert tensor contract");
        static prefetch_t pf;
        pthread_t pool[64];
        int npool = 0;
        {
            const char *e = getenv("GLM53F_STAGE_PREFETCH");
            int want = e && *e ? atoi(e) : 16;
            if (want > 64) want = 64;
            pf.gate = &gate; pf.up = &up; pf.down = &down; pf.n = 0; pf.next = 0; pf.consumed = 0;
            pf.lead = getenv("GLM53F_STAGE_LEAD") ? atoi(getenv("GLM53F_STAGE_LEAD")) : 40;
            for (int expert = 0; expert < NEXPERTS; ++expert) {
                int part = owned_part(expert, rank);
                if (part >= 0) { pf.expert[pf.n] = expert; pf.part[pf.n] = part; ++pf.n; }
            }
            if (!dry) for (int t = 0; t < want && t < pf.n; ++t)
                if (!pthread_create(&pool[npool], NULL, prefetch_worker, &pf)) ++npool;
        }
        int done_items = 0;
        for (int expert = 0; expert < NEXPERTS; ++expert) {
            int part = owned_part(expert, rank);
            if (part < 0) continue;
            if (put_gate_up(out, manifest, &offset, &hash, &gate, &up,
                            layer, expert, part, buf, dry) ||
                put_down(out, manifest, &offset, &hash, &down,
                         layer, expert, part, in, packed, dry))
                die(rank, "expert payload");
            __atomic_store_n(&pf.consumed, ++done_items, __ATOMIC_RELEASE);
        }
        __atomic_store_n(&pf.consumed, 1 << 30, __ATOMIC_RELEASE);
        for (int t = 0; t < npool; ++t) pthread_join(pool[t], NULL);
        if (!dry && (layer & 3) == 3) {
            fdatasync(out); posix_fadvise(out, 0, 0, POSIX_FADV_DONTNEED);
        }
        fprintf(stdout, "GLM53F_Q2_STAGE rank=%d layer=%d bytes=%" PRIu64 "\n",
                rank, layer, offset); fflush(stdout);
    }
    if (!dry) {
        fprintf(manifest, "# COMPLETE bytes=%" PRIu64 " fnv1a=%016" PRIx64 "\n", offset, hash);
        if (fflush(manifest) || fsync(fileno(manifest)) || fdatasync(out))
            die(rank, "sync stage");
        fclose(manifest); manifest = NULL; close(out); out = -1;
        if (rename(blob_tmp, blob) || rename(manifest_tmp, manifest_path))
            die(rank, "commit stage");
    }
    printf("SENTINEL glm53f_q2_stage=%s rank=%d bytes=%" PRIu64 " hash=%016" PRIx64 "\n",
           dry ? "DRY_RUN" : "OK", rank, offset, hash);
    free(packed); free(in); free(buf); gguf_close(g); MPI_Finalize(); return 0;
}
