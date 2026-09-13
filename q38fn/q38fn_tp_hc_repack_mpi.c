#define _GNU_SOURCE
#include <mpi.h>
#define Q38FN_TP_BLOB_IMPLEMENTATION
#include "../common/q38fn_tp_blob.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

enum { BUFFER_BYTES = 16 << 20 };

static uint64_t hash_bytes(const void *data, size_t bytes)
{
    const unsigned char *p = data; uint64_t h = UINT64_C(1469598103934665603);
    while (bytes--) { h ^= *p++; h *= UINT64_C(1099511628211); }
    return h;
}

static int read_at(int fd, uint64_t offset, void *data, size_t bytes)
{
    unsigned char *p = data;
    while (bytes) {
        ssize_t n = pread(fd, p, bytes, (off_t)offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += (size_t)n; bytes -= (size_t)n; offset += (uint64_t)n;
    }
    return 0;
}

static int write_at(int fd, uint64_t offset, const void *data, size_t bytes)
{
    const unsigned char *p = data;
    while (bytes) {
        ssize_t n = pwrite(fd, p, bytes, (off_t)offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += (size_t)n; bytes -= (size_t)n; offset += (uint64_t)n;
    }
    return 0;
}

static int layer_mix(const char *name)
{
    return strstr(name, ".layers.") &&
           (q38fn_tp_ends_with(name, ".input_mix_weight_down.weight") ||
            q38fn_tp_ends_with(name, ".input_mix_weight_up.weight"));
}

static void write_entry(FILE *f, const q38fn_tp_blob_entry *e,
                        const q38fn_tp_plan *p, uint64_t offset,
                        uint64_t bytes, uint64_t hash)
{
    fprintf(f, "%llu %llu %016llx %d %d %d",
            (unsigned long long)offset, (unsigned long long)bytes,
            (unsigned long long)hash, (int)p->kind, p->axis, e->ndims);
    for (int d = 0; d < e->ndims; ++d)
        fprintf(f, " %llu", (unsigned long long)e->shape[d]);
    fprintf(f, " %d", p->n_ranges);
    for (int r = 0; r < p->n_ranges; ++r)
        fprintf(f, " %llu %llu", (unsigned long long)p->range[r].start,
                (unsigned long long)p->range[r].count);
    fprintf(f, " %s\n", e->name);
}

int main(int argc, char **argv)
{
    int rank, ranks, rc = 1, fd = -1, tensors = 0;
    char dir[4096], old_blob[8192], old_manifest[8192];
    char new_blob[8192], new_manifest[8192], partial[16384], line[8192];
    FILE *source = NULL, *destination = NULL; unsigned char *buffer = NULL;
    uint64_t original_bytes = 0, append;

    MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 2 || ranks != Q38FN_TP_RANKS) {
        if (!rank) fprintf(stderr, "usage: %s LOCAL_BASE (12 ranks)\n", argv[0]);
        goto done;
    }
    snprintf(dir, sizeof(dir), "%s/rank-%02d", argv[1], rank);
    snprintf(old_blob, sizeof(old_blob), "%s/tp12-v3.blob", dir);
    snprintf(old_manifest, sizeof(old_manifest), "%s/tp12-v3.manifest", dir);
    snprintf(new_blob, sizeof(new_blob), "%s/tp12-v%d.blob", dir, Q38FN_TP_LAYOUT_VERSION);
    snprintf(new_manifest, sizeof(new_manifest), "%s/tp12-v%d.manifest", dir, Q38FN_TP_LAYOUT_VERSION);
    snprintf(partial, sizeof(partial), "%s.partial", new_manifest);
    if (!access(new_blob, F_OK) && !access(new_manifest, F_OK)) { rc = 0; goto done; }
    source = fopen(old_manifest, "r");
    if (!source || !fgets(line, sizeof(line), source)) goto done;
    while (fgets(line, sizeof(line), source)) {
        unsigned long long n;
        if (sscanf(line, "# COMPLETE blob_bytes=%llu", &n) == 1) original_bytes = n;
    }
    fclose(source); source = NULL;
    if (!original_bytes || (access(new_blob, F_OK) && rename(old_blob, new_blob))) goto done;
    fd = open(new_blob, O_RDWR); if (fd < 0 || ftruncate(fd, (off_t)original_bytes)) goto done;
    append = original_bytes; buffer = malloc(BUFFER_BYTES);
    source = fopen(old_manifest, "r"); destination = fopen(partial, "w");
    if (!buffer || !source || !destination || !fgets(line, sizeof(line), source)) goto done;
    fprintf(destination, "# Q38FNTP layout=%d rank=%d ranks=%d layers=0\n",
            Q38FN_TP_LAYOUT_VERSION, rank, ranks);
    while (fgets(line, sizeof(line), source)) {
        char parsed[8192]; q38fn_tp_blob_entry e = {0}; q38fn_tp_plan p;
        if (!strncmp(line, "# COMPLETE", 10)) break;
        memcpy(parsed, line, sizeof(parsed));
        if (q38fn_tp_blob_parse_entry(parsed, &e)) goto done;
        if (!layer_mix(e.name)) fputs(line, destination);
        else {
            uint64_t full = e.shape[0] * e.shape[1] * sizeof(uint16_t);
            uint64_t offset, bytes, hash;
            if (e.kind != Q38FN_TP_FULL || e.ndims != 2 || full != e.bytes ||
                full > BUFFER_BYTES ||
                q38fn_tp_make_plan(e.name, e.shape, e.ndims, rank, ranks, &p)) {
                free(e.name); goto done;
            }
            if (p.kind == Q38FN_TP_AXIS0) {
                bytes = p.range[0].count * e.shape[1] * sizeof(uint16_t);
                offset = e.offset + p.range[0].start * e.shape[1] * sizeof(uint16_t);
                if (read_at(fd, offset, buffer, (size_t)bytes)) { free(e.name); goto done; }
            } else if (p.kind == Q38FN_TP_AXIS1) {
                uint64_t rows = e.shape[0], width = e.shape[1], start = p.range[0].start;
                uint64_t count = p.range[0].count;
                bytes = rows * count * sizeof(uint16_t);
                if (read_at(fd, e.offset, buffer, (size_t)full)) { free(e.name); goto done; }
                uint16_t *v = (uint16_t *)buffer;
                for (uint64_t row = 0; row < rows; ++row)
                    memmove(v + row * count, v + row * width + start,
                            (size_t)count * sizeof(uint16_t));
                append = (append + 255u) & ~UINT64_C(255); offset = append;
                if (write_at(fd, offset, buffer, (size_t)bytes)) { free(e.name); goto done; }
                append += bytes;
            } else { free(e.name); goto done; }
            hash = hash_bytes(buffer, (size_t)bytes);
            write_entry(destination, &e, &p, offset, bytes, hash);
        }
        free(e.name); tensors++;
    }
    if (ftruncate(fd, (off_t)append)) goto done;
    fprintf(destination, "# COMPLETE blob_bytes=%llu tensors=%d\n",
            (unsigned long long)append, tensors);
    if (fflush(destination) || fsync(fileno(destination)) || fdatasync(fd)) goto done;
    fclose(destination); destination = NULL; close(fd); fd = -1;
    if (rename(partial, new_manifest)) goto done;
    fprintf(stderr, "Q38FN_TP_HC_REPACK rank=%d append=%llu tensors=%d\n", rank,
            (unsigned long long)(append - original_bytes), tensors); rc = 0;
done:
    if (source) fclose(source);
    if (destination) fclose(destination);
    if (fd >= 0) close(fd);
    free(buffer);
    if (rc) MPI_Abort(MPI_COMM_WORLD, rc);
    MPI_Finalize(); return rc;
}
