/* Private image v1: little-endian, canonical tensor/CMG order, no pointers.
 * Hashes detect accidental corruption; they are not cryptographic signatures.
 * Source identity includes inventory, shard sizes and nanosecond mtimes. */
typedef struct {
    char magic[8];
    uint64_t version, format, tensors, source, bytes, record_bytes, reserved;
} q38_image_header;
typedef struct {
    uint64_t offset[4], bytes[4], hash[4];
    uint64_t format, rows, cols, reserved;
} q38_image_record;
_Static_assert(sizeof(q38_image_header) == 64, "image header ABI");
_Static_assert(sizeof(q38_image_record) == 128, "image record ABI");

static uint64_t image_mix(uint64_t h, uint64_t word) {
    h ^= word;
    h *= UINT64_C(0x9e3779b185ebca87);
    return (h << 27) | (h >> 37);
}
static uint64_t image_hash(const void *data, size_t n) {
    const unsigned char *p = data;
    uint64_t h = image_mix(UINT64_C(0x514c4231494d4147), n), word;
    while (n >= 8) {
        memcpy(&word, p, 8);
        h = image_mix(h, word);
        p += 8; n -= 8;
    }
    word = 0;
    memcpy(&word, p, n);
    return image_mix(h, word);
}

static int image_identity(const gguf_context *g, uint64_t *identity) {
    uint64_t h = image_mix(UINT64_C(0x514c4231494d4147), g->n_tensors);
    int previous_fd = -1;
    uint64_t file_identity = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        int fd = g->tensor_fds ? g->tensor_fds[i] : g->fd;
        if (fd != previous_fd) {
            struct stat st;
            if (fstat(fd, &st) || st.st_size < 0) return 0;
            file_identity = image_mix((uint64_t)st.st_size, (uint64_t)st.st_mtim.tv_sec);
            file_identity = image_mix(file_identity, (uint64_t)st.st_mtim.tv_nsec);
            previous_fd = fd;
        }
        const gguf_tensor_info *t = &g->tensors[i];
        h = image_mix(h, file_identity);
        h = image_mix(h, image_hash(t->name.str, strlen(t->name.str)));
        h = image_mix(h, t->type);
        h = image_mix(h, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) h = image_mix(h, t->dims[d]);
        h = image_mix(h, g->tensor_file_offsets ? g->tensor_file_offsets[i] :
                      g->data_offset + t->offset);
    }
    *identity = h;
    return 1;
}

static q38_image_record *image_records(const q38_lowbit_model *m, int format) {
    size_t count = (size_t)m->gguf->n_tensors;
    q38_image_record *records = calloc(count, sizeof(*records));
    if (!records) return NULL;
    uint64_t off = sizeof(q38_image_header) + count * sizeof(*records);
    for (size_t i = 0; i < count; i++) {
        q38_image_record *r = records + i;
        const q38_lowbit_matrix *mat = m->matrix + i;
        r->format = mat->format;
        r->rows = mat->rows; r->cols = mat->cols;
        for (int c = 0; c < 4; c++) {
            size_t bytes = mat->format ? q38_lowbit_bytes(format,
                mat->first[c+1] - mat->first[c], mat->cols) : c == 0 ? m->raw_bytes[i] : 0;
            if (!bytes) continue;
            if (bytes > (uint64_t)INT64_MAX - off) { free(records); return NULL; }
            r->offset[c] = off; r->bytes[c] = bytes;
            off += bytes;
        }
    }
    return records;
}

static int image_read(int fd, void *data, size_t n, uint64_t off) {
    size_t done = 0;
    while (done < n) {
        size_t chunk = n - done < 8u*1024u*1024u ? n - done : 8u*1024u*1024u;
        ssize_t got = pread(fd, (char *)data + done, chunk, (off_t)(off + done));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return 0;
        posix_fadvise(fd, (off_t)(off + done), got, POSIX_FADV_DONTNEED);
        done += (size_t)got;
    }
    return 1;
}

static int image_open(const gguf_context *g, int format, const char *path,
                      q38_image_record *expected) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { perror("lowbit image open"); return -1; }
    q38_image_header h;
    uint64_t identity = 0;
    struct stat st;
    uint32_t endian = 1;
    int ok = *(unsigned char *)&endian == 1 && !fstat(fd, &st) &&
        image_read(fd, &h, sizeof(h), 0) && image_identity(g, &identity) &&
        !memcmp(h.magic, "Q38LBIM1", 8) && h.version == 1 && h.format == (uint64_t)format &&
        h.tensors == g->n_tensors && h.source == identity && h.reserved == 0 &&
        h.record_bytes == sizeof(q38_image_record) && h.bytes == (uint64_t)st.st_size;
    uint64_t end = sizeof(h) + g->n_tensors * sizeof(*expected);
    for (uint64_t i = 0; ok && i < g->n_tensors; i++) {
        q38_image_record r;
        ok = image_read(fd, &r, sizeof(r), sizeof(h) + i * sizeof(r));
        if (!ok) break;
        for (int c = 0; c < 4; c++) {
            if (r.offset[c] != expected[i].offset[c] || r.bytes[c] != expected[i].bytes[c] ||
                (!r.bytes[c] && r.hash[c])) ok = 0;
            expected[i].hash[c] = r.hash[c];
            if (r.bytes[c]) end = expected[i].offset[c] + expected[i].bytes[c];
        }
        if (r.format != expected[i].format || r.rows != expected[i].rows ||
            r.cols != expected[i].cols || r.reserved) ok = 0;
    }
    if (ok && end != h.bytes) ok = 0;
    if (!ok) {
        fprintf(stderr, "lowbit image: stale source, incompatible metadata, or truncated image: %s\n", path);
        close(fd); return -1;
    }
    return fd;
}

static int image_read_part(int fd, const q38_image_record *r, int c, void *out) {
    if (!image_read(fd, out, (size_t)r->bytes[c], r->offset[c]) ||
        image_hash(out, (size_t)r->bytes[c]) != r->hash[c]) {
        fprintf(stderr, "lowbit image: payload checksum mismatch\n"); return 0;
    }
    return 1;
}

static int image_write(int fd, const void *data, size_t n, uint64_t off) {
    size_t done = 0;
    while (done < n) {
        size_t chunk = n - done < 8u*1024u*1024u ? n - done : 8u*1024u*1024u;
        size_t written = 0;
        while (written < chunk) {
            ssize_t got = pwrite(fd, (const char *)data + done + written,
                chunk - written, (off_t)(off + done + written));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) return 0;
            written += (size_t)got;
        }
        /* At most 8 MiB dirty per iteration, even with a model resident. */
        if (fdatasync(fd)) return 0;
        posix_fadvise(fd, (off_t)(off + done), (off_t)chunk, POSIX_FADV_DONTNEED);
        done += chunk;
    }
    return 1;
}

int q38_lowbit_model_save_image(const q38_lowbit_model *m, const char *path) {
    if (!m || !path || !*path || m->active_x) return 0;
    int format = 0;
    for (uint64_t i = 0; i < m->gguf->n_tensors; i++)
        if (m->matrix[i].format) { format = m->matrix[i].format; break; }
    q38_image_record *records = image_records(m, format);
    q38_image_header h = {{0}, 1, (uint64_t)format, m->gguf->n_tensors, 0, 0,
                          sizeof(q38_image_record), 0};
    memcpy(h.magic, "Q38LBIM1", 8);
    uint32_t endian = 1;
    if (!records || *(unsigned char *)&endian != 1 || !image_identity(m->gguf, &h.source) || h.source != m->source_identity) {
        free(records); return 0;
    }
    size_t npath = strlen(path) + 32;
    char *temporary = malloc(npath);
    if (!temporary) { free(records); return 0; }
    snprintf(temporary, npath, "%s.partial.XXXXXX", path);
    int fd = mkstemp(temporary), ok = fd >= 0;
    h.bytes = sizeof(h) + h.tensors * sizeof(*records);
    for (uint64_t i = 0; ok && i < h.tensors; i++) for (int c = 0; ok && c < 4; c++) {
        q38_image_record *r = records + i;
        if (!r->bytes[c]) continue;
        const void *data = r->format ? m->matrix[i].part[c] : m->raw[i];
        r->hash[c] = image_hash(data, (size_t)r->bytes[c]);
        ok = image_write(fd, data, (size_t)r->bytes[c], r->offset[c]);
        h.bytes = r->offset[c] + r->bytes[c];
    }
    if (ok) ok = image_write(fd, records, h.tensors * sizeof(*records), sizeof(h)) &&
                 image_write(fd, &h, sizeof(h), 0);
    if (fd >= 0 && close(fd)) ok = 0;
    if (ok) ok = rename(temporary, path) == 0;
    if (!ok) { perror("lowbit image write"); unlink(temporary); }
    free(temporary); free(records);
    return ok;
}
