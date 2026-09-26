/* Experimental, little-endian prefill/decode handoff: FP32 version 1,
 * packed INT6 K/V plus FP32 row scales version 2.
 * Global SSM/KV head order on disk, independent of the consumer's TP layout.
 * A fresh directory and a meta file published last make incomplete exports
 * fail closed. Payload hashes detect accidental corruption, not tampering. */
static const char *state_in, *state_out;
static uint64_t state_model_id, state_prompt_id;
static double state_read_sec, state_write_sec;
static int state_first_token;
static int state_prefix_tokens;
static int state_out_i6;
static int state_prompt_tokens(void) { return state_prefix_tokens > 0 ? state_prefix_tokens : JOB.n_prompt; }

typedef struct {
    uint64_t magic, version, model, prompt, pn, layer, fmt, arith;
} q38_state_header;
_Static_assert(sizeof(q38_state_header) == 64, "state header ABI");

static uint64_t state_hash(const void *data, size_t n) {
    const unsigned char *p = data;
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}
static void state_fail(const char *what) {
    fprintf(stderr, "q38d_state: FATAL rank=%d %s (errno=%d)\n", tp_r, what, errno);
#ifdef Q38P_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    exit(1);
}
static void state_io(int fd, void *data, size_t n, off_t off, int writing) {
    unsigned char *p = data;
    while (n) {
        size_t chunk = n < (8u << 20) ? n : (8u << 20);
        ssize_t got = writing ? pwrite(fd, p, chunk, off) : pread(fd, p, chunk, off);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) state_fail(writing ? "write" : "truncated read");
        p += got; n -= got; off += got;
    }
}
static int state_open(const char *dir, int layer, int writing) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s%02d.bin", dir, layer == NLAYER ? "meta" : "layer", layer);
    if (n < 0 || n >= (int)sizeof(path)) state_fail("path too long");
    int fd = open(path, writing ? O_CREAT | O_EXCL | O_WRONLY : O_RDONLY, 0600);
    if (fd < 0) state_fail(path);
    q38_state_header want = {UINT64_C(0x5133385354415445),
                             pf_kv_i6_mode || kv_i6_mode || state_out_i6 ? 2 : 1,
                             state_model_id, state_prompt_id,
                             state_prompt_tokens(), layer, E.fmt, E.arith}, got;
    if (writing) state_io(fd, &want, sizeof(want), 0, 1);
    else {
        state_io(fd, &got, sizeof(got), 0, 0);
        if (memcmp(&want, &got, sizeof(want))) state_fail("incompatible model/prompt/state header");
    }
    return fd;
}
static void state_close(int fd, int writing) {
    if (writing && fsync(fd)) state_fail("fsync");
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (close(fd)) state_fail("close");
}
/* The validated image's inventory includes checksums of every weight payload. */
static void state_identity(const char *image) {
    if (!image) state_fail("state handoff requires --image for weight identity");
    size_t n = 64 + (size_t)G->n_tensors * 128;
    void *p = malloc(n);
    int fd = open(image, O_RDONLY);
    if (!p || fd < 0) state_fail("image identity");
    state_io(fd, p, n, 0, 0);
    state_model_id = state_hash(p, n);
    free(p); close(fd);
    state_prompt_id = state_hash(JOB.tok, (size_t)state_prompt_tokens() * sizeof(int32_t));
}
static void state_record(int fd, off_t off, void *a, size_t na, void *b, size_t nb, int writing) {
    uint64_t hashes[2];
    if (writing) {
        hashes[0] = state_hash(a, na); hashes[1] = state_hash(b, nb);
        state_io(fd, hashes, sizeof(hashes), off, 1);
    } else state_io(fd, hashes, sizeof(hashes), off, 0);
    state_io(fd, a, na, off + 16, writing);
    state_io(fd, b, nb, off + 16 + na, writing);
    if (!writing && (hashes[0] != state_hash(a, na) || hashes[1] != state_hash(b, nb)))
        state_fail("payload checksum mismatch");
    /* Bound dirty page cache during large exports. */
    if (writing && fdatasync(fd)) state_fail("fdatasync");
    posix_fadvise(fd, off, 16 + na + nb, POSIX_FADV_DONTNEED);
}
static void state_export_layer(int layer) {
    int fd = state_open(state_out, layer, 1);
    if (E.L[layer].ssm) {
        size_t ns = (DS * DS + 2 * DS) * sizeof(float), nc = 8 * 384 * sizeof(float);
        for (int h = 0; h < NVH; h++)
            state_record(fd, 64 + h * (16 + ns + nc), ssm_st(layer, h, 0), ns, E.conv_hist[layer][h], nc, 1);
    } else {
        int ai = E.L[layer].ai;
        if (pf_kv_i6_mode || state_out_i6) {
            size_t rows = (size_t)state_prompt_tokens();
            size_t data_bytes = rows * q38d_kv_i6_row_bytes(HD), scale_bytes = rows * sizeof(float);
            size_t record_bytes = 16 + data_bytes + scale_bytes;
            for (int h = 0; h < NKV; h++) {
                uint8_t *kb = pf_kv_i6_mode ? (uint8_t *)KI.k[ai][h][0] : malloc(data_bytes);
                uint8_t *vb = pf_kv_i6_mode ? (uint8_t *)KI.v[ai][h][0] : malloc(data_bytes);
                float *ks = pf_kv_i6_mode ? KI.ks[ai][h][0] : malloc(scale_bytes);
                float *vs = pf_kv_i6_mode ? KI.vs[ai][h][0] : malloc(scale_bytes);
                if (!kb || !vb || !ks || !vs) state_fail("INT6 state export buffer allocation");
                if (!pf_kv_i6_mode)
                    for (size_t t = 0; t < rows; t++)
                        if (q38d_kv_i6_pack_row(E.kcp[ai][h][0] + t * HD,
                                                 kb + t * q38d_kv_i6_row_bytes(HD), HD, ks + t) ||
                            q38d_kv_i6_pack_row(E.vcp[ai][h][0] + t * HD,
                                                 vb + t * q38d_kv_i6_row_bytes(HD), HD, vs + t))
                            state_fail("nonfinite FP32 KV during INT6 state export");
                off_t off = 64 + (off_t)h * 2 * record_bytes;
                state_record(fd, off, kb, data_bytes, ks, scale_bytes, 1);
                state_record(fd, off + record_bytes, vb, data_bytes, vs, scale_bytes, 1);
                if (!pf_kv_i6_mode) { free(kb); free(vb); free(ks); free(vs); }
            }
        } else {
            size_t n = (size_t)state_prompt_tokens() * HD * sizeof(float);
            for (int h = 0; h < NKV; h++)
                state_record(fd, 64 + h * (16 + 2 * n), E.kcp[ai][h][0], n, E.vcp[ai][h][0], n, 1);
        }
    }
    state_close(fd, 1);
}
static void state_export_meta(void) {
    int fd = state_open(state_out, NLAYER, 1);
    uint64_t token = E.next_token;
    state_record(fd, 64, E.x, EMBD * sizeof(float), &token, sizeof(token), 1);
    state_close(fd, 1);
}
static void state_import(void) {
    double t0 = now_sec();
    int fd = state_open(state_in, NLAYER, 0);
    uint64_t token;
    state_record(fd, 64, E.x, EMBD * sizeof(float), &token, sizeof(token), 0);
    if (token >= (uint64_t)E.n_vocab) state_fail("invalid next token");
    state_first_token = (int)token;
    state_close(fd, 0);
    size_t rows = (size_t)state_prompt_tokens();
    size_t n = rows * (kv_i6_mode ? kv_row_bytes() : HD * sizeof(float));
    void *kbuf = malloc(n), *vbuf = malloc(n);
    float *ksbuf = kv_i6_mode ? malloc(rows * sizeof(float)) : NULL;
    float *vsbuf = kv_i6_mode ? malloc(rows * sizeof(float)) : NULL;
    if (!kbuf || !vbuf) state_fail("KV read buffer allocation");
    if (kv_i6_mode && (!ksbuf || !vsbuf)) state_fail("INT6 KV scale read buffer allocation");
    for (int layer = 0; layer < NLAYER; layer++) {
        fd = state_open(state_in, layer, 0);
        if (E.L[layer].ssm) {
            size_t ns = (DS * DS + 2 * DS) * sizeof(float), nc = 8 * 384 * sizeof(float);
            for (int tid = 0; tid < NT; tid++) {
                int h = ssm_head_of(tid);
                if (h >= 0) state_record(fd, 64 + h * (16 + ns + nc), ssm_st(layer, h, 0), ns,
                                         E.conv_hist[layer][h], nc, 0);
            }
        } else {
            int ai = E.L[layer].ai, local_heads = NKV / tp_n, cpk = kv_cpk();
            for (int h = 0; h < local_heads; h++) {
                int global_h = tp_r * local_heads + h;
                if (kv_i6_mode) {
                    size_t scale_bytes = rows * sizeof(float), record_bytes = 16 + n + scale_bytes;
                    off_t off = 64 + (off_t)global_h * 2 * record_bytes;
                    state_record(fd, off, kbuf, n, ksbuf, scale_bytes, 0);
                    state_record(fd, off + record_bytes, vbuf, n, vsbuf, scale_bytes, 0);
                } else state_record(fd, 64 + global_h * (16 + 2 * n), kbuf, n, vbuf, n, 0);
                for (int t = 0; t < state_prompt_tokens(); t += KVB) {
                    int part, u, count = state_prompt_tokens() - t;
                    if (count > KVB) count = KVB;
                    kv_map(t, cpk, &part, &u);
                    if (kv_i6_mode) {
                        size_t row_bytes = kv_row_bytes();
                        memcpy((uint8_t *)KI.k[ai][h][part] + (size_t)u * row_bytes,
                               (uint8_t *)kbuf + (size_t)t * row_bytes, (size_t)count * row_bytes);
                        memcpy((uint8_t *)KI.v[ai][h][part] + (size_t)u * row_bytes,
                               (uint8_t *)vbuf + (size_t)t * row_bytes, (size_t)count * row_bytes);
                        memcpy(KI.ks[ai][h][part] + u, ksbuf + t, (size_t)count * sizeof(float));
                        memcpy(KI.vs[ai][h][part] + u, vsbuf + t, (size_t)count * sizeof(float));
                    } else {
                        memcpy(E.kcp[ai][h][part] + (size_t)u * HD,
                               (float *)kbuf + (size_t)t * HD, (size_t)count * HD * 4);
                        memcpy(E.vcp[ai][h][part] + (size_t)u * HD,
                               (float *)vbuf + (size_t)t * HD, (size_t)count * HD * 4);
                    }
                }
            }
        }
        state_close(fd, 0);
    }
    free(kbuf); free(vbuf); free(ksbuf); free(vsbuf);
    ssq_valid = 0; tp_ssq_ok = 0; tp_prod_ok = 0;
    state_read_sec = now_sec() - t0;
}
