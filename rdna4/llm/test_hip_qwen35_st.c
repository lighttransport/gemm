/* test_hip_qwen35_st.c (RDNA4 port of cuda/llm/test_qwen35_st.c) - drive the DIRECT Qwen3.5 (hybrid DeltaNet+attn) safetensors loader.
 * Runs the text tower on token ids and writes per-layer hidden states
 * (layer_00..layer_24 = HF hidden_states[0..24]; layer_24 is post-final-norm)
 * plus cond.npy (= hidden_states[-3] = output of layer 21, pre-final-norm).
 * See --help, cuda/llm/QWEN35_ST.md and rdna4/llm/QWEN35_ST_HIP.md.
 */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include "hip_qwen35_st.h"

/* minimal .npy reader for int64/int32 1-D/2-D arrays (C order) */
static int32_t *read_npy_ids(const char *path, int *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char hdr[10];
    if (fread(hdr, 1, 10, f) != 10 || memcmp(hdr, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
    int hl = hdr[8] | (hdr[9] << 8);
    char *h = (char *)calloc(hl + 1, 1);
    if (fread(h, 1, hl, f) != (size_t)hl) { fclose(f); free(h); return NULL; }
    int is64 = strstr(h, "<i8") != NULL, is32 = strstr(h, "<i4") != NULL;
    free(h);
    if (!is64 && !is32) { fclose(f); return NULL; }
    long start = ftell(f); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, start, SEEK_SET);
    int cnt = (int)((sz - start) / (is64 ? 8 : 4));
    int32_t *ids = (int32_t *)malloc(cnt * sizeof(int32_t));
    for (int i = 0; i < cnt; i++) {
        if (is64) { int64_t v; if (fread(&v, 8, 1, f) != 1) break; ids[i] = (int32_t)v; }
        else { int32_t v; if (fread(&v, 4, 1, f) != 1) break; ids[i] = v; }
    }
    fclose(f);
    *n = cnt;
    return ids;
}

static int write_npy_f32_2d(const char *path, const float *d, int rows, int cols) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    char hdr[128];
    int n = snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", rows, cols);
    int total = 10 + n + 1;
    int pad = (64 - total % 64) % 64;
    int hl = n + pad + 1;
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fputc(hl & 0xff, f); fputc(hl >> 8, f);
    fwrite(hdr, 1, n, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(d, sizeof(float), (size_t)rows * cols, f);
    fclose(f);
    return 0;
}


static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void usage(const char *a0) {
    fprintf(stderr,
"Usage: %s <model_dir> [options]\n"
"  Qwen3.5-2B text tower (HF safetensors, bf16->F16) -> hidden-state .npy dumps.\n"
"  -i ids.npy     token ids (int32/int64, default tmp/praxis_ref/text/ids.npy)\n"
"  -o out_dir     output dir (default tmp/praxis_out/text_hip)\n"
"  -p N           right-pad ids with <|endoftext|> to a multiple of N (causal => outputs\n"
"                 for real rows are unchanged); padding rows are trimmed from outputs\n"
"  --cond-only    write only cond.npy (hidden_states[-3]); runs just 22 layers\n"
"  --chunk N      feed the prompt in chunks of N tokens (continuation/start_pos>0 path)\n"
"  --max-seq N    max sequence length (default: max(2048, padded length))\n"
"  --serve        persistent mode: load once, then read one request per stdin line:\n"
"                   <ids.npy> <out_dir>            or   - <out_dir> <id> <id> ...\n"
"                 answers 'OK <n_tokens> <ms>' / 'ERR <msg>' on stdout; EOF or 'quit' exits.\n"
"                 (-p/--cond-only/--chunk apply to every request)\n"
"  --repeat N     run the request N times in one process (timing: first = cold, rest = warm)\n"
"  --prof         print load / prefill timing breakdown to stderr\n"
"  -v             verbose runner output\n"
"  -h, --help     this help\n"
"Exit status: 0 ok, 1 usage/input error, 2 load error, 3 run error.\n", a0);
}

static int mk_dir(const char *d) {
    char cmd[1200];
    if (strchr(d, '\'')) return -1;
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", d);
    return system(cmd);
}

typedef struct {
    int pad_to, cond_only, chunk, prof, n_embd, n_layers;
} opts_t;

/* returns 0 ok. ids is consumed (not freed here). */
static int run_one(hip_qwen35 *r, int32_t *ids, int n_real, const char *out_dir, const opts_t *o, double *ms_out) {
    int n_tok = n_real;
    int32_t *pids = ids;
    if (o->pad_to > 1 && n_tok % o->pad_to) {
        int np = (n_tok + o->pad_to - 1) / o->pad_to * o->pad_to;
        pids = (int32_t *)malloc((size_t)np * sizeof(int32_t));
        memcpy(pids, ids, (size_t)n_real * sizeof(int32_t));
        for (int i = n_tok; i < np; i++) pids[i] = 151643;
        n_tok = np;
    }
    int n_embd = o->n_embd, n_layers = o->n_layers, rc = 3;
    size_t per = (size_t)n_tok * n_embd;
    int cond_slot = 22;                       /* hidden_states[-3] of a 24-layer model */
    int n_run = o->cond_only ? cond_slot : n_layers;
    float *all = NULL, *normed = NULL, *chunk_buf = NULL;
    double t0 = now_ms();
    if (o->cond_only) all = (float *)malloc(per * sizeof(float));
    else { all = (float *)malloc((size_t)(n_layers + 1) * per * sizeof(float)); normed = (float *)malloc(per * sizeof(float)); }
    if (!all || (!o->cond_only && !normed)) { fprintf(stderr, "out of memory\n"); goto done; }
    int ch = (o->chunk > 0 && o->chunk < n_tok) ? o->chunk : n_tok;
    if (ch == n_tok) {
        rc = hip_qwen35_prefill_hidden_ex(r, pids, n_tok, n_run, 0, o->cond_only ? cond_slot : -1, all, normed) ? 3 : 0;
    } else {   /* chunked prefill via start_pos continuation; gather per-chunk dumps into [slot][token][embd] */
        size_t cper = (size_t)ch * n_embd;
        chunk_buf = (float *)malloc((size_t)(n_run + 1) * cper * sizeof(float));
        rc = 0;
        for (int s0 = 0; s0 < n_tok && rc == 0; s0 += ch) {
            int n = (n_tok - s0 < ch) ? n_tok - s0 : ch;
            if (hip_qwen35_prefill_hidden_ex(r, pids + s0, n, n_run, s0, o->cond_only ? cond_slot : -1,
                                           chunk_buf, o->cond_only ? NULL : normed + (size_t)s0 * n_embd)) { rc = 3; break; }
            if (o->cond_only) memcpy(all + (size_t)s0 * n_embd, chunk_buf, (size_t)n * n_embd * sizeof(float));
            else for (int k = 0; k <= n_layers; k++)
                memcpy(all + (size_t)k * per + (size_t)s0 * n_embd, chunk_buf + (size_t)k * n * n_embd, (size_t)n * n_embd * sizeof(float));
        }
    }
    if (rc) { fprintf(stderr, "prefill failed\n"); goto done; }
    double t1 = now_ms();
    if (mk_dir(out_dir) != 0) { fprintf(stderr, "cannot create %s\n", out_dir); rc = 1; goto done; }
    char path[1400];
    if (!o->cond_only) {
        for (int k = 0; k < n_layers; k++) {
            snprintf(path, sizeof(path), "%s/layer_%02d.npy", out_dir, k);
            if (write_npy_f32_2d(path, all + (size_t)k * per, n_real, n_embd)) { rc = 1; goto done; }
        }
        snprintf(path, sizeof(path), "%s/layer_%02d.npy", out_dir, n_layers);   /* post-final-norm */
        if (write_npy_f32_2d(path, normed, n_real, n_embd)) { rc = 1; goto done; }
    }
    snprintf(path, sizeof(path), "%s/cond.npy", out_dir);
    if (write_npy_f32_2d(path, o->cond_only ? all : all + (size_t)cond_slot * per, n_real, n_embd)) { rc = 1; goto done; }
    if (o->prof) fprintf(stderr, "[prof] prefill %d tok (%d padded, %d layers): %.1f ms, write %.1f ms\n",
                         n_real, n_tok, n_run, t1 - t0, now_ms() - t1);
    if (ms_out) *ms_out = now_ms() - t0;
    rc = 0;
done:
    free(all); free(normed); free(chunk_buf);
    if (pids != ids) free(pids);
    return rc;
}

int main(int argc, char **argv) {
    const char *model = NULL, *ids_path = "tmp/praxis_ref/text/ids.npy", *out_dir = "tmp/praxis_out/text_hip";
    int verbose = 0, serve = 0, max_seq = 0, repeat = 1;
    opts_t o = { 1, 0, 0, 0, 2048, 24 };
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has = i + 1 < argc;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else if (!strcmp(a, "-p") && has) o.pad_to = atoi(argv[++i]);
        else if (!strcmp(a, "--cond-only")) o.cond_only = 1;
        else if (!strcmp(a, "--chunk") && has) o.chunk = atoi(argv[++i]);
        else if (!strcmp(a, "--max-seq") && has) max_seq = atoi(argv[++i]);
        else if (!strcmp(a, "--serve")) serve = 1;
        else if (!strcmp(a, "--repeat") && has) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--prof")) o.prof = 1;
        else if (!strcmp(a, "-i") && has) ids_path = argv[++i];
        else if (!strcmp(a, "-o") && has) out_dir = argv[++i];
        else if (!strcmp(a, "-v")) verbose = 1;
        else if (a[0] != '-' && !model) model = a;
        else { fprintf(stderr, "unknown or incomplete option: %s\n", a); usage(argv[0]); return 1; }
    }
    if (!model) { usage(argv[0]); return 1; }
    if (access(model, R_OK) != 0) { fprintf(stderr, "model dir not readable: %s\n", model); return 1; }

    int32_t *ids = NULL; int n_tok = 0;
    if (!serve) {
        ids = read_npy_ids(ids_path, &n_tok);
        if (!ids || n_tok <= 0) { fprintf(stderr, "cannot read ids %s\n", ids_path); return 1; }
        if (!max_seq) {
            int np = (o.pad_to > 1) ? (n_tok + o.pad_to - 1) / o.pad_to * o.pad_to : n_tok;
            max_seq = np > 2048 ? np : 2048;
        }
    } else if (!max_seq) max_seq = 8192;

    double t0 = now_ms();
    hip_qwen35 *r = hip_qwen35_init(0, verbose);
    if (!r) { fprintf(stderr, "hip_qwen35_init failed\n"); return 2; }
    double t1 = now_ms();
    if (hip_qwen35_load_safetensors(r, model, max_seq) != 0) { fprintf(stderr, "load failed\n"); return 2; }
    double t2 = now_ms();
    if (o.prof) fprintf(stderr, "[prof] init %.1f ms, load %.1f ms\n", t1 - t0, t2 - t1);

    if (!serve) {
        double ms = 0;
        int rc = 0;
        for (int it = 0; it < repeat && rc == 0; it++) rc = run_one(r, ids, n_tok, out_dir, &o, &ms);
        if (rc == 0) fprintf(stderr, "wrote %s/ (%d tokens, %.1f ms)\n", out_dir, n_tok, ms);
        return rc;
    }
    char line[65536];
    int rc_all = 0;
    while (fgets(line, sizeof(line), stdin)) {
        char *save = NULL, *tok = strtok_r(line, " \t\r\n", &save);
        if (!tok) continue;
        if (!strcmp(tok, "quit")) break;
        char *od = NULL; int32_t *rid = NULL; int rn = 0;
        if (!strcmp(tok, "-")) {
            od = strtok_r(NULL, " \t\r\n", &save);
            rid = (int32_t *)malloc(sizeof(int32_t) * 16384);
            char *t;
            while (od && rn < 16384 && (t = strtok_r(NULL, " \t\r\n", &save))) rid[rn++] = atoi(t);
        } else {
            od = strtok_r(NULL, " \t\r\n", &save);
            if (od) rid = read_npy_ids(tok, &rn);
        }
        if (!od || !rid || rn <= 0 || rn + o.pad_to > max_seq) {
            printf("ERR bad request\n"); fflush(stdout); free(rid); continue;
        }
        double ms = 0;
        int rc = run_one(r, rid, rn, od, &o, &ms);
        if (rc == 0) printf("OK %d %.1f\n", rn, ms); else { printf("ERR run rc=%d\n", rc); rc_all = rc; }
        fflush(stdout);
        free(rid);
    }
    return rc_all;
}
