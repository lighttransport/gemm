/*
 * test_hip_praxis.c - driver of the praxis-1 RDNA4 HIP runner, validated against the
 * PyTorch reference dumps of ref/praxis/gen_backbone_ref.py (praxis' tiny policy).
 *
 *   rdna4/praxis/test_hip_praxis --model $PRAXIS_ROOT/tiny --ref tmp/praxis_ref/backbone \
 *        --out tmp/praxis_out/backbone [--stream N] [--selftest-stream]
 *
 * Writes enc_, dec_, ae_ and smp_ .npy files under --out (same names as the reference) and
 * compares the final tensors itself; ref/praxis/compare.py checks every dump.
 *   --stream N          stream routed expert banks through N device ring slots (0: resident)
 *   --selftest-stream   run resident, ring=2, ring=1: outputs must be bit-identical and the
 *                       routed-weight device residency must equal the ring (bounded)
 */
#define SAFETENSORS_IMPLEMENTATION
#define HIP_PRAXIS_IMPLEMENTATION
#include "hip_praxis_runner.h"
#include <errno.h>
#include <time.h>
#include "../../common/praxis_paths.h"

static float *npy_read(const char *path, int *nd, long long shape[8], size_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    unsigned char m[10];
    if (fread(m, 1, 10, f) != 10 || memcmp(m, "\x93NUMPY", 6) || m[6] == 2) { fclose(f); return NULL; }
    int hl = m[8] | (m[9] << 8);
    char *h = (char *)calloc(1, hl + 1);
    if (fread(h, 1, hl, f) != (size_t)hl) { fclose(f); return NULL; }
    if (!strstr(h, "<f4")) { fprintf(stderr, "%s: only <f4 supported\n", path); fclose(f); return NULL; }
    char *p = strchr(strstr(h, "shape"), '(') + 1;
    *nd = 0;
    size_t n = 1;
    while (*p && *p != ')') {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ')') break;
        shape[*nd] = strtoll(p, &p, 10);
        n *= shape[*nd];
        (*nd)++;
    }
    free(h);
    float *d = (float *)malloc(n * 4);
    if (fread(d, 4, n, f) != n) { free(d); d = NULL; }
    fclose(f);
    if (count) *count = n;
    return d;
}
static void npy_write(const char *path, const float *d, int nd, const long long *shape) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    char hdr[256], sh[200] = "";
    size_t n = 1;
    for (int i = 0; i < nd; i++) { char t[32]; snprintf(t, sizeof t, "%lld, ", shape[i]); strcat(sh, t); n *= shape[i]; }
    int hl = snprintf(hdr, sizeof hdr, "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", sh);
    int pad = (64 - (10 + hl + 1) % 64) % 64;
    unsigned short hlen = (unsigned short)(hl + pad + 1);
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hlen, 2, 1, f);
    fwrite(hdr, 1, hl, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(d, 4, n, f);
    fclose(f);
}
static void mkdirs(const char *path) {
    char b[1024];
    snprintf(b, sizeof b, "%s", path);
    for (char *p = b + 1; *p; p++) if (*p == '/') { *p = 0; mkdir(b, 0755); *p = '/'; }
    mkdir(b, 0755);
}

typedef struct {
    float *text, *text_u, *anchor, *noised, *clean, *t, *proprio, *noised_a, *noise_v, *noise_a;
    int S, Su, h, w, K;
    double cfg_v, cfg_a, t_shift; int steps;
} inputs_t;

static float *rd(const char *dir, const char *name, int *nd, long long *sh, size_t *n) {
    char p[1200];
    snprintf(p, sizeof p, "%s/%s.npy", dir, name);
    float *d = npy_read(p, nd, sh, n);
    if (!d) exit(2);
    return d;
}
static void load_inputs(const char *ref, inputs_t *in) {
    int nd; long long sh[8]; size_t n;
    in->text = rd(ref, "in_text", &nd, sh, &n); in->S = (int)sh[1];
    in->text_u = rd(ref, "in_text_u", &nd, sh, &n); in->Su = (int)sh[1];
    in->anchor = rd(ref, "in_anchor", &nd, sh, &n); in->h = (int)sh[3]; in->w = (int)sh[4];
    in->noised = rd(ref, "in_noised", &nd, sh, &n);
    { char pc[1200]; snprintf(pc, sizeof pc, "%s/in_clean.npy", ref); in->clean = access(pc, R_OK) == 0 ? rd(ref, "in_clean", &nd, sh, &n) : NULL; }   /* K'=1 (depth 0): no clean chunk */
    in->t = rd(ref, "in_t_trunk", &nd, sh, &n); in->K = (int)sh[1];
    in->proprio = rd(ref, "in_proprio", &nd, sh, &n);
    in->noised_a = rd(ref, "in_noised_a", &nd, sh, &n);
    in->noise_v = rd(ref, "smp_noise.salt0.0", &nd, sh, &n);
    in->noise_a = rd(ref, "smp_noise.salt1.0", &nd, sh, &n);
    char p[1200];
    snprintf(p, sizeof p, "%s/meta.json", ref);
    FILE *f = fopen(p, "rb");
    in->cfg_v = 4.0; in->cfg_a = 1.0; in->t_shift = 12.5; in->steps = 2;
    if (f) {
        char buf[4096]; size_t m = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[m] = 0;
        json_val *j = json_parse(buf, (int)m);
        json_val *v;
        if ((v = json_obj_get(j, "cfg_video"))) in->cfg_v = v->num;
        if ((v = json_obj_get(j, "cfg_action"))) in->cfg_a = v->num;
        if ((v = json_obj_get(j, "t_shift"))) in->t_shift = v->num;
        if ((v = json_obj_get(j, "num_steps"))) in->steps = (int)v->num;
        json_free(j);
    }
}

typedef struct { float *vel, *ae, *video, *action; size_t nvel, nae, nvid, nact; } outputs_t;

static int run_all(praxis_runner *r, const inputs_t *in, const char *out, outputs_t *o, int do_sample, int verbose) {
    int F = praxis_lat_per_chunk(r), vc = praxis_video_c(r), A = praxis_action_dim(r);
    int K = in->K, h = in->h, w = in->w;
    o->nvel = (size_t)vc * K * F * h * w; o->nae = (size_t)K * F * A; o->nvid = (size_t)vc * F * h * w; o->nact = (size_t)F * A;
    o->vel = (float *)malloc(o->nvel * 4); o->ae = (float *)malloc(o->nae * 4);
    o->video = (float *)malloc(o->nvid * 4); o->action = (float *)malloc(o->nact * 4);
    char p[1200];
    praxis_set_dump_dir(r, out);
    if (praxis_encoder(r, in->text, in->S, in->anchor, h, w, in->noised, in->clean, in->t, K)) return -1;
    if (praxis_decoder(r, in->noised, in->t, o->vel)) return -1;
    if (praxis_expert(r, in->noised_a, in->t, in->proprio, o->ae)) return -1;
    long long s5[5] = { 1, vc, K * F, h, w }, s3[3] = { 1, K * F, A };
    snprintf(p, sizeof p, "%s/dec_vel.npy", out); npy_write(p, o->vel, 5, s5);
    snprintf(p, sizeof p, "%s/ae_out.npy", out); npy_write(p, o->ae, 3, s3);
    if (do_sample) {
        praxis_set_dump_dir(r, NULL);
        praxis_set_sample_dump_dir(r, out);
        if (praxis_sample_window_joint_ex2(r, in->text, in->S, in->text_u, in->Su, in->anchor, h, w, in->clean, in->proprio, in->noise_v, in->noise_a,
                                           K, in->steps, in->t_shift, in->cfg_v, in->cfg_a, o->video, o->action, 0, 0)) return -1;
        long long v5[5] = { 1, vc, F, h, w }, a3[3] = { 1, F, A };
        snprintf(p, sizeof p, "%s/smp_out_video.npy", out); npy_write(p, o->video, 5, v5);
        snprintf(p, sizeof p, "%s/smp_out_action.npy", out); npy_write(p, o->action, 3, a3);
    }
    (void)verbose;
    return 0;
}

static void relcos(const char *name, const float *a, const float *b, size_t n, double *rel, double *cos) {
    double d = 0, na = 0, nb = 0, ab = 0;
    for (size_t i = 0; i < n; i++) { double x = a[i], y = b[i]; d += (x - y) * (x - y); na += x * x; nb += y * y; ab += x * y; }
    *rel = sqrt(d) / (sqrt(na) + 1e-30);
    *cos = ab / (sqrt(na) * sqrt(nb) + 1e-30);
    printf("  %-14s rel_l2=%.3e cos=%.6f\n", name, *rel, *cos);
}

static int check_vs_ref(const char *ref, const outputs_t *o, int do_sample, double tol) {
    int nd; long long sh[8]; size_t n; double rel, cs; int bad = 0;
    struct { const char *n; const float *d; size_t cnt; int on; } t[] = {
        { "dec_vel", o->vel, o->nvel, 1 }, { "ae_out", o->ae, o->nae, 1 },
        { "smp_out_video", o->video, o->nvid, do_sample }, { "smp_out_action", o->action, o->nact, do_sample } };
    for (int i = 0; i < 4; i++) {
        if (!t[i].on) continue;
        float *r = rd(ref, t[i].n, &nd, sh, &n);
        if (n != t[i].cnt) { printf("  %s size mismatch %zu vs %zu\n", t[i].n, n, t[i].cnt); bad++; free(r); continue; }
        relcos(t[i].n, r, t[i].d, n, &rel, &cs);
        if (!(rel <= tol) || !(cs >= 0.999)) bad++;
        free(r);
    }
    return bad;
}

static void print_stats(praxis_runner *r, const char *tag) {
    praxis_stats s;
    praxis_stats_get(r, &s);
    printf("[%s] device bytes now %.2f MB peak %.2f MB | weights %.2f MB ws %.2f MB | routed on device %.2f MB of %.2f MB total"
           " | ring %d x %.2f MB | H2D %.2f MB (%ld loads, %ld prefetch hits, %ld MoE layer runs) | hipMemGetInfo peak use %.2f MB\n",
           tag, s.dev_bytes_now / 1048576.0, s.dev_bytes_peak / 1048576.0, s.weights_bytes / 1048576.0, s.ws_bytes / 1048576.0,
           s.routed_dev_bytes / 1048576.0, s.routed_total_bytes / 1048576.0, s.ring_slots, s.ring_slot_bytes / 1048576.0,
           s.h2d_bytes / 1048576.0, s.n_loads, s.n_prefetch_hits, s.n_moe_layers_run, s.free_min_bytes_used / 1048576.0);
}

int main(int argc, char **argv) {
    const char *model = px_root("tiny"), *ref = "tmp/praxis_ref/backbone", *out = "tmp/praxis_out/backbone";
    int stream = 0, selftest = 0, verbose = 0, no_sample = 0, selftest_smp = 0, repeat = 0;
    double tol = 0.02;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--stream") && i + 1 < argc) stream = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--selftest-stream")) selftest = 1;
        else if (!strcmp(argv[i], "--no-sample")) no_sample = 1;
        else if (!strcmp(argv[i], "--selftest-sampler")) selftest_smp = 1;
        else if (!strcmp(argv[i], "--tol") && i + 1 < argc) tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    mkdirs(out);
    inputs_t in;
    load_inputs(ref, &in);
    praxis_runner *r = praxis_load(model, 0, verbose);
    if (!r) return 1;
    printf("tiny: K=%d S=%d latent %dx%d, %d euler steps, cfg video %.2f action %.2f\n", in.K, in.S, in.h, in.w, in.steps, in.cfg_v, in.cfg_a);
    int bad = 0;
    if (praxis_set_stream(r, stream)) return 1;
    outputs_t o;
    clock_t t0 = clock();
    if (run_all(r, &in, out, &o, !no_sample, verbose)) { fprintf(stderr, "run failed\n"); return 1; }
    printf("run (stream=%d): %.2f s\n", stream, (double)(clock() - t0) / CLOCKS_PER_SEC);
    bad += check_vs_ref(ref, &o, !no_sample, tol);
    print_stats(r, stream ? "stream" : "resident");
    if (repeat > 0 && !no_sample) {
        /* warm per-call latency of the 4-pass sampler (no dumps): plain, skip-last-video, and paired+skip (timing only: both branches use the cond text, equal lengths) */
        int F = praxis_lat_per_chunk(r), vc = praxis_video_c(r), A = praxis_action_dim(r);
        float *v1 = (float *)malloc((size_t)vc * F * in.h * in.w * 4), *a1 = (float *)malloc((size_t)F * A * 4);
        const char *nm[3] = { "plain", "skip_last_video", "paired+skip (timing only)" };
        for (int var = 0; var < 3; var++) {
            double ts[64]; int n = repeat < 64 ? repeat : 64;
            for (int it = -1; it < n; it++) {
                double t1 = pr_now();
                int pair = var == 2, skip = var >= 1;
                if (praxis_sample_window_joint_ex2(r, in.text, in.S, pair ? in.text : in.text_u, pair ? in.S : in.Su, in.anchor, in.h, in.w, in.clean, in.proprio, in.noise_v, in.noise_a,
                                                   in.K, in.steps, in.t_shift, in.cfg_v, in.cfg_a, v1, a1, pair, skip)) return 1;
                if (it >= 0) ts[it] = pr_now() - t1;
            }
            for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (ts[j] < ts[i]) { double t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            printf("latency %-26s warm median of %d: %.3f s\n", nm[var], n, ts[n / 2]);
        }
        print_stats(r, "after-latency");
        const char *pe = getenv("PRAXIS_PROFILE");
        if (pe && *pe && *pe != '0') {   /* one synchronised, phase-profiled plain call */
            praxis_prof_reset(r);
            if (praxis_sample_window_joint_ex2(r, in.text, in.S, in.text_u, in.Su, in.anchor, in.h, in.w, in.clean, in.proprio, in.noise_v, in.noise_a,
                                               in.K, in.steps, in.t_shift, in.cfg_v, in.cfg_a, v1, a1, 0, 0)) return 1;
            praxis_prof_report(r, stdout);
        }
    }
    if (selftest) {
        char out2[1200];
        printf("== expert streaming self-test ==\n");
        for (int ring = 2; ring >= 1; ring--) {
            if (praxis_set_stream(r, ring)) return 1;
            snprintf(out2, sizeof out2, "%s/stream%d", out, ring);
            mkdirs(out2);
            outputs_t s;
            if (run_all(r, &in, out2, &s, !no_sample, verbose)) { fprintf(stderr, "stream run failed\n"); return 1; }
            int same = !memcmp(s.vel, o.vel, o.nvel * 4) && !memcmp(s.ae, o.ae, o.nae * 4) &&
                       (no_sample || (!memcmp(s.video, o.video, o.nvid * 4) && !memcmp(s.action, o.action, o.nact * 4)));
            praxis_stats st;
            praxis_stats_get(r, &st);
            int bounded = st.routed_dev_bytes == (size_t)ring * st.ring_slot_bytes && st.routed_dev_bytes < st.routed_total_bytes;
            printf("ring=%d: outputs bit-identical to resident: %s | routed device residency %.3f MB == %d x slot %.3f MB < total %.3f MB: %s\n",
                   ring, same ? "yes" : "NO", st.routed_dev_bytes / 1048576.0, ring, st.ring_slot_bytes / 1048576.0,
                   st.routed_total_bytes / 1048576.0, bounded ? "yes" : "NO");
            print_stats(r, ring == 2 ? "ring2" : "ring1");
            if (!same || !bounded) bad++;
            bad += check_vs_ref(ref, &s, !no_sample, tol);
        }
    }
    if (selftest_smp && !no_sample) {
        /* sampler shortcuts vs the plain sampler: skip_last_video must keep the actions (it only drops the last decoder passes), the paired
         * encoder runs both guidance branches as one batch (same maths; row-batched GEMMs differ only in rounding) */
        printf("== sampler shortcut self-test ==\n");
        if (praxis_set_stream(r, stream)) return 1;
        int F = praxis_lat_per_chunk(r), vc = praxis_video_c(r), A = praxis_action_dim(r);
        size_t nvid = (size_t)vc * F * in.h * in.w, nact = (size_t)F * A;
        float *v0 = (float *)malloc(nvid * 4), *a0 = (float *)malloc(nact * 4), *v1 = (float *)malloc(nvid * 4), *a1 = (float *)malloc(nact * 4);
        if (praxis_sample_window_joint_ex(r, in.text, in.text_u, in.S, in.anchor, in.h, in.w, in.clean, in.proprio, in.noise_v, in.noise_a, in.K, in.steps, in.t_shift, in.cfg_v, in.cfg_a, v0, a0, 0, 0)) return 1;
        for (int var = 1; var < 4; var++) {
            int pair = var & 1, skip = var >> 1;
            praxis_stats s0, s1; praxis_stats_get(r, &s0);
            if (praxis_sample_window_joint_ex(r, in.text, in.text_u, in.S, in.anchor, in.h, in.w, in.clean, in.proprio, in.noise_v, in.noise_a, in.K, in.steps, in.t_shift, in.cfg_v, in.cfg_a, v1, a1, pair, skip)) return 1;
            praxis_stats_get(r, &s1);
            double rv, cv, ra, ca;
            printf("  pair=%d skip_last_video=%d: MoE layer runs %ld, expert loads %ld\n", pair, skip, s1.n_moe_layers_run - s0.n_moe_layers_run, s1.n_loads - s0.n_loads);
            relcos("action", a0, a1, nact, &ra, &ca);
            if (!skip) relcos("video", v0, v1, nvid, &rv, &cv); else rv = 0;
            double lim = (pair ? 0.03 : 1e-6);
            if (!pair && skip && !(ra <= 1e-6)) { printf("  FAIL: skip_last_video changed the actions\n"); bad++; }
            if (pair && !(ra <= 0.03 && rv <= 0.03)) { printf("  FAIL: paired encoder deviates\n"); bad++; }
            (void)lim;
        }
    }
    if (selftest_smp && !no_sample) {
        /* encoder context cache: cached sampler vs the uncached one (4 steps: 3 cached ones), plain / paired / skip-last-video */
        printf("== encoder context cache self-test (ctx_cache on vs off, 4 Euler steps) ==\n");
        int F = praxis_lat_per_chunk(r), vc = praxis_video_c(r), A = praxis_action_dim(r);
        size_t nvid = (size_t)vc * F * in.h * in.w, nact = (size_t)F * A;
        float *v0 = (float *)malloc(nvid * 4), *a0 = (float *)malloc(nact * 4), *v1 = (float *)malloc(nvid * 4), *a1 = (float *)malloc(nact * 4);
        praxis_set_ctx_cache(r, 0);
        if (praxis_sample_window_joint_ex(r, in.text, in.text_u, in.S, in.anchor, in.h, in.w, in.clean, in.proprio, in.noise_v, in.noise_a, in.K, 4, in.t_shift, in.cfg_v, in.cfg_a, v0, a0, 0, 0)) return 1;
        praxis_set_ctx_cache(r, 1);
        for (int var = 0; var < 4; var++) {
            int pair = var & 1, skip = var >> 1;
            if (praxis_sample_window_joint_ex(r, in.text, in.text_u, in.S, in.anchor, in.h, in.w, in.clean, in.proprio, in.noise_v, in.noise_a, in.K, 4, in.t_shift, in.cfg_v, in.cfg_a, v1, a1, pair, skip)) return 1;
            double rv = 0, cv, ra, ca;
            printf("  ctx_cache pair=%d skip_last_video=%d:\n", pair, skip);
            relcos("action", a0, a1, nact, &ra, &ca);
            if (!skip) relcos("video", v0, v1, nvid, &rv, &cv);
            if (!(ra <= 0.03 && rv <= 0.03)) { printf("  FAIL: ctx cache deviates\n"); bad++; }
        }
        /* and the cached run is repeatable (second call of the same window re-stores the context) */
        if (praxis_sample_window_joint_ex(r, in.text, in.text_u, in.S, in.anchor, in.h, in.w, in.clean, in.proprio, in.noise_v, in.noise_a, in.K, 4, in.t_shift, in.cfg_v, in.cfg_a, v1, a1, 0, 0)) return 1;
        double r1, c1; relcos("action(repeat)", a0, a1, nact, &r1, &c1);
        if (!(r1 <= 0.03)) bad++;
        praxis_set_ctx_cache(r, 0);
    }
    praxis_free(r);
    printf(bad ? "FAILED (%d)\n" : "ALL OK\n", bad);
    return bad ? 1 : 0;
}
