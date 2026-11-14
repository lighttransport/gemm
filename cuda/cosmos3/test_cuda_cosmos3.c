/*
 * test_cuda_cosmos3.c - driver of the Cosmos3-Edge-Policy CUDA transformer runner, validated against ref/cosmos3/gen_ref.py dumps.
 *
 *   cuda/cosmos3/test_cuda_cosmos3 --model $PRAXIS_ROOT/cosmos3-edge-droid --ref tmp/cosmos3_ref --out tmp/cosmos3_out [--teacher-force] [--steps 4] [--guidance 3]
 *
 * Reads token ids, noise latents (`latents_init`, `action_init`) from --ref, runs the und tower once, then either
 *   --teacher-force : every step's transformer input is the ORACLE's (isolates per-step transformer error), or
 *   (default)       : the full UniPC loop with our own latents (end-to-end drift).
 * Writes oracle-named .npy dumps under --out; compare with ref/cosmos3/compare.py.
 */
#define SAFETENSORS_IMPLEMENTATION
#define CUDA_COSMOS3_IMPLEMENTATION
#include "cuda_cosmos3_runner.h"
#include "../../common/praxis_paths.h"

static long long *read_i8(const char *path, int *n) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    unsigned char m[10]; if (fread(m, 1, 10, f) != 10) { fclose(f); return NULL; }
    int hl = m[8] | (m[9] << 8); char *h = (char *)calloc(1, hl + 1); if (fread(h, 1, hl, f) != (size_t)hl) { fclose(f); return NULL; }
    long long cnt = strtoll(strchr(strstr(h, "shape"), '(') + 1, NULL, 10);
    long long *d = (long long *)malloc(8 * (size_t)cnt); if (fread(d, 8, (size_t)cnt, f) != (size_t)cnt) { fclose(f); return NULL; }
    fclose(f); free(h); *n = (int)cnt; return d;
}
static double nowms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }

int main(int argc, char **argv) {
    const char *model = px_root("cosmos3-edge-droid"), *ref = "tmp/cosmos3_ref", *out = "tmp/cosmos3_out";
    int steps = 4, tf = 0, T = 9, H = 11, W = 20, chunk = 32, domain = 8, raw = 10, verbose = 1, repeat = 1, dump_steps = 1, nocfg = 0;
    float guidance = 3.0f, fps = 15.0f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--guidance") && i + 1 < argc) guidance = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--teacher-force")) tf = 1;
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-steps") && i + 1 < argc) dump_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) verbose = 0;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    nocfg = guidance == 1.0f;
    char p[1400];
    double t0 = nowms();
    cosmos3_runner *r = cosmos3_load(model, 0, verbose);
    if (!r) return 1;
    printf("load: %.2f s\n", (nowms() - t0) / 1e3);
    cosmos3_set_dump_dir(r, out); cosmos3_set_dump_steps(r, dump_steps);
    cosmos3_set_geometry(r, T, H, W, chunk, fps, domain, raw);
    int nc = 0, nu = 0; snprintf(p, sizeof p, "%s/input_ids_cond.npy", ref); long long *idc = read_i8(p, &nc);
    snprintf(p, sizeof p, "%s/input_ids_uncond.npy", ref); long long *idu = read_i8(p, &nu);
    if (!idc || !idu) return 1;
    if (cosmos3_set_prompt(r, 0, idc, nc) || (!nocfg && cosmos3_set_prompt(r, 1, idu, nu))) return 1;
    long long nlat = 0, nact = 0;
    snprintf(p, sizeof p, "%s/latents_init.npy", ref); float *lat0 = cosmos3_npy_read(p, &nlat);
    snprintf(p, sizeof p, "%s/action_init.npy", ref); float *act0 = cosmos3_npy_read(p, &nact);
    if (!lat0 || !act0) return 1;
    if (nlat != 48LL * T * H * W || nact != chunk * 64) { fprintf(stderr, "latent/action size mismatch (%lld %lld)\n", nlat, nact); return 1; }
    cosmos3_sample_cfg sc = { steps, guidance, 5.0, !nocfg };
    for (int rep = 0; rep < repeat; rep++) {
        double ta = nowms();
        if (cosmos3_prepare_und(r, nocfg ? 1 : 2)) return 1;
        cosmos3_stats st; cosmos3_get_stats(r, &st);
        double tb = nowms();
        printf("[%d] und tower (%d prompt%s, weights streamed once: %.0f MB H2D): %.0f ms\n", rep, nocfg ? 1 : 2, nocfg ? "" : "s", st.h2d_und_mb, tb - ta);
        if (tf) {
            float *cv = (float *)malloc(4 * (size_t)nlat), *ca = (float *)malloc(4 * (size_t)nact);
            for (int i = 0; i < steps; i++) {
                snprintf(p, sizeof p, "%s/s%d_cond_in_vision_tokens.npy", ref, i); float *lv = cosmos3_npy_read(p, NULL);
                snprintf(p, sizeof p, "%s/s%d_cond_in_action_tokens.npy", ref, i); float *la = cosmos3_npy_read(p, NULL);
                if (!lv || !la) { fprintf(stderr, "missing teacher-force inputs for step %d\n", i); return 1; }
                int ts = cosmos3_timestep_for(r, steps, i, 5.0);
                double tc = nowms();
                if (cosmos3_gen_pass(r, 0, i, lv, la, ts, cv, ca)) return 1;
                if (!nocfg && cosmos3_gen_pass(r, 1, i, lv, la, ts, cv, ca)) return 1;
                printf("  step %d t=%d: %.0f ms per cond+uncond pair\n", i, ts, nowms() - tc);
                free(lv); free(la);
            }
        } else {
            float *ao = (float *)malloc(4 * (size_t)nact), *lo = (float *)malloc(4 * (size_t)nlat);
            if (cosmos3_sample(r, &sc, lat0, act0, ao, lo)) return 1;
            double td = nowms(); cosmos3_get_stats(r, &st);
            printf("[%d] denoise %d steps (%s): %.0f ms; total policy call (und + denoise, excl. VAE) %.0f ms; last gen pass %.0f ms\n", rep, steps, nocfg ? "no cfg" : "cfg", td - tb, td - ta, st.gen_ms_last);
            long long sa[2] = { chunk, 64 }, sl[5] = { 1, 48, T, H, W };
            if (rep == repeat - 1) {
                snprintf(p, sizeof p, "%s/final_action.npy", out);
                float *a10 = (float *)malloc(4ull * chunk * raw); for (int a = 0; a < chunk; a++) for (int d = 0; d < raw; d++) a10[a * raw + d] = ao[a * 64 + d];
                long long s10[2] = { chunk, raw }; cosmos3_npy_write(p, a10, 2, s10);
                snprintf(p, sizeof p, "%s/final_video_latent.npy", out); cosmos3_npy_write(p, lo, 5, sl); (void)sa;
                printf("action[0]:"); for (int d = 0; d < raw; d++) printf(" %.4f", a10[d]); printf("\n");
            }
        }
        cosmos3_stats s2; cosmos3_get_stats(r, &s2);
        printf("[%d] device memory: own accounting peak %.0f MB (weights %.0f MB); cuMemGetInfo min-free drop vs. before-load %.0f MB (excl. CUDA ctx)\n",
               rep, s2.peak_alloc_mb, s2.weights_mb, s2.base_free_mb - s2.min_free_mb);
    }
    return 0;
}
