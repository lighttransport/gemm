/*
 * cosmos3_policy.c - end-to-end Cosmos3-Edge-Policy-DROID policy call on one GPU (all compute in CUDA):
 *
 *   conditioning canvas frame -> Wan2.2 VAE encode (cuda/praxis/cuda_wan22_vae.h, weights converted by ref/cosmos3/convert_vae.py)
 *   -> clean latent frame 0 + seeded noise -> und tower (once) -> UniPC denoise (cond + uncond generation passes) -> action chunk [32][10]
 *
 *   cuda/cosmos3/cosmos3_policy --model $PRAXIS_ROOT/cosmos3-edge-droid --vae $PRAXIS_ROOT/cosmos3-vae-wan --ref tmp/cosmos3_ref
 *        [--out tmp/cosmos3_e2e] [--steps 4] [--guidance 3] [--keep-vae] [--oracle-latent] [--repeat N]
 *
 * Python-side helper inputs (exported by ref/cosmos3/gen_ref.py, no C tokenizer / image resampler needed): token ids (input_ids_{cond,uncond}.npy),
 * the padded conditioning canvas frame (cond_canvas.npy, [3][H][W] in [-1,1]), the seeded noise (noise_vision.npy, noise_action.npy).
 * Stage sequence with device memory released between stages: VAE load+encode+free, then transformer load, und, denoise.
 *   --repeat N      N policy calls in one process (N>1: VAE + transformer stay resident, steady-state latency)
 *   --cache-und     reuse the und-tower K/V cache from the previous call (same instruction text)
 *   --keep-vae      keep the VAE resident while the transformer runs (the "everything resident" budget)
 *   --oracle-latent skip the CUDA VAE and take the oracle's latents_init (isolates the transformer + sampler error)
 */
#define SAFETENSORS_IMPLEMENTATION
#define CUDA_WAN22_VAE_IMPLEMENTATION
#define CUDA_COSMOS3_IMPLEMENTATION
#include "../praxis/cuda_wan22_vae.h"
#include "cuda_cosmos3_runner.h"
#include "../../common/praxis_paths.h"

static double nowms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
static double used_mb(void) { size_t fr = 0, tot = 0; cuMemGetInfo(&fr, &tot); return (tot - fr) / 1048576.0; }
static long long *read_i8(const char *path, int *n) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    unsigned char m[10]; if (fread(m, 1, 10, f) != 10) { fclose(f); return NULL; }
    int hl = m[8] | (m[9] << 8); char *h = (char *)calloc(1, hl + 1); if (fread(h, 1, hl, f) != (size_t)hl) { fclose(f); return NULL; }
    long long cnt = strtoll(strchr(strstr(h, "shape"), '(') + 1, NULL, 10);
    long long *d = (long long *)malloc(8 * (size_t)cnt); if (fread(d, 8, (size_t)cnt, f) != (size_t)cnt) { fclose(f); return NULL; }
    fclose(f); free(h); *n = (int)cnt; return d;
}
static float *rd(const char *ref, const char *name, long long *n) { char p[1400]; snprintf(p, sizeof p, "%s/%s.npy", ref, name); float *d = cosmos3_npy_read(p, n); if (!d) fprintf(stderr, "cannot read %s\n", p); return d; }

int main(int argc, char **argv) {
    const char *model = px_root("cosmos3-edge-droid"), *vae_dir = px_root("cosmos3-vae-wan"), *ref = "tmp/cosmos3_ref", *out = "tmp/cosmos3_e2e";
    int steps = 4, keep_vae = 0, cache_und = 0, oracle_lat = 0, repeat = 1, T = 9, LH = 11, LW = 20, CH = 192, CW = 320, chunk = 32, domain = 8, raw = 10;
    float guidance = 3.0f, fps = 15.0f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--vae") && i + 1 < argc) vae_dir = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--guidance") && i + 1 < argc) guidance = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--keep-vae")) keep_vae = 1;
        else if (!strcmp(argv[i], "--cache-und")) cache_und = 1;
        else if (!strcmp(argv[i], "--oracle-latent")) oracle_lat = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0); { CUdevice d; CUcontext c; cuDeviceGet(&d, 0); cuDevicePrimaryCtxRetain(&c, d); cuCtxSetCurrent(c); }
    double base = used_mb();
    printf("device used before anything (other processes + driver): %.0f MB\n", base);
    char p[1400]; (void)p;
    long long n1;
    int nc = 0, nu = 0; snprintf(p, sizeof p, "%s/input_ids_cond.npy", ref); long long *idc = read_i8(p, &nc); snprintf(p, sizeof p, "%s/input_ids_uncond.npy", ref); long long *idu = read_i8(p, &nu);
    float *canvas = rd(ref, "cond_canvas", &n1), *nv = rd(ref, "noise_vision", &n1), *na = rd(ref, "noise_action", &n1);
    if (!idc || !idu || !canvas || !nv || !na) return 1;
    size_t nlat = (size_t)48 * T * LH * LW, nact = (size_t)chunk * 64;
    float *lat0 = (float *)malloc(4 * nlat), *act0 = (float *)calloc(nact, 4);
    for (int a = 0; a < chunk; a++) for (int d = 0; d < raw; d++) act0[a * 64 + d] = na[a * 64 + d];   /* action noise (cols >= raw dim zero) */
    double peak_vae = 0, peak_tr = 0, t_vae = 0, t_load = 0;
    cuda_wan22_vae *vae = NULL; cosmos3_runner *r = NULL; double vae_load_ms = 0, vae_enc_ms = 0;
    for (int rep = 0; rep < repeat; rep++) {
        double t0 = nowms();
        /* ---- stage 1: VAE encode of the conditioning frame ---- */
        double tv = nowms();
        if (oracle_lat) { float *l = rd(ref, "latents_init", &n1); memcpy(lat0, l, 4 * nlat); free(l); }
        else {
            if (!vae) { double tq = nowms(); vae = cuda_wan22_vae_load(vae_dir, 0, 0); if (!vae) return 1; vae_load_ms = nowms() - tq; }
            float *pol = NULL; int Tl = 0;
            if (cuda_wan22_vae_encode(vae, canvas, 1, CH, CW, &pol, &Tl)) return 1;
            vae_enc_ms = cuda_wan22_vae_last_ms(vae);
            peak_vae = used_mb() > peak_vae ? used_mb() : peak_vae;
            { size_t pb = cuda_wan22_vae_peak_bytes(vae); double m = pb / 1048576.0 + 0; if (m + base > peak_vae) peak_vae = m + base; }
            int hp = CH / 32, wp = CW / 32;   /* policy latent [192][1][6][10] -> raw normalised z [48][12][20], crop to LH rows */
            for (int c = 0; c < 48; c++) for (int t = 0; t < T; t++) for (int y = 0; y < LH; y++) for (int x = 0; x < LW; x++) {
                size_t o = (((size_t)c * T + t) * LH + y) * LW + x;
                if (t == 0) lat0[o] = pol[(((size_t)(c * 4 + (y & 1) * 2 + (x & 1)) * Tl + 0) * hp + (y >> 1)) * wp + (x >> 1)];
                else lat0[o] = nv[o];   /* noisy frames: seeded noise */
            }
            free(pol);
            if (!keep_vae && repeat == 1) { cuda_wan22_vae_free(vae); vae = NULL; }
        }
        t_vae = nowms() - tv;
        double after_vae = used_mb();
        /* ---- stage 2: transformer ---- */
        double tl = nowms();
        if (!r) {
            r = cosmos3_load(model, 0, 0); if (!r) return 1;
            t_load = nowms() - tl;
            cosmos3_set_dump_dir(r, out); cosmos3_set_dump_steps(r, 0);
            cosmos3_set_geometry(r, T, LH, LW, chunk, fps, domain, raw);
            if (cosmos3_set_prompt(r, 0, idc, nc) || cosmos3_set_prompt(r, 1, idu, nu)) return 1;
        }
        double ta = nowms();
        if (!(cache_und && rep > 0) && cosmos3_prepare_und(r, 2)) return 1;   /* the und tower depends on the text only: cacheable across control steps with the same instruction */
        double tb = nowms(); double used_und = used_mb();
        float *ao = (float *)malloc(4 * nact), *lo = (float *)malloc(4 * nlat);
        cosmos3_sample_cfg sc = { steps, guidance, 5.0, guidance != 1.0f };
        if (cosmos3_sample(r, &sc, lat0, act0, ao, lo)) return 1;
        double tc = nowms(); double used_den = used_mb(); cosmos3_stats st; cosmos3_get_stats(r, &st);
        peak_tr = (base + (st.base_free_mb - st.min_free_mb)) > peak_tr ? (base + (st.base_free_mb - st.min_free_mb)) : peak_tr;
        printf("[%d] VAE %s: %.0f ms (load %.0f + encode %.0f) | transformer load %.0f ms (once per process in a server) | und %.0f ms | denoise %d steps %.0f ms | policy call (VAE enc + und + denoise) %.0f ms\n",
               rep, oracle_lat ? "(skipped, oracle latent)" : "stage", t_vae, rep == 0 ? vae_load_ms : 0.0, vae_enc_ms, t_load, tb - ta, steps, tc - tb, (oracle_lat ? 0 : vae_enc_ms) + (tc - ta));
        printf("[%d] device used (cuMemGetInfo total-free, incl. ctx + other procs): after VAE stage %.0f MB, after und %.0f MB, after denoise %.0f MB; min-free-derived peak %.0f MB (base %.0f)\n",
               rep, after_vae, used_und, used_den, peak_tr, base);
        if (rep == repeat - 1) {
            char q[1400]; long long s2[2] = { chunk, raw }, s5[5] = { 1, 48, T, LH, LW };
            float *a10 = (float *)malloc(4ull * chunk * raw); for (int a = 0; a < chunk; a++) for (int d = 0; d < raw; d++) a10[a * raw + d] = ao[a * 64 + d];
            cosmos3_set_dump_dir(r, NULL);
            { char b[1300]; snprintf(b, sizeof b, "mkdir -p %s", out); (void)!system(b); }
            snprintf(q, sizeof q, "%s/final_action.npy", out); cosmos3_npy_write(q, a10, 2, s2);
            snprintf(q, sizeof q, "%s/final_video_latent.npy", out); cosmos3_npy_write(q, lo, 5, s5);
            { long long s3[4] = { 1, 48, LH, LW }; float *f0 = (float *)malloc(4ull * 48 * LH * LW);
              for (int c = 0; c < 48; c++) for (int y = 0; y < LH; y++) for (int x = 0; x < LW; x++) f0[((size_t)c * LH + y) * LW + x] = lat0[(((size_t)c * T) * LH + y) * LW + x];
              long long s4[3] = { 48, LH, LW }; (void)s3; snprintf(q, sizeof q, "%s/x0_latent_frame0.npy", out); cosmos3_npy_write(q, f0, 3, s4); free(f0); }
            { snprintf(q, sizeof q, "%s/latents_init.npy", out); cosmos3_npy_write(q, lat0, 5, s5); }
            printf("action[0]:"); for (int d = 0; d < raw; d++) printf(" %.4f", a10[d]); printf("\n");
        }
        free(ao); free(lo);
        (void)t0;
    }
    if (vae) cuda_wan22_vae_free(vae);
    printf("peak device memory per stage (used incl. driver/ctx baseline %.0f MB): VAE stage %.0f MB, transformer stage %.0f MB\n", base, peak_vae, peak_tr);
    return 0;
}
