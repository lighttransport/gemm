/*
 * test_cuda_wan22_vae.c - driver for the standalone CUDA Wan2.2 VAE runner.
 *
 *   ./test_cuda_wan22_vae --weights /mnt/disk1/praxis/wan2.2-vae \
 *        --ref tmp/praxis_ref/vae --out tmp/praxis_out/vae [--taps] [--repeat N]
 *
 * Reads <ref>/video.npy ([1,3,T,H,W] f32) and encodes it -> <out>/latent.npy;
 * reads <ref>/latent.npy (the oracle's policy latent) and decodes it ->
 * <out>/recon_unit.npy ([1,3,T',H,W] in [-1,1]) and <out>/recon.npy
 * ([T',3,H,W], uint8-valued like vae.decode). Compare with
 *   python ref/praxis/compare.py --ref-dir <ref> --ours-dir <out> --tol 0.02
 * --taps also dumps intermediate tensors (needs the ref made with --inter).
 * --self-decode additionally decodes OUR latent -> recon_selfdec_unit.npy.
 */
#define SAFETENSORS_IMPLEMENTATION
#define CUDA_WAN22_VAE_IMPLEMENTATION
#include "cuda_wan22_vae.h"
#include <sys/stat.h>
#include <errno.h>
#include <math.h>

static float *npy_read(const char *path, int *nd, long long shape[8]) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    unsigned char m[10];
    if (fread(m, 1, 10, f) != 10 || memcmp(m, "\x93NUMPY", 6)) { fclose(f); return NULL; }
    int hl = m[8] | (m[9] << 8);
    if (m[6] == 2) { fclose(f); return NULL; }
    char *h = (char *)calloc(1, hl + 1);
    if (fread(h, 1, hl, f) != (size_t)hl) { fclose(f); return NULL; }
    if (!strstr(h, "<f4")) { fprintf(stderr, "%s: only <f4 supported\n", path); fclose(f); return NULL; }
    char *p = strstr(h, "shape");
    p = strchr(p, '(') + 1;
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
    for (char *p = b + 1; *p; p++)
        if (*p == '/') { *p = 0; mkdir(b, 0755); *p = '/'; }
    mkdir(b, 0755);
}

static void write_decoded(const char *out, const char *unit_name, const char *u8_name, const float *fr, int To, int H, int W) {
    char p[1100];
    long long s1[5] = { 1, 3, To, H, W };
    snprintf(p, sizeof p, "%s/%s", out, unit_name);
    npy_write(p, fr, 5, s1);
    if (!u8_name) return;
    /* vae.decode: [T',3,H,W], (clamp(x*0.5+0.5,0,1)*255) truncated to uint8 */
    size_t hw = (size_t)H * W;
    float *u = (float *)malloc(sizeof(float) * (size_t)To * 3 * hw);
    for (int t = 0; t < To; t++)
        for (int c = 0; c < 3; c++)
            for (size_t i = 0; i < hw; i++) {
                float v = fr[((size_t)c * To + t) * hw + i] * 0.5f + 0.5f;
                v = v < 0 ? 0 : v > 1 ? 1 : v;
                u[((size_t)t * 3 + c) * hw + i] = (float)(int)(v * 255.0f);
            }
    long long s2[4] = { To, 3, H, W };
    snprintf(p, sizeof p, "%s/%s", out, u8_name);
    npy_write(p, u, 4, s2);
    free(u);
}


/* synthetic video [3][T][H][W] in [-1,1]: smooth colour fields + moving edges/disks + texture noise */
static float *synth_video(int T, int H, int W) {
    float *v = (float *)malloc(sizeof(float) * 3 * T * H * W);
    unsigned s = 12345;
    for (int c = 0; c < 3; c++)
        for (int t = 0; t < T; t++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    float fx = (float)x / W, fy = (float)y / H, ft = (float)t / (T > 1 ? T - 1 : 1);
                    float b = 0.5f * sinf(6.f * fx + 3.f * c + 2.f * ft) * cosf(5.f * fy - c) + 0.2f * sinf(40.f * fx * fy + t);
                    float cx = 0.2f + 0.6f * ft, cy = 0.5f + 0.2f * sinf(6.f * ft);
                    float d = (fx - cx) * (fx - cx) * (float)W / H + (fy - cy) * (fy - cy) / (float)H * H;
                    if (d < 0.02f) b += 0.6f * (c == 0 ? 1.f : -0.5f);
                    if (((x / 16 + y / 16) & 1) && fx > 0.6f) b += 0.25f;
                    s = s * 1664525u + 1013904223u;
                    b += 0.05f * ((float)(s >> 9) / 8388608.f - 0.5f);
                    v[(((size_t)c * T + t) * H + y) * W + x] = b < -1.f ? -1.f : b > 1.f ? 1.f : b;
                }
    return v;
}

static double rel_l2(const float *a, const float *b, size_t n, double *maxabs) {
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < n; i++) { double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; if (fabs(d) > mx) mx = fabs(d); }
    if (maxabs) *maxabs = mx;
    return sqrt(num / (den > 0 ? den : 1));
}

static double tile_e2e, tile_lat, tile_dec; static int tile_recon_ok = 1;
static int synth_main(cuda_wan22_vae *r, int T, int H, int W, int tile, int tov, int cmp_untiled, int use_u8, int repeat) {
    float *video = synth_video(T, H, W);
    size_t vn = (size_t)3 * T * H * W;
    unsigned char *u8 = NULL;
    if (use_u8) {
        u8 = (unsigned char *)malloc(vn);
        for (size_t i = 0; i < vn; i++) { float q = floorf((video[i] * 0.5f + 0.5f) * 255.f + 0.5f); u8[i] = (unsigned char)q; video[i] = (float)u8[i] / 127.5f - 1.f; }
    }
    int Hp = (H + 31) & ~31, Wp = (W + 31) & ~31;
    float *lat = NULL, *fr = NULL, *lat0 = NULL, *fr0 = NULL;
    int Tl = 0, To = 0;
    if (tile != -2) cuda_wan22_vae_set_tiling(r, tile, tov > 0 ? tov : 0);
    cuda_wan22_vae_reset_peak(r);
    size_t base = cuda_wan22_vae_peak_bytes(r);
    for (int it = 0; it < repeat; it++) {
        free(lat); lat = NULL;
        int rc = use_u8 ? cuda_wan22_vae_encode_u8(r, u8, T, H, W, &lat, &Tl) : cuda_wan22_vae_encode(r, video, T, H, W, &lat, &Tl);
        if (rc) { fprintf(stderr, "encode failed: %s\n", cuda_wan22_vae_last_error(r)); return 1; }
        printf("encode %dx%dx%d (padded %dx%d) -> latent [192,%d,%d,%d]: %.1f ms\n", T, H, W, Hp, Wp, Tl, Hp / 32, Wp / 32, cuda_wan22_vae_last_ms(r));
    }
    printf("  encode peak device memory: %.2f GB (weights %.2f GB)\n", cuda_wan22_vae_peak_bytes(r) / 1073741824.0, base / 1073741824.0);
    cuda_wan22_vae_reset_peak(r);
    for (int it = 0; it < repeat; it++) {
        free(fr); fr = NULL;
        int rc = cuda_wan22_vae_decode_crop(r, lat, Tl, Hp / 32, Wp / 32, H, W, &fr, &To);
        if (rc) { fprintf(stderr, "decode failed: %s\n", cuda_wan22_vae_last_error(r)); return 1; }
        printf("decode latent [192,%d,%d,%d] -> %d frames %dx%d: %.1f ms\n", Tl, Hp / 32, Wp / 32, To, H, W, cuda_wan22_vae_last_ms(r));
    }
    printf("  decode peak device memory: %.2f GB\n", cuda_wan22_vae_peak_bytes(r) / 1073741824.0);
    if (cmp_untiled) {
        cuda_wan22_vae_set_tiling(r, 0, 0);
        cuda_wan22_vae_reset_peak(r);
        int rc = use_u8 ? cuda_wan22_vae_encode_u8(r, u8, T, H, W, &lat0, &Tl) : cuda_wan22_vae_encode(r, video, T, H, W, &lat0, &Tl);
        if (rc) { fprintf(stderr, "untiled encode failed: %s\n", cuda_wan22_vae_last_error(r)); return 1; }
        printf("untiled encode: %.1f ms, peak %.2f GB\n", cuda_wan22_vae_last_ms(r), cuda_wan22_vae_peak_bytes(r) / 1073741824.0);
        double mx, e = rel_l2(lat, lat0, (size_t)192 * Tl * (Hp / 32) * (Wp / 32), &mx);
        printf("  latent tiled-vs-untiled rel-L2 %.3e maxabs %.3e\n", e, mx);
        tile_lat = e;
        cuda_wan22_vae_reset_peak(r);
        rc = cuda_wan22_vae_decode_crop(r, lat0, Tl, Hp / 32, Wp / 32, H, W, &fr0, &To);
        if (rc) { fprintf(stderr, "untiled decode failed: %s\n", cuda_wan22_vae_last_error(r)); return 1; }
        printf("untiled decode: %.1f ms, peak %.2f GB\n", cuda_wan22_vae_last_ms(r), cuda_wan22_vae_peak_bytes(r) / 1073741824.0);
        /* tiled decode of the SAME (untiled) latent vs untiled decode isolates decoder seams */
        float *frt = NULL;
        cuda_wan22_vae_set_tiling(r, tile, tov > 0 ? tov : 0);
        rc = cuda_wan22_vae_decode_crop(r, lat0, Tl, Hp / 32, Wp / 32, H, W, &frt, &To);
        if (rc) { fprintf(stderr, "tiled decode failed: %s\n", cuda_wan22_vae_last_error(r)); return 1; }
        e = rel_l2(frt, fr0, (size_t)3 * To * H * W, &mx);
        printf("  decode (same latent) tiled-vs-untiled rel-L2 %.3e maxabs %.3e\n", e, mx);
        tile_dec = e;
        e = rel_l2(fr, fr0, (size_t)3 * To * H * W, &mx);
        printf("  end-to-end tiled enc+dec vs untiled enc+dec rel-L2 %.3e maxabs %.3e\n", e, mx);
        tile_e2e = e;
        {   /* reconstruction error vs the input frames (decoded T' = 1+4*((T-1)/4) frames; compare those) */
            size_t fn = (size_t)H * W; int tt = To < T ? To : T; double ea = 0, eb = 0, den = 0;
            for (int c = 0; c < 3; c++) for (int t = 0; t < tt; t++) for (size_t i = 0; i < fn; i++) {
                size_t k = ((size_t)c * T + t) * fn + i, k2 = ((size_t)c * To + t) * fn + i;
                double d0 = fr0[k2] - video[k], d1 = fr[k2] - video[k];
                ea += d0 * d0; eb += d1 * d1; den += (double)video[k] * video[k];
            }
            printf("  recon error vs input: untiled %.3e  tiled %.3e (rel-L2)\n", sqrt(ea / den), sqrt(eb / den));
            tile_recon_ok = sqrt(eb / den) <= sqrt(ea / den) * 1.05;
        }
        free(frt); free(lat0); free(fr0);
    }
    free(video); free(u8); free(lat); free(fr);
    cuda_wan22_vae_free(r);
    if (cmp_untiled) {
        int ok = tile_lat < 0.15 && tile_dec < 0.03 && tile_recon_ok;
        printf("tiling check: %s (latent<0.15, same-latent decode<0.03, tiled recon error <= 1.05x untiled)\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *wdir = "/mnt/disk1/praxis/wan2.2-vae", *ref = "tmp/praxis_ref/vae", *out = "tmp/praxis_out/vae";
    int taps = 0, repeat = 1, selfdec = 0, verbose = 1, do_enc = 1, do_dec = 1;
    int synT = 0, synH = 0, synW = 0, tile = -2, tov = -1, cmp_untiled = 0, use_u8 = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--weights") && i + 1 < argc) wdir = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--taps")) taps = 1;
        else if (!strcmp(argv[i], "--self-decode")) selfdec = 1;
        else if (!strcmp(argv[i], "--no-enc")) do_enc = 0;
        else if (!strcmp(argv[i], "--no-dec")) do_dec = 0;
        else if (!strcmp(argv[i], "-v")) verbose = 2;
        else if (!strcmp(argv[i], "--synth") && i + 1 < argc) sscanf(argv[++i], "%d,%d,%d", &synT, &synH, &synW);
        else if (!strcmp(argv[i], "--tile") && i + 1 < argc) { tov = -1; sscanf(argv[++i], "%d,%d", &tile, &tov); }
        else if (!strcmp(argv[i], "--compare-untiled")) cmp_untiled = 1;
        else if (!strcmp(argv[i], "--u8")) use_u8 = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    mkdirs(out);
    char p[1100];
    cuda_wan22_vae *r = cuda_wan22_vae_load(wdir, 0, verbose);
    if (!r) return 1;
    if (taps) cuda_wan22_vae_set_tap_dir(r, out);

    if (synT > 0) return synth_main(r, synT, synH, synW, tile, tov, cmp_untiled, use_u8, repeat);
    float *lat = NULL; int Tl = 0, nd; long long sh[8];
    int T = 0, H = 0, W = 0;
    if (do_enc) {
        snprintf(p, sizeof p, "%s/video.npy", ref);
        float *video = npy_read(p, &nd, sh);
        if (!video || nd != 5) { fprintf(stderr, "bad video.npy\n"); return 1; }
        T = (int)sh[2]; H = (int)sh[3]; W = (int)sh[4];
        for (int it = 0; it < repeat; it++) {
            free(lat);
            if (cuda_wan22_vae_encode(r, video, T, H, W, &lat, &Tl)) return 1;
            printf("encode %dx%dx%d -> latent [192,%d,%d,%d]: %.1f ms\n", T, H, W, Tl, H / 32, W / 32, cuda_wan22_vae_last_ms(r));
        }
        long long s[5] = { 1, 192, Tl, H / 32, W / 32 };
        snprintf(p, sizeof p, "%s/latent.npy", out);
        npy_write(p, lat, 5, s);
        free(video);
    }
    if (do_dec) {
        /* decode the oracle's latent when available (isolates decoder error) */
        snprintf(p, sizeof p, "%s/latent.npy", ref);
        int nd2; long long sh2[8];
        float *rl = npy_read(p, &nd2, sh2);
        float *dl = rl; int dTl, dh, dw;
        if (rl && nd2 == 5) { dTl = (int)sh2[2]; dh = (int)sh2[3]; dw = (int)sh2[4]; }
        else if (lat) { dl = lat; dTl = Tl; dh = H / 32; dw = W / 32; }
        else { fprintf(stderr, "no latent to decode\n"); return 1; }
        float *fr = NULL; int To = 0;
        for (int it = 0; it < repeat; it++) {
            free(fr);
            if (cuda_wan22_vae_decode(r, dl, dTl, dh, dw, &fr, &To)) return 1;
            printf("decode latent [192,%d,%d,%d] -> %d frames %dx%d: %.1f ms\n", dTl, dh, dw, To, dh * 32, dw * 32, cuda_wan22_vae_last_ms(r));
        }
        write_decoded(out, "recon_unit.npy", "recon.npy", fr, To, dh * 32, dw * 32);
        free(fr);
        if (selfdec && lat && rl) {
            cuda_wan22_vae_set_tap_dir(r, NULL);
            if (cuda_wan22_vae_decode(r, lat, Tl, H / 32, W / 32, &fr, &To)) return 1;
            write_decoded(out, "recon_selfdec_unit.npy", NULL, fr, To, H, W);
            free(fr);
        }
        free(rl);
    }
    printf("peak device memory: %.2f GB\n", cuda_wan22_vae_peak_bytes(r) / 1073741824.0);
    cuda_wan22_vae_free(r);
    return 0;
}
