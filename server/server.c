/*
 * diffusion-server: tiny C HTTP/MCP inference server for local model runners.
 *
 * This is intentionally dependency-light: POSIX sockets, existing single-header
 * model code, stb_image_write for PNG encoding, and the safetensors JSON parser.
 */

#define _POSIX_C_SOURCE 200809L

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#define TRANSFORMER_IMPLEMENTATION
#define QIMG_SCHEDULER_IMPLEMENTATION
#define QIMG_DIT_IMPLEMENTATION
#define QIMG_VAE_IMPLEMENTATION
#define QIMG_TEXT_ENCODER_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <pthread.h>

#include "../common/safetensors.h"
#include "../common/ggml_dequant.h"
#include "../common/gguf_loader.h"
#include "../common/bpe_tokenizer.h"
#include "../common/transformer.h"
#include "../common/qwen_image_scheduler.h"

#if defined(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_HIP)
#include "../rdna4/llm/hip_llm_runner.h"
#include "../rdna4/qimg/hip_qimg_runner.h"
#endif

#if defined(DIFFUSION_SERVER_ENABLE_SAM3)
#include "server_sam3.h"
#endif

#include "../common/qwen_image_text_encoder.h"
#include "../common/qwen_image_dit.h"
#include "../common/qwen_image_vae.h"
#include "../common/stb_image_write.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <netdb.h>
#include <strings.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <glob.h>
#include <dirent.h>

#define MAX_BODY_BYTES (64u * 1024u * 1024u)
#define READ_CHUNK 8192

typedef struct {
    char *name;
    char *model_dir;
    char *dit;
    char *vae;
    char *enc;
    char *bias;
    char *kind;   /* "gguf" or "safetensors" */
} qwen_variant;

typedef struct {
    const char *host;
    int port;
    const char *web_root;
    const char *qwen_dit;
    const char *qwen_vae;
    const char *qwen_enc;
    const char *qwen_enc_bias;
    const char *qwen_variants_spec;   /* raw "name:dir,name:dir" string */
    qwen_variant *qwen_variants;
    int qwen_variant_count;
    const char *sam3_ckpt;
    const char *sam3_ckpt_v31;
    const char *sam3_vocab;
    const char *sam3_merges;
    const char *sam3_ref_url;     /* http://host:port forwarded via /v1/ref/sam3   */
    const char *sam3_1_ref_url;   /* http://host:port forwarded via /v1/ref/sam3.1 */
    int device;
    int mcp_stdio;
    int stdio_mode;
} server_config;

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} sbuf;

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
} bbuf;

typedef struct {
    uint8_t *png;
    int png_len;
    int width;
    int height;
} image_result;

static pthread_mutex_t g_infer_mu = PTHREAD_MUTEX_INITIALIZER;
static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

static void sbuf_init(sbuf *b) {
    b->cap = 4096;
    b->len = 0;
    b->ptr = (char *)malloc(b->cap);
    if (b->ptr) b->ptr[0] = 0;
}

static void sbuf_free(sbuf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static int sbuf_reserve(sbuf *b, size_t need) {
    if (need <= b->cap) return 0;
    size_t nc = b->cap ? b->cap : 4096;
    while (nc < need) nc *= 2;
    char *p = (char *)realloc(b->ptr, nc);
    if (!p) return -1;
    b->ptr = p;
    b->cap = nc;
    return 0;
}

static int sbuf_appendn(sbuf *b, const char *s, size_t n) {
    if (sbuf_reserve(b, b->len + n + 1) != 0) return -1;
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = 0;
    return 0;
}

static int sbuf_append(sbuf *b, const char *s) {
    return sbuf_appendn(b, s, strlen(s));
}

static int sbuf_printf(sbuf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) { va_end(ap2); return -1; }
    if (sbuf_reserve(b, b->len + (size_t)n + 1) != 0) { va_end(ap2); return -1; }
    vsnprintf(b->ptr + b->len, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
    return 0;
}

static void bbuf_init(bbuf *b) {
    b->cap = 16384;
    b->len = 0;
    b->data = (uint8_t *)malloc(b->cap);
}

static void bbuf_free(bbuf *b) {
    free(b->data);
    memset(b, 0, sizeof(*b));
}

static int bbuf_reserve(bbuf *b, size_t need) {
    if (need <= b->cap) return 0;
    size_t nc = b->cap ? b->cap : 16384;
    while (nc < need) nc *= 2;
    uint8_t *p = (uint8_t *)realloc(b->data, nc);
    if (!p) return -1;
    b->data = p;
    b->cap = nc;
    return 0;
}

static int bbuf_append(bbuf *b, const void *data, size_t n) {
    if (bbuf_reserve(b, b->len + n) != 0) return -1;
    memcpy(b->data + b->len, data, n);
    b->len += n;
    return 0;
}

static char *xstrdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

/* Qwen-Image variant registry.
 *
 * Directory layout (hyphen/underscore tolerant): diffusion_models/ for the
 * DiT, vae/ for the VAE, text_encoders/ or text_encoder/ for the encoder.
 * Parsed at startup from --qwen-variants "name:dir,name:dir,..."  The client
 * may pass params.variant to auto-fill the three weight paths.
 */
static int str_has_suffix(const char *s, const char *suf) {
    if (!s || !suf) return 0;
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && strcmp(s + ls - lf, suf) == 0;
}

static char *variant_first_glob(const char *root, const char *const *patterns, int n_patterns) {
    char pat[1024];
    for (int i = 0; i < n_patterns; i++) {
        snprintf(pat, sizeof(pat), "%s/%s", root, patterns[i]);
        glob_t gl = {0};
        if (glob(pat, 0, NULL, &gl) == 0 && gl.gl_pathc > 0) {
            char *out = xstrdup(gl.gl_pathv[0]);
            globfree(&gl);
            return out;
        }
        globfree(&gl);
    }
    return NULL;
}

static int variant_detect_paths(qwen_variant *v) {
    static const char *dit_pats[] = {
        "diffusion_models/*.safetensors",
        "diffusion_models/*.gguf",
        "diffusion-models/*.safetensors",
        "diffusion-models/*.gguf",
    };
    static const char *vae_pats[] = {
        "vae/*.safetensors",
        "vae/*.gguf",
    };
    static const char *enc_pats[] = {
        "text_encoders/*.safetensors",
        "text_encoders/*.gguf",
        "text-encoder/*.gguf",
        "text-encoder/*.safetensors",
        "text_encoder/*.safetensors",
        "text_encoder/*.gguf",
    };
    v->dit = variant_first_glob(v->model_dir, dit_pats, (int)(sizeof(dit_pats)/sizeof(*dit_pats)));
    v->vae = variant_first_glob(v->model_dir, vae_pats, (int)(sizeof(vae_pats)/sizeof(*vae_pats)));
    v->enc = variant_first_glob(v->model_dir, enc_pats, (int)(sizeof(enc_pats)/sizeof(*enc_pats)));
    /* Skip multimodal projector if a real encoder sits alongside. */
    if (v->enc) {
        const char *base = strrchr(v->enc, '/');
        base = base ? base + 1 : v->enc;
        if (strncasecmp(base, "mmproj", 6) == 0) {
            static const char *enc2_pats[] = {
                "text-encoder/Qwen*",
                "text_encoders/Qwen*",
                "text_encoder/Qwen*",
            };
            char *alt = variant_first_glob(v->model_dir, enc2_pats,
                                           (int)(sizeof(enc2_pats)/sizeof(*enc2_pats)));
            if (alt) { free(v->enc); v->enc = alt; }
        }
    }
    v->kind = v->dit && str_has_suffix(v->dit, ".gguf") ? xstrdup("gguf") : xstrdup("safetensors");
    return v->dit && v->vae && v->enc ? 0 : -1;
}

static void variant_free(qwen_variant *v) {
    free(v->name); free(v->model_dir); free(v->dit); free(v->vae);
    free(v->enc); free(v->bias); free(v->kind);
    memset(v, 0, sizeof(*v));
}

static int qwen_variants_parse(server_config *cfg, const char *spec) {
    cfg->qwen_variants = NULL;
    cfg->qwen_variant_count = 0;
    if (!spec || !*spec) return 0;
    /* First pass: count commas + 1. */
    int n = 1;
    for (const char *p = spec; *p; p++) if (*p == ',') n++;
    qwen_variant *arr = (qwen_variant *)calloc((size_t)n, sizeof(*arr));
    if (!arr) return -1;
    int k = 0;
    char *dup = xstrdup(spec);
    char *save = NULL;
    for (char *tok = strtok_r(dup, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        while (*tok == ' ' || *tok == '\t') tok++;
        if (!*tok) continue;
        char *colon = strchr(tok, ':');
        if (!colon) {
            fprintf(stderr, "[qwen-variants] bad entry '%s' (want name:dir)\n", tok);
            continue;
        }
        *colon = '\0';
        arr[k].name = xstrdup(tok);
        arr[k].model_dir = xstrdup(colon + 1);
        if (variant_detect_paths(&arr[k]) != 0) {
            fprintf(stderr, "[qwen-variants] '%s' at %s: missing dit/vae/enc (dit=%s vae=%s enc=%s)\n",
                    arr[k].name, arr[k].model_dir,
                    arr[k].dit ? arr[k].dit : "(none)",
                    arr[k].vae ? arr[k].vae : "(none)",
                    arr[k].enc ? arr[k].enc : "(none)");
            variant_free(&arr[k]);
            continue;
        }
        fprintf(stderr, "[qwen-variants] %s (%s): dit=%s\n",
                arr[k].name, arr[k].kind, arr[k].dit);
        k++;
    }
    free(dup);
    cfg->qwen_variants = arr;
    cfg->qwen_variant_count = k;
    return k;
}

static const qwen_variant *qwen_variants_find(const server_config *cfg, const char *name) {
    if (!cfg || !name || !*name) return NULL;
    for (int i = 0; i < cfg->qwen_variant_count; i++) {
        if (strcmp(cfg->qwen_variants[i].name, name) == 0) return &cfg->qwen_variants[i];
    }
    return NULL;
}

static char *json_escape_dup(const char *s) {
    if (!s) return xstrdup("");
    sbuf b;
    sbuf_init(&b);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '\\': sbuf_append(&b, "\\\\"); break;
        case '"': sbuf_append(&b, "\\\""); break;
        case '\n': sbuf_append(&b, "\\n"); break;
        case '\r': sbuf_append(&b, "\\r"); break;
        case '\t': sbuf_append(&b, "\\t"); break;
        default:
            if (*p < 32) sbuf_printf(&b, "\\u%04x", (unsigned)*p);
            else sbuf_appendn(&b, (const char *)p, 1);
            break;
        }
    }
    return b.ptr;
}

static const char *json_str(const json_val *obj, const char *key, const char *def) {
    json_val *v = json_obj_get(obj, key);
    if (!v || v->type != JSON_STRING) return def;
    return v->str.ptr ? v->str.ptr : def;
}

static int json_int(const json_val *obj, const char *key, int def) {
    json_val *v = json_obj_get(obj, key);
    if (!v || v->type != JSON_NUMBER) return def;
    return (int)v->num;
}

static unsigned long long json_u64(const json_val *obj, const char *key, unsigned long long def) {
    json_val *v = json_obj_get(obj, key);
    if (!v || v->type != JSON_NUMBER) return def;
    return (unsigned long long)v->num;
}

static double json_f64(const json_val *obj, const char *key, double def) {
    json_val *v = json_obj_get(obj, key);
    if (!v || v->type != JSON_NUMBER) return def;
    return v->num;
}

static json_val *json_obj(const json_val *obj, const char *key) {
    json_val *v = json_obj_get(obj, key);
    return (v && v->type == JSON_OBJECT) ? v : NULL;
}

static const char b64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static char *base64_encode(const uint8_t *data, size_t len) {
    size_t out_len = ((len + 2) / 3) * 4;
    char *out = (char *)malloc(out_len + 1);
    if (!out) return NULL;
    size_t o = 0;
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        int rem = (int)(len - i);
        if (rem > 1) v |= (uint32_t)data[i + 1] << 8;
        if (rem > 2) v |= data[i + 2];
        out[o++] = b64_table[(v >> 18) & 63];
        out[o++] = b64_table[(v >> 12) & 63];
        out[o++] = rem > 1 ? b64_table[(v >> 6) & 63] : '=';
        out[o++] = rem > 2 ? b64_table[v & 63] : '=';
    }
    out[o] = 0;
    return out;
}

/* base64 decode (whitespace-tolerant, stops at '=' or end). Returns caller-free
 * byte buffer or NULL on invalid input. */
uint8_t *base64_decode_buf(const char *s, size_t in_len, size_t *out_len);
uint8_t *base64_decode_buf(const char *s, size_t in_len, size_t *out_len) {
    size_t n = 0;
    for (size_t i = 0; i < in_len; i++) {
        char c = s[i];
        if (c == ' ' || c == '\r' || c == '\n' || c == '\t') continue;
        if (c == '=') break;
        n++;
    }
    size_t pad = (4 - (n % 4)) % 4;
    size_t out_cap = ((n + pad) / 4) * 3;
    uint8_t *out = (uint8_t *)malloc(out_cap + 4);
    if (!out) return NULL;
    static const int8_t tbl[256] = {
        ['A']=0,['B']=1,['C']=2,['D']=3,['E']=4,['F']=5,['G']=6,['H']=7,['I']=8,
        ['J']=9,['K']=10,['L']=11,['M']=12,['N']=13,['O']=14,['P']=15,['Q']=16,
        ['R']=17,['S']=18,['T']=19,['U']=20,['V']=21,['W']=22,['X']=23,['Y']=24,
        ['Z']=25,
        ['a']=26,['b']=27,['c']=28,['d']=29,['e']=30,['f']=31,['g']=32,['h']=33,
        ['i']=34,['j']=35,['k']=36,['l']=37,['m']=38,['n']=39,['o']=40,['p']=41,
        ['q']=42,['r']=43,['s']=44,['t']=45,['u']=46,['v']=47,['w']=48,['x']=49,
        ['y']=50,['z']=51,
        ['0']=52,['1']=53,['2']=54,['3']=55,['4']=56,['5']=57,['6']=58,['7']=59,
        ['8']=60,['9']=61,['+']=62,['/']=63,
    };
    size_t oi = 0;
    int buf[4]; int bc = 0;
    for (size_t i = 0; i < in_len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == ' ' || c == '\r' || c == '\n' || c == '\t') continue;
        if (c == '=') break;
        int v = tbl[c];
        if (v == 0 && c != 'A') { free(out); return NULL; }
        buf[bc++] = v;
        if (bc == 4) {
            out[oi++] = (uint8_t)((buf[0] << 2) | (buf[1] >> 4));
            out[oi++] = (uint8_t)(((buf[1] & 0xF) << 4) | (buf[2] >> 2));
            out[oi++] = (uint8_t)(((buf[2] & 0x3) << 6) | buf[3]);
            bc = 0;
        }
    }
    if (bc == 2) out[oi++] = (uint8_t)((buf[0] << 2) | (buf[1] >> 4));
    else if (bc == 3) {
        out[oi++] = (uint8_t)((buf[0] << 2) | (buf[1] >> 4));
        out[oi++] = (uint8_t)(((buf[1] & 0xF) << 4) | (buf[2] >> 2));
    }
    *out_len = oi;
    return out;
}

static uint64_t rng_state = 42;
static int rng_cached_valid = 0;
static float rng_cached = 0.0f;

static float randn_local(void) {
    if (rng_cached_valid) { rng_cached_valid = 0; return rng_cached; }
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-10) u1 = 1e-10;
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    rng_cached = (float)(r * sin(theta));
    rng_cached_valid = 1;
    return (float)(r * cos(theta));
}

static uint8_t *chw_float_to_rgb8(const float *rgb, int h, int w) {
    uint8_t *u8 = (uint8_t *)malloc((size_t)h * w * 3);
    if (!u8) return NULL;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                float v = rgb[(size_t)c * h * w + (size_t)y * w + x];
                v = v * 0.5f + 0.5f;
                if (v < 0) v = 0;
                if (v > 1) v = 1;
                u8[((size_t)y * w + x) * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
            }
        }
    }
    return u8;
}

static int png_from_chw_float(const float *rgb, int h, int w, image_result *res, char *err, size_t err_cap) {
    uint8_t *u8 = chw_float_to_rgb8(rgb, h, w);
    if (!u8) {
        snprintf(err, err_cap, "out of memory converting RGB");
        return -1;
    }
    int png_len = 0;
    unsigned char *png = stbi_write_png_to_mem(u8, w * 3, w, h, 3, &png_len);
    free(u8);
    if (!png || png_len <= 0) {
        snprintf(err, err_cap, "failed to encode PNG");
        return -1;
    }
    res->png = png;
    res->png_len = png_len;
    res->width = w;
    res->height = h;
    return 0;
}

static int has_suffix(const char *s, const char *suffix) {
    if (!s || !suffix) return 0;
    size_t a = strlen(s), b = strlen(suffix);
    return a >= b && strcmp(s + a - b, suffix) == 0;
}

static int qwen_cpu_generate(const char *dit_path, const char *vae_path,
                             const char *enc_path, const char *bias_path,
                             const char *prompt, int out_h, int out_w,
                             int n_steps, uint64_t seed,
                             image_result *result, char *err, size_t err_cap) {
#if defined(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE) && defined(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_CPU)
    if (!dit_path || !vae_path || !enc_path) {
        snprintf(err, err_cap, "qwen-image cpu requires dit, vae, and text_encoder paths");
        return -1;
    }
    qimg_dit_model *dit = has_suffix(dit_path, ".safetensors")
        ? qimg_dit_load_safetensors(dit_path)
        : qimg_dit_load_gguf(dit_path);
    if (!dit) { snprintf(err, err_cap, "failed to load DiT: %s", dit_path); return -1; }

    qimg_vae_model *vae = qimg_vae_load(vae_path);
    if (!vae) {
        qimg_dit_free(dit);
        snprintf(err, err_cap, "failed to load VAE: %s", vae_path);
        return -1;
    }

    qimg_text_enc *enc = bias_path && bias_path[0]
        ? qimg_text_enc_load_gguf_with_biases(enc_path, bias_path)
        : qimg_text_enc_load(enc_path);
    if (!enc) {
        qimg_vae_free(vae);
        qimg_dit_free(dit);
        snprintf(err, err_cap, "failed to load text encoder: %s", enc_path);
        return -1;
    }

    int n_txt = 0;
    float *txt_tokens = qimg_text_enc_encode(enc, prompt ? prompt : "", &n_txt);
    qimg_text_enc_free(enc);
    if (!txt_tokens || n_txt <= 0) {
        qimg_vae_free(vae);
        qimg_dit_free(dit);
        snprintf(err, err_cap, "failed to encode prompt");
        return -1;
    }

    int ps = dit->patch_size;
    int lat_h = out_h / 8;
    int lat_w = out_w / 8;
    int hp = lat_h / ps;
    int wp = lat_w / ps;
    int n_img = hp * wp;
    int in_ch = dit->in_channels;
    int lat_ch = 16;
    size_t lat_total = (size_t)lat_ch * lat_h * lat_w;

    float *latent = (float *)malloc(lat_total * sizeof(float));
    float *img_tokens = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
    if (!latent || !img_tokens) {
        free(latent); free(img_tokens); free(txt_tokens);
        qimg_vae_free(vae); qimg_dit_free(dit);
        snprintf(err, err_cap, "out of memory allocating latent");
        return -1;
    }
    rng_state = seed;
    rng_cached_valid = 0;
    for (size_t i = 0; i < lat_total; i++) latent[i] = randn_local();

    qimg_scheduler sched;
    qimg_sched_init(&sched);
    qimg_sched_set_timesteps(&sched, n_steps, n_img);

    for (int step = 0; step < n_steps; step++) {
        qimg_dit_patchify(img_tokens, latent, lat_ch, lat_h, lat_w, ps);
        float *velocity = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        float *vel_latent = (float *)malloc(lat_total * sizeof(float));
        if (!velocity || !vel_latent) {
            free(velocity); free(vel_latent); free(latent); free(img_tokens); free(txt_tokens);
            qimg_vae_free(vae); qimg_dit_free(dit);
            snprintf(err, err_cap, "out of memory during denoise");
            return -1;
        }
        qimg_dit_forward(velocity, img_tokens, n_img, txt_tokens, n_txt,
                         sched.timesteps[step], dit, 1);
        qimg_dit_unpatchify(vel_latent, velocity, n_img, lat_ch, lat_h, lat_w, ps);
        qimg_sched_step(latent, vel_latent, (int)lat_total, step, &sched);
        free(velocity);
        free(vel_latent);
    }

    free(img_tokens);
    free(txt_tokens);
    qimg_dit_unnormalize_latent(latent, lat_ch, lat_h, lat_w);

    float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
    if (!rgb) {
        free(latent); qimg_vae_free(vae); qimg_dit_free(dit);
        snprintf(err, err_cap, "out of memory allocating RGB");
        return -1;
    }
    qimg_vae_decode(rgb, latent, lat_h, lat_w, vae);
    int rc = png_from_chw_float(rgb, out_h, out_w, result, err, err_cap);
    free(rgb);
    free(latent);
    qimg_vae_free(vae);
    qimg_dit_free(dit);
    return rc;
#else
    (void)dit_path; (void)vae_path; (void)enc_path; (void)bias_path; (void)prompt;
    (void)out_h; (void)out_w; (void)n_steps; (void)seed; (void)result;
    snprintf(err, err_cap, "qwen-image cpu backend was not compiled");
    return -1;
#endif
}

static int qwen_hip_generate(const char *dit_path, const char *vae_path,
                             const char *enc_path, const char *prompt,
                             int out_h, int out_w, int n_steps,
                             uint64_t seed, int device,
                             image_result *result, char *err, size_t err_cap) {
#if defined(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE) && defined(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_HIP)
    if (!dit_path || !vae_path || !enc_path) {
        snprintf(err, err_cap, "qwen-image hip requires dit, vae, and text_encoder paths");
        return -1;
    }
    hip_qimg_runner *r = hip_qimg_init(device, 1);
    if (!r) { snprintf(err, err_cap, "failed to initialize HIP runner"); return -1; }

    int n_txt = 0;
    qimg_text_enc *enc = qimg_text_enc_load_gpu(enc_path, NULL, device);
    float *txt_tokens = NULL;
    if (enc) {
        txt_tokens = qimg_text_enc_encode(enc, prompt ? prompt : "", &n_txt);
        qimg_text_enc_free(enc);
    }
    if (!txt_tokens) {
        hip_qimg_free(r);
        snprintf(err, err_cap, "failed to encode prompt with HIP text encoder");
        return -1;
    }

    if (hip_qimg_load_dit(r, dit_path) != 0) {
        free(txt_tokens);
        hip_qimg_free(r);
        snprintf(err, err_cap, "failed to load HIP DiT: %s", dit_path);
        return -1;
    }

    int ps = 2;
    int lat_ch = 16;
    int in_ch = 64;
    int lat_h = out_h / 8;
    int lat_w = out_w / 8;
    int hp = lat_h / ps;
    int wp = lat_w / ps;
    int n_img = hp * wp;
    size_t lat_total = (size_t)lat_ch * lat_h * lat_w;
    float *latent = (float *)malloc(lat_total * sizeof(float));
    float *img_tokens = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
    float *vel = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
    if (!latent || !img_tokens || !vel) {
        free(latent); free(img_tokens); free(vel); free(txt_tokens); hip_qimg_free(r);
        snprintf(err, err_cap, "out of memory allocating HIP inputs");
        return -1;
    }

    rng_state = seed;
    rng_cached_valid = 0;
    for (size_t i = 0; i < lat_total; i++) latent[i] = randn_local();
    qimg_dit_patchify(img_tokens, latent, lat_ch, lat_h, lat_w, ps);

    qimg_scheduler sched;
    qimg_sched_init(&sched);
    qimg_sched_set_timesteps(&sched, n_steps, n_img + n_txt);
    for (int step = 0; step < n_steps; step++) {
        float t = sched.sigmas[step] * 1000.0f;
        float dt = sched.dt[step];
        if (hip_qimg_dit_step(r, img_tokens, n_img, txt_tokens, n_txt, t, vel) != 0) {
            free(latent); free(img_tokens); free(vel); free(txt_tokens); hip_qimg_free(r);
            snprintf(err, err_cap, "HIP DiT step failed at step %d", step + 1);
            return -1;
        }
        for (int i = 0; i < n_img * in_ch; i++) img_tokens[i] += dt * vel[i];
    }

    qimg_dit_unpatchify(latent, img_tokens, n_img, lat_ch, lat_h, lat_w, ps);
    qimg_dit_unnormalize_latent(latent, lat_ch, lat_h, lat_w);

    if (hip_qimg_load_vae(r, vae_path) != 0) {
        free(latent); free(img_tokens); free(vel); free(txt_tokens); hip_qimg_free(r);
        snprintf(err, err_cap, "failed to load HIP VAE: %s", vae_path);
        return -1;
    }

    float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
    if (!rgb) {
        free(latent); free(img_tokens); free(vel); free(txt_tokens); hip_qimg_free(r);
        snprintf(err, err_cap, "out of memory allocating RGB");
        return -1;
    }
    if (hip_qimg_vae_decode(r, latent, lat_h, lat_w, rgb) != 0) {
        free(rgb); free(latent); free(img_tokens); free(vel); free(txt_tokens); hip_qimg_free(r);
        snprintf(err, err_cap, "HIP VAE decode failed");
        return -1;
    }

    int rc = png_from_chw_float(rgb, out_h, out_w, result, err, err_cap);
    free(rgb);
    free(latent);
    free(img_tokens);
    free(vel);
    free(txt_tokens);
    hip_qimg_free(r);
    return rc;
#else
    (void)dit_path; (void)vae_path; (void)enc_path; (void)prompt; (void)out_h; (void)out_w;
    (void)n_steps; (void)seed; (void)device; (void)result;
    snprintf(err, err_cap, "qwen-image hip backend was not compiled");
    return -1;
#endif
}

static char *json_error_response(const char *code, const char *message) {
    char *ec = json_escape_dup(code);
    char *em = json_escape_dup(message);
    sbuf b;
    sbuf_init(&b);
    sbuf_printf(&b, "{\"ok\":false,\"error\":{\"code\":\"%s\",\"message\":\"%s\"}}", ec, em);
    free(ec);
    free(em);
    return b.ptr;
}

static void json_append_value(sbuf *out, const json_val *v) {
    if (!v) { sbuf_append(out, "null"); return; }
    switch (v->type) {
    case JSON_NULL: sbuf_append(out, "null"); break;
    case JSON_TRUE: sbuf_append(out, "true"); break;
    case JSON_FALSE: sbuf_append(out, "false"); break;
    case JSON_NUMBER: sbuf_printf(out, "%.17g", v->num); break;
    case JSON_STRING: {
        char *s = json_escape_dup(v->str.ptr);
        sbuf_printf(out, "\"%s\"", s);
        free(s);
        break;
    }
    case JSON_ARRAY:
        sbuf_append(out, "[");
        for (int i = 0; i < v->arr.count; i++) {
            if (i) sbuf_append(out, ",");
            json_append_value(out, &v->arr.items[i]);
        }
        sbuf_append(out, "]");
        break;
    case JSON_OBJECT:
        sbuf_append(out, "{");
        for (int i = 0; i < v->obj.count; i++) {
            char *k = json_escape_dup(v->obj.keys[i]);
            if (i) sbuf_append(out, ",");
            sbuf_printf(out, "\"%s\":", k);
            free(k);
            json_append_value(out, &v->obj.vals[i]);
        }
        sbuf_append(out, "}");
        break;
    }
}

static char *build_qwen_infer_body_from_mcp_args(json_val *args) {
    const char *backend = args ? json_str(args, "backend", "cpu") : "cpu";
    const char *prompt = args ? json_str(args, "prompt", "") : "";
    char *be = json_escape_dup(backend);
    char *pr = json_escape_dup(prompt);
    sbuf b;
    sbuf_init(&b);
    sbuf_printf(&b,
        "{\"model\":\"qwen-image\",\"task\":\"text-to-image\",\"backend\":\"%s\","
        "\"inputs\":{\"text\":\"%s\"},\"weights\":",
        be, pr);
    json_val *weights = args ? json_obj(args, "weights") : NULL;
    if (weights) json_append_value(&b, weights);
    else sbuf_append(&b, "{}");
    sbuf_append(&b, ",\"params\":");
    json_val *params = args ? json_obj(args, "params") : NULL;
    if (params) {
        json_append_value(&b, params);
    } else {
        int width = args ? json_int(args, "width", 256) : 256;
        int height = args ? json_int(args, "height", 256) : 256;
        int steps = args ? json_int(args, "steps", 20) : 20;
        int device = args ? json_int(args, "device", 0) : 0;
        unsigned long long seed = args ? json_u64(args, "seed", 42) : 42;
        sbuf_printf(&b, "{\"width\":%d,\"height\":%d,\"steps\":%d,\"seed\":%llu,\"device\":%d}",
                    width, height, steps, seed, device);
    }
    sbuf_append(&b, "}");
    free(be);
    free(pr);
    return b.ptr;
}

/* Build a sam3 segmentation infer-body from MCP tool arguments. */
static char *build_sam3_infer_body_from_mcp_args(json_val *args, const char *model_id) {
    const char *backend = args ? json_str(args, "backend", "cpu") : "cpu";
    const char *text = args ? json_str(args, "text", "") : "";
    const char *image_b64 = args ? json_str(args, "image_base64", "") : "";
    char *be = json_escape_dup(backend);
    char *tx = json_escape_dup(text);
    char *ib = json_escape_dup(image_b64);
    sbuf b;
    sbuf_init(&b);
    sbuf_printf(&b,
        "{\"model\":\"%s\",\"task\":\"segmentation\",\"backend\":\"%s\","
        "\"inputs\":{\"text\":\"%s\",\"image_base64\":\"%s\"},\"weights\":",
        model_id, be, tx, ib);
    json_val *weights = args ? json_obj(args, "weights") : NULL;
    if (weights) json_append_value(&b, weights); else sbuf_append(&b, "{}");
    sbuf_append(&b, ",\"params\":");
    json_val *params = args ? json_obj(args, "params") : NULL;
    if (params) json_append_value(&b, params);
    else {
        double st = args ? json_f64(args, "score_threshold", 0.3) : 0.3;
        double mt = args ? json_f64(args, "mask_threshold",  0.5) : 0.5;
        int mm    = args ? json_int(args, "max_masks",       16)  : 16;
        sbuf_printf(&b, "{\"score_threshold\":%.3f,\"mask_threshold\":%.3f,\"max_masks\":%d}",
                    st, mt, mm);
    }
    sbuf_append(&b, "}");
    free(be); free(tx); free(ib);
    return b.ptr;
}

static char *mcp_error_result_json(const char *idbuf, const char *message) {
    char *em = json_escape_dup(message ? message : "error");
    sbuf out;
    sbuf_init(&out);
    sbuf_printf(&out,
        "{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"isError\":true,"
        "\"content\":[{\"type\":\"text\",\"text\":\"%s\"}]}}",
        idbuf, em);
    free(em);
    return out.ptr;
}

static char *mcp_qwen_result_from_infer(const char *idbuf, const char *infer_resp, int status) {
    json_val *root = json_parse(infer_resp, (int)strlen(infer_resp));
    if (!root || root->type != JSON_OBJECT) {
        if (root) json_free(root);
        return mcp_error_result_json(idbuf, "failed to parse inference response");
    }
    json_val *okv = json_obj_get(root, "ok");
    int ok = (okv && okv->type == JSON_TRUE);
    if (!ok || status != 200) {
        json_val *err = json_obj(root, "error");
        const char *msg = err ? json_str(err, "message", "inference failed") : "inference failed";
        char *msg_copy = xstrdup(msg ? msg : "inference failed");
        json_free(root);
        char *out = mcp_error_result_json(idbuf, msg_copy ? msg_copy : "inference failed");
        free(msg_copy);
        return out;
    }

    json_val *outputs = json_obj_get(root, "outputs");
    if (!outputs || outputs->type != JSON_ARRAY || outputs->arr.count <= 0) {
        json_free(root);
        return mcp_error_result_json(idbuf, "inference response has no outputs");
    }
    json_val *first = &outputs->arr.items[0];
    if (first->type != JSON_OBJECT) {
        json_free(root);
        return mcp_error_result_json(idbuf, "unexpected output format");
    }
    const char *mime = json_str(first, "mime", "image/png");
    const char *b64 = json_str(first, "data_base64", "");
    int width = json_int(first, "width", 0);
    int height = json_int(first, "height", 0);
    if (!b64 || !b64[0]) {
        json_free(root);
        return mcp_error_result_json(idbuf, "image output missing data");
    }

    char *emime = json_escape_dup(mime);
    sbuf out;
    sbuf_init(&out);
    sbuf_printf(&out,
        "{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"isError\":false,"
        "\"content\":[{\"type\":\"text\",\"text\":\"generated image %dx%d\"},"
        "{\"type\":\"image\",\"mimeType\":\"%s\",\"data\":\"%s\"}]}}",
        idbuf, width, height, emime, b64);
    free(emime);
    json_free(root);
    return out.ptr;
}

/* Wrap a sam3 segmentation infer response into MCP tool-call result form. */
static char *mcp_sam3_result_from_infer(const char *idbuf, const char *infer_resp, int status) {
    json_val *root = json_parse(infer_resp, (int)strlen(infer_resp));
    if (!root || root->type != JSON_OBJECT) {
        if (root) json_free(root);
        return mcp_error_result_json(idbuf, "failed to parse inference response");
    }
    json_val *okv = json_obj_get(root, "ok");
    int ok = (okv && okv->type == JSON_TRUE);
    if (!ok || status != 200) {
        json_val *err = json_obj(root, "error");
        const char *msg = err ? json_str(err, "message", "inference failed") : "inference failed";
        char *msg_copy = xstrdup(msg ? msg : "inference failed");
        json_free(root);
        char *out = mcp_error_result_json(idbuf, msg_copy ? msg_copy : "inference failed");
        free(msg_copy);
        return out;
    }
    json_val *outputs = json_obj_get(root, "outputs");
    int nm = (outputs && outputs->type == JSON_ARRAY) ? outputs->arr.count : 0;
    sbuf out; sbuf_init(&out);
    sbuf_printf(&out,
        "{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"isError\":false,"
        "\"content\":[{\"type\":\"text\",\"text\":\"sam3: %d masks\"}",
        idbuf, nm);
    for (int i = 0; i < nm; i++) {
        json_val *m = &outputs->arr.items[i];
        if (m->type != JSON_OBJECT) continue;
        const char *mime = json_str(m, "mime", "image/png");
        const char *b64  = json_str(m, "data_base64", "");
        if (!b64 || !b64[0]) continue;
        char *em = json_escape_dup(mime);
        sbuf_printf(&out,
            ",{\"type\":\"image\",\"mimeType\":\"%s\",\"data\":\"%s\"}", em, b64);
        free(em);
    }
    sbuf_append(&out, "]}}");
    json_free(root);
    return out.ptr;
}

static char *infer_json(const server_config *cfg, const char *body, size_t body_len, int *status) {
    *status = 200;
    json_val *root = json_parse(body, (int)body_len);
    if (!root || root->type != JSON_OBJECT) {
        if (root) json_free(root);
        *status = 400;
        return json_error_response("invalid_json", "request body must be a JSON object");
    }

    const char *model = json_str(root, "model", "");
    const char *task = json_str(root, "task", "");
    const char *backend = json_str(root, "backend", "cpu");
    json_val *inputs = json_obj(root, "inputs");
    json_val *weights = json_obj(root, "weights");
    json_val *params = json_obj(root, "params");

    const char *prompt = inputs ? json_str(inputs, "text", "") : "";
    /* Variant lookup: if params.variant matches a registered qwen variant,
     * use its auto-detected weight paths as fallbacks. Explicit per-request
     * weights.{dit,vae,text_encoder} still override. */
    const char *variant_name = params ? json_str(params, "variant", "") : "";
    const qwen_variant *qv = qwen_variants_find(cfg, variant_name);
    const char *qv_dit  = qv ? qv->dit  : cfg->qwen_dit;
    const char *qv_vae  = qv ? qv->vae  : cfg->qwen_vae;
    const char *qv_enc  = qv ? qv->enc  : cfg->qwen_enc;
    const char *qv_bias = qv && qv->bias ? qv->bias : cfg->qwen_enc_bias;
    const char *dit = weights ? json_str(weights, "dit", qv_dit) : qv_dit;
    const char *vae = weights ? json_str(weights, "vae", qv_vae) : qv_vae;
    const char *enc = weights ? json_str(weights, "text_encoder", qv_enc) : qv_enc;
    const char *bias = weights ? json_str(weights, "text_encoder_bias", qv_bias) : qv_bias;
    int width = params ? json_int(params, "width", 256) : 256;
    int height = params ? json_int(params, "height", 256) : 256;
    int steps = params ? json_int(params, "steps", 20) : 20;
    int device = params ? json_int(params, "device", cfg->device) : cfg->device;
    uint64_t seed = params ? (uint64_t)json_u64(params, "seed", 42) : 42;

    char err[1024] = {0};
    image_result img = {0};
    int rc = -1;

    if (strcmp(model, "qwen-image") == 0 && strcmp(task, "text-to-image") == 0) {
        if (width <= 0 || height <= 0 || width % 16 != 0 || height % 16 != 0 || steps <= 0) {
            *status = 400;
            json_free(root);
            return json_error_response("invalid_params", "width and height must be positive multiples of 16, steps must be positive");
        }
        if ((strcmp(backend, "cpu") == 0 || strcmp(backend, "hip") == 0) &&
            (!dit || !dit[0] || !vae || !vae[0] || !enc || !enc[0])) {
            *status = 400;
            snprintf(err, sizeof(err), "qwen-image backend %s requires dit, vae, and text_encoder paths", backend);
        } else
        if (pthread_mutex_trylock(&g_infer_mu) != 0) {
            snprintf(err, sizeof(err), "another inference request is currently running");
            *status = 429;
        } else {
            if (strcmp(backend, "cpu") == 0) {
                rc = qwen_cpu_generate(dit, vae, enc, bias, prompt, height, width, steps, seed, &img, err, sizeof(err));
            } else if (strcmp(backend, "hip") == 0) {
                rc = qwen_hip_generate(dit, vae, enc, prompt, height, width, steps, seed, device, &img, err, sizeof(err));
            } else if (strcmp(backend, "cuda") == 0 || strcmp(backend, "vulkan") == 0) {
                snprintf(err, sizeof(err), "qwen-image backend %s is parsed but not implemented in this server build", backend);
                *status = 501;
            } else {
                snprintf(err, sizeof(err), "unknown backend: %s", backend);
                *status = 400;
            }
            pthread_mutex_unlock(&g_infer_mu);
        }
    } else if ((strcmp(model, "sam3") == 0 || strcmp(model, "sam3.1") == 0)
               && strcmp(task, "segmentation") == 0) {
#if defined(DIFFUSION_SERVER_ENABLE_SAM3)
        if (strcmp(model, "sam3.1") == 0) {
            /* sam3.1 only has a CUDA runner (see cuda/sam3.1/). CPU path is
             * not implemented. */
            if (strcmp(backend, "cuda") != 0) {
                snprintf(err, sizeof(err),
                    "sam3.1 backend '%s' not implemented; only backend=cuda is supported",
                    backend);
                *status = 501;
            } else {
#if defined(DIFFUSION_SERVER_ENABLE_SAM3_1_CUDA)
                const char *ckpt = weights ? json_str(weights, "ckpt",
                                        cfg->sam3_ckpt_v31 ? cfg->sam3_ckpt_v31 : "")
                                           : (cfg->sam3_ckpt_v31 ? cfg->sam3_ckpt_v31 : "");
                const char *vocab = weights ? json_str(weights, "vocab",
                                        cfg->sam3_vocab ? cfg->sam3_vocab : "")
                                            : (cfg->sam3_vocab ? cfg->sam3_vocab : "");
                const char *merges = weights ? json_str(weights, "merges",
                                        cfg->sam3_merges ? cfg->sam3_merges : "")
                                             : (cfg->sam3_merges ? cfg->sam3_merges : "");
                const char *phrase  = inputs ? json_str(inputs, "text", "") : "";
                const char *img_b64 = inputs ? json_str(inputs, "image_base64", "") : "";
                float score_thr = params ? (float)json_f64(params, "score_threshold", 0.3) : 0.3f;
                float mask_thr  = params ? (float)json_f64(params, "mask_threshold",  0.5) : 0.5f;
                int device_ord  = params ? json_int(params, "device", cfg->device) : cfg->device;
                int max_masks   = params ? json_int(params, "max_masks", 16) : 16;
                const char *precision = params ? json_str(params, "precision", "fp16") : "fp16";
                if (max_masks < 1) max_masks = 1;
                if (max_masks > 64) max_masks = 64;

                size_t img_len = 0;
                uint8_t *img_bytes = NULL;
                if (img_b64 && *img_b64) {
                    extern uint8_t *base64_decode_buf(const char *s, size_t in_len, size_t *out_len);
                    img_bytes = base64_decode_buf(img_b64, strlen(img_b64), &img_len);
                }
                if (!img_bytes) {
                    snprintf(err, sizeof(err), "inputs.image_base64 missing or invalid");
                    *status = 400;
                } else if (pthread_mutex_trylock(&g_infer_mu) != 0) {
                    free(img_bytes);
                    snprintf(err, sizeof(err), "another inference request is currently running");
                    *status = 429;
                } else {
                    extern int server_sam3_1_cuda_segment(const char *, const char *, const char *,
                                                           const uint8_t *, size_t, const char *,
                                                           float, float, int, const char *,
                                                           server_sam3_mask *, int, int *,
                                                           char *, size_t);
                    server_sam3_mask *masks = (server_sam3_mask *)calloc((size_t)max_masks, sizeof(*masks));
                    int n_out = 0;
                    int src = server_sam3_1_cuda_segment(
                        ckpt, vocab, merges, img_bytes, img_len, phrase,
                        score_thr, mask_thr, device_ord, precision,
                        masks, max_masks, &n_out, err, sizeof(err));
                    free(img_bytes);
                    pthread_mutex_unlock(&g_infer_mu);
                    if (src != 0) {
                        for (int i = 0; i < n_out; i++) server_sam3_free_mask(&masks[i]);
                        free(masks);
                        *status = 500;
                    } else {
                        char *mm = json_escape_dup(model);
                        char *tt = json_escape_dup(task);
                        char *bb = json_escape_dup(backend);
                        sbuf out; sbuf_init(&out);
                        sbuf_printf(&out,
                            "{\"ok\":true,\"model\":\"%s\",\"task\":\"%s\",\"backend\":\"%s\","
                            "\"outputs\":[", mm, tt, bb);
                        for (int i = 0; i < n_out; i++) {
                            char *b64 = base64_encode(masks[i].png, (size_t)masks[i].png_len);
                            if (!b64) { b64 = xstrdup(""); }
                            sbuf_printf(&out,
                                "%s{\"type\":\"mask\",\"mime\":\"image/png\","
                                "\"width\":%d,\"height\":%d,\"score\":%.6f,"
                                "\"box\":[%.3f,%.3f,%.3f,%.3f],"
                                "\"data_base64\":\"%s\"}",
                                i ? "," : "",
                                masks[i].width, masks[i].height, masks[i].score,
                                masks[i].box[0], masks[i].box[1], masks[i].box[2], masks[i].box[3],
                                b64);
                            free(b64);
                            server_sam3_free_mask(&masks[i]);
                        }
                        free(masks);
                        sbuf_printf(&out,
                            "],\"num_masks\":%d,\"timings_ms\":{\"total\":0}}", n_out);
                        free(mm); free(tt); free(bb);
                        json_free(root);
                        return out.ptr;
                    }
                }
#else
                snprintf(err, sizeof(err),
                    "sam3.1 cuda backend not compiled "
                    "(rebuild with -DDIFFUSION_SERVER_ENABLE_SAM3_1_CUDA=ON)");
                *status = 501;
#endif
            }
        } else if (strcmp(backend, "cpu") != 0 && strcmp(backend, "cuda") != 0) {
            snprintf(err, sizeof(err),
                "sam3 backend '%s' not yet wired into the server; use backend=cpu or cuda", backend);
            *status = 501;
        } else if (strcmp(backend, "cuda") == 0) {
#if defined(DIFFUSION_SERVER_ENABLE_SAM3_CUDA)
            const char *ckpt = weights ? json_str(weights, "ckpt",
                                    cfg->sam3_ckpt ? cfg->sam3_ckpt : "")
                                       : (cfg->sam3_ckpt ? cfg->sam3_ckpt : "");
            const char *vocab = weights ? json_str(weights, "vocab",
                                    cfg->sam3_vocab ? cfg->sam3_vocab : "")
                                        : (cfg->sam3_vocab ? cfg->sam3_vocab : "");
            const char *merges = weights ? json_str(weights, "merges",
                                    cfg->sam3_merges ? cfg->sam3_merges : "")
                                         : (cfg->sam3_merges ? cfg->sam3_merges : "");
            const char *phrase  = inputs ? json_str(inputs, "text", "") : "";
            const char *img_b64 = inputs ? json_str(inputs, "image_base64", "") : "";
            float score_thr = params ? (float)json_f64(params, "score_threshold", 0.3) : 0.3f;
            float mask_thr  = params ? (float)json_f64(params, "mask_threshold",  0.5) : 0.5f;
            int device_ord  = params ? json_int(params, "device", cfg->device) : cfg->device;
            int max_masks   = params ? json_int(params, "max_masks", 16) : 16;
            const char *precision = params ? json_str(params, "precision", "fp16") : "fp16";
            if (max_masks < 1) max_masks = 1;
            if (max_masks > 64) max_masks = 64;

            size_t img_len = 0;
            uint8_t *img_bytes = NULL;
            if (img_b64 && *img_b64) {
                extern uint8_t *base64_decode_buf(const char *s, size_t in_len, size_t *out_len);
                img_bytes = base64_decode_buf(img_b64, strlen(img_b64), &img_len);
            }
            if (!img_bytes) {
                snprintf(err, sizeof(err), "inputs.image_base64 missing or invalid");
                *status = 400;
            } else if (pthread_mutex_trylock(&g_infer_mu) != 0) {
                free(img_bytes);
                snprintf(err, sizeof(err), "another inference request is currently running");
                *status = 429;
            } else {
                server_sam3_mask *masks = (server_sam3_mask *)calloc((size_t)max_masks, sizeof(*masks));
                int n_out = 0;
                int src = server_sam3_cuda_segment(
                    ckpt, vocab, merges, img_bytes, img_len, phrase,
                    score_thr, mask_thr, device_ord, precision,
                    masks, max_masks, &n_out, err, sizeof(err));
                free(img_bytes);
                pthread_mutex_unlock(&g_infer_mu);
                if (src != 0) {
                    for (int i = 0; i < n_out; i++) server_sam3_free_mask(&masks[i]);
                    free(masks);
                    *status = 500;
                } else {
                    char *mm = json_escape_dup(model);
                    char *tt = json_escape_dup(task);
                    char *bb = json_escape_dup(backend);
                    sbuf out; sbuf_init(&out);
                    sbuf_printf(&out,
                        "{\"ok\":true,\"model\":\"%s\",\"task\":\"%s\",\"backend\":\"%s\","
                        "\"outputs\":[", mm, tt, bb);
                    for (int i = 0; i < n_out; i++) {
                        char *b64 = base64_encode(masks[i].png, (size_t)masks[i].png_len);
                        if (!b64) { b64 = xstrdup(""); }
                        sbuf_printf(&out,
                            "%s{\"type\":\"mask\",\"mime\":\"image/png\","
                            "\"width\":%d,\"height\":%d,\"score\":%.6f,"
                            "\"box\":[%.3f,%.3f,%.3f,%.3f],"
                            "\"data_base64\":\"%s\"}",
                            i ? "," : "",
                            masks[i].width, masks[i].height, masks[i].score,
                            masks[i].box[0], masks[i].box[1], masks[i].box[2], masks[i].box[3],
                            b64);
                        free(b64);
                        server_sam3_free_mask(&masks[i]);
                    }
                    free(masks);
                    sbuf_printf(&out,
                        "],\"num_masks\":%d,\"timings_ms\":{\"total\":0}}", n_out);
                    free(mm); free(tt); free(bb);
                    json_free(root);
                    return out.ptr;
                }
            }
#else
            snprintf(err, sizeof(err),
                "sam3 cuda backend not compiled (rebuild with -DDIFFUSION_SERVER_ENABLE_SAM3_CUDA=ON)");
            *status = 501;
#endif
        } else {
            const char *ckpt = weights ? json_str(weights, "ckpt",
                                    cfg->sam3_ckpt ? cfg->sam3_ckpt : "")
                                       : (cfg->sam3_ckpt ? cfg->sam3_ckpt : "");
            const char *vocab = weights ? json_str(weights, "vocab",
                                    cfg->sam3_vocab ? cfg->sam3_vocab : "")
                                        : (cfg->sam3_vocab ? cfg->sam3_vocab : "");
            const char *merges = weights ? json_str(weights, "merges",
                                    cfg->sam3_merges ? cfg->sam3_merges : "")
                                         : (cfg->sam3_merges ? cfg->sam3_merges : "");
            const char *phrase = inputs ? json_str(inputs, "text", "") : "";
            const char *img_b64 = inputs ? json_str(inputs, "image_base64", "") : "";
            float score_thr = params ? (float)json_f64(params, "score_threshold", 0.3) : 0.3f;
            float mask_thr  = params ? (float)json_f64(params, "mask_threshold",  0.5) : 0.5f;
            int threads     = params ? json_int(params, "threads", 0) : 0;
            int max_masks   = params ? json_int(params, "max_masks", 16) : 16;
            if (max_masks < 1) max_masks = 1;
            if (max_masks > 64) max_masks = 64;

            size_t img_len = 0;
            uint8_t *img_bytes = NULL;
            if (img_b64 && *img_b64) {
                extern uint8_t *base64_decode_buf(const char *s, size_t in_len, size_t *out_len);
                img_bytes = base64_decode_buf(img_b64, strlen(img_b64), &img_len);
            }
            if (!img_bytes) {
                snprintf(err, sizeof(err), "inputs.image_base64 missing or invalid");
                *status = 400;
            } else if (pthread_mutex_trylock(&g_infer_mu) != 0) {
                free(img_bytes);
                snprintf(err, sizeof(err), "another inference request is currently running");
                *status = 429;
            } else {
                server_sam3_mask *masks = (server_sam3_mask *)calloc((size_t)max_masks, sizeof(*masks));
                int n_out = 0;
                int src = server_sam3_cpu_segment(
                    ckpt, vocab, merges, img_bytes, img_len, phrase,
                    score_thr, mask_thr, threads,
                    masks, max_masks, &n_out, err, sizeof(err));
                free(img_bytes);
                pthread_mutex_unlock(&g_infer_mu);
                if (src != 0) {
                    for (int i = 0; i < n_out; i++) server_sam3_free_mask(&masks[i]);
                    free(masks);
                    *status = 500;
                } else {
                    /* Build success JSON here and early-return. */
                    char *mm = json_escape_dup(model);
                    char *tt = json_escape_dup(task);
                    char *bb = json_escape_dup(backend);
                    sbuf out; sbuf_init(&out);
                    sbuf_printf(&out,
                        "{\"ok\":true,\"model\":\"%s\",\"task\":\"%s\",\"backend\":\"%s\","
                        "\"outputs\":[", mm, tt, bb);
                    for (int i = 0; i < n_out; i++) {
                        char *b64 = base64_encode(masks[i].png, (size_t)masks[i].png_len);
                        if (!b64) { b64 = xstrdup(""); }
                        sbuf_printf(&out,
                            "%s{\"type\":\"mask\",\"mime\":\"image/png\","
                            "\"width\":%d,\"height\":%d,\"score\":%.6f,"
                            "\"box\":[%.3f,%.3f,%.3f,%.3f],"
                            "\"data_base64\":\"%s\"}",
                            i ? "," : "",
                            masks[i].width, masks[i].height, masks[i].score,
                            masks[i].box[0], masks[i].box[1], masks[i].box[2], masks[i].box[3],
                            b64);
                        free(b64);
                        server_sam3_free_mask(&masks[i]);
                    }
                    free(masks);
                    sbuf_printf(&out,
                        "],\"num_masks\":%d,\"timings_ms\":{\"total\":0}}", n_out);
                    free(mm); free(tt); free(bb);
                    json_free(root);
                    return out.ptr;
                }
            }
        }
#else
        snprintf(err, sizeof(err),
            "sam3 support not compiled (rebuild with -DDIFFUSION_SERVER_ENABLE_SAM3=ON)");
        *status = 501;
#endif
    } else {
        snprintf(err, sizeof(err), "unsupported model/task: %s/%s", model, task);
        *status = 400;
    }

    json_free(root);

    if (rc != 0) {
        if (*status == 200) *status = 500;
        if (*status == 501) return json_error_response("not_implemented", err);
        if (*status == 429) return json_error_response("busy", err);
        if (*status == 400) return json_error_response("bad_request", err);
        return json_error_response("inference_failed", err);
    }

    char *b64 = base64_encode(img.png, (size_t)img.png_len);
    free(img.png);
    if (!b64) {
        *status = 500;
        return json_error_response("encode_failed", "failed to base64 encode image");
    }
    char *m = json_escape_dup(model);
    char *t = json_escape_dup(task);
    char *be = json_escape_dup(backend);
    sbuf out;
    sbuf_init(&out);
    sbuf_printf(&out,
        "{\"ok\":true,\"model\":\"%s\",\"task\":\"%s\",\"backend\":\"%s\","
        "\"outputs\":[{\"type\":\"image\",\"mime\":\"image/png\",\"width\":%d,\"height\":%d,"
        "\"data_base64\":\"%s\"}],\"timings_ms\":{\"total\":0}}",
        m, t, be, img.width, img.height, b64);
    free(m); free(t); free(be); free(b64);
    return out.ptr;
}

static const char *mime_for_path(const char *path) {
    if (has_suffix(path, ".html")) return "text/html; charset=utf-8";
    if (has_suffix(path, ".css")) return "text/css; charset=utf-8";
    if (has_suffix(path, ".js")) return "application/javascript; charset=utf-8";
    if (has_suffix(path, ".png")) return "image/png";
    if (has_suffix(path, ".jpg") || has_suffix(path, ".jpeg")) return "image/jpeg";
    if (has_suffix(path, ".webp")) return "image/webp";
    if (has_suffix(path, ".glb")) return "model/gltf-binary";
    if (has_suffix(path, ".gltf")) return "model/gltf+json";
    if (has_suffix(path, ".obj")) return "text/plain; charset=utf-8";
    if (has_suffix(path, ".ply")) return "application/octet-stream";
    return "application/octet-stream";
}

static void send_all(int fd, const void *data, size_t len) {
    const char *p = (const char *)data;
    while (len) {
        ssize_t n = send(fd, p, len, 0);
        if (n <= 0) return;
        p += n;
        len -= (size_t)n;
    }
}

static void send_response(int fd, int status, const char *ctype, const void *body, size_t body_len) {
    const char *reason = status == 200 ? "OK" :
                         status == 400 ? "Bad Request" :
                         status == 404 ? "Not Found" :
                         status == 405 ? "Method Not Allowed" :
                         status == 429 ? "Too Many Requests" :
                         status == 413 ? "Payload Too Large" :
                         status == 501 ? "Not Implemented" : "Internal Server Error";
    char hdr[512];
    int n = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: content-type\r\n"
        "Access-Control-Allow-Methods: GET,POST,OPTIONS\r\n"
        "Connection: close\r\n\r\n",
        status, reason, ctype, body_len);
    send_all(fd, hdr, (size_t)n);
    if (body_len) send_all(fd, body, body_len);
}

static char *models_json(const server_config *cfg) {
    sbuf out;
    sbuf_init(&out);
    sbuf_append(&out, "{\"ok\":true,\"models\":[");
    /* qwen-image with variants (if registered) */
    sbuf_append(&out,
        "{\"id\":\"qwen-image\",\"tasks\":[\"text-to-image\"],"
        "\"backends\":[\"cpu\""
#if defined(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_HIP)
        ",\"hip\""
#endif
        "],\"variants\":[");
    if (cfg) {
        for (int i = 0; i < cfg->qwen_variant_count; i++) {
            const qwen_variant *v = &cfg->qwen_variants[i];
            char *en = json_escape_dup(v->name);
            char *ek = json_escape_dup(v->kind ? v->kind : "");
            sbuf_printf(&out, "%s{\"name\":\"%s\",\"kind\":\"%s\"}",
                        i ? "," : "", en, ek);
            free(en); free(ek);
        }
    }
    sbuf_append(&out, "]},");
    sbuf_append(&out,
        "{\"id\":\"sam3\",\"tasks\":[\"segmentation\"],\"backends\":["
        "\"cpu\""
#if defined(DIFFUSION_SERVER_ENABLE_SAM3_CUDA)
        ",\"cuda\""
#endif
        "],\"status\":\""
#if defined(DIFFUSION_SERVER_ENABLE_SAM3)
        "ready"
#else
        "stub"
#endif
        "\"},");
    sbuf_append(&out,
        "{\"id\":\"sam3.1\",\"tasks\":[\"segmentation\"],\"backends\":[\"cpu\",\"cuda\"],"
        "\"status\":\"pending_runner\",\"note\":\"see cuda/sam3.1/PORT.md\"}"
        "]}");
    return out.ptr;
}

static char *health_json(void) {
    sbuf b;
    sbuf_init(&b);
    sbuf_printf(&b,
        "{\"ok\":true,\"server\":\"diffusion-server\",\"compiled\":{\"qwen_image\":%s,"
        "\"qwen_image_cpu\":%s,\"qwen_image_hip\":%s,\"sam3\":%s,\"mcp\":%s}}",
#if defined(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE)
        "true",
#else
        "false",
#endif
#if defined(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_CPU)
        "true",
#else
        "false",
#endif
#if defined(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_HIP)
        "true",
#else
        "false",
#endif
#if defined(DIFFUSION_SERVER_ENABLE_SAM3)
        "true",
#else
        "false",
#endif
#if defined(DIFFUSION_SERVER_ENABLE_MCP)
        "true"
#else
        "false"
#endif
    );
    return b.ptr;
}

static int read_file(const char *path, bbuf *out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    uint8_t buf[READ_CHUNK];
    for (;;) {
        size_t n = fread(buf, 1, sizeof(buf), fp);
        if (n) bbuf_append(out, buf, n);
        if (n < sizeof(buf)) break;
    }
    int ok = ferror(fp) ? -1 : 0;
    fclose(fp);
    return ok;
}

static int path_is_safe(const char *p) {
    return p && !strstr(p, "..") && !strchr(p, '\\');
}

static void handle_static(int fd, const server_config *cfg, const char *url_path) {
    const char *rel = url_path;
    if (strcmp(rel, "/") == 0 || strcmp(rel, "/index.html") == 0) rel = "/index.html";
    else if (strcmp(rel, "/sam3") == 0)   rel = "/sam3.html";
    else if (strcmp(rel, "/sam3.1") == 0) rel = "/sam3.1.html";
    else if (strcmp(rel, "/sam3_compare") == 0) rel = "/sam3_compare.html";
    else if (strcmp(rel, "/sam3_ref") == 0) rel = "/sam3_ref.html";
    else if (strcmp(rel, "/sam3_1_ref") == 0) rel = "/sam3_1_ref.html";
    else if (strcmp(rel, "/qwen-image") == 0) rel = "/qwen-image.html";
    else if (strcmp(rel, "/qwen_image_compare") == 0) rel = "/qwen_image_compare.html";
    else if (strcmp(rel, "/flux2") == 0) rel = "/flux2.html";
    else if (strcmp(rel, "/flux2_compare") == 0) rel = "/flux2_compare.html";
    else if (strcmp(rel, "/hy3d") == 0) rel = "/hy3d.html";
    else if (strcmp(rel, "/hy3d_compare") == 0) rel = "/hy3d.html";
    else if (strcmp(rel, "/trellis2") == 0) rel = "/trellis2.html";
    else if (strcmp(rel, "/trellis2_compare") == 0) rel = "/trellis2.html";
    else if (strcmp(rel, "/llm") == 0) rel = "/llm_chat.html";
    else if (strcmp(rel, "/llm_chat") == 0) rel = "/llm_chat.html";
    if (!path_is_safe(rel)) {
        char *e = json_error_response("bad_path", "invalid static path");
        send_response(fd, 400, "application/json", e, strlen(e));
        free(e);
        return;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s%s", cfg->web_root, rel);
    bbuf body;
    bbuf_init(&body);
    if (read_file(path, &body) != 0) {
        const char msg[] = "not found";
        send_response(fd, 404, "text/plain; charset=utf-8", msg, sizeof(msg) - 1);
    } else {
        send_response(fd, 200, mime_for_path(path), body.data, body.len);
    }
    bbuf_free(&body);
}

static char *find_header_end(const char *data, size_t len) {
    for (size_t i = 3; i < len; i++) {
        if (data[i-3] == '\r' && data[i-2] == '\n' && data[i-1] == '\r' && data[i] == '\n')
            return (char *)(data + i + 1);
    }
    return NULL;
}

static int starts_with_icase(const char *s, const char *prefix) {
    while (*prefix) {
        unsigned char a = (unsigned char)*s++;
        unsigned char b = (unsigned char)*prefix++;
        if ((unsigned char)tolower(a) != (unsigned char)tolower(b)) return 0;
    }
    return 1;
}

static size_t parse_content_length(const char *headers) {
    const char *p = headers;
    while (*p) {
        const char *line = p;
        const char *nl = strchr(line, '\n');
        size_t line_len = nl ? (size_t)(nl - line) : strlen(line);
        if (line_len >= 15 && starts_with_icase(line, "content-length:")) {
            p = line + 15;
            while (*p == ' ' || *p == '\t') p++;
            return (size_t)strtoull(p, NULL, 10);
        }
        if (!nl) break;
        p = nl + 1;
    }
    return 0;
#if 0
    while ((p = strcasestr(p, "content-length:")) != NULL) {
        if (p == headers || p[-1] == '\n') {
            p += 15;
            while (*p == ' ' || *p == '\t') p++;
            return (size_t)strtoull(p, NULL, 10);
        }
        p += 15;
    }
    return 0;
#endif
}

/* ------------------------------------------------------------------ *
 * Ref-server HTTP proxy
 *
 * Parses http://host[:port] from ref_url, opens a TCP connection,
 * writes METHOD upstream_path HTTP/1.1 + headers + body, then reads the
 * response and forwards the body back to the browser with our CORS
 * headers. Deliberately minimal — no chunked encoding, no keep-alive.
 * ------------------------------------------------------------------ */
static int parse_http_url(const char *url, char *host, size_t host_cap, int *port_out, const char **tail) {
    if (!url) return -1;
    const char *p = url;
    if (strncmp(p, "http://", 7) == 0) p += 7;
    else if (strncmp(p, "https://", 8) == 0) return -1; /* TLS not supported */
    const char *slash = strchr(p, '/');
    const char *hostend = slash ? slash : p + strlen(p);
    const char *colon = memchr(p, ':', (size_t)(hostend - p));
    size_t hlen = (size_t)((colon ? colon : hostend) - p);
    if (hlen == 0 || hlen >= host_cap) return -1;
    memcpy(host, p, hlen); host[hlen] = 0;
    *port_out = colon ? atoi(colon + 1) : 80;
    if (tail) *tail = slash ? slash : "";
    return 0;
}

static int tcp_connect(const char *host, int port) {
    char portstr[16]; snprintf(portstr, sizeof(portstr), "%d", port);
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, portstr, &hints, &res) != 0 || !res) return -1;
    int s = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s < 0) continue;
        if (connect(s, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(s); s = -1;
    }
    freeaddrinfo(res);
    return s;
}

/* Send a fixed-body request and read the response into resp (including
 * status line + headers + body). Returns 0 on success. */
static int http_forward(const char *ref_url, const char *method,
                        const char *upstream_path,
                        const char *body, size_t body_len,
                        bbuf *resp) {
    char host[256]; int port = 0; const char *tail = NULL;
    if (parse_http_url(ref_url, host, sizeof(host), &port, &tail) != 0) return -1;
    int s = tcp_connect(host, port);
    if (s < 0) return -2;
    char hdr[1024];
    int n = snprintf(hdr, sizeof(hdr),
        "%s %s HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n\r\n",
        method, upstream_path, host, port, body_len);
    if (n <= 0 || (size_t)n >= sizeof(hdr)) { close(s); return -3; }
    send_all(s, hdr, (size_t)n);
    if (body_len) send_all(s, body, body_len);
    char tmp[READ_CHUNK];
    for (;;) {
        ssize_t r = recv(s, tmp, sizeof(tmp), 0);
        if (r <= 0) break;
        bbuf_append(resp, tmp, (size_t)r);
        if (resp->len > MAX_BODY_BYTES) break;
    }
    close(s);
    return 0;
}

/* Extract body + content-type + status from a full HTTP response.
 * Returns pointer into resp->data (no copy). */
static const char *parse_http_response(const bbuf *resp, int *status_out,
                                       char *ctype, size_t ctype_cap,
                                       size_t *body_off, size_t *body_len) {
    const char *d = (const char *)resp->data;
    size_t L = resp->len;
    if (L < 12 || strncmp(d, "HTTP/", 5) != 0) return NULL;
    const char *sp = memchr(d, ' ', L);
    if (!sp) return NULL;
    *status_out = atoi(sp + 1);
    /* find header end */
    size_t hend = 0;
    for (size_t i = 3; i < L; i++) {
        if (d[i-3] == '\r' && d[i-2] == '\n' && d[i-1] == '\r' && d[i] == '\n') {
            hend = i + 1; break;
        }
    }
    if (!hend) return NULL;
    ctype[0] = 0;
    const char *p = d;
    while (p < d + hend) {
        const char *eol = memchr(p, '\n', (size_t)(d + hend - p));
        if (!eol) break;
        size_t ll = (size_t)(eol - p);
        if (ll > 14 && strncasecmp(p, "content-type:", 13) == 0) {
            const char *v = p + 13;
            while (v < eol && (*v == ' ' || *v == '\t')) v++;
            size_t vl = (size_t)(eol - v);
            while (vl && (v[vl-1] == '\r' || v[vl-1] == ' ')) vl--;
            if (vl >= ctype_cap) vl = ctype_cap - 1;
            memcpy(ctype, v, vl); ctype[vl] = 0;
        }
        p = eol + 1;
    }
    *body_off = hend;
    *body_len = L - hend;
    return d;
}

static void proxy_ref(int fd, const char *ref_url, const char *method,
                      const char *upstream_path,
                      const char *body, size_t body_len) {
    if (!ref_url || !*ref_url) {
        char *e = json_error_response("ref_not_configured",
            "ref URL not configured — pass --sam3-ref-url / --sam3-1-ref-url to the server");
        send_response(fd, 503, "application/json", e, strlen(e));
        free(e);
        return;
    }
    bbuf resp; bbuf_init(&resp);
    int rc = http_forward(ref_url, method, upstream_path, body, body_len, &resp);
    if (rc != 0) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "proxy to %s failed (rc=%d) — is the ref server running?", ref_url, rc);
        char *e = json_error_response("ref_unreachable", msg);
        send_response(fd, 502, "application/json", e, strlen(e));
        free(e);
        bbuf_free(&resp);
        return;
    }
    int status = 500;
    char ctype[128] = "application/json";
    size_t body_off = 0, rlen = 0;
    const char *raw = parse_http_response(&resp, &status, ctype, sizeof(ctype), &body_off, &rlen);
    if (!raw) {
        char *e = json_error_response("ref_bad_response", "malformed response from ref server");
        send_response(fd, 502, "application/json", e, strlen(e));
        free(e);
        bbuf_free(&resp);
        return;
    }
    send_response(fd, status, ctype, raw + body_off, rlen);
    bbuf_free(&resp);
}

static void handle_client(int fd, const server_config *cfg) {
    bbuf req;
    bbuf_init(&req);
    char tmp[READ_CHUNK];
    char *header_end = NULL;
    size_t header_len = 0, content_len = 0;
    while (!header_end) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) { bbuf_free(&req); return; }
        bbuf_append(&req, tmp, (size_t)n);
        header_end = find_header_end((const char *)req.data, req.len);
        if (req.len > 1024 * 1024) { bbuf_free(&req); return; }
    }
    header_len = (size_t)(header_end - (char *)req.data);
    req.data[header_len - 4] = 0;
    content_len = parse_content_length((const char *)req.data);
    if (content_len > MAX_BODY_BYTES) {
        char *e = json_error_response("payload_too_large", "request body exceeds server limit");
        send_response(fd, 413, "application/json", e, strlen(e));
        free(e);
        bbuf_free(&req);
        return;
    }
    while (req.len < header_len + content_len) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        bbuf_append(&req, tmp, (size_t)n);
    }

    char method[16] = {0}, path[512] = {0};
    sscanf((const char *)req.data, "%15s %511s", method, path);
    const char *body = (const char *)req.data + header_len;

    if (strcmp(method, "OPTIONS") == 0) {
        send_response(fd, 200, "text/plain", "", 0);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
        char *j = health_json();
        send_response(fd, 200, "application/json", j, strlen(j));
        free(j);
    } else if (strcmp(method, "GET") == 0 &&
               (strcmp(path, "/models") == 0 || strcmp(path, "/v1/models") == 0)) {
        char *j = models_json(cfg);
        send_response(fd, 200, "application/json", j, strlen(j));
        free(j);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/infer") == 0) {
        int status = 200;
        char *j = infer_json(cfg, body, content_len, &status);
        send_response(fd, status, "application/json", j, strlen(j));
        free(j);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/ref/sam3/infer") == 0) {
        proxy_ref(fd, cfg->sam3_ref_url, "POST", "/v1/infer", body, content_len);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/ref/sam3.1/infer") == 0) {
        proxy_ref(fd, cfg->sam3_1_ref_url, "POST", "/v1/infer", body, content_len);
    } else if (strcmp(method, "GET") == 0 &&
               (strcmp(path, "/v1/ref/sam3/models") == 0 ||
                strcmp(path, "/v1/ref/sam3/health") == 0)) {
        const char *up = strchr(path + 8, '/'); /* /v1/ref/sam3<here> */
        proxy_ref(fd, cfg->sam3_ref_url, "GET", up, NULL, 0);
    } else if (strcmp(method, "GET") == 0 &&
               (strcmp(path, "/v1/ref/sam3.1/models") == 0 ||
                strcmp(path, "/v1/ref/sam3.1/health") == 0)) {
        const char *up = strchr(path + 10, '/'); /* /v1/ref/sam3.1<here> */
        proxy_ref(fd, cfg->sam3_1_ref_url, "GET", up, NULL, 0);
    } else if (strcmp(method, "GET") == 0) {
        handle_static(fd, cfg, path);
    } else {
        char *e = json_error_response("method_not_allowed", "unsupported method");
        send_response(fd, 405, "application/json", e, strlen(e));
        free(e);
    }
    bbuf_free(&req);
}

static int run_http(const server_config *cfg) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)cfg->port);
    if (inet_pton(AF_INET, cfg->host, &addr.sin_addr) != 1) {
        fprintf(stderr, "invalid host: %s\n", cfg->host);
        close(fd);
        return 1;
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("bind");
        close(fd);
        return 1;
    }
    if (listen(fd, 16) != 0) {
        perror("listen");
        close(fd);
        return 1;
    }
    fprintf(stderr, "diffusion-server listening on http://%s:%d\n", cfg->host, cfg->port);
    while (!g_stop) {
        struct sockaddr_in peer;
        socklen_t peer_len = sizeof(peer);
        int cfd = accept(fd, (struct sockaddr *)&peer, &peer_len);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        handle_client(cfd, cfg);
        close(cfd);
    }
    close(fd);
    return 0;
}

static void mcp_write_json(const char *json) {
    printf("%s\n", json);
    fflush(stdout);
}

static char *stdio_wrap(int status, const char *resp_json) {
    sbuf out;
    sbuf_init(&out);
    sbuf_printf(&out, "{\"status\":%d,\"response\":%s}", status, resp_json ? resp_json : "null");
    return out.ptr;
}

static int run_stdio(const server_config *cfg) {
    char line[4 * 1024 * 1024];
    while (fgets(line, sizeof(line), stdin)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ' || line[n - 1] == '\t'))
            line[--n] = 0;
        if (n == 0) continue;

        int status = 200;
        char *resp = NULL;
        json_val *root = json_parse(line, (int)n);
        if (!root || root->type != JSON_OBJECT) {
            if (root) json_free(root);
            status = 400;
            resp = json_error_response("invalid_json", "stdin line must be a JSON object");
        } else {
            const char *cmd = json_str(root, "cmd", "");
            if (strcmp(cmd, "health") == 0) {
                resp = health_json();
            } else if (strcmp(cmd, "models") == 0) {
                resp = models_json(cfg);
            } else {
                char *infer_body = NULL;
                if (strcmp(cmd, "infer") == 0) {
                    json_val *req = json_obj(root, "request");
                    if (!req) {
                        status = 400;
                        resp = json_error_response("bad_request", "cmd=infer requires object field request");
                    } else {
                        sbuf b;
                        sbuf_init(&b);
                        json_append_value(&b, req);
                        infer_body = b.ptr;
                    }
                } else if (cmd[0] == 0) {
                    infer_body = xstrdup(line);
                } else {
                    status = 400;
                    resp = json_error_response("bad_request", "unknown cmd; use health, models, infer, or direct infer JSON");
                }
                if (infer_body) {
                    resp = infer_json(cfg, infer_body, strlen(infer_body), &status);
                    free(infer_body);
                }
            }
            json_free(root);
        }

        char *wrapped = stdio_wrap(status, resp ? resp : "null");
        printf("%s\n", wrapped);
        fflush(stdout);
        free(wrapped);
        free(resp);
    }
    return 0;
}

static int run_mcp_stdio(const server_config *cfg) {
#if defined(DIFFUSION_SERVER_ENABLE_MCP)
    char line[1024 * 1024];
    while (fgets(line, sizeof(line), stdin)) {
        json_val *root = json_parse(line, (int)strlen(line));
        json_val *idv = root && root->type == JSON_OBJECT ? json_obj_get(root, "id") : NULL;
        const char *method = root && root->type == JSON_OBJECT ? json_str(root, "method", "") : "";
        char idbuf[64] = "null";
        if (idv && idv->type == JSON_NUMBER) snprintf(idbuf, sizeof(idbuf), "%.0f", idv->num);
        else if (idv && idv->type == JSON_STRING) snprintf(idbuf, sizeof(idbuf), "\"%s\"", idv->str.ptr);

        if (strcmp(method, "initialize") == 0) {
            printf("{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"diffusion-server\",\"version\":\"0.1\"}}}\n", idbuf);
            fflush(stdout);
        } else if (strcmp(method, "tools/list") == 0) {
            printf("{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"tools\":["
                   "{\"name\":\"qwen_image_generate\",\"description\":\"Generate an image with Qwen-Image\","
                   "\"inputSchema\":{\"type\":\"object\",\"additionalProperties\":false,"
                   "\"properties\":{\"backend\":{\"type\":\"string\",\"enum\":[\"cpu\",\"hip\",\"cuda\",\"vulkan\"]},"
                   "\"prompt\":{\"type\":\"string\"},\"width\":{\"type\":\"integer\",\"minimum\":16},"
                   "\"height\":{\"type\":\"integer\",\"minimum\":16},\"steps\":{\"type\":\"integer\",\"minimum\":1},"
                   "\"seed\":{\"type\":\"integer\",\"minimum\":0},\"device\":{\"type\":\"integer\",\"minimum\":0},"
                   "\"weights\":{\"type\":\"object\",\"additionalProperties\":false,"
                   "\"properties\":{\"dit\":{\"type\":\"string\"},\"vae\":{\"type\":\"string\"},"
                   "\"text_encoder\":{\"type\":\"string\"},\"text_encoder_bias\":{\"type\":\"string\"}}}}"
                   ",\"required\":[\"prompt\"]}},"
                   "{\"name\":\"sam3_segment\",\"description\":\"Segment an image by text phrase with SAM 3 (CPU backend)\","
                   "\"inputSchema\":{\"type\":\"object\",\"additionalProperties\":false,"
                   "\"properties\":{\"backend\":{\"type\":\"string\",\"enum\":[\"cpu\"]},"
                   "\"text\":{\"type\":\"string\"},\"image_base64\":{\"type\":\"string\"},"
                   "\"score_threshold\":{\"type\":\"number\"},\"mask_threshold\":{\"type\":\"number\"},"
                   "\"max_masks\":{\"type\":\"integer\",\"minimum\":1,\"maximum\":64},"
                   "\"weights\":{\"type\":\"object\",\"additionalProperties\":false,"
                   "\"properties\":{\"ckpt\":{\"type\":\"string\"},\"vocab\":{\"type\":\"string\"},"
                   "\"merges\":{\"type\":\"string\"}}}},\"required\":[\"text\",\"image_base64\"]}},"
                   "{\"name\":\"sam3_1_segment\",\"description\":\"Segment with SAM 3.1 (runner port in progress; currently 501)\","
                   "\"inputSchema\":{\"type\":\"object\",\"additionalProperties\":false,"
                   "\"properties\":{\"text\":{\"type\":\"string\"},\"image_base64\":{\"type\":\"string\"}}}}"
                   "]}}\n", idbuf);
            fflush(stdout);
        } else if (strcmp(method, "tools/call") == 0) {
            json_val *params = json_obj(root, "params");
            const char *name = params ? json_str(params, "name", "") : "";
            json_val *args = params ? json_obj(params, "arguments") : NULL;
            if (strcmp(name, "qwen_image_generate") == 0) {
                char *body = build_qwen_infer_body_from_mcp_args(args);
                int status = 200;
                char *resp = infer_json(cfg, body, strlen(body), &status);
                char *mcp_resp = mcp_qwen_result_from_infer(idbuf, resp, status);
                mcp_write_json(mcp_resp);
                free(mcp_resp);
                free(resp);
                free(body);
            } else if (strcmp(name, "sam3_segment") == 0 ||
                       strcmp(name, "sam3_1_segment") == 0) {
                const char *model_id = strcmp(name, "sam3_1_segment") == 0 ? "sam3.1" : "sam3";
                char *body = build_sam3_infer_body_from_mcp_args(args, model_id);
                int status = 200;
                char *resp = infer_json(cfg, body, strlen(body), &status);
                char *mcp_resp = mcp_sam3_result_from_infer(idbuf, resp, status);
                mcp_write_json(mcp_resp);
                free(mcp_resp);
                free(resp);
                free(body);
            } else {
                printf("{\"jsonrpc\":\"2.0\",\"id\":%s,\"error\":{\"code\":-32602,\"message\":\"unknown tool\"}}\n", idbuf);
                fflush(stdout);
            }
        } else if (strcmp(method, "notifications/initialized") == 0) {
        } else {
            printf("{\"jsonrpc\":\"2.0\",\"id\":%s,\"error\":{\"code\":-32601,\"message\":\"method not found\"}}\n", idbuf);
            fflush(stdout);
        }
        if (root) json_free(root);
    }
    return 0;
#else
    (void)cfg;
    mcp_write_json("{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{\"code\":-32601,\"message\":\"MCP support not compiled\"}}");
    return 1;
#endif
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --host <ip>              default 127.0.0.1\n"
        "  --port <n>               default 8080\n"
        "  --web-root <path>        default ../web\n"
        "  --qwen-dit <path>\n"
        "  --qwen-vae <path>\n"
        "  --qwen-enc <path>\n"
        "  --qwen-enc-bias <path>\n"
        "  --qwen-variants <spec>   name:dir,name:dir (or env QWEN_IMAGE_VARIANTS)\n"
        "  --sam3-ckpt <path>       default sam3.model.safetensors\n"
        "  --sam3-ckpt-v31 <path>   default sam3.1.model.safetensors (cuda only)\n"
        "  --sam3-vocab <path>      CLIP BPE vocab.json\n"
        "  --sam3-merges <path>     CLIP BPE merges.txt\n"
        "  --sam3-ref-url <url>     proxy /v1/ref/sam3/*   to URL (pytorch ref, sam3)\n"
        "  --sam3-1-ref-url <url>   proxy /v1/ref/sam3.1/* to URL (pytorch ref, sam3.1)\n"
        "  --device <n>             default 0\n"
        "  --stdio                  line-delimited JSON transport for local debugging\n"
        "  --mcp-stdio              run MCP stdio mode when compiled\n",
        prog);
}

int main(int argc, char **argv) {
    server_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.host = "127.0.0.1";
    cfg.port = 8080;
    cfg.web_root = "../web";
    cfg.device = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) cfg.host = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) cfg.port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--web-root") == 0 && i + 1 < argc) cfg.web_root = argv[++i];
        else if (strcmp(argv[i], "--qwen-dit") == 0 && i + 1 < argc) cfg.qwen_dit = argv[++i];
        else if (strcmp(argv[i], "--qwen-vae") == 0 && i + 1 < argc) cfg.qwen_vae = argv[++i];
        else if (strcmp(argv[i], "--qwen-enc") == 0 && i + 1 < argc) cfg.qwen_enc = argv[++i];
        else if (strcmp(argv[i], "--qwen-enc-bias") == 0 && i + 1 < argc) cfg.qwen_enc_bias = argv[++i];
        else if (strcmp(argv[i], "--qwen-variants") == 0 && i + 1 < argc) cfg.qwen_variants_spec = argv[++i];
        else if (strcmp(argv[i], "--sam3-ckpt") == 0 && i + 1 < argc) cfg.sam3_ckpt = argv[++i];
        else if (strcmp(argv[i], "--sam3-ckpt-v31") == 0 && i + 1 < argc) cfg.sam3_ckpt_v31 = argv[++i];
        else if (strcmp(argv[i], "--sam3-vocab") == 0 && i + 1 < argc) cfg.sam3_vocab = argv[++i];
        else if (strcmp(argv[i], "--sam3-merges") == 0 && i + 1 < argc) cfg.sam3_merges = argv[++i];
        else if (strcmp(argv[i], "--sam3-ref-url") == 0 && i + 1 < argc) cfg.sam3_ref_url = argv[++i];
        else if (strcmp(argv[i], "--sam3-1-ref-url") == 0 && i + 1 < argc) cfg.sam3_1_ref_url = argv[++i];
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) cfg.device = atoi(argv[++i]);
        else if (strcmp(argv[i], "--stdio") == 0) cfg.stdio_mode = 1;
        else if (strcmp(argv[i], "--mcp-stdio") == 0) cfg.mcp_stdio = 1;
        else { usage(argv[0]); return 1; }
    }
    /* Env fallback for variant spec (easier to set in launchers). */
    if (!cfg.qwen_variants_spec) {
        const char *e = getenv("QWEN_IMAGE_VARIANTS");
        if (e && *e) cfg.qwen_variants_spec = e;
    }
    qwen_variants_parse(&cfg, cfg.qwen_variants_spec);
    if (cfg.stdio_mode) return run_stdio(&cfg);
    if (cfg.mcp_stdio) return run_mcp_stdio(&cfg);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    return run_http(&cfg);
}
