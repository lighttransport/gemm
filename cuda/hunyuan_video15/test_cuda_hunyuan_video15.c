#define _POSIX_C_SOURCE 200809L
#include "hunyuan_video15.h"
#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

static volatile sig_atomic_t interrupted;
static const char *output;
static void stop(int sig) { (void)sig; interrupted = 1; }
static int cancelled(void *user) { (void)user; return interrupted != 0; }
static void progress(int step, int total, void *user) {
    (void)user; printf("PROGRESS %d %d\n", step, total); fflush(stdout);
}
static int frame(int index, int width, int height, const unsigned char *rgb, void *user) {
    (void)user;
    char path[PATH_MAX];
    if (snprintf(path, sizeof(path), "%s/frame_%05d.ppm", output, index) >= (int)sizeof(path)) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int ok = fprintf(f, "P6\n%d %d\n255\n", width, height) > 0 &&
        fwrite(rgb, 3, (size_t)width * height, f) == (size_t)width * height;
    if (fclose(f)) ok = 0;
    return ok ? 0 : -1;
}
static int integer(const char *text, long long low, long long high, long long *value) {
    char *end; errno = 0; long long n = strtoll(text, &end, 10);
    if (errno || *end || end == text || n < low || n > high) return -1;
    *value = n; return 0;
}
static const char *string(json_val *obj, const char *key) {
    json_val *v = json_obj_get(obj, key);
    return v && v->type == JSON_STRING ? v->str.ptr : NULL;
}
static char *component(json_val *obj, const char *key, const char *root) {
    const char *name = string(obj, key);
    if (!name || !*name || name[0] == '/' || strstr(name, "..")) return NULL;
    size_t size = strlen(root) + strlen(name) + 2;
    char *path = malloc(size);
    if (path) snprintf(path, size, "%s/%s", root, name);
    return path;
}
static void usage(void) {
    puts("HunyuanVideo-1.5 native experimental runner\n"
         "  --generate --model DIR --task i2v|t2v --image FILE --prompt TEXT\n"
         "  --preset quality|fast12 --frames 81|121 --seed N --out-dir DIR\n"
         "  --width 480 --height 848 --device 0 --vram-budget-mib 14336\n"
         "  --offload block --allow-experimental\n"
         "  --validate checks the request without loading checkpoints\n"
         "Outputs RGB PPM frames; native_generate.py packages MP4, poster and provenance.");
}
int main(int argc, char **argv) {
    hv15_request request; hv15_defaults(&request);
    hv15_model model = {0}; model.vram_budget_mib = 14336;
    const char *root = NULL; int validate = 0, generate = 0;
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help")) { usage(); return 0; }
        if (!strcmp(arg, "--validate")) { validate = 1; continue; }
        if (!strcmp(arg, "--generate")) { generate = 1; continue; }
        if (!strcmp(arg, "--allow-experimental")) { model.allow_experimental = 1; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", arg); return 2; }
        const char *value = argv[++i]; long long n;
        if (!strcmp(arg, "--task")) request.task = value;
        else if (!strcmp(arg, "--preset")) request.preset = value;
        else if (!strcmp(arg, "--prompt")) request.prompt = value;
        else if (!strcmp(arg, "--negative-prompt")) request.negative_prompt = value;
        else if (!strcmp(arg, "--image")) request.image = value;
        else if (!strcmp(arg, "--vision-pixels")) request.vision_pixels = value;
        else if (!strcmp(arg, "--model")) root = value;
        else if (!strcmp(arg, "--out-dir")) output = value;
        else if (!strcmp(arg, "--offload")) {
            if (strcmp(value, "block")) { fprintf(stderr, "only block offload is supported\n"); return 2; }
        } else {
            if (integer(value, 0, LLONG_MAX, &n)) { fprintf(stderr, "invalid integer: %s\n", value); return 2; }
            if (!strcmp(arg, "--seed")) request.seed = n;
            else if (n > INT_MAX) return 2;
            else if (!strcmp(arg, "--frames")) request.frames = (int)n;
            else if (!strcmp(arg, "--width")) request.width = (int)n;
            else if (!strcmp(arg, "--height")) request.height = (int)n;
            else if (!strcmp(arg, "--device")) model.device = (int)n;
            else if (!strcmp(arg, "--vram-budget-mib")) model.vram_budget_mib = (int)n;
            else { fprintf(stderr, "unknown argument %s\n", arg); return 2; }
        }
    }
    char error[512];
    if (hv15_validate(&request, error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 2; }
    if (validate) { puts("request valid"); return 0; }
    if (!generate || !root || !output) { usage(); return 2; }
    char manifest_path[PATH_MAX];
    snprintf(manifest_path, sizeof(manifest_path), "%s/model.json", root);
    FILE *f = fopen(manifest_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", manifest_path); return 2; }
    char *buffer = malloc(1048577);
    if (!buffer) { fclose(f); return 1; }
    size_t length = fread(buffer, 1, 1048577, f); fclose(f);
    if (length > 1048576) { free(buffer); return 2; }
    buffer[length] = 0;
    json_val *manifest = json_parse(buffer, (int)length); free(buffer);
    const char *schema = string(manifest, "schema");
    if (!schema || strcmp(schema, "hunyuan_video15.model.v1")) {
        fprintf(stderr, "invalid model manifest schema\n"); json_free(manifest); return 2;
    }
    json_val *components = json_obj_get(manifest, "components");
    char checkpoint[64]; snprintf(checkpoint, sizeof(checkpoint), "%s_%s", request.preset, request.task);
    model.diffusion = component(json_obj_get(manifest, "checkpoints"), checkpoint, root);
    model.vae = component(components, "vae", root); model.qwen = component(components, "qwen", root);
    model.byt5 = component(components, "byt5", root); model.vision = component(components, "vision", root);
    model.tokenizer = component(components, "tokenizer", root);
    json_free(manifest);
    if (mkdir(output, 0700) && errno != EEXIST) { perror(output); return 1; }
    signal(SIGINT, stop); signal(SIGTERM, stop);
    hv15_context *ctx = hv15_load(&model, error, sizeof(error));
    if (!ctx) { fprintf(stderr, "%s\n", error); return 1; }
    hv15_callbacks cb = {progress, frame, cancelled, NULL};
    int result = hv15_generate(ctx, &request, &cb, error, sizeof(error));
    hv15_free(ctx);
    free((void *)model.diffusion); free((void *)model.vae); free((void *)model.qwen);
    free((void *)model.byt5); free((void *)model.vision); free((void *)model.tokenizer);
    if (result) fprintf(stderr, "%s\n", error);
    return result ? (interrupted ? 130 : 1) : 0;
}
