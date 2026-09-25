#include "glm53f_q8_resident.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static uint64_t hash_bytes(const unsigned char *p, size_t n) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}

static int selftest(void) {
    const char *dir = "/local/glm53f-q8-resident-selftest";
    const char *name = "blk.0.ffn_gate.weight";
    unsigned char payload[256];
    for (size_t i = 0; i < sizeof(payload); ++i) payload[i] = (unsigned char)i;
    char blob[256], manifest[256];
    snprintf(blob, sizeof(blob), "%s/rank00.blob", dir);
    snprintf(manifest, sizeof(manifest), "%s/rank00.manifest", dir);
    if (mkdir(dir, 0755) && errno != EEXIST) return 1;
    int fd = open(blob, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0 || write(fd, payload, sizeof(payload)) != (ssize_t)sizeof(payload) || close(fd)) return 1;
    FILE *f = fopen(manifest, "w");
    if (!f) return 1;
    fprintf(f, "# GLM53F_Q8_IMAGE_V1 rank=0 ranks=12 tensors=1\n");
    fprintf(f, "T 7 %s 8 2 32 1 1 1 0 256 0 1 0\n", name);
    fprintf(f, "# COMPLETE bytes=256 tensors=1 fnv1a=%016" PRIx64 "\n", hash_bytes(payload, sizeof(payload)));
    fclose(f);
    glm53f_q8_resident *r = glm53f_q8_resident_load(dir, 0);
    const glm53f_q8_resident_entry *e = glm53f_q8_resident_find(r, name);
    int ok = r && e && e->tensor_index == 7 && e->bytes == sizeof(payload) &&
             !memcmp(glm53f_q8_resident_data(r), payload, sizeof(payload));
    glm53f_q8_resident_free(r);
    unlink(blob); unlink(manifest); rmdir(dir);
    printf("GLM53F_Q8_RESIDENT_SELFTEST %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--selftest")) return selftest();
    if (argc != 3) { fprintf(stderr, "usage: %s IMAGE_DIR RANK\n", argv[0]); return 2; }
    glm53f_q8_resident *r = glm53f_q8_resident_load(argv[1], atoi(argv[2]));
    if (!r) { perror("glm53f_q8_resident_load"); return 1; }
    printf("GLM53F_Q8_RESIDENT size=%zu entries=%zu hash=%016" PRIx64 " data=%p PASS\n",
           glm53f_q8_resident_size(r), glm53f_q8_resident_entry_count(r),
           glm53f_q8_resident_hash(r), glm53f_q8_resident_data(r));
    glm53f_q8_resident_free(r);
    return 0;
}
