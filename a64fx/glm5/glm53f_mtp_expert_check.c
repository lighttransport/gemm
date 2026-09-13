/* Compare one real staged MTP expert shard against the scalar FP8 reference. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <arm_sve.h>
#include <omp.h>
#include "glm53f_expert_kern.h"
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int first_offsets(const char *path, uint64_t off[4], int *inter) {
    FILE *f = fopen(path, "r");
    char line[2048], dtype[16], name[1024];
    int n = 0, nd, rows, cols;
    unsigned long long at;
    if (!f) return -1;
    while (n < 4 && fgets(line, sizeof(line), f)) {
        char *last;
        if (line[0] == '#' || sscanf(line, "%llu %15s %d %d %d",
                                     &at, dtype, &nd, &rows, &cols) != 5)
            continue;
        last = strrchr(line, ' ');
        if (!last) continue;
        snprintf(name, sizeof(name), "%s", last + 1);
        name[strcspn(name, "\r\n")] = 0;
        if (!strstr(name, ".experts.0.")) continue;
        off[n++] = at;
        if (strstr(name, "down_proj.weight") && !strstr(name, "scale"))
            *inter = cols;
    }
    fclose(f);
    return n == 4 && *inter > 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    uint64_t off[4];
    int inter = 0, fd;
    struct stat st;
    unsigned char *blob;
    float *x, *up, *act, *got, *ref, *rup, *ract;
    glm53f_expert_part part;
    if (argc != 3 || first_offsets(argv[1], off, &inter)) {
        fprintf(stderr, "usage: %s MANIFEST BLOB\n", argv[0]);
        return 2;
    }
    fd = open(argv[2], O_RDONLY);
    if (fd < 0 || fstat(fd, &st)) return 2;
    blob = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (blob == MAP_FAILED) return 2;
    x = aligned_alloc(256, 4096 * sizeof(float));
    up = aligned_alloc(256, (size_t)2 * inter * sizeof(float));
    act = aligned_alloc(256, (size_t)inter * sizeof(float));
    got = aligned_alloc(256, 4096 * sizeof(float));
    ref = aligned_alloc(256, 4096 * sizeof(float));
    rup = aligned_alloc(256, (size_t)2 * inter * sizeof(float));
    ract = aligned_alloc(256, (size_t)inter * sizeof(float));
    if (!x || !up || !act || !got || !ref || !rup || !ract) return 2;
    for (int i = 0; i < 4096; ++i) x[i] = (float)((i % 31) - 15) * .001f;
    part = (glm53f_expert_part){blob + off[0], (const float *)(blob + off[1]),
                                blob + off[2], (const float *)(blob + off[3]), inter};
    glm53f_expert_batch_bits(&part, 1, x, up, act, got);
    for (int r = 0; r < 2 * inter; ++r)
        rup[r] = glm53f_dot_fp8_block128(part.gate_up + (size_t)r * 4096,
            part.gate_up_scale + (size_t)(r / 128) * 32, x, 4096);
    for (int i = 0; i < inter; ++i) {
        float g = fmaxf(-100, fminf(10, rup[i]));
        float u = fmaxf(-10, fminf(10, rup[inter + i]));
        ract[i] = (g / (1 + expf(-g))) * u;
    }
    for (int r = 0; r < 4096; ++r)
        ref[r] = glm53f_dot_fp8_block128(part.down + (size_t)r * inter,
            part.down_scale + (size_t)(r / 128) * (inter / 128), ract, inter);
    double se = 0, sr = 0, max_abs = 0;
    int finite = 1;
    for (int i = 0; i < 4096; ++i) {
        double d = (double)got[i] - ref[i];
        se += d * d; sr += (double)ref[i] * ref[i];
        if (fabs(d) > max_abs) max_abs = fabs(d);
        finite &= isfinite(got[i]);
    }
    printf("GLM53F_MTP_EXPERT_CHECK inter=%d finite=%s max_abs=%.9g rel_l2=%.9g checksum=%.9g %s\n",
           inter, finite ? "YES" : "NO", max_abs, sqrt(se / (sr + 1e-30)), got[0],
           finite && sqrt(se / (sr + 1e-30)) < 2e-5 ? "PASS" : "FAIL");
    return finite && sqrt(se / (sr + 1e-30)) < 2e-5 ? 0 : 1;
}
