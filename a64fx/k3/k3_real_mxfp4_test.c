#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "k3_kernels.h"
#include "ggml_dequant.h"
#include "k3_moe.h"

typedef struct {
    uint64_t offset, nbytes, rows, cols;
    char dtype[16], name[512];
} tensor_entry;

static int read_full_at(int fd, void *buf, size_t bytes, uint64_t offset) {
    uint8_t *p = (uint8_t *)buf;
    while (bytes) {
        ssize_t n = pread(fd, p, bytes, (off_t)offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += n; offset += (uint64_t)n; bytes -= (size_t)n;
    }
    return 0;
}

static int load_manifest(const char *path, tensor_entry *entries, int capacity) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[1024]; int count = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        tensor_entry e; int ndims = 0;
        unsigned long long off, bytes, rows, cols;
        int n = sscanf(line, "%llu %llu %15s %d %llu %llu %511s",
                       &off, &bytes, e.dtype, &ndims, &rows, &cols, e.name);
        if (n != 7 || ndims != 2) continue;
        if (count >= capacity) { fclose(f); return -1; }
        e.offset = off; e.nbytes = bytes; e.rows = rows; e.cols = cols;
        entries[count++] = e;
    }
    fclose(f);
    return count;
}

static tensor_entry *find_entry(tensor_entry *entries, int count, const char *suffix) {
    for (int i = 0; i < count; ++i) {
        size_t nl = strlen(entries[i].name), sl = strlen(suffix);
        if (nl >= sl && strcmp(entries[i].name + nl - sl, suffix) == 0) return &entries[i];
    }
    return NULL;
}

static uint64_t rng_state = UINT64_C(0x4b335245414c0001);
static uint64_t rng_next(void) {
    uint64_t z = (rng_state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static float random_float(void) { return ((rng_next() >> 40) / 8388608.0f) - 1.0f; }

static float mxfp4_ref(const uint8_t *w, const uint8_t *scale, const float *x, int k) {
    double sum = 0.0;
    for (int b = 0; b < k / 32; ++b) {
        /* compressed-tensors packs consecutive values and uses standard OCP
         * E2M1.  ds4f_kvalues_mxfp4_f32 is the legacy doubled table. */
        float s = 0.5f * ggml_e8m0_to_fp32(scale[b]);
        for (int j = 0; j < 16; ++j) {
            uint8_t p = w[b * 16 + j];
            sum += (double)ds4f_kvalues_mxfp4_f32[p & 15] * s * x[b * 32 + 2*j];
            sum += (double)ds4f_kvalues_mxfp4_f32[p >> 4] * s * x[b * 32 + 2*j + 1];
        }
    }
    return (float)sum;
}

static int test_matrix(int fd, tensor_entry *packed, tensor_entry *scale, int first_row) {
    if (!packed || !scale || strcmp(packed->dtype, "U8") || strcmp(scale->dtype, "U8")) return 1;
    int k = (int)packed->cols * 2;
    if (k % 32 || scale->cols != (uint64_t)k / 32 || packed->rows != scale->rows
            || first_row < 0 || first_row + 8 > (int)packed->rows) return 1;
    size_t wb = (size_t)8 * packed->cols, sb = (size_t)8 * scale->cols;
    uint8_t *w = malloc(wb), *s = malloc(sb); float *x = malloc((size_t)k * sizeof(float));
    if (!w || !s || !x) return 1;
    if (read_full_at(fd, w, wb, packed->offset + (uint64_t)first_row * packed->cols)
            || read_full_at(fd, s, sb, scale->offset + (uint64_t)first_row * scale->cols)) return 1;
    for (int i = 0; i < k; ++i) x[i] = random_float();
    const uint8_t *wr[8], *sr[8]; float got[8], ref[8];
    for (int r = 0; r < 8; ++r) {
        wr[r] = w + (size_t)r * packed->cols;
        sr[r] = s + (size_t)r * scale->cols;
        ref[r] = mxfp4_ref(wr[r], sr[r], x, k);
    }
    matvec_mxfp4_8row(got,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],
                      sr[0],sr[1],sr[2],sr[3],sr[4],sr[5],sr[6],sr[7],x,k);
    double max_abs = 0.0, max_rel = 0.0;
    for (int r = 0; r < 8; ++r) {
        double ae = fabs((double)got[r] - ref[r]);
        double re = ae / (fabs((double)ref[r]) + 1e-6);
        if (ae > max_abs) max_abs = ae;
        if (re > max_rel) max_rel = re;
    }
    const char *which = strstr(packed->name, ".w1.") ? "w1" :
                        strstr(packed->name, ".w2.") ? "w2" : "w3";
    int bad = max_rel > 3e-5 && max_abs > 3e-4;
    printf("[real-%s] rows=%d..%d K=%d max_abs=%.3e max_rel=%.3e %s\n",
           which, first_row, first_row + 7, k, max_abs, max_rel, bad ? "FAIL" : "OK");
    enum { batch = 8 };
    float *xb = malloc((size_t)batch * k * sizeof(float));
    float tiled[batch * 8], tiled_ref[batch * 8];
    if (!xb) return 1;
    for (int m = 0; m < batch; ++m)
        for (int i = 0; i < k; ++i) xb[(size_t)m * k + i] = random_float();
    k3_mxfp4_group_tile(tiled, 8, w, s, xb, k, batch, k, NULL);
    for (int m = 0; m < batch; ++m)
        for (int r = 0; r < 8; ++r)
            tiled_ref[m * 8 + r] = mxfp4_ref(wr[r], sr[r], xb + (size_t)m * k, k);
    double tile_abs = 0.0, tile_rel = 0.0;
    for (int i = 0; i < batch * 8; ++i) {
        double ae = fabs((double)tiled[i] - tiled_ref[i]);
        double re = ae / (fabs((double)tiled_ref[i]) + 1e-6);
        if (ae > tile_abs) tile_abs = ae;
        if (re > tile_rel) tile_rel = re;
    }
    int tile_bad = tile_rel > 3e-3 && tile_abs > 3e-3;
    printf("[real-%s-tile] batch=%d max_abs=%.3e max_rel=%.3e %s\n",
           which, batch, tile_abs, tile_rel, tile_bad ? "FAIL" : "OK");
    bad |= tile_bad;
    free(xb);
    free(w); free(s); free(x);
    return bad;
}

static int matvec_all(float *out, const uint8_t *w, const uint8_t *s,
                      const float *x, int rows, int k) {
    if (rows % 8 || k % 32) return -1;
    size_t wstride = (size_t)k / 2, sstride = (size_t)k / 32;
    for (int r = 0; r < rows; r += 8) {
        const uint8_t *wr[8], *sr[8];
        for (int j = 0; j < 8; ++j) {
            wr[j] = w + (size_t)(r + j) * wstride;
            sr[j] = s + (size_t)(r + j) * sstride;
        }
        matvec_mxfp4_8row(out+r,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],
                          sr[0],sr[1],sr[2],sr[3],sr[4],sr[5],sr[6],sr[7],x,k);
    }
    return 0;
}

static int compare_projection(const char *name, const float *got, const float *ref, int n) {
    double max_abs = 0.0, max_rel = 0.0;
    for (int i = 0; i < n; ++i) {
        double ae = fabs((double)got[i] - ref[i]);
        double re = ae / (fabs((double)ref[i]) + 1e-6);
        if (ae > max_abs) max_abs = ae;
        if (re > max_rel) max_rel = re;
    }
    int bad = max_rel > 5e-4 && max_abs > 5e-4;
    printf("[expert-%s] n=%d max_abs=%.3e max_rel=%.3e %s\n",
           name,n,max_abs,max_rel,bad?"FAIL":"OK");
    return bad;
}

static int test_full_expert(int fd, tensor_entry *entries, int count) {
    tensor_entry *w1=find_entry(entries,count,"w1.weight_packed"), *s1=find_entry(entries,count,"w1.weight_scale");
    tensor_entry *w2=find_entry(entries,count,"w2.weight_packed"), *s2=find_entry(entries,count,"w2.weight_scale");
    tensor_entry *w3=find_entry(entries,count,"w3.weight_packed"), *s3=find_entry(entries,count,"w3.weight_scale");
    if (!w1||!s1||!w2||!s2||!w3||!s3) return 1;
    struct stat st;
    if (fstat(fd,&st) || st.st_size <= 0) return 1;
    uint8_t *blob = mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if (blob == MAP_FAILED) return 1;
    int latent=(int)w1->cols*2, inter=(int)w1->rows, outdim=(int)w2->rows;
    if (latent != K3_LATENT || inter != K3_EXPERT_INTER || outdim != K3_LATENT
            || (int)w2->cols*2 != inter) { munmap(blob,(size_t)st.st_size); return 1; }
    float *x=malloc((size_t)latent*4), *gate=malloc((size_t)inter*4), *up=malloc((size_t)inter*4);
    float *gate_ref=malloc((size_t)inter*4), *up_ref=malloc((size_t)inter*4), *hidden=malloc((size_t)inter*4);
    float *out=malloc((size_t)outdim*4), *out_ref=malloc((size_t)outdim*4);
    if(!x||!gate||!up||!gate_ref||!up_ref||!hidden||!out||!out_ref) return 1;
    for(int i=0;i<latent;++i) x[i]=random_float()*.125f;
    matvec_all(gate,blob+w1->offset,blob+s1->offset,x,inter,latent);
    matvec_all(up,blob+w3->offset,blob+s3->offset,x,inter,latent);
    for(int r=0;r<inter;++r){
        gate_ref[r]=mxfp4_ref(blob+w1->offset+(size_t)r*w1->cols,
                             blob+s1->offset+(size_t)r*s1->cols,x,latent);
        up_ref[r]=mxfp4_ref(blob+w3->offset+(size_t)r*w3->cols,
                           blob+s3->offset+(size_t)r*s3->cols,x,latent);
    }
    int fail=compare_projection("w1",gate,gate_ref,inter)|compare_projection("w3",up,up_ref,inter);
    k3_situ_sve(hidden,gate,up,inter);
    matvec_all(out,blob+w2->offset,blob+s2->offset,hidden,outdim,inter);
    for(int r=0;r<outdim;++r)
        out_ref[r]=mxfp4_ref(blob+w2->offset+(size_t)r*w2->cols,
                            blob+s2->offset+(size_t)r*s2->cols,hidden,inter);
    fail|=compare_projection("w2",out,out_ref,outdim);
    double sum=0.0,ss=0.0; int finite=1;
    for(int i=0;i<outdim;++i){ sum+=out[i]; ss+=(double)out[i]*out[i]; finite&=isfinite(out[i]); }
    printf("[expert-ffn] checksum=%+.9e l2=%.9e finite=%s\n",sum,sqrt(ss),finite?"yes":"NO");
    fail|=!finite;
    munmap(blob,(size_t)st.st_size);
    free(x);free(gate);free(up);free(gate_ref);free(up_ref);free(hidden);free(out);free(out_ref);
    return fail;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s BLOB MANIFEST [FIRST_ROW]\n", argv[0]); return 2;
    }
    int first_row = argc == 4 ? atoi(argv[3]) : 0;
    tensor_entry *entries = calloc(10000, sizeof(*entries));
    if (!entries) return 2;
    int count = load_manifest(argv[2], entries, 10000);
    if (count < 6) { fprintf(stderr, "expected at least 6 manifest entries, got %d\n", count); return 2; }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open blob"); return 2; }
    int fail = 0;
    fail |= test_matrix(fd, find_entry(entries,count,"w1.weight_packed"),
                        find_entry(entries,count,"w1.weight_scale"), first_row);
    fail |= test_matrix(fd, find_entry(entries,count,"w2.weight_packed"),
                        find_entry(entries,count,"w2.weight_scale"), first_row);
    fail |= test_matrix(fd, find_entry(entries,count,"w3.weight_packed"),
                        find_entry(entries,count,"w3.weight_scale"), first_row);
    fail |= test_full_expert(fd, entries, count);
    close(fd);
    free(entries);
    printf("K3 real MXFP4 partial test: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
