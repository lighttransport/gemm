/* Unit test for triton_spconv_pp.h: validate gray/sorted_idx/vk/vk_seg
 * against PyTorch-generated reference dumps.
 *
 * Build:
 *   gcc -O2 -o test_pp test_pp.c
 * Run:
 *   ./test_pp
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include "triton_spconv_pp.h"

/* Minimal .npy loader (header parser + raw bytes). Assumes little-endian,
 * non-fortran-order, 1- or 2-D, dtype = the one declared. */
static void *
read_npy(const char *path, int *ndim, int *shape, size_t *elt_sz, char *dtype_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: ", path); perror(""); return NULL; }
    char magic[6]; fread(magic, 1, 6, f);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
    unsigned char vmaj, vmin; fread(&vmaj, 1, 1, f); fread(&vmin, 1, 1, f);
    uint32_t hdr_len = 0;
    if (vmaj == 1) { uint16_t hl; fread(&hl, 2, 1, f); hdr_len = hl; }
    else { fread(&hdr_len, 4, 1, f); }
    char *hdr = (char *)malloc(hdr_len + 1); fread(hdr, 1, hdr_len, f); hdr[hdr_len] = 0;
    /* parse descr e.g. '<i4', '<u4', '<i8' */
    char *p = strstr(hdr, "'descr':");
    if (!p) { fclose(f); free(hdr); return NULL; }
    p += strlen("'descr':");
    while (*p && *p != '\'') p++;            /* skip to opening quote of value */
    if (*p != '\'') { fclose(f); free(hdr); return NULL; }
    p++;                                      /* now at e.g. '<u4' content */
    /* may have leading byteorder char */
    if (*p == '<' || *p == '>' || *p == '|' || *p == '=') p++;
    *dtype_out = *p;                          /* 'i', 'u', 'f' */
    p++;
    *elt_sz = (size_t)(*p - '0');             /* '4' or '8' */
    /* parse shape */
    p = strstr(hdr, "'shape':"); p = strchr(p, '(') + 1;
    *ndim = 0;
    while (*p && *p != ')') {
        while (*p == ' ' || *p == ',') p++;
        if (!isdigit((unsigned char)*p)) break;
        shape[(*ndim)++] = atoi(p);
        while (isdigit((unsigned char)*p)) p++;
    }
    free(hdr);
    size_t total = *elt_sz;
    for (int i = 0; i < *ndim; i++) total *= (size_t)shape[i];
    void *buf = malloc(total);
    fread(buf, 1, total, f);
    fclose(f);
    return buf;
}

int main(void)
{
    const char *DIR = "/mnt/disk1/work/gemm/trellis2/rdna4/trellis2/triton_aot/test_data";
    int nd, sh[4]; size_t es; char dt;
    char path[512];

    snprintf(path, sizeof path, "%s/nmap_stage0.npy", DIR);
    int32_t *nmap = (int32_t *)read_npy(path, &nd, sh, &es, &dt);
    if (!nmap) return 1;
    int N = sh[0], V = sh[1];
    printf("nmap (%d, %d) dtype=%c%zu\n", N, V, dt, es);

    snprintf(path, sizeof path, "%s/gray_stage0.npy", DIR);
    int32_t *gray_ref = (int32_t *)read_npy(path, &nd, sh, &es, &dt);

    snprintf(path, sizeof path, "%s/sorted_idx_stage0.npy", DIR);
    int64_t *sorted_ref = (int64_t *)read_npy(path, &nd, sh, &es, &dt);

    snprintf(path, sizeof path, "%s/vk_stage0_B64.npy", DIR);
    int32_t *vk_ref = (int32_t *)read_npy(path, &nd, sh, &es, &dt);
    int vk_ref_len = sh[0];

    snprintf(path, sizeof path, "%s/vk_seg_stage0_B64.npy", DIR);
    int32_t *seg_ref = (int32_t *)read_npy(path, &nd, sh, &es, &dt);
    int seg_ref_len = sh[0];

    /* Run our impl. */
    uint32_t *gray = (uint32_t *)malloc(N * 4);
    uint32_t *binary = (uint32_t *)malloc(N * 4);
    int64_t *sorted = (int64_t *)malloc(N * 8);
    t2_neigh_to_gray_binary(N, V, nmap, gray, binary);
    t2_argsort_binary(N, binary, sorted);

    /* Compare gray. */
    int gd = 0;
    for (int i = 0; i < N; i++) if ((int32_t)gray[i] != gray_ref[i]) { gd++; if (gd<3) printf("  gray[%d]: ours=0x%x ref=0x%x\n", i, gray[i], (uint32_t)gray_ref[i]); }
    printf("gray: %d mismatches / %d\n", gd, N);

    /* Compare sorted_idx — there can be ties; check binary[sorted[i]] sequence equals binary[sorted_ref[i]] sequence. */
    int sd_strict = 0, sd_seq = 0;
    for (int i = 0; i < N; i++) {
        if (sorted[i] != sorted_ref[i]) sd_strict++;
        if (binary[sorted[i]] != binary[sorted_ref[i]]) sd_seq++;
    }
    printf("sorted: %d strict / %d sequence-equiv mismatches / %d\n", sd_strict, sd_seq, N);

    /* Build vk + seg. */
    int32_t *vk = NULL; int32_t *seg = NULL; int vk_len = 0;
    t2_build_valid_kernel(N, 64, gray, sorted, &vk, &seg, &vk_len);
    printf("vk_len: ours=%d ref=%d  seg_len: ours=%d ref=%d\n",
           vk_len, vk_ref_len, (N+63)/64+1, seg_ref_len);
    int seg_diff = 0;
    for (int i = 0; i <= (N+63)/64; i++) if (seg[i] != seg_ref[i]) seg_diff++;
    printf("vk_seg: %d mismatches / %d\n", seg_diff, (N+63)/64+1);

    int vk_diff = 0;
    /* vk values within a block depend only on the OR of gray[]; ours is sorted by ctz which matches CUDA __ffs. */
    if (vk_len == vk_ref_len) {
        for (int i = 0; i < vk_len; i++) if (vk[i] != vk_ref[i]) vk_diff++;
    }
    printf("vk: %d mismatches / %d\n", vk_diff, vk_len);

    return (gd || sd_seq || seg_diff || vk_diff) ? 1 : 0;
}
