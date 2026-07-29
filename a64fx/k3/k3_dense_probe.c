#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3_dense.h"
#include "k3_runtime.h"

typedef struct {
    uint64_t offset, nbytes;
    int rows, cols;
    char name[512];
} entry;
typedef k3_bf16_matrix matrix;
static void evict(float *b, size_t n, int th);
static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static int manifest(const char *path, entry *out, int cap) {
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char line[1024];
    int n = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#')
            continue;
        unsigned long long o, b;
        char dt[16];
        int nd, r, c;
        if (sscanf(line, "%llu %llu %15s %d %d %d %511s", &o, &b, dt, &nd, &r,
                   &c, out[n].name) != 7 ||
            strcmp(dt, "BF16") || nd != 2 || n >= cap) {
            fclose(f);
            return -1;
        }
        out[n].offset = o;
        out[n].nbytes = b;
        out[n].rows = r;
        out[n].cols = c;
        ++n;
    }
    fclose(f);
    return n;
}
static entry *find(entry *e, int n, const char *s) {
    for (int i = 0; i < n; ++i)
        if (strstr(e[i].name, s))
            return &e[i];
    return NULL;
}
static int q8_projection(const matrix *m,const float*x,const float*ref,int threads,const char*label){
    size_t wn=(size_t)m->rows*m->cols;int8_t*qw=malloc(wn),*qx=malloc((size_t)m->cols);float*sc=malloc((size_t)m->rows*4),*out=malloc((size_t)m->rows*4);
    size_t en=(size_t)192*1024*1024/4;float*eb=calloc(en,4);if(!qw||!qx||!sc||!out||!eb)return 1;
    k3_q8_quantize_bf16_rows(qw,sc,m->weight,m->rows,m->cols);k3_q8_matrix q={qw,sc,m->rows,m->cols};k3_matvec_q8(out,&q,x,qx,threads);
    double se=0,sr=0,dot=0,so=0;for(int i=0;i<m->rows;++i){double d=out[i]-ref[i];se+=d*d;sr+=(double)ref[i]*ref[i];dot+=(double)out[i]*ref[i];so+=(double)out[i]*out[i];}
    double rel=sqrt(se/(sr+1e-30)),cos=dot/sqrt((sr+1e-30)*(so+1e-30));double sec=0;int iters=10;for(int it=0;it<iters;++it){evict(eb,en,threads);double t=now_sec();k3_matvec_q8(out,&q,x,qx,threads);sec+=now_sec()-t;}
    printf("[dense-q8-%s] rel_l2=%.3e cosine=%.8f %s\n",label,rel,cos,rel<5e-3&&cos>=.99995?"GATE-PASS":"GATE-REJECT");
    printf("PROBE dense mode=%s-q8 threads=%d us=%.3f GB/s=%.2f memory_MiB=%.2f\n",label,threads,sec/iters*1e6,k3_q8_matrix_bytes(m->rows,m->cols)/(sec/iters)/1e9,k3_q8_matrix_bytes(m->rows,m->cols)/1048576.0);
    free(qw);free(qx);free(sc);free(out);free(eb);return !isfinite(rel)||!isfinite(cos);
}
static uint64_t rs = 0x4b3344454e534501ULL;
static float rnd(void) {
    rs += 0x9e3779b97f4a7c15ULL;
    uint64_t z = rs;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return ((z ^ (z >> 31)) >> 40) / 8388608.f - 1.f;
}
static void mv(float *y, const matrix *m, const float *x, int threads) {
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < m->rows; r += 8) {
        const uint16_t *w = m->weight + (size_t)r * m->cols;
        matvec_bf16_8row(y + r, w, w + m->cols, w + 2 * m->cols,
                         w + 3 * m->cols, w + 4 * m->cols, w + 5 * m->cols,
                         w + 6 * m->cols, w + 7 * m->cols, x, m->cols);
    }
}
static void evict(float *b, size_t n, int th) {
    omp_set_num_threads(th);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += 16)
        b[i] += 1;
}
static void perf(const matrix *r, const matrix *d, const float *x, float *yr,
                 float *yd, int th, int use_fused) {
    size_t en = (size_t)192 * 1024 * 1024 / 4;
    float *eb = calloc(en, 4);
    double sec = 0;
    int iters = 10;
    for (int i = 0; i < iters; ++i) {
        evict(eb, en, th);
        double t = now_sec();
        if (use_fused)
            k3_dense_pair_bf16(yr, yd, r, d, x, th);
        else {
            mv(yr, r, x, th);
            mv(yd, d, x, th);
        }
        sec += now_sec() - t;
    }
    double bytes =
               2.0 * (r->rows * (double)r->cols + d->rows * (double)d->cols),
           sum = 0;
    for (int i = 0; i < r->rows; ++i)
        sum += yr[i];
    for (int i = 0; i < d->rows; ++i)
        sum += yd[i];
    printf("PROBE dense mode=%s threads=%d us=%.3f GB/s=%.2f checksum=%+.6e\n",
           use_fused ? "fused" : "separate", th, sec / iters * 1e6,
           bytes / (sec / iters) / 1e9, sum);
    free(eb);
}
static int q8_correctness_perf(const matrix *r, const matrix *d, const float *x,
                               const float *down_ref, int threads) {
    size_t wn = (size_t)d->rows * d->cols;
    int8_t *qw = malloc(wn), *qx = malloc((size_t)d->cols);
    float *scale = malloc((size_t)d->rows * sizeof(float));
    float *out = malloc((size_t)d->rows * sizeof(float));
    float *router_out = malloc((size_t)r->rows * sizeof(float));
    if (!qw || !qx || !scale || !out || !router_out) return 1;
    k3_q8_quantize_bf16_rows(qw, scale, d->weight, d->rows, d->cols);
    k3_q8_matrix q = {qw, scale, d->rows, d->cols};
    k3_matvec_q8(out, &q, x, qx, threads);
    double se = 0.0, sr = 0.0, dot = 0.0, so = 0.0;
    float ma = 0.0f;
    for (int i = 0; i < d->rows; ++i) {
        double diff = out[i] - down_ref[i];
        se += diff * diff; sr += (double)down_ref[i] * down_ref[i];
        dot += (double)out[i] * down_ref[i]; so += (double)out[i] * out[i];
        ma = fmaxf(ma, fabsf((float)diff));
    }
    double rel = sqrt(se / (sr + 1e-30));
    double cosine = dot / sqrt((sr + 1e-30) * (so + 1e-30));
    int accepted = rel < 5e-3 && cosine >= .99995;
    printf("[dense-q8] max_abs=%.3e rel_l2=%.3e cosine=%.8f %s\n",
           ma, rel, cosine, accepted ? "GATE-PASS" : "GATE-REJECT(BF16 fallback)");
    size_t en = (size_t)192 * 1024 * 1024 / 4;
    float *eb = calloc(en, sizeof(float));
    double sec = 0.0; int iters = 10;
    double qsec = 0.0;
    for (int i = 0; i < iters; ++i) {
        evict(eb, en, threads); double t = now_sec();
        k3_matvec_q8(out, &q, x, qx, threads); qsec += now_sec() - t;
    }
    printf("PROBE dense mode=down-q8-dot24 threads=%d us=%.3f GB/s=%.2f\n",
           threads, qsec / iters * 1e6,
           k3_q8_matrix_bytes(d->rows,d->cols)/(qsec/iters)/1e9);
    for (int i = 0; i < iters; ++i) {
        evict(eb, en, threads); double t = now_sec();
        k3_dense_router_bf16_down_q8(router_out, out, r, &q, x, qx, threads);
        sec += now_sec() - t;
    }
    double bytes = 2.0 * r->rows * r->cols + (double)wn + d->rows * 4.0;
    printf("PROBE dense mode=router-bf16+down-q8 threads=%d us=%.3f GB/s=%.2f memory_MiB=%.2f\n",
           threads, sec / iters * 1e6, bytes / (sec / iters) / 1e9,
           k3_q8_matrix_bytes(d->rows, d->cols) / 1048576.0);
    free(eb); free(qw); free(qx); free(scale); free(out); free(router_out);
    return !isfinite(rel) || !isfinite(cosine);
}
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s BLOB MANIFEST\n", argv[0]);
        return 2;
    }
    entry e[3];
    int ne=manifest(argv[2], e, 3);
    if (ne < 2)
        return 2;
    k3_apply_numa_interleave();
    size_t sz;
    uint8_t *b = k3_load_blob_anon(argv[1], &sz);
    if (!b)
        return 2;
    entry *re = find(e, ne, "gate.weight"),
          *de = find(e, ne, "routed_expert_down_proj"),
          *ue = find(e, ne, "routed_expert_up_proj");
    if (!re || !de)
        return 2;
    matrix r = {(uint16_t *)(b + re->offset), re->rows, re->cols},
           d = {(uint16_t *)(b + de->offset), de->rows, de->cols};
    float *x = malloc((size_t)7168 * 4), *yr = malloc((size_t)r.rows * 4),
          *yd = malloc((size_t)d.rows * 4);
    for (int i = 0; i < 7168; ++i)
        x[i] = rnd() * .125f;
    mv(yr, &r, x, 48);
    mv(yd, &d, x, 48);
    double ref = 0;
    for (int i = 0; i < 7168; ++i)
        ref += (double)bf16_to_f32_scalar(r.weight[i]) * x[i];
    double err = fabs(ref - yr[0]);
    printf("[dense-row0] abs_err=%.3e %s\n", err, err < 2e-5 ? "OK" : "FAIL");
    int ts[] = {24, 28, 32, 36, 40, 44, 47, 48};
    float *down_ref = malloc((size_t)d.rows * sizeof(float));
    memcpy(down_ref, yd, (size_t)d.rows * sizeof(float));
    for (int i = 0; i < 8; ++i) {
        perf(&r, &d, x, yr, yd, ts[i], 0);
        perf(&r, &d, x, yr, yd, ts[i], 1);
    }
    int q8_fail = q8_correctness_perf(&r, &d, x, down_ref, 47);
    if(ue){matrix up={(uint16_t*)(b+ue->offset),ue->rows,ue->cols};float*ux=malloc((size_t)up.cols*4),*uref=malloc((size_t)up.rows*4);for(int i=0;i<up.cols;++i)ux[i]=rnd()*.125f;mv(uref,&up,ux,47);q8_fail|=q8_projection(&up,ux,uref,47,"up");free(ux);free(uref);}
    free(down_ref);
    free(x);
    free(yr);
    free(yd);
    free(b);
    return err < 2e-5 && !q8_fail ? 0 : 1;
}
