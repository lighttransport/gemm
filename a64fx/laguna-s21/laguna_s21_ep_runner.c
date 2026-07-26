/* Laguna S 2.1 INT4 expert-parallel runner, A64FX / Fugaku.
 *
 * Distributed forward pass + greedy generation over pure uTofu (one rank/node).
 * Experts are EP-sharded (rank owns expert e where e % N == rank); attention,
 * dense layer-0 MLP, shared expert, router, embed, lm_head and norms are
 * replicated on every rank. The only per-MoE-layer communication is one
 * tp_allreduce_sum over the routed partial [hidden].
 *
 * Architecture (verified from config.json / safetensors index):
 *   48 layers, hidden 3072, vocab 100352, GQA 8x128, head_dim 128.
 *   full_attention layers (layer%4==0) = 48 q-heads + YaRN rope (rot 64);
 *   sliding_attention layers = 72 q-heads + default rope (rot 128), window 512.
 *   QK-norm per head before rope; softplus per-head attention gate before o_proj.
 *   MoE: sigmoid+bias top-10 of 256, shared expert, routed_scale 2.5, SwiGLU.
 *   Only routed experts are INT4 group-32; everything else bf16.
 *
 * Build:  make -C a64fx/laguna-s21 all CC=fcc
 * Stage:  mpiexec -np 12 build/laguna_s21_stage      (LAGUNA_EP_SIZE=12)
 * Run:    mpiexec -np 12 [-vcoordfile vc] build/laguna_s21_ep_runner \
 *             --generate --ids prompt.ids --max-new 64 --stage-dir /local/... \
 *             (after tofu_topo_helper produced tofu_topo.txt)
 */
#define _GNU_SOURCE
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "laguna_s21.h"

#if defined(LAGUNA_FP8)
float laguna_fp8_lut[256];   /* definition for the extern in laguna_s21.h */
#endif

/* ============================ small helpers ============================ */
static FILE *g_log = NULL;
static int   g_rank = 0;
static void logmsg(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    if (g_log) { va_list a2; va_copy(a2, ap); vfprintf(g_log, fmt, a2); va_end(a2); fflush(g_log); }
    vfprintf(stderr, fmt, ap); va_end(ap);
}
static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* ============================ ABI self-tests ============================ */
static int test_i4(void) {
    uint32_t w[4] = {0xFEDCBA98u,0x76543210u,0x01234567u,0x89ABCDEFu};
    uint16_t s[1] = {laguna_f32_to_bf16(0.25f)}; float x[32]; for(int i=0;i<32;i++)x[i]=(float)(i-15);
    float a=laguna_i4g32_dot(w,s,x,32), b=0; for(int i=0;i<32;i++)b+=(float)laguna_i4_at(w,i)*x[i]*0.25f;
    if(fabsf(a-b)>1e-5f){fprintf(stderr,"INT4 dot mismatch %.8g %.8g\n",a,b);return 1;} return 0;
}
static int test_fht(void) { float a[128],b[128]; for(int i=0;i<128;i++)a[i]=b[i]=(float)(i-63)*0.03125f; laguna_fht128(a);laguna_fht128(a);for(int i=0;i<128;i++)if(fabsf(a[i]-b[i])>2e-5f){fprintf(stderr,"FHT mismatch at %d\n",i);return 1;}return 0; }
static int test_route(void) { float l[256]={0},b[256]={0},w[10];int id[10];for(int i=0;i<256;i++)l[i]=(float)(i-128)*.01f;laguna_top10(l,b,id,w);float z=0;for(int i=0;i<10;i++){if(id[i]!=255-i){fprintf(stderr,"route id %d=%d\n",i,id[i]);return 1;}z+=w[i];}return fabsf(z-1)>1e-6f; }
/* cross-check the group-32 matvec against a scalar per-element reconstruction */
static int test_i4_matvec(void) {
    enum { R=5, C=64 };
    uint32_t packed[R*(C/8)]; uint16_t scales[R*(C/32)]; float x[C], y[R];
    uint64_t st=0x1234;
    for (int i=0;i<R*(C/8);i++){ st=st*6364136223846793005ull+1; packed[i]=(uint32_t)(st>>32); }
    for (int i=0;i<R*(C/32);i++) scales[i]=laguna_f32_to_bf16(0.1f+0.01f*i);
    for (int i=0;i<C;i++) x[i]=(float)((int)(i%7)-3)*0.5f;
    laguna_matvec_i4g32(y, packed, scales, x, R, C);
    for (int r=0;r<R;r++){
        float ref=0; const uint32_t*pw=packed+r*(C/8); const uint16_t*sc=scales+r*(C/32);
        for (int c=0;c<C;c++) ref += (float)laguna_i4_at(pw,c)*laguna_bf16_to_f32(sc[c/32])*x[c];
        if (fabsf(ref-y[r])>1e-4f){ fprintf(stderr,"i4 matvec row %d: %.6g vs %.6g\n",r,y[r],ref); return 1; }
    }
    return 0;
}

/* Batched int8 GEMM must agree with C separate int8 matvecs (chunked prefill
 * uses the former, the last-token forward the latter). */
static int test_i8_matmat(void) {
    enum { R=136, C=256, N=11 };
    static int8_t q[R*C]; static float s[R], X[N*C], Y[N*R], y1[R];
    uint64_t st=0x9e3779b9ull;
    for (int i=0;i<R*C;i++){ st=st*6364136223846793005ull+1; q[i]=(int8_t)((st>>33)%255-127); }
    for (int i=0;i<R;i++) s[i]=0.001f*(1+i%7);
    for (int i=0;i<N*C;i++){ st=st*6364136223846793005ull+1; X[i]=(float)((int)((st>>34)%2000)-1000)/1000.0f; }
    laguna_w8 w={q,s};
    laguna_matmat_i8(Y,&w,X,R,C,N);
    for (int n=0;n<N;n++){
        laguna_matvec_i8(y1,&w,X+(size_t)n*C,R,C);
        for (int r=0;r<R;r++){
            float a=Y[(size_t)n*R+r], b=y1[r];
            if (fabsf(a-b) > 1e-4f*(1+fabsf(b))) {
                fprintf(stderr,"i8 matmat tok %d row %d: %.6g vs %.6g\n",n,r,a,b); return 1; }
        }
    }
    return 0;
}
#if defined(LAGUNA_FP8)
/* The fp8 -> int8-per-block re-quantization must track the exact e4m3 kernel, and
 * the batched form must agree with the matvec form. */
static int test_fp8_i8blk(void) {
    enum { R=256, C=256, N=9 };
    static uint8_t W[R*C]; static uint16_t bs[(R/LAGUNA_FP8_BLK)*(C/LAGUNA_FP8_BLK)];
    static float x[C], yex[R], yq[R], X[N*C], Y[N*R];
    laguna_fp8_init_lut();
    uint64_t st=0xdeadbeefull;
    /* near-Gaussian block content (what a real block-scaled checkpoint holds) */
    for (int i=0;i<R*C;i++){
        double u=0; for(int k=0;k<4;k++){ st=st*6364136223846793005ull+1; u+=(double)((st>>34)%1000)/1000.0; }
        float v=(float)(u-2.0)*0.35f;                       /* ~N(0,1) scaled */
        int best=0; float bd=1e30f;
        for (int b=0;b<256;b++){ if((b&0x7f)==0x7f) continue;
            float d=fabsf(laguna_fp8_lut[b]-v); if(d<bd){bd=d;best=b;} }
        W[i]=(uint8_t)best;
    }
    for (size_t i=0;i<sizeof bs/sizeof *bs;i++) bs[i]=laguna_f32_to_bf16(0.0037f);
    for (int i=0;i<C;i++){ st=st*6364136223846793005ull+1; x[i]=(float)((int)((st>>34)%2000)-1000)/1000.0f; }
    for (int i=0;i<N*C;i++){ st=st*6364136223846793005ull+1; X[i]=(float)((int)((st>>34)%2000)-1000)/1000.0f; }

    laguna_matvec_fp8blk(yex,W,bs,x,R,C);
    laguna_w8b q;
    if (posix_memalign((void**)&q.q,256,(size_t)R*C)!=0 ||
        posix_memalign((void**)&q.s,256,(size_t)R*(C/LAGUNA_FP8_BLK)*sizeof(float))!=0) return 1;
    laguna_fp8_to_i8blk(&q,W,bs,R,C);
    laguna_matvec_i8blk(yq,&q,x,R,C);

    double num=0,den=0;
    for (int r=0;r<R;r++){ double d=(double)yq[r]-yex[r]; num+=d*d; den+=(double)yex[r]*yex[r]; }
    double rel=sqrt(num/(den>0?den:1));
    if (!(rel < 0.05)) { fprintf(stderr,"fp8->i8blk relerr %.3e too large\n",rel); free(q.q);free(q.s); return 1; }

    laguna_matmat_i8blk(Y,&q,X,R,C,N);
    for (int n=0;n<N;n++){
        laguna_matvec_i8blk(yq,&q,X+(size_t)n*C,R,C);
        for (int r=0;r<R;r++){
            float a=Y[(size_t)n*R+r], b=yq[r];
            if (fabsf(a-b) > 1e-4f*(1+fabsf(b))) {
                fprintf(stderr,"i8blk matmat tok %d row %d: %.6g vs %.6g\n",n,r,a,b);
                free(q.q);free(q.s); return 1; }
        }
    }
    fprintf(stderr,"  fp8->i8blk relerr vs exact e4m3 kernel = %.3e\n",rel);
    free(q.q); free(q.s);
    return 0;
}
#endif

/* ============================ stage validation (unchanged) ============================ */
typedef struct { uint64_t off; size_t bytes; char dtype[8]; int nd; long shape[5]; char name[320]; } ent;
static int stage_check(const char *dir, int layers) {
    char path[1200], line[1024]; snprintf(path,sizeof path,"%s/rank00.manifest",dir);
    FILE *f=fopen(path,"r"); if(!f){perror(path);return 2;}
    long cap=4096,n=0,packed=0,scale=0; ent *v=calloc((size_t)cap,sizeof(*v));
    while(fgets(line,sizeof line,f)) { if(line[0]=='#')continue; if(n==cap){cap*=2;v=realloc(v,(size_t)cap*sizeof(*v));}
        ent *e=&v[n]; int pos=0,got=0; got=sscanf(line,"%" SCNu64 " %zu %7s %d%n",&e->off,&e->bytes,e->dtype,&e->nd,&pos);
        if (got != 4 || e->nd < 0 || e->nd > 5) continue;
        for (int d = 0; d < e->nd; d++) {
            int used = 0;
            if (sscanf(line + pos, " %ld%n", &e->shape[d], &used) != 1) { got = 0; break; }
            pos += used;
        }
        if (!got) continue;
        while (line[pos] == ' ') pos++;
        snprintf(e->name, sizeof e->name, "%s", line + pos);
        e->name[strcspn(e->name, "\n")] = 0;
        if (strstr(e->name, ".weight_packed")) packed++;
        if (strstr(e->name, ".weight_scale")) scale++;
        n++;
    }
    fclose(f); struct stat sb; char blob[1200];snprintf(blob,sizeof blob,"%s/rank00.blob",dir);if(stat(blob,&sb)){perror(blob);free(v);return 2;}
    int bad=0; for(long i=0;i<n;i++){ if(v[i].off+v[i].bytes>(uint64_t)sb.st_size){fprintf(stderr,"out of range: %s\n",v[i].name);bad=1;} }
    if(layers>1 && (!packed || packed!=scale)){fprintf(stderr,"packed/scale mismatch %ld/%ld\n",packed,scale);bad=1;}
    printf("stage rank00: %ld tensors, %ld packed INT4 tensors, %.3f GB, %s\n",n,packed,(double)sb.st_size/1e9,bad?"FAIL":"PASS"); free(v);return bad;
}
static int probe_stage(const char *dir) {
    static const char *need[] = {
        "model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.weight", "model.layers.1.mlp.experts.0.gate_proj.weight_packed",
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale", NULL};
    char mp[1200],bp[1200],line[1024]; snprintf(mp,sizeof mp,"%s/rank00.manifest",dir); snprintf(bp,sizeof bp,"%s/rank00.blob",dir);
    FILE *f=fopen(mp,"r"); if(!f){perror(mp);return 2;} int fd=open(bp,O_RDONLY);struct stat sb;if(fd<0||fstat(fd,&sb)){perror(bp);if(fd>=0)close(fd);fclose(f);return 2;}
    const unsigned char *base=mmap(NULL,(size_t)sb.st_size,PROT_READ,MAP_PRIVATE,fd,0);if(base==MAP_FAILED){perror("mmap");close(fd);fclose(f);return 2;}
    int found=0; while(fgets(line,sizeof line,f)){uint64_t off;size_t bytes;char dt[8],name[320];int nd,pos=0;if(line[0]=='#'||sscanf(line,"%" SCNu64 " %zu %7s %d%n",&off,&bytes,dt,&nd,&pos)!=4)continue;for(int d=0;d<nd;d++){long q;int used;if(sscanf(line+pos," %ld%n",&q,&used)!=1)break;pos+=used;}while(line[pos]==' ')pos++;snprintf(name,sizeof name,"%s",line+pos);name[strcspn(name,"\n")]=0;for(int i=0;need[i];i++)if(!strcmp(name,need[i])){if(off+bytes>(uint64_t)sb.st_size){fprintf(stderr,"bad tensor %s\n",name);found=-99;}else{printf("probe %s: %s %zu B first=0x%02x\n",name,dt,bytes,base[off]);found++;}}}
    munmap((void*)base,(size_t)sb.st_size);close(fd);fclose(f);if(found!=(int)(sizeof need/sizeof need[0]-1)){fprintf(stderr,"probe missing tensors (%d/8)\n",found);return 1;}return 0;
}

/* ============================ manifest -> model ============================ */
typedef struct { char name[320]; uint64_t off; size_t bytes; char dtype[8]; long shape[5]; int nd; } manifest_ent;

typedef struct {
    const unsigned char *blob; size_t blob_bytes;
    manifest_ent *ents; long n_ents;
} laguna_stage;

static const void *stage_find(const laguna_stage *s, const char *name) {
    for (long i = 0; i < s->n_ents; ++i)
        if (!strcmp(s->ents[i].name, name)) return s->blob + s->ents[i].off;
    return NULL;
}
static const void *stage_req(const laguna_stage *s, const char *name) {
    const void *p = stage_find(s, name);
    if (!p) { fprintf(stderr, "FATAL: missing tensor %s\n", name); exit(1); }
    return p;
}
static int stage_dtype_is_bf16(const laguna_stage *s, const char *name) {
    for (long i = 0; i < s->n_ents; ++i)
        if (!strcmp(s->ents[i].name, name)) return !strcmp(s->ents[i].dtype, "BF16");
    return 0;
}

/* use_arena=1: copy the blob into anonymous memory with NUMA-local first-touch
 * (required when weights are used IN PLACE, i.e. the bf16/int4 builds).
 * use_arena=0: point straight at the mmap and skip the copy.  Valid only when
 * every bandwidth-hot tensor is re-quantized at load into its own allocation
 * (the fp8 build: experts -> int8-per-block, linears -> int8), which does its own
 * NUMA-local first-touch.  Saves a full blob-sized allocation (17.8 GB for fp8),
 * which is what makes room for the int8 copies on a 32 GB node. */
static int stage_load(laguna_stage *s, const char *dir, int rank, int use_arena) {
    char mp[1200], bp[1200], line[1024];
    snprintf(mp,sizeof mp,"%s/rank%02d.manifest",dir,rank);
    snprintf(bp,sizeof bp,"%s/rank%02d.blob",dir,rank);
    FILE *f=fopen(mp,"r"); if(!f){perror(mp);return -1;}
    int fd=open(bp,O_RDONLY); struct stat sb;
    if(fd<0||fstat(fd,&sb)){perror(bp); if(fd>=0)close(fd); fclose(f); return -1;}
    const unsigned char *fmap=mmap(NULL,(size_t)sb.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(fmap==MAP_FAILED){perror("mmap"); close(fd); fclose(f); return -1;}
    size_t nb=(size_t)sb.st_size;
    unsigned char *arena=NULL;
    if (use_arena) {
        if(posix_memalign((void**)&arena,256,nb)!=0){fprintf(stderr,"arena alloc %.1f GB failed\n",nb/1e9);munmap((void*)fmap,nb);close(fd);fclose(f);return -1;}
        s->blob=arena;
    } else {
        s->blob=fmap;   /* mapping is retained for the process lifetime */
    }
    s->blob_bytes=nb;
    /* Parse the manifest first so the arena copy can first-touch each tensor with
     * the SAME per-row static schedule the matvecs use.  Copying the whole blob in
     * arbitrary chunks lands each tensor's pages on a CMG unrelated to the thread
     * that later reads that row -> cross-CMG reads collapse bandwidth (~3x slower
     * decode).  Per-tensor parallel copy keeps weights CMG-local to their reader. */
    long cap=8192; s->n_ents=0; s->ents=malloc((size_t)cap*sizeof(manifest_ent));
    while (fgets(line,sizeof line,f)) {
        if (line[0]=='#') continue;
        if (s->n_ents==cap){ cap*=2; s->ents=realloc(s->ents,(size_t)cap*sizeof(manifest_ent)); }
        manifest_ent *e=&s->ents[s->n_ents]; int pos=0;
        if (sscanf(line,"%" SCNu64 " %zu %7s %d%n",&e->off,&e->bytes,e->dtype,&e->nd,&pos)!=4) continue;
        if (e->nd<0||e->nd>5) continue;
        int ok=1;
        for (int d=0; d<e->nd; d++){ int used=0; if(sscanf(line+pos," %ld%n",&e->shape[d],&used)!=1){ok=0;break;} pos+=used; }
        if(!ok) continue;
        while(line[pos]==' ')pos++;
        snprintf(e->name,sizeof e->name,"%s",line+pos);
        e->name[strcspn(e->name,"\n")]=0;
        if (e->off+e->bytes>s->blob_bytes){ fprintf(stderr,"tensor %s out of blob range\n",e->name); fclose(f); return -1; }
        s->n_ents++;
    }
    fclose(f);
    /* Per-tensor parallel copy for NUMA-local first-touch.  Large tensors are split
     * into page-chunks static across threads (thread t owns the same byte/row range
     * it later reads in the matvec).  Tiny tensors (norms/biases) don't matter for
     * bandwidth, so they're copied one-per-thread via a single dynamic loop -- this
     * avoids thousands of tiny omp-for dispatches that ballooned load time. */
    const size_t PG=(size_t)64<<10, BIG=(size_t)256<<10;
    long ne=s->n_ents;
    if (!use_arena) { close(fd); return 0; }   /* no copy: read from the mapping */
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        for (long i=0;i<ne;i++) {
            const manifest_ent *e=&s->ents[i];
            if (e->bytes < BIG) continue;
            long nchunk=(long)((e->bytes+PG-1)/PG);
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (long c=0;c<nchunk;c++) {
                size_t o=(size_t)c*PG, len=e->bytes-o<PG?e->bytes-o:PG;
                memcpy(arena+e->off+o, fmap+e->off+o, len);
            }
        }
#ifdef _OPENMP
        #pragma omp for schedule(dynamic,16) nowait
#endif
        for (long i=0;i<ne;i++) {
            const manifest_ent *e=&s->ents[i];
            if (e->bytes >= BIG) continue;
            memcpy(arena+e->off, fmap+e->off, e->bytes);
        }
    }
    munmap((void*)fmap,nb); close(fd);
    return 0;
}

/* ---------------- allocation: checked, accounted, and reported ----------------
 * Every large allocation goes through these so that (a) a failure names what was
 * being allocated and how big it was instead of dying on a NULL deref deep in a
 * kernel, and (b) the totals can be reported up front, which is what makes an
 * over-large context fail with a budget rather than an OOM kill. */
static size_t g_mem_weights = 0, g_mem_kv = 0, g_mem_scratch = 0, g_mem_other = 0;

static const char *laguna_hsize(size_t b, char *buf, size_t n) {
    const char *u[] = {"B","KB","MB","GB","TB"}; int i=0; double v=(double)b;
    while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
    snprintf(buf, n, "%.2f %s", v, u[i]); return buf;
}
/* Total memory the OS is willing to give us right now (MemAvailable accounts for
 * reclaimable page cache, unlike MemFree).  0 if unknown. */
static size_t laguna_mem_available(void) {
    FILE *f = fopen("/proc/meminfo", "r"); if (!f) return 0;
    char line[256]; size_t kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    fclose(f); return kb * 1024;
}
static void *laguna_xalloc(size_t bytes, size_t *bucket, const char *what) {
    void *p = NULL;
    if (bytes == 0) return NULL;
    if (posix_memalign(&p, 256, bytes) != 0 || !p) {
        char b1[32], b2[32];
        fprintf(stderr, "FATAL: could not allocate %s for %s (available %s)\n",
                laguna_hsize(bytes, b1, sizeof b1), what,
                laguna_hsize(laguna_mem_available(), b2, sizeof b2));
        exit(1);
    }
    if (bucket) *bucket += bytes;
    return p;
}

/* Bump allocator for re-quantized weights, handing out of big (1 GB) chunks.
 *
 * These MUST NOT be thousands of separate posix_memalign blocks.  Decode streams
 * every byte of the quantized weights each token, and ~3000 mid-size mappings
 * cost ~30% of decode versus the same bytes in a handful of large ones (measured:
 * 13.7 -> 19.9 tok/s) -- the big contiguous mappings are what the OS backs with
 * large pages, and at 13 GB streamed per token the TLB behaviour dominates.
 * Kernel speed in isolation says nothing about this; the microbenchmarks reuse one
 * small resident array and cannot see it.
 *
 * Chunks are never freed (process lifetime).  NUMA first-touch still comes from
 * the quantization loops, which write each row from the thread that later reads it. */
#define LAGUNA_QCHUNK ((size_t)1<<30)
static unsigned char *g_qcur = NULL; static size_t g_qleft = 0;
static void *qalloc(size_t n) {
    n = (n + 255u) & ~(size_t)255u;
    if (n > g_qleft) {
        size_t want = n > LAGUNA_QCHUNK ? n : LAGUNA_QCHUNK;
        void *p = mmap(NULL, want, PROT_READ|PROT_WRITE,
                       MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            char b1[32], b2[32], b3[32];
            fprintf(stderr, "FATAL: weight arena could not grow by %s "
                    "(%s already committed, %s available)\n",
                    laguna_hsize(want,b1,sizeof b1), laguna_hsize(g_mem_weights,b2,sizeof b2),
                    laguna_hsize(laguna_mem_available(),b3,sizeof b3));
            exit(1);
        }
        g_qcur = p; g_qleft = want; g_mem_weights += want;
    }
    void *r = g_qcur; g_qcur += n; g_qleft -= n; return r;
}

/* Load one linear weight from the stage.  int8 and fp8 builds: quantize bf16 ->
 * per-row W8 at load.  bf16 build (-DLAGUNA_BF16): point straight at the staged
 * bf16. */
#if defined(LAGUNA_BF16)
static laguna_lin stage_lin(const laguna_stage *s, const char *name, int rows, int cols) {
    (void)rows; (void)cols; return (laguna_lin)stage_req(s, name);
}
#else
static laguna_lin stage_lin(const laguna_stage *s, const char *name, int rows, int cols) {
    const uint16_t *w = stage_req(s, name);
    laguna_w8 r;
    r.q = qalloc((size_t)rows*cols);
    r.s = qalloc((size_t)rows*sizeof(float));
    laguna_quant_w8(r.q, r.s, w, rows, cols);
    return r;
}
#endif

#if defined(LAGUNA_FP8)
/* --fp8-exact keeps the exact e4m3 LUT-gather kernels (A/B reference) instead of
 * the int8-per-block re-quantization.  Set from argv before model_build. */
static int g_fp8_exact = 0;
static void alloc_w8b(laguna_w8b *w, int rows, int cols) {
    int cblk = cols/LAGUNA_FP8_BLK;
    w->q = qalloc((size_t)rows*cols);
    w->s = qalloc((size_t)rows*cblk*sizeof(float));
}
#endif

#if defined(LAGUNA_FP8)
/* Once every hot tensor has been re-quantized, the blob is dead weight -- but as a
 * mapping its pages stay in the page cache, and 17.8 GB of those alongside ~13 GB
 * of int8 weights does not fit a 32 GB node.  The kernel then evicts exactly the
 * pages the still-bf16 norms live on, and decode faults them back from local SSD
 * on EVERY token.  Chunked prefill barely notices (one fault amortized over 256
 * tokens); decode lost ~45% (17.2 -> 9.8 tok/s) before this.
 * So: copy the few tensors that stay bf16 (embed + norms, ~620 MB) into anonymous
 * memory and unmap the blob. */
static const void *dup_blob(const void *p, size_t bytes) {
    void *q = malloc(bytes);
    if (!q) { fprintf(stderr,"FATAL: dup_blob %zu bytes failed\n",bytes); exit(1); }
    memcpy(q, p, bytes);
    return q;
}
static void fp8_release_blob(laguna_model *m, laguna_stage *s) {
    if (g_fp8_exact || !s->blob) return;      /* EXACT mode reads e4m3 bytes hot */
    size_t H = LAGUNA_HIDDEN, hd = LAGUNA_HEAD_DIM;
    m->embed      = dup_blob(m->embed, (size_t)LAGUNA_VOCAB*H*sizeof(uint16_t));
    m->final_norm = dup_blob(m->final_norm, H*sizeof(uint16_t));
    for (int L = 0; L < m->n_layers; ++L) {
        laguna_layer *ly = &m->layers[L];
        ly->in_ln   = dup_blob(ly->in_ln,   H*sizeof(uint16_t));
        ly->post_ln = dup_blob(ly->post_ln, H*sizeof(uint16_t));
        ly->q_norm  = dup_blob(ly->q_norm,  hd*sizeof(uint16_t));
        ly->k_norm  = dup_blob(ly->k_norm,  hd*sizeof(uint16_t));
        for (int e = 0; e < LAGUNA_EXPERTS; ++e) {   /* e4m3 sources now dead */
            laguna_expert *ex = &ly->experts[e];
            ex->gate = ex->up = ex->down = NULL;
            ex->gs = ex->us = ex->ds = NULL;
        }
    }
    munmap((void*)s->blob, s->blob_bytes);
    s->blob = NULL; s->blob_bytes = 0;
    free(s->ents); s->ents = NULL; s->n_ents = 0;
}
#endif


/* ---------------- long-context budget ----------------
 * Everything that scales with context is known in closed form before a byte is
 * allocated, so an over-large --maxpos can be refused with a budget and the
 * largest workable value, rather than being discovered by the OOM killer part way
 * through a 20-minute prefill.
 *
 * Per position the only growing term is the FULL-attention layers' KV: sliding
 * layers are ringed at LAGUNA_SLIDING_CAP and cost a constant.  Scratch is
 * dominated by sc->scores, which is MAX_HEADS*maxpos floats. */
static size_t laguna_ctx_bytes_per_pos(int n_layers) {
    int n_full = 0;
    for (int L = 0; L < n_layers; ++L) if (L % 4 == 0) ++n_full;
    size_t kv  = (size_t)n_full * LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM * 2 /*K+V*/ * sizeof(uint16_t);
    size_t rope = (size_t)(LAGUNA_ROPE_FULL_DIM/2 + LAGUNA_ROPE_SLIDING_DIM/2) * 2 /*cos+sin*/ * sizeof(float);
    size_t scores = (size_t)LAGUNA_MAX_HEADS * sizeof(float);
    return kv + rope + scores;
}
static size_t laguna_ctx_fixed_bytes(int n_layers) {
    int n_slide = n_layers - (n_layers + 3) / 4;
    return (size_t)n_slide * LAGUNA_SLIDING_CAP * LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM * 2 * sizeof(uint16_t);
}
static void laguna_report_context_budget(const laguna_model *m, int n_layers, int max_pos, int rank) {
    (void)m;
    size_t per   = laguna_ctx_bytes_per_pos(n_layers);
    size_t fixed = laguna_ctx_fixed_bytes(n_layers);
    size_t need  = fixed + per * (size_t)max_pos;
    size_t avail = laguna_mem_available();
    char b1[32], b2[32], b3[32], b4[32], b5[32];
    if (rank == 0)
        fprintf(stderr, "  context budget: maxpos %d needs %s (%s/token growing + %s fixed); "
                        "weights hold %s, %s available\n",
                max_pos, laguna_hsize(need,b1,sizeof b1), laguna_hsize(per,b2,sizeof b2),
                laguna_hsize(fixed,b3,sizeof b3), laguna_hsize(g_mem_weights,b4,sizeof b4),
                laguna_hsize(avail,b5,sizeof b5));
    /* Leave headroom for the chunk scratch and transient copies; refuse early if
     * the growing part alone cannot fit. */
    if (avail && need + (size_t)(256u<<20) > avail) {
        long fits = avail > fixed + (size_t)(256u<<20)
                  ? (long)((avail - fixed - (size_t)(256u<<20)) / per) : 0;
        fprintf(stderr, "FATAL: context of %d does not fit: needs %s, only %s available.\n"
                        "       Largest --maxpos that fits here is about %ld.\n",
                max_pos, laguna_hsize(need,b1,sizeof b1), laguna_hsize(avail,b2,sizeof b2), fits);
        (void)b3; (void)b4; (void)b5;
        exit(1);
    }
}

/* Fill the model from a rank-local stage. n_layers lets a bring-up run truncate.
 * Attention/dense/shared/router/lm_head are quantized to int8 (W8) at load to
 * halve the per-token weight bandwidth; norms/embed stay bf16, experts stay int4. */
static void model_build(laguna_model *m, const laguna_stage *s, int n_layers,
                        int max_pos, int rank, int ep_size) {
    memset(m,0,sizeof *m);
#if defined(LAGUNA_FP8)
    laguna_fp8_init_lut();
#endif
    m->n_layers=n_layers; m->max_pos=max_pos; m->ep_rank=rank; m->ep_size=ep_size;
    m->embed      = stage_req(s,"model.embed_tokens.weight");
    m->lm_head    = stage_lin(s,"lm_head.weight", LAGUNA_VOCAB, LAGUNA_HIDDEN);
    m->final_norm = stage_req(s,"model.norm.weight");
    char nm[320];
    for (int L=0; L<n_layers; ++L) {
        laguna_layer *ly=&m->layers[L];
        int full = (L % 4 == 0);            /* layer_types: layer%4==0 => full_attention */
        ly->is_sliding = !full;
        ly->num_heads  = full ? LAGUNA_FULL_HEADS : LAGUNA_SLIDING_HEADS;
        ly->is_moe     = (L != 0);          /* mlp_only_layers=[0] => layer 0 dense */
        int nh=ly->num_heads, H=LAGUNA_HIDDEN, hd=LAGUNA_HEAD_DIM;
        #define REQ(field,suffix) do{ snprintf(nm,sizeof nm,"model.layers.%d." suffix,L); ly->field=stage_req(s,nm);}while(0)
        #define QREQ(field,suffix,rows,cols) do{ snprintf(nm,sizeof nm,"model.layers.%d." suffix,L); ly->field=stage_lin(s,nm,(rows),(cols));}while(0)
        QREQ(q_proj,  "self_attn.q_proj.weight", nh*hd, H);
        QREQ(k_proj,  "self_attn.k_proj.weight", LAGUNA_KV_HEADS*hd, H);
        QREQ(v_proj,  "self_attn.v_proj.weight", LAGUNA_KV_HEADS*hd, H);
        QREQ(o_proj,  "self_attn.o_proj.weight", H, nh*hd);
        QREQ(g_proj,  "self_attn.g_proj.weight", nh, H);
        REQ(q_norm,  "self_attn.q_norm.weight");
        REQ(k_norm,  "self_attn.k_norm.weight");
        REQ(in_ln,   "input_layernorm.weight");
        REQ(post_ln, "post_attention_layernorm.weight");
        if (!ly->is_moe) {
            QREQ(dense_gate,"mlp.gate_proj.weight", LAGUNA_DENSE_INTER, H);
            QREQ(dense_up,  "mlp.up_proj.weight",   LAGUNA_DENSE_INTER, H);
            QREQ(dense_down,"mlp.down_proj.weight", H, LAGUNA_DENSE_INTER);
        } else {
            QREQ(shared_gate,"mlp.shared_expert.gate_proj.weight", LAGUNA_SHARED_INTER, H);
            QREQ(shared_up,  "mlp.shared_expert.up_proj.weight",   LAGUNA_SHARED_INTER, H);
            QREQ(shared_down,"mlp.shared_expert.down_proj.weight", H, LAGUNA_SHARED_INTER);
            QREQ(router_w,   "mlp.gate.weight", LAGUNA_EXPERTS, H);
            /* router bias: F32 in the int4 checkpoint, BF16 in the bf16 one.
             * Read the manifest dtype and convert to a f32 [256] so the router
             * kernel is uniform. */
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.e_score_correction_bias",L);
            { float *bias=malloc(LAGUNA_EXPERTS*sizeof(float));
              const void *bp=stage_req(s,nm);
              if (stage_dtype_is_bf16(s,nm)) { const uint16_t *b=bp; for(int e=0;e<LAGUNA_EXPERTS;++e) bias[e]=laguna_bf16_to_f32(b[e]); }
              else { memcpy(bias, bp, LAGUNA_EXPERTS*sizeof(float)); }
              ly->router_bias=bias; }
            for (int e=0;e<LAGUNA_EXPERTS;++e) {
                if (e % ep_size != rank) continue;   /* not owned by this rank */
                laguna_expert *ex=&ly->experts[e];
#if defined(LAGUNA_FP8)
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.gate_proj.weight",      L,e); ex->gate=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.gate_proj.weight_scale",L,e); ex->gs  =stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.up_proj.weight",        L,e); ex->up  =stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.up_proj.weight_scale",  L,e); ex->us  =stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.down_proj.weight",      L,e); ex->down=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.down_proj.weight_scale",L,e); ex->ds  =stage_req(s,nm);
                if (!g_fp8_exact) {
                    /* Re-quantize e4m3 -> int8 per 128-col block (halves the bytes and
                     * drops the LUT gather; see laguna_fp8_to_i8blk). */
                    int inter=LAGUNA_EXPERT_INTER;
                    alloc_w8b(&ex->qg, inter, H); laguna_fp8_to_i8blk(&ex->qg, ex->gate, ex->gs, inter, H);
                    alloc_w8b(&ex->qu, inter, H); laguna_fp8_to_i8blk(&ex->qu, ex->up,   ex->us, inter, H);
                    alloc_w8b(&ex->qd, H, inter); laguna_fp8_to_i8blk(&ex->qd, ex->down, ex->ds, H, inter);
                }
#elif defined(LAGUNA_BF16)
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.gate_proj.weight",L,e); ex->gate=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.up_proj.weight",  L,e); ex->up  =stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.down_proj.weight", L,e); ex->down=stage_req(s,nm);
#else
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.gate_proj.weight_packed",L,e); ex->gp=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.gate_proj.weight_scale", L,e); ex->gs=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.up_proj.weight_packed",  L,e); ex->up=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.up_proj.weight_scale",   L,e); ex->us=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.down_proj.weight_packed",L,e); ex->dp=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.down_proj.weight_scale", L,e); ex->ds=stage_req(s,nm);
#endif
                ex->present=1;
            }
        }
        #undef REQ
        #undef QREQ
    }
    /* rope tables + KV cache */
    int hf=LAGUNA_ROPE_FULL_DIM/2, hs=LAGUNA_ROPE_SLIDING_DIM/2;
    laguna_report_context_budget(m, n_layers, max_pos, rank);
    m->full_cos=laguna_xalloc((size_t)max_pos*hf*sizeof(float),&g_mem_other,"rope full_cos");
    m->full_sin=laguna_xalloc((size_t)max_pos*hf*sizeof(float),&g_mem_other,"rope full_sin");
    m->swa_cos =laguna_xalloc((size_t)max_pos*hs*sizeof(float),&g_mem_other,"rope swa_cos");
    m->swa_sin =laguna_xalloc((size_t)max_pos*hs*sizeof(float),&g_mem_other,"rope swa_sin");
    laguna_build_rope_tables(m);
    /* Per-layer KV: full-attention layers keep the whole context; sliding layers
     * use a SLIDING_WINDOW ring buffer.  Keeps 128k KV at ~6.5 GB not ~26. */
    size_t kv_elems=0; int slot=LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM;
    for (int L=0; L<n_layers; ++L) {
        int cap = m->layers[L].is_sliding ? LAGUNA_SLIDING_CAP : max_pos;
        m->kv_cap[L]=cap; m->kv_off[L]=kv_elems; kv_elems += (size_t)cap*slot;
    }
    m->kcache=laguna_xalloc(kv_elems*sizeof(uint16_t),&g_mem_kv,"K cache");
    m->vcache=laguna_xalloc(kv_elems*sizeof(uint16_t),&g_mem_kv,"V cache");
}

/* ============================ forward pass ============================ */
/* per-token scratch (allocated once) */
typedef struct {
    float *n1, *n2;        /* [hidden] normed activations */
    float *qf, *kf, *vf;   /* q [maxheads*128], k/v [8*128] */
    float *gf;             /* [maxheads] attention gate */
    float *ao;             /* [maxheads*128] attention output pre-o_proj */
    float *attn_out;       /* [hidden] */
    float *inter_a, *inter_b; /* [dense_inter] swiglu scratch */
    float *partial;        /* [hidden] routed accumulator */
    float *shared;         /* [hidden] */
    float *logits;         /* [vocab] */
    float *scores;         /* [MAX_HEADS * max_pos] per-head attention scores */
    /* batched chunked-prefill scratch: PCHUNK tokens (token-major layout) */
    float *cn1,*cn2,*cq,*ck,*cv,*cg,*cao,*cattn,*cia,*cib,*cpart,*cshared;
    /* query-block flash-attention state (full layers): per (head, query) */
    float *fm,*fl,*facc;   /* fm/fl [MAX_HEADS*PCHUNK], facc [MAX_HEADS*PCHUNK*HEAD_DIM] */
    /* batched-MoE (chunked prefill): routing table + per-expert gather buffer */
    int   *rids;           /* [PCHUNK*ACTIVE] selected expert ids per token */
    float *rrw;            /* [PCHUNK*ACTIVE] routing weights */
    float *xe, *ye;        /* [PCHUNK*hidden] gathered expert in/out */
    float *crouter;        /* [PCHUNK*EXPERTS] batched router logits */
} laguna_scratch;

#define LAGUNA_PCHUNK 256
/* attention_slide_flash writes a whole chunk's K/V before any of the chunk's
 * queries attend, so the sliding ring must hold the window plus the chunk. */
_Static_assert(LAGUNA_SLIDING_CAP >= LAGUNA_SLIDING_WINDOW + LAGUNA_PCHUNK - 1,
               "sliding ring too small for LAGUNA_PCHUNK: chunk writes would clobber "
               "slots the chunk's own earlier queries still need");
static void scratch_alloc(laguna_scratch *sc, int max_pos) {
#define SALLOC(field, nbytes, what) sc->field = laguna_xalloc((nbytes), &g_mem_scratch, what)
    { int C=LAGUNA_PCHUNK, H=LAGUNA_HIDDEN;
      size_t cH = (size_t)C*H*sizeof(float);
      SALLOC(cn1, cH, "chunk n1");                    SALLOC(cn2, cH, "chunk n2");
      SALLOC(cq,  (size_t)C*LAGUNA_MAX_HEADS*LAGUNA_HEAD_DIM*sizeof(float), "chunk q");
      SALLOC(ck,  (size_t)C*LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM*sizeof(float),  "chunk k");
      SALLOC(cv,  (size_t)C*LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM*sizeof(float),  "chunk v");
      SALLOC(cg,  (size_t)C*LAGUNA_MAX_HEADS*sizeof(float), "chunk gate");
      SALLOC(cao, (size_t)C*LAGUNA_MAX_HEADS*LAGUNA_HEAD_DIM*sizeof(float), "chunk attn out");
      SALLOC(cattn, cH, "chunk attn proj");
      SALLOC(cia, (size_t)C*LAGUNA_DENSE_INTER*sizeof(float), "chunk inter a");
      SALLOC(cib, (size_t)C*LAGUNA_DENSE_INTER*sizeof(float), "chunk inter b");
      SALLOC(cpart, cH, "chunk routed partial");      SALLOC(cshared, cH, "chunk shared");
      SALLOC(fm, (size_t)LAGUNA_MAX_HEADS*C*sizeof(float), "flash max");
      SALLOC(fl, (size_t)LAGUNA_MAX_HEADS*C*sizeof(float), "flash denom");
      SALLOC(facc, (size_t)LAGUNA_MAX_HEADS*C*LAGUNA_HEAD_DIM*sizeof(float), "flash acc");
      SALLOC(rids, (size_t)C*LAGUNA_ACTIVE*sizeof(int), "routing ids");
      SALLOC(rrw,  (size_t)C*LAGUNA_ACTIVE*sizeof(float), "routing weights");
      SALLOC(xe, cH, "expert gather in");              SALLOC(ye, cH, "expert gather out");
      SALLOC(crouter, (size_t)C*LAGUNA_EXPERTS*sizeof(float), "chunk router logits"); }
    SALLOC(n1, LAGUNA_HIDDEN*sizeof(float), "n1");
    SALLOC(n2, LAGUNA_HIDDEN*sizeof(float), "n2");
    SALLOC(qf, (size_t)LAGUNA_MAX_HEADS*LAGUNA_HEAD_DIM*sizeof(float), "q");
    SALLOC(kf, LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM*sizeof(float), "k");
    SALLOC(vf, LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM*sizeof(float), "v");
    SALLOC(gf, LAGUNA_MAX_HEADS*sizeof(float), "attention gate");
    SALLOC(ao, (size_t)LAGUNA_MAX_HEADS*LAGUNA_HEAD_DIM*sizeof(float), "attn out");
    SALLOC(attn_out, LAGUNA_HIDDEN*sizeof(float), "attn proj out");
    SALLOC(inter_a, LAGUNA_DENSE_INTER*sizeof(float), "inter a");
    SALLOC(inter_b, LAGUNA_DENSE_INTER*sizeof(float), "inter b");
    SALLOC(partial, LAGUNA_HIDDEN*sizeof(float), "routed partial");
    SALLOC(shared,  LAGUNA_HIDDEN*sizeof(float), "shared expert out");
    SALLOC(logits,  LAGUNA_VOCAB*sizeof(float), "logits");
    /* the only maxpos-sized scratch: MAX_HEADS * maxpos floats (~38 MB at 128k) */
    SALLOC(scores, (size_t)LAGUNA_MAX_HEADS*max_pos*sizeof(float), "attention scores");
#undef SALLOC
}

/* One line summarising where the process memory went; printed by rank 0 after the
 * model is built so a long-context run shows its budget before it commits to it. */
static void laguna_report_memory(void) {
    char b1[32],b2[32],b3[32],b4[32],b5[32],b6[32];
    size_t tot = g_mem_weights+g_mem_kv+g_mem_scratch+g_mem_other;
    fprintf(stderr, "  memory: weights %s + KV %s + scratch %s + other %s = %s (%s still available)\n",
            laguna_hsize(g_mem_weights,b1,sizeof b1), laguna_hsize(g_mem_kv,b2,sizeof b2),
            laguna_hsize(g_mem_scratch,b3,sizeof b3), laguna_hsize(g_mem_other,b4,sizeof b4),
            laguna_hsize(tot,b5,sizeof b5), laguna_hsize(laguna_mem_available(),b6,sizeof b6));
}

/* Attention for one token at `pos`. `x` is input_layernorm output (n1). Writes attn_out. */
/* SVE-vectorized per-position attention inner ops (bf16 KV widened to f32).
 * These are the O(context) hot loops that dominate long-context prefill. */
#if defined(__ARM_FEATURE_SVE)
static inline float laguna_qkdot(const float *restrict q, const uint16_t *restrict k, int hd) {
    svfloat32_t a = svdup_f32(0);
    for (int d=0; d<hd; d+=(int)svcntw()) { svbool_t pg=svwhilelt_b32(d,hd);
        a = svmla_f32_x(pg, a, svld1_f32(pg,q+d), laguna_ld_bf16(pg,k+d)); }
    return svaddv_f32(svptrue_b32(), a);
}
static inline void laguna_vaxpy(float *restrict acc, const uint16_t *restrict v,
                                float p, float corr, int hd) {
    svfloat32_t sp=svdup_f32(p), scv=svdup_f32(corr);
    for (int d=0; d<hd; d+=(int)svcntw()) { svbool_t pg=svwhilelt_b32(d,hd);
        svfloat32_t a = svmul_f32_x(pg, svld1_f32(pg,acc+d), scv);
        svst1_f32(pg, acc+d, svmla_f32_x(pg, a, sp, laguna_ld_bf16(pg,v+d))); }
}

/* ---- run-based attention inner loops (head_dim == 8 vectors at VL=16) ----
 * These replace per-key laguna_qkdot / laguna_vaxpy calls over a CONTIGUOUS run of
 * key slots.  Two things dominate attention at long context and both are fixed by
 * keeping state in registers across the run:
 *   qk : one accumulator per dot is an 8-long serial FMLA chain (~9-cycle latency
 *        each).  Four keys at a time with two accumulators each gives 8 independent
 *        chains of depth 4, and q stays in registers instead of being re-read.
 *   av : the old laguna_vaxpy read AND wrote all 128 floats of acc for every key --
 *        1 KB of traffic per 256-byte V row.  Holding acc in 8 registers for the
 *        whole run removes that entirely.
 * hd is always LAGUNA_HEAD_DIM (128); the generic path is kept as a fallback. */
#define LAGUNA_AV_NV 8      /* 128 / 16 */
/* bf16 halves of a full-width u16 load, widened to f32 */
#define LAGUNA_UNLO(pg,h) svreinterpret_f32_u32(svlsl_n_u32_x((pg),svunpklo_u32(h),16))
#define LAGUNA_UNHI(pg,h) svreinterpret_f32_u32(svlsl_n_u32_x((pg),svunpkhi_u32(h),16))
/* uzp1(a,b)+uzp2(a,b) = [a0+a1, a2+a3, ..., b0+b1, ...] -- four stages reduce 16
 * accumulators to one vector of 16 lane-sums without any svaddv. */
#define LAGUNA_BFLY(pg,a,b) svadd_f32_x((pg), svuzp1_f32((a),(b)), svuzp2_f32((a),(b)))
static inline int laguna_run_ok(int hd) { return hd == LAGUNA_AV_NV*(int)svcntw(); }

/* sco[i] = dot(q, k[i]) * scale, for i in [0,n), keys at k + i*kvstride */
static inline void laguna_qk_run(float *restrict sco, const float *restrict q,
                                 const uint16_t *restrict k, int kvstride,
                                 int n, float scale, int hd) {
    if (!laguna_run_ok(hd)) {
        for (int i=0;i<n;++i) sco[i]=laguna_qkdot(q,k+(size_t)i*kvstride,hd)*scale;
        return;
    }
    svbool_t pt=svptrue_b32(), ph=svptrue_b16(); int VL=(int)svcntw();
    svfloat32_t q0=svld1_f32(pt,q+0*VL),q1=svld1_f32(pt,q+1*VL),
                q2=svld1_f32(pt,q+2*VL),q3=svld1_f32(pt,q+3*VL),
                q4=svld1_f32(pt,q+4*VL),q5=svld1_f32(pt,q+5*VL),
                q6=svld1_f32(pt,q+6*VL),q7=svld1_f32(pt,q+7*VL);
    int i=0;
    /* Same full-width-load trick as laguna_av_run: 4 svld1_u16 + unpack instead of
     * 8 widening svld1uh_u32.  Each q register still meets the same dims in the
     * same order, so this is bit-identical. */
    for (; i+4<=n; i+=4) {
        const uint16_t *k0=k+(size_t)(i+0)*kvstride,*k1=k+(size_t)(i+1)*kvstride,
                       *k2=k+(size_t)(i+2)*kvstride,*k3=k+(size_t)(i+3)*kvstride;
        svuint16_t g0=svld1_u16(ph,k0+0*2*VL),g1=svld1_u16(ph,k0+1*2*VL),
                   g2=svld1_u16(ph,k0+2*2*VL),g3=svld1_u16(ph,k0+3*2*VL);
        svuint16_t m0=svld1_u16(ph,k1+0*2*VL),m1=svld1_u16(ph,k1+1*2*VL),
                   m2=svld1_u16(ph,k1+2*2*VL),m3=svld1_u16(ph,k1+3*2*VL);
        svuint16_t n0=svld1_u16(ph,k2+0*2*VL),n1=svld1_u16(ph,k2+1*2*VL),
                   n2=svld1_u16(ph,k2+2*2*VL),n3=svld1_u16(ph,k2+3*2*VL);
        svuint16_t r0=svld1_u16(ph,k3+0*2*VL),r1=svld1_u16(ph,k3+1*2*VL),
                   r2=svld1_u16(ph,k3+2*2*VL),r3=svld1_u16(ph,k3+3*2*VL);
        svfloat32_t a0=svmul_f32_x(pt,q0,LAGUNA_UNLO(pt,g0));
        svfloat32_t b0=svmul_f32_x(pt,q4,LAGUNA_UNLO(pt,g2));
        svfloat32_t a1=svmul_f32_x(pt,q0,LAGUNA_UNLO(pt,m0));
        svfloat32_t b1=svmul_f32_x(pt,q4,LAGUNA_UNLO(pt,m2));
        svfloat32_t a2=svmul_f32_x(pt,q0,LAGUNA_UNLO(pt,n0));
        svfloat32_t b2=svmul_f32_x(pt,q4,LAGUNA_UNLO(pt,n2));
        svfloat32_t a3=svmul_f32_x(pt,q0,LAGUNA_UNLO(pt,r0));
        svfloat32_t b3=svmul_f32_x(pt,q4,LAGUNA_UNLO(pt,r2));
        a0=svmla_f32_x(pt,a0,q1,LAGUNA_UNHI(pt,g0)); b0=svmla_f32_x(pt,b0,q5,LAGUNA_UNHI(pt,g2));
        a1=svmla_f32_x(pt,a1,q1,LAGUNA_UNHI(pt,m0)); b1=svmla_f32_x(pt,b1,q5,LAGUNA_UNHI(pt,m2));
        a2=svmla_f32_x(pt,a2,q1,LAGUNA_UNHI(pt,n0)); b2=svmla_f32_x(pt,b2,q5,LAGUNA_UNHI(pt,n2));
        a3=svmla_f32_x(pt,a3,q1,LAGUNA_UNHI(pt,r0)); b3=svmla_f32_x(pt,b3,q5,LAGUNA_UNHI(pt,r2));
        a0=svmla_f32_x(pt,a0,q2,LAGUNA_UNLO(pt,g1)); b0=svmla_f32_x(pt,b0,q6,LAGUNA_UNLO(pt,g3));
        a1=svmla_f32_x(pt,a1,q2,LAGUNA_UNLO(pt,m1)); b1=svmla_f32_x(pt,b1,q6,LAGUNA_UNLO(pt,m3));
        a2=svmla_f32_x(pt,a2,q2,LAGUNA_UNLO(pt,n1)); b2=svmla_f32_x(pt,b2,q6,LAGUNA_UNLO(pt,n3));
        a3=svmla_f32_x(pt,a3,q2,LAGUNA_UNLO(pt,r1)); b3=svmla_f32_x(pt,b3,q6,LAGUNA_UNLO(pt,r3));
        a0=svmla_f32_x(pt,a0,q3,LAGUNA_UNHI(pt,g1)); b0=svmla_f32_x(pt,b0,q7,LAGUNA_UNHI(pt,g3));
        a1=svmla_f32_x(pt,a1,q3,LAGUNA_UNHI(pt,m1)); b1=svmla_f32_x(pt,b1,q7,LAGUNA_UNHI(pt,m3));
        a2=svmla_f32_x(pt,a2,q3,LAGUNA_UNHI(pt,n1)); b2=svmla_f32_x(pt,b2,q7,LAGUNA_UNHI(pt,n3));
        a3=svmla_f32_x(pt,a3,q3,LAGUNA_UNHI(pt,r1)); b3=svmla_f32_x(pt,b3,q7,LAGUNA_UNHI(pt,r3));
        sco[i+0]=svaddv_f32(pt,svadd_f32_x(pt,a0,b0))*scale;
        sco[i+1]=svaddv_f32(pt,svadd_f32_x(pt,a1,b1))*scale;
        sco[i+2]=svaddv_f32(pt,svadd_f32_x(pt,a2,b2))*scale;
        sco[i+3]=svaddv_f32(pt,svadd_f32_x(pt,a3,b3))*scale;
    }
    for (; i<n; ++i) {
        const uint16_t *ki=k+(size_t)i*kvstride;
        svfloat32_t a=svmul_f32_x(pt,q0,laguna_ld_bf16(pt,ki+0*VL));
        svfloat32_t b=svmul_f32_x(pt,q4,laguna_ld_bf16(pt,ki+4*VL));
        a=svmla_f32_x(pt,a,q1,laguna_ld_bf16(pt,ki+1*VL)); b=svmla_f32_x(pt,b,q5,laguna_ld_bf16(pt,ki+5*VL));
        a=svmla_f32_x(pt,a,q2,laguna_ld_bf16(pt,ki+2*VL)); b=svmla_f32_x(pt,b,q6,laguna_ld_bf16(pt,ki+6*VL));
        a=svmla_f32_x(pt,a,q3,laguna_ld_bf16(pt,ki+3*VL)); b=svmla_f32_x(pt,b,q7,laguna_ld_bf16(pt,ki+7*VL));
        sco[i]=svaddv_f32(pt,svadd_f32_x(pt,a,b))*scale;
    }
}

/* acc[d] = acc[d]*corr + sum_i w[i]*v[i][d], keys at v + i*kvstride */
static inline void laguna_av_run(float *restrict acc, const float *restrict w,
                                 const uint16_t *restrict v, int kvstride,
                                 int n, float corr, int hd) {
    if (!laguna_run_ok(hd)) {
        for (int i=0;i<n;++i) laguna_vaxpy(acc, v+(size_t)i*kvstride, w[i], i==0?corr:1.0f, hd);
        return;
    }
    svbool_t pt=svptrue_b32(), ph=svptrue_b16(); int VL=(int)svcntw();
    svfloat32_t c=svdup_f32(corr);
    svfloat32_t a0=svmul_f32_x(pt,svld1_f32(pt,acc+0*VL),c),a1=svmul_f32_x(pt,svld1_f32(pt,acc+1*VL),c),
                a2=svmul_f32_x(pt,svld1_f32(pt,acc+2*VL),c),a3=svmul_f32_x(pt,svld1_f32(pt,acc+3*VL),c),
                a4=svmul_f32_x(pt,svld1_f32(pt,acc+4*VL),c),a5=svmul_f32_x(pt,svld1_f32(pt,acc+5*VL),c),
                a6=svmul_f32_x(pt,svld1_f32(pt,acc+6*VL),c),a7=svmul_f32_x(pt,svld1_f32(pt,acc+7*VL),c);
    /* Read V with FULL-WIDTH u16 loads and unpack, rather than 8 widening
     * svld1uh_u32.  A widening load fills 16 f32 lanes from only 32 bytes, so it
     * spends a whole load slot on half a vector; 4 full loads + unpack move the
     * same bytes in half the slots and measure 1.44x (35.2 -> 24.5 cyc/key).
     * Each accumulator still sees the same dims in the same order, so the result
     * is bit-identical to the widening-load form. */
    for (int i=0;i<n;++i) {
        const uint16_t *vi=v+(size_t)i*kvstride; svfloat32_t p=svdup_f32(w[i]);
        svuint16_t h0=svld1_u16(ph,vi+0*2*VL), h1=svld1_u16(ph,vi+1*2*VL);
        svuint16_t h2=svld1_u16(ph,vi+2*2*VL), h3=svld1_u16(ph,vi+3*2*VL);
        a0=svmla_f32_x(pt,a0,p,LAGUNA_UNLO(pt,h0));
        a1=svmla_f32_x(pt,a1,p,LAGUNA_UNHI(pt,h0));
        a2=svmla_f32_x(pt,a2,p,LAGUNA_UNLO(pt,h1));
        a3=svmla_f32_x(pt,a3,p,LAGUNA_UNHI(pt,h1));
        a4=svmla_f32_x(pt,a4,p,LAGUNA_UNLO(pt,h2));
        a5=svmla_f32_x(pt,a5,p,LAGUNA_UNHI(pt,h2));
        a6=svmla_f32_x(pt,a6,p,LAGUNA_UNLO(pt,h3));
        a7=svmla_f32_x(pt,a7,p,LAGUNA_UNHI(pt,h3));
    }
    svst1_f32(pt,acc+0*VL,a0); svst1_f32(pt,acc+1*VL,a1);
    svst1_f32(pt,acc+2*VL,a2); svst1_f32(pt,acc+3*VL,a3);
    svst1_f32(pt,acc+4*VL,a4); svst1_f32(pt,acc+5*VL,a5);
    svst1_f32(pt,acc+6*VL,a6); svst1_f32(pt,acc+7*VL,a7);
}
#else
static inline float laguna_qkdot(const float *q, const uint16_t *k, int hd) {
    float s=0; for(int d=0;d<hd;++d) s+=q[d]*laguna_bf16_to_f32(k[d]); return s;
}
static inline void laguna_vaxpy(float *acc, const uint16_t *v, float p, float corr, int hd) {
    for(int d=0;d<hd;++d) acc[d]=acc[d]*corr + p*laguna_bf16_to_f32(v[d]);
}
static inline void laguna_qk_run(float *sco, const float *q, const uint16_t *k,
                                 int kvstride, int n, float scale, int hd) {
    for (int i=0;i<n;++i) sco[i]=laguna_qkdot(q,k+(size_t)i*kvstride,hd)*scale;
}
static inline void laguna_av_run(float *acc, const float *w, const uint16_t *v,
                                 int kvstride, int n, float corr, int hd) {
    for (int i=0;i<n;++i) laguna_vaxpy(acc, v+(size_t)i*kvstride, w[i], i==0?corr:1.0f, hd);
}
#endif

/* Attention core for ONE token: qk-norm + rope + KV-write + head loop, on the
 * pre-projected qf/kf/vf/gf, writing ao.  Shared by decode (forward_token) and
 * chunked prefill (which batches the q/k/v/g/o matvecs and calls this per token). */
static void attention_core(const laguna_model *m, const laguna_layer *ly, laguna_scratch *sc,
                           int layer, int pos, int nh,
                           float *qf, float *kf, float *vf, float *gf, float *ao) {
    int hd=LAGUNA_HEAD_DIM;
    int rot = ly->is_sliding ? LAGUNA_ROPE_SLIDING_DIM : LAGUNA_ROPE_FULL_DIM;
    const float *cosp = ly->is_sliding ? m->swa_cos : m->full_cos;
    const float *sinp = ly->is_sliding ? m->swa_sin : m->full_sin;
    int half = rot/2;
    const float *rc = cosp + (size_t)pos*half, *rs = sinp + (size_t)pos*half;

    /* q_norm + rope per query head */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h=0; h<nh; ++h) {
        float *q=qf+(size_t)h*hd; float tmp[LAGUNA_HEAD_DIM];
        laguna_rmsnorm(tmp, q, ly->q_norm, hd, LAGUNA_RMS_EPS);
        memcpy(q,tmp,sizeof tmp);
        laguna_rope_half(q, rc, rs, rot);
    }
    /* k_norm + rope per kv head, then store to cache at pos (ring slot pos%cap) */
    int cap=m->kv_cap[layer], kvstride=LAGUNA_KV_HEADS*hd;
    uint16_t *kbase=m->kcache+m->kv_off[layer], *vbase=m->vcache+m->kv_off[layer];
    uint16_t *kdst=kbase+(size_t)(pos%cap)*kvstride;
    uint16_t *vdst=vbase+(size_t)(pos%cap)*kvstride;
    for (int h=0; h<LAGUNA_KV_HEADS; ++h) {
        float *k=kf+(size_t)h*hd; float tmp[LAGUNA_HEAD_DIM];
        laguna_rmsnorm(tmp, k, ly->k_norm, hd, LAGUNA_RMS_EPS);
        memcpy(k,tmp,sizeof tmp);
        laguna_rope_half(k, rc, rs, rot);
        for (int d=0; d<hd; ++d) kdst[h*hd+d]=laguna_f32_to_bf16(k[d]);
        for (int d=0; d<hd; ++d) vdst[h*hd+d]=laguna_f32_to_bf16(vf[h*hd+d]);
    }
    /* attention per query head.  Sliding layers attend to the last `cap` positions
     * (the ring holds exactly those); full layers attend to all [0..pos]. */
    float scale=1.0f/sqrtf((float)hd);
    int kv_groups=nh/LAGUNA_KV_HEADS;
    /* Attended range is the WINDOW, never the ring capacity (cap >= window, and
     * they are no longer equal -- see LAGUNA_SLIDING_CAP). */
    int lo = ly->is_sliding ? (pos-LAGUNA_SLIDING_WINDOW+1) : 0; if(lo<0)lo=0;
    /* Per-head attention, parallel over heads.  Two passes over the key range with
     * a per-head score buffer: (1) scores = q.k*scale, track max; (2) VECTORIZED
     * softmax exp (FEXPA) -- the O(context) prefill bottleneck; (3) weighted sum of
     * V.  Contiguous key positions when cap doesn't wrap (full layers, or sliding
     * once wrapped) let qk/av stream. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h=0; h<nh; ++h) {
        const float *q=qf+(size_t)h*hd;
        int kvh=h/kv_groups;
        float *sco=sc->scores+(size_t)h*m->max_pos;   /* this head's scores */
        int nkeys=pos-lo+1;
        /* [lo,pos] is at most two contiguous slot runs (one if the ring doesn't
         * wrap, which is always the case for full-attention layers). */
        int s0=lo%cap, n1=cap-s0; if(n1>nkeys) n1=nkeys; int n2=nkeys-n1;
        laguna_qk_run(sco,    q, kbase+(size_t)s0*kvstride+(size_t)kvh*hd, kvstride, n1, scale, hd);
        if(n2) laguna_qk_run(sco+n1, q, kbase+(size_t)kvh*hd, kvstride, n2, scale, hd);
        float mx=-INFINITY;
        for (int i=0;i<nkeys;++i) if(sco[i]>mx) mx=sco[i];
        float l_i=laguna_exp_shift_sum(sco, nkeys, mx);   /* sco[i]=exp(sco[i]-mx) */
        float acc[LAGUNA_HEAD_DIM]; for(int d=0;d<hd;++d)acc[d]=0.0f;
        laguna_av_run(acc, sco,    vbase+(size_t)s0*kvstride+(size_t)kvh*hd, kvstride, n1, 0.0f, hd);
        if(n2) laguna_av_run(acc, sco+n1, vbase+(size_t)kvh*hd, kvstride, n2, 1.0f, hd);
        float gate=laguna_softplus(gf[h]);
        float s=gate/l_i; float *o=ao+(size_t)h*hd;
        for(int d=0;d<hd;++d) o[d]=acc[d]*s;
    }
}

/* Decode attention for one token: project q/k/v/g from n1, run the core, o_proj. */
static void attention(const laguna_model *m, const laguna_layer *ly, laguna_scratch *sc,
                      int layer, int pos) {
    int nh=ly->num_heads, hd=LAGUNA_HEAD_DIM;
    float *ys[4]={sc->qf,sc->kf,sc->vf,sc->gf};
    const laguna_lin *Ws[4]={&ly->q_proj,&ly->k_proj,&ly->v_proj,&ly->g_proj};
    int rws[4]={nh*hd, LAGUNA_KV_HEADS*hd, LAGUNA_KV_HEADS*hd, nh};
    laguna_lin_mv_multi(ys, Ws, rws, 4, sc->n1, LAGUNA_HIDDEN);
    attention_core(m, ly, sc, layer, pos, nh, sc->qf, sc->kf, sc->vf, sc->gf, sc->ao);
    laguna_lin_mv(sc->attn_out, &ly->o_proj, sc->ao, LAGUNA_HIDDEN, nh*hd);
}

/* Query-block flash attention for a chunk of C tokens on a FULL-attention layer.
 * Batched q/k/v/g are token-major (Q[c*nh*hd], K/V[c*8*hd], G[c*nh]); writes AO.
 * Amortizes KV bandwidth ~C x: for each key block, the C queries reuse it from
 * cache, so KV is read from HBM once instead of once per query.  Online (flash)
 * softmax with FEXPA exp; causal via a prefix pass [0,pos0) + a diagonal pass. */
#ifndef LAGUNA_QT
/* Query-tile size for the flash paths.  nh*ceil(C/QT) tasks must comfortably
 * exceed the thread count or the tail round wastes most of the machine. */
#define LAGUNA_QT 32
#endif
#ifndef LAGUNA_KB
/* Key-block size for the flash paths.  K+V for one block is 2*KB*head_dim*2 bytes
 * (64 KB at KB=128, i.e. the whole of A64FX's 64 KB L1D). */
#define LAGUNA_KB 128
#endif
static void attention_full_flash(const laguna_model *m, const laguna_layer *ly, laguna_scratch *sc,
                                 int layer, int pos0, int C, int nh,
                                 float *Q, float *K, float *V, float *G, float *AO) {
    int hd=LAGUNA_HEAD_DIM, kv_groups=nh/LAGUNA_KV_HEADS, kvstride=LAGUNA_KV_HEADS*hd;
    int rot=LAGUNA_ROPE_FULL_DIM, half=rot/2;
    uint16_t *kbase=m->kcache+m->kv_off[layer], *vbase=m->vcache+m->kv_off[layer];
    float scale=1.0f/sqrtf((float)hd);
    /* 1. qk-norm + rope + write KV (parallel over chunk tokens; full cache slot=pos) */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int c=0;c<C;++c) {
        int pos=pos0+c;
        const float *rc=m->full_cos+(size_t)pos*half, *rs=m->full_sin+(size_t)pos*half;
        for (int h=0;h<nh;++h){ float *q=Q+(size_t)c*nh*hd+(size_t)h*hd; float tmp[LAGUNA_HEAD_DIM];
            laguna_rmsnorm(tmp,q,ly->q_norm,hd,LAGUNA_RMS_EPS); memcpy(q,tmp,sizeof tmp); laguna_rope_half(q,rc,rs,rot); }
        uint16_t *kdst=kbase+(size_t)pos*kvstride, *vdst=vbase+(size_t)pos*kvstride;
        for (int h=0;h<LAGUNA_KV_HEADS;++h){ float *k=K+(size_t)c*kvstride+(size_t)h*hd; float tmp[LAGUNA_HEAD_DIM];
            laguna_rmsnorm(tmp,k,ly->k_norm,hd,LAGUNA_RMS_EPS); memcpy(k,tmp,sizeof tmp); laguna_rope_half(k,rc,rs,rot);
            for(int d=0;d<hd;++d) kdst[h*hd+d]=laguna_f32_to_bf16(k[d]);
            for(int d=0;d<hd;++d) vdst[h*hd+d]=laguna_f32_to_bf16(V[(size_t)c*kvstride+h*hd+d]); }
    }
    /* 2. flash query-block, parallel over (head, query tile).
     * Parallelising over heads alone leaves 47 threads running ceil(48/47)=2 rounds
     * with the second round 1/47 utilised -- measured, 24 threads were as fast as 47.
     * Tiling the queries gives nh*ntile tasks so every thread stays busy. */
    int ntile=(C+LAGUNA_QT-1)/LAGUNA_QT;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic,1)
#endif
    for (int h=0;h<nh;++h) {
      for (int t=0;t<ntile;++t) {
        int cbeg=t*LAGUNA_QT, cend=cbeg+LAGUNA_QT; if(cend>C)cend=C;
        int kvh=h/kv_groups;
        float *fm=sc->fm+(size_t)h*C, *fl=sc->fl+(size_t)h*C, *acc=sc->facc+(size_t)h*C*hd;
        for (int c=cbeg;c<cend;++c){ fm[c]=-INFINITY; fl[c]=0.0f; float *a=acc+(size_t)c*hd; for(int d=0;d<hd;++d)a[d]=0.0f; }
        /* prefix keys [0, pos0): every query attends all -> block, reuse KV across C queries */
        for (int kb=0; kb<pos0; kb+=LAGUNA_KB) {
            int bn=pos0-kb; if(bn>LAGUNA_KB)bn=LAGUNA_KB;
            for (int c=cbeg;c<cend;++c) {
                const float *q=Q+(size_t)c*nh*hd+(size_t)h*hd; float sb[LAGUNA_KB]; float bmax=-INFINITY;
                laguna_qk_run(sb, q, kbase+(size_t)kb*kvstride+(size_t)kvh*hd, kvstride, bn, scale, hd);
                for (int b=0;b<bn;++b) if(sb[b]>bmax) bmax=sb[b];
                float m_old=fm[c], m_new=m_old>bmax?m_old:bmax, corr=expf(m_old-m_new);
                float psum=laguna_exp_shift_sum(sb, bn, m_new);
                fl[c]=fl[c]*corr+psum; float *a=acc+(size_t)c*hd;
                laguna_av_run(a, sb, vbase+(size_t)kb*kvstride+(size_t)kvh*hd, kvstride, bn, corr, hd);
                fm[c]=m_new;
            }
        }
        /* Diagonal keys [pos0, pos0+c]: causal within the chunk.  Blocked exactly
         * like the prefix pass above -- one key at a time was the worst-shaped
         * loop in the file (a per-key svaddv inside laguna_qkdot, a full-accumulator
         * read-modify-write inside laguna_vaxpy, and two scalar expf), running at
         * 67 GFLOP/s against the prefix pass's 382.  Its cost is C^2/2 per head
         * regardless of context, so it was 27% of this kernel at pos0=2048 and 33%
         * at pos0=1024, though only ~1% at 64k.
         * Key order per query is still strictly increasing (kb ascends, and within
         * a block j ascends), which is what the online softmax requires. */
        for (int kb=0; kb<cend; kb+=LAGUNA_KB) {
            int kend = kb+LAGUNA_KB-1; if (kend > cend-1) kend = cend-1;
            int c0 = kb > cbeg ? kb : cbeg;      /* queries below kb have no key here */
            for (int c=c0;c<cend;++c) {
                int je = kend < c ? kend : c;    /* causal cutoff inside the block */
                int n = je - kb + 1;
                if (n <= 0) continue;
                const float *q=Q+(size_t)c*nh*hd+(size_t)h*hd; float *a=acc+(size_t)c*hd;
                const uint16_t *kblk = kbase+(size_t)(pos0+kb)*kvstride+(size_t)kvh*hd;
                const uint16_t *vblk = vbase+(size_t)(pos0+kb)*kvstride+(size_t)kvh*hd;
                float sb[LAGUNA_KB], bmax=-INFINITY;
                laguna_qk_run(sb, q, kblk, kvstride, n, scale, hd);
                for (int b=0;b<n;++b) if (sb[b]>bmax) bmax=sb[b];
                float m_old=fm[c], m_new=m_old>bmax?m_old:bmax, corr=expf(m_old-m_new);
                float psum=laguna_exp_shift_sum(sb, n, m_new);
                fl[c]=fl[c]*corr+psum;
                laguna_av_run(a, sb, vblk, kvstride, n, corr, hd);
                fm[c]=m_new;
            }
        }
        for (int c=cbeg;c<cend;++c){ float gate=laguna_softplus(G[(size_t)c*nh+h]); float s=gate/fl[c];
            float *a=acc+(size_t)c*hd, *o=AO+(size_t)c*nh*hd+(size_t)h*hd; for(int d=0;d<hd;++d)o[d]=a[d]*s; }
      }
    }
}

/* Query-block flash attention for a chunk of C tokens on a SLIDING layer.
 *
 * The per-token path re-reads a 512-key window for every query, and consecutive
 * queries' windows overlap by 511/512 -- so the same KV is pulled from memory ~C
 * times per chunk.  Blocking over key positions and sweeping the C queries inside
 * each block reads it once, exactly as attention_full_flash does for full layers.
 *
 * This requires writing the whole chunk's K/V before any query attends, which is
 * why the sliding ring is LAGUNA_SLIDING_CAP (768) rather than the 512-wide window:
 * at cap==window the chunk's own writes would clobber slots its earlier queries
 * still need.  Key ranges are per query c: [pos0+c-511, pos0+c] (clamped at 0),
 * so the union over the chunk is a band of C+511 positions. */
static void attention_slide_flash(const laguna_model *m, const laguna_layer *ly,
                                  laguna_scratch *sc, int layer, int pos0, int C, int nh,
                                  float *Q, float *K, float *V, float *G, float *AO) {
    int hd=LAGUNA_HEAD_DIM, kv_groups=nh/LAGUNA_KV_HEADS, kvstride=LAGUNA_KV_HEADS*hd;
    int rot=LAGUNA_ROPE_SLIDING_DIM, half=rot/2;
    int cap=m->kv_cap[layer], W=LAGUNA_SLIDING_WINDOW;
    uint16_t *kbase=m->kcache+m->kv_off[layer], *vbase=m->vcache+m->kv_off[layer];
    float scale=1.0f/sqrtf((float)hd);

    /* 1. qk-norm + rope + write KV for the whole chunk (ring slot pos%cap) */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int c=0;c<C;++c) {
        int pos=pos0+c;
        const float *rc=m->swa_cos+(size_t)pos*half, *rs=m->swa_sin+(size_t)pos*half;
        for (int h=0;h<nh;++h){ float *q=Q+(size_t)c*nh*hd+(size_t)h*hd; float tmp[LAGUNA_HEAD_DIM];
            laguna_rmsnorm(tmp,q,ly->q_norm,hd,LAGUNA_RMS_EPS); memcpy(q,tmp,sizeof tmp); laguna_rope_half(q,rc,rs,rot); }
        uint16_t *kdst=kbase+(size_t)(pos%cap)*kvstride, *vdst=vbase+(size_t)(pos%cap)*kvstride;
        for (int h=0;h<LAGUNA_KV_HEADS;++h){ float *k=K+(size_t)c*kvstride+(size_t)h*hd; float tmp[LAGUNA_HEAD_DIM];
            laguna_rmsnorm(tmp,k,ly->k_norm,hd,LAGUNA_RMS_EPS); memcpy(k,tmp,sizeof tmp); laguna_rope_half(k,rc,rs,rot);
            for(int d=0;d<hd;++d) kdst[h*hd+d]=laguna_f32_to_bf16(k[d]);
            for(int d=0;d<hd;++d) vdst[h*hd+d]=laguna_f32_to_bf16(V[(size_t)c*kvstride+h*hd+d]); }
    }

    /* 2. flash over key blocks, parallel over query heads */
    int glo = pos0-(W-1); if(glo<0) glo=0;
    int ghi = pos0+C-1;
    int ntile=(C+LAGUNA_QT-1)/LAGUNA_QT;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic,1)
#endif
    for (int h=0;h<nh;++h) {
      for (int t=0;t<ntile;++t) {
        int cbeg=t*LAGUNA_QT, cend=cbeg+LAGUNA_QT; if(cend>C)cend=C;
        int kvh=h/kv_groups;
        float *fm=sc->fm+(size_t)h*C, *fl=sc->fl+(size_t)h*C, *acc=sc->facc+(size_t)h*C*hd;
        for (int c=cbeg;c<cend;++c){ fm[c]=-INFINITY; fl[c]=0.0f; float *a=acc+(size_t)c*hd; for(int d=0;d<hd;++d)a[d]=0.0f; }
        for (int kb=glo; kb<=ghi; kb+=LAGUNA_KB) {
            int kend = kb+LAGUNA_KB-1; if(kend>ghi) kend=ghi;
            /* queries whose window intersects [kb,kend]: pos0+c-(W-1) <= kend and pos0+c >= kb */
            int c0 = kb-pos0;            if(c0<cbeg) c0=cbeg;
            int c1 = kend-pos0+(W-1);    if(c1>cend-1) c1=cend-1;
            for (int c=c0;c<=c1;++c) {
                int lo_c = pos0+c-(W-1); if(lo_c<0) lo_c=0;
                int hi_c = pos0+c;
                int js = kb>lo_c?kb:lo_c, je = kend<hi_c?kend:hi_c;
                if (js>je) continue;
                const float *q=Q+(size_t)c*nh*hd+(size_t)h*hd;
                float sb[LAGUNA_KB]; float bmax=-INFINITY; int n=je-js+1;
                /* the ring makes [js,je] at most two contiguous slot runs */
                int s0=js%cap, n1=cap-s0; if(n1>n) n1=n; int n2=n-n1;
                laguna_qk_run(sb,    q, kbase+(size_t)s0*kvstride+(size_t)kvh*hd, kvstride, n1, scale, hd);
                if(n2) laguna_qk_run(sb+n1, q, kbase+(size_t)kvh*hd, kvstride, n2, scale, hd);
                for (int b=0;b<n;++b) if(sb[b]>bmax) bmax=sb[b];
                float m_old=fm[c], m_new=m_old>bmax?m_old:bmax, corr=expf(m_old-m_new);
                float psum=laguna_exp_shift_sum(sb, n, m_new);
                fl[c]=fl[c]*corr+psum; float *a=acc+(size_t)c*hd;
                laguna_av_run(a, sb,    vbase+(size_t)s0*kvstride+(size_t)kvh*hd, kvstride, n1, corr, hd);
                if(n2) laguna_av_run(a, sb+n1, vbase+(size_t)kvh*hd, kvstride, n2, 1.0f, hd);
                fm[c]=m_new;
            }
        }
        for (int c=cbeg;c<cend;++c){ float gate=laguna_softplus(G[(size_t)c*nh+h]); float s=gate/fl[c];
            float *a=acc+(size_t)c*hd, *o=AO+(size_t)c*nh*hd+(size_t)h*hd; for(int d=0;d<hd;++d)o[d]=a[d]*s; }
      }
    }
}

/* dense/shared SwiGLU: out[hidden] = down( silu(gate(x)) * up(x) ). */
static void swiglu_lin(laguna_scratch *sc, const laguna_lin *gate_w, const laguna_lin *up_w,
                       const laguna_lin *down_w, const float *x, float *out, int inter) {
    float *ys[2]={sc->inter_a, sc->inter_b};
    const laguna_lin *Ws[2]={gate_w, up_w}; int rws[2]={inter, inter};
    laguna_lin_mv_multi(ys, Ws, rws, 2, x, LAGUNA_HIDDEN);   /* gate & up share x */
    for (int i=0;i<inter;++i) sc->inter_a[i]=laguna_silu(sc->inter_a[i])*sc->inter_b[i];
    laguna_lin_mv(out, down_w, sc->inter_a, LAGUNA_HIDDEN, inter);
}

/* One routed expert's SwiGLU into `out` (hidden). inter=1024. INT4 (production)
 * or bf16 (-DLAGUNA_BF16). */
static void expert_mv(laguna_scratch *sc, const laguna_expert *ex, const float *x, float *out) {
    int inter=LAGUNA_EXPERT_INTER;
#if defined(LAGUNA_FP8)
    if (g_fp8_exact) {
        laguna_matvec_fp8blk(sc->inter_a, ex->gate, ex->gs, x, inter, LAGUNA_HIDDEN);
        laguna_matvec_fp8blk(sc->inter_b, ex->up,   ex->us, x, inter, LAGUNA_HIDDEN);
        for (int i=0;i<inter;++i) sc->inter_a[i]=laguna_silu(sc->inter_a[i])*sc->inter_b[i];
        laguna_matvec_fp8blk(out, ex->down, ex->ds, sc->inter_a, LAGUNA_HIDDEN, inter);
    } else {
        float *ys[2]={sc->inter_a, sc->inter_b};
        const laguna_w8b *ws[2]={&ex->qg, &ex->qu};
        laguna_matvec_i8blk_multi(ys, ws, 2, x, inter, LAGUNA_HIDDEN);  /* gate & up share x */
        for (int i=0;i<inter;++i) sc->inter_a[i]=laguna_silu(sc->inter_a[i])*sc->inter_b[i];
        laguna_matvec_i8blk(out, &ex->qd, sc->inter_a, LAGUNA_HIDDEN, inter);
    }
#elif defined(LAGUNA_BF16)
    laguna_matvec_bf16(sc->inter_a, ex->gate, x, inter, LAGUNA_HIDDEN);
    laguna_matvec_bf16(sc->inter_b, ex->up,   x, inter, LAGUNA_HIDDEN);
    for (int i=0;i<inter;++i) sc->inter_a[i]=laguna_silu(sc->inter_a[i])*sc->inter_b[i];
    laguna_matvec_bf16(out, ex->down, sc->inter_a, LAGUNA_HIDDEN, inter);
#else
    laguna_matvec_i4g32(sc->inter_a, ex->gp, ex->gs, x, inter, LAGUNA_HIDDEN);
    laguna_matvec_i4g32(sc->inter_b, ex->up, ex->us, x, inter, LAGUNA_HIDDEN);
    for (int i=0;i<inter;++i) sc->inter_a[i]=laguna_silu(sc->inter_a[i])*sc->inter_b[i];
    laguna_matvec_i4g32(out, ex->dp, ex->ds, sc->inter_a, LAGUNA_HIDDEN, inter);
#endif
}

/* Async all-reduce interface: launch() posts the reduction (on a comm thread),
 * join() waits for it. Between them the caller runs independent compute (the
 * shared expert) so the per-MoE-layer allreduce overlaps it. NULL => single node
 * (no reduction needed). */
typedef struct {
    void (*launch)(void *ctx, float *buf, int count);
    void (*join)(void *ctx);
    void *ctx;
} laguna_async_ar;

/* Batched SwiGLU over C tokens (token-major).  out[C][hidden]. */
static void swiglu_lin_batch(laguna_scratch *sc, const laguna_lin *gate_w, const laguna_lin *up_w,
                             const laguna_lin *down_w, const float *X, float *out, int inter, int C) {
    laguna_lin_mm(sc->cia, gate_w, X, inter, LAGUNA_HIDDEN, C);
    laguna_lin_mm(sc->cib, up_w,   X, inter, LAGUNA_HIDDEN, C);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long i=0;i<(long)C*inter;++i) sc->cia[i]=laguna_silu(sc->cia[i])*sc->cib[i];
    laguna_lin_mm(out, down_w, sc->cia, LAGUNA_HIDDEN, inter, C);
}

static int g_dbg=0;
static double vnorm(const float*v,int n){ double s=0; for(int i=0;i<n;i++)s+=(double)v[i]*v[i]; return sqrt(s); }
/* lightweight phase profiling (enabled by the bench) */
double g_t_attn=0, g_t_mlp=0, g_t_norm=0; int g_prof=0;
/* chunked-prefill phase timers (rank-0, seconds) */
double g_c_qkv=0, g_c_attn=0, g_c_op=0, g_c_router=0, g_c_expert=0, g_c_shared=0, g_c_ar=0;
static double prof_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void forward_token(const laguna_model *m, laguna_scratch *sc, float *x, int pos,
                          laguna_async_ar *aar, int compute_logits) {
    for (int L=0; L<m->n_layers; ++L) {
        const laguna_layer *ly=&m->layers[L];
        double pt0=g_prof?prof_now():0;
        /* --- attention block --- */
        laguna_rmsnorm(sc->n1, x, ly->in_ln, LAGUNA_HIDDEN, LAGUNA_RMS_EPS);
        if(g_prof){double n=prof_now();g_t_norm+=n-pt0;pt0=n;}
        attention(m, ly, sc, L, pos);
        if(g_prof){double n=prof_now();g_t_attn+=n-pt0;pt0=n;}
        if(g_dbg&&compute_logits) logmsg("L%02d pos%d ||x_in||=%.3f ||attn||=%.3f\n",L,pos,vnorm(x,LAGUNA_HIDDEN),vnorm(sc->attn_out,LAGUNA_HIDDEN));
        for (int i=0;i<LAGUNA_HIDDEN;++i) x[i]+=sc->attn_out[i];
        double pm0=g_prof?prof_now():0;
        /* --- mlp block --- */
        laguna_rmsnorm(sc->n2, x, ly->post_ln, LAGUNA_HIDDEN, LAGUNA_RMS_EPS);
        if (!ly->is_moe) {
            swiglu_lin(sc, &ly->dense_gate, &ly->dense_up, &ly->dense_down, sc->n2,
                       sc->attn_out, LAGUNA_DENSE_INTER);
            for (int i=0;i<LAGUNA_HIDDEN;++i) x[i]+=sc->attn_out[i];
        } else {
            /* router: sigmoid + bias, top-10, normalized */
            laguna_lin_mv(sc->logits, &ly->router_w, sc->n2, LAGUNA_EXPERTS, LAGUNA_HIDDEN);
            int ids[LAGUNA_ACTIVE]; float rw[LAGUNA_ACTIVE];
            laguna_top10(sc->logits, ly->router_bias, ids, rw);
            for (int i=0;i<LAGUNA_HIDDEN;++i) sc->partial[i]=0.0f;
            for (int k=0;k<LAGUNA_ACTIVE;++k) {
                int e=ids[k]; const laguna_expert *ex=&ly->experts[e];
                if (!ex->present) continue;
                expert_mv(sc, ex, sc->n2, sc->attn_out);   /* reuse attn_out as expert out */
                float w=rw[k];
                for (int i=0;i<LAGUNA_HIDDEN;++i) sc->partial[i]+=w*sc->attn_out[i];
            }
            /* Launch the routed-partial allreduce on the comm thread, then compute
             * the (independent) shared expert on the OMP team so they overlap. */
            if (aar && aar->launch) aar->launch(aar->ctx, sc->partial, LAGUNA_HIDDEN);
            swiglu_lin(sc, &ly->shared_gate, &ly->shared_up, &ly->shared_down, sc->n2,
                       sc->shared, LAGUNA_SHARED_INTER);
            if (aar && aar->join) aar->join(aar->ctx);
            if(g_dbg&&compute_logits) logmsg("L%02d pos%d ||shared||=%.3f ||routed||=%.3f e0=%d\n",L,pos,vnorm(sc->shared,LAGUNA_HIDDEN),vnorm(sc->partial,LAGUNA_HIDDEN),ids[0]);
            for (int i=0;i<LAGUNA_HIDDEN;++i) x[i]+=sc->shared[i]+LAGUNA_ROUTED_SCALE*sc->partial[i];
        }
        if(g_prof) g_t_mlp+=prof_now()-pm0;
    }
    if (compute_logits) {
        laguna_rmsnorm(sc->n1, x, m->final_norm, LAGUNA_HIDDEN, LAGUNA_RMS_EPS);
        laguna_lin_mv(sc->logits, &m->lm_head, sc->n1, LAGUNA_VOCAB, LAGUNA_HIDDEN);
    }
}

/* Chunked prefill: process C tokens (positions pos0..pos0+C-1) in one pass.  The
 * weight-heavy matvecs are BATCHED (weight bandwidth amortized ~8x) and the routed
 * allreduce is one per chunk (C-token payload) instead of per token.  KV-write +
 * attention stay per token (the sliding ring needs write-then-attend order).
 * Does not compute logits (prefill builds KV; the last token's logits are produced
 * by a final per-token forward). */
static void forward_prefill_chunk(const laguna_model *m, laguna_scratch *sc, float *X,
                                  int pos0, int C, laguna_async_ar *aar) {
    int H=LAGUNA_HIDDEN, hd=LAGUNA_HEAD_DIM;
    for (int L=0; L<m->n_layers; ++L) {
        const laguna_layer *ly=&m->layers[L];
        int nh=ly->num_heads;
        double _t=prof_now();
        /* attention */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int c=0;c<C;++c) laguna_rmsnorm(sc->cn1+(size_t)c*H, X+(size_t)c*H, ly->in_ln, H, LAGUNA_RMS_EPS);
        laguna_lin_mm(sc->cq, &ly->q_proj, sc->cn1, nh*hd, H, C);
        laguna_lin_mm(sc->ck, &ly->k_proj, sc->cn1, LAGUNA_KV_HEADS*hd, H, C);
        laguna_lin_mm(sc->cv, &ly->v_proj, sc->cn1, LAGUNA_KV_HEADS*hd, H, C);
        laguna_lin_mm(sc->cg, &ly->g_proj, sc->cn1, nh, H, C);
        g_c_qkv+=prof_now()-_t; _t=prof_now();
        if (!ly->is_sliding) {
            /* full attention: query-block flash (amortizes KV bandwidth across the chunk) */
            attention_full_flash(m, ly, sc, L, pos0, C, nh, sc->cq, sc->ck, sc->cv, sc->cg, sc->cao);
        } else if (C > 1) {
            /* sliding: query-block flash (KV read once per block, not once per query) */
            attention_slide_flash(m, ly, sc, L, pos0, C, nh, sc->cq, sc->ck, sc->cv, sc->cg, sc->cao);
        } else {
            /* single token: per-token write-then-attend (ring, no clobber) */
            for (int c=0;c<C;++c)
                attention_core(m, ly, sc, L, pos0+c, nh,
                               sc->cq+(size_t)c*nh*hd, sc->ck+(size_t)c*LAGUNA_KV_HEADS*hd,
                               sc->cv+(size_t)c*LAGUNA_KV_HEADS*hd, sc->cg+(size_t)c*nh,
                               sc->cao+(size_t)c*nh*hd);
        }
        g_c_attn+=prof_now()-_t; _t=prof_now();
        laguna_lin_mm(sc->cattn, &ly->o_proj, sc->cao, H, nh*hd, C);
        for (long i=0;i<(long)C*H;++i) X[i]+=sc->cattn[i];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int c=0;c<C;++c) laguna_rmsnorm(sc->cn2+(size_t)c*H, X+(size_t)c*H, ly->post_ln, H, LAGUNA_RMS_EPS);
        g_c_op+=prof_now()-_t;
        /* mlp */
        if (!ly->is_moe) {
            _t=prof_now();
            swiglu_lin_batch(sc, &ly->dense_gate, &ly->dense_up, &ly->dense_down, sc->cn2, sc->cattn, LAGUNA_DENSE_INTER, C);
            for (long i=0;i<(long)C*H;++i) X[i]+=sc->cattn[i];
            g_c_shared+=prof_now()-_t;
        } else {
            _t=prof_now();
            for (long i=0;i<(long)C*H;++i) sc->cpart[i]=0.0f;
            /* route all C tokens: one batched router GEMM (was C matvecs, i.e. C
             * OpenMP fork/joins per MoE layer), then top-10 per token. */
            laguna_lin_mm(sc->crouter, &ly->router_w, sc->cn2, LAGUNA_EXPERTS, H, C);
            for (int c=0;c<C;++c)
                laguna_top10(sc->crouter+(size_t)c*LAGUNA_EXPERTS, ly->router_bias,
                             sc->rids+(size_t)c*LAGUNA_ACTIVE, sc->rrw+(size_t)c*LAGUNA_ACTIVE);
            double _te=prof_now();
#if defined(LAGUNA_FP8)
            /* BATCHED experts: for each owned expert, gather its tokens and run one
             * (dequant-once) fp8 GEMM -> the fp8 gather is amortized across tokens. */
            int inter=LAGUNA_EXPERT_INTER; int tok[LAGUNA_PCHUNK]; float wgt[LAGUNA_PCHUNK];
            for (int e=m->ep_rank; e<LAGUNA_EXPERTS; e+=m->ep_size) {
                const laguna_expert *ex=&ly->experts[e]; if(!ex->present) continue;
                int ne=0;
                for (int c=0;c<C;++c){ const int *id=sc->rids+(size_t)c*LAGUNA_ACTIVE;
                    for (int k=0;k<LAGUNA_ACTIVE;++k) if(id[k]==e){ tok[ne]=c; wgt[ne]=sc->rrw[(size_t)c*LAGUNA_ACTIVE+k]; ne++; break; } }
                if(ne==0) continue;
                for (int i=0;i<ne;++i) memcpy(sc->xe+(size_t)i*H, sc->cn2+(size_t)tok[i]*H, (size_t)H*sizeof(float));
                if (g_fp8_exact) {
                    laguna_matmat_fp8blk(sc->cia, ex->gate, ex->gs, sc->xe, inter, H, ne);
                    laguna_matmat_fp8blk(sc->cib, ex->up,   ex->us, sc->xe, inter, H, ne);
                    for (long i=0;i<(long)ne*inter;++i) sc->cia[i]=laguna_silu(sc->cia[i])*sc->cib[i];
                    laguna_matmat_fp8blk(sc->ye, ex->down, ex->ds, sc->cia, H, inter, ne);
                } else {
                    laguna_matmat_i8blk(sc->cia, &ex->qg, sc->xe, inter, H, ne);
                    laguna_matmat_i8blk(sc->cib, &ex->qu, sc->xe, inter, H, ne);
                    for (long i=0;i<(long)ne*inter;++i) sc->cia[i]=laguna_silu(sc->cia[i])*sc->cib[i];
                    laguna_matmat_i8blk(sc->ye, &ex->qd, sc->cia, H, inter, ne);
                }
                for (int i=0;i<ne;++i){ float w=wgt[i]; float *pc=sc->cpart+(size_t)tok[i]*H; const float *ye=sc->ye+(size_t)i*H;
                    for (int d=0;d<H;++d) pc[d]+=w*ye[d]; }
            }
#else
            for (int c=0;c<C;++c) {
                const float *n2=sc->cn2+(size_t)c*H; float *pc=sc->cpart+(size_t)c*H;
                const int *id=sc->rids+(size_t)c*LAGUNA_ACTIVE; const float *rw=sc->rrw+(size_t)c*LAGUNA_ACTIVE;
                for (int k=0;k<LAGUNA_ACTIVE;++k) {
                    const laguna_expert *ex=&ly->experts[id[k]]; if(!ex->present) continue;
                    expert_mv(sc, ex, n2, sc->attn_out);
                    for (int i=0;i<H;++i) pc[i]+=rw[k]*sc->attn_out[i];
                }
            }
#endif
            g_c_expert+=prof_now()-_te;
            g_c_router+=prof_now()-_t; _t=prof_now();
            if (aar && aar->launch) aar->launch(aar->ctx, sc->cpart, C*H);   /* one AR for C tokens */
            swiglu_lin_batch(sc, &ly->shared_gate, &ly->shared_up, &ly->shared_down, sc->cn2, sc->cshared, LAGUNA_SHARED_INTER, C);
            if (aar && aar->join) aar->join(aar->ctx);
            for (long i=0;i<(long)C*H;++i) X[i]+=sc->cshared[i]+LAGUNA_ROUTED_SCALE*sc->cpart[i];
            g_c_shared+=prof_now()-_t;
        }
    }
}

static int argmax(const float *v, int n) {
    int best=0; float bv=v[0];
    for (int i=1;i<n;++i) if(v[i]>bv){bv=v[i];best=i;}
    return best;
}

/* ---- sampling ----
 * The checkpoint's generation_config.json asks for do_sample=true, temperature=1.0,
 * top_k=20, top_p=1.0, min_p=0.0.  Greedy stays the default here because it is
 * deterministic (and is what every benchmark in fp8-optimization.md used); pass
 * --sample to follow the checkpoint's own configuration.  Warper order matches
 * HuggingFace: temperature -> top_k -> top_p -> min_p. */
typedef struct {
    int   do_sample;
    float temp, top_p, min_p;
    int   top_k;
    uint64_t rng;
} laguna_sampler;

static inline double laguna_rng_next(uint64_t *s) {   /* splitmix64 -> [0,1) */
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) * 0x1.0p-53;
}

/* Candidate cap when top_k is disabled but a distribution warper still needs a
 * ranked list.  A full 100352-wide sort per token is far too slow, and no
 * plausible configuration reaches this far down the tail. */
#define LAGUNA_TOPK_CAP 1024

static int laguna_sample(const float *logits, int n, laguna_sampler *sp) {
    if (!sp->do_sample || sp->temp <= 0.0f) return argmax(logits, n);

    int need_rank = (sp->top_k > 0) || (sp->top_p < 1.0f) || (sp->min_p > 0.0f);
    float inv_t = 1.0f / sp->temp;

    if (!need_rank) {                     /* plain multinomial over the full vocab */
        float mx = logits[argmax(logits,n)];
        double sum = 0.0;
        for (int i=0;i<n;++i) sum += exp((double)(logits[i]-mx)*inv_t);
        double r = laguna_rng_next(&sp->rng) * sum, c = 0.0;
        for (int i=0;i<n;++i) { c += exp((double)(logits[i]-mx)*inv_t); if (c >= r) return i; }
        return n-1;
    }

    int k = sp->top_k > 0 ? sp->top_k : LAGUNA_TOPK_CAP;
    if (k > n) k = n;
    if (k > LAGUNA_TOPK_CAP) k = LAGUNA_TOPK_CAP;

    /* top-k by insertion into a descending list (k is small; one pass over vocab) */
    static int   idx[LAGUNA_TOPK_CAP];
    static float val[LAGUNA_TOPK_CAP];
    int cnt = 0;
    for (int i=0;i<n;++i) {
        float v = logits[i];
        if (cnt == k && v <= val[cnt-1]) continue;
        int j = cnt < k ? cnt : k-1;
        while (j > 0 && val[j-1] < v) { val[j]=val[j-1]; idx[j]=idx[j-1]; --j; }
        val[j]=v; idx[j]=i;
        if (cnt < k) ++cnt;
    }

    /* softmax over the candidates (temperature applied to logits) */
    double p[LAGUNA_TOPK_CAP], sum = 0.0;
    for (int i=0;i<cnt;++i) { p[i] = exp((double)(val[i]-val[0])*inv_t); sum += p[i]; }
    for (int i=0;i<cnt;++i) p[i] /= sum;

    int m = cnt;
    if (sp->top_p < 1.0f) {               /* smallest prefix with cumulative >= top_p */
        double c = 0.0; m = cnt;
        for (int i=0;i<cnt;++i) { c += p[i]; if (c >= (double)sp->top_p) { m = i+1; break; } }
        if (m < 1) m = 1;
    }
    if (sp->min_p > 0.0f) {               /* drop p < min_p * p_max */
        double thr = (double)sp->min_p * p[0];
        int mm = m; for (int i=0;i<m;++i) if (p[i] < thr) { mm = i; break; }
        if (mm < 1) mm = 1;
        m = mm;
    }

    double tot = 0.0; for (int i=0;i<m;++i) tot += p[i];
    double r = laguna_rng_next(&sp->rng) * tot, c = 0.0;
    for (int i=0;i<m;++i) { c += p[i]; if (c >= r) return idx[i]; }
    return idx[m-1];
}
static void embed_token(const laguna_model *m, int tok, float *x) {
    const uint16_t *row=m->embed+(size_t)tok*LAGUNA_HIDDEN;
    for (int i=0;i<LAGUNA_HIDDEN;++i) x[i]=laguna_bf16_to_f32(row[i]);
}
static int count_nan(const float *v, int n){ int c=0; for(int i=0;i<n;++i) if(!isfinite(v[i]))c++; return c; }

/* ============================ generate driver ============================ */
static int load_ids(const char *path, int **out) {
    FILE *f=fopen(path,"r"); if(!f){perror(path);return -1;}
    int cap=1024,n=0,*v=malloc((size_t)cap*sizeof(int)),t;
    while (fscanf(f,"%d",&t)==1){ if(n==cap){cap*=2;v=realloc(v,(size_t)cap*sizeof(int));} v[n++]=t; }
    fclose(f); *out=v; return n;
}

static void usage(const char *n){
    fprintf(stderr,
      "usage: %s --self-test | --describe | --check-stage DIR [LAYERS] | --probe-stage DIR\n"
      "       %s --generate --ids FILE --max-new N [--maxpos P] [--layers L]\n"
      "                     [--stage-dir DIR] [--gen-out FILE]\n", n, n);
}

#ifndef LAGUNA_BENCH
#include "laguna_tofu.inc"   /* uTofu bootstrap + allreduce glue (single EP group) */

int main(int argc, char **argv) {
    if (argc==2 && !strcmp(argv[1],"--self-test")) {
        int rc=test_i4()|test_fht()|test_route()|test_i4_matvec()|test_i8_matmat();
#if defined(LAGUNA_FP8)
        rc|=test_fp8_i8blk();
#endif
        if(!rc)puts("Laguna S21 ABI self-test: PASS"); return rc;
    }
    if (argc==2 && !strcmp(argv[1],"--describe")) {
        puts("Laguna S21: 48L H=3072 GQA=8x128, full=48h(YaRN rot64)/sliding=72h(rope rot128,win512), 256 experts top-10 INT4 g32, shared+dense0"); return 0;
    }
    if (argc>=3 && argc<=4 && !strcmp(argv[1],"--check-stage")) return stage_check(argv[2],argc==4?atoi(argv[3]):0);
    if (argc==3 && !strcmp(argv[1],"--probe-stage")) return probe_stage(argv[2]);
    if (argc>=2 && !strcmp(argv[1],"--generate")) return run_generate(argc,argv);
    if (argc>=2 && !strcmp(argv[1],"--serve"))    return run_serve(argc,argv);
    usage(argv[0]); return 2;
}
#endif /* LAGUNA_BENCH */
