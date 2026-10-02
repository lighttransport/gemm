/* Stage native GGUF KDA projections for every linear-attention layer into one
 * rank-local image.  Head-owned rows of Q/K/V, f_b/g_b and beta are sliced by
 * the production 12-way head partition; f_a/g_a are replicated.  The output
 * projection is column-sliced when the type's block divides one head (Q8_0),
 * otherwise replicated (256-value K-quant blocks do not align to 5/6 heads).
 * Quantized blocks are preserved byte-for-byte. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

#include <mpi.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef GLM53F_PP_KDA_STAGE
#include "glm53f_parallel.h"
#include "glm53f_pp_source.h"
#include "glm53f_pp_blob.h"
static glm53f_parallel_config pp_config;
static glm53f_parallel_map pp_map;
static uint64_t pp_hash;
#define KDA_RANKS 4
#else
#define KDA_RANKS 12
#endif
enum { RANKS = KDA_RANKS, LAYERS = 45, HIDDEN = 4096, HEADS = 64, DIM = 128,
       QKV = HEADS * DIM };

typedef struct { int fd; uint64_t base; const gguf_tensor_info *info; } tensor_ref;

static void die(int rank, const char *message, const char *tensor) {
    fprintf(stderr, "rank=%d glm53f_q2_kda_stage: %s%s%s: %s\n", rank, message,
            tensor ? " " : "", tensor ? tensor : "",
            errno ? strerror(errno) : "contract failure");
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static tensor_ref find_tensor(const gguf_context *g, const char *name) {
    tensor_ref t = {-1, 0, NULL};
    for (uint64_t i = 0; i < g->n_tensors; ++i)
        if (g->tensors[i].name.str && !strcmp(g->tensors[i].name.str, name)) {
            t.info = &g->tensors[i];
            t.fd = g->tensor_fds ? g->tensor_fds[i] : g->fd;
            t.base = g->tensor_file_offsets ? g->tensor_file_offsets[i] :
                     g->data_offset + g->tensors[i].offset;
            break;
        }
    return t;
}

static int supported(uint32_t type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

static size_t row_bytes(uint32_t type, int columns) {
    if (type >= GGML_TYPE_COUNT || ggml_type_info[type].block_size <= 0 ||
        columns % ggml_type_info[type].block_size) return 0;
    return (size_t)(columns / ggml_type_info[type].block_size) *
           ggml_type_info[type].type_size;
}

static int read_exact(const tensor_ref *t, uint64_t rel, void *dst, size_t n) {
    unsigned char *p = dst;
#ifdef GLM53F_PP_KDA_STAGE
    const uint64_t original_rel=rel;const size_t original_n=n;
#endif
    while (n) {
        ssize_t z = pread(t->fd, p, n, (off_t)(t->base + rel));
        if (z < 0) { if (errno == EINTR) continue; return -1; }
        if (!z) { errno = EIO; return -1; }
        p += z; rel += (uint64_t)z; n -= (size_t)z;
    }
#ifdef GLM53F_PP_KDA_STAGE
    (void)posix_fadvise(t->fd, (off_t)(t->base+original_rel), original_n, POSIX_FADV_DONTNEED);
#endif
    return 0;
}

static int write_all(int fd, const void *src, size_t n) {
    const unsigned char *p = src;
#ifdef GLM53F_PP_KDA_STAGE
    for(size_t i=0;i<n;i++){pp_hash^=p[i];pp_hash*=UINT64_C(1099511628211);}
#endif
    while (n) {
        ssize_t z = write(fd, p, n);
        if (z < 0) { if (errno == EINTR) continue; return -1; }
        if (!z) { errno = EIO; return -1; }
        p += z; n -= (size_t)z;
    }
    return 0;
}

/* Model identity: a stage image is reusable only for the same first shard. */
static void model_identity(const char *path, char *out, size_t n) {
    struct stat st;
    if (stat(path, &st)) st.st_size = 0;
    snprintf(out, n, "model_bytes=%lld model=%s", (long long)st.st_size, path);
}

static void stage_header(char *out,size_t size,int rank,const char *identity){
#ifdef GLM53F_PP_KDA_STAGE
    snprintf(out,size,"# GLM53F_PP_KDA_V1 layout=pp3-tp4 world_rank=%d stage=%d tp_rank=%d tp_size=4 cuts=%d,%d layers=%d:%d %s\n",rank,pp_map.stage,pp_map.tp_rank,pp_config.cuts[0],pp_config.cuts[1],pp_map.first_layer,pp_map.end_layer,identity);
#else
    snprintf(out,size,"# GLM53F_Q2_KDA_V3 rank=%d ranks=12 %s\n",rank,identity);
#endif
}
#ifdef GLM53F_PP_KDA_STAGE
static void payload(FILE *m,uint64_t off,uint64_t bytes,uint64_t hash,const char *name,int r0,int rows,int c0,int cols){
    fprintf(m,"# PAYLOAD offset=%" PRIu64 " bytes=%" PRIu64 " fnv1a=%016" PRIx64 " source=%s rows=%d:%d columns=%d:%d\n",off,bytes,hash,name,r0,r0+rows,c0,c0+cols);
}
#endif
static int complete(const char *manifest, const char *blob, int rank,
                    const char *identity) {
    FILE *f = fopen(manifest, "r");
    struct stat st;
    char line[8192], want[8192];
    int header = 0;
    uint64_t bytes = UINT64_MAX;
#ifdef GLM53F_PP_KDA_STAGE
    unsigned long long hash=0;int have_hash=0;
#endif
    if (!f || stat(blob, &st)) { if (f) fclose(f); return 0; }
    stage_header(want,sizeof(want),rank,identity);
    while (fgets(line, sizeof line, f)) {
        unsigned long long z;
        if (!strcmp(line, want)) header = 1;
        if (sscanf(line, "# COMPLETE bytes=%llu", &z) == 1) bytes = z;
#ifdef GLM53F_PP_KDA_STAGE
        if(sscanf(line,"# COMPLETE bytes=%llu fnv1a=%llx",&z,&hash)==2)have_hash=1;
#endif
    }
    fclose(f);
    return header && bytes == (uint64_t)st.st_size
#ifdef GLM53F_PP_KDA_STAGE
        && have_hash && !glm53f_pp_blob_verify(blob,bytes,hash)
#endif
        ;
}

/* Rows [row0, row0 + rows) of a [src_rows x columns] matrix. */
static void stage_rows(int rank, int fd, FILE *m, uint64_t *off,
                       const gguf_context *g, const char *name, int src_rows,
                       int columns, int row0, int rows, void *buf) {
    tensor_ref t = find_tensor(g, name);
    if (!t.info || !supported(t.info->type) || t.info->n_dims != 2 ||
        t.info->dims[0] != (uint64_t)columns ||
        t.info->dims[1] != (uint64_t)src_rows) die(rank, "tensor contract", name);
    size_t rb = row_bytes(t.info->type, columns), bytes = (size_t)rows * rb;
#ifdef GLM53F_PP_KDA_STAGE
    uint64_t tensor_hash=UINT64_C(1469598103934665603);
    if(!rb||row0<0||rows<1||row0>src_rows-rows)die(rank,"row range",name);
    for(int start=0;start<rows;start+=64){int count=rows-start;if(count>64)count=64;
        if(read_exact(&t,(uint64_t)(row0+start)*rb,buf,(size_t)count*rb))die(rank,"tensor read",name);
        const unsigned char *p=buf;for(size_t j=0;j<(size_t)count*rb;j++){tensor_hash^=p[j];tensor_hash*=UINT64_C(1099511628211);}
        if(write_all(fd,buf,(size_t)count*rb))die(rank,"tensor write",name);
    }
#else
    if (!rb || read_exact(&t, (uint64_t)row0 * rb, buf, bytes) ||
        write_all(fd, buf, bytes)) die(rank, "tensor stage", name);
#endif
    fprintf(m, "%" PRIu64 " %u %s %d %d %s\n", *off, t.info->type,
            ggml_type_name(t.info->type), rows, columns, name);
#ifdef GLM53F_PP_KDA_STAGE
    payload(m,*off,bytes,tensor_hash,name,row0,rows,0,columns);
#endif
    *off += bytes;
}

/* Output projection [HIDDEN rows x QKV columns]: local head columns when the
 * block aligns to a head, else the full replicated matrix. */
static void stage_output(int rank, int fd, FILE *m, uint64_t *off,
                         const gguf_context *g, const char *name, int col0,
                         int columns, void *full, void *local) {
    tensor_ref t = find_tensor(g, name);
    if (!t.info || !supported(t.info->type) || t.info->n_dims != 2 ||
        t.info->dims[0] != QKV || t.info->dims[1] != HIDDEN)
        die(rank, "output contract", name);
    const int bs = ggml_type_info[t.info->type].block_size;
    #ifdef GLM53F_PP_KDA_STAGE
    if(col0%bs||columns%bs)die(rank,"unaligned PP head slice",name);
    uint64_t tensor_hash=UINT64_C(1469598103934665603);
#else
    if (DIM % bs) { col0 = 0; columns = QKV; }
#endif
    size_t fr = row_bytes(t.info->type, QKV), lr = row_bytes(t.info->type, columns);
    size_t byte0 = (size_t)(col0 / bs) * ggml_type_info[t.info->type].type_size;
    for (int r0 = 0; r0 < HIDDEN; r0 += 64) {
        if (read_exact(&t, (uint64_t)r0 * fr, full, 64 * fr))
            die(rank, "output read", name);
        for (int r = 0; r < 64; ++r)
            memcpy((unsigned char *)local + (size_t)r * lr,
                   (unsigned char *)full + (size_t)r * fr + byte0, lr);
#ifdef GLM53F_PP_KDA_STAGE
        const unsigned char *p=local;for(size_t j=0;j<64*lr;j++){tensor_hash^=p[j];tensor_hash*=UINT64_C(1099511628211);}
#endif
        if (write_all(fd, local, 64 * lr)) die(rank, "output write", name);
    }
    fprintf(m, "%" PRIu64 " %u %s %d %d %s\n", *off, t.info->type,
            ggml_type_name(t.info->type), HIDDEN, columns, name);
#ifdef GLM53F_PP_KDA_STAGE
    payload(m,*off,(uint64_t)HIDDEN*lr,tensor_hash,name,0,HIDDEN,col0,columns);
#endif
    *off += (uint64_t)HIDDEN * lr;
}

int main(int argc, char **argv) {
    (void)gguf_type_name;
    int rank, nr, fd = -1;
    uint64_t off = 0;
    char blob[4096], manifest[4096], bt[4096], mt[4096], identity[4096], name[128];
    gguf_context *g = NULL;
    FILE *m = NULL;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nr);
    int first=0,end=LAYERS,tp_rank=rank;
#ifdef GLM53F_PP_KDA_STAGE
    pp_config=glm53f_parallel_default();pp_config.layout=GLM53F_PP3_TP4;
    for(int i=3;i<argc;i++)if(glm53f_parallel_option(&pp_config,argc,argv,&i)!=1)die(rank,"pipeline option",NULL);
    if(pp_config.layout!=GLM53F_PP3_TP4||glm53f_parallel_map_rank(&pp_config,rank,nr,&pp_map))die(rank,"PP layout",NULL);
    int low[2],high[2];MPI_Allreduce(pp_config.cuts,low,2,MPI_INT,MPI_MIN,MPI_COMM_WORLD);MPI_Allreduce(pp_config.cuts,high,2,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(memcmp(low,high,sizeof(low)))die(rank,"inconsistent cuts",NULL);
    first=pp_map.first_layer;end=pp_map.end_layer;tp_rank=pp_map.tp_rank;
    if(argc<3||nr!=12||strncmp(argv[2],"/local/",7))die(rank,"PP usage",NULL);
#else
    if (argc != 3 || nr != RANKS || strncmp(argv[2], "/local/", 7))
        die(rank, "usage: GGUF /local/STAGE", NULL);
#endif
    if (mkdir(argv[2], 0755) && errno != EEXIST) die(rank, "mkdir", argv[2]);
    snprintf(blob, sizeof blob, "%s/rank%02d.blob", argv[2], rank);
    snprintf(manifest, sizeof manifest, "%s/rank%02d.manifest", argv[2], rank);
    model_identity(argv[1], identity, sizeof identity);
#ifdef GLM53F_PP_KDA_STAGE
    uint64_t stamp=0;
    if(!(g=gguf_open_multi(argv[1],3))||g->n_tensors!=1412||glm53f_pp_source_stamp(g,argv[1],&stamp))die(rank,"GGUF source identity",argv[1]);
    snprintf(identity,sizeof(identity),"source_metadata_fnv1a=%016" PRIx64,stamp);
    pp_hash=UINT64_C(1469598103934665603);
#endif
    if (complete(manifest, blob, rank, identity)) {
        printf("SENTINEL glm53f_q2_kda_stage=REUSE rank=%d\n", rank);
        gguf_close(g);MPI_Finalize();
        return 0;
    }
    snprintf(bt, sizeof bt, "%s/.rank%02d.blob.%ld", argv[2], rank, (long)getpid());
    snprintf(mt, sizeof mt, "%s/.rank%02d.manifest.%ld", argv[2], rank, (long)getpid());
    if ((!g && !(g = gguf_open_multi(argv[1], 3))) || g->n_tensors != 1412)
        die(rank, "GGUF metadata", argv[1]);
    if ((fd = open(bt, O_CREAT | O_EXCL | O_WRONLY, 0644)) < 0 || !(m = fopen(mt, "wx")))
        die(rank, "create", bt);
    char header[8192];stage_header(header,sizeof(header),rank,identity);fputs(header,m);
    const int h0 = HEADS * tp_rank / RANKS, hn = HEADS * (tp_rank + 1) / RANKS - h0;
    const int qd = hn * DIM;
    /* Worst case is Q8_0 (34 bytes / 32 values); K-quants are smaller. */
    const size_t q8 = row_bytes(GGML_TYPE_Q8_0, HIDDEN);
    #ifdef GLM53F_PP_KDA_STAGE
    void *buf=malloc(64*q8);
#else
    void *buf = malloc((size_t)QKV * q8);
#endif
    void *full = malloc(64 * row_bytes(GGML_TYPE_Q8_0, QKV));
    void *local = malloc(64 * row_bytes(GGML_TYPE_Q8_0, QKV));
    if (!buf || !full || !local) die(rank, "scratch", NULL);
    int staged = 0;
    for (int layer = first; layer < end; ++layer) {
        if (layer % 4 == 3) continue;   /* MLA/DSA sparse layers. */
        static const char *const qkv[3] = {"attn_q", "attn_k", "attn_v"};
        for (int i = 0; i < 3; ++i) {
            snprintf(name, sizeof name, "blk.%d.%s.weight", layer, qkv[i]);
            stage_rows(rank, fd, m, &off, g, name, QKV, HIDDEN, h0 * DIM, qd, buf);
        }
        snprintf(name, sizeof name, "blk.%d.ssm_f_a.weight", layer);
        stage_rows(rank, fd, m, &off, g, name, DIM, HIDDEN, 0, DIM, buf);
        snprintf(name, sizeof name, "blk.%d.ssm_f_b.weight", layer);
        stage_rows(rank, fd, m, &off, g, name, QKV, DIM, h0 * DIM, qd, buf);
        snprintf(name, sizeof name, "blk.%d.ssm_beta.weight", layer);
        stage_rows(rank, fd, m, &off, g, name, HEADS, HIDDEN, h0, hn, buf);
        snprintf(name, sizeof name, "blk.%d.ssm_g_a.weight", layer);
        stage_rows(rank, fd, m, &off, g, name, DIM, HIDDEN, 0, DIM, buf);
        snprintf(name, sizeof name, "blk.%d.ssm_g_b.weight", layer);
        stage_rows(rank, fd, m, &off, g, name, QKV, DIM, h0 * DIM, qd, buf);
        snprintf(name, sizeof name, "blk.%d.attn_output.weight", layer);
        stage_output(rank, fd, m, &off, g, name, h0 * DIM, qd, full, local);
        ++staged;
    }
    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        int fdi = g->tensor_fds ? g->tensor_fds[i] : g->fd;
        (void)posix_fadvise(fdi, 0, 0, POSIX_FADV_DONTNEED);
    }
    #ifdef GLM53F_PP_KDA_STAGE
    fprintf(m,"# COMPLETE bytes=%" PRIu64 " fnv1a=%016" PRIx64 "\n",off,pp_hash);
#else
    fprintf(m, "# COMPLETE bytes=%" PRIu64 "\n", off);
#endif
    if (fflush(m) || fsync(fileno(m)) || fsync(fd) || fclose(m) || close(fd) ||
        rename(bt, blob) || rename(mt, manifest)) die(rank, "publish", blob);
    printf("SENTINEL glm53f_q2_kda_stage=OK rank=%d layers=%d bytes=%" PRIu64 "\n",
           rank, staged, off);
    free(local); free(full); free(buf);
    gguf_close(g);
    MPI_Finalize();
    return 0;
}
