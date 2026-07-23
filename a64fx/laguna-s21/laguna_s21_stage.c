/* Safe-ish, rank-local Laguna S 2.1 INT4 safetensors stager.
 * It never creates a whole-model copy: experts are EP-owned, dense tensors are
 * retained for now so the loader can apply the exact TP slice after validation. */
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

#define ALIGN 256u
static int envi(const char *k, int d) { const char *v=getenv(k); return v&&*v ? atoi(v) : d; }
static int rank_env(void) { const char *ks[]={"LAGUNA_EP_RANK","PMIX_RANK","OMPI_COMM_WORLD_RANK","PMI_RANK",NULL}; for(int i=0;ks[i];++i){const char*v=getenv(ks[i]);if(v&&*v)return atoi(v);} return 0; }
static long expert_id(const char *s) { const char *p=strstr(s,".mlp.experts."); if(!p) return -1; p+=13; return (*p>='0'&&*p<='9') ? strtol(p,NULL,10) : -1; }
static long layer_id(const char *s) { const char *p=strstr(s,"model.layers."); if(!p) return -1; p+=13; return (*p>='0'&&*p<='9') ? strtol(p,NULL,10) : -1; }
static int write_all(int fd,const void *p,size_t n) { const unsigned char *b=p; while(n){ ssize_t w=write(fd,b,n>(size_t)(1u<<30)?(size_t)(1u<<30):n); if(w<0){if(errno==EINTR)continue;return -1;} b+=w;n-=(size_t)w;} return 0; }
static int mkdir_p(const char *path) {
    char tmp[1200]; size_t n = strlen(path);
    if (n == 0 || n >= sizeof tmp) return -1;
    memcpy(tmp, path, n + 1);
    for (char *p = tmp + 1; *p; ++p) if (*p == '/') {
        *p = 0;
        if (mkdir(tmp, 0755) && errno != EEXIST) return -1;
        *p = '/';
    }
    return mkdir(tmp, 0755) && errno != EEXIST ? -1 : 0;
}
static double now_sec(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void drop_source(int fd, const void *p, size_t n) {
    long page = sysconf(_SC_PAGESIZE);
    uintptr_t lo = (uintptr_t)p & ~((uintptr_t)page - 1u);
    uintptr_t hi = ((uintptr_t)p + n + (uintptr_t)page - 1u) & ~((uintptr_t)page - 1u);
    (void)madvise((void *)lo, hi - lo, MADV_DONTNEED);
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
}

int main(int argc, char **argv) {
    const char *model=getenv("LAGUNA_MODEL_DIR"), *stage=getenv("LAGUNA_STAGE_DIR"), *status=getenv("LAGUNA_STATUS_DIR");
    if(argc==3 && !strcmp(argv[1],"--model")) model=argv[2];
    if(!model) model="/home/u14346/models/laguna-s21-int4";
    if(!stage) stage="/local/laguna-s21";
    int rank=rank_env(), ep=envi("LAGUNA_EP_SIZE",12), nshards=envi("LAGUNA_NSHARDS",15), dense=envi("LAGUNA_STAGE_DENSE",1);
    int shard_limit=envi("LAGUNA_SHARD_LIMIT",0);
    int layer_limit=envi("LAGUNA_STAGE_LAYERS",0);
    if (shard_limit <= 0 || shard_limit > nshards) shard_limit = nshards;
    if(rank<0||rank>=ep||ep<1||ep>12){fprintf(stderr,"bad rank/ep-size %d/%d\n",rank,ep);return 2;}
    if(mkdir_p(stage)){perror(stage);return 2;}
    char bp[1200], mp[1200], bt[1220], mt[1220];
    snprintf(bp,sizeof bp,"%s/rank%02d.blob",stage,rank); snprintf(mp,sizeof mp,"%s/rank%02d.manifest",stage,rank);
    snprintf(bt,sizeof bt,"%s.tmp",bp); snprintf(mt,sizeof mt,"%s.tmp",mp);
    int out=open(bt,O_WRONLY|O_CREAT|O_TRUNC,0644); if(out<0){perror(bt);return 2;}
    FILE *mf=fopen(mt,"w"); if(!mf){perror(mt);close(out);return 2;}
    fprintf(mf,"# LAGUNA-S21 rank=%d ep_size=%d; offset bytes dtype ndims shape... name\n",rank,ep);
    uint64_t off=0, bytes=0; long nt=0, ne=0; double t0=now_sec();
    for(int s=1;s<=shard_limit;s++) {
        char sp[1200]; snprintf(sp,sizeof sp,"%s/model-%05d-of-%05d.safetensors",model,s,nshards);
        int sfd=open(sp,O_RDONLY); st_context *st=safetensors_open(sp);
        if(!st||sfd<0){fprintf(stderr,"cannot open %s\n",sp); if(st)safetensors_close(st); if(sfd>=0)close(sfd); goto fail;}
        for(int i=0;i<st->n_tensors;i++) {
            const char *name=safetensors_name(st,i); long e=expert_id(name);
            long layer=layer_id(name);
            if (layer_limit > 0 && layer >= layer_limit) continue;
            if(e>=0 && e%ep!=rank) continue;
            if(e<0 && !dense) continue;
            size_t nb=safetensors_nbytes(st,i); uint64_t ao=(off+(ALIGN-1))&~(uint64_t)(ALIGN-1);
            if(ao!=off && lseek(out,(off_t)ao,SEEK_SET)<0){perror("lseek");safetensors_close(st);close(sfd);goto fail;}
            if(write_all(out,safetensors_data(st,i),nb)){perror("write");safetensors_close(st);close(sfd);goto fail;}
            const uint64_t *sh=safetensors_shape(st,i); int nd=safetensors_ndims(st,i);
            fprintf(mf,"%" PRIu64 " %zu %s %d",ao,nb,safetensors_dtype(st,i),nd);
            for (int d = 0; d < nd; d++)
                fprintf(mf, " %" PRIu64, sh[d]);
            fprintf(mf, " %s\n", name);
            off=ao+nb; bytes+=nb; nt++; if(e>=0)ne++; drop_source(sfd,safetensors_data(st,i),nb);
        }
        safetensors_close(st); close(sfd);
    }
    if(fsync(out)||close(out)){perror("stage blob");goto fail_no_fd;} fclose(mf);
    if(rename(bt,bp)||rename(mt,mp)){perror("rename");return 2;}
    if(status&&*status){char q[1200];snprintf(q,sizeof q,"%s/laguna_stage_rank%02d.txt",status,rank);FILE*f=fopen(q,"w");if(f){fprintf(f,"rank=%d ep=%d tensors=%ld experts=%ld bytes=%" PRIu64 " sec=%.1f DONE\n",rank,ep,nt,ne,bytes,now_sec()-t0);fclose(f);}}
    printf("laguna stage rank %d/%d: %ld tensors (%ld expert), %.3f GB -> %s\n",rank,ep,nt,ne,bytes/1e9,bp); return 0;
fail: close(out); fail_no_fd: fclose(mf); unlink(bt); unlink(mt); return 2;
}
