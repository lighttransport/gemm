/* glm5_stage.c — shard the GLM-5.2 (text) safetensors load and stage each EP
 * rank's slice to that node's local scratch (/local/glm5).
 *
 * Text-only: stage model layers 0..77 plus embed/lm_head/norm. The next-N/speculative
 * layer 78 is skipped for the bring-up path. Dense tensors are staged WHOLE on every
 * rank; the loader applies TP slicing. Routed experts (mlp.experts.E) are kept iff
 * E % ep_size == ep_rank.
 *
 * Ownership from the tensor NAME alone (no index.json):
 *   - "model.layers.78.*" -> SKIP by default
 *   - "...mlp.experts.E..." -> KEEP iff E%ep_size==rank
 *   - everything else -> KEEP (replicated dense)
 *
 *   out_dir/rank<rr>.blob      packed weights (256B aligned per tensor)
 *   out_dir/rank<rr>.manifest  header + per tensor: <off> <nbytes> <dtype> <ndims> <d..> <name>
 *
 * Build (native A64FX):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -std=c11 -D_GNU_SOURCE -I../common \
 *       -o build/glm5_stage common/glm5_stage.c
 * Run (per node):
 *   GLM5_EP_RANK=0 GLM5_EP_SIZE=96 ./build/glm5_stage
 * Env: GLM5_MODEL_DIR ($HOME/models/glm5.2), GLM5_STAGE_DIR (/local/glm5), GLM5_EP_RANK/SIZE,
 *      GLM5_NSHARDS (282), GLM5_STAGE_LAYERS (0=main 78; keep layers.L.* with L<N),
 *      GLM5_STAGE_FLUSH_GB (2), GLM5_STATUS_DIR.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

#define ALIGN 256u
static double now_sec(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int envi(const char*k,int def){ const char*e=getenv(k); return (e&&*e)?atoi(e):def; }

static int detect_rank(void){
    const char*keys[]={"GLM5_EP_RANK","PMIX_RANK","OMPI_COMM_WORLD_RANK","PMI_RANK","MV2_COMM_WORLD_RANK",NULL};
    for(int i=0;keys[i];i++){ const char*e=getenv(keys[i]); if(e&&*e) return atoi(e); }
    return 0;
}
/* expert id from "...mlp.experts.<E>...."; -1 if not an expert */
static long expert_id(const char*name){
    const char*p=strstr(name,".experts."); if(!p) return -1; p+=9;
    if(*p<'0'||*p>'9') return -1; return strtol(p,NULL,10);
}
/* layer index from "...layers.<L>...."; -1 if none */
static long layer_id(const char*name){
    const char*p=strstr(name,".layers."); if(!p) return -1; p+=8;
    if(*p<'0'||*p>'9') return -1; return strtol(p,NULL,10);
}

/* Mixed-precision: layers listed in GLM5_BF16_LAYERS ("0,1,2,77" or "0-2,77") are staged
 * from the bf16 model (GLM5_BF16_DIR) WITHOUT a _scale_inv companion, so the loader uses
 * the bf16 path for them; all other layers stay FP8 from GLM5_MODEL_DIR. Targets the
 * sensitive first/last layers to recover instruction-following lost to E4M3 quantization. */
static int g_nbf16=0; static int g_bf16[256];
static int is_bf16_layer(long L){ for(int i=0;i<g_nbf16;i++) if(g_bf16[i]==L) return 1; return 0; }
static void parse_bf16_layers(void){
    const char*e=getenv("GLM5_BF16_LAYERS"); if(!e||!*e) return;
    char*p=(char*)e;            /* accepts ',' ':' or ' ' separators ("0-2,77" or "0:1:2:77") */
    while(*p){
        while(*p==','||*p==' '||*p==':')p++; if(!*p)break;
        long a=strtol(p,&p,10), b=a;
        if(*p=='-'){ p++; b=strtol(p,&p,10); }
        for(long L=a;L<=b && g_nbf16<256;L++) g_bf16[g_nbf16++]=(int)L;
    }
}

enum { CLS_SKIP, CLS_DENSE, CLS_EXPERT };
/* bf16_pass=0: FP8 source pass (skip bf16-designated layers); 1: bf16 source pass (keep only them). */
static int classify(const char*name,int rank,int ep_size,int bf16_pass){
    static int slay=-2; if(slay==-2) slay=envi("GLM5_STAGE_LAYERS",78);
    long L=layer_id(name);
    if(slay>0){ if(L>=0 && L>=slay) return CLS_SKIP; }
    if(g_nbf16>0){
        int isb=(L>=0 && is_bf16_layer(L));
        if(bf16_pass){ if(!isb) return CLS_SKIP; }   /* bf16 pass keeps only designated layers */
        else { if(isb) return CLS_SKIP; }            /* fp8 pass skips designated layers */
    } else if(bf16_pass) return CLS_SKIP;            /* no designated layers -> bf16 pass empty */
    long e=expert_id(name);
    if(e>=0) return (e%ep_size==rank)?CLS_EXPERT:CLS_SKIP;
    return CLS_DENSE;
}
static int write_all(int fd,const void*buf,size_t n){
    const uint8_t*p=buf; while(n){ ssize_t w=write(fd,p,n>(1u<<30)?(1u<<30):n); if(w<0){ if(errno==EINTR)continue; return -1;} p+=w; n-=(size_t)w; } return 0;
}

int main(void){
    const char*home=getenv("HOME"); if(!home) home=".";
    char model_dir[1024], stage_dir[1024];
    { const char*e=getenv("GLM5_MODEL_DIR"); if(e&&*e) snprintf(model_dir,sizeof model_dir,"%s",e);
      else snprintf(model_dir,sizeof model_dir,"%s/models/glm5.2",home); }
    { const char*e=getenv("GLM5_STAGE_DIR"); if(e&&*e) snprintf(stage_dir,sizeof stage_dir,"%s",e);
      else { struct stat sb; if(stat("/local",&sb)==0&&S_ISDIR(sb.st_mode)) snprintf(stage_dir,sizeof stage_dir,"/local/glm5");
             else snprintf(stage_dir,sizeof stage_dir,"%s/tmp/glm5",home); } }
    int rank=detect_rank();
    int ep_size=envi("GLM5_EP_SIZE",192);
    /* data-parallel groups: ep_size is the GROUP size; ranks beyond it belong to sibling groups that
     * stage the SAME group-local expert shard (e%ep_size==rank) to their own node's /local. Map the
     * global MPI rank to its group-local index so each group is a complete, independently-staged model. */
    if(ep_size>0) rank %= ep_size;
    int nshards=envi("GLM5_NSHARDS",282);
    int slimit=envi("GLM5_SHARD_LIMIT",0);
    int fp8_last=(slimit>0&&slimit<nshards)?slimit:nshards;
    parse_bf16_layers();
    char bf16_dir[1024];
    { const char*e=getenv("GLM5_BF16_DIR"); if(e&&*e) snprintf(bf16_dir,sizeof bf16_dir,"%s",e);
      else snprintf(bf16_dir,sizeof bf16_dir,"%s/models/glm5.2",home); }
    int bf16_nshards=envi("GLM5_BF16_NSHARDS",282);
    uint64_t flush_bytes=(uint64_t)(envi("GLM5_STAGE_FLUSH_GB",2)>0?envi("GLM5_STAGE_FLUSH_GB",2):2)<<30;
    if(rank<0||rank>=ep_size){ fprintf(stderr,"glm5_stage: bad rank %d for ep_size %d\n",rank,ep_size); return 2; }
    mkdir(stage_dir,0755);

    char blob_path[1100], mani_path[1100];
    snprintf(blob_path,sizeof blob_path,"%s/rank%02d.blob",stage_dir,rank);
    snprintf(mani_path,sizeof mani_path,"%s/rank%02d.manifest",stage_dir,rank);
    int bfd=open(blob_path,O_WRONLY|O_CREAT|O_TRUNC,0644);
    if(bfd<0){ fprintf(stderr,"glm5_stage: cannot create %s: %s\n",blob_path,strerror(errno)); return 2; }
    FILE*mf=fopen(mani_path,"w");
    if(!mf){ fprintf(stderr,"glm5_stage: cannot create %s: %s\n",mani_path,strerror(errno)); close(bfd); return 2; }

    printf("glm5_stage: rank %d/%d  model=%s  out=%s  shards=%d\n",rank,ep_size,model_dir,stage_dir,nshards); fflush(stdout);
    long hdr_pos=ftell(mf);
    fprintf(mf,"# GLM5MANIFEST rank=%02d ep_size=%02d n_tensors=%-12d blob_bytes=%-18lld\n",rank,ep_size,0,0LL);

    uint64_t off=0,last_sync=0; long long n_dense=0,n_expert=0; uint64_t b_dense=0,b_expert=0; double t0=now_sec();
    int npass=(g_nbf16>0)?2:1;
    if(g_nbf16>0){ printf("glm5_stage: mixed-precision %d bf16 layers from %s (rest FP8)\n",g_nbf16,bf16_dir); fflush(stdout); }
    for(int pass=0;pass<npass;pass++){
      const char*mdir = pass==0?model_dir:bf16_dir;
      int nsh = pass==0?nshards:bf16_nshards;
      int last = pass==0?fp8_last:bf16_nshards;
      for(int s=1;s<=last;s++){
        char shard[1200]; snprintf(shard,sizeof shard,"%s/model-%05d-of-%05d.safetensors",mdir,s,nsh);
        st_context*st=safetensors_open(shard);
        if(!st){ fprintf(stderr,"glm5_stage: skip unreadable shard %s\n",shard); continue; }
        int kept=0;
        for(int i=0;i<st->n_tensors;i++){
            const char*name=st->tensors[i].name; int cls=classify(name,rank,ep_size,pass);
            if(cls==CLS_SKIP) continue;
            size_t nb=st->tensors[i].nbytes;
            uint64_t aligned=(off+(ALIGN-1))&~(uint64_t)(ALIGN-1);
            if(aligned!=off && lseek(bfd,(off_t)aligned,SEEK_SET)<0){ fprintf(stderr,"glm5_stage: lseek: %s\n",strerror(errno)); goto fail; }
            if(write_all(bfd,safetensors_data(st,i),nb)!=0){ fprintf(stderr,"glm5_stage: write %s: %s\n",name,strerror(errno)); goto fail; }
            st_tensor_info*t=&st->tensors[i];
            fprintf(mf,"%llu %zu %s %d",(unsigned long long)aligned,nb,t->dtype_str,t->n_dims);
            for(int d=0;d<t->n_dims;d++) fprintf(mf," %llu",(unsigned long long)t->shape[d]);
            fprintf(mf," %s\n",name);
            off=aligned+nb;
            if(cls==CLS_EXPERT){ n_expert++; b_expert+=nb; } else { n_dense++; b_dense+=nb; }
            kept++;
            if(off-last_sync>=flush_bytes){ fdatasync(bfd); posix_fadvise(bfd,0,0,POSIX_FADV_DONTNEED); last_sync=off; }
        }
        madvise(st->map_base,st->map_size,MADV_DONTNEED);
        safetensors_close(st);
        double el=now_sec()-t0; double gb=(b_dense+b_expert)/1e9;
        printf("  [p%d] shard %2d/%d  kept %4d  cum %5.1f GB  %5.1f s  %5.2f GB/s\n",pass,s,nsh,kept,gb,el,el>0?gb/el:0.0); fflush(stdout);
      }
    }
    double tel=now_sec()-t0; long long n_total=n_dense+n_expert; uint64_t b_total=b_dense+b_expert;
    fseek(mf,hdr_pos,SEEK_SET);
    fprintf(mf,"# GLM5MANIFEST rank=%02d ep_size=%02d n_tensors=%-12lld blob_bytes=%-18llu\n",rank,ep_size,n_total,(unsigned long long)b_total);
    fclose(mf); fdatasync(bfd); posix_fadvise(bfd,0,0,POSIX_FADV_DONTNEED);
    if(close(bfd)<0){ fprintf(stderr,"glm5_stage: close blob: %s\n",strerror(errno)); return 2; }
    printf("\nrank %d done: %lld tensors (%lld dense / %lld expert)  staged %.2f GB  blob %.2f GB  %.1fs %.2f GB/s\n",
           rank,n_total,n_dense,n_expert,b_total/1e9,off/1e9,tel,tel>0?b_total/1e9/tel:0.0);
    { const char*sdir=getenv("GLM5_STATUS_DIR"); char sp[1200];
      snprintf(sp,sizeof sp,"%s/glm5_stage_rank%02d.txt",(sdir&&*sdir)?sdir:".",rank);
      FILE*sf=fopen(sp,"w"); if(sf){ fprintf(sf,"rank=%02d ep_size=%d tensors=%lld dense=%lld expert=%lld staged_GB=%.3f blob_GB=%.3f sec=%.1f blob=%s DONE\n",
            rank,ep_size,n_total,n_dense,n_expert,b_total/1e9,off/1e9,tel,blob_path); fclose(sf);} }
    return 0;
fail: fclose(mf); close(bfd); return 2;
}
