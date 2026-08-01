/* m3_stage.c — shard the MiniMax-M3 (text) safetensors load and stage each EP
 * rank's slice to that node's local scratch (/local/m3).
 *
 * Text-only: the language_model tensors (layers 0..59 + embed_tokens + lm_head +
 * model.norm) are staged; vision_tower / multi_modal_projector / patch_merge_mlp
 * are SKIPPED. Dense (replicated) tensors are staged WHOLE on every rank; the
 * loader (m3_load_real) applies TP slicing. Routed experts (block_sparse_moe.
 * experts.E) are kept iff E % ep_size == ep_rank.
 *
 * Ownership from the tensor NAME alone (no index.json):
 *   - vision_tower. / multi_modal_projector. / patch_merge_mlp.  -> SKIP
 *   - "...block_sparse_moe.experts.E..."  -> KEEP iff E%ep_size==rank
 *     (".experts." has a leading dot, so "shared_experts" never matches)
 *   - everything else (language_model.*)  -> KEEP (replicated dense)
 *
 *   out_dir/rank<rr>.blob      packed weights (256B aligned per tensor)
 *   out_dir/rank<rr>.manifest  header + per tensor: <off> <nbytes> <dtype> <ndims> <d..> <name>
 *
 * Build (native A64FX):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -std=c11 -D_GNU_SOURCE -I../common \
 *       -o build/m3_stage common/m3_stage.c
 * Run (per node):
 *   M3_EP_RANK=0 M3_EP_SIZE=96 ./build/m3_stage
 * Env: M3_MODEL_DIR ($HOME/models/m3), M3_STAGE_DIR (/local/m3), M3_EP_RANK/SIZE,
 *      M3_NSHARDS (59), M3_STAGE_LAYERS (0=all; keep layers.L.* with L<N),
 *      M3_STAGE_FLUSH_GB (2), M3_STATUS_DIR.
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
    const char*keys[]={"M3_EP_RANK","PMIX_RANK","OMPI_COMM_WORLD_RANK","PMI_RANK","MV2_COMM_WORLD_RANK",NULL};
    for(int i=0;keys[i];i++){ const char*e=getenv(keys[i]); if(e&&*e) return atoi(e); }
    return 0;
}
/* expert id from "...block_sparse_moe.experts.<E>...."; -1 if not an expert */
static long expert_id(const char*name){
    const char*p=strstr(name,".experts."); if(!p) return -1; p+=9;
    if(*p<'0'||*p>'9') return -1; return strtol(p,NULL,10);
}
/* layer index from "...layers.<L>...."; -1 if none */
static long layer_id(const char*name){
    const char*p=strstr(name,".layers."); if(!p) return -1; p+=8;
    if(*p<'0'||*p>'9') return -1; return strtol(p,NULL,10);
}

enum { CLS_SKIP, CLS_DENSE, CLS_EXPERT };
static int classify(const char*name,int rank,int ep_size){
    if(strncmp(name,"vision_tower.",13)==0) return CLS_SKIP;
    if(strncmp(name,"multi_modal_projector.",22)==0) return CLS_SKIP;
    if(strncmp(name,"patch_merge_mlp.",16)==0) return CLS_SKIP;
    static int slay=-2; if(slay==-2) slay=envi("M3_STAGE_LAYERS",0);
    if(slay>0){ long L=layer_id(name); if(L>=0 && L>=slay) return CLS_SKIP; }
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
    { const char*e=getenv("M3_MODEL_DIR"); if(e&&*e) snprintf(model_dir,sizeof model_dir,"%s",e);
      else snprintf(model_dir,sizeof model_dir,"%s/models/m3",home); }
    { const char*e=getenv("M3_STAGE_DIR"); if(e&&*e) snprintf(stage_dir,sizeof stage_dir,"%s",e);
      else { struct stat sb; if(stat("/local",&sb)==0&&S_ISDIR(sb.st_mode)) snprintf(stage_dir,sizeof stage_dir,"/local/m3");
             else snprintf(stage_dir,sizeof stage_dir,"%s/tmp/m3",home); } }
    int rank=detect_rank();
    int ep_size=envi("M3_EP_SIZE",96);
    int nshards=envi("M3_NSHARDS",59);
    int slimit=envi("M3_SHARD_LIMIT",0);
    int last=(slimit>0&&slimit<nshards)?slimit:nshards;
    uint64_t flush_bytes=(uint64_t)(envi("M3_STAGE_FLUSH_GB",2)>0?envi("M3_STAGE_FLUSH_GB",2):2)<<30;
    if(rank<0||rank>=ep_size){ fprintf(stderr,"m3_stage: bad rank %d for ep_size %d\n",rank,ep_size); return 2; }
    mkdir(stage_dir,0755);

    char blob_path[1100], mani_path[1100];
    snprintf(blob_path,sizeof blob_path,"%s/rank%02d.blob",stage_dir,rank);
    snprintf(mani_path,sizeof mani_path,"%s/rank%02d.manifest",stage_dir,rank);
    int bfd=open(blob_path,O_WRONLY|O_CREAT|O_TRUNC,0644);
    if(bfd<0){ fprintf(stderr,"m3_stage: cannot create %s: %s\n",blob_path,strerror(errno)); return 2; }
    FILE*mf=fopen(mani_path,"w");
    if(!mf){ fprintf(stderr,"m3_stage: cannot create %s: %s\n",mani_path,strerror(errno)); close(bfd); return 2; }

    printf("m3_stage: rank %d/%d  model=%s  out=%s  shards=%d\n",rank,ep_size,model_dir,stage_dir,nshards); fflush(stdout);
    long hdr_pos=ftell(mf);
    fprintf(mf,"# M3MANIFEST rank=%02d ep_size=%02d n_tensors=%-12d blob_bytes=%-18lld\n",rank,ep_size,0,0LL);

    uint64_t off=0,last_sync=0; long long n_dense=0,n_expert=0; uint64_t b_dense=0,b_expert=0; double t0=now_sec();
    for(int s=1;s<=last;s++){
        char shard[1200]; snprintf(shard,sizeof shard,"%s/model-%05d-of-%05d.safetensors",model_dir,s,nshards);
        st_context*st=safetensors_open(shard);
        if(!st){ fprintf(stderr,"m3_stage: skip unreadable shard %s\n",shard); continue; }
        int kept=0;
        for(int i=0;i<st->n_tensors;i++){
            const char*name=st->tensors[i].name; int cls=classify(name,rank,ep_size);
            if(cls==CLS_SKIP) continue;
            size_t nb=st->tensors[i].nbytes;
            uint64_t aligned=(off+(ALIGN-1))&~(uint64_t)(ALIGN-1);
            if(aligned!=off && lseek(bfd,(off_t)aligned,SEEK_SET)<0){ fprintf(stderr,"m3_stage: lseek: %s\n",strerror(errno)); goto fail; }
            if(write_all(bfd,safetensors_data(st,i),nb)!=0){ fprintf(stderr,"m3_stage: write %s: %s\n",name,strerror(errno)); goto fail; }
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
        printf("  shard %2d/%d  kept %4d  cum %5.1f GB  %5.1f s  %5.2f GB/s\n",s,nshards,kept,gb,el,el>0?gb/el:0.0); fflush(stdout);
    }
    double tel=now_sec()-t0; long long n_total=n_dense+n_expert; uint64_t b_total=b_dense+b_expert;
    fseek(mf,hdr_pos,SEEK_SET);
    fprintf(mf,"# M3MANIFEST rank=%02d ep_size=%02d n_tensors=%-12lld blob_bytes=%-18llu\n",rank,ep_size,n_total,(unsigned long long)b_total);
    fclose(mf); fdatasync(bfd); posix_fadvise(bfd,0,0,POSIX_FADV_DONTNEED);
    if(close(bfd)<0){ fprintf(stderr,"m3_stage: close blob: %s\n",strerror(errno)); return 2; }
    printf("\nrank %d done: %lld tensors (%lld dense / %lld expert)  staged %.2f GB  blob %.2f GB  %.1fs %.2f GB/s\n",
           rank,n_total,n_dense,n_expert,b_total/1e9,off/1e9,tel,tel>0?b_total/1e9/tel:0.0);
    { const char*sdir=getenv("M3_STATUS_DIR"); char sp[1200];
      snprintf(sp,sizeof sp,"%s/m3_stage_rank%02d.txt",(sdir&&*sdir)?sdir:".",rank);
      FILE*sf=fopen(sp,"w"); if(sf){ fprintf(sf,"rank=%02d ep_size=%d tensors=%lld dense=%lld expert=%lld staged_GB=%.3f blob_GB=%.3f sec=%.1f blob=%s DONE\n",
            rank,ep_size,n_total,n_dense,n_expert,b_total/1e9,off/1e9,tel,blob_path); fclose(sf);} }
    return 0;
fail: fclose(mf); close(bfd); return 2;
}
