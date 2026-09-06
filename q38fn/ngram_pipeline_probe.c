/* Deep outstanding-request benchmark for the Qwen3.8 n-gram pipeline. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../common/glm53f_safetensors.h"
#include "../common/q38fn_arch.h"
#include "../common/q38fn_ngram_pipeline.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void usage(const char *p){fprintf(stderr,"usage: %s MODEL_DIR STORAGE_DIR iterations partition workers depth max_span max_gap duplicate_period [cache_rows]\n",p);}

int main(int argc,char **argv){
    if(argc<3){usage(argv[0]);return 2;}
    const char *model=argv[1],*storage=argv[2]; int iters=argc>3?atoi(argv[3]):10000;
    int partition=argc>4?atoi(argv[4]):0, workers=argc>5?atoi(argv[5]):4;
    int depth=argc>6?atoi(argv[6]):64, span=argc>7?atoi(argv[7]):32;
    int gap=argc>8?atoi(argv[8]):2, dup=argc>9?atoi(argv[9]):0;
    int cache_rows=argc>10?atoi(argv[10]):0;
    if(iters<1||partition<0||partition>=Q38FN_NGRAM_SHARDS||workers<1||depth<1||span<1||gap<0||dup<0||dup>16){usage(argv[0]);return 2;}
    glm53f_st_context *ctx=glm53f_st_open(model);const st_context *owner=NULL;char name[180],path[4096];int fd=-1,sid=-1;
    snprintf(name,sizeof(name),"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%d.weight",partition);
    const st_tensor_info *t=ctx?glm53f_st_find(ctx,name,&owner):NULL;
    if(!t||!owner){fprintf(stderr,"missing %s\n",name);goto fail;}
    for(int i=0;i<ctx->n_shards;i++)if(ctx->shards[i].st==owner){sid=i;break;}
    if(sid<0||snprintf(path,sizeof(path),"%s/%s",storage,ctx->shards[sid].name)>=(int)sizeof(path)){goto fail;}
    fd=open(path,O_RDONLY);if(fd<0){perror(path);goto fail;}
    q38fn_ngram_fd_source fs={fd,(off_t)(owner->data_offset+t->offset)};q38fn_ngram_source src[Q38FN_NGRAM_SHARDS];
    memset(src,0,sizeof(src));src[partition]=(q38fn_ngram_source){q38fn_ngram_fd_read_span,&fs,0};
    q38fn_ngram_pipeline *p=NULL;int rc=q38fn_ngram_pipeline_init_ex(&p,src,Q38FN_NGRAM_SHARDS,(uint32_t)depth,(uint32_t)workers,(uint32_t)span,(uint32_t)gap,(uint32_t)cache_rows);
    if(rc){fprintf(stderr,"pipeline init: %s\n",strerror(rc));goto fail;}
    q38fn_ngram_ticket *tickets=calloc((size_t)depth,sizeof(*tickets));uint16_t *out=malloc((size_t)Q38FN_NGRAM_HEADS*Q38FN_NGRAM_ROW_BYTES);uint64_t checksum=0;
    if(!tickets||!out){fprintf(stderr,"allocation failed\n");q38fn_ngram_pipeline_destroy(p);goto fail;}
    int submitted=0,completed=0;double t0=now_sec();
    while(completed<iters){
        if(submitted<iters && submitted-completed<depth){uint64_t rows[Q38FN_NGRAM_HEADS];
            for(int h=0;h<Q38FN_NGRAM_HEADS;h++){uint64_t local=dup?(uint64_t)(h%dup):(uint64_t)(((uint64_t)submitted*7919+h*104729)%Q38FN_NGRAM_ROWS_PER_SHARD);rows[h]=(uint64_t)partition*Q38FN_NGRAM_ROWS_PER_SHARD+local;}
            rc=q38fn_ngram_submit(p,rows,Q38FN_NGRAM_HEADS,&tickets[submitted%depth]);if(rc){fprintf(stderr,"submit: %s\n",strerror(rc));break;}submitted++;
        }else{rc=q38fn_ngram_wait(p,tickets[completed%depth],out,(size_t)Q38FN_NGRAM_HEADS*Q38FN_NGRAM_ROW_BYTES);if(rc){fprintf(stderr,"wait: %s\n",strerror(rc));break;}for(int i=0;i<Q38FN_NGRAM_HEADS;i++)checksum^=out[i*Q38FN_NGRAM_HEAD_DIM+(completed%Q38FN_NGRAM_HEAD_DIM)];completed++;}
    }
    double elapsed=now_sec()-t0;q38fn_ngram_stats st={0};q38fn_ngram_get_stats(p,&st);
    printf("Q38FN_NGRAM_PIPELINE iterations=%d workers=%d depth=%d span=%d gap=%d duplicate_period=%d cache_rows=%d elapsed=%.6f logical_s=%.2f unique_s=%.2f spans_s=%.2f useful_GB_s=%.3f physical_GB_s=%.3f dedup=%llu cache_hits=%llu cache_misses=%llu direct_spans=%llu reorder_spans=%llu coalesce=%.3f wait_ms=%.3f checksum=%llu\n",completed,workers,depth,span,gap,dup,cache_rows,elapsed,st.logical_rows/elapsed,st.unique_rows/elapsed,st.spans/elapsed,st.useful_bytes/elapsed/1e9,st.physical_bytes/elapsed/1e9,(unsigned long long)st.deduplicated_rows,(unsigned long long)st.cache_hits,(unsigned long long)st.cache_misses,(unsigned long long)st.direct_spans,(unsigned long long)st.reorder_spans,st.spans?(double)st.physical_bytes/st.spans/Q38FN_NGRAM_ROW_BYTES:0.0,(double)st.wait_ns/1e6,(unsigned long long)checksum);
    free(out);free(tickets);q38fn_ngram_pipeline_destroy(p);close(fd);glm53f_st_close(ctx);return completed==iters?0:1;
fail: if(fd>=0)close(fd);glm53f_st_close(ctx);return 1;
}
