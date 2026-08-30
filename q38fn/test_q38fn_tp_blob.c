#define _GNU_SOURCE
#define Q38FN_TP_BLOB_IMPLEMENTATION
#include "../common/q38fn_tp_blob.h"
#include <sys/stat.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

int main(void)
{
    char dir[256], blob_path[320], manifest_path[320];
    if (mkdir("../tmp",0700) && errno != EEXIST) return 1;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    int made=0;
    for(int attempt=0;attempt<100&&!made;attempt++){
        snprintf(dir,sizeof(dir),"../tmp/q38fn-tp-blob-%ld-%ld-%d",
                 (long)getpid(),ts.tv_nsec,attempt);
        if(!mkdir(dir,0700))made=1;else if(errno!=EEXIST)return 1;
    }
    if(!made)return 1;
    snprintf(blob_path,sizeof(blob_path),"%s/tp%d-v%d.blob",dir,Q38FN_TP_RANKS,Q38FN_TP_LAYOUT_VERSION);
    snprintf(manifest_path,sizeof(manifest_path),"%s/tp%d-v%d.manifest",dir,Q38FN_TP_RANKS,Q38FN_TP_LAYOUT_VERSION);
    uint16_t values[8]={1,2,3,4,5,6,7,8};
    FILE *f=fopen(blob_path,"wb"); if(!f||fwrite(values,sizeof(values),1,f)!=1)return 1; fclose(f);
    uint64_t hash=q38fn_tp_blob_hash(values,sizeof(values));
    f=fopen(manifest_path,"w"); if(!f)return 1;
    fprintf(f,"# Q38FNTP layout=%d rank=0 ranks=12 layers=1\n",Q38FN_TP_LAYOUT_VERSION);
    fprintf(f,"0 16 %016llx 2 0 2 4 2 1 0 4 test.weight\n",(unsigned long long)hash);
    fprintf(f,"# COMPLETE blob_bytes=16 tensors=1\n"); fclose(f);
    q38fn_tp_blob b;
    if(q38fn_tp_blob_open(&b,dir,1)||b.n_entries!=1||b.bytes!=sizeof(values))return 1;
    const q38fn_tp_blob_entry*e=q38fn_tp_blob_find(&b,"test.weight");
    int ok=e&&e->data[7]==8&&e->shape[0]==4&&e->range[0].count==4;
    q38fn_tp_blob_close(&b);
    uint16_t ngram[2*Q38FN_NGRAM_HEAD_DIM];
    for(size_t i=0;i<sizeof(ngram)/sizeof(*ngram);i++){
        float value=sinf((float)i)*0.0625f;uint32_t bits;
        memcpy(&bits,&value,sizeof(bits));ngram[i]=(uint16_t)(bits>>16);
    }
    f=fopen(blob_path,"wb");if(!f||fwrite(ngram,sizeof(ngram),1,f)!=1)return 1;fclose(f);
    f=fopen(manifest_path,"w");if(!f)return 1;
    fprintf(f,"# Q38FNTP layout=%d rank=0 ranks=12 layers=1\n",Q38FN_TP_LAYOUT_VERSION);
    fprintf(f,"0 %zu 0 5 -1 2 2 %d 0 model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight\n",sizeof(ngram),Q38FN_NGRAM_HEAD_DIM);
    fprintf(f,"# COMPLETE blob_bytes=%zu tensors=1\n",sizeof(ngram));fclose(f);
    if(q38fn_tp_blob_open(&b,dir,0)||b.n_entries!=1)return 1;
    e=&b.entries[0];float decoded[Q38FN_NGRAM_HEAD_DIM];
    if(!e->q5_data||e->q5_bytes!=q38fn_q5_bytes(2,Q38FN_NGRAM_HEAD_DIM)||
       q38fn_q5_dequantize_row(decoded,e->q5_data,Q38FN_NGRAM_HEAD_DIM))return 1;
    double error2=0,source2=0;
    for(int i=0;i<Q38FN_NGRAM_HEAD_DIM;i++){float source=q38fn_q5_bf16(ngram[i]);double d=decoded[i]-source;error2+=d*d;source2+=(double)source*source;}
    if(sqrt(error2/source2)>0.08)return 1;
    q38fn_tp_blob_close(&b);
    q38fn_q5_block q5[2*Q38FN_NGRAM_HEAD_DIM/32];
    if(q38fn_q5_quantize_bf16(q5,ngram,2,Q38FN_NGRAM_HEAD_DIM))return 1;
    f=fopen(blob_path,"wb");if(!f||fwrite(q5,sizeof(q5),1,f)!=1)return 1;fclose(f);
    f=fopen(manifest_path,"w");if(!f)return 1;
    fprintf(f,"# Q38FNTP layout=%d rank=0 ranks=12 layers=1\n",Q38FN_TP_LAYOUT_VERSION);
    fprintf(f,"0 %zu 0 5 -1 2 2 %d 0 model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight\n",sizeof(q5),Q38FN_NGRAM_HEAD_DIM);
    fprintf(f,"# COMPLETE blob_bytes=%zu tensors=1\n",sizeof(q5));fclose(f);
    if(q38fn_tp_blob_open(&b,dir,0)||!b.entries[0].q5_data||
       b.entries[0].q5_bytes!=sizeof(q5))return 1;
    q38fn_tp_blob_close(&b);unlink(blob_path);unlink(manifest_path);rmdir(dir);
    if(!ok)return 1;
    puts("Q38FN_TP_BLOB_TEST ok");return 0;
}
