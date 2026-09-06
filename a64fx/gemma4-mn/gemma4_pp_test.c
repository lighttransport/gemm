/* gemma4_pp_test.c - validate the PP shard loader: run a deterministic synthetic
 * hidden state through one stage's layer range, from (a) the per-rank blob shard and
 * (b) the full model -- the output-hidden checksums must MATCH (same weights).
 *
 * Usage: gemma4_pp_test <gguf> <blob_dir|FULL> <rank> <nranks> [max_seq]
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *          -I../../common gemma4_pp_test.c -lm -o gemma4_pp_test
 */
#define _GNU_SOURCE
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include "gemma4_pp_load.h"

static float hashf(unsigned i){ unsigned long long h=i*0x9E3779B97F4A7C15ULL; h^=h>>29; h*=0xBF58476D1CE4E5B9ULL; h^=h>>32; return (float)((h&0xFFFF)/65536.0-0.5); }

int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: %s <gguf> <blob_dir|FULL> <rank> <nranks> [max_seq]\n",argv[0]); return 1; }
    const char *gguf=argv[1], *src=argv[2]; int rank=atoi(argv[3]), nranks=atoi(argv[4]);
    int max_seq=argc>5?atoi(argv[5]):512;
    int nthr=getenv("LLM_THREADS")?atoi(getenv("LLM_THREADS")):48;
    transformer_model *m; int L0,L1;

    if(!strcmp(src,"FULL")){
        gguf_context *g=gguf_open(gguf,1); if(!g){fprintf(stderr,"open fail\n");return 1;}
        m=transformer_load(g,max_seq); if(!m){fprintf(stderr,"load fail\n");return 1;}
        int ki=gguf_find_key_internal(g,"gemma4.block_count"); int nl=(int)g->kv[ki].value.u32;
        L0=rank*nl/nranks; L1=(rank+1)*nl/nranks;
        fprintf(stderr,"FULL model, testing layer range [%d,%d)\n",L0,L1);
    } else {
        char blob[1100],mani[1100];
        snprintf(blob,sizeof blob,"%s/rank%02d.blob",src,rank);
        snprintf(mani,sizeof mani,"%s/rank%02d.manifest",src,rank);
        m=gemma4_pp_load(gguf,blob,mani,rank,nranks,max_seq,&L0,&L1);
        if(!m){fprintf(stderr,"pp load fail\n");return 1;}
    }
    transformer_set_threads(m,nthr);

    /* deterministic synthetic input hidden -> m->x */
    for(int i=0;i<m->n_embd;i++) m->x[i]=hashf((unsigned)i);
    /* run this stage's layers at cache_pos 0 */
    float *h=transformer_forward_partial(m,0,L0,L1);
    if(!h){fprintf(stderr,"forward_partial returned NULL\n");return 1;}
    double s=0,a=0; for(int i=0;i<m->n_embd;i++){ s+=(double)h[i]*(i+1); a+=h[i]<0?-h[i]:h[i]; }
    fprintf(stderr,"[%s rank %d] layers [%d,%d): hidden checksum=%.6f  L1norm=%.6f  h[0..2]=%.5f %.5f %.5f\n",
        src,rank,L0,L1,s,a,h[0],h[1],h[2]);
    printf("PPTEST src=%s rank=%d L=[%d,%d) checksum=%.6f l1=%.6f\n",src,rank,L0,L1,s,a);
    return 0;
}
