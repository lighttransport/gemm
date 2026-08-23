/* Pipeline-parallel Qwen3.8 runner: one MPI rank per A64FX node. */
#include <mpi.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define BPE_TOKENIZER_IMPLEMENTATION
#include "bpe_tokenizer.h"
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double qtime(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static int qargmax(const float*x,int n){int b=0;for(int i=1;i<n;i++)if(x[i]>x[b])b=i;return b;}
static int pp_vocab_argmax(transformer_model *m, int rank, int nr) {
    int v0=m->n_vocab*rank/nr, v1=m->n_vocab*(rank+1)/nr;
    float *logits=transformer_compute_logits_slice(m,v0,v1);
    int li=qargmax(logits,v1-v0), gi=v0+li;
    float local=logits[li], best;
    MPI_Allreduce(&local,&best,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
    int owner=(local==best)?rank:nr, winner;
    MPI_Allreduce(&owner,&winner,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    MPI_Bcast(&gi,1,MPI_INT,winner,MPI_COMM_WORLD);
    return gi;
}
static void cuts(const gguf_context*g,int nl,int nr,int*c){
    uint64_t*w=calloc((size_t)nl,sizeof(*w)),sum=0;
    for(uint64_t i=0;i<g->n_tensors;i++){int l=-1;const char*n=gguf_tensor_name(g,(int)i);if(n&&sscanf(n,"blk.%d.",&l)==1&&l>=0&&l<nl)w[l]+=gguf_tensor_size(g,(int)i);}
    for(int l=0;l<nl;l++)sum+=w[l];c[0]=0;int l=0;uint64_t a=0;
    for(int r=1;r<nr;r++){uint64_t goal=sum*(uint64_t)r/nr;while(l<nl&&a+w[l]/2<goal)a+=w[l++];c[r]=l;}c[nr]=nl;free(w);
}
static int pp_trunk_layers(const gguf_context *g) {
    /* Keep this in sync with transformer_load()'s architecture detection.  PP
     * ranges must be installed before transformer_load(), so it cannot use the
     * model's m->n_layers yet. */
    static const char *archs[] = {
        "gemma4", "qwen2vl", "qwen35", "qwen3vlmoe", "qwen3moe",
        "qwen3vl", "qwen3", "qwen2"
    };
    for (size_t i = 0; i < sizeof(archs) / sizeof(archs[0]); i++) {
        char key[96];
        snprintf(key, sizeof(key), "%s.block_count", archs[i]);
        if (gguf_find_key(g, key) >= 0) {
            int total = tf_get_int(g, key, 0);
            snprintf(key, sizeof(key), "%s.nextn_predict_layers", archs[i]);
            int nextn = tf_get_int(g, key, 0);
            return total - nextn;
        }
    }
    return 0;
}
static void usage(const char*p){fprintf(stderr,"usage: %s MODEL [--prompt TEXT|--token-id ID] [--max-gen N] [--max-seq N] [--threads N] [--spec-k 0..4]\n",p);}

int main(int ac,char**av){
    MPI_Init(&ac,&av);int rank,nr;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);
    FILE *dbg=NULL;
    if (getenv("TF_PP_DEBUG")) { char dn[96]; snprintf(dn,sizeof(dn),"/home/u14346/q38pp-rank%d.log",rank); dbg=fopen(dn,"w"); }
    const char*path=NULL,*prompt="Hello";int ngen=16,nseq=512,nth=48,sk=0,synth=-1;
    for(int i=1;i<ac;i++){if(!strcmp(av[i],"--prompt")&&++i<ac)prompt=av[i];else if(!strcmp(av[i],"--token-id")&&++i<ac)synth=atoi(av[i]);else if(!strcmp(av[i],"--max-gen")&&++i<ac)ngen=atoi(av[i]);else if(!strcmp(av[i],"--max-seq")&&++i<ac)nseq=atoi(av[i]);else if(!strcmp(av[i],"--threads")&&++i<ac)nth=atoi(av[i]);else if(!strcmp(av[i],"--spec-k")&&++i<ac)sk=atoi(av[i]);else if(av[i][0]!='-'&&!path)path=av[i];else{if(!rank)usage(av[0]);MPI_Abort(MPI_COMM_WORLD,2);}}
    if(!path||nr<2||sk<0||sk>4){if(!rank)usage(av[0]);MPI_Abort(MPI_COMM_WORLD,2);}
    /* Mode 2 maps tensor shards lazily. Each rank touches only its assigned layers. */
    gguf_context*g=gguf_open_multi(path,2);if(!g)MPI_Abort(MPI_COMM_WORLD,1);
    int nl=pp_trunk_layers(g); int*c=malloc((size_t)(nr+1)*sizeof(*c));
    if (!c || nl <= 0) MPI_Abort(MPI_COMM_WORLD, 1);
    cuts(g,nl,nr,c); int l0=c[rank],l1=c[rank+1];
    char l0buf[32], l1buf[32];
    snprintf(l0buf,sizeof(l0buf),"%d",l0); snprintf(l1buf,sizeof(l1buf),"%d",l1);
    setenv("TF_PP_L0",l0buf,1); setenv("TF_PP_L1",l1buf,1);
    bpe_vocab*v=bpe_vocab_load(g);transformer_model*m=transformer_load(g,nseq);if(!v||!m)MPI_Abort(MPI_COMM_WORLD,1);
    if (m->n_layers != nl) MPI_Abort(MPI_COMM_WORLD, 1);
    transformer_set_threads(m,nth);
    if (getenv("TF_PP_RESIDENT")) transformer_materialize_pp(m,l0,l1);
    transformer_free_unused_kv(m,l0,l1);
    if (dbg) { fprintf(dbg,"loaded layers=%d,%d n_vocab=%d head=%d\n",l0,l1,m->n_vocab,m->has_lm_head); fflush(dbg); }
    fprintf(stderr,"qwen38-pp rank=%d/%d layers=[%d,%d)\n",rank,nr,l0,l1);
    int32_t*ts=NULL;int nt=0;if(!rank){ts=malloc((size_t)nseq*sizeof(*ts));if(synth>=0){ts[0]=synth;nt=1;}else nt=bpe_tokenize(v,prompt,-1,ts,nseq);}MPI_Bcast(&nt,1,MPI_INT,0,MPI_COMM_WORLD);if(nt<=0||nt+ngen+sk>=nseq)MPI_Abort(MPI_COMM_WORLD,2);
    float*h=transformer_get_hidden(m);int cur=0,pending=-1;long ok=0,all=0;double t0=qtime();
    for(int pos=0;pos<nt+ngen;pos++){
        int token;if(pos<nt){if(!rank)token=ts[pos];MPI_Bcast(&token,1,MPI_INT,0,MPI_COMM_WORLD);}else token=cur;
        if(pos>=nt){
            if(rank==nr-1&&pending>=0){ok+=pending==token;all++;}
            if(!rank){const char*s=bpe_token_to_str(v,token);if(s)fputs(s,stdout);fflush(stdout);}
            if(token==v->eos_id||token==v->eot_id)break;
        }
        if(!rank)transformer_embed_token(m,token);else MPI_Recv(h,m->n_embd,MPI_FLOAT,rank-1,100,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if (dbg) { fprintf(dbg,"forward pos=%d\n",pos); fflush(dbg); }
        transformer_forward_partial(m,pos,l0,l1);if(rank<nr-1)MPI_Send(h,m->n_embd,MPI_FLOAT,rank+1,100,MPI_COMM_WORLD);
        else{
            /* The final stage used to do the entire vocabulary projection.  That
             * makes PP decode-bound on one node (and is especially painful on a
             * cold BF16 mmap).  Replicate only the final hidden vector, then let
             * every stage project a disjoint vocabulary slice. */
            MPI_Bcast(h,m->n_embd,MPI_FLOAT,nr-1,MPI_COMM_WORLD);
            transformer_set_hidden(m,h);
            if (dbg) { fprintf(dbg,"head-bcast pos=%d\n",pos); fflush(dbg); }
            cur=pp_vocab_argmax(m,rank,nr); pending=-1;
            if (dbg) { fprintf(dbg,"head-done pos=%d cur=%d\n",pos,cur); fflush(dbg); }
            if(sk){const float*dh=h;int prev=token;for(int k=0;k<sk;k++){int d=qargmax(transformer_nextn_logits(m,prev,dh,pos+k),m->n_vocab);if(!k)pending=d;prev=d;dh=transformer_nextn_hidden(m);}}
        }
        if(rank<nr-1){
            /* All ranks participate in the vocab-sharded head after the final
             * stage's hidden broadcast. */
            MPI_Bcast(h,m->n_embd,MPI_FLOAT,nr-1,MPI_COMM_WORLD);
            transformer_set_hidden(m,h);
            if (dbg) { fprintf(dbg,"head-bcast pos=%d\n",pos); fflush(dbg); }
            cur=pp_vocab_argmax(m,rank,nr);
            if (dbg) { fprintf(dbg,"head-done pos=%d cur=%d\n",pos,cur); fflush(dbg); }
        }
        MPI_Bcast(&cur,1,MPI_INT,nr-1,MPI_COMM_WORLD);
    }
    long gok=0,gall=0;MPI_Reduce(&ok,&gok,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);MPI_Reduce(&all,&gall,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);if(!rank){fputc('\n',stdout);fprintf(stderr,"qwen38-pp elapsed=%.3fs",qtime()-t0);if(gall)fprintf(stderr," mtp_greedy_match=%ld/%ld alpha=%.4f",gok,gall,(double)gok/gall);fputc('\n',stderr);}
    if (dbg) { fprintf(dbg,"elapsed=%.6f\n",qtime()-t0); fclose(dbg); }
    free(ts);free(c);transformer_free(m);bpe_vocab_free(v);gguf_close(g);MPI_Finalize();return 0;
}
