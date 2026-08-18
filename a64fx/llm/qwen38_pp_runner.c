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
static void cuts(const gguf_context*g,int nl,int nr,int*c){
    uint64_t*w=calloc((size_t)nl,sizeof(*w)),sum=0;
    for(uint64_t i=0;i<g->n_tensors;i++){int l=-1;const char*n=gguf_tensor_name(g,(int)i);if(n&&sscanf(n,"blk.%d.",&l)==1&&l>=0&&l<nl)w[l]+=gguf_tensor_size(g,(int)i);}
    for(int l=0;l<nl;l++)sum+=w[l];c[0]=0;int l=0;uint64_t a=0;
    for(int r=1;r<nr;r++){uint64_t goal=sum*(uint64_t)r/nr;while(l<nl&&a+w[l]/2<goal)a+=w[l++];c[r]=l;}c[nr]=nl;free(w);
}
static void usage(const char*p){fprintf(stderr,"usage: %s MODEL --prompt TEXT [--max-gen N] [--max-seq N] [--threads N] [--spec-k 0..4]\n",p);}

int main(int ac,char**av){
    MPI_Init(&ac,&av);int rank,nr;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);
    const char*path=NULL,*prompt="Hello";int ngen=16,nseq=512,nth=48,sk=0;
    for(int i=1;i<ac;i++){if(!strcmp(av[i],"--prompt")&&++i<ac)prompt=av[i];else if(!strcmp(av[i],"--max-gen")&&++i<ac)ngen=atoi(av[i]);else if(!strcmp(av[i],"--max-seq")&&++i<ac)nseq=atoi(av[i]);else if(!strcmp(av[i],"--threads")&&++i<ac)nth=atoi(av[i]);else if(!strcmp(av[i],"--spec-k")&&++i<ac)sk=atoi(av[i]);else if(av[i][0]!='-'&&!path)path=av[i];else{if(!rank)usage(av[0]);MPI_Abort(MPI_COMM_WORLD,2);}}
    if(!path||nr<2||sk<0||sk>4){if(!rank)usage(av[0]);MPI_Abort(MPI_COMM_WORLD,2);}
    /* Mode 2 maps tensor shards lazily. Each rank touches only its assigned layers. */
    gguf_context*g=gguf_open_multi(path,2);if(!g)MPI_Abort(MPI_COMM_WORLD,1);
    bpe_vocab*v=bpe_vocab_load(g);transformer_model*m=transformer_load(g,nseq);if(!v||!m)MPI_Abort(MPI_COMM_WORLD,1);
    transformer_set_threads(m,nth);int*c=malloc((size_t)(nr+1)*sizeof(*c));cuts(g,m->n_layers,nr,c);int l0=c[rank],l1=c[rank+1];transformer_free_unused_kv(m,l0,l1);
    fprintf(stderr,"qwen38-pp rank=%d/%d layers=[%d,%d)\n",rank,nr,l0,l1);
    int32_t*ts=NULL;int nt=0;if(!rank){ts=malloc((size_t)nseq*sizeof(*ts));nt=bpe_tokenize(v,prompt,-1,ts,nseq);}MPI_Bcast(&nt,1,MPI_INT,0,MPI_COMM_WORLD);if(nt<=0||nt+ngen+sk>=nseq)MPI_Abort(MPI_COMM_WORLD,2);
    float*h=transformer_get_hidden(m);int cur=0,pending=-1;long ok=0,all=0;double t0=qtime();
    for(int pos=0;pos<nt+ngen;pos++){
        int token;if(pos<nt){if(!rank)token=ts[pos];MPI_Bcast(&token,1,MPI_INT,0,MPI_COMM_WORLD);}else token=cur;
        if(pos>=nt){
            if(rank==nr-1&&pending>=0){ok+=pending==token;all++;}
            if(!rank){const char*s=bpe_token_to_str(v,token);if(s)fputs(s,stdout);fflush(stdout);}
            if(token==v->eos_id||token==v->eot_id)break;
        }
        if(!rank)transformer_embed_token(m,token);else MPI_Recv(h,m->n_embd,MPI_FLOAT,rank-1,100,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        transformer_forward_partial(m,pos,l0,l1);if(rank<nr-1)MPI_Send(h,m->n_embd,MPI_FLOAT,rank+1,100,MPI_COMM_WORLD);
        else{cur=qargmax(transformer_compute_logits(m),m->n_vocab);pending=-1;if(sk){const float*dh=h;int prev=token;for(int k=0;k<sk;k++){int d=qargmax(transformer_nextn_logits(m,prev,dh,pos+k),m->n_vocab);if(!k)pending=d;prev=d;dh=transformer_nextn_hidden(m);}}}
        MPI_Bcast(&cur,1,MPI_INT,nr-1,MPI_COMM_WORLD);
    }
    long gok=0,gall=0;MPI_Reduce(&ok,&gok,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);MPI_Reduce(&all,&gall,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);if(!rank){fputc('\n',stdout);fprintf(stderr,"qwen38-pp elapsed=%.3fs",qtime()-t0);if(gall)fprintf(stderr," mtp_greedy_match=%ld/%ld alpha=%.4f",gok,gall,(double)gok/gall);fputc('\n',stderr);}
    free(ts);free(c);transformer_free(m);bpe_vocab_free(v);gguf_close(g);MPI_Finalize();return 0;
}
