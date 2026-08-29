/* Complete decode-front + head-parallel sparse layer integration check. */
#define main glm53f_sparse_core_standalone_main
#include "glm53f_sparse_core_12n.c"
#undef main
#include "glm53f_sparse_12n.h"

enum { QA=1536,IH=32,ID=128,KPOOL=4,TOPK=2048 };
static void b16dot8(float*y,const uint16_t*w,const float*x,int n){svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0),a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);int vl=(int)svcntw();for(int i=0;i<n;i+=vl){svbool_t p=svwhilelt_b32(i,n);svfloat32_t xv=svld1(p,x+i);
#define R(N,A) do{svuint32_t z=svlsl_n_u32_x(p,svld1uh_u32(p,w+(size_t)(N)*n+i),16);A=svmla_x(p,A,svreinterpret_f32_u32(z),xv);}while(0)
R(0,a0);R(1,a1);R(2,a2);R(3,a3);R(4,a4);R(5,a5);R(6,a6);R(7,a7);
#undef R
    }svbool_t p=svptrue_b32();y[0]=svaddv_f32(p,a0);y[1]=svaddv_f32(p,a1);y[2]=svaddv_f32(p,a2);y[3]=svaddv_f32(p,a3);y[4]=svaddv_f32(p,a4);y[5]=svaddv_f32(p,a5);y[6]=svaddv_f32(p,a6);y[7]=svaddv_f32(p,a7);}
static void mv_b16(float*y,const uint16_t*w,const float*x,int rows,int cols){
    int nb=rows/8;
#pragma omp parallel for schedule(static)
    for(int b=0;b<nb;b++)b16dot8(y+b*8,w+(size_t)b*8*cols,x,cols);
#pragma omp parallel for schedule(static)
    for(int r=nb*8;r<rows;r++)y[r]=bf16dot(w+(size_t)r*cols,x,cols);
}
static void f8dot8(float*y,const uint8_t*w,const float*s,const float*x,int n){svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0),a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);int vl=(int)svcntw();for(int b=0;b<n;b+=128){int e=b+128<n?b+128:n;for(int i=b;i<e;i+=vl){svbool_t p=svwhilelt_b32(i,e);svfloat32_t xv=svmul_n_f32_x(p,svld1(p,x+i),s[b/128]);
#define R(N,A) A=svmla_x(p,A,glm53f_fp8_e4m3_bits(p,w+(size_t)(N)*n,i),xv)
R(0,a0);R(1,a1);R(2,a2);R(3,a3);R(4,a4);R(5,a5);R(6,a6);R(7,a7);
#undef R
    }}svbool_t p=svptrue_b32();y[0]=svaddv_f32(p,a0);y[1]=svaddv_f32(p,a1);y[2]=svaddv_f32(p,a2);y[3]=svaddv_f32(p,a3);y[4]=svaddv_f32(p,a4);y[5]=svaddv_f32(p,a5);y[6]=svaddv_f32(p,a6);y[7]=svaddv_f32(p,a7);}
static void mv_f8(float*y,const uint8_t*w,const float*s,const float*x,int rows,int cols){int sb=cols/128,nb=rows/8;
#pragma omp parallel for schedule(static)
    for(int b=0;b<nb;b++)f8dot8(y+b*8,w+(size_t)b*8*cols,s+(size_t)(b*8/128)*sb,x,cols);
#pragma omp parallel for schedule(static)
    for(int r=nb*8;r<rows;r++)y[r]=fp8dot(w+(size_t)r*cols,s+(size_t)(r/128)*sb,x,cols);
}
static void *tensor(glm53f_st_context*st,const char*n,int rank){const st_tensor_info*t=glm53f_st_find(st,n,NULL);if(!t){fprintf(stderr,"missing %s\n",n);MPI_Abort(MPI_COMM_WORLD,2);}void*p=a256(t->nbytes);readp(st,n,0,p,t->nbytes,rank);return p;}
#ifndef GLM53F_SPARSE_NO_MAIN
typedef struct{float score;int id;} pool_score;
static int pool_cmp(const void*a,const void*b){const pool_score*x=a,*y=b;if(x->score>y->score)return -1;if(x->score<y->score)return 1;return x->id<y->id?-1:x->id>y->id;}
static int select_cached(const float*pool,int*selected,const float*q,const float*hw,
                         int tokens){int np=tokens/KPOOL,nc=TOPK/KPOOL;if(nc>np)nc=np;pool_score*score=a256((size_t)(np?np:1)*sizeof(*score));int out=0;
#pragma omp parallel for schedule(static)
    for(int p=0;p<np;p++){score[p].score=0;score[p].id=p;for(int h=0;h<IH;h++){float z=f32dot(q+(size_t)h*ID,pool+(size_t)p*ID,ID)/sqrtf((float)ID);if(z>0)score[p].score+=hw[h]*z/sqrtf((float)IH);}}qsort(score,np,sizeof(*score),pool_cmp);for(int i=0;i<nc;i++)for(int z=0;z<KPOOL;z++)selected[out++]=score[i].id*KPOOL+z;for(int z=np*KPOOL;z<tokens;z++)selected[out++]=z;free(score);return out;}
#endif

struct glm53f_sparse_context_12n {
    int rank,h0,hn,qd,capacity,length;
    uint8_t *qa,*qb,*kva,*op;
    float *qas,*qbs,*kvas,*ops;
    uint16_t *qan,*kvan,*kvb,*wk,*knw,*knb,*gatew,*ape,*wqb,*wp;
    float *qres,*query,*latent,*key,*gcache,*iq,*iw,*pool,*attn,*partial,*apef;
    int *selected;
};

glm53f_sparse_context_12n*glm53f_sparse_create_12n(const char*model,int layer,int capacity){int rank,nr,h0,hn,qd;char n[256];glm53f_st_context*st;glm53f_sparse_context_12n*c;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(nr!=12||capacity<1)return NULL;glm53f_balanced_slice(NH,rank,nr,&h0,&hn);qd=hn*KD;st=glm53f_st_open(model);if(!st)return NULL;c=calloc(1,sizeof(*c));if(!c)MPI_Abort(MPI_COMM_WORLD,2);c->rank=rank;c->h0=h0;c->hn=hn;c->qd=qd;c->capacity=capacity;
#define N(S) snprintf(n,sizeof n,"model.language_model.layers.%d.self_attn.%s",layer,S)
    N("q_a_proj.weight");c->qa=tensor(st,n,rank);N("q_a_proj.weight_scale_inv");c->qas=tensor(st,n,rank);N("q_a_layernorm.weight");c->qan=tensor(st,n,rank);N("q_b_proj.weight");c->qb=a256((size_t)qd*QA);readp(st,n,(size_t)h0*KD*QA,c->qb,(size_t)qd*QA,rank);N("q_b_proj.weight_scale_inv");{int nb=QA/128,tiles=qd/128;c->qbs=a256((size_t)tiles*nb*4);readp(st,n,(size_t)(h0*KD/128)*nb*4,c->qbs,(size_t)tiles*nb*4,rank);}N("kv_a_proj_with_mqa.weight");c->kva=tensor(st,n,rank);N("kv_a_proj_with_mqa.weight_scale_inv");c->kvas=tensor(st,n,rank);N("kv_a_layernorm.weight");c->kvan=tensor(st,n,rank);N("kv_b_proj.weight");c->kvb=a256((size_t)hn*(KD+VD)*LAT*2);readp(st,n,(size_t)h0*(KD+VD)*LAT*2,c->kvb,(size_t)hn*(KD+VD)*LAT*2,rank);{int oc0=h0*VD,ocn=hn*VD,ob=QKV/128,ob0=oc0/128,obn=ocn/128;N("o_proj.weight");c->op=a256((size_t)H*ocn);if(glm53f_st_read_columns(st,n,QKV,oc0,ocn,c->op))MPI_Abort(MPI_COMM_WORLD,2);N("o_proj.weight_scale_inv");c->ops=a256((size_t)(H/128)*obn*sizeof(float));if(glm53f_st_read_columns(st,n,(size_t)ob*sizeof(float),(size_t)ob0*sizeof(float),(size_t)obn*sizeof(float),c->ops))MPI_Abort(MPI_COMM_WORLD,2);}N("indexer.wk.weight");c->wk=tensor(st,n,rank);N("indexer.k_norm.weight");c->knw=tensor(st,n,rank);N("indexer.k_norm.bias");c->knb=tensor(st,n,rank);N("indexer.index_kpool_compress_gate");c->gatew=tensor(st,n,rank);N("indexer.index_kpool_compress_ape");c->ape=tensor(st,n,rank);N("indexer.wq_b.weight");c->wqb=tensor(st,n,rank);N("indexer.weights_proj.weight");c->wp=tensor(st,n,rank);
#undef N
    glm53f_st_close(st);c->qres=a256(QA*4);c->query=a256((size_t)qd*4);c->latent=a256((size_t)capacity*LAT*4);c->key=a256((size_t)capacity*ID*4);c->gcache=a256((size_t)capacity*ID*4);c->iq=a256((size_t)IH*ID*4);c->iw=a256(IH*4);c->pool=a256((size_t)(capacity/KPOOL+1)*ID*4);c->attn=a256((size_t)hn*VD*4);c->partial=a256(H*4);c->selected=a256((size_t)(TOPK+KPOOL)*sizeof(int));c->apef=a256(KPOOL*ID*4);for(int i=0;i<KPOOL*ID;i++)c->apef[i]=glm53f_bf16_to_f32(c->ape[i]);return c;}
void glm53f_sparse_reset_12n(glm53f_sparse_context_12n*c){if(c)c->length=0;}
int glm53f_sparse_length_12n(const glm53f_sparse_context_12n*c){return c?c->length:-1;}
int glm53f_sparse_sublayer_12n(void*context,float*out,const float*x){glm53f_sparse_context_12n*c=context;if(!c||c->length>=c->capacity)return-1;int pos=c->length,tokens=pos+1,local_cols=c->hn*VD,local_blocks=local_cols/128;mv_f8(c->qres,c->qa,c->qas,x,QA,H);glm53f_rmsnorm_bf16(c->qres,c->qres,c->qan,QA,1e-5f);mv_f8(c->query,c->qb,c->qbs,c->qres,c->qd,QA);mv_f8(c->latent+(size_t)pos*LAT,c->kva,c->kvas,x,LAT,H);glm53f_rmsnorm_bf16(c->latent+(size_t)pos*LAT,c->latent+(size_t)pos*LAT,c->kvan,LAT,1e-5f);float raw[ID];mv_b16(raw,c->wk,x,ID,H);glm53f_layernorm_bf16(c->key+(size_t)pos*ID,raw,c->knw,c->knb,ID,1e-5f);mv_b16(c->gcache+(size_t)pos*ID,c->gatew,x,ID,H);mv_b16(c->iq,c->wqb,c->qres,IH*ID,QA);mv_b16(c->iw,c->wp,x,IH,H);int ns=glm53f_index_select_decode(c->pool,c->selected,c->iq,c->iw,c->key,c->gcache,c->apef,tokens,KPOOL,TOPK,IH,ID);if(ns<1||mla_heads(c->attn,c->query,c->latent,c->kvb,c->selected,ns,c->hn))return-1;
#pragma omp parallel for schedule(static)
    for(int r=0;r<H;r++)c->partial[r]=fp8dot(c->op+(size_t)r*local_cols,c->ops+(size_t)(r/128)*local_blocks,c->attn,local_cols);if(MPI_Allreduce(c->partial,out,H,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD)!=MPI_SUCCESS)return-1;c->length++;return 0;}
void glm53f_sparse_free_12n(glm53f_sparse_context_12n*c){if(!c)return;free(c->apef);free(c->selected);free(c->partial);free(c->attn);free(c->pool);free(c->iw);free(c->iq);free(c->gcache);free(c->key);free(c->latent);free(c->query);free(c->qres);free(c->wp);free(c->wqb);free(c->ape);free(c->gatew);free(c->knb);free(c->knw);free(c->wk);free(c->ops);free(c->op);free(c->kvb);free(c->kvan);free(c->kvas);free(c->kva);free(c->qbs);free(c->qb);free(c->qan);free(c->qas);free(c->qa);free(c);}
#ifndef GLM53F_SPARSE_NO_MAIN
int main(int argc,char**argv){
    int rank,nr,layer=argc>3?atoi(argv[3]):43,tokens=argc>2?atoi(argv[2]):512,h0,hn,qd;char n[256];glm53f_st_context*st;
    uint8_t *qa,*qb,*kva,*op;float *qas,*qbs,*kvas,*ops;uint16_t *qan,*kvan,*kvb,*wk,*knw,*knb,*gatew,*ape,*wqb,*wp;
    float *x,*qres,*query,*latent,*key,*gcache,*iq,*iw,*pool,*attn,*partial,*out[2];int*sel;MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);
    if(argc<2||nr!=12||tokens<1||tokens>4096){if(!rank)fprintf(stderr,"usage: mpiexec -np 12 %s MODEL_DIR [tokens=512] [layer=43]\n",argv[0]);MPI_Finalize();return 2;}glm53f_balanced_slice(NH,rank,nr,&h0,&hn);qd=hn*KD;st=glm53f_st_open(argv[1]);if(!st)MPI_Abort(MPI_COMM_WORLD,2);
#define N(S) snprintf(n,sizeof n,"model.language_model.layers.%d.self_attn.%s",layer,S)
    N("q_a_proj.weight");qa=tensor(st,n,rank);N("q_a_proj.weight_scale_inv");qas=tensor(st,n,rank);N("q_a_layernorm.weight");qan=tensor(st,n,rank);
    N("q_b_proj.weight");qb=a256((size_t)qd*QA);readp(st,n,(size_t)h0*KD*QA,qb,(size_t)qd*QA,rank);N("q_b_proj.weight_scale_inv");{int nb=QA/128,tiles=qd/128;qbs=a256((size_t)tiles*nb*4);readp(st,n,(size_t)(h0*KD/128)*nb*4,qbs,(size_t)tiles*nb*4,rank);}
    N("kv_a_proj_with_mqa.weight");kva=tensor(st,n,rank);N("kv_a_proj_with_mqa.weight_scale_inv");kvas=tensor(st,n,rank);N("kv_a_layernorm.weight");kvan=tensor(st,n,rank);
    N("kv_b_proj.weight");kvb=a256((size_t)hn*(KD+VD)*LAT*2);readp(st,n,(size_t)h0*(KD+VD)*LAT*2,kvb,(size_t)hn*(KD+VD)*LAT*2,rank);
    {int oc0=h0*VD,ocn=hn*VD,ob=QKV/128,ob0=oc0/128,obn=ocn/128;
     N("o_proj.weight");op=a256((size_t)H*ocn);if(glm53f_st_read_columns(st,n,QKV,oc0,ocn,op))MPI_Abort(MPI_COMM_WORLD,2);
     N("o_proj.weight_scale_inv");ops=a256((size_t)(H/128)*obn*sizeof(float));if(glm53f_st_read_columns(st,n,(size_t)ob*sizeof(float),(size_t)ob0*sizeof(float),(size_t)obn*sizeof(float),ops))MPI_Abort(MPI_COMM_WORLD,2);}
    N("indexer.wk.weight");wk=tensor(st,n,rank);N("indexer.k_norm.weight");knw=tensor(st,n,rank);N("indexer.k_norm.bias");knb=tensor(st,n,rank);N("indexer.index_kpool_compress_gate");gatew=tensor(st,n,rank);N("indexer.index_kpool_compress_ape");ape=tensor(st,n,rank);N("indexer.wq_b.weight");wqb=tensor(st,n,rank);N("indexer.weights_proj.weight");wp=tensor(st,n,rank);
#undef N
    glm53f_st_close(st);x=a256(H*4);qres=a256(QA*4);query=a256((size_t)qd*4);latent=a256((size_t)tokens*LAT*4);key=a256((size_t)tokens*ID*4);gcache=a256((size_t)tokens*ID*4);iq=a256((size_t)IH*ID*4);iw=a256(IH*4);pool=a256((size_t)(tokens/KPOOL+1)*ID*4);attn=a256((size_t)hn*VD*4);partial=a256(H*4);out[0]=a256(H*4);out[1]=a256(H*4);sel=a256((size_t)(TOPK+KPOOL)*4);
    for(int t=0;t<tokens;t++){for(int d=0;d<LAT;d++)latent[(size_t)t*LAT+d]=(float)(((t*29+d*11+3)%257)-128)/128.0f;for(int d=0;d<ID;d++){key[(size_t)t*ID+d]=(float)(((t*13+d*17+5)%251)-125)/125.0f;gcache[(size_t)t*ID+d]=(float)(((t*7+d*19+1)%127)-63)/63.0f;}}
    for(int i=0;i<H;i++)x[i]=(float)(((i*17+tokens*3+3)%251)-125)/125.0f;float apef[KPOOL*ID];for(int i=0;i<KPOOL*ID;i++)apef[i]=glm53f_bf16_to_f32(ape[i]);float ph[2][4],el[2];int ns[2];int local_cols=hn*VD,local_blocks=local_cols/128;
    /* Build the persistent cache state once. Decode only refreshes the current
     * token and scores already-compressed completed pools. */
    mv_f8(qres,qa,qas,x,QA,H);glm53f_rmsnorm_bf16(qres,qres,qan,QA,1e-5f);mv_f8(latent+(size_t)(tokens-1)*LAT,kva,kvas,x,LAT,H);glm53f_rmsnorm_bf16(latent+(size_t)(tokens-1)*LAT,latent+(size_t)(tokens-1)*LAT,kvan,LAT,1e-5f);{float raw[ID];mv_b16(raw,wk,x,ID,H);glm53f_layernorm_bf16(key+(size_t)(tokens-1)*ID,raw,knw,knb,ID,1e-5f);}mv_b16(gcache+(size_t)(tokens-1)*ID,gatew,x,ID,H);mv_b16(iq,wqb,qres,IH*ID,QA);mv_b16(iw,wp,x,IH,H);glm53f_index_select_decode(pool,sel,iq,iw,key,gcache,apef,tokens,KPOOL,TOPK,IH,ID);
    for(int pass=0;pass<2;pass++){double t0=MPI_Wtime();mv_f8(qres,qa,qas,x,QA,H);glm53f_rmsnorm_bf16(qres,qres,qan,QA,1e-5f);mv_f8(query,qb,qbs,qres,qd,QA);mv_f8(latent+(size_t)(tokens-1)*LAT,kva,kvas,x,LAT,H);glm53f_rmsnorm_bf16(latent+(size_t)(tokens-1)*LAT,latent+(size_t)(tokens-1)*LAT,kvan,LAT,1e-5f);float raw[ID];mv_b16(raw,wk,x,ID,H);glm53f_layernorm_bf16(key+(size_t)(tokens-1)*ID,raw,knw,knb,ID,1e-5f);mv_b16(gcache+(size_t)(tokens-1)*ID,gatew,x,ID,H);mv_b16(iq,wqb,qres,IH*ID,QA);mv_b16(iw,wp,x,IH,H);ns[pass]=select_cached(pool,sel,iq,iw,tokens);double t1=MPI_Wtime();if(mla_heads(attn,query,latent,kvb,sel,ns[pass],hn))MPI_Abort(MPI_COMM_WORLD,2);double t2=MPI_Wtime();
#pragma omp parallel for schedule(static)
        for(int r=0;r<H;r++)partial[r]=fp8dot(op+(size_t)r*local_cols,ops+(size_t)(r/128)*local_blocks,attn,local_cols);double t3=MPI_Wtime();MPI_Allreduce(partial,out[pass],H,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);double t4=MPI_Wtime();ph[pass][0]=t1-t0;ph[pass][1]=t2-t1;ph[pass][2]=t3-t2;ph[pass][3]=t4-t3;el[pass]=t4-t0;}
    int ok=ns[0]==ns[1]&&!memcmp(out[0],out[1],H*4),all;MPI_Allreduce(&ok,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);float me,mp[4];MPI_Allreduce(&el[1],&me,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);MPI_Allreduce(ph[1],mp,4,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);double ss=0;for(int i=0;i<H;i++)ss+=(double)out[0][i]*out[0][i];if(!rank)printf("GLM53F_SPARSE_LAYER_12N layer=%d tokens=%d selected=%d max_ms=%.3f front_ms=%.3f mla_ms=%.3f oproj_ms=%.3f ar_ms=%.3f rms=%.9g repeat=%s %s\n",layer,tokens,ns[0],me*1e3f,mp[0]*1e3f,mp[1]*1e3f,mp[2]*1e3f,mp[3]*1e3f,sqrt(ss/H),all?"BIT_EXACT":"FAIL",all?"PASS":"FAIL");MPI_Finalize();return all?0:1;
}
#endif
