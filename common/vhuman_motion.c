/* Stateful two-layer PyTorch-compatible GRU evaluation, no framework runtime. */
#define SWIN_NO_MAIN
#include "swin_runner.c"

typedef struct {
    st_context *s;
    int hidden,controls;
    const float *mean,*scale,*bounds,*embedding,*projection_w,*projection_b,*output_w,*output_b;
    const float *iw[2],*hw[2],*ib[2],*hb[2];
    float state[256];
    float *normalized,*output;
} vh_motion;

static const float *motion_weight(st_context *s,const char *name,int rank,int a,int b)
{
    int i=safetensors_find(s,name);
    if(i<0 || strcmp(safetensors_dtype(s,i),"F32") || safetensors_ndims(s,i)!=rank ||
       safetensors_shape(s,i)[0]!=(uint64_t)a || (rank==2 && safetensors_shape(s,i)[1]!=(uint64_t)b) ||
       safetensors_nbytes(s,i)!=(size_t)a*(rank==2?b:1)*sizeof(float)) return NULL;
    const float *p=safetensors_data(s,i);
    for(size_t j=0;j<(size_t)a*(rank==2?b:1);j++)if(!isfinite(p[j]))return NULL;
    return p;
}
void vh_motion_close(vh_motion *m)
{ if(m){if(m->s)safetensors_close(m->s);free(m->normalized);free(m->output);free(m);} }
vh_motion *vh_motion_open(const char *path,int hidden,int controls)
{
    if(!path || hidden<1 || hidden>16384 || controls<1 || controls>512)return NULL;
    vh_motion *m=calloc(1,sizeof(*m));if(!m)return NULL;
    m->s=safetensors_open(path);if(!m->s)goto bad;
    m->hidden=hidden;m->controls=controls;
#define MW(dst,name,rank,a,b) do {m->dst=motion_weight(m->s,name,rank,a,b);if(!m->dst)goto bad;}while(0)
    MW(mean,"hidden_mean",1,hidden,0);MW(scale,"hidden_scale",1,hidden,0);
    MW(bounds,"bounds",2,controls,2);MW(embedding,"code_embedding.weight",2,2048,16);
    MW(projection_w,"hidden_projection.weight",2,128,hidden);MW(projection_b,"hidden_projection.bias",1,128,0);
    MW(output_w,"output.weight",2,8*controls,128);MW(output_b,"output.bias",1,8*controls,0);
    for(int l=0;l<2;l++) {
        char name[80];snprintf(name,sizeof(name),"recurrent.weight_ih_l%d",l);MW(iw[l],name,2,384,l?128:384);
        snprintf(name,sizeof(name),"recurrent.weight_hh_l%d",l);MW(hw[l],name,2,384,128);
        snprintf(name,sizeof(name),"recurrent.bias_ih_l%d",l);MW(ib[l],name,1,384,0);
        snprintf(name,sizeof(name),"recurrent.bias_hh_l%d",l);MW(hb[l],name,1,384,0);
    }
#undef MW
    for(int i=0;i<hidden;i++)if(m->scale[i]<=0)goto bad;
    for(int i=0;i<controls;i++)if(m->bounds[2*i]>m->bounds[2*i+1])goto bad;
    m->normalized=malloc((size_t)hidden*4);m->output=malloc((size_t)8*controls*4);
    if(!m->normalized || !m->output)goto bad;
    return m;
bad:
    vh_motion_close(m);return NULL;
}
void vh_motion_reset(vh_motion *m) { if(m)memset(m->state,0,sizeof(m->state)); }
static float motion_sigmoid(float x)
{return x>=0?1/(1+expf(-x)):expf(x)/(1+expf(x));}
int vh_motion_step(vh_motion *m,const float *hidden,const int32_t *codes,float *output)
{
    if(!m || !hidden || !codes || !output)return -1;
    for(int i=0;i<m->hidden;i++) {
        if(!isfinite(hidden[i]))return -1;
        m->normalized[i]=(hidden[i]-m->mean[i])/m->scale[i];
        if(!isfinite(m->normalized[i]))return -1;
    }
    for(int i=0;i<16;i++)if(codes[i]<0 || codes[i]>=2048)return -1;
    /* Small streaming GEMVs: no thread pool fanout per audio feature packet. */
    omp_set_num_threads(1);
    float input[384],state[256],gi[384],gh[384];memcpy(state,m->state,sizeof(state));
    swin_linear(input,m->projection_w,m->projection_b,m->normalized,1,128,m->hidden);
    for(int i=0;i<16;i++)memcpy(input+128+16*i,m->embedding+16*codes[i],16*4);
    for(int l=0;l<2;l++) {
        swin_linear(gi,m->iw[l],m->ib[l],l?state:input,1,384,l?128:384);
        swin_linear(gh,m->hw[l],m->hb[l],m->state+128*l,1,384,128);
        for(int i=0;i<128;i++) {
            float r=motion_sigmoid(gi[i]+gh[i]),z=motion_sigmoid(gi[128+i]+gh[128+i]);
            float candidate=tanhf(gi[256+i]+r*gh[256+i]);
            state[l*128+i]=(1-z)*candidate+z*m->state[l*128+i];
            if(!isfinite(state[l*128+i]))return -1;
        }
    }
    swin_linear(m->output,m->output_w,m->output_b,state+128,1,8*m->controls,128);
    for(int i=0;i<8*m->controls;i++) {
        int c=i%m->controls;float lo=m->bounds[c*2],hi=m->bounds[c*2+1];
        if(!isfinite(m->output[i]))return -1;
        m->output[i]=lo+motion_sigmoid(m->output[i])*(hi-lo);
    }
    memcpy(output,m->output,(size_t)8*m->controls*4);memcpy(m->state,state,sizeof(state));return 0;
}
