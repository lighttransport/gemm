#include "ds41f_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
static void near(float a,float b){if(!isfinite(a)||fabsf(a-b)>1e-5f){fprintf(stderr,"FAIL got=%g expected=%g\n",a,b);exit(1);}}
int main(void)
{
    float logits[4]={0,0,0,0},bias[4]={0,1,0,2},weights[2];int ids[2];
    if(ds41f_gate(logits,bias,4,2,1,1.5,ids,weights))return 1;
    if(ids[0]!=3||ids[1]!=1)return 1;
    near(weights[0],.75);near(weights[1],.75);
    float mix[24]={0},base[24]={0},scale[3]={1,1,1},pre[4],post[4],comb[16];
    ds41f_hc_split(mix,scale,base,20,1e-6,pre,post,comb);
    for(int i=0;i<4;++i){near(pre[i],.500001);near(post[i],1);for(int j=0;j<4;++j)near(comb[i*4+j],.25);}
    float x[8]={1,2,3,4,5,6,7,8},out[8],y[2]={10,20};
    for(int i=0;i<16;++i)comb[i]=0;
    comb[1]=1;comb[6]=1;comb[11]=1;comb[12]=1;
    ds41f_hc_post(out,y,x,post,comb,2);near(out[0],17);near(out[2],11);near(out[4],13);near(out[6],15);
    /* A small residual must survive cancellation with the sublayer output.
     * Exercise all output streams and the in-place residual update. */
    float cancellation[4]={1,-16777216.f,0,0},large[1]={16777216.f};
    for(int i=0;i<16;++i)comb[i]=1;
    ds41f_hc_post(out,large,cancellation,post,comb,1);
    for(int i=0;i<4;++i)near(out[i],1);
    ds41f_hc_post(cancellation,large,cancellation,post,comb,1);
    for(int i=0;i<4;++i)near(cancellation[i],1);
    float products[4]={-1,1-0x1p-13f,0,0},zero[1]={0};
    for(int i=0;i<4;++i)comb[4+i]=1+0x1p-13f;
    ds41f_hc_post(out,zero,products,post,comb,1);
    for(int i=0;i<4;++i)if(out[i]!=0){fprintf(stderr,"FAIL mHC product rounding: %g\n",out[i]);return 1;}
    float q[2]={0,0},kv[4]={2,4,6,8},sink[1]={0};int sel[2]={0,1};
    if(ds41f_sparse_attention(out,q,kv,sink,sel,2,2,1,2))return 1;
    near(out[0],8.f/3);near(out[1],4);
    sel[1]=2;if(!ds41f_sparse_attention(out,q,kv,sink,sel,2,2,1,2))return 1;
    float a[4]={1,2,3,4};ds41f_rope(a,1,4,4,17,10000,16,65536,0);ds41f_rope(a,1,4,4,17,10000,16,65536,1);
    for(int i=0;i<4;++i)near(a[i],i+1);
    float g[2]={0,100},u[2]={5,100};ds41f_swiglu(out,g,u,2,10);near(out[0],0);near(out[1],100/(1+expf(-10)));
    float pa[2]={2,4},pb[2]={6,8},sa[2]={0,1000},sb[2]={0,-1000};
    ds41f_pool_pair(out,pa,pb,sa,sb,2);near(out[0],4);near(out[1],4);
    float streams[8],key[8],qw[8],kw[8],value[2]={2,4};
    for(int i=0;i<8;++i){streams[i]=1;key[i]=1;qw[i]=1;kw[i]=1;}
    ds41f_engram_fuse(streams,key,value,qw,kw,2,1e-20f);
    float eg=1/(1+expf(-sqrtf(sqrtf(2))));
    for(int i=0;i<8;++i)near(streams[i],1+eg*value[i%2]);
    puts("DS41F_OPS PASS routing mHC SwiGLU RoPE sparse_attention pooling Engram_fuse");return 0;
}
