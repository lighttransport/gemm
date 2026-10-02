/* Original inference kernels for vhuman triangle-bound Gaussian assets.
 * Pinhole/EWA and alpha conventions match the pinned gsplat reference.
 * No gsplat, Torch, CUB, cuBLAS or other inference library is linked. */
#pragma once
static const char vh_splat_source[] = R"CUDA(
__device__ float bound(float x,float a,float b){return fminf(b,fmaxf(a,x));}
__device__ void mul3(const float *a,const float *b,float *c){
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){
        float v=0;for(int k=0;k<3;k++)v+=a[3*i+k]*b[3*k+j];c[3*i+j]=v;
    }
}
__device__ void congruence(const float *a,const float *b,float *c){
    float d[9];mul3(a,b,d);
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){
        float v=0;for(int k=0;k<3;k++)v+=d[3*i+k]*a[3*j+k];c[3*i+j]=v;
    }
}
__device__ void clamp_eigen(float *a){
    float u[9]={1,0,0,0,1,0,0,0,1};
    for(int sweep=0;sweep<12;sweep++)for(int p=0;p<2;p++)for(int q=p+1;q<3;q++){
        float off=a[3*p+q];if(fabsf(off)<1.e-20f)continue;
        float tau=(a[3*q+q]-a[3*p+p])/(2*off);
        float t=copysignf(1.f,tau)/(fabsf(tau)+sqrtf(1+tau*tau));
        float c=rsqrtf(1+t*t),s=t*c;
        float app=a[3*p+p],aqq=a[3*q+q];
        a[3*p+p]=app-t*off;a[3*q+q]=aqq+t*off;a[3*p+q]=a[3*q+p]=0;
        for(int k=0;k<3;k++)if(k!=p&&k!=q){
            float kp=a[3*k+p],kq=a[3*k+q];
            a[3*k+p]=a[3*p+k]=c*kp-s*kq;a[3*k+q]=a[3*q+k]=s*kp+c*kq;
        }
        for(int k=0;k<3;k++){
            float kp=u[3*k+p],kq=u[3*k+q];u[3*k+p]=c*kp-s*kq;u[3*k+q]=s*kp+c*kq;
        }
    }
    float d[3]={bound(a[0],1.e-10f,1.e-4f),bound(a[4],1.e-10f,1.e-4f),bound(a[8],1.e-10f,1.e-4f)};
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){
        float v=0;for(int k=0;k<3;k++)v+=u[3*i+k]*d[k]*u[3*j+k];a[3*i+j]=v;
    }
}
// Packed asset row: bary[3], offset, covariance[9], opacity, rgb[3], color_basis[24].
// Geometry row: center[3], covariance[9], opacity, rgb[3].
extern "C" __global__ void deform(const float *vertices,const int *attachments,
    const float *assets,const float *coeff,int n,int trace,float *geometry){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const float *a=assets+41*i;float *g=geometry+16*i;
    float p[9];for(int k=0;k<3;k++)for(int j=0;j<3;j++)p[3*k+j]=vertices[3*attachments[3*i+k]+j];
    float b[9],normal[3];for(int j=0;j<3;j++){b[3*j]=p[3+j]-p[j];b[3*j+1]=p[6+j]-p[j];}
    normal[0]=b[3]*b[7]-b[6]*b[4];normal[1]=b[6]*b[1]-b[0]*b[7];normal[2]=b[0]*b[4]-b[3]*b[1];
    float len=sqrtf(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
    for(int j=0;j<3;j++){
        b[3*j+2]=normal[j]/fmaxf(len,1.e-12f);
        g[j]=p[j]*a[0]+p[3+j]*a[1]+p[6+j]*a[2]+b[3*j+2]*a[3];
    }
    congruence(b,a+4,g+3);
    if(trace){
        g[3]+=1.e-10f;g[7]+=1.e-10f;g[11]+=1.e-10f;
        float scale=fminf(1.f,1.e-4f/fmaxf(g[3]+g[7]+g[11],1.e-10f));
        for(int j=0;j<9;j++)g[3+j]*=scale;
    }else clamp_eigen(g+3);
    g[12]=len>1.e-10f?a[13]:0;
    for(int j=0;j<3;j++){
        float v=a[14+j];for(int k=0;k<8;k++)v+=a[17+3*k+j]*coeff[k];
        g[13+j]=trace?bound(v,0,1):fmaxf(0,v);
    }
}
// Projection row: mean[2], depth, conic[3], opacity, rgb[3], radius[2].
extern "C" __global__ void project(const float *geometry,const float *camera,int n,int w,int h,float *proj){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const float *g=geometry+16*i;float *o=proj+12*i;for(int k=0;k<12;k++)o[k]=0;
    float r[9],p[3],c[9];
    for(int j=0;j<3;j++){
        p[j]=camera[4*j+3];for(int k=0;k<3;k++){r[3*j+k]=camera[4*j+k];p[j]+=r[3*j+k]*g[k];}
    }
    if(p[2]<.01f||p[2]>1.e10f||g[12]<1.f/255.f)return;
    congruence(r,g+3,c);
    float fx=camera[16],fy=camera[20],cx=camera[18],cy=camera[21],z=p[2];
    float tx=bound(p[0]/z,-(cx+.15f*w)/fx,(w-cx+.15f*w)/fx);
    float ty=bound(p[1]/z,-(cy+.15f*h)/fy,(h-cy+.15f*h)/fy);
    float j[6]={fx/z,0,-fx*tx/z,0,fy/z,-fy*ty/z};
    float s[4]={0,0,0,0};
    for(int u=0;u<2;u++)for(int v=0;v<2;v++)for(int k=0;k<3;k++)for(int l=0;l<3;l++)
        s[u*2+v]+=j[u*3+k]*c[k*3+l]*j[v*3+l];
    s[0]+=.3f;s[3]+=.3f;float det=s[0]*s[3]-s[1]*s[2];if(det<=0||!isfinite(det))return;
    float extent=fminf(3.33f,sqrtf(2*logf(g[12]*255.f)));
    float rx=ceilf(extent*sqrtf(s[0])),ry=ceilf(extent*sqrtf(s[3]));
    float x=fx*p[0]/z+cx,y=fy*p[1]/z+cy;
    if(!isfinite(x)||!isfinite(y)||!isfinite(z)||!isfinite(rx)||!isfinite(ry))return;
    if(x+rx<=0||x-rx>=w||y+ry<=0||y-ry>=h)return;
    o[0]=x;o[1]=y;o[2]=z;o[3]=s[3]/det;o[4]=-s[1]/det;o[5]=s[0]/det;o[6]=g[12];
    for(int k=0;k<3;k++)o[7+k]=g[13+k];o[10]=rx;o[11]=ry;
}
extern "C" __global__ void raster(const float *proj,const int *offsets,const int *ids,int w,int h,float *rgba){
    int tile=blockIdx.x,tw=(w+15)/16,x=(tile%tw)*16+threadIdx.x%16,y=(tile/tw)*16+threadIdx.x/16;
    bool done=x>=w||y>=h;float trans=1,r=0,g=0,b=0;
    __shared__ float batch[256][10];
    int begin=offsets[tile],end=offsets[tile+1];
    for(int base=begin;base<end;base+=256){
        if(__syncthreads_count(done)==256)break;
        int index=base+threadIdx.x;
        if(index<end){const float *p=proj+12*ids[index];for(int k=0;k<10;k++)batch[threadIdx.x][k]=p[k];}
        __syncthreads();
        if(!done)for(int t=0;t<min(256,end-base);t++){
            const float *p=batch[t];float dx=p[0]-(x+.5f),dy=p[1]-(y+.5f);
            float sigma=.5f*(p[3]*dx*dx+p[5]*dy*dy)+p[4]*dx*dy;
            float a=fminf(.99f,p[6]*expf(-sigma));if(sigma<0||a<1.f/255.f)continue;
            float next=trans*(1-a);if(next<=1.e-4f){done=true;break;}
            float weight=a*trans;r+=weight*p[7];g+=weight*p[8];b+=weight*p[9];trans=next;
        }
        __syncthreads();
    }
    if(x<w&&y<h){int i=4*(y*w+x);rgba[i]=r;rgba[i+1]=g;rgba[i+2]=b;rgba[i+3]=1-trans;}
}
)CUDA";
