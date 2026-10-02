#ifndef PIXAL3D_HV15N_KERNELS_HPP
#define PIXAL3D_HV15N_KERNELS_HPP
namespace hv15n {
inline const char *ops_source = R"CUDA(
#include <cuda_fp16.h>
#define INFINITY __int_as_float(0x7f800000)
extern "C" {
__global__ void element(float*y,const float*x,const float*z,const float*b,int count,int channels,int mode,float scale,float offset) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count)return;
    float v=x[i];
    if(mode==0)v=v*scale+offset;
    if(mode==1)v=v+z[i]*scale;
    if(mode==2)v=v*z[i];
    if(mode==3)v=v/(1.f+expf(-v));
    if(mode==4)v=0.5f*v*(1.f+tanhf(0.7978845608f*(v+0.044715f*v*v*v)));
    if(mode==5)v=0.5f*v*(1.f+erff(v*0.7071067812f));
    if(mode==6)v+=b[i%channels];
    if(mode==7)v=v*(1.f+z[i%channels])+b[i%channels];
    if(mode==8)v=v*z[i%channels];
    if(mode==9)v=fmaxf(v,0.f);
    y[i]=v;
}
__global__ void convert_half(half*y,const float*x,int rows,int channels,int padded) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*padded)return;
    int r=i/padded,c=i%padded; y[i]=__float2half(c<channels?x[r*channels+c]:0.f);
}
__global__ void convert_float(float*y,const half*x,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<count)y[i]=__half2float(x[i]);
}
__global__ void convert_bfloat(float*y,const unsigned short*x,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=__uint_as_float((unsigned)x[i]<<16);
}
__global__ void hv15n_norm(float*y,const float*x,const float*w,const float*b,int rows,int channels,int rms,float eps) {
    int row=blockIdx.x,tid=threadIdx.x;if(row>=rows)return;
    __shared__ float s[256],q[256];float sum=0.f,sq=0.f;
    for(int c=tid;c<channels;c+=256){float v=x[row*channels+c];sum+=v;sq+=v*v;}
    s[tid]=sum;q[tid]=sq;__syncthreads();
    for(int n=128;n;n>>=1){if(tid<n){s[tid]+=s[tid+n];q[tid]+=q[tid+n];}__syncthreads();}
    float mean=rms?0.f:s[0]/channels;
    float inverse=rms==2?sqrtf(float(channels))/fmaxf(sqrtf(q[0]),1.e-12f):rsqrtf(fmaxf(q[0]/channels-mean*mean,0.f)+eps);
    for(int c=tid;c<channels;c+=256)y[row*channels+c]=(x[row*channels+c]-mean)*inverse*(w?w[c]:1.f)+(b?b[c]:0.f);
}
__global__ void norm_silu_half(half*y,const float*x,const float*w,const float*b,int rows,int c) {
    int row=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(row>=rows)return;
    float square=0.f;for(int j=lane;j<c;j+=32){float v=x[row*c+j];square+=v*v;}
    for(int d=16;d;d>>=1)square+=__shfl_xor_sync(0xffffffff,square,d);
    float inverse=sqrtf(float(c))/fmaxf(sqrtf(square),1.e-12f);
    for(int j=lane;j<c;j+=32){float v=x[row*c+j]*inverse*w[j]+(b?b[j]:0.f);y[row*c+j]=__float2half(v/(1.f+expf(-v)));}
}
__global__ void modulate_norm(float*y,const float*x,const float*mod,int rows,int c,int start) {
    int row=blockIdx.x,tid=threadIdx.x;__shared__ float sum[256],sq[256];float a=0.f,b=0.f;
    for(int j=tid;j<c;j+=256){float v=x[row*c+j];a+=v;b+=v*v;}
    sum[tid]=a;sq[tid]=b;__syncthreads();
    for(int d=128;d;d>>=1){if(tid<d){sum[tid]+=sum[tid+d];sq[tid]+=sq[tid+d];}__syncthreads();}
    float mean=sum[0]/c,inverse=rsqrtf(fmaxf(sq[0]/c-mean*mean,0.f)+1.e-6f);
    for(int j=tid;j<c;j+=256)y[row*c+j]=(x[row*c+j]-mean)*inverse*(1.f+mod[start+c+j])+mod[start+j];
}
__global__ void gate_add(float*y,const float*x,const float*z,const float*mod,int count,int c,int start) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=x[i]+z[i]*mod[start+i%c];
}
__global__ void qkv_heads(half*q,half*k,half*v,const float*packed,const float*qw,const float*kw,
                         int rows,int height,int width,int image) {
    int row=blockIdx.x,head=blockIdx.y,d=threadIdx.x;__shared__ float qa[128],ka[128],qs[128],ks[128];
    int base=row*6144+head*128;
    float a=packed[base+d],b=packed[base+2048+d];qs[d]=a*a;ks[d]=b*b;__syncthreads();
    for(int s=64;s;s>>=1){if(d<s){qs[d]+=qs[d+s];ks[d]+=ks[d+s];}__syncthreads();}
    qa[d]=a*rsqrtf(qs[0]/128.f+1.e-6f)*qw[d];ka[d]=b*rsqrtf(ks[0]/128.f+1.e-6f)*kw[d];__syncthreads();
    if(image){int first=d&~1,axis=first<16?0:first<72?1:2,axisdim=axis==0?16:56;
        int frequency=(first-(axis==0?0:axis==1?16:72))/2;
        int position=axis==0?row/(height*width):axis==1?(row/width)%height:row%width;
        float angle=position*powf(256.f,-2.f*frequency/axisdim),c=cosf(angle),s=sinf(angle);
        a=(d&1)?qa[d]*c+qa[d-1]*s:qa[d]*c-qa[d+1]*s;
        b=(d&1)?ka[d]*c+ka[d-1]*s:ka[d]*c-ka[d+1]*s;
    }else{a=qa[d];b=ka[d];}
    int out=(head*rows+row)*128+d;q[out]=__float2half(a);k[out]=__float2half(b);v[out]=__float2half(packed[base+4096+d]);
}
__global__ void concat_heads(half*y,const half*a,const half*b,int ar,int br,int heads,int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=(ar+br)*heads*dim)return;
    int d=i%dim,r=(i/dim)%(ar+br),h=i/(dim*(ar+br));
    y[i]=r<ar?a[(h*ar+r)*dim+d]:b[(h*br+r-ar)*dim+d];
}
__global__ void columns(float*y,const float*x,int rows,int oldc,int start,int nc) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*nc)y[i]=x[(i/nc)*oldc+start+i%nc];
}
__global__ void transpose(float*y,const float*x,int rows,int channels) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*channels)y[(i%channels)*rows+i/channels]=x[i];
}
__global__ void gemm_ieee(float*y,const float*x,const float*w,int m,int n,int k) {
    __shared__ float a[16][16],b[16][16];int r=blockIdx.y*16+threadIdx.y,c=blockIdx.x*16+threadIdx.x;float s=0.f;
    for(int base=0;base<k;base+=16){int ka=base+threadIdx.x,kb=base+threadIdx.y;
        a[threadIdx.y][threadIdx.x]=(r<m&&ka<k)?x[r*k+ka]:0.f;
        b[threadIdx.y][threadIdx.x]=(c<n&&kb<k)?w[c*k+kb]:0.f;__syncthreads();
        for(int j=0;j<16;j++)s=fmaf(a[threadIdx.y][j],b[j][threadIdx.x],s);__syncthreads();}
    if(r<m&&c<n)y[r*n+c]=s;
}
// Four-by-four register outputs retain IEEE FP32 arithmetic for encoders.
__global__ void gemm_ieee_tiled(float*y,const float*x,const float*w,int m,int n,int k) {
    __shared__ float a[64][17],b[64][17];
    int tr=threadIdx.y,tc=threadIdx.x,tid=tr*16+tc;
    float sum[4][4]={};
    for(int base=0;base<k;base+=16) {
        #pragma unroll
        for(int i=0;i<4;i++) {
            int at=tid+256*i,row=at/16,col=at%16,r=blockIdx.y*64+row,c=blockIdx.x*64+row;
            a[row][col]=(r<m&&base+col<k)?x[(size_t)r*k+base+col]:0.f;
            b[row][col]=(c<n&&base+col<k)?w[(size_t)c*k+base+col]:0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int j=0;j<16;j++) {
            #pragma unroll
            for(int i=0;i<4;i++) {
                float av=a[tr+16*i][j];
                #pragma unroll
                for(int c=0;c<4;c++)sum[i][c]=fmaf(av,b[tc+16*c][j],sum[i][c]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;i++) {
        int r=blockIdx.y*64+tr+16*i;
        #pragma unroll
        for(int j=0;j<4;j++) {
            int c=blockIdx.x*64+tc+16*j;if(r<m&&c<n)y[(size_t)r*n+c]=sum[i][j];
        }
    }
}
__global__ void rope(float*y,int rows,int heads,int dim,int kind,int height,int width,float theta) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*heads*(dim/2))return;
    int p=i%(dim/2),head=(i/(dim/2))%heads,row=i/(heads*(dim/2));
    int first,second,position,frequency,axisdim;
    if(kind==0){first=p;second=p+dim/2;position=row;frequency=p;axisdim=dim;}
    else {first=p*2;second=first+1;int axis=first<16?0:(first<72?1:2);
        axisdim=axis==0?16:56;frequency=(first-(axis==0?0:(axis==1?16:72)))/2;
        position=axis==0?row/(height*width):(axis==1?(row/width)%height:row%width);}
    float angle=position*powf(theta,-2.f*frequency/axisdim),c=cosf(angle),s=sinf(angle);
    int base=(row*heads+head)*dim;float a=y[base+first],b=y[base+second];
    y[base+first]=a*c-b*s;y[base+second]=a*s+b*c;
}
__global__ void pack_heads(half*y,const float*x,int rows,int heads,int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*heads*dim)return;
    int d=i%dim,r=(i/dim)%rows,h=i/(dim*rows);y[i]=__float2half(x[(r*heads+h)*dim+d]);
}
__global__ void unpack_heads_half(float*y,const half*x,int rows,int heads,int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*heads*dim)return;
    int d=i%dim,h=(i/dim)%heads,r=i/(dim*heads);y[i]=__half2float(x[(h*rows+r)*dim+d]);
}
__global__ void join_condition(float*y,const float*x,const float*c,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*65)return;
    int r=i/65,d=i%65;y[i]=d<32?x[r*32+d]:c[r*33+d-32];
}
__global__ void pack_keys(half*y,const float*x,int rows,int heads,int dim,int start,int tile,int transpose_mode) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=heads*tile*dim)return;
    int h=i/(tile*dim),within=i%(tile*dim),r=transpose_mode?within%tile:within/dim,d=transpose_mode?within/tile:within%dim;
    y[i]=__float2half(start+r<rows?x[((start+r)*heads+h)*dim+d]:0.f);
}
__global__ void softmax_tile(float*scores,half*p,float*acc,float*maxima,float*sums,int rows,int heads,int dim,int keys,int start,int tile,int mask,int frame_hw,float scale) {
    int row=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(row>=rows*heads)return;
    int qr=row%rows;float best=-INFINITY;
    for(int j=lane;j<tile;j+=32){bool valid=j<keys&&(mask==0||(mask==1?start+j<=qr:(start+j)/frame_hw<=qr/frame_hw));
        float v=valid?scores[row*tile+j]*scale:-INFINITY;scores[row*tile+j]=v;best=fmaxf(best,v);}
    for(int d=16;d;d>>=1)best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,d));
    float previous=maxima[row],next=fmaxf(previous,best),factor=isfinite(previous)?expf(previous-next):0.f,total=0.f;
    for(int j=lane;j<tile;j+=32){float v=scores[row*tile+j];float e=isfinite(v)?expf(v-next):0.f;p[row*tile+j]=__float2half(e);total+=e;}
    for(int d=16;d;d>>=1)total+=__shfl_xor_sync(0xffffffff,total,d);
    for(int d=lane;d<dim;d+=32)acc[row*dim+d]*=factor;
    if(lane==0){maxima[row]=next;sums[row]=sums[row]*factor+total;}
}
__global__ void finish_attention(float*y,const float*x,const float*sums,int rows,int heads,int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*heads*dim)return;
    int d=i%dim,r=(i/dim)%rows,h=i/(dim*rows);float s=sums[h*rows+r];y[(r*heads+h)*dim+d]=s>0.f?x[i]/s:0.f;
}
// FP32 reference-preserving attention, including GQA and T5 relative bias.
__global__ void attention_ieee(float*y,const float*q,const float*k,const float*v,const float*bias,int rows,int heads,int kvheads,int dim,int mask,int frame_hw,float scale) {
    int qr=blockIdx.x,h=blockIdx.y,tid=threadIdx.x,kh=h/(heads/kvheads);
    __shared__ float scores[256],maximum,total;float local_max=-INFINITY;
    for(int j=tid;j<rows;j+=256){if(mask==1&&j>qr)break;if(mask==2&&j/frame_hw>qr/frame_hw)break;
        float dot=0.f;for(int d=0;d<dim;d++)dot=fmaf(q[(qr*heads+h)*dim+d],k[(j*kvheads+kh)*dim+d],dot);
        if(bias){int delta=j-qr,n=abs(delta),exact=8;int bucket=(delta>0?16:0)+(n<exact?n:min(15,exact+int(logf(float(n)/exact)/logf(16.f)*8)));
            dot=dot*scale+bias[bucket*heads+h];}else dot*=scale;local_max=fmaxf(local_max,dot);}
    scores[tid]=local_max;__syncthreads();for(int d=128;d;d>>=1){if(tid<d)scores[tid]=fmaxf(scores[tid],scores[tid+d]);__syncthreads();}
    if(tid==0){maximum=scores[0];total=0.f;}__syncthreads();
    // Process bounded key tiles. One thread owns each output channel.
    float out[8];for(int a=0;a<8;a++)out[a]=0.f;
    for(int start=0;start<rows;start+=256){int j=start+tid;float dot=-INFINITY;
        if(j<rows&&(mask==0||(mask==1?j<=qr:j/frame_hw<=qr/frame_hw))){dot=0.f;for(int d=0;d<dim;d++)dot=fmaf(q[(qr*heads+h)*dim+d],k[(j*kvheads+kh)*dim+d],dot);
            dot*=scale;if(bias){int delta=j-qr,n=abs(delta);int bucket=(delta>0?16:0)+(n<8?n:min(15,8+int(logf(float(n)/8)/logf(16.f)*8)));dot+=bias[bucket*heads+h];}}
        scores[tid]=isfinite(dot)?expf(dot-maximum):0.f;__syncthreads();
        if(tid==0)for(int z=0;z<256&&start+z<rows;z++)total+=scores[z];
        for(int a=0;a<8;a++){int d=tid+a*256;if(d<dim)for(int z=0;z<256&&start+z<rows;z++)out[a]=fmaf(scores[z],v[((start+z)*kvheads+kh)*dim+d],out[a]);}__syncthreads();}
    for(int a=0;a<8;a++){int d=tid+a*256;if(d<dim)y[(qr*heads+h)*dim+d]=total>0.f?out[a]/total:0.f;}
}
__global__ void im2col(float*y,const float*x,int t,int h,int w,int c,int kt,int kh,int kw,int pt,int ph,int pw,int replicate,int start,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,k=c*kt*kh*kw;if(i>=count*k)return;
    int r=start+i/k,j=i%k,dx=j%kw;j/=kw;int dy=j%kh;j/=kh;int dt=j%kt,channel=j/kt;
    int xx=r%w+dx-pw,yy=(r/w)%h+dy-ph,tt=r/(h*w)+dt-pt;
    if(replicate){xx=max(0,min(w-1,xx));yy=max(0,min(h-1,yy));tt=max(0,min(t-1,tt));}
    y[i]=(xx>=0&&xx<w&&yy>=0&&yy<h&&tt>=0&&tt<t)?x[((tt*h+yy)*w+xx)*c+channel]:0.f;
}
__global__ void im2col_half(half*y,const float*x,int t,int h,int w,int c,int kt,int kh,int kw,int pt,int ph,int pw,int replicate,int start,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,k=c*kt*kh*kw;if(i>=count*k)return;
    int r=start+i/k,j=i%k,dx=j%kw;j/=kw;int dy=j%kh;j/=kh;int dt=j%kt,channel=j/kt;
    int xx=r%w+dx-pw,yy=(r/w)%h+dy-ph,tt=r/(h*w)+dt-pt;
    if(replicate){xx=max(0,min(w-1,xx));yy=max(0,min(h-1,yy));tt=max(0,min(t-1,tt));}
    y[i]=(xx>=0&&xx<w&&yy>=0&&yy<h&&tt>=0&&tt<t)?x[((tt*h+yy)*w+xx)*c+channel]:0.f;
}
__global__ void reorder_conv_half(half*y,const half*x,int outputs,int channels,int volume) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=outputs*channels*volume)return;
    int c=i%channels,p=(i/channels)%volume,n=i/(channels*volume);
    y[i]=x[(n*channels+c)*volume+p];
}
__global__ void mean_rows(float*y,const float*x,int rows,int channels) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=channels)return;float sum=0.f;for(int r=0;r<rows;r++)sum+=x[r*channels+c];y[c]=sum/rows;
}
__global__ void channel_map(float*y,const float*x,int rows,int cin,int cout,int mean) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*cout)return;int r=i/cout,c=i%cout;
    if(mean){int group=cin/cout;float v=0.f;for(int j=0;j<group;j++)v+=x[r*cin+c*group+j];y[i]=v/group;}
    else y[i]=x[r*cin+c/(cout/cin)];
}
__global__ void downshuffle(float*y,const float*conv,const float*x,int t,int h,int w,int ci,int cc,int co,int temporal) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,nt=temporal?(t+1)/2:t,nh=h/2,nw=w/2;if(i>=nt*nh*nw*co)return;
    int outc=i%co,r=i/co,ox=r%nw,oy=(r/nw)%nh,ot=r/(nw*nh),rt=temporal?2:1;
    int part=outc/cc,ch=outc%cc,dx=part%2,dy=(part/2)%2,dt=part/4;
    int it=temporal?(ot==0?0:1+(ot-1)*rt+dt):ot;
    float value=conv[((it*h+oy*2+dy)*w+ox*2+dx)*cc+ch];
    int group=rt*4*ci/co;if(temporal&&ot==0)group/=2;float skip=0.f;
    for(int j=0;j<group;j++){int packed=outc*group+j,ic=packed%ci,pos=packed/ci,sx=pos%2,sy=(pos/2)%2,st=pos/4;
        int tt=temporal?(ot==0?0:1+(ot-1)*2+st):ot;skip+=x[((tt*h+oy*2+sy)*w+ox*2+sx)*ci+ic];}
    y[i]=value+skip/group;
}
__global__ void upshuffle(float*y,const float*conv,const float*x,int t,int h,int w,int ci,int co,int temporal) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,nt=temporal?(t-1)*2+1:t,nh=h*2,nw=w*2,rt=temporal?2:1;if(i>=nt*nh*nw*co)return;
    int c=i%co,r=i/co,ox=r%nw,oy=(r/nw)%nh,ot=r/(nw*nh),it=temporal?(ot==0?0:1+(ot-1)/2):ot;
    int dt=temporal&&ot>0?(ot-1)%2:0,part=(dt*2+oy%2)*2+ox%2;
    int cc=rt*4*co,conv_c;
    if(temporal&&ot==0)conv_c=((oy%2)*2+ox%2)*(co*2)+c;else conv_c=part*co+c;
    int repeats=rt*4*co/ci,skipc;
    if(temporal&&ot==0)skipc=((oy%2)*2+ox%2)*(ci/4)+c/(repeats/2);
    else if(temporal)skipc=part*(ci/(rt*4))+c/repeats;
    else skipc=(part*co+c)/repeats;
    int src=(it*h+oy/2)*w+ox/2;y[i]=conv[src*cc+conv_c]+x[src*ci+skipc];
}
__global__ void vision_patches(float*y,const float*x,int patches,int side,int patch) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,k=3*patch*patch;if(i>=patches*patches*k)return;
    int r=i/k,j=i%k,c=j/(patch*patch),dy=(j/patch)%patch,dx=j%patch;
    y[i]=x[(c*side+(r/patches)*patch+dy)*side+(r%patches)*patch+dx];
}
}
)CUDA";
}
#endif
