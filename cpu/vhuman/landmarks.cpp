/* Inference for the pinned face detector, landmarks and blendshape graphs.
 * NHWC float32 tensors, repository AVX2 GEMM; no TFLite/ONNX/Torch runtime.
 * Asset conversion folds float16 constants and validates the original task hash.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#ifdef __AVX2__
#include "../../ryzen/gemm_avx2.h"
#endif

namespace {
thread_local std::string error;
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
struct tensor {
    int rank = 0;
    std::array<int,4> shape{1,1,1,1};
    std::vector<float> data;
    size_t size() const { return data.size(); }
    int dim(int i) const { return shape.at(i); }
};
struct node { int code, output; std::array<int,8> a; std::vector<int> in; };
struct graph {
    std::vector<tensor> t;
    std::vector<node> nodes;
    std::vector<int> inputs, outputs;
    size_t output_size = 0;
    int threads = 4;
};
struct close_file { void operator()(FILE *f) const { fclose(f); } };
int read_int(FILE *f) { int32_t v; require(fread(&v,4,1,f)==1,"truncated graph"); return v; }
size_t index(const tensor &t, const std::array<int,4> &coords) {
    size_t offset=0;
    for (int d=0;d<t.rank;d++) { require(coords[d]>=0 && coords[d]<t.dim(d),"tensor index"); offset=offset*t.dim(d)+coords[d]; }
    return offset;
}
std::array<int,4> coordinates(const tensor &t, size_t offset) {
    std::array<int,4> c{};
    for(int d=t.rank-1;d>=0;d--) { c[d]=offset%t.dim(d); offset/=t.dim(d); }
    return c;
}
size_t broadcast(const tensor &t,const tensor &out,const std::array<int,4> &c) {
    require(t.rank<=out.rank,"broadcast rank"); size_t p=0;
    for(int d=0;d<t.rank;d++) {
        int j=d+out.rank-t.rank;
        require(t.dim(d)==1 || t.dim(d)==out.dim(j),"broadcast dimension");
        p=p*t.dim(d)+(t.dim(d)==1?0:c[j]);
    }
    return p;
}
float activation(float x,int kind) {
    switch(kind) {
    case 0:return x;
    case 1:return std::max(0.f,x);
    case 2:return std::clamp(x,-1.f,1.f);
    case 3:return std::clamp(x,0.f,6.f);
    default:throw std::runtime_error("unsupported fused activation");
    }
}
void gemm(float *y,const float *x,const float *w,int m,int n,int k) {
    // Weights are OHWI: compute W * X^T using the repository row-major GEMM.
    std::vector<float> xt(size_t(k)*m),yt(size_t(n)*m);
    for(int i=0;i<m;i++) for(int j=0;j<k;j++) xt[size_t(j)*m+i]=x[size_t(i)*k+j];
#ifdef __AVX2__
    sgemm_avx2(n,m,k,1,w,k,xt.data(),m,0,yt.data(),m);
#else
    for(int i=0;i<n;i++) for(int j=0;j<m;j++) {
        double s=0;for(int d=0;d<k;d++) s+=double(w[size_t(i)*k+d])*xt[size_t(d)*m+j];
        yt[size_t(i)*m+j]=float(s);
    }
#endif
    for(int i=0;i<m;i++) for(int j=0;j<n;j++) y[size_t(i)*n+j]=yt[size_t(j)*m+i];
}
void conv(graph &g,const node &op) {
    const tensor &x=g.t.at(op.in.at(0)),&w=g.t.at(op.in.at(1)),&b=g.t.at(op.in.at(2));
    tensor &y=g.t.at(op.output);
    require(x.rank==4 && w.rank==4 && y.rank==4 && x.dim(0)==1 && y.dim(0)==1,"conv shape");
    int h=x.dim(1),wid=x.dim(2),ci=x.dim(3),oh=y.dim(1),ow=y.dim(2),co=y.dim(3);
    int kh=w.dim(1),kw=w.dim(2),sw=op.a[1],sh=op.a[2],dw=op.a[4],dh=op.a[5];
    require(sw>0 && sh>0 && dw>0 && dh>0 && b.size()==size_t(co),"conv parameters");
    int py=op.a[0]==0?std::max(0,(oh-1)*sh+(kh-1)*dh+1-h)/2:0;
    int px=op.a[0]==0?std::max(0,(ow-1)*sw+(kw-1)*dw+1-wid)/2:0;
    if(op.code==4) {
        int multiplier=op.a[6];
        require(multiplier>0 && co==ci*multiplier && w.dim(0)==1 && w.dim(3)==co,"depthwise shape");
        #pragma omp parallel for schedule(static)
        for(int p=0;p<oh*ow;p++) for(int c=0;c<co;c++) {
            float v=b.data[c];
            for(int yy=0;yy<kh;yy++) for(int xx=0;xx<kw;xx++) {
                int iy=p/ow*sh-py+yy*dh,ix=p%ow*sw-px+xx*dw;
                if(iy>=0 && iy<h && ix>=0 && ix<wid)
                    v+=x.data[(size_t(iy)*wid+ix)*ci+c/multiplier]*w.data[(size_t(yy)*kw+xx)*co+c];
            }
            y.data[size_t(p)*co+c]=activation(v,op.a[3]);
        }
    } else {
        require(w.dim(0)==co && w.dim(3)==ci,"conv channels");
        int k=kh*kw*ci;
        #pragma omp parallel for schedule(static)
        for(int start=0;start<oh*ow;start+=128) {
            int count=std::min(128,oh*ow-start);
            std::vector<float> cols(size_t(count)*k);
            for(int p=0;p<count;p++) for(int yy=0;yy<kh;yy++) for(int xx=0;xx<kw;xx++) {
                int iy=(start+p)/ow*sh-py+yy*dh,ix=(start+p)%ow*sw-px+xx*dw;
                if(iy>=0 && iy<h && ix>=0 && ix<wid)
                    memcpy(cols.data()+size_t(p)*k+(yy*kw+xx)*ci,x.data.data()+(size_t(iy)*wid+ix)*ci,ci*4);
            }
            float *dst=y.data.data()+size_t(start)*co;
            gemm(dst,cols.data(),w.data.data(),count,co,k);
            for(int p=0;p<count;p++) for(int c=0;c<co;c++) dst[size_t(p)*co+c]=activation(dst[size_t(p)*co+c]+b.data[c],op.a[3]);
        }
    }
}
void run_node(graph &g,const node &op) {
    tensor &y=g.t.at(op.output);
    const tensor &x=g.t.at(op.in.at(0));
    if(op.code==3 || op.code==4) { conv(g,op);return; }
    if(op.code==22) { require(y.size()==x.size(),"reshape size"); y.data=x.data;return; }
    if(op.code==17) {
        require(x.rank==4 && y.rank==4 && x.dim(0)==1 && y.dim(0)==1 && x.dim(3)==y.dim(3),"pool shape");
        int h=x.dim(1),w=x.dim(2),c=x.dim(3),oh=y.dim(1),ow=y.dim(2);
        int sw=op.a[1],sh=op.a[2],kw=op.a[4],kh=op.a[5];
        require(sw>0 && sh>0 && kw>0 && kh>0,"pool parameters");
        int py=op.a[0]==0?std::max(0,(oh-1)*sh+kh-h)/2:0,px=op.a[0]==0?std::max(0,(ow-1)*sw+kw-w)/2:0;
        for(int p=0;p<oh*ow;p++) for(int ch=0;ch<c;ch++) {
            float v=-INFINITY;
            for(int yy=0;yy<kh;yy++) for(int xx=0;xx<kw;xx++) {
                int iy=p/ow*sh-py+yy,ix=p%ow*sw-px+xx;
                if(iy>=0 && iy<h && ix>=0 && ix<w) v=std::max(v,x.data[(size_t(iy)*w+ix)*c+ch]);
            }
            y.data[size_t(p)*c+ch]=activation(v,op.a[3]);
        }
        return;
    }
    if(op.code==2) {
        int axis=op.a[0];if(axis<0)axis+=y.rank;
        require(axis>=0 && axis<y.rank,"concat axis");
        int offset=0;
        for(int id:op.in) {
            const auto &a=g.t.at(id);require(a.rank==y.rank,"concat rank");
            for(int d=0;d<y.rank;d++) require(d==axis || a.dim(d)==y.dim(d),"concat dimension");
            for(size_t i=0;i<a.size();i++) { auto c=coordinates(a,i);c[axis]+=offset;y.data.at(index(y,c))=activation(a.data[i],op.a[1]); }
            offset+=a.dim(axis);
        }
        require(offset==y.dim(axis),"concat size");return;
    }
    if(op.code==34) {
        const auto &pads=g.t.at(op.in.at(1));require(x.rank==y.rank && pads.size()==size_t(x.rank*2),"padding rank");
        std::fill(y.data.begin(),y.data.end(),0);
        for(size_t i=0;i<x.size();i++) { auto c=coordinates(x,i);for(int d=0;d<x.rank;d++)c[d]+=int(pads.data[d*2]);y.data.at(index(y,c))=x.data[i]; }
        return;
    }
    if(op.code==39) {
        const auto &perm=g.t.at(op.in.at(1)); require(perm.size()==size_t(x.rank) && x.rank==y.rank,"transpose rank");
        for(size_t i=0;i<y.size();i++) {
            auto c=coordinates(y,i);std::array<int,4> in{};
            for(int d=0;d<y.rank;d++) { int j=int(perm.data[d]);require(j>=0 && j<x.rank,"transpose axis");in[j]=c[d]; }
            y.data[i]=x.data.at(index(x,in));
        }
        return;
    }
    if(op.code==40 || op.code==74) {
        std::array<bool,4> axes{};const auto &a=g.t.at(op.in.at(1));
        size_t count=1;
        for(float v:a.data) { int d=int(v);if(d<0)d+=x.rank;require(d>=0 && d<x.rank,"reduction axis"); if(!axes[d])count*=x.dim(d); axes[d]=true; }
        std::fill(y.data.begin(),y.data.end(),0);
        for(size_t i=0;i<x.size();i++) {
            auto c=coordinates(x,i);std::array<int,4> out{};int j=0;
            for(int d=0;d<x.rank;d++) { if(!axes[d])out[j++]=c[d];else if(op.a[0])out[j++]=0; }
            require(j==y.rank,"reduction rank");y.data.at(index(y,out))+=x.data[i];
        }
        if(op.code==40)for(float &v:y.data)v/=float(count);
        return;
    }
    if(op.code==45) {
        const auto &begin=g.t.at(op.in.at(1)),&end=g.t.at(op.in.at(2)),&strides=g.t.at(op.in.at(3));
        require(begin.size()==size_t(x.rank) && end.size()==begin.size() && strides.size()==begin.size() && !op.a[2] && !op.a[3],"slice rank/masks");
        std::array<int,4> starts{},steps{};
        for(int d=0;d<x.rank;d++) {
            steps[d]=int(strides.data[d]);require(steps[d]!=0,"slice stride");
            starts[d]=int(begin.data[d]);if(starts[d]<0)starts[d]+=x.dim(d);
            if(op.a[0]&(1<<d))starts[d]=steps[d]>0?0:x.dim(d)-1;
        }
        for(size_t i=0;i<y.size();i++) {
            auto c=coordinates(y,i);std::array<int,4> in{};int j=0;
            for(int d=0;d<x.rank;d++) in[d]=starts[d]+((op.a[4]&(1<<d))?0:c[j++]*steps[d]);
            require(j==y.rank,"slice output rank");y.data[i]=x.data.at(index(x,in));
        }
        return;
    }
    for(size_t i=0;i<y.size();i++) {
        auto c=coordinates(y,i);float v=x.data.at(broadcast(x,y,c));
        float b=0;
        if(op.in.size()>1) { const auto &other=g.t.at(op.in[1]);b=other.data.at(broadcast(other,y,c)); }
        switch(op.code) {
        case 0:v=activation(v+b,op.a[0]);break;
        case 18:v=activation(v*b,op.a[0]);break;
        case 41:v=activation(v-b,op.a[0]);break;
        case 42:v=activation(v/b,op.a[0]);break;
        case 19:v=std::max(0.f,v);break;
        case 54:if(v<0)v*=b;break;
        case 14:v=v>=0?1/(1+std::exp(-v)):std::exp(v)/(1+std::exp(v));break;
        case 59:v=-v;break;
        case 75:v=std::sqrt(v);break;
        case 76:v=1/std::sqrt(v);break;
        case 99:v=(v-b)*(v-b);break;
        default:throw std::runtime_error("unsupported graph operator "+std::to_string(op.code));
        }
        y.data[i]=v;
    }
}
}
extern "C" const char *vh_face_error() { return error.c_str(); }
extern "C" void vh_face_close(graph *g) { delete g; }
extern "C" graph *vh_face_open(const char *path,int threads) {
    try {
        require(threads>=1 && threads<=64,"invalid thread count");
        std::unique_ptr<FILE,close_file> f(fopen(path,"rb"));require(bool(f),"cannot open native graph");
        char magic[8];require(fread(magic,1,8,f.get())==8 && !memcmp(magic,"VHFACE1\0",8),"invalid graph header");
        int nt=read_int(f.get()),nn=read_int(f.get()),ni=read_int(f.get()),no=read_int(f.get());
        require(nt>0 && nt<=2048 && nn>0 && nn<=1024 && ni==1 && no>0 && no<=8,"invalid graph counts");
        auto g=std::make_unique<graph>();g->threads=threads;g->t.resize(nt);
        for(int i=0;i<ni+no;i++) { int id=read_int(f.get());require(id>=0 && id<nt,"invalid graph IO");(i<ni?g->inputs:g->outputs).push_back(id); }
        size_t total=0;
        for(auto &t:g->t) {
            t.rank=read_int(f.get());require(t.rank>=0 && t.rank<=4,"invalid tensor rank");size_t size=1;
            for(int d=0;d<4;d++) { t.shape[d]=read_int(f.get());require(t.shape[d]>0 && t.shape[d]<=65536,"invalid dimension");if(d<t.rank)size*=t.shape[d];require(size<=16000000,"oversized tensor"); }
            total+=size;require(total<=64000000,"oversized graph");t.data.resize(size);
            int stored=read_int(f.get());require(stored==0 || size_t(stored)==size,"invalid constant size");
            if(stored)require(fread(t.data.data(),4,stored,f.get())==size_t(stored),"truncated constant");
        }
        for(int i=0;i<nn;i++) {
            node n;n.code=read_int(f.get());int count=read_int(f.get());n.output=read_int(f.get());
            require(count>0 && count<=16 && n.output>=0 && n.output<nt,"invalid node");
            for(int &a:n.a)a=read_int(f.get());
            for(int j=0;j<count;j++) { int id=read_int(f.get());require(id>=0 && id<nt,"invalid input");n.in.push_back(id); }
            g->nodes.push_back(std::move(n));
        }
        require(fgetc(f.get())==EOF,"trailing graph data");
        for(int id:g->outputs)g->output_size+=g->t[id].size();
        return g.release();
    } catch(const std::exception &e) { error=e.what();return nullptr; }
}
extern "C" size_t vh_face_size(graph *g,int output) { return !g?0:output?g->output_size:g->t[g->inputs[0]].size(); }
extern "C" int vh_face_run(graph *g,const float *input,size_t count,float *output,size_t capacity) {
    try {
        require(g && input && output && count==vh_face_size(g,0) && capacity==g->output_size,"invalid graph IO sizes");
        omp_set_num_threads(g->threads);
        std::copy(input,input+count,g->t[g->inputs[0]].data.begin());
        for(const auto &n:g->nodes)run_node(*g,n);
        for(int id:g->outputs) { auto &t=g->t[id];for(float v:t.data)require(std::isfinite(v),"nonfinite graph output");std::copy(t.data.begin(),t.data.end(),output);output+=t.size(); }
        return 0;
    } catch(const std::exception &e) { error=e.what();return -1; }
}
