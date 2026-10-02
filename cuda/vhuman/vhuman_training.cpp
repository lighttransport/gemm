/* Persistent CUDA cue CNN/Gaussian training. Driver + NVRTC + own GEMM only.
 * Host depth sort/tile bins are retained; weights, gradients and Adam moments
 * stay on the device across steps. No tensor/BLAS/rasterizer runtime. */
#include "../cuew.h"
#include "vhuman_training.h"
#include "../gemm/cuda_gemm_f32_kernels.h"
#include "vhuman_training_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static thread_local std::string last_error;
static void require(bool ok,const char *message) { if(!ok)throw std::runtime_error(message); }
static void check(CUresult code) { if(code)throw std::runtime_error("CUDA driver error "+std::to_string(int(code))); }
static void finite(const float *p,size_t n) {
    require(p,"missing training array");for(size_t i=0;i<n;++i)require(std::isfinite(p[i]),"nonfinite training array");
}
struct context_scope {
    explicit context_scope(CUcontext c) { check(cuCtxPushCurrent(c)); }
    ~context_scope() { CUcontext c;cuCtxPopCurrent(&c); }
};
struct training_buffer { CUdeviceptr p=0;size_t capacity=0; };
struct trainer {
    CUdevice device=0;CUcontext context=nullptr;CUstream stream=nullptr;CUmodule module=nullptr;
    std::map<std::string,CUfunction> kernels;
    std::map<std::string,training_buffer> buffers;
    int count=0,iteration=0;size_t budget=0,allocated=0,peak=0,overlaps=0;
    double lr=0,decay=0;
    ~trainer() {
        if(!context)return;
        CUcontext previous;
        if(cuCtxPushCurrent(context)==CUDA_SUCCESS) {
            if(stream)cuStreamSynchronize(stream);
            for(auto &entry:buffers)if(entry.second.p)cuMemFree(entry.second.p);
            if(module)cuModuleUnload(module);
            if(stream)cuStreamDestroy(stream);
            cuCtxPopCurrent(&previous);
        }
        cuDevicePrimaryCtxRelease(device);
    }
    CUdeviceptr reserve(const std::string &name,size_t bytes) {
        auto &b=buffers[name];if(b.capacity>=bytes)return b.p;
        require(bytes<=budget&&allocated-b.capacity<=budget-bytes,"CUDA training memory budget exceeded");
        if(b.p){check(cuMemFree(b.p));allocated-=b.capacity;b={};}
        check(cuMemAlloc(&b.p,bytes));b.capacity=bytes;allocated+=bytes;peak=std::max(peak,allocated);return b.p;
    }
    CUdeviceptr upload(const std::string &name,const void *data,size_t bytes) {
        auto p=reserve(name,std::max(size_t(4),bytes));if(bytes)check(cuMemcpyHtoD(p,data,bytes));return p;
    }
    void zero(CUdeviceptr p,size_t bytes) { check(cuMemsetD8Async(p,0,bytes,stream)); }
    void launch(const char *name,int grid,int block,std::initializer_list<void *> args,int gy=1,int by=1) {
        auto found=kernels.find(name);
        if(found==kernels.end()) { CUfunction f;check(cuModuleGetFunction(&f,module,name));found=kernels.emplace(name,f).first; }
        std::vector<void *> arguments(args);
        check(cuLaunchKernel(found->second,grid,gy,1,block,by,1,0,stream,arguments.data(),nullptr));
    }
    void gemm(CUdeviceptr out,CUdeviceptr a,CUdeviceptr b,int m,int n,int k,int ta=0,int tb=0) {
        launch("gemm_f32_train",(n+15)/16,16,{&out,&a,&b,&m,&n,&k,&ta,&tb},(m+15)/16,16);
    }
    void download(void *out,CUdeviceptr p,size_t bytes) { check(cuStreamSynchronize(stream));check(cuMemcpyDtoH(out,p,bytes)); }
    void step() {
        auto p=buffers.at("parameters").p,g=buffers.at("gradient").p,m=buffers.at("moment1").p,v=buffers.at("moment2").p;
        double b1=1-std::pow(.9,iteration+1),b2=1-std::pow(.999,iteration+1);
        launch("adam",(count+255)/256,256,{&p,&g,&m,&v,&count,&lr,&decay,&b1,&b2});++iteration;
    }
    void validate(CUdeviceptr out,int elements,bool backward) {
        auto invalid=reserve("invalid",4);zero(invalid,4);
        launch("check_finite",(elements+255)/256,256,{&out,&elements,&invalid});
        if(backward){auto g=buffers.at("gradient").p;launch("check_finite",(count+255)/256,256,{&g,&count,&invalid});}
        int bad=0;download(&bad,invalid,4);require(!bad,"nonfinite CUDA training result");
    }
};
static std::vector<char> compile(int major,int minor) {
    std::string src=std::string("extern \"C\" {\n")+CUDA_GEMM_F32_TRAIN_SRC+"}\n"+vh_training_source;
    nvrtcProgram program=nullptr;
    require(nvrtcCreateProgram(&program,src.c_str(),"vhuman_training.cu",0,nullptr,nullptr)==NVRTC_SUCCESS,"NVRTC create failed");
    std::string arch="--gpu-architecture=compute_"+std::to_string(major)+std::to_string(minor);
    const char *options[]={arch.c_str(),"--std=c++14","--fmad=false"};auto code=nvrtcCompileProgram(program,3,options);
    size_t bytes=0;
    if(code!=NVRTC_SUCCESS) {
        nvrtcGetProgramLogSize(program,&bytes);std::vector<char> log(bytes+1);nvrtcGetProgramLog(program,log.data());
        nvrtcDestroyProgram(&program);throw std::runtime_error(log.data());
    }
    nvrtcGetPTXSize(program,&bytes);std::vector<char> ptx(bytes);code=nvrtcGetPTX(program,ptx.data());nvrtcDestroyProgram(&program);
    require(code==NVRTC_SUCCESS,"NVRTC PTX failed");return ptx;
}
extern "C" const char *vht_error() { return last_error.c_str(); }
extern "C" int vht_compile_probe() {
    try { require(cuewInit(CUEW_INIT_NVRTC)==CUEW_SUCCESS,"NVRTC unavailable");compile(12,0);return 0; }
    catch(const std::exception &e){last_error=e.what();return -1;}
}
extern "C" void vht_close(trainer *t) { delete t; }
extern "C" trainer *vht_open(int device,const float *parameters,int count,double lr,double decay,size_t budget) {
    trainer *t=nullptr;
    try {
        require(count>0&&count<=6404096&&budget>=1048576&&budget<=size_t(4096)*1048576,"invalid CUDA training allocation");finite(parameters,count);
        require(std::isfinite(lr)&&lr>0&&std::isfinite(decay)&&decay>=0,"invalid CUDA optimizer");
        require(cuewInit(CUEW_INIT_CUDA|CUEW_INIT_NVRTC)==CUEW_SUCCESS,"CUDA/NVRTC unavailable");check(cuInit(0));
        t=new trainer;t->count=count;t->budget=budget;t->lr=lr;t->decay=decay;
        check(cuDeviceGet(&t->device,device));check(cuDevicePrimaryCtxRetain(&t->context,t->device));context_scope current(t->context);
        check(cuStreamCreate(&t->stream,CU_STREAM_NON_BLOCKING));int major,minor;
        check(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,t->device));
        check(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,t->device));
        require(major>=6,"native training needs FP64 atomics (SM60+)");auto ptx=compile(major,minor);check(cuModuleLoadData(&t->module,ptx.data()));
        size_t bytes=size_t(count)*4;t->upload("parameters",parameters,bytes);t->reserve("gradient",bytes);
        t->zero(t->reserve("moment1",bytes),bytes);t->zero(t->reserve("moment2",bytes),bytes);check(cuStreamSynchronize(t->stream));return t;
    }catch(const std::exception &e){last_error=e.what();delete t;return nullptr;}
}
extern "C" int vht_parameters(trainer *t,float *host,int upload) {
    try { require(t&&host,"invalid parameter access");context_scope current(t->context);check(cuStreamSynchronize(t->stream));
        if(upload){finite(host,t->count);t->upload("parameters",host,size_t(t->count)*4);}
        else t->download(host,t->buffers.at("parameters").p,size_t(t->count)*4);
        return 0;
    }catch(const std::exception &e){last_error=e.what();return -1;}
}
extern "C" size_t vht_peak_bytes(trainer *t) { return t?t->peak:0; }
extern "C" size_t vht_overlaps(trainer *t) { return t?t->overlaps:0; }
extern "C" int vht_optimizer(trainer *t,double lr,double decay) {
    try {
        require(t&&std::isfinite(lr)&&lr>0&&std::isfinite(decay)&&decay>=0,"invalid CUDA optimizer");
        t->lr=lr;t->decay=decay;return 0;
    }catch(const std::exception &e){last_error=e.what();return -1;}
}
extern "C" int vht_gemm(trainer *t,float *out,const float *a,const float *b,int m,int n,int k,int ta,int tb) {
    try {
        require(t&&out&&m>0&&n>0&&k>0&&m<=32768&&n<=32768&&k<=32768&&(ta==0||ta==1)&&(tb==0||tb==1),"invalid CUDA GEMM dimensions");
        finite(a,size_t(m)*k);finite(b,size_t(k)*n);context_scope current(t->context);check(cuStreamSynchronize(t->stream));
        auto ap=t->upload("gemm_a",a,size_t(m)*k*4),bp=t->upload("gemm_b",b,size_t(k)*n*4),yp=t->reserve("gemm_y",size_t(m)*n*4);
        t->gemm(yp,ap,bp,m,n,k,ta,tb);t->download(out,yp,size_t(m)*n*4);finite(out,size_t(m)*n);return 0;
    }catch(const std::exception &e){last_error=e.what();return -1;}
}

extern "C" int vht_appearance(trainer *t,const float *vertices,const int *attachments,const float *bary,
    const float *controls,const float *camera,int n,int v,int c,int w,int h,const float *truth,const float *mask,
    float *rgba,float *gradient,double *loss,int update)
{
    try {
        require(t&&n>0&&n<=200000&&v>=3&&v<=2000000&&c>0&&c<=512&&t->count==n*32+c*8&&w>0&&w<=4096&&h>0&&h<=4096&&loss,"invalid CUDA appearance dimensions");
        int backward=truth!=nullptr;require(!update||backward,"missing CUDA appearance supervision");
        finite(vertices,size_t(v)*3);finite(bary,size_t(n)*3);finite(controls,c);finite(camera,25);require(attachments,"missing attachment indices");
        require(camera[16]>0&&camera[20]>0,"invalid focal length");
        for(int i=0;i<n*3;++i)require(attachments[i]>=0&&attachments[i]<v,"attachment index out of bounds");
        size_t pixels=size_t(w)*h;
        if(backward){finite(truth,pixels*3);finite(mask,pixels);for(size_t i=0;i<pixels;++i)require(mask[i]>=0&&mask[i]<=1,"invalid appearance mask");}
        context_scope current(t->context);check(cuStreamSynchronize(t->stream));
        auto p=t->buffers.at("parameters").p,g=t->buffers.at("gradient").p;
        auto vp=t->upload("vertices",vertices,size_t(v)*12),ip=t->upload("attachments",attachments,size_t(n)*12),bp=t->upload("bary",bary,size_t(n)*12);
        auto cp=t->upload("controls",controls,size_t(c)*4),cam=t->upload("camera",camera,100),coeff=t->reserve("coeff",32),expression=p+size_t(n)*128;
        t->gemm(coeff,cp,expression,1,8,c);t->launch("tanh8",1,8,{&coeff});
        auto projected=t->reserve("projection",size_t(n)*12*8),jac=t->reserve("jacobian",size_t(n)*20*8);
        t->launch("project_train",(n+127)/128,128,{&p,&vp,&ip,&bp,&cam,&coeff,&n,&w,&h,&projected,&jac});
        std::vector<double> projection(size_t(n)*12);t->download(projection.data(),projected,projection.size()*8);
        int tw=(w+15)/16,th=(h+15)/16,tiles=tw*th;std::vector<int> order(n),offsets(tiles+1,0);std::iota(order.begin(),order.end(),0);
        for(double value:projection)require(std::isfinite(value),"nonfinite CUDA Gaussian projection");
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return projection[a*12+2]<projection[b*12+2];});
        auto visit=[&](int i,auto append){const double *q=projection.data()+i*12;if(q[10]<=0||q[11]<=0)return;
            int x0=int(std::clamp(std::floor((q[0]-q[10])/16),0.,double(tw))),x1=int(std::clamp(std::ceil((q[0]+q[10])/16),0.,double(tw)));
            int y0=int(std::clamp(std::floor((q[1]-q[11])/16),0.,double(th))),y1=int(std::clamp(std::ceil((q[1]+q[11])/16),0.,double(th)));
            for(int y=y0;y<y1;++y)for(int x=x0;x<x1;++x)append(y*tw+x,i);
        };
        size_t count=0;for(int i:order)visit(i,[&](int tile,int){require(++count<=8000000,"Gaussian overlap budget exceeded");++offsets[tile+1];});
        std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());std::vector<int> ids(count),cursor=offsets;
        for(int i:order)visit(i,[&](int tile,int id){ids[cursor[tile]++]=id;});
        t->overlaps=count;
        auto op=t->upload("offsets",offsets.data(),offsets.size()*4),ids_p=t->upload("ids",ids.data(),ids.size()*4),out=t->reserve("output",pixels*16);
        auto pg=t->reserve("projected_gradient",size_t(n)*9*8),lp=t->reserve("loss",16),dcoeff=t->reserve("dcoeff",64),dc=t->reserve("dc",32);
        CUdeviceptr target=0,mp=0;
        if(backward){target=t->upload("truth",truth,pixels*12);mp=t->upload("mask",mask,pixels*4);}
        t->zero(pg,size_t(n)*9*8);t->zero(lp,16);t->zero(dcoeff,64);
        t->launch("raster_train",tiles,256,{&projected,&op,&ids_p,&w,&h,&target,&mp,&out,&pg,&lp,&backward});
        if(backward){
            t->launch("chain_train",(n+127)/128,128,{&p,&pg,&jac,&coeff,&n,&g,&dcoeff,&lp});
            t->launch("coeff_reverse",1,8,{&dcoeff,&coeff,&dc});
            auto eg=g+size_t(n)*128;t->gemm(eg,cp,dc,c,8,1);
        }
        t->download(loss,lp,16);require(std::isfinite(loss[0])&&std::isfinite(loss[1]),"nonfinite CUDA appearance loss");
        t->validate(out,int(pixels*4),backward);
        if(update)t->step();
        if(rgba)t->download(rgba,out,pixels*16);
        if(gradient&&backward)t->download(gradient,g,size_t(t->count)*4);
        check(cuStreamSynchronize(t->stream));
        return 0;
    }catch(const std::exception &e){last_error=e.what();return -1;}
}

extern "C" int vht_cues(trainer *t,const float *inputs,const float *truth,const float *mask,int n,int h,int w,
    float *output,float *gradient,double *loss,int update)
{
    try {
        require(t&&t->count==19876&&n>0&&n<=8&&h>0&&h<=128&&w>0&&w<=128&&loss,"invalid CUDA cue dimensions");
        int backward=truth!=nullptr;require(!update||backward,"missing CUDA cue supervision");size_t pixels=size_t(n)*h*w;
        finite(inputs,pixels*6);double denominator=1;
        if(backward){finite(truth,pixels*3);finite(mask,pixels);denominator=0;for(size_t i=0;i<pixels;++i){require(mask[i]>=0&&mask[i]<=1,"invalid cue mask");denominator+=mask[i];}denominator=std::max(denominator,1.);}
        context_scope current(t->context);check(cuStreamSynchronize(t->stream));auto p=t->buffers.at("parameters").p,g=t->buffers.at("gradient").p;
        auto input=t->upload("inputs",inputs,pixels*24);CUdeviceptr a[6],z[5];int hs[6]={h},ws[6]={w},ic[]={6,16,24,32,24},oc[]={16,24,32,24,4},ks[]={5,3,3,3,1},ss[]={1,2,2,1,1};
        int weights[]={0,2416,5896,12840,19776},biases[]={2400,5872,12808,19752,19872};
        int area=h*w,np=int(pixels),channels=6;a[0]=t->reserve("a0",pixels*24);
        t->launch("layout_input",(np*6+255)/256,256,{&input,&a[0],&np,&channels,&area});
        for(int i=0;i<5;++i){
            hs[i+1]=(hs[i]+ss[i]-1)/ss[i];ws[i+1]=(ws[i]+ss[i]-1)/ss[i];int rows=n*hs[i+1]*ws[i+1],k=ic[i]*ks[i]*ks[i],elements=rows*oc[i],active=i<4;
            auto col=t->reserve("columns",size_t(rows)*k*4);z[i]=t->reserve("z"+std::to_string(i),size_t(elements)*4);a[i+1]=t->reserve("a"+std::to_string(i+1),size_t(elements)*4);
            t->launch("columns",(rows*k+255)/256,256,{&a[i],&col,&n,&ic[i],&hs[i],&ws[i],&hs[i+1],&ws[i+1],&ks[i],&ss[i]});
            auto weight=p+size_t(weights[i])*4,bias=p+size_t(biases[i])*4;t->gemm(z[i],col,weight,rows,oc[i],k,0,1);
            t->launch("conv_finish",(elements+255)/256,256,{&z[i],&a[i+1],&bias,&elements,&oc[i],&active});
        }
        auto raw=t->reserve("raw",pixels*16),draw=t->reserve("draw",pixels*16),out=t->reserve("output",pixels*16),lp=t->reserve("loss",16);
        int four=4,reverse=0;t->zero(lp,16);t->launch("resize_train",(np*4+255)/256,256,{&a[5],&raw,&n,&four,&hs[5],&ws[5],&h,&w,&reverse});
        CUdeviceptr target=0,mp=0;if(backward){target=t->upload("truth",truth,pixels*12);mp=t->upload("mask",mask,pixels*4);}
        t->launch("cue_loss",(np+255)/256,256,{&raw,&a[0],&target,&mp,&out,&draw,&np,&area,&denominator,&lp,&backward});
        if(backward){
            auto dy=t->reserve("dy5",size_t(n)*hs[5]*ws[5]*16);t->zero(dy,size_t(n)*hs[5]*ws[5]*16);reverse=1;
            t->launch("resize_train",(np*4+255)/256,256,{&draw,&dy,&n,&four,&hs[5],&ws[5],&h,&w,&reverse});
            for(int i=4;i>=0;--i){
                int rows=n*hs[i+1]*ws[i+1],k=ic[i]*ks[i]*ks[i],elements=rows*oc[i],active=i<4;
                auto col=t->reserve("columns",size_t(rows)*k*4),dcol=t->reserve("dcolumns",size_t(rows)*k*4),dx=t->reserve("dy"+std::to_string(i),size_t(n)*hs[i]*ws[i]*ic[i]*4);
                t->launch("conv_reverse",(elements+255)/256,256,{&dy,&z[i],&elements,&active});
                auto gw=g+size_t(weights[i])*4,gb=g+size_t(biases[i])*4,weight=p+size_t(weights[i])*4;
                t->launch("bias_reverse",oc[i],256,{&dy,&gb,&rows,&oc[i]});
                t->launch("columns",(rows*k+255)/256,256,{&a[i],&col,&n,&ic[i],&hs[i],&ws[i],&hs[i+1],&ws[i+1],&ks[i],&ss[i]});
                t->gemm(gw,dy,col,oc[i],k,rows,1,0);t->gemm(dcol,dy,weight,rows,k,oc[i]);
                int in_elements=n*hs[i]*ws[i]*ic[i];t->launch("col_reverse",(in_elements+255)/256,256,{&dcol,&dx,&n,&ic[i],&hs[i],&ws[i],&hs[i+1],&ws[i+1],&ks[i],&ss[i]});dy=dx;
            }
        }
        t->download(loss,lp,8);require(std::isfinite(*loss),"nonfinite CUDA cue loss");
        t->validate(out,int(pixels*4),backward);
        if(update)t->step();
        if(output)t->download(output,out,pixels*16);
        if(gradient&&backward)t->download(gradient,g,size_t(t->count)*4);
        check(cuStreamSynchronize(t->stream));
        return 0;
    }catch(const std::exception &e){last_error=e.what();return -1;}
}
