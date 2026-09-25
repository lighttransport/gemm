#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <stdio.h>

__global__ static void f32_to_bf16(__nv_bfloat16 *dst, const float *src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16_rn(src[i]);
}

__global__ static void f32_to_bf16_pad(__nv_bfloat16 *dst, const float *src,
                                       int channels, int h, int w, int pad) {
    int hp=h+2*pad,wp=w+2*pad;
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=(size_t)channels*hp*wp;
    if(i<n){int c=(int)(i/(hp*wp)),p=(int)(i%(hp*wp)),y=p/wp-pad,x=p%wp-pad;
        dst[i]=(y>=0&&y<h&&x>=0&&x<w)?__float2bfloat16_rn(src[(c*h+y)*w+x]):__float2bfloat16_rn(0.0f);}
}

__global__ static void f32_nchw_to_bf16_nhwc_pad(__nv_bfloat16 *dst, const float *src,
                                                  int channels, int h, int w, int pad) {
    int hp=h+2*pad,wp=w+2*pad;
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=(size_t)channels*hp*wp;
    if(i<n){int c=(int)(i%channels),p=(int)(i/channels),y=p/wp-pad,x=p%wp-pad;
        dst[i]=(y>=0&&y<h&&x>=0&&x<w)?__float2bfloat16_rn(src[(c*h+y)*w+x]):__float2bfloat16_rn(0.0f);}
}

__global__ static void f32_oihw_to_bf16_ohwi(__nv_bfloat16 *dst, const float *src,
                                              int co, int ci, int kh, int kw) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=(size_t)co*ci*kh*kw;
    if(i<n){int c=(int)(i%ci),p=(int)(i/ci),x=p%kw;p/=kw;int y=p%kh,o=p/kh;
        dst[i]=__float2bfloat16_rn(src[((o*ci+c)*kh+y)*kw+x]);}
}

__global__ static void bf16_to_f32_bias(float *dst, const __nv_bfloat16 *src,
                                        const __nv_bfloat16 *bias,
                                        int channels, int spatial) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)channels * spatial) {
        int c = (int)(i / spatial);
        __nv_bfloat16 value = __float2bfloat16_rn(__bfloat162float(src[i]) +
                                                  __bfloat162float(bias[c]));
        dst[i] = __bfloat162float(value);
    }
}

__global__ static void bf16_nhwc_to_f32_nchw_bias(float *dst, const __nv_bfloat16 *src,
                                                   const __nv_bfloat16 *bias,
                                                   int channels, int spatial) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=(size_t)channels*spatial;
    if(i<n){int c=(int)(i/spatial),p=(int)(i%spatial);
        __nv_bfloat16 value=__float2bfloat16_rn(__bfloat162float(src[(size_t)p*channels+c])+
                                                __bfloat162float(bias[c]));
        dst[i]=__bfloat162float(value);}
}

#define CCHK(expr) do { cudnnStatus_t status_ = (expr); if (status_ != CUDNN_STATUS_SUCCESS) { \
    fprintf(stderr, "q21_cudnn_vae: %s failed at line %d: %s\n", #expr, __LINE__, \
            cudnnGetErrorString(status_)); goto done; } } while (0)

static int q21_backend_conv(cudnnHandle_t handle, __nv_bfloat16 *y,
                            const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                            int ci, int h, int width, int co, int kh, int kw,
                            int stride_value, int oh, int ow) {
    cudnnBackendDescriptor_t xd=nullptr,yd=nullptr,wd=nullptr,cd=nullptr,op=nullptr;
    cudnnBackendDescriptor_t graph=nullptr,engine=nullptr,config=nullptr,plan=nullptr,pack=nullptr;
    cudnnBackendDescriptor_t choices[8]={}; void *workspace=nullptr; int rc=1;
    int channels_last=kh==3&&((ci==4)||(ci==96&&co==96&&h==1026&&stride_value==1));
    int64_t xdims[4]={1,ci,h,width}, xstr[4]={(int64_t)ci*h*width,(int64_t)h*width,width,1};
    int64_t ydims[4]={1,co,oh,ow}, ystr[4]={(int64_t)co*oh*ow,(int64_t)oh*ow,ow,1};
    int64_t wdims[4]={co,ci,kh,kw}, wstr[4]={(int64_t)ci*kh*kw,(int64_t)kh*kw,kw,1};
    if(channels_last){xstr[1]=1;xstr[2]=(int64_t)width*ci;xstr[3]=ci;
        ystr[1]=1;ystr[2]=(int64_t)ow*co;ystr[3]=co;
        wstr[1]=1;wstr[2]=(int64_t)kw*ci;wstr[3]=ci;}
    int64_t pad[2]={0,0},stride[2]={stride_value,stride_value},dilation[2]={1,1},spatial=2,alignment=32;
    int64_t uids[3]={120,121,119},engine_index=1,workspace_bytes=0;
    void *ptrs[3]={(void*)x,y,(void*)w};
    cudnnDataType_t bf16=CUDNN_DATA_BFLOAT16,f32=CUDNN_DATA_FLOAT;
    cudnnConvolutionMode_t mode=CUDNN_CROSS_CORRELATION;float alpha=1,beta=0;
    cudnnBackendKnobType_t kt[8]={CUDNN_KNOB_TYPE_TILE_SIZE,CUDNN_KNOB_TYPE_USE_TEX};
    int64_t kv[8]={3,0}; int nchoices=2;
    if(kh==1) {
        if(ci==192&&co==384){engine_index=22;nchoices=5;kt[1]=CUDNN_KNOB_TYPE_TILEK;kt[2]=CUDNN_KNOB_TYPE_STAGES;kt[3]=CUDNN_KNOB_TYPE_IDX_MODE;kt[4]=CUDNN_KNOB_TYPE_SPECFILT;kv[0]=6;kv[1]=0;kv[2]=2;kv[3]=1;kv[4]=0;}
        else if(ci==384&&co==768){engine_index=22;nchoices=5;kt[1]=CUDNN_KNOB_TYPE_TILEK;kt[2]=CUDNN_KNOB_TYPE_STAGES;kt[3]=CUDNN_KNOB_TYPE_IDX_MODE;kt[4]=CUDNN_KNOB_TYPE_SPECFILT;
            if(h>=128){kv[0]=0;kv[1]=0;kv[2]=3;kv[3]=0;kv[4]=1;}
            else{kv[0]=5;kv[1]=1;kv[2]=3;kv[3]=0;kv[4]=0;}}
        else if(ci==768){engine_index=22;nchoices=5;kt[1]=CUDNN_KNOB_TYPE_TILEK;kt[2]=CUDNN_KNOB_TYPE_STAGES;kt[3]=CUDNN_KNOB_TYPE_IDX_MODE;kt[4]=CUDNN_KNOB_TYPE_SPECFILT;kv[0]=5;kv[1]=1;kv[2]=3;kv[3]=0;kv[4]=1;}
        else if(ci==128&&co==128){engine_index=22;nchoices=5;kt[1]=CUDNN_KNOB_TYPE_TILEK;kt[2]=CUDNN_KNOB_TYPE_STAGES;kt[3]=CUDNN_KNOB_TYPE_IDX_MODE;kt[4]=CUDNN_KNOB_TYPE_SPECFILT;kv[0]=2;kv[1]=0;kv[2]=3;kv[3]=0;kv[4]=0;}
    } else if(ci==4) {
        engine_index=28;kv[0]=4;
    } else if((ci<=192&&stride_value==1)||(ci==96&&stride_value==2)) {
        engine_index=45;nchoices=3;kt[1]=CUDNN_KNOB_TYPE_KBLOCK;kt[2]=CUDNN_KNOB_TYPE_STAGES;kv[0]=0;kv[1]=3;kv[2]=2;
    } else if(ci==192&&stride_value==2) {
        engine_index=22;nchoices=5;kt[1]=CUDNN_KNOB_TYPE_TILEK;kt[2]=CUDNN_KNOB_TYPE_STAGES;kt[3]=CUDNN_KNOB_TYPE_IDX_MODE;kt[4]=CUDNN_KNOB_TYPE_SPECFILT;kv[0]=0;kv[1]=1;kv[2]=2;kv[3]=0;kv[4]=0;
    } else if(ci==384) {
        engine_index=22;nchoices=5;kt[1]=CUDNN_KNOB_TYPE_TILEK;kt[2]=CUDNN_KNOB_TYPE_STAGES;kt[3]=CUDNN_KNOB_TYPE_IDX_MODE;kt[4]=CUDNN_KNOB_TYPE_SPECFILT;kv[0]=0;kv[1]=(co==768)?1:0;kv[2]=(co==768)?2:3;kv[3]=0;kv[4]=(co==768)?0:2;
    } else if(ci==768&&co==768&&h>=129) {
        engine_index=45;nchoices=3;kt[1]=CUDNN_KNOB_TYPE_KBLOCK;kt[2]=CUDNN_KNOB_TYPE_STAGES;kv[0]=1;kv[1]=4;kv[2]=2;
    } else if(ci==768&&co==768) {
        engine_index=23;nchoices=8;kt[1]=CUDNN_KNOB_TYPE_SPLIT_K_BUF;kt[2]=CUDNN_KNOB_TYPE_TILEK;kt[3]=CUDNN_KNOB_TYPE_STAGES;kt[4]=CUDNN_KNOB_TYPE_REDUCTION_MODE;kt[5]=CUDNN_KNOB_TYPE_SPLIT_K_SLC;kt[6]=CUDNN_KNOB_TYPE_IDX_MODE;kt[7]=CUDNN_KNOB_TYPE_SPECFILT;kv[0]=0;kv[1]=-1;kv[2]=0;kv[3]=3;kv[4]=0;kv[5]=256;kv[6]=0;kv[7]=2;
    } else {
        engine_index=23;nchoices=8;kt[1]=CUDNN_KNOB_TYPE_SPLIT_K_BUF;kt[2]=CUDNN_KNOB_TYPE_TILEK;kt[3]=CUDNN_KNOB_TYPE_STAGES;kt[4]=CUDNN_KNOB_TYPE_REDUCTION_MODE;kt[5]=CUDNN_KNOB_TYPE_SPLIT_K_SLC;kt[6]=CUDNN_KNOB_TYPE_IDX_MODE;kt[7]=CUDNN_KNOB_TYPE_SPECFILT;kv[0]=7;kv[1]=3;kv[2]=0;kv[3]=4;kv[4]=0;kv[5]=192;kv[6]=1;kv[7]=0;
    }
#define BMAKE(v,t) do{CCHK(cudnnBackendCreateDescriptor((t),&(v)));}while(0)
#define BSET(d,a,t,n,p) CCHK(cudnnBackendSetAttribute((d),(a),(t),(n),(p)))
    BMAKE(xd,CUDNN_BACKEND_TENSOR_DESCRIPTOR);BSET(xd,CUDNN_ATTR_TENSOR_DATA_TYPE,CUDNN_TYPE_DATA_TYPE,1,&bf16);BSET(xd,CUDNN_ATTR_TENSOR_DIMENSIONS,CUDNN_TYPE_INT64,4,xdims);BSET(xd,CUDNN_ATTR_TENSOR_STRIDES,CUDNN_TYPE_INT64,4,xstr);BSET(xd,CUDNN_ATTR_TENSOR_UNIQUE_ID,CUDNN_TYPE_INT64,1,&uids[0]);BSET(xd,CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,CUDNN_TYPE_INT64,1,&alignment);CCHK(cudnnBackendFinalize(xd));
    BMAKE(yd,CUDNN_BACKEND_TENSOR_DESCRIPTOR);BSET(yd,CUDNN_ATTR_TENSOR_DATA_TYPE,CUDNN_TYPE_DATA_TYPE,1,&bf16);BSET(yd,CUDNN_ATTR_TENSOR_DIMENSIONS,CUDNN_TYPE_INT64,4,ydims);BSET(yd,CUDNN_ATTR_TENSOR_STRIDES,CUDNN_TYPE_INT64,4,ystr);BSET(yd,CUDNN_ATTR_TENSOR_UNIQUE_ID,CUDNN_TYPE_INT64,1,&uids[1]);BSET(yd,CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,CUDNN_TYPE_INT64,1,&alignment);CCHK(cudnnBackendFinalize(yd));
    BMAKE(wd,CUDNN_BACKEND_TENSOR_DESCRIPTOR);BSET(wd,CUDNN_ATTR_TENSOR_DATA_TYPE,CUDNN_TYPE_DATA_TYPE,1,&bf16);BSET(wd,CUDNN_ATTR_TENSOR_DIMENSIONS,CUDNN_TYPE_INT64,4,wdims);BSET(wd,CUDNN_ATTR_TENSOR_STRIDES,CUDNN_TYPE_INT64,4,wstr);BSET(wd,CUDNN_ATTR_TENSOR_UNIQUE_ID,CUDNN_TYPE_INT64,1,&uids[2]);BSET(wd,CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,CUDNN_TYPE_INT64,1,&alignment);CCHK(cudnnBackendFinalize(wd));
    BMAKE(cd,CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);BSET(cd,CUDNN_ATTR_CONVOLUTION_COMP_TYPE,CUDNN_TYPE_DATA_TYPE,1,&f32);BSET(cd,CUDNN_ATTR_CONVOLUTION_CONV_MODE,CUDNN_TYPE_CONVOLUTION_MODE,1,&mode);BSET(cd,CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,CUDNN_TYPE_INT64,1,&spatial);BSET(cd,CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,CUDNN_TYPE_INT64,2,pad);BSET(cd,CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,CUDNN_TYPE_INT64,2,pad);BSET(cd,CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,CUDNN_TYPE_INT64,2,stride);BSET(cd,CUDNN_ATTR_CONVOLUTION_DILATIONS,CUDNN_TYPE_INT64,2,dilation);CCHK(cudnnBackendFinalize(cd));
    BMAKE(op,CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);BSET(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&xd);BSET(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&wd);BSET(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&yd);BSET(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&cd);BSET(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,CUDNN_TYPE_FLOAT,1,&alpha);BSET(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,CUDNN_TYPE_FLOAT,1,&beta);CCHK(cudnnBackendFinalize(op));
    BMAKE(graph,CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);BSET(graph,CUDNN_ATTR_OPERATIONGRAPH_HANDLE,CUDNN_TYPE_HANDLE,1,&handle);BSET(graph,CUDNN_ATTR_OPERATIONGRAPH_OPS,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&op);CCHK(cudnnBackendFinalize(graph));
    BMAKE(engine,CUDNN_BACKEND_ENGINE_DESCRIPTOR);BSET(engine,CUDNN_ATTR_ENGINE_OPERATION_GRAPH,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&graph);BSET(engine,CUDNN_ATTR_ENGINE_GLOBAL_INDEX,CUDNN_TYPE_INT64,1,&engine_index);CCHK(cudnnBackendFinalize(engine));
    for(int i=0;i<nchoices;i++){BMAKE(choices[i],CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);BSET(choices[i],CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,CUDNN_TYPE_KNOB_TYPE,1,&kt[i]);BSET(choices[i],CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,CUDNN_TYPE_INT64,1,&kv[i]);CCHK(cudnnBackendFinalize(choices[i]));}
    BMAKE(config,CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);BSET(config,CUDNN_ATTR_ENGINECFG_ENGINE,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&engine);BSET(config,CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,CUDNN_TYPE_BACKEND_DESCRIPTOR,nchoices,choices);CCHK(cudnnBackendFinalize(config));
    BMAKE(plan,CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);BSET(plan,CUDNN_ATTR_EXECUTION_PLAN_HANDLE,CUDNN_TYPE_HANDLE,1,&handle);BSET(plan,CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&config);CCHK(cudnnBackendFinalize(plan));CCHK(cudnnBackendGetAttribute(plan,CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,CUDNN_TYPE_INT64,1,nullptr,&workspace_bytes));
    if(workspace_bytes&&cudaMalloc(&workspace,(size_t)workspace_bytes)!=cudaSuccess)goto done;
    BMAKE(pack,CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);BSET(pack,CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,CUDNN_TYPE_VOID_PTR,3,ptrs);BSET(pack,CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,CUDNN_TYPE_INT64,3,uids);BSET(pack,CUDNN_ATTR_VARIANT_PACK_WORKSPACE,CUDNN_TYPE_VOID_PTR,1,&workspace);CCHK(cudnnBackendFinalize(pack));CCHK(cudnnBackendExecute(handle,plan,pack));rc=0;
done:
    if(workspace)cudaFree(workspace);if(pack)cudnnBackendDestroyDescriptor(pack);if(plan)cudnnBackendDestroyDescriptor(plan);if(config)cudnnBackendDestroyDescriptor(config);for(int i=0;i<8;i++)if(choices[i])cudnnBackendDestroyDescriptor(choices[i]);if(engine)cudnnBackendDestroyDescriptor(engine);if(graph)cudnnBackendDestroyDescriptor(graph);if(op)cudnnBackendDestroyDescriptor(op);if(cd)cudnnBackendDestroyDescriptor(cd);if(wd)cudnnBackendDestroyDescriptor(wd);if(yd)cudnnBackendDestroyDescriptor(yd);if(xd)cudnnBackendDestroyDescriptor(xd);return rc;
#undef BSET
#undef BMAKE
}

static int q21_cudnn_vae_conv2d_impl(float *output, const float *input,
                                     const float *weight, const float *bias,
                                     int ci, int h, int w, int co, int kh, int kw,
                                     int stride, int pad, int oh, int ow,
                                     cudaStream_t stream) {
    if (!output || !input || !weight || !bias || ci <= 0 || h <= 0 || w <= 0 ||
        co <= 0 || kh <= 0 || kw <= 0) return 1;
    cudnnHandle_t handle = nullptr;
    cudnnTensorDescriptor_t xd = nullptr, yd = nullptr;
    cudnnFilterDescriptor_t wd = nullptr;
    cudnnConvolutionDescriptor_t cd = nullptr;
    __nv_bfloat16 *xb = nullptr, *wb = nullptr, *yb = nullptr, *bb = nullptr;
    void *workspace = nullptr;
    int rc = 1, returned = 0;
    int source_h=h,source_w=w,conv_h=h+2*pad,conv_w=w+2*pad;
    int channels_last=kh==3&&((ci==4)||(ci==96&&co==96&&conv_h==1026&&stride==1));
    size_t source_n=(size_t)ci*h*w,xn=(size_t)ci*conv_h*conv_w;
    size_t wn = (size_t)co*ci*kh*kw, yn = (size_t)co*oh*ow;
    size_t workspace_bytes = 0;
    cudnnConvolutionFwdAlgoPerf_t perf[8];
    float alpha = 1.0f, beta = 0.0f;

    if (cudaMalloc(&xb,xn*sizeof(*xb)) != cudaSuccess ||
        cudaMalloc(&wb,wn*sizeof(*wb)) != cudaSuccess ||
        cudaMalloc(&yb,yn*sizeof(*yb)) != cudaSuccess ||
        cudaMalloc(&bb,(size_t)co*sizeof(*bb)) != cudaSuccess) goto done;
    if(channels_last) f32_nchw_to_bf16_nhwc_pad<<<(xn+255)/256,256,0,stream>>>(xb,input,ci,source_h,source_w,pad);
    else if(pad) f32_to_bf16_pad<<<(xn+255)/256,256,0,stream>>>(xb,input,ci,source_h,source_w,pad);
    else f32_to_bf16<<<(source_n+255)/256,256,0,stream>>>(xb,input,source_n);
    if(channels_last) f32_oihw_to_bf16_ohwi<<<(wn+255)/256,256,0,stream>>>(wb,weight,co,ci,kh,kw);
    else f32_to_bf16<<<(wn+255)/256,256,0,stream>>>(wb,weight,wn);
    f32_to_bf16<<<(co+255)/256,256,0,stream>>>(bb,bias,(size_t)co);
    if (cudaGetLastError() != cudaSuccess) goto done;
    CCHK(cudnnCreate(&handle)); CCHK(cudnnSetStream(handle,stream));
    if ((kh==1 && kw==1 && stride==1 && pad==0) || (kh==3 && kw==3)) {
        if(q21_backend_conv(handle,yb,xb,wb,ci,conv_h,conv_w,co,kh,kw,
                            stride,oh,ow))goto done;
        if(channels_last) bf16_nhwc_to_f32_nchw_bias<<<(yn+255)/256,256,0,stream>>>(output,yb,bb,co,oh*ow);
        else bf16_to_f32_bias<<<(yn+255)/256,256,0,stream>>>(output,yb,bb,co,oh*ow);
        rc=cudaGetLastError()==cudaSuccess?0:1;goto done;
    }
    CCHK(cudnnCreateTensorDescriptor(&xd)); CCHK(cudnnCreateTensorDescriptor(&yd));
    CCHK(cudnnCreateFilterDescriptor(&wd)); CCHK(cudnnCreateConvolutionDescriptor(&cd));
    CCHK(cudnnSetTensor4dDescriptor(xd,CUDNN_TENSOR_NCHW,CUDNN_DATA_BFLOAT16,1,ci,conv_h,conv_w));
    CCHK(cudnnSetTensor4dDescriptor(yd,CUDNN_TENSOR_NCHW,CUDNN_DATA_BFLOAT16,1,co,oh,ow));
    CCHK(cudnnSetFilter4dDescriptor(wd,CUDNN_DATA_BFLOAT16,CUDNN_TENSOR_NCHW,co,ci,kh,kw));
    CCHK(cudnnSetConvolution2dDescriptor(cd,0,0,stride,stride,1,1,
                                         CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
    CCHK(cudnnSetConvolutionMathType(cd,CUDNN_TENSOR_OP_MATH));
    CCHK(cudnnGetConvolutionForwardAlgorithm_v7(handle,xd,wd,cd,yd,8,&returned,perf));
    if (returned < 1 || perf[0].status != CUDNN_STATUS_SUCCESS) goto done;
    CCHK(cudnnGetConvolutionForwardWorkspaceSize(handle,xd,wd,cd,yd,
                                                  perf[0].algo,&workspace_bytes));
    if (workspace_bytes && cudaMalloc(&workspace,workspace_bytes) != cudaSuccess) goto done;
    CCHK(cudnnConvolutionForward(handle,&alpha,xd,xb,wd,wb,cd,perf[0].algo,
                                  workspace,workspace_bytes,&beta,yd,yb));
    bf16_to_f32_bias<<<(yn+255)/256,256,0,stream>>>(output,yb,bb,co,oh*ow);
    rc = cudaGetLastError() == cudaSuccess ? 0 : 1;
done:
    if (workspace) cudaFree(workspace);
    if (cd) cudnnDestroyConvolutionDescriptor(cd); if (wd) cudnnDestroyFilterDescriptor(wd);
    if (yd) cudnnDestroyTensorDescriptor(yd); if (xd) cudnnDestroyTensorDescriptor(xd);
    if (handle) cudnnDestroy(handle);
    if (bb) cudaFree(bb); if (yb) cudaFree(yb); if (wb) cudaFree(wb); if (xb) cudaFree(xb);
    return rc;
}

extern "C" int q21_cudnn_vae_conv2d(float *output, const float *input,
                                      const float *weight, const float *bias,
                                      int ci, int h, int w, int co, int kh, int kw,
                                      cudaStream_t stream) {
    return q21_cudnn_vae_conv2d_impl(output,input,weight,bias,ci,h,w,co,kh,kw,
                                     1,kh/2,h,w,stream);
}

extern "C" int q21_cudnn_vae_conv2d_ex(float *output, const float *input,
                                         const float *weight, const float *bias,
                                         int ci, int h, int w, int co, int kh, int kw,
                                         int stride, int pad, int oh, int ow,
                                         cudaStream_t stream) {
    return q21_cudnn_vae_conv2d_impl(output,input,weight,bias,ci,h,w,co,kh,kw,
                                     stride,pad,oh,ow,stream);
}

/* F32 convolution for the native VAE decoder (--conv cudnn): F32 data with
 * FMA math (no TF32), zero padding kh/2, bias added in F32. The handle, the
 * algorithm per shape and a grow-only workspace are cached across calls;
 * FFT and Winograd algorithms are excluded to keep direct-sum accuracy. */
__global__ static void f32_add_bias(float *y, const float *bias, int channels, int spatial) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(size_t)channels*spatial) y[i]+=bias[i/spatial];
}

namespace {
struct f32_algo { int ci,h,w,co,kh,kw; cudnnConvolutionFwdAlgo_t algo; size_t workspace; };
cudnnHandle_t f32_handle;
f32_algo f32_algos[64];
int f32_count;
void *f32_workspace;
size_t f32_workspace_bytes;
}

extern "C" int q21_cudnn_conv2d_f32(float *output, const float *input, const float *weight, const float *bias,
                                    int ci, int h, int w, int co, int kh, int kw, cudaStream_t stream) {
    if (!output || !input || !weight || ci <= 0 || h <= 0 || w <= 0 || co <= 0 || kh <= 0 || kw <= 0) return 1;
    cudnnTensorDescriptor_t xd = nullptr, yd = nullptr;
    cudnnFilterDescriptor_t wd = nullptr;
    cudnnConvolutionDescriptor_t cd = nullptr;
    const f32_algo *found = nullptr;
    float alpha = 1.0f, beta = 0.0f;
    int rc = 1;
    if (!f32_handle) CCHK(cudnnCreate(&f32_handle));
    CCHK(cudnnSetStream(f32_handle, stream));
    CCHK(cudnnCreateTensorDescriptor(&xd)); CCHK(cudnnCreateTensorDescriptor(&yd));
    CCHK(cudnnCreateFilterDescriptor(&wd)); CCHK(cudnnCreateConvolutionDescriptor(&cd));
    CCHK(cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, ci, h, w));
    CCHK(cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, co, h, w));
    CCHK(cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, co, ci, kh, kw));
    CCHK(cudnnSetConvolution2dDescriptor(cd, kh / 2, kw / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CCHK(cudnnSetConvolutionMathType(cd, CUDNN_FMA_MATH));
    for (int i = 0; i < f32_count; i++) {
        const f32_algo *a = &f32_algos[i];
        if (a->ci == ci && a->h == h && a->w == w && a->co == co && a->kh == kh && a->kw == kw) found = a;
    }
    if (!found) {
        cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        int returned = 0;
        CCHK(cudnnGetConvolutionForwardAlgorithm_v7(f32_handle, xd, wd, cd, yd, CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                                                     &returned, perf));
        for (int i = 0; i < returned && !found; i++) {
            cudnnConvolutionFwdAlgo_t algo = perf[i].algo;
            if (perf[i].status != CUDNN_STATUS_SUCCESS || perf[i].mathType != CUDNN_FMA_MATH ||
                algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT || algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING ||
                algo == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD || algo == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
                continue;
            size_t bytes = 0;
            if (cudnnGetConvolutionForwardWorkspaceSize(f32_handle, xd, wd, cd, yd, algo, &bytes) !=
                CUDNN_STATUS_SUCCESS || f32_count == 64)
                continue;
            f32_algos[f32_count] = {ci, h, w, co, kh, kw, algo, bytes};
            found = &f32_algos[f32_count++];
        }
        if (!found) goto done;
    }
    if (found->workspace > f32_workspace_bytes) {
        if (f32_workspace) cudaFree(f32_workspace);
        f32_workspace = nullptr;
        f32_workspace_bytes = 0;
        if (cudaMalloc(&f32_workspace, found->workspace) != cudaSuccess) goto done;
        f32_workspace_bytes = found->workspace;
    }
    CCHK(cudnnConvolutionForward(f32_handle, &alpha, xd, input, wd, weight, cd, found->algo, f32_workspace,
                                 found->workspace, &beta, yd, output));
    if (bias) f32_add_bias<<<(unsigned)(((size_t)co * h * w + 255) / 256), 256, 0, stream>>>(output, bias, co, h * w);
    rc = cudaGetLastError() == cudaSuccess ? 0 : 1;
done:
    if (cd) cudnnDestroyConvolutionDescriptor(cd);
    if (wd) cudnnDestroyFilterDescriptor(wd);
    if (yd) cudnnDestroyTensorDescriptor(yd);
    if (xd) cudnnDestroyTensorDescriptor(xd);
    return rc;
}
#undef CCHK
