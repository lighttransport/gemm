/* Native single-frame RGBA -> raw posterior moments and optional normalized
 * posterior-mode tokens. No image preprocessing, sampling or temporal continuation. */
#define main q21_decoder_main
#include "test_cuda_qimg21_vae.c"
#undef main

static const char *q21_encoder_src =
"extern \"C\" {\n"
"__global__ void avg_first(float*y,const float*x,int ci,int co,int h,int w,int ft,int fs){\n"
" int i=blockIdx.x*blockDim.x+threadIdx.x,oh=h/fs,ow=w/fs;if(i>=co*oh*ow)return;\n"
" int c=i/(oh*ow),py=(i/ow)%oh,px=i%ow,factor=ft*fs*fs,group=ci*factor/co;float sum=0;\n"
" for(int g=0;g<group;g++){int e=c*group+g,ic=e/factor,tf=(e/(fs*fs))%ft;\n"
" if(tf==ft-1)sum+=x[(ic*h+py*fs+(e/fs)%fs)*w+px*fs+e%fs];}y[i]=sum/group;}\n"
"__global__ void sample_odd(float*y,const float*x,int c,int h,int w){\n"
" int i=blockIdx.x*blockDim.x+threadIdx.x,oh=h/2,ow=w/2;if(i>=c*oh*ow)return;\n"
" int ch=i/(oh*ow),py=(i/ow)%oh,px=i%ow;y[i]=x[(ch*h+2*py+1)*w+2*px+1];}\n"
"}\n";

static int q21_encode(cuda_qimg_runner *r, const st_context *st,
                      const float *input, int h, int w, float *output) {
    CUmodule module=NULL;
    CUfunction avg, sample;
    CUdeviceptr x=0,y=0,shortcut=0,gamma=0;
    int rc=1,c=96;
    if(cu_compile_kernels(&module,r->device,q21_encoder_src,"qimg21_encoder.cu",1,"qimg21_encoder")<0 ||
       cuModuleGetFunction(&avg,module,"avg_first") || cuModuleGetFunction(&sample,module,"sample_odd"))goto done;
    x=checked_cuMemAlloc((size_t)4*h*w*4);
    if(!x || cuMemcpyHtoD(x,input,(size_t)4*h*w*4) || cuCtxSynchronize())goto done;
    y=q21_conv(r,st,x,4,h,w,c,"encoder.conv_in.weight","encoder.conv_in.bias");
    q21_free(&x);x=y;y=0;if(!x)goto done;
    q21_dump(r,x,c,h,w,"encoder_conv_in");
    const int channels[5]={96,192,384,768,768};
    for(int stage=0;stage<5;stage++) {
        int co=channels[stage],fs=stage<4?2:1,ft=stage>0 && stage<4?2:1;
        int oh=h/fs,ow=w/fs,n=co*oh*ow;
        shortcut=checked_cuMemAlloc((size_t)n*4);if(!shortcut)goto done;
        void *args[]={&shortcut,&x,&c,&co,&h,&w,&ft,&fs};
        if(cuLaunchKernel(avg,(n+255)/256,1,1,256,1,1,0,r->stream,args,NULL) || cuCtxSynchronize())goto done;
        char name[256],bias[256];
        for(int block=0;block<2;block++) {
            snprintf(name,sizeof(name),"encoder.down_blocks.%d.resnets.%d",stage,block);
            y=q21_resblock(r,x,st,name,c,co,h,w);
            q21_free(&x);x=y;y=0;if(!x)goto done;c=co;
        }
        if(fs==2) {
            snprintf(name,sizeof(name),"encoder.down_blocks.%d.downsampler.resample.1.weight",stage);
            snprintf(bias,sizeof(bias),"encoder.down_blocks.%d.downsampler.resample.1.bias",stage);
            /* Stride-2 with right/bottom padding equals odd-position samples
             * of the same convolution with symmetric one-pixel padding. */
            y=q21_conv(r,st,x,c,h,w,c,name,bias);
            q21_free(&x);if(!y)goto done;
            x=checked_cuMemAlloc((size_t)n*4);if(!x)goto done;
            void *a[]={&x,&y,&c,&h,&w};
            if(cuLaunchKernel(sample,(n+255)/256,1,1,256,1,1,0,r->stream,a,NULL)||cuCtxSynchronize())goto done;
            q21_free(&y);h=oh;w=ow;
        }
        float one=1;
        void *a[]={&x,&shortcut,&one,&n};
        if(cuLaunchKernel(r->euler_step,(n+255)/256,1,1,256,1,1,0,r->stream,a,NULL)||cuCtxSynchronize())goto done;
        q21_free(&shortcut);
        snprintf(name,sizeof(name),"encoder_down_%d",stage);q21_dump(r,x,c,h,w,name);
    }
    y=q21_resblock(r,x,st,"encoder.mid_block.resnets.0",c,c,h,w);
    q21_free(&x);x=y;y=0;if(!x)goto done;
    if(!q21_mid_attention_named(r,st,x,c,h,w,"encoder.mid_block.attentions.0"))goto done;
    y=q21_resblock(r,x,st,"encoder.mid_block.resnets.1",c,c,h,w);
    q21_free(&x);x=y;y=0;if(!x)goto done;
    gamma=q21_load_weight(st,"encoder.norm_out.gamma");
    y=checked_cuMemAlloc((size_t)c*h*w*4);if(!gamma||!y)goto done;
    vae_op_gn(r,y,x,gamma,c,h*w);vae_op_silu(r,y,c*h*w);
    q21_free(&x);q21_free(&gamma);x=y;y=0;
    y=q21_conv(r,st,x,c,h,w,128,"encoder.conv_out.weight","encoder.conv_out.bias");
    q21_free(&x);x=y;y=0;if(!x)goto done;
    y=q21_conv1(r,st,x,128,h,w,128,"quant_conv.weight","quant_conv.bias");
    if(!y||cuCtxSynchronize()||cuMemcpyDtoH(output,y,(size_t)128*h*w*4))goto done;
    rc=0;
done:
    q21_free(&x);q21_free(&y);q21_free(&shortcut);q21_free(&gamma);
    if(module)cuModuleUnload(module);
    return rc;
}

int main(int argc,char **argv) {
    const char *model=NULL,*input=NULL,*output=NULL,*latent_output=NULL;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--model")&&i+1<argc)model=argv[++i];
        else if(!strcmp(argv[i],"--image")&&i+1<argc)input=argv[++i];
        else if(!strcmp(argv[i],"--out")&&i+1<argc)output=argv[++i];
        else if(!strcmp(argv[i],"--normalized-latents")&&i+1<argc)latent_output=argv[++i];
        else return 2;
    }
    if(!model||!input||!output){fprintf(stderr,"usage: %s --model VAE_DIR --image RGBA_CHW.npy --out MOMENTS.npy [--normalized-latents TOKENS.npy]\n",argv[0]);return 2;}
    if(latent_output && !strcmp(latent_output,output))return 2;
    q21_npy a={0};if(q21_npy_read_f32(input,&a))return 1;
    if(a.ndim!=3||a.shape[0]!=4||a.shape[1]>1024||a.shape[2]>1024||a.shape[1]%16||a.shape[2]%16){q21_npy_free(&a);return 2;}
    for(size_t i=0;i<a.n;i++)if(!isfinite(a.data[i])||a.data[i]<-1||a.data[i]>1){q21_npy_free(&a);return 2;}
    int h=a.shape[1],w=a.shape[2];
    char path[2048];snprintf(path,sizeof(path),"%s/diffusion_pytorch_model.safetensors",model);
    st_context *st=safetensors_open(path);if(!st){q21_npy_free(&a);return 1;}
    cuda_qimg_runner *r=cuda_qimg_init(0,1);if(!r){safetensors_close(st);q21_npy_free(&a);return 1;}
    r->use_fp8_pipe=0;r->use_fp8_pipe_perrow=0;
    CUstream original=r->stream;cuStreamSynchronize(original);r->stream=NULL;
    q21_vae_dump_dir=getenv("QIMG21_VAE_DUMP_DIR");if(q21_vae_dump_dir)mkdir(q21_vae_dump_dir,0755);
    size_t count=(size_t)128*(h/16)*(w/16);
    float *moments=malloc(count*4);
    int rc=moments?q21_encode(r,st,a.data,h,w,moments):1;
    if(!rc)for(size_t i=0;i<count;i++)if(!isfinite(moments[i])){rc=1;break;}
    if(!rc)rc=q21_npy_write_chw(output,moments,count,128,h/16,w/16);
    if(!rc && latent_output) {
        int tokens=(h/16)*(w/16);
        float mean[64],std[64];q21_latent_stats(mean,std);
        float *latents=malloc((size_t)tokens*64*sizeof(float));
        if(!latents)rc=1;
        else {
            for(int i=0;i<tokens;i++)for(int c=0;c<64;c++)
                latents[i*64+c]=(moments[c*tokens+i]-mean[c])/std[c];
            char shape[64];snprintf(shape,sizeof(shape),"(%d, 64)",tokens);
            rc=q21_npy_write_shape(latent_output,latents,(size_t)tokens*64,shape);
            free(latents);
        }
    }
    cuStreamSynchronize(r->stream);r->stream=original;
    free(moments);cuda_qimg_free(r);safetensors_close(st);q21_npy_free(&a);return rc?1:0;
}
