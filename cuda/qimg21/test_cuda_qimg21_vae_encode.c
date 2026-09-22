/* Native single-frame RGBA -> raw posterior moments and optional normalized
 * posterior-mode tokens. No image preprocessing, sampling or temporal continuation. */
#define main q21_decoder_main
#include "test_cuda_qimg21_vae.c"
#undef main
#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../common/stb_image_resize2.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../common/stb_image_write.h"

static double q21_lanczos(double x) {
    x = fabs(x);
    if (x < 1e-12) return 1.0;
    if (x >= 3.0) return 0.0;
    double p = M_PI * x;
    return sin(p) * sin(p / 3.0) / (p * (p / 3.0));
}

static int q21_lanczos_resize(const unsigned char *source, int sw, int sh,
                              unsigned char *output, int ow, int oh) {
    const int precision = 22;
    double scales[2] = {(double)sw / ow, (double)sh / oh};
    int sizes[2][2] = {{sw, ow}, {sh, oh}}, *bounds[2] = {NULL, NULL};
    int32_t *coeff[2] = {NULL, NULL};
    int ksize[2] = {0, 0};
    for (int axis = 0; axis < 2; axis++) {
        double filter_scale = fmax(1.0, scales[axis]), support = 3.0 * filter_scale;
        ksize[axis] = (int)ceil(support) * 2 + 1;
        int input_size = sizes[axis][0], output_size = sizes[axis][1];
        bounds[axis] = malloc((size_t)output_size * 2 * sizeof(int));
        coeff[axis] = calloc((size_t)output_size * ksize[axis], sizeof(int32_t));
        if (!bounds[axis] || !coeff[axis]) goto fail;
        for (int out_index = 0; out_index < output_size; out_index++) {
            double center = (out_index + 0.5) * scales[axis];
            int first = (int)(center - support + 0.5);
            int last = (int)(center + support + 0.5);
            if (first < 0) first = 0;
            if (last > input_size) last = input_size;
            int count = last - first;
            double weights[64], sum = 0.0;
            if (count > (int)(sizeof(weights) / sizeof(weights[0]))) goto fail;
            for (int i = 0; i < count; i++) {
                weights[i] = q21_lanczos((i + first - center + 0.5) / filter_scale);
                sum += weights[i];
            }
            for (int i = 0; i < count; i++) {
                double normalized = weights[i] / sum * (1 << precision);
                coeff[axis][(size_t)out_index * ksize[axis] + i] =
                    (int32_t)(normalized < 0 ? normalized - 0.5 : normalized + 0.5);
            }
            bounds[axis][out_index * 2] = first;
            bounds[axis][out_index * 2 + 1] = count;
        }
    }
    unsigned char *premultiplied = malloc((size_t)sw * sh * 4);
    unsigned char *horizontal = malloc((size_t)sh * ow * 4);
    unsigned char *resized_premultiplied = malloc((size_t)oh * ow * 4);
    if (!premultiplied || !horizontal || !resized_premultiplied) {
        free(premultiplied); free(horizontal); free(resized_premultiplied); goto fail;
    }
    for (int i = 0; i < sw * sh; i++) {
        int alpha = source[i * 4 + 3];
        for (int c = 0; c < 3; c++) premultiplied[i * 4 + c] =
            (unsigned char)((source[i * 4 + c] * alpha + 127) / 255);
        premultiplied[i * 4 + 3] = (unsigned char)alpha;
    }
    for (int y = 0; y < sh; y++) for (int x = 0; x < ow; x++) for (int c = 0; c < 4; c++) {
        int64_t sum = 1 << (precision - 1);
        int first = bounds[0][x * 2], count = bounds[0][x * 2 + 1];
        const int32_t *weights = coeff[0] + (size_t)x * ksize[0];
        for (int i = 0; i < count; i++) sum += premultiplied[((size_t)y * sw + first + i) * 4 + c] * weights[i];
        int value = (int)(sum >> precision);
        horizontal[((size_t)y * ow + x) * 4 + c] = (unsigned char)(value < 0 ? 0 : value > 255 ? 255 : value);
    }
    for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) for (int c = 0; c < 4; c++) {
        int64_t sum = 1 << (precision - 1);
        int first = bounds[1][y * 2], count = bounds[1][y * 2 + 1];
        const int32_t *weights = coeff[1] + (size_t)y * ksize[1];
        for (int i = 0; i < count; i++) sum += horizontal[((size_t)(first + i) * ow + x) * 4 + c] * weights[i];
        int value = (int)(sum >> precision);
        resized_premultiplied[((size_t)y * ow + x) * 4 + c] =
            (unsigned char)(value < 0 ? 0 : value > 255 ? 255 : value);
    }
    for (int i = 0; i < ow * oh; i++) {
        int alpha = resized_premultiplied[i * 4 + 3];
        for (int c = 0; c < 3; c++) output[i * 4 + c] = alpha ?
            (unsigned char)fmin(255, resized_premultiplied[i * 4 + c] * 255 / alpha) : 0;
        output[i * 4 + 3] = (unsigned char)alpha;
    }
    free(premultiplied); free(horizontal); free(resized_premultiplied);
    for (int i = 0; i < 2; i++) { free(bounds[i]); free(coeff[i]); }
    return 0;
fail:
    for (int i = 0; i < 2; i++) { free(bounds[i]); free(coeff[i]); }
    return -1;
}

static int q21_load_image(const char *path, int resolution, const char *resized_output,
                          float **out, int *oh, int *ow) {
    int sw, sh, channels;
    unsigned char *source = stbi_load(path, &sw, &sh, &channels, 4);
    if (!source || sw < 1 || sh < 1 || resolution < 32 || resolution > 1024) {
        stbi_image_free(source); return -1;
    }
    double ratio = (double)sw / sh;
    int w = (int)nearbyint(sqrt((double)resolution * resolution * ratio) / 32.0) * 32;
    int h = (int)nearbyint(sqrt((double)resolution * resolution / ratio) / 32.0) * 32;
    if (w < 32 || h < 32 || w > 1024 || h > 1024) { stbi_image_free(source); return -1; }
    unsigned char *resized = malloc((size_t)w * h * 4);
    float *chw = malloc((size_t)w * h * 4 * sizeof(float));
    if (!resized || !chw || q21_lanczos_resize(source, sw, sh, resized, w, h)) {
        free(resized); free(chw); stbi_image_free(source); return -1;
    }
    if (resized_output && !stbi_write_png(resized_output, w, h, 4, resized, w * 4)) {
        free(resized); free(chw); stbi_image_free(source); return -1;
    }
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) for (int c = 0; c < 4; c++)
        chw[((size_t)c * h + y) * w + x] =
            ((float)resized[((size_t)y * w + x) * 4 + c] / 255.0f) * 2.0f - 1.0f;
    free(resized); stbi_image_free(source);
    *out = chw; *oh = h; *ow = w;
    return 0;
}

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
    const char *model=NULL,*input=NULL,*input_image=NULL,*output=NULL,*latent_output=NULL;
    const char *preprocessed_output=NULL,*resized_output=NULL;
    int resolution=1024,preprocess_only=0;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--model")&&i+1<argc)model=argv[++i];
        else if(!strcmp(argv[i],"--image")&&i+1<argc)input=argv[++i];
        else if(!strcmp(argv[i],"--input-image")&&i+1<argc)input_image=argv[++i];
        else if(!strcmp(argv[i],"--resolution")&&i+1<argc)resolution=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--preprocessed-out")&&i+1<argc)preprocessed_output=argv[++i];
        else if(!strcmp(argv[i],"--resized-out")&&i+1<argc)resized_output=argv[++i];
        else if(!strcmp(argv[i],"--preprocess-only"))preprocess_only=1;
        else if(!strcmp(argv[i],"--out")&&i+1<argc)output=argv[++i];
        else if(!strcmp(argv[i],"--normalized-latents")&&i+1<argc)latent_output=argv[++i];
        else return 2;
    }
    if((!preprocess_only&&!model)||!!input==!!input_image||(!preprocess_only&&!output)||
       (preprocess_only&&!preprocessed_output)){
        fprintf(stderr,"usage: %s --model VAE_DIR (--image RGBA_CHW.npy | --input-image IMAGE) "
                       "--out MOMENTS.npy [--resolution N --preprocessed-out IMAGE.npy "
                       "--preprocess-only --normalized-latents TOKENS.npy]\n",argv[0]);return 2;}
    if(latent_output && !strcmp(latent_output,output))return 2;
    q21_npy a={0};
    if(input_image) {
        int image_h,image_w;
        if(q21_load_image(input_image,resolution,resized_output,&a.data,&image_h,&image_w))return 1;
        a.n=(size_t)4*image_h*image_w;a.ndim=3;
        a.shape[0]=4;a.shape[1]=image_h;a.shape[2]=image_w;
    } else if(q21_npy_read_f32(input,&a))return 1;
    if(a.ndim!=3||a.shape[0]!=4||a.shape[1]>1024||a.shape[2]>1024||a.shape[1]%16||a.shape[2]%16){q21_npy_free(&a);return 2;}
    for(size_t i=0;i<a.n;i++)if(!isfinite(a.data[i])||a.data[i]<-1||a.data[i]>1){q21_npy_free(&a);return 2;}
    int h=a.shape[1],w=a.shape[2];
    if(preprocessed_output&&q21_npy_write_chw(preprocessed_output,a.data,a.n,4,h,w)){q21_npy_free(&a);return 1;}
    if(preprocess_only){q21_npy_free(&a);return 0;}
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
