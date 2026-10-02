/* Framework-free image model runner. All dense math uses repository GEMM. */
#define _GNU_SOURCE
#define SWIN_NO_MAIN
#include "swin_runner.c"
#include "rmbg2.h"
#include "moge2_camera.h"

static vh_tensor vh_cues(st_context *s,vh_tensor x)
{
    const char *names[]={"encoder.0","encoder.2","encoder.4","decoder.0","decoder.2"};
    vh_tensor p=x;
    for(int i=0;i<5;i++) {
        vh_tensor next=vh_conv(s,names[i],p,i==1||i==2?2:1,i==0?2:i==4?0:1,0);
        if(i)vh_drop(p);
        p=next;if(i<4)vh_act(p,2);
    }
    vh_tensor y=vh_resize(p,x.h,x.w,0);vh_drop(p);
    if(y.c!=4)vh_fail("invalid cue head");
    int n=x.h*x.w;
    for(int i=0;i<n;i++) {
        float norm=0;
        for(int c=0;c<3;c++){float v=x.d[(size_t)(c+3)*n+i]+.15f*tanhf(y.d[(size_t)c*n+i]);y.d[(size_t)c*n+i]=v;norm+=v*v;}
        norm=fmaxf(sqrtf(norm),1e-6f);
        for(int c=0;c<3;c++)y.d[(size_t)c*n+i]/=norm;
        y.d[3*(size_t)n+i]=vh_sigmoid(y.d[3*(size_t)n+i]);
    }
    return y;
}

#ifndef VH_MODELS_NO_MAIN
int main(int argc,char **argv)
{
    const char *task=NULL,*model=NULL,*input=NULL,*output=NULL,*backend="cpu",*backbone=NULL;
    int h=0,w=0,threads=4,device=0,gh=0,gw=0;
    for(int i=1;i<argc;i++) {
        if(i+1>=argc)goto usage;
        const char *flag=argv[i],*v=argv[++i];
        if(!strcmp(flag,"--task"))task=v;
        else if(!strcmp(flag,"--model"))model=v;
        else if(!strcmp(flag,"--input"))input=v;
        else if(!strcmp(flag,"--output"))output=v;
        else if(!strcmp(flag,"--backend"))backend=v;
        else if(!strcmp(flag,"--backbone"))backbone=v;
        else if(!strcmp(flag,"--grid-height"))gh=swin_number(v);
        else if(!strcmp(flag,"--grid-width"))gw=swin_number(v);
        else if(!strcmp(flag,"--height"))h=swin_number(v);
        else if(!strcmp(flag,"--width"))w=swin_number(v);
        else if(!strcmp(flag,"--threads"))threads=swin_number(v);
        else if(!strcmp(flag,"--device"))device=!strcmp(v,"0")?0:swin_number(v);
        else goto usage;
    }
    if(!task || !model || !input || !output || h<1 || w<1 || threads<1 || threads>128 || device<0 ||
       (strcmp(task,"rmbg") && strcmp(task,"cues") && strcmp(task,"moge")) || (strcmp(backend,"cpu") && strcmp(backend,"cuda")))goto usage;
    int limit=!strcmp(task,"moge")?2048:1024;
    if(h>limit || w>limit)goto usage;
    if(!strcmp(task,"rmbg") && (h!=w || h%32))goto usage;
    if(!strcmp(task,"moge") && (!backbone || gh<1 || gw<1 || gh>128 || gw>128 || gh*gw>4096))goto usage;
    omp_set_num_threads(threads);
#ifdef SWIN_CUDA
    swin_gpu=!strcmp(backend,"cuda");
    if(swin_gpu && cuda_linear_f32_init(device)){cuda_linear_f32_free();return 4;}
#else
    if(strcmp(backend,"cpu")){fprintf(stderr,"Use cuda/vhuman/vhuman_models for CUDA\n");return 2;}
#endif
    struct timespec start,end;clock_gettime(CLOCK_MONOTONIC,&start);
    vh_tensor x=vh_read(input,!strcmp(task,"cues")?6:3,h,w),y;
    if(!strcmp(task,"rmbg")) {
        swin_model *m=swin_load(model);if(!m)vh_fail("invalid Swin backbone");
        y=rmbg_predict(m,x);swin_free(m);
    }else{
        st_context *s=safetensors_open(model);if(!s)vh_fail("cannot load model");
        y=!strcmp(task,"moge")?moge_camera_predict(s,backbone,x,gh,gw,threads):vh_cues(s,x);
        safetensors_close(s);
    }
    vh_write(output,y);vh_drop(x);vh_drop(y);clock_gettime(CLOCK_MONOTONIC,&end);
#ifdef SWIN_CUDA
    cuda_linear_f32_free();
#endif
    printf("{\"task\":\"%s\",\"backend\":\"%s\",\"seconds\":%.6f}\n",task,backend,
           end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)*1e-9);return 0;
usage:
    fprintf(stderr,"vhuman_models --task rmbg|cues|moge --model FILE --input CHW.f32 --output FILE "
                   "--height H --width W [--backend cpu|cuda] [--device N] [--threads N] "
                   "[--backbone FILE --grid-height H --grid-width W]\n");return 2;
}
#endif
