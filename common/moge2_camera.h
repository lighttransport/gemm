/* MoGe-2 ViT-L camera path: DINOv2, neck, affine points and mask heads.
 * Metric scale is not used in camera recovery and is deliberately not run.
 */
#ifndef MOGE2_CAMERA_H
#define MOGE2_CAMERA_H
static void vh_dino_linear(float *y,const float *w,const float *b,const float *x,int m,int n,int k,int threads)
{ (void)threads;swin_linear(y,w,b,x,m,n,k); }
#define DINOV2_GEMM_F32 vh_dino_linear
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define DINOV2_IMPLEMENTATION
#include "dinov2.h"

/* Separable antialiased triangle resize, matching interpolate(...,
 * bilinear, align_corners=False, antialias=True), including edge renormalization. */
static vh_tensor vh_resize_aa(vh_tensor x,int h,int w)
{
    vh_tensor a=vh_new(x.c,x.h,w),y=vh_new(x.c,h,w);
    for(int axis=0;axis<2;axis++) {
        int ni=axis?x.h:x.w,no=axis?h:w;
        float scale=(float)ni/no,support=fmaxf(scale,1);
        vh_tensor in=axis?a:x,out=axis?y:a;
        #pragma omp parallel for schedule(static)
        for(int c=0;c<x.c;c++) for(int p=0;p<no;p++) {
            float center=(p+.5f)*scale;
            int lo=(int)floorf(center-support+.5f),hi=(int)floorf(center+support+.5f);
            lo=lo<0?0:lo;hi=hi>ni?ni:hi;
            float total=0;for(int i=lo;i<hi;i++)total+=fmaxf(0,1-fabsf((i+.5f-center)/support));
            int lines=axis?w:x.h;
            for(int l=0;l<lines;l++) {
                float sum=0;
                for(int i=lo;i<hi;i++) {
                    float weight=fmaxf(0,1-fabsf((i+.5f-center)/support))/total;
                    sum+=weight*in.d[axis?((size_t)c*in.h+i)*in.w+l:((size_t)c*in.h+l)*in.w+i];
                }
                out.d[axis?((size_t)c*out.h+p)*out.w+l:((size_t)c*out.h+l)*out.w+p]=sum;
            }
        }
    }
    vh_drop(a);return y;
}
static vh_tensor moge_uv(int h,int w,float aspect)
{
    vh_tensor uv=vh_new(2,h,w);float sy=1/sqrtf(1+aspect*aspect),sx=aspect*sy;
    for(int y=0;y<h;y++)for(int x=0;x<w;x++) {
        uv.d[(size_t)y*w+x]=sx*(2*(x+.5f)/w-1);
        uv.d[(size_t)h*w+y*w+x]=sy*(2*(y+.5f)/h-1);
    }
    return uv;
}
static vh_tensor moge_transpose(st_context *s,const char *prefix,vh_tensor x)
{
    char name[512];vh_name(name,prefix,"weight");int wi=vh_index(s,name);
    const uint64_t *sh=safetensors_shape(s,wi);
    if(safetensors_ndims(s,wi)!=4 || sh[0]!=(uint64_t)x.c || sh[2]!=2 || sh[3]!=2)vh_fail(name);
    int co=sh[1],n=x.h*x.w;const float *weight=safetensors_data(s,wi);
    vh_name(name,prefix,"bias");const float *bias=vh_param(s,name,co);
    float *w=swin_alloc((size_t)co*4*x.c*4),*b=swin_alloc((size_t)co*4*4);
    for(int c=0;c<co;c++)for(int k=0;k<4;k++) {
        b[c*4+k]=bias[c];for(int i=0;i<x.c;i++)w[(size_t)(c*4+k)*x.c+i]=weight[((size_t)i*co+c)*4+k];
    }
    vh_tensor y=vh_new(co,x.h*2,x.w*2);
    int tile=256;float *a=swin_alloc((size_t)tile*x.c*4),*out=swin_alloc((size_t)tile*co*4*4);
    for(int start=0;start<n;start+=tile) {
        int count=n-start<tile?n-start:tile;
        for(int p=0;p<count;p++)for(int c=0;c<x.c;c++)a[(size_t)p*x.c+c]=x.d[(size_t)c*n+start+p];
        swin_linear(out,w,b,a,count,4*co,x.c);
        for(int p=0;p<count;p++)for(int c=0;c<co;c++)for(int k=0;k<4;k++)
            y.d[((size_t)c*y.h+(start+p)/x.w*2+k/2)*y.w+(start+p)%x.w*2+k%2]=out[(size_t)p*4*co+c*4+k];
    }
    free(w);free(b);free(a);free(out);return y;
}
static void moge_stack(st_context *s,const char *prefix,vh_tensor input[5],vh_tensor output[5],int neck)
{
    vh_tensor p={0};char name[512];
    for(int i=0;i<5;i++) {
        snprintf(name,sizeof(name),"%s.input_blocks.%d",prefix,i);vh_tensor feature=vh_conv(s,name,input[i],1,0,1);
        if(i==0)p=feature;else {vh_add(p,feature);vh_drop(feature);}
        int blocks=(i>0 && i<4)?(neck?2:1):0;
        for(int j=0;j<blocks;j++) {
            vh_tensor a=vh_copy(p);vh_act(a,1);
            snprintf(name,sizeof(name),"%s.res_blocks.%d.%d.layers.2",prefix,i,j);
            vh_tensor b=vh_conv(s,name,a,1,1,1);vh_drop(a);vh_act(b,1);
            snprintf(name,sizeof(name),"%s.res_blocks.%d.%d.layers.5",prefix,i,j);
            a=vh_conv(s,name,b,1,1,1);vh_drop(b);vh_add(p,a);vh_drop(a);
        }
        if(!neck && i==4) {snprintf(name,sizeof(name),"%s.output_blocks.4",prefix);output[i]=vh_conv(s,name,p,1,0,1);}
        else output[i]=vh_copy(p);
        if(i<4) {
            vh_tensor a;
            if(i<3) {snprintf(name,sizeof(name),"%s.resamplers.%d.0",prefix,i);a=moge_transpose(s,name,p);}
            else a=vh_resize(p,p.h*2,p.w*2,0);
            vh_drop(p);snprintf(name,sizeof(name),"%s.resamplers.%d.1",prefix,i);
            p=vh_conv(s,name,a,1,1,1);vh_drop(a);
        }
    }
    vh_drop(p);
}
static vh_tensor moge_camera_predict(st_context *s,const char *backbone,vh_tensor image,int gh,int gw,int threads)
{
    dinov2_model *e=dinov2_load_safetensors(backbone);if(!e)vh_fail("invalid MoGe DINOv2 weights");
    if(e->dim!=1024 || e->n_blocks!=24 || e->n_register || e->patch_size!=14 || e->n_heads!=16 || e->ffn_hidden!=4096)
        vh_fail("MoGe supports the ViT-L/14 checkpoint only");
    e->grid_h=gh;e->grid_w=gw;e->n_patches=gh*gw;e->n_tokens=gh*gw+1;
    vh_tensor x=vh_resize_aa(image,gh*14,gw*14);
    float mean[3]={.485f,.456f,.406f},std[3]={.229f,.224f,.225f};
    for(int c=0;c<3;c++)for(int i=0;i<x.h*x.w;i++)x.d[(size_t)c*x.h*x.w+i]=(x.d[(size_t)c*x.h*x.w+i]-mean[c])/std[c];
    float *features[4]={0};int layers[]={5,11,17,23};
    if(dinov2_intermediates_f32(e,x.d,x.w,x.h,layers,4,features,threads))vh_fail("MoGe DINOv2 failed");
    vh_drop(x);dinov2_free(e);
    vh_tensor feature={0};char name[512];
    for(int l=0;l<4;l++) {
        vh_tensor a=vh_new(1024,gh,gw);
        for(int i=0;i<gh*gw;i++)for(int c=0;c<1024;c++)a.d[(size_t)c*gh*gw+i]=features[l][(size_t)i*1024+c];
        free(features[l]);snprintf(name,sizeof(name),"encoder.output_projections.%d",l);
        vh_tensor b=vh_conv(s,name,a,1,0,0);vh_drop(a);
        if(l==0)feature=b;else {vh_add(feature,b);vh_drop(b);}
    }
    vh_tensor input[5],neck[5],points[5],mask[5];
    for(int i=0;i<5;i++) {
        input[i]=moge_uv(gh*(1<<i),gw*(1<<i),(float)image.w/image.h);
        if(i==0){vh_tensor cat=vh_cat(feature,input[i]);vh_drop(feature);vh_drop(input[i]);input[i]=cat;}
    }
    moge_stack(s,"neck",input,neck,1);for(int i=0;i<5;i++)vh_drop(input[i]);
    moge_stack(s,"points_head",neck,points,0);moge_stack(s,"mask_head",neck,mask,0);
    vh_tensor p=vh_resize(points[4],image.h,image.w,0),m=vh_resize(mask[4],image.h,image.w,0);
    for(int i=0;i<5;i++){vh_drop(neck[i]);vh_drop(points[i]);vh_drop(mask[i]);}
    if(p.c!=3 || m.c!=1)vh_fail("invalid MoGe camera heads");
    int n=p.h*p.w;
    for(int i=0;i<n;i++){float z=expf(p.d[2*(size_t)n+i]);p.d[i]*=z;p.d[n+i]*=z;p.d[2*(size_t)n+i]=z;}
    vh_act(m,3);vh_tensor result=vh_cat(p,m);vh_drop(p);vh_drop(m);return result;
}
#endif
