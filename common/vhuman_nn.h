/* Small CHW inference primitives shared by vhuman model ports.
 * Requires safetensors + swin_linear (repository GEMM) from swin_runner.c.
 * This is an executable-internal API: malformed assets terminate the child.
 */
#ifndef VHUMAN_NN_H
#define VHUMAN_NN_H
typedef struct { float *d; int c, h, w; } vh_tensor;
static void vh_fail(const char *s) { fprintf(stderr,"vhuman: %s\n",s); exit(3); }
static size_t vh_size(vh_tensor x) { return (size_t)x.c*x.h*x.w; }
static vh_tensor vh_new(int c,int h,int w)
{
    if (c<1 || c>16384 || h<1 || w<1 || h>2048 || w>2048 || (size_t)c*h*w>536870912)
        vh_fail("unsupported tensor dimensions");
    return (vh_tensor){swin_alloc((size_t)c*h*w*sizeof(float)),c,h,w};
}
static void vh_drop(vh_tensor x) { free(x.d); }
static vh_tensor vh_copy(vh_tensor x)
{ vh_tensor y=vh_new(x.c,x.h,x.w); memcpy(y.d,x.d,vh_size(x)*sizeof(float)); return y; }
static int vh_index(st_context *st,const char *name)
{
    int i=safetensors_find(st,name);
    if (i<0 || strcmp(safetensors_dtype(st,i),"F32")) vh_fail(name);
    size_t n=1;
    for(int d=0;d<safetensors_ndims(st,i);d++) {
        uint64_t v=safetensors_shape(st,i)[d];
        if(!v || v>536870912 || n>536870912/v) vh_fail(name);
        n*=v;
    }
    if(n*sizeof(float)!=safetensors_nbytes(st,i)) vh_fail(name);
    return i;
}
static const float *vh_param(st_context *s,const char *name,int n)
{
    int i=vh_index(s,name);
    if(safetensors_ndims(s,i)!=1 || safetensors_shape(s,i)[0]!=(uint64_t)n) vh_fail(name);
    return safetensors_data(s,i);
}
static void vh_name(char out[512],const char *prefix,const char *suffix)
{ if(snprintf(out,512,"%s.%s",prefix,suffix)>=512) vh_fail("tensor name too long"); }
static float vh_sigmoid(float x)
{ return x>=0 ? 1/(1+expf(-x)) : expf(x)/(1+expf(x)); }
static void vh_act(vh_tensor x,int kind)
{
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<vh_size(x);i++) {
        float v=x.d[i];
        x.d[i]=kind==1 ? fmaxf(0,v) : kind==2 ? v*vh_sigmoid(v) : vh_sigmoid(v);
    }
}
static float vh_sample(vh_tensor x,int c,int y,int col,int replicate)
{
    if(replicate) { y=y<0?0:y>=x.h?x.h-1:y; col=col<0?0:col>=x.w?x.w-1:col; }
    return y<0 || y>=x.h || col<0 || col>=x.w ? 0 : x.d[((size_t)c*x.h+y)*x.w+col];
}
static vh_tensor vh_conv(st_context *s,const char *prefix,vh_tensor x,int stride,int pad,int replicate)
{
    char name[512]; vh_name(name,prefix,"weight"); int wi=vh_index(s,name);
    const uint64_t *sh=safetensors_shape(s,wi);
    if(safetensors_ndims(s,wi)!=4 || sh[1]!=(uint64_t)x.c || sh[2]>7 || sh[3]>7) vh_fail(name);
    int co=sh[0],kh=sh[2],kw=sh[3],k=x.c*kh*kw;
    vh_name(name,prefix,"bias"); int bi=safetensors_find(s,name);
    const float *b=bi<0?NULL:vh_param(s,name,co), *weight=safetensors_data(s,wi);
    vh_tensor y=vh_new(co,(x.h+2*pad-kh)/stride+1,(x.w+2*pad-kw)/stride+1);
    int tile=256,total=y.h*y.w;
    float *a=swin_alloc((size_t)tile*k*sizeof(float)),*out=swin_alloc((size_t)tile*co*sizeof(float));
    for(int start=0;start<total;start+=tile) {
        int n=total-start<tile?total-start:tile;
        #pragma omp parallel for schedule(static)
        for(int i=0;i<n;i++) {
            int py=(start+i)/y.w*stride-pad,px=(start+i)%y.w*stride-pad;
            for(int c=0;c<x.c;c++) for(int yy=0;yy<kh;yy++) for(int xx=0;xx<kw;xx++)
                a[(size_t)i*k+(c*kh+yy)*kw+xx]=vh_sample(x,c,py+yy,px+xx,replicate);
        }
        swin_linear(out,weight,b,a,n,co,k);
        for(int i=0;i<n;i++) for(int c=0;c<co;c++) y.d[(size_t)c*total+start+i]=out[(size_t)i*co+c];
    }
    free(a);free(out);return y;
}
static void vh_bn(st_context *s,const char *prefix,vh_tensor x)
{
    char name[512];vh_name(name,prefix,"weight");const float *w=vh_param(s,name,x.c);
    vh_name(name,prefix,"bias");const float *b=vh_param(s,name,x.c);
    vh_name(name,prefix,"running_mean");const float *m=vh_param(s,name,x.c);
    vh_name(name,prefix,"running_var");const float *v=vh_param(s,name,x.c);
    #pragma omp parallel for schedule(static)
    for(int c=0;c<x.c;c++) {
        float scale=w[c]/sqrtf(v[c]+1e-5f),bias=b[c]-m[c]*scale;
        for(int i=0;i<x.h*x.w;i++) x.d[(size_t)c*x.h*x.w+i]=x.d[(size_t)c*x.h*x.w+i]*scale+bias;
    }
}
static vh_tensor vh_resize(vh_tensor x,int h,int w,int align)
{
    vh_tensor y=vh_new(x.c,h,w);
    float sy=align?(h>1?(float)(x.h-1)/(h-1):0):(float)x.h/h;
    float sx=align?(w>1?(float)(x.w-1)/(w-1):0):(float)x.w/w;
    #pragma omp parallel for schedule(static)
    for(int c=0;c<x.c;c++) for(int yy=0;yy<h;yy++) {
        float py=fmaxf(0,align?sy*yy:sy*(yy+.5f)-.5f);int iy=(int)py;float fy=py-iy;
        for(int xx=0;xx<w;xx++) {
            float px=fmaxf(0,align?sx*xx:sx*(xx+.5f)-.5f);int ix=(int)px;float fx=px-ix;
            float a=(1-fx)*vh_sample(x,c,iy,ix,1)+fx*vh_sample(x,c,iy,ix+1,1);
            float b=(1-fx)*vh_sample(x,c,iy+1,ix,1)+fx*vh_sample(x,c,iy+1,ix+1,1);
            y.d[((size_t)c*h+yy)*w+xx]=(1-fy)*a+fy*b;
        }
    }
    return y;
}
static vh_tensor vh_cat(vh_tensor a,vh_tensor b)
{
    if(a.h!=b.h || a.w!=b.w) vh_fail("concat shape mismatch");
    vh_tensor y=vh_new(a.c+b.c,a.h,a.w);
    memcpy(y.d,a.d,vh_size(a)*4);memcpy(y.d+vh_size(a),b.d,vh_size(b)*4);return y;
}
static void vh_add(vh_tensor a,vh_tensor b)
{
    if(a.c!=b.c || a.h!=b.h || a.w!=b.w) vh_fail("add shape mismatch");
    for(size_t i=0;i<vh_size(a);i++) a.d[i]+=b.d[i];
}
static float vh_bilinear_zero(vh_tensor x,int c,float y,float col)
{
    /* torchvision deform_conv2d excludes samples outside (-1,H)x(-1,W). */
    if(!isfinite(y) || !isfinite(col))vh_fail("nonfinite deformable-convolution coordinates");
    if(y<=-1 || y>=x.h || col<=-1 || col>=x.w) return 0;
    int iy=(int)floorf(y),ix=(int)floorf(col);float fy=y-iy,fx=col-ix;
    return (1-fy)*((1-fx)*vh_sample(x,c,iy,ix,0)+fx*vh_sample(x,c,iy,ix+1,0))+
           fy*((1-fx)*vh_sample(x,c,iy+1,ix,0)+fx*vh_sample(x,c,iy+1,ix+1,0));
}
static vh_tensor vh_deform(st_context *s,const char *prefix,vh_tensor x,int kernel)
{
    char name[512];int pad=kernel/2,kk=kernel*kernel;
    vh_name(name,prefix,"offset_conv");vh_tensor offset=vh_conv(s,name,x,1,pad,0);
    vh_name(name,prefix,"modulator_conv");vh_tensor mask=vh_conv(s,name,x,1,pad,0);
    if(offset.c!=2*kk || mask.c!=kk || offset.h!=x.h || offset.w!=x.w || mask.h!=x.h || mask.w!=x.w)
        vh_fail("deform offsets/mask shape");
    vh_name(name,prefix,"regular_conv.weight");int wi=vh_index(s,name);
    const uint64_t *sh=safetensors_shape(s,wi);
    if(safetensors_ndims(s,wi)!=4 || sh[1]!=(uint64_t)x.c || sh[2]!=(uint64_t)kernel || sh[3]!=(uint64_t)kernel) vh_fail(name);
    int co=sh[0],n=x.h*x.w,k=x.c*kk,tile=128;
    vh_tensor y=vh_new(co,x.h,x.w);
    float *a=swin_alloc((size_t)tile*k*4),*o=swin_alloc((size_t)tile*co*4);
    for(int start=0;start<n;start+=tile) {
        int count=n-start<tile?n-start:tile;
        #pragma omp parallel for schedule(static)
        for(int i=0;i<count;i++) for(int c=0;c<x.c;c++) for(int j=0;j<kk;j++) {
            int p=start+i;
            float yy=p/x.w-pad+j/kernel+offset.d[(size_t)(2*j)*n+p];
            float xx=p%x.w-pad+j%kernel+offset.d[(size_t)(2*j+1)*n+p];
            a[(size_t)i*k+c*kk+j]=vh_bilinear_zero(x,c,yy,xx)*2*vh_sigmoid(mask.d[(size_t)j*n+p]);
        }
        swin_linear(o,safetensors_data(s,wi),NULL,a,count,co,k);
        for(int i=0;i<count;i++)for(int c=0;c<co;c++)y.d[(size_t)c*n+start+i]=o[(size_t)i*co+c];
    }
    free(a);free(o);vh_drop(offset);vh_drop(mask);return y;
}
static void vh_write(const char *path,vh_tensor x)
{
    for(size_t i=0;i<vh_size(x);i++) if(!isfinite(x.d[i])) vh_fail("nonfinite output");
    FILE *f=fopen(path,"wb");if(!f)vh_fail(path);
    int bad=fwrite(x.d,4,vh_size(x),f)!=vh_size(x);bad|=fclose(f)!=0;
    if(bad)vh_fail("output write failed");
}
static vh_tensor vh_read(const char *path,int c,int h,int w)
{
    vh_tensor x=vh_new(c,h,w);FILE *f=fopen(path,"rb");if(!f)vh_fail(path);
    int bad=fread(x.d,4,vh_size(x),f)!=vh_size(x) || fgetc(f)!=EOF;fclose(f);
    for(size_t i=0;!bad && i<vh_size(x);i++)bad=!isfinite(x.d[i]);
    if(bad)vh_fail("invalid finite F32 input size/values");
    return x;
}
#endif
