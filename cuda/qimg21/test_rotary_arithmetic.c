#define main qimg21_original_main
#include "test_cuda_qimg21_native.c"
#undef main

static const char *rotary_src =
"extern \"C\" __global__ void rotate(float*y,const float*x,const float*f,int n,int mode){\n"
" int z=(blockIdx.x*blockDim.x+threadIdx.x)*2;if(z>=n)return;int t=z/4096,j=z%128;\n"
" float a=x[z],b=x[z+1],c=f[t*128+j],s=f[t*128+j+1],u,v;\n"
" if(mode==0){u=fmaf(a,c,-b*s);v=fmaf(a,s,b*c);}\n"
" else if(mode==1){u=__fsub_rn(__fmul_rn(a,c),__fmul_rn(b,s));v=__fadd_rn(__fmul_rn(a,s),__fmul_rn(b,c));}\n"
" else {u=(float)((double)a*c-(double)b*s);v=(float)((double)a*s+(double)b*c);}\n"
" unsigned q=__float_as_uint(u);y[z]=__uint_as_float((q+0x7fff+((q>>16)&1))&0xffff0000);\n"
" q=__float_as_uint(v);y[z+1]=__uint_as_float((q+0x7fff+((q>>16)&1))&0xffff0000);}\n";

int main(int argc,char **argv) {
    if(argc!=3 && argc!=6)return 2;
    int mode=atoi(argv[2]);if(mode<0||mode>2)return 2;
    char path[2048];npy_f32 x={0},f={0};int rc=1;
    snprintf(path,sizeof(path),"%s/normalized.npy",argv[1]);if(npy_read_f32(path,&x))goto host_done;
    snprintf(path,sizeof(path),"%s/frequencies.npy",argv[1]);if(npy_read_f32(path,&f))goto host_done;
    if(x.ndim!=2||x.shape[1]!=4096||f.ndim!=2||f.shape[1]!=128||f.shape[0]!=x.shape[0])goto host_done;
    if(argc==6) {
        int prefix=atoi(argv[3]),height=atoi(argv[4]),width=atoi(argv[5]);
        if(prefix<1||height<1||height>1024||width<1||width>1024||
           (size_t)(prefix+height*width)!=x.shape[0])goto host_done;
        for(int t=0;t<(int)x.shape[0];t++)for(int j=0;j<128;j+=2) {
            int axis=j<16?16:56,off=j<16?0:(j<72?16:72);
            int pos=t<prefix?t:(j<16?prefix:(j<72?-(height-height/2)+(t-prefix)/width:
                                                       -(width-width/2)+(t-prefix)%width));
            float inverse=1.f/powf(10000.f,(float)(j-off)/(float)axis);
            float angle=(float)pos*inverse;
            f.data[t*128+j]=cosf(angle);f.data[t*128+j+1]=sinf(angle);
        }
        snprintf(path,sizeof(path),"%s/host_frequencies.npy",argv[1]);
        if(npy_write_f32(path,f.data,f.n,f.shape[0],128))goto host_done;
    }
    cuda_qimg_runner*r=cuda_qimg_init(0,1);if(!r)goto host_done;
    CUmodule m=NULL;CUfunction fn;CUdeviceptr dx=0,df=0,dy=0;
    if(cu_compile_kernels(&m,r->device,rotary_src,"qimg21_rotary_arithmetic.cu",1,"qimg21_rotary_arithmetic")<0||cuModuleGetFunction(&fn,m,"rotate"))goto done;
    dx=checked_cuMemAlloc(x.n*4);dy=checked_cuMemAlloc(x.n*4);df=checked_cuMemAlloc(f.n*4);
    if(!dx||!dy||!df||cuMemcpyHtoD(dx,x.data,x.n*4)||cuMemcpyHtoD(df,f.data,f.n*4)||cuCtxSynchronize())goto done;
    int n=x.n;void *a[]={&dy,&dx,&df,&n,&mode};
    if(cuLaunchKernel(fn,(n+511)/512,1,1,256,1,1,0,r->stream,a,NULL)||cuCtxSynchronize()||cuMemcpyDtoH(x.data,dy,x.n*4))goto done;
    snprintf(path,sizeof(path),"%s/%smode_%d.npy",argv[1],argc==6?"host_":"",mode);rc=npy_write_f32(path,x.data,x.n,x.shape[0],4096);
done:
    free_d(&dx);free_d(&dy);free_d(&df);if(m)cuModuleUnload(m);cuda_qimg_free(r);
host_done:
    npy_free(&x);npy_free(&f);return rc;
}
