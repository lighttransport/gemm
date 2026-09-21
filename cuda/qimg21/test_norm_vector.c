#define main qimg21_original_main
#include "test_cuda_qimg21_native.c"
#undef main
#include "norm_vector_kernels.h"

int main(int argc,char **argv) {
    if(argc!=3)return 2;
    char path[2048];npy_f32 x={0},m={0},text={0};int rc=1;
    snprintf(path,sizeof(path),"%s/hidden0.npy",argv[1]);if(npy_read_f32(path,&x))goto host_done;
    snprintf(path,sizeof(path),"%s/mod.npy",argv[1]);if(npy_read_f32(path,&m))goto host_done;
    snprintf(path,sizeof(path),"%s/txt_input.npy",argv[1]);if(npy_read_f32(path,&text))goto host_done;
    if(x.ndim!=2||x.shape[1]!=4096||m.n!=32768||text.ndim!=2||text.shape[0]>=x.shape[0])goto host_done;
    cuda_qimg_runner*r=cuda_qimg_init(0,1);if(!r)goto host_done;
    CUmodule module=NULL;CUfunction fn;CUdeviceptr dx=0,dm=0,dy=0;
    if(cu_compile_kernels(&module,r->device,q21_norm_vector_src,"qimg21_norm_vector.cu",1,"qimg21_norm_vector")<0||
       cuModuleGetFunction(&fn,module,"mod_ln_vector"))goto done;
    dx=checked_cuMemAlloc(x.n*4);dy=checked_cuMemAlloc(x.n*4);dm=checked_cuMemAlloc(m.n*4);
    if(!dx||!dy||!dm||cuMemcpyHtoD(dx,x.data,x.n*4)||cuMemcpyHtoD(dm,m.data,m.n*4)||cuCtxSynchronize())goto done;
    int n=x.shape[0],d=4096,p=text.shape[0],which=0;
    void*args[]={&dy,&dx,&dm,&n,&d,&p,&which};
    if(cuLaunchKernel(fn,n,1,1,128,1,1,0,r->stream,args,NULL)||cuCtxSynchronize()||cuMemcpyDtoH(x.data,dy,x.n*4))goto done;
    rc=npy_write_f32(argv[2],x.data,x.n,n,d);
done:
    free_d(&dx);free_d(&dm);free_d(&dy);if(module)cuModuleUnload(module);cuda_qimg_free(r);
host_done:
    npy_free(&x);npy_free(&m);npy_free(&text);return rc;
}
