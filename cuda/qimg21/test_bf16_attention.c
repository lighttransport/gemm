/* Replay saved editing Q/K/V with the existing native BF16 tensor-core kernel. */
#define main qimg21_original_main
#include "test_cuda_qimg21_native.c"
#undef main
#include "mma64_kernels.h"

int main(int argc, char **argv) {
    if(argc!=4 && !(argc==5 && !strcmp(argv[4],"--reverse64"))){
        fprintf(stderr,"usage: %s STAGE_DIR LAYOUT.txt OUT.npy [--reverse64]\n",argv[0]);return 2;
    }
    q21_joint_layout layout={0};
    int nt,ih,iw,rc=1;
    if(q21_layout_read(argv[2],&layout,&nt,&ih,&iw))return 2;
    npy_f32 input[3]={{0}};
    const char *names[]={"rope_q","rope_k","v"};
    char path[2048];
    for(int i=0;i<3;i++) {
        snprintf(path,sizeof(path),"%s/%s.npy",argv[1],names[i]);
        if(npy_read_f32(path,&input[i]) || input[i].ndim!=2 ||
           input[i].shape[0]!=(size_t)layout.n || input[i].shape[1]!=4096)goto host_done;
    }
    cuda_qimg_runner *r=cuda_qimg_init(0,1);
    if(!r)goto host_done;
    CUdeviceptr packed[3]={0},scratch=0,output=0;
    CUfunction attention;
    CUmodule module=NULL;
    CUmodule mma_module=NULL;
    int shared_bytes=4*32*136*2;
    qimg21_kernels kernels;
    size_t count=(size_t)layout.n*4096;
    if(cuModuleGetFunction(&attention,r->module,"flash_attn_bf16_xq"))goto done;
    if(argc==5) {
        shared_bytes=4*64*136*2;
        if(cu_compile_kernels(&mma_module,r->device,q21_mma64_src,"qimg21_mma64.cu",1,"qimg21_mma64")<0 ||
           cuModuleGetFunction(&attention,mma_module,"q21_flash_reverse64") ||
           cuFuncSetAttribute(attention,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared_bytes))goto done;
    }
    if(cu_compile_kernels(&module,r->device,qimg21_src,"qimg21_native.cu",1,"qimg21_native")<0 ||
       get_kernel(&kernels,module))goto done;
    scratch=checked_cuMemAlloc(count*4);output=checked_cuMemAlloc(count*4);
    if(!scratch || !output)goto done;
    for(int i=0;i<3;i++) {
        packed[i]=checked_cuMemAlloc(count*2);
        if(!packed[i] || cuMemcpyHtoD(scratch,input[i].data,count*4) || cuCtxSynchronize() ||
           launch_cast(r,packed[i],scratch,(int)count))goto done;
    }
    for(int start=0;start<layout.n;) {
        int end=start+1;
        if(layout.image_id[start]>=0)
            while(end<layout.n && layout.image_id[end]==layout.image_id[start])end++;
        int nq=end-start,nkv=end,heads=32,hd=128;
        CUdeviceptr q=packed[0]+(size_t)start*4096*2;
        CUdeviceptr out=output+(size_t)start*4096*4;
        void *args[]={&out,&q,&packed[1],&packed[2],&nq,&nkv,&heads,&hd};
        if(cuLaunchKernel(attention,heads,(nq+63)/64,1,128,1,1,shared_bytes,r->stream,args,NULL) ||
           cuCtxSynchronize())goto done;
        start=end;
    }
    if(launch_vec(kernels.round_bf16,r->stream,(int)count,output) ||
       cuMemcpyDtoH(input[0].data,output,count*4))goto done;
    rc=npy_write_f32(argv[3],input[0].data,count,layout.n,4096);
done:
    for(int i=0;i<3;i++)free_d(&packed[i]);
    free_d(&scratch);free_d(&output);
    if(module)cuModuleUnload(module);
    if(mma_module)cuModuleUnload(mma_module);
    cuda_qimg_free(r);
host_done:
    for(int i=0;i<3;i++)npy_free(&input[i]);
    q21_layout_free(&layout);
    return rc;
}
