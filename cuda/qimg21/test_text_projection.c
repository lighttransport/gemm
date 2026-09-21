/* Isolate the denoiser text projection, not the Qwen3-VL text encoder. */
#define main qimg21_original_main
#include "test_cuda_qimg21_native.c"
#undef main

int main(int argc,char **argv) {
    if(argc!=4 && !(argc==5&&(!strcmp(argv[4],"--ordered-gelu")||!strcmp(argv[4],"--legacy-gelu"))))return 2;
    if(mkdir(argv[3],0755))return 2;
    npy_f32 input={0};if(npy_read_f32(argv[2],&input))return 1;
    int nt=input.ndim==3?(int)input.shape[1]:(int)input.shape[0],d=4096;
    if(input.n!=(size_t)nt*d)return 2;
    qimg21_stage_dir=argv[3];
    cuda_qimg_runner*r=cuda_qimg_init(0,1);if(!r)return 1;
    qimg21_shards shards={{0},0};char path[2048];int rc=1;
    CUmodule module=NULL;qimg21_kernels kernels;
    CUdeviceptr x=0,y=0,bf=0,wn=0,wi=0,wo=0;
    for(int i=1;i<=2;i++) {
        snprintf(path,sizeof(path),"%s/transformer/diffusion_pytorch_model-%05d-of-00002.safetensors",argv[1],i);
        shards.st[shards.n]=safetensors_open(path);if(!shards.st[shards.n])goto done;shards.n++;
    }
    if(cublasewCreate(&r->cublaslt_ctx,r->stream) ||
       cu_compile_kernels(&module,r->device,qimg21_src,"qimg21_native.cu",1,"qimg21_native")<0 ||
       get_kernel(&kernels,module))goto done;
    if(argc==5 && cuModuleGetFunction(&kernels.gelu,module,
        !strcmp(argv[4],"--legacy-gelu")?"gelu_tanh":"gelu_tanh_ordered"))goto done;
    x=checked_cuMemAlloc(input.n*4);y=checked_cuMemAlloc(input.n*4);bf=checked_cuMemAlloc(input.n*2);
    wn=upload_f32(&shards,"txt_in.text_norm.weight");wi=upload_bf16(&shards,"txt_in.in_layer.weight");
    wo=upload_bf16(&shards,"txt_in.out_layer.weight");
    if(!x||!y||!bf||!wn||!wi||!wo||cuMemcpyHtoD(x,input.data,input.n*4)||cuCtxSynchronize())goto done;
    float eps=1e-6f;void *a[]={&y,&x,&wn,&nt,&d,&eps};
    if(cuLaunchKernel(kernels.zero_rms,nt,1,1,256,1,1,256*4,r->stream,a,NULL)||cuCtxSynchronize() ||
       launch_vec(kernels.round_bf16,r->stream,nt*d,y))goto done;
    dump_stage("norm",y,input.n,nt,d);
    if(launch_cast(r,bf,y,nt*d)||gemm(r,x,wi,bf,nt,d,d)||launch_vec(kernels.round_bf16,r->stream,nt*d,x))goto done;
    dump_stage("in",x,input.n,nt,d);
    if(launch_vec(kernels.gelu,r->stream,nt*d,x)||launch_vec(kernels.round_bf16,r->stream,nt*d,x))goto done;
    dump_stage("gelu",x,input.n,nt,d);
    if(launch_cast(r,bf,x,nt*d)||gemm(r,y,wo,bf,nt,d,d)||launch_vec(kernels.round_bf16,r->stream,nt*d,y))goto done;
    dump_stage("out",y,input.n,nt,d);rc=0;
done:
    free_d(&x);free_d(&y);free_d(&bf);free_d(&wn);free_d(&wi);free_d(&wo);
    if(module)cuModuleUnload(module);
    for(int i=0;i<shards.n;i++)safetensors_close(shards.st[i]);
    cuda_qimg_free(r);npy_free(&input);return rc;
}
