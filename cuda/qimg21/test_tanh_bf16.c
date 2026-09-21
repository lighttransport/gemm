/* Exhaustive BF16 tanh replay; compiled with the runner's NVRTC settings. */
#define main qimg21_runner_main
#include "test_cuda_qimg21_native.c"
#undef main

static const char *tanh_source =
"extern \"C\" __global__ void tanh_bf16(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=tanhf(x[i]);unsigned u=__float_as_uint(v);x[i]=__uint_as_float((u+0x7fff+((u>>16)&1))&0xffff0000);}}\n";

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s INPUT.npy OUTPUT.npy\n", argv[0]);
        return 2;
    }
    npy_f32 a = {0};
    if (npy_read_f32(argv[1], &a)) return 1;
    if (a.n > 65536 || a.ndim != 2 || a.shape[1] != 1) { npy_free(&a); return 2; }
    cuda_qimg_runner *r = cuda_qimg_init(0, 1);
    if (!r) { npy_free(&a); return 1; }
    CUmodule module = NULL;
    CUfunction function;
    CUdeviceptr data = 0;
    int rc = 1;
    if (cu_compile_kernels(&module, r->device, tanh_source, "qimg21_tanh.cu", 1, "qimg21_tanh") < 0 ||
        cuModuleGetFunction(&function, module, "tanh_bf16")) goto done;
    data = checked_cuMemAlloc(a.n * 4);
    if (!data || cuMemcpyHtoD(data, a.data, a.n * 4) || cuCtxSynchronize() ||
        launch_vec(function, r->stream, (int)a.n, data) || cuMemcpyDtoH(a.data, data, a.n * 4)) goto done;
    rc = npy_write_f32(argv[2], a.data, a.n, (int)a.n, 1);
done:
    free_d(&data);
    if (module) cuModuleUnload(module);
    cuda_qimg_free(r);
    npy_free(&a);
    return rc;
}
