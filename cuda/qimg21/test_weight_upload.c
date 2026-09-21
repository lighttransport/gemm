/* Guard-page regression for F32/BF16 safetensors uploads. Requires CUDA. */
#define main qimg21_runner_main
#include "test_cuda_qimg21_native.c"
#undef main
#include <unistd.h>

int main(void) {
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) return 1;
    unsigned char *memory = mmap(NULL, (size_t)page * 2, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) return 1;
    if (mprotect(memory + page, (size_t)page, PROT_NONE)) return 1;
    cuda_qimg_runner *r = cuda_qimg_init(0, 0);
    if (!r) return 1;
    st_tensor_info tensor = {.name="test", .shape={4}, .n_dims=1};
    st_context context = {.tensors=&tensor, .n_tensors=1};
    qimg21_shards shards = {.st={&context}, .n=1};
    const float values[4] = {1.0f, -2.0f, 1.00390625f, 1.01171875f};
    uint16_t expected[4], actual[4];
    for (int i=0;i<4;i++) expected[i]=qimg_f32_to_bf16_rne(values[i]);
    int failed=0;
    for (int bf16=0;bf16<2;bf16++) {
        strcpy(tensor.dtype_str,bf16 ? "BF16" : "F32");
        tensor.nbytes=bf16 ? sizeof(expected) : sizeof(values);
        context.data=memory+page-tensor.nbytes;
        memcpy(context.data,bf16 ? (const void *)expected : (const void *)values,
               tensor.nbytes);
        CUdeviceptr data=upload_bf16(&shards,"test");
        if (!data || cuMemcpyDtoH(actual,data,sizeof(actual))!=CUDA_SUCCESS ||
            memcmp(actual,expected,sizeof(actual))) failed=1;
        if(data)cuMemFree(data);
    }
    cuda_qimg_free(r);
    munmap(memory,(size_t)page*2);
    puts(failed ? "weight upload: FAIL" : "weight upload: PASS (F32/BF16 guard page)");
    return failed;
}
