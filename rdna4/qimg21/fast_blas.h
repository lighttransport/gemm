/* BF16-output gfx12 WMMA boundary for the shared resident graph. */
typedef struct { void *library; CUstream stream; int (*bf16)(void*,int,const void*,const void*,int,int,int,int,void*); } cublasew_context;
static int cublasewCreate(cublasew_context **out, CUstream stream) {
    cublasew_context *c = calloc(1,sizeof(*c));
    if (!c) return -1;
    c->library=dlopen("rdna4/qimg21/libq21_hip_fast_gemm.so",RTLD_NOW|RTLD_LOCAL);
    if (c->library) *(void**)&c->bf16=dlsym(c->library,"q21f_bf16_gemm");
    if (!c->bf16) { if(c->library)dlclose(c->library);free(c);return -1; }
    c->stream=stream; *out=c; return 0;
}
static int cublasew_disallow_reduced_precision_reduction(cublasew_context *c) { return c ? 0 : -1; }
static int cublasew_gemm_bf16_bf16_bf16_rowmajor_nt_ld(cublasew_context*c,CUdeviceptr y,int ldy,CUdeviceptr w,CUdeviceptr x,int ldx,int m,int n,int k) {
    return c->bf16((void*)(uintptr_t)y,ldy,(const void*)(uintptr_t)w,(const void*)(uintptr_t)x,ldx,m,n,k,c->stream);
}
/* INT8 always uses the fused plugin; reject unavailable implementations. */
static int cublasew_gemm_int8_s32_rowmajor_nt(cublasew_context*c,CUdeviceptr y,CUdeviceptr w,CUdeviceptr x,int m,int n,int k) {
    (void)c;(void)y;(void)w;(void)x;(void)m;(void)n;(void)k;return -1;
}

static void q21f_hip_blas_destroy(cublasew_context *c) {
    if(c){if(c->library)dlclose(c->library);free(c);}
}
