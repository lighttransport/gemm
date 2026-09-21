/* Experimental streamed Qwen3-VL text encoder, original BF16 checkpoint.
 * Tokenization stays external. Input is unpadded text-only token IDs, one
 * whitespace-separated integer per token. No vision, KV cache, or LM head.
 * Reuse the denoiser's checked weight upload and BF16 GEMM infrastructure. */
#define main qimg21_denoiser_main
#include "test_cuda_qimg21_native.c"
#undef main
#include "text_kernels.h"

static int text_bf16_gemm_output = 1;

static int text_linear(cuda_qimg_runner *r, qimg21_kernels *k,
                       const qimg21_shards *s, const char *name,
                       CUdeviceptr out, CUdeviceptr in_bf, int n, int no, int ni) {
    int idx;
    st_context *st = find_tensor(s, name, &idx);
    if (!st || strcmp(safetensors_dtype(st, idx), "BF16") || safetensors_ndims(st, idx) != 2 ||
        safetensors_shape(st, idx)[0] != (uint64_t)no || safetensors_shape(st, idx)[1] != (uint64_t)ni ||
        safetensors_nbytes(st, idx) != (size_t)no * ni * 2) {
        fprintf(stderr, "text: unsupported/missing matrix %s\n", name);
        return -1;
    }
    CUdeviceptr w = upload_bf16(s, name);
    if (!w) return -1;
    /* The cuBLAS handle owns a separate stream. Pageable host-to-device
     * copies may return after staging, before the device transfer finishes. */
    int rc = cuCtxSynchronize();
    if (!rc && text_bf16_gemm_output) {
        CUdeviceptr result = checked_cuMemAlloc((size_t)n * no * 2), bias = 0;
        if (!result) { free_d(&w); return -1; }
        rc = cublasew_gemm_bf16_bf16_bf16_rowmajor_nt(r->cublaslt_ctx, result, w, in_bf, n, no, ni);
        if (!rc) rc = cuCtxSynchronize();
        void *args[] = {&out, &result, &bias, &no, &n};
        if (!rc) rc = cuLaunchKernel(r->bf16_to_f32_add_bias, (n * no + 255) / 256, 1, 1,
                                     256, 1, 1, 0, r->stream, args, NULL);
        if (!rc) rc = cuCtxSynchronize();
        free_d(&result);
    } else if (!rc) rc = gemm(r, out, w, in_bf, n, no, ni);
    if (!rc) rc = launch_vec(k->round_bf16, r->stream, n * no, out);
    free_d(&w);
    return rc;
}

static int text_norm(cuda_qimg_runner *r, CUfunction fn, const qimg21_shards *s,
                     const char *name, CUdeviceptr out, CUdeviceptr in, int rows, int d) {
    int idx;
    st_context *st = find_tensor(s, name, &idx);
    if (!st || strcmp(safetensors_dtype(st, idx), "BF16") ||
        safetensors_nbytes(st, idx) != (size_t)d * 2) return -1;
    CUdeviceptr w = upload_f32(s, name);
    if (!w) return -1;
    void *a[] = {&out, &in, &w, &d};
    int rc = cuCtxSynchronize();
    if (!rc) rc = cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, r->stream, a, NULL);
    if (!rc) rc = cuStreamSynchronize(r->stream);
    free_d(&w);
    return rc;
}

int main(int argc, char **argv) {
    const char *model = NULL, *tokens = NULL, *out = NULL, *dump_dir = NULL;
    int drop = 0, layers = 36;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--dump-dir") && i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--bf16-gemm-output")) text_bf16_gemm_output = 1;
        else if (!strcmp(argv[i], "--f32-gemm-output")) text_bf16_gemm_output = 0;
        else if (!strcmp(argv[i], "--drop-prefix") && i + 1 < argc) drop = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-layers") && i + 1 < argc) layers = atoi(argv[++i]);
        else { fprintf(stderr, "text: unknown/incomplete option %s\n", argv[i]); return 2; }
    }
    if (!model || !tokens || !out || drop < 0 || layers < 1 || layers > 36) {
        fprintf(stderr, "usage: %s --model DIR --tokens ids.txt --out embeds.npy "
                        "[--drop-prefix N --max-layers 36]\n", argv[0]); return 2;
    }
    int ids[4096], n = 0, scanned;
    char word[64];
    FILE *fp = fopen(tokens, "r");
    if (!fp) return 1;
    while ((scanned = fscanf(fp, "%63s", word)) == 1) {
        char *end;
        errno = 0;
        long token = strtol(word, &end, 10);
        if (errno || *end || n == 4096 || token < 0 || token >= 151936 ||
            token == 151655 || token == 151656 || token == 151652 || token == 151653) {
            fprintf(stderr, "text: invalid token/vision input or more than 4096 tokens\n");
            fclose(fp); return 1;
        }
        ids[n++] = token;
    }
    fclose(fp);
    if (scanned != EOF || n <= drop) return 1;

    int rc = 1;
    qimg21_shards shards = {{0}, 0};
    cuda_qimg_runner *r = NULL;
    CUmodule module = NULL, base_module = NULL;
    CUdeviceptr x=0, norm=0, bf=0, q=0, key=0, v=0, att=0, tmp=0, gate=0, up=0;
    float *host = calloc((size_t)n * 4096, sizeof(float));
    if (!host) return 1;
    char path[2048], name[256];
    for (int i = 1; i <= 4; i++) {
        snprintf(path, sizeof(path), "%s/text_encoder/model-%05d-of-00004.safetensors", model, i);
        st_context *st = safetensors_open(path);
        if (!st) goto done;
        shards.st[shards.n++] = st;
    }
    int idx;
    st_context *st = find_tensor(&shards, "model.language_model.embed_tokens.weight", &idx);
    if (!st || strcmp(safetensors_dtype(st,idx),"BF16") ||
        safetensors_nbytes(st,idx) != (size_t)151936 * 4096 * 2) goto done;
    const uint16_t *embedding = safetensors_data(st,idx);
    for (int t = 0; t < n; t++) for (int j = 0; j < 4096; j++) {
        uint32_t bits = (uint32_t)embedding[(size_t)ids[t]*4096+j] << 16;
        memcpy(host+(size_t)t*4096+j, &bits, 4);
    }
    r = cuda_qimg_init(0, 1);
    if (!r) goto done;
    qimg21_kernels base;
    CUfunction rms, rope, add, attention;
    if (cu_compile_kernels(&module,r->device,q21_text_src,"qimg21_text.cu",1,"qimg21_text")<0 ||
        cu_compile_kernels(&base_module,r->device,qimg21_src,"qimg21_native.cu",1,"qimg21_native")<0 ||
        get_kernel(&base,base_module) || cuModuleGetFunction(&rms,module,"text_rms") ||
        cuModuleGetFunction(&rope,module,"text_rope") || cuModuleGetFunction(&add,module,"text_add") ||
        cuModuleGetFunction(&attention,module,"text_attn")) goto done;
    #define ALLOC(p,count,bytes) do { p=checked_cuMemAlloc((size_t)(count)*(bytes)); if(!p)goto done; } while(0)
    ALLOC(x,n*4096,4); ALLOC(norm,n*4096,4); ALLOC(bf,n*12288,2);
    ALLOC(q,n*4096,4); ALLOC(key,n*1024,4); ALLOC(v,n*1024,4);
    ALLOC(att,n*4096,4); ALLOC(tmp,n*4096,4); ALLOC(gate,n*12288,4); ALLOC(up,n*12288,4);
    #undef ALLOC
    if(cuMemcpyHtoD(x,host,(size_t)n*4096*4))goto done;
    if(dump_dir && mkdir(dump_dir,0755) && errno!=EEXIST)goto done;
    #define CHECK(call) do { if((call)!=0)goto done; } while(0)
    #define NAME(suffix) snprintf(name,sizeof(name),"model.language_model.layers.%d.%s",l,suffix)
    #define LINEAR(suffix,dst,no,ni) do { NAME(suffix); CHECK(text_linear(r,&base,&shards,name,dst,bf,n,no,ni)); } while(0)
    #define DUMP(label,ptr,width) do { if(dump_dir && l==0) { CHECK(cuCtxSynchronize()); qimg21_stage_dir=dump_dir; dump_stage("stage_" label,ptr,(size_t)n*(width),n,width); } } while(0)
    for(int l=0;l<layers;l++) {
        fprintf(stderr,"text: layer %d/%d (%d tokens)\n",l+1,layers,n);
        NAME("input_layernorm.weight"); CHECK(text_norm(r,rms,&shards,name,norm,x,n,4096));
        DUMP("input_layernorm",norm,4096);
        CHECK(launch_cast(r,bf,norm,n*4096));
        LINEAR("self_attn.q_proj.weight",q,4096,4096);
        LINEAR("self_attn.k_proj.weight",key,1024,4096);
        LINEAR("self_attn.v_proj.weight",v,1024,4096);
        DUMP("self_attn.q_proj",q,4096); DUMP("self_attn.k_proj",key,1024); DUMP("self_attn.v_proj",v,1024);
        NAME("self_attn.q_norm.weight"); CHECK(text_norm(r,rms,&shards,name,q,q,n*32,128));
        NAME("self_attn.k_norm.weight"); CHECK(text_norm(r,rms,&shards,name,key,key,n*8,128));
        DUMP("self_attn.q_norm",q,4096); DUMP("self_attn.k_norm",key,1024);
        int heads=32; void *qa[]={&q,&heads};
        CHECK(cuLaunchKernel(rope,n,32,1,64,1,1,0,r->stream,qa,NULL));
        heads=8; void *ka[]={&key,&heads};
        CHECK(cuLaunchKernel(rope,n,8,1,64,1,1,0,r->stream,ka,NULL));
        void *aa[]={&att,&q,&key,&v,&n};
        CHECK(cuLaunchKernel(attention,32,n,1,32,1,1,0,r->stream,aa,NULL));
        CHECK(launch_vec(base.round_bf16,r->stream,n*4096,att));
        DUMP("self_attn.o_proj.input",att,4096);
        CHECK(launch_cast(r,bf,att,n*4096));
        LINEAR("self_attn.o_proj.weight",tmp,4096,4096);
        DUMP("self_attn.o_proj",tmp,4096);
        int count=n*4096; void *ra[]={&x,&tmp,&count};
        CHECK(cuLaunchKernel(add,(count+255)/256,1,1,256,1,1,0,r->stream,ra,NULL));
        NAME("post_attention_layernorm.weight"); CHECK(text_norm(r,rms,&shards,name,norm,x,n,4096));
        DUMP("post_attention_layernorm",norm,4096);
        CHECK(launch_cast(r,bf,norm,n*4096));
        LINEAR("mlp.gate_proj.weight",gate,12288,4096);
        LINEAR("mlp.up_proj.weight",up,12288,4096);
        DUMP("mlp.gate_proj",gate,12288); DUMP("mlp.up_proj",up,12288);
        int ffcount=n*12288; void *ma[]={&gate,&gate,&up,&ffcount};
        CHECK(cuLaunchKernel(base.mul_silu,(ffcount+255)/256,1,1,256,1,1,0,r->stream,ma,NULL));
        CHECK(launch_cast(r,bf,gate,ffcount));
        LINEAR("mlp.down_proj.weight",tmp,4096,12288);
        DUMP("mlp.down_proj",tmp,4096);
        CHECK(cuLaunchKernel(add,(count+255)/256,1,1,256,1,1,0,r->stream,ra,NULL));
        if(dump_dir) {
            CHECK(cuStreamSynchronize(r->stream));
            CHECK(cuMemcpyDtoH(host,x,(size_t)n*4096*4));
            snprintf(path,sizeof(path),"%s/layer_%02d.npy",dump_dir,l);
            CHECK(npy_write_f32(path,host,(size_t)n*4096,n,4096));
        }
    }
    CHECK(cuStreamSynchronize(r->stream));
    CHECK(cuMemcpyDtoH(host,x,(size_t)n*4096*4));
    for(size_t i=0;i<(size_t)n*4096;i++)if(!isfinite(host[i]))goto done;
    rc=npy_write_f32(out,host+(size_t)drop*4096,(size_t)(n-drop)*4096,n-drop,4096);
    #undef CHECK
    #undef NAME
    #undef LINEAR
    #undef DUMP
done:
    if(rc)fprintf(stderr,"text: encoder failed\n");
    free_d(&x);free_d(&norm);free_d(&bf);free_d(&q);free_d(&key);free_d(&v);
    free_d(&att);free_d(&tmp);free_d(&gate);free_d(&up);
    if(module)cuModuleUnload(module);
    if(base_module)cuModuleUnload(base_module);
    if(r)cuda_qimg_free(r);
    for(int i=0;i<shards.n;i++)safetensors_close(shards.st[i]);
    free(host);
    return rc;
}
