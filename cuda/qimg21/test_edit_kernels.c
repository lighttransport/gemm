/* Synthetic GPU validation for native editing primitives; no model weights. */
#define main qimg21_denoiser_main
#include "test_cuda_qimg21_native.c"
#undef main
#include "joint_layout.h"
#include "edit_kernels.h"

int main(int argc, char **argv) {
    if(argc!=2)return 2;
    const char *folder=argv[1];
    if(mkdir(folder,0755) && errno!=EEXIST)return 1;
    int mask[]={0,1,1,0,1},height[]={2,2,2},width[]={2,2,2};
    q21_joint_layout layout;
    if(q21_layout_build(&layout,mask,5,4,height,width,3))return 1;
    int n=layout.n,heads=2,dim=256,rc=1;
    float *host=malloc((size_t)n*dim*4);
    if(!host){q21_layout_free(&layout);return 1;}
    cuda_qimg_runner *r=cuda_qimg_init(0,1);
    CUmodule module=NULL;
    CUdeviceptr text=0,img=0,ti=0,ii=0,ids=0,pos=0,q=0,k=0,v=0,y=0,qw=0,kw=0;
    if(!r)goto done;
    CUfunction scatter,rope,attn;
    if(cu_compile_kernels(&module,r->device,q21_edit_src,"qimg21_edit.cu",1,"qimg21_edit")<0 ||
       cuModuleGetFunction(&scatter,module,"edit_scatter") || cuModuleGetFunction(&rope,module,"edit_qk_rope") ||
       cuModuleGetFunction(&attn,module,"edit_attention"))goto done;
    #define CHECK(call) do {if((call)!=0)goto done;}while(0)
    #define ALLOC(ptr,count) do {ptr=checked_cuMemAlloc((size_t)(count)*4);if(!ptr)goto done;}while(0)
    ALLOC(text,4*dim);ALLOC(img,12*dim);ALLOC(ti,n);ALLOC(ii,n);ALLOC(ids,n);ALLOC(pos,n*3);
    ALLOC(q,n*dim);ALLOC(k,n*dim);ALLOC(v,n*dim);ALLOC(y,n*dim);ALLOC(qw,128);ALLOC(kw,128);
    CHECK(cuMemcpyHtoD(ti,layout.text_index,n*4));CHECK(cuMemcpyHtoD(ii,layout.image_index,n*4));
    CHECK(cuMemcpyHtoD(ids,layout.image_id,n*4));CHECK(cuMemcpyHtoD(pos,layout.position,n*3*4));
    char path[2048];
    #define SAVE(label,rows,cols) do {snprintf(path,sizeof(path),"%s/%s.npy",folder,label);CHECK(npy_write_f32(path,host,(size_t)(rows)*(cols),rows,cols));}while(0)
    #define READ_SAVE(label,ptr) do {CHECK(cuCtxSynchronize());CHECK(cuMemcpyDtoH(host,ptr,(size_t)n*dim*4));SAVE(label,n,dim);}while(0)
    for(int i=0;i<4*dim;i++)host[i]=(float)i;
    CHECK(cuMemcpyHtoD(text,host,4*dim*4));SAVE("text",4,dim);
    for(int i=0;i<12*dim;i++)host[i]=10000.f+i;
    CHECK(cuMemcpyHtoD(img,host,12*dim*4));SAVE("image",12,dim);
    CUdeviceptr tensors[]={q,k,v};const char *names[]={"q_input","k_input","v_input"};
    for(int j=0;j<3;j++){
        for(int i=0;i<n*dim;i++)host[i]=qimg21_round_bf16_host(sinf((i+17*j)*.031f));
        CHECK(cuMemcpyHtoD(tensors[j],host,(size_t)n*dim*4));SAVE(names[j],n,dim);
    }
    for(int i=0;i<128;i++)host[i]=qimg21_round_bf16_host(.75f+i*.002f);
    CHECK(cuMemcpyHtoD(qw,host,128*4));SAVE("qw",1,128);
    for(int i=0;i<128;i++)host[i]=qimg21_round_bf16_host(1.25f-i*.002f);
    CHECK(cuMemcpyHtoD(kw,host,128*4));SAVE("kw",1,128);
    CHECK(cuCtxSynchronize());
    void *sa[]={&y,&text,&img,&ti,&ii,&n,&dim};
    CHECK(cuLaunchKernel(scatter,(n*dim+255)/256,1,1,256,1,1,0,r->stream,sa,NULL));READ_SAVE("scatter",y);
    void *ra[]={&q,&k,&qw,&kw,&pos,&n,&heads};
    CHECK(cuLaunchKernel(rope,n,heads,1,128,1,1,0,r->stream,ra,NULL));READ_SAVE("rope_q",q);READ_SAVE("rope_k",k);
    void *aa[]={&y,&q,&k,&v,&ids,&n,&heads};
    CHECK(cuLaunchKernel(attn,heads,n,1,32,1,1,0,r->stream,aa,NULL));READ_SAVE("attention",y);
    rc=0;
    #undef CHECK
    #undef ALLOC
    #undef SAVE
    #undef READ_SAVE
done:
    free_d(&text);free_d(&img);free_d(&ti);free_d(&ii);free_d(&ids);free_d(&pos);
    free_d(&q);free_d(&k);free_d(&v);free_d(&y);free_d(&qw);free_d(&kw);
    if(module)cuModuleUnload(module);
    if(r)cuda_qimg_free(r);
    free(host);q21_layout_free(&layout);
    return rc;
}
