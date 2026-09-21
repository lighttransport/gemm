#ifndef QIMG21_EDIT_RUNTIME_H
#define QIMG21_EDIT_RUNTIME_H
#include "joint_layout.h"
#include "edit_kernels.h"

typedef struct {
    q21_joint_layout layout;
    CUmodule module;
    CUfunction scatter, rope, attention;
    CUdeviceptr text_index, image_index, image_id, position;
} q21_edit_context;

static void q21_edit_free(q21_edit_context *edit) {
    if(edit->text_index)cuMemFree(edit->text_index);
    if(edit->image_index)cuMemFree(edit->image_index);
    if(edit->image_id)cuMemFree(edit->image_id);
    if(edit->position)cuMemFree(edit->position);
    if(edit->module)cuModuleUnload(edit->module);
    q21_layout_free(&edit->layout);
    memset(edit,0,sizeof(*edit));
}

/* The layout is validated before allocating CUDA memory. */
static int q21_edit_init(q21_edit_context *edit, cuda_qimg_runner *r,
                         const char *path, int nt, int image_tokens, int ih, int iw, int reverse64) {
    memset(edit,0,sizeof(*edit));
    int slots,h,w;
    if(q21_layout_read(path,&edit->layout,&slots,&h,&w) || slots!=nt || h!=ih || w!=iw ||
       edit->layout.image_tokens!=image_tokens)goto fail;
    if(cu_compile_kernels(&edit->module,r->device,q21_edit_src,"qimg21_edit.cu",1,"qimg21_edit")<0 ||
       cuModuleGetFunction(&edit->scatter,edit->module,"edit_scatter") ||
       cuModuleGetFunction(&edit->rope,edit->module,"edit_qk_rope") ||
       cuModuleGetFunction(&edit->attention,edit->module,reverse64?"edit_attention_reverse64":"edit_attention"))goto fail;
    int n=edit->layout.n;
    #define UPLOAD(field,source,count) do { \
        if(cuMemAlloc(&edit->field,(size_t)(count)*sizeof(int)) || \
           cuMemcpyHtoD(edit->field,edit->layout.source,(size_t)(count)*sizeof(int)))goto fail; \
    }while(0)
    UPLOAD(text_index,text_index,n);UPLOAD(image_index,image_index,n);
    UPLOAD(image_id,image_id,n);UPLOAD(position,position,3*n);
    #undef UPLOAD
    if(cuCtxSynchronize())goto fail;
    return 0;
fail:
    q21_edit_free(edit);
    fprintf(stderr,"native: invalid editing layout or CUDA initialization failure\n");
    return -1;
}
#endif
