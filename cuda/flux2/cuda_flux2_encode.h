/* FLUX.2 VAE image encoder. Reuses repository CUDA VAE/GEMM primitives.
 * Include after cuda_flux2_runner.h's implementation. Weights are streamed
 * one residual block at a time; no inference framework or vendor GEMM needed.
 */
#ifndef CUDA_FLUX2_ENCODE_H
#define CUDA_FLUX2_ENCODE_H

static void flux2_encode_drop_res(flux2_vae_resblock *b) {
    free(b->norm1_w);free(b->norm1_b);free(b->conv1_w);free(b->conv1_b);
    free(b->norm2_w);free(b->norm2_b);free(b->conv2_w);free(b->conv2_b);
    free(b->skip_w);free(b->skip_b);
}
static void flux2_encode_drop_attn(flux2_vae_attn *b) {
    free(b->norm_w);free(b->norm_b);free(b->q_w);free(b->q_b);
    free(b->k_w);free(b->k_b);free(b->v_w);free(b->v_b);free(b->out_w);free(b->out_b);
}
static CUdeviceptr flux2_encode_conv(cuda_flux2_runner *r,CUdeviceptr x,const char *prefix,
                                    int ci,int co,int h,int w,int kernel) {
    st_context *st=(st_context *)r->vae->st_ctx;
    char name[512];snprintf(name,sizeof(name),"%s.weight",prefix);
    int wi=safetensors_find(st,name);
    if(wi<0 || safetensors_ndims(st,wi)!=4) return 0;
    const uint64_t *shape=safetensors_shape(st,wi);
    if(shape[0]!=(uint64_t)co || shape[1]!=(uint64_t)ci || shape[2]!=(uint64_t)kernel || shape[3]!=(uint64_t)kernel) return 0;
    float *weight=flux2_vae_load_tensor(st,name,NULL);
    snprintf(name,sizeof(name),"%s.bias",prefix);
    float *bias=flux2_vae_load_tensor(st,name,NULL);
    CUdeviceptr dw=kernel==3?vae_upload_w_3x3(r,weight,co*ci*9):gpu_upload_f32(weight,co*ci);
    CUdeviceptr db=gpu_upload_f32_or0(bias,co),y=0;
    free(weight);free(bias);
    if(!dw || !db || cuMemAlloc(&y,(size_t)co*h*w*4)!=CUDA_SUCCESS) goto done;
    if(kernel==3) vae_conv_3x3(r,y,x,dw,db,ci,h,w,co);
    else op_vae_conv2d(r,y,x,dw,db,ci,h,w,co,1,1,0);
    if(cuStreamSynchronize(r->stream)!=CUDA_SUCCESS) CU_FREE(y);
done:
    CU_FREE(dw);CU_FREE(db);return y;
}
static CUdeviceptr flux2_encode_res(cuda_flux2_runner *r,CUdeviceptr x,const char *prefix,int ci,int co,int h,int w) {
    flux2_vae_resblock rb={0};flux2_gpu_vae_resblock_t gpu={0};
    flux2_vae_load_resblock(&rb,(st_context *)r->vae->st_ctx,prefix,ci,co);
    CUdeviceptr y=0;
    if(flux2_vae_upload_resblock(r,&gpu,&rb)==0)
        y=flux2_vae_resblock_gpu(r,x,&rb,&gpu,h,w,32);
    if(cuStreamSynchronize(r->stream)!=CUDA_SUCCESS) CU_FREE(y);
    flux2_vae_free_resblock(&gpu);flux2_encode_drop_res(&rb);return y;
}
static int cuda_flux2_vae_encode(cuda_flux2_runner *r,const float *image,int height,int width,float *latent) {
    if(!r || !r->vae || !image || !latent || height<16 || width<16 || height>1024 || width>1024 || height%16 || width%16) return -1;
    CUdeviceptr x=gpu_upload_f32(image,3*height*width),y=0,norm_w=0,norm_b=0;
    float *host_w=NULL,*host_b=NULL;
    int h=height,w=width,c=128,rc=-1;
    st_context *st=(st_context *)r->vae->st_ctx;
    CUfunction down;
    if(cuModuleGetFunction(&down,r->mod,"flux2_vae_downsample2x_f32")!=CUDA_SUCCESS || !x) goto done;
    y=flux2_encode_conv(r,x,"encoder.conv_in",3,128,h,w,3);if(!y)goto done;CU_FREE(x);x=y;y=0;
    for(int block=0;block<4;block++) {
        int co=block==0?128:block==1?256:512;
        for(int layer=0;layer<2;layer++) {
            char name[128];snprintf(name,sizeof(name),"encoder.down_blocks.%d.resnets.%d",block,layer);
            y=flux2_encode_res(r,x,name,c,co,h,w);if(!y)goto done;CU_FREE(x);x=y;y=0;c=co;
        }
        if(block<3) {
            char name[128];snprintf(name,sizeof(name),"encoder.down_blocks.%d.downsamplers.0.conv",block);
            y=flux2_encode_conv(r,x,name,c,c,h,w,3);if(!y)goto done;CU_FREE(x);x=y;y=0;
            int total=c*(h/2)*(w/2);
            if(cuMemAlloc(&y,(size_t)total*4)!=CUDA_SUCCESS)goto done;
            void *args[]={&y,&x,&c,&h,&w};
            if(cuLaunchKernel(down,(total+255)/256,1,1,256,1,1,0,r->stream,args,NULL)!=CUDA_SUCCESS || cuStreamSynchronize(r->stream)!=CUDA_SUCCESS)goto done;
            CU_FREE(x);x=y;y=0;h/=2;w/=2;
        }
    }
    y=flux2_encode_res(r,x,"encoder.mid_block.resnets.0",c,c,h,w);if(!y)goto done;CU_FREE(x);x=y;y=0;
    {
        flux2_vae_attn attn={0};flux2_gpu_vae_attn_t gpu={0};
        flux2_vae_load_attn(&attn,st,"encoder.mid_block.attentions.0",c);
        if(flux2_vae_upload_attn(r,&gpu,&attn,flux2_vae_attn_use_bf16(r))==0)
            y=flux2_vae_mid_attn_gpu(r,x,&attn,&gpu,h,w,32);
        if(cuStreamSynchronize(r->stream)!=CUDA_SUCCESS)CU_FREE(y);
        flux2_vae_free_attn(&gpu);flux2_encode_drop_attn(&attn);
        if(!y)goto done;
        CU_FREE(x);x=y;y=0;
    }
    y=flux2_encode_res(r,x,"encoder.mid_block.resnets.1",c,c,h,w);if(!y)goto done;CU_FREE(x);x=y;y=0;
    host_w=flux2_vae_load_tensor(st,"encoder.conv_norm_out.weight",NULL);
    host_b=flux2_vae_load_tensor(st,"encoder.conv_norm_out.bias",NULL);
    norm_w=gpu_upload_f32(host_w,c);norm_b=gpu_upload_f32(host_b,c);
    if(!norm_w || !norm_b || cuMemAlloc(&y,(size_t)c*h*w*4)!=CUDA_SUCCESS)goto done;
    op_vae_groupnorm(r,y,x,norm_w,norm_b,c,h*w,32);
    op_silu(r,y,c*h*w);
    if(cuStreamSynchronize(r->stream)!=CUDA_SUCCESS)goto done;
    CU_FREE(x);x=y;y=0;
    y=flux2_encode_conv(r,x,"encoder.conv_out",c,64,h,w,3);if(!y)goto done;CU_FREE(x);x=y;y=0;
    y=flux2_encode_conv(r,x,"quant_conv",64,64,h,w,1);if(!y)goto done;CU_FREE(x);x=y;y=0;
    // Deterministic posterior mode: the first 32 channels are the mean.
    if(cuMemcpyDtoH(latent,x,(size_t)32*h*w*4)!=CUDA_SUCCESS)goto done;
    rc=0;
done:
    CU_FREE(x);CU_FREE(y);CU_FREE(norm_w);CU_FREE(norm_b);free(host_w);free(host_b);
    if(rc)fprintf(stderr,"cuda_flux2: VAE encoder failed\n");
    return rc;
}
#endif
