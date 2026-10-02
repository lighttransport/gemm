/* Depth Anything V2 Small: existing DINOv2 encoder + DA3 DPT primitives.
 * Define DEPTH_ANYTHING2_IMPLEMENTATION after safetensors/dequant implementations.
 * Inputs are normalized RGB CHW; outputs are relative inverse depth. */
#ifndef DEPTH_ANYTHING2_H
#define DEPTH_ANYTHING2_H
#include "safetensors.h"
#include "dinov2.h"
#include "depth_anything3.h"

typedef struct {
    dinov2_model *encoder;
    da3_dpt_head head;
    st_context *weights;
} da2_model;

da2_model *da2_load(const char *backbone, const char *head);
void da2_free(da2_model *m);
float *da2_predict(da2_model *m, const float *chw, int w, int h,
                   int out_w, int out_h, int threads, const char *dump_dir);

#ifdef DEPTH_ANYTHING2_IMPLEMENTATION

static int da2_tensor(st_context *s, const char *name, qtensor *out,
                      int nd, int d0, int d1, int d2, int d3)
{
    int i=safetensors_find(s,name), dims[]={d0,d1,d2,d3};
    if (i<0 || strcmp(safetensors_dtype(s,i),"F32") || safetensors_ndims(s,i)!=nd) goto bad;
    const uint64_t *shape=safetensors_shape(s,i);
    size_t bytes=sizeof(float);
    for (int d=0; d<nd; d++) {
        if (shape[d]!=(uint64_t)dims[d]) goto bad;
        bytes*=shape[d];
    }
    if (safetensors_nbytes(s,i)!=bytes) goto bad;
    *out=qt_make_tensor(s,i);
    return 0;
bad:
    fprintf(stderr,"da2: missing/invalid F32 tensor %s\n",name);
    return -1;
}

void da2_free(da2_model *m)
{
    if (!m) return;
    if (m->encoder) dinov2_free(m->encoder);
    if (m->weights) safetensors_close(m->weights);
    free(m);
}

da2_model *da2_load(const char *backbone, const char *head)
{
    da2_model *m=calloc(1,sizeof(*m));
    if (!m) return NULL;
    m->encoder=dinov2_load_safetensors(backbone);
    m->weights=safetensors_open(head);
    if (!m->encoder || !m->weights) goto fail;
    dinov2_model *e=m->encoder;
    if (e->dim!=384 || e->n_blocks!=12 || e->n_heads!=6 || e->ffn_hidden!=1536 ||
        e->patch_size!=14 || e->n_register!=0 || e->orig_grid!=37) goto fail;
    /* Verify every required backbone tensor, not merely inferred dimensions. */
    st_context *s=e->st_ctx;
    qtensor t;
    if (da2_tensor(s,"cls_token",&t,3,1,1,384,0) ||
        da2_tensor(s,"pos_embed",&t,3,1,1370,384,0) ||
        da2_tensor(s,"patch_embed.proj.weight",&t,4,384,3,14,14) ||
        da2_tensor(s,"patch_embed.proj.bias",&t,1,384,0,0,0) ||
        da2_tensor(s,"norm.weight",&t,1,384,0,0,0) ||
        da2_tensor(s,"norm.bias",&t,1,384,0,0,0)) goto fail;
    char name[160];
    const char *vectors[]={"norm1.weight","norm1.bias","norm2.weight","norm2.bias","ls1.gamma","ls2.gamma"};
    const char *linears[]={"attn.qkv","attn.proj","mlp.fc1","mlp.fc2"};
    const int inputs[]={384,384,384,1536}, outputs[]={1152,384,1536,384};
    for (int l=0; l<12; l++) {
        for (int j=0; j<6; j++) {
            snprintf(name,sizeof(name),"blocks.%d.%s",l,vectors[j]);
            if (da2_tensor(s,name,&t,1,384,0,0,0)) goto fail;
        }
        for (int j=0; j<4; j++) {
            snprintf(name,sizeof(name),"blocks.%d.%s.weight",l,linears[j]);
            if (da2_tensor(s,name,&t,2,outputs[j],inputs[j],0,0)) goto fail;
            snprintf(name,sizeof(name),"blocks.%d.%s.bias",l,linears[j]);
            if (da2_tensor(s,name,&t,1,outputs[j],0,0,0)) goto fail;
        }
    }
    s=m->weights;
    da3_dpt_head *d=&m->head;
    const int channels[]={48,96,192,384};
    for (int l=0; l<4; l++) {
        snprintf(name,sizeof(name),"projects.%d.weight",l);
        if (da2_tensor(s,name,&d->proj_w[l],4,channels[l],384,1,1)) goto fail;
        snprintf(name,sizeof(name),"projects.%d.bias",l);
        if (da2_tensor(s,name,&d->proj_b[l],1,channels[l],0,0,0)) goto fail;
        snprintf(name,sizeof(name),"scratch.layer%d_rn.weight",l+1);
        if (da2_tensor(s,name,&d->adapter_w[l],4,64,channels[l],3,3)) goto fail;
        for (int r=1; r<=2; r++) for (int c=1; c<=2; c++) {
            qtensor *w=r==1 ? (c==1 ? d->fuse_rcu1_c1_w : d->fuse_rcu1_c2_w)
                              : (c==1 ? d->fuse_rcu2_c1_w : d->fuse_rcu2_c2_w);
            qtensor *b=r==1 ? (c==1 ? d->fuse_rcu1_c1_b : d->fuse_rcu1_c2_b)
                              : (c==1 ? d->fuse_rcu2_c1_b : d->fuse_rcu2_c2_b);
            snprintf(name,sizeof(name),"scratch.refinenet%d.resConfUnit%d.conv%d.weight",l+1,r,c);
            if (da2_tensor(s,name,&w[l],4,64,64,3,3)) goto fail;
            snprintf(name,sizeof(name),"scratch.refinenet%d.resConfUnit%d.conv%d.bias",l+1,r,c);
            if (da2_tensor(s,name,&b[l],1,64,0,0,0)) goto fail;
        }
        snprintf(name,sizeof(name),"scratch.refinenet%d.out_conv.weight",l+1);
        if (da2_tensor(s,name,&d->fuse_out_w[l],4,64,64,1,1)) goto fail;
        snprintf(name,sizeof(name),"scratch.refinenet%d.out_conv.bias",l+1);
        if (da2_tensor(s,name,&d->fuse_out_b[l],1,64,0,0,0)) goto fail;
    }
#define DA2_PAIR(base,field,ci,co,k) \
    if (da2_tensor(s,base ".weight",&d->field##_w,4,co,ci,k,k) || \
        da2_tensor(s,base ".bias",&d->field##_b,1,co,0,0,0)) goto fail
    DA2_PAIR("resize_layers.0",upsample_0,48,48,4);
    DA2_PAIR("resize_layers.1",upsample_1,96,96,2);
    DA2_PAIR("resize_layers.3",downsample,384,384,3);
    DA2_PAIR("scratch.output_conv1",neck,64,32,3);
    DA2_PAIR("scratch.output_conv2.0",out_0,32,32,3);
    DA2_PAIR("scratch.output_conv2.2",out_2,32,1,1);
#undef DA2_PAIR
    return m;
fail:
    fprintf(stderr,"da2: only the verified Depth Anything V2 Small F32 architecture is supported\n");
    da2_free(m); return NULL;
}

static int da2_dump(const char *dir, const char *name, const float *p, size_t count)
{
    if (!dir) return 0;
    char path[4096];
    if (snprintf(path,sizeof(path),"%s/%s.f32",dir,name)>=(int)sizeof(path)) return -1;
    FILE *f=fopen(path,"wb");
    if (!f) return -1;
    int rc=fwrite(p,sizeof(float),count,f)==count ? 0 : -1;
    return fclose(f) ? -1 : rc;
}

float *da2_predict(da2_model *m, const float *chw, int w, int h,
                   int out_w, int out_h, int threads, const char *dump_dir)
{
    if (!m || !chw || w<14 || h<14 || w%14 || h%14 ||
        (int64_t)(w/14)*(h/14)>4096 || out_w<1 || out_h<1 ||
        (int64_t)out_w*out_h>4194304) return NULL;
    dinov2_model *e=m->encoder;
    e->grid_h=h/14; e->grid_w=w/14; e->n_patches=e->grid_h*e->grid_w;
    e->n_tokens=e->n_patches+1;
    int layers[]={2,5,8,11}, channels[]={48,96,192,384}, ah[4],aw[4];
    float *features[4]={0}, *adapted[4]={0}, *fused=NULL, *result=NULL;
    if (dinov2_intermediates_f32(e,chw,w,h,layers,4,features,threads)) return NULL;
    for (int l=0; l<4; l++) {
        char name[64]; snprintf(name,sizeof(name),"feature_%d",l);
        if (da2_dump(dump_dir,name,features[l],(size_t)e->n_patches*384)) goto done;
        float *channel=malloc((size_t)e->n_patches*384*sizeof(float));
        if (!channel) goto done;
        for (int p=0; p<e->n_patches; p++) for (int c=0; c<384; c++)
            channel[(size_t)c*e->n_patches+p]=features[l][(size_t)p*384+c];
        free(features[l]); features[l]=NULL;
        int ph,pw,sh,sw;
        float *project=da3_conv2d_qt(channel,&m->head.proj_w[l],&m->head.proj_b[l],
                                    h/14,w/14,384,channels[l],1,1,1,0,&ph,&pw);
        free(channel);
        float *spatial;
        if (l<2) {
            int k=l==0 ? 4 : 2;
            spatial=da3_conv_transpose2d_qt(project,l==0 ? &m->head.upsample_0_w : &m->head.upsample_1_w,
                    l==0 ? &m->head.upsample_0_b : &m->head.upsample_1_b,
                    ph,pw,channels[l],channels[l],k,k,k,&sh,&sw);
            free(project);
        } else if (l==3) {
            spatial=da3_conv2d_qt(project,&m->head.downsample_w,&m->head.downsample_b,
                                  ph,pw,384,384,3,3,2,1,&sh,&sw);
            free(project);
        } else { spatial=project; sh=ph; sw=pw; }
        adapted[l]=da3_conv2d_qt(spatial,&m->head.adapter_w[l],NULL,
                                 sh,sw,channels[l],64,3,3,1,1,&ah[l],&aw[l]);
        free(spatial);
    }
    int fh=ah[3],fw=aw[3];
    for (int l=3; l>=0; l--) {
        int oh=l ? ah[l-1] : ah[0]*2, ow=l ? aw[l-1] : aw[0]*2;
        float *next=da3_refinenet(&m->head,l,adapted[l],ah[l],aw[l],fused,fh,fw,64,oh,ow);
        free(fused); fused=next; fh=oh; fw=ow;
    }
    if (da2_dump(dump_dir,"fused",fused,(size_t)64*fh*fw)) goto done;
    int nh,nw;
    float *neck=da3_conv2d_qt(fused,&m->head.neck_w,&m->head.neck_b,fh,fw,64,32,3,3,1,1,&nh,&nw);
    free(fused); fused=NULL;
    float *up=malloc((size_t)32*h*w*sizeof(float));
    if (!up) { free(neck); goto done; }
    da3_bilinear(up,neck,32,nh,nw,h,w); free(neck);
    float *out0=da3_conv2d_qt(up,&m->head.out_0_w,&m->head.out_0_b,h,w,32,32,3,3,1,1,&nh,&nw);
    free(up);
    for (size_t i=0; i<(size_t)32*h*w; i++) if (out0[i]<0) out0[i]=0;
    float *depth=da3_conv2d_qt(out0,&m->head.out_2_w,&m->head.out_2_b,h,w,32,1,1,1,1,0,&nh,&nw);
    free(out0);
    for (int i=0; i<h*w; i++) if (depth[i]<0) depth[i]=0;
    if (da2_dump(dump_dir,"depth_model",depth,(size_t)h*w)) { free(depth); goto done; }
    result=malloc((size_t)out_h*out_w*sizeof(float));
    if (result) da3_bilinear(result,depth,1,h,w,out_h,out_w);
    free(depth);
done:
    for (int l=0; l<4; l++) { free(features[l]); free(adapted[l]); }
    free(fused);
    return result;
}
#endif
#endif
