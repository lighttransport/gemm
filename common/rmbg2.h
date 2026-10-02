/* BiRefNet evaluation graph for the pinned RMBG-2.0 model. */
#ifndef RMBG2_H
#define RMBG2_H
#include "vhuman_nn.h"
static vh_tensor rmbg_aspp(st_context *s,const char *prefix,vh_tensor x)
{
    char name[512],branch[512];vh_tensor joined={0};
    for(int i=0;i<4;i++) {
        if(i==0)vh_name(branch,prefix,"aspp1");
        else {char suffix[64];snprintf(suffix,sizeof(suffix),"aspp_deforms.%d",i-1);vh_name(branch,prefix,suffix);}
        vh_name(name,branch,"atrous_conv");vh_tensor part=vh_deform(s,name,x,i<2?1:i==2?3:7);
        vh_name(name,branch,"bn");vh_bn(s,name,part);vh_act(part,1);
        if(i==0)joined=part;
        else {vh_tensor cat=vh_cat(joined,part);vh_drop(joined);vh_drop(part);joined=cat;}
    }
    vh_tensor pool=vh_new(x.c,1,1);
    for(int c=0;c<x.c;c++){double sum=0;for(int i=0;i<x.h*x.w;i++)sum+=x.d[(size_t)c*x.h*x.w+i];pool.d[c]=sum/(x.h*x.w);}
    vh_name(name,prefix,"global_avg_pool.1");vh_tensor tmp=vh_conv(s,name,pool,1,0,0);vh_drop(pool);
    vh_name(name,prefix,"global_avg_pool.2");vh_bn(s,name,tmp);vh_act(tmp,1);
    pool=vh_resize(tmp,x.h,x.w,1);vh_drop(tmp);tmp=vh_cat(joined,pool);vh_drop(joined);vh_drop(pool);
    vh_name(name,prefix,"conv1");joined=vh_conv(s,name,tmp,1,0,0);vh_drop(tmp);
    vh_name(name,prefix,"bn1");vh_bn(s,name,joined);vh_act(joined,1);return joined;
}
static vh_tensor rmbg_block(st_context *s,const char *prefix,vh_tensor x)
{
    char name[512];vh_name(name,prefix,"conv_in");vh_tensor a=vh_conv(s,name,x,1,1,0);
    vh_name(name,prefix,"bn_in");vh_bn(s,name,a);vh_act(a,1);
    vh_name(name,prefix,"dec_att");vh_tensor b=rmbg_aspp(s,name,a);vh_drop(a);
    vh_name(name,prefix,"conv_out");a=vh_conv(s,name,b,1,1,0);vh_drop(b);
    vh_name(name,prefix,"bn_out");vh_bn(s,name,a);return a;
}
static vh_tensor rmbg_input(st_context *s,int block,vh_tensor image,int h,int w)
{
    if(image.h%h || image.w%w)vh_fail("RMBG input patches must divide image");
    vh_tensor patches=vh_new(3*(image.h/h)*(image.w/w),h,w);
    for(int px=0;px<image.w/w;px++)for(int py=0;py<image.h/h;py++)for(int c=0;c<3;c++)
        for(int y=0;y<h;y++)for(int col=0;col<w;col++) {
            int dstc=(px*(image.h/h)+py)*3+c;
            patches.d[((size_t)dstc*h+y)*w+col]=image.d[((size_t)c*image.h+py*h+y)*image.w+px*w+col];
        }
    char name[512];snprintf(name,sizeof(name),"decoder.ipt_blk%d.conv1",block);
    vh_tensor a=vh_conv(s,name,patches,1,1,0);vh_drop(patches);
    snprintf(name,sizeof(name),"decoder.ipt_blk%d.conv_out",block);
    vh_tensor b=vh_conv(s,name,a,1,1,0);vh_drop(a);return b;
}
static vh_tensor rmbg_predict(swin_model *m,vh_tensor image)
{
    if(image.c!=3 || image.h!=image.w || image.h%32)vh_fail("RMBG expects square RGB multiples of 32");
    swin_feature full[4]={{0}},half[4]={{0}};
    swin_predict(m,image.d,image.h,image.w,full);
    vh_tensor small=vh_resize(image,image.h/2,image.w/2,1);
    swin_predict(m,small.d,small.h,small.w,half);vh_drop(small);
    vh_tensor feat[4];
    for(int i=0;i<4;i++) {
        vh_tensor a={full[i].data,full[i].c,full[i].h,full[i].w};
        vh_tensor b={half[i].data,half[i].c,half[i].h,half[i].w};
        vh_tensor up=vh_resize(b,a.h,a.w,1);feat[i]=vh_cat(a,up);vh_drop(a);vh_drop(b);vh_drop(up);
    }
    vh_tensor context={0};
    for(int i=0;i<4;i++) {
        vh_tensor t=vh_resize(feat[i],feat[3].h,feat[3].w,1);
        if(i==0)context=t;else {vh_tensor cat=vh_cat(context,t);vh_drop(context);vh_drop(t);context=cat;}
    }
    st_context *s=m->weights;vh_tensor p=rmbg_block(s,"squeeze_module.0",context);vh_drop(context);
    char name[512];
    for(int block=4;block>=1;block--) {
        vh_tensor ipt=rmbg_input(s,block+1,image,p.h,p.w),cat=vh_cat(p,ipt);vh_drop(p);vh_drop(ipt);
        snprintf(name,sizeof(name),"decoder.decoder_block%d",block);p=rmbg_block(s,name,cat);vh_drop(cat);
        if(block>1) {
            snprintf(name,sizeof(name),"decoder.gdt_convs_%d.0",block);vh_tensor a=vh_conv(s,name,p,1,1,0);
            snprintf(name,sizeof(name),"decoder.gdt_convs_%d.1",block);vh_bn(s,name,a);vh_act(a,1);
            snprintf(name,sizeof(name),"decoder.gdt_convs_attn_%d.0",block);vh_tensor gate=vh_conv(s,name,a,1,0,0);vh_drop(a);vh_act(gate,3);
            for(int c=0;c<p.c;c++)for(int i=0;i<p.h*p.w;i++)p.d[(size_t)c*p.h*p.w+i]*=gate.d[i];
            vh_drop(gate);
            a=vh_resize(p,feat[block-2].h,feat[block-2].w,1);vh_drop(p);
            snprintf(name,sizeof(name),"decoder.lateral_block%d.conv",block);
            vh_tensor lateral=vh_conv(s,name,feat[block-2],1,0,0);vh_add(a,lateral);vh_drop(lateral);p=a;
        }
    }
    for(int i=0;i<4;i++)vh_drop(feat[i]);
    vh_tensor up=vh_resize(p,image.h,image.w,1);vh_drop(p);
    vh_tensor ipt=rmbg_input(s,1,image,image.h,image.w),cat=vh_cat(up,ipt);vh_drop(up);vh_drop(ipt);
    p=vh_conv(s,"decoder.conv_out1.0",cat,1,0,0);vh_drop(cat);return p;
}
#endif
