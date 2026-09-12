/* Included only by ds41f_ops.c. SDOT QK uses group-16 query/raw-key
 * quantization; compressed keys retain exact E2M1*2 codes and E4M3/2 scales.
 * PV remains FP32 and the original sink-aware softmax is independently gated. */
static float sparse_e4m3(uint8_t code)
{
    unsigned m=code&127;if(m==127)return NAN;
    float x=m<8?(float)m*(1.f/512):ldexpf((float)(8+(m&7)),(int)(m>>3)-10);
    return code&128?-x:x;
}
static int sparse_quant16(int8_t *out,float *scale,const float *x)
{
#if defined(__ARM_FEATURE_SVE)
    if(svcntw()==16){svbool_t pg=svptrue_b32();svfloat32_t v=svld1_f32(pg,x),a=svabs_f32_x(pg,v);
        if(svptest_any(pg,svcmpge_n_f32(pg,a,INFINITY))||svptest_any(pg,svcmpuo_f32(pg,v,v)))return EDOM;
        float maximum=svmaxv_f32(pg,a),step=maximum?fmaxf(maximum/127.f,0x1p-149f):1.f;*scale=step;
        v=step<0x1p-126f?svdiv_n_f32_x(pg,v,step):svmul_n_f32_x(pg,v,1.f/step);
        v=svmax_n_f32_x(pg,svmin_n_f32_x(pg,svrintn_f32_x(pg,v),127.f),-127.f);
        svst1b_s32(pg,out,svcvt_s32_f32_x(pg,v));return 0;
    }
#endif
    float maximum=0;for(size_t i=0;i<16;++i){if(!isfinite(x[i]))return EDOM;maximum=fmaxf(maximum,fabsf(x[i]));}
    float step=maximum?fmaxf(maximum/127.f,0x1p-149f):1.f;*scale=step;
    for(size_t i=0;i<16;++i)out[i]=(int8_t)fmaxf(-127.f,fminf(127.f,nearbyintf(x[i]/step)));
    return 0;
}
static void sparse_sdot4x4(float *scores,size_t stride,size_t heads,size_t count,
                            size_t h,size_t r,const int8_t *query,const float *qs,
                            const int8_t *key,const float *ks,const int *ids,int reference)
{
#if defined(__ARM_FEATURE_SVE)
    if(!reference&&svcntw()==16&&h+4<=heads){
        svbool_t pg=svptrue_b32(),p4=svwhilelt_b32((uint64_t)0,(uint64_t)4);
        svuint32_t expand=svlsr_n_u32_x(pg,svindex_u32(0,1),2);
        svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0;
        for(size_t b=0;b<32;++b){svint8_t k=svld1_s8(svptrue_b8(),key+r*512+b*64);
            svfloat32_t scales=svtbl_f32(svld1_f32(p4,ks+r*32+b*4),expand);
            #define QK_HEAD(acc,j) {svint8_t q=svld1rq_s8(svptrue_b8(),query+(h+(j))*512+b*16); \
                svint32_t d=svdot_s32(svdup_s32(0),k,q); \
                acc=svmla_f32_x(pg,acc,svcvt_f32_s32_x(pg,d),svmul_n_f32_x(pg,scales,qs[(h+(j))*32+b]));}
            QK_HEAD(a0,0);QK_HEAD(a1,1);QK_HEAD(a2,2);QK_HEAD(a3,3);
            #undef QK_HEAD
        }
        for(size_t j=0;j<4&&r+j<count;++j){svuint32_t ix=svindex_u32(0,1);
            svbool_t p=svand_b_z(pg,svcmpge_n_u32(pg,ix,j*4),svcmplt_n_u32(pg,ix,j*4+4));
            float factor=1/sqrtf(512.f);
            scores[(h+0)*stride+r+j]=ids[r+j]<0?-INFINITY:svaddv_f32(p,a0)*factor;
            scores[(h+1)*stride+r+j]=ids[r+j]<0?-INFINITY:svaddv_f32(p,a1)*factor;
            scores[(h+2)*stride+r+j]=ids[r+j]<0?-INFINITY:svaddv_f32(p,a2)*factor;
            scores[(h+3)*stride+r+j]=ids[r+j]<0?-INFINITY:svaddv_f32(p,a3)*factor;
        }
        return;
    }
#endif
    (void)reference;
    for(size_t t=h;t<h+4&&t<heads;++t)for(size_t j=0;j<4&&r+j<count;++j){double sum=0;
        for(size_t b=0;b<32;++b){int dot=0;
            for(size_t c=0;c<16;++c)dot+=query[t*512+b*16+c]*key[r*512+b*64+j*16+c];
            sum+=(double)dot*qs[t*32+b]*ks[r*32+b*4+j];}
        scores[t*stride+r+j]=ids[r+j]<0?-INFINITY:(float)sum*(1/sqrtf(512.f));
    }
}
int ds41f_sparse_attention_sdot(float *out,const float *q,const float *kv,
                                const float *sink,const uint8_t *const *compressed,
                                size_t raw,size_t extra,size_t heads,int math,int reference)
{
    if(!out||!q||!kv||!sink||raw>128||extra>512||(!compressed&&extra)||!heads||heads>64||math<0||math>3)return EINVAL;
    size_t count=raw+extra,padded=(count+3)/4*4;
    if(!count){for(size_t i=0;i<heads*512;++i)out[i]=0;return 0;}
    int8_t *query=malloc(heads*512),*key=malloc(padded*512);
    float *qs=malloc(heads*32*sizeof(float)),*ks=malloc(padded*32*sizeof(float));
    float *scores=malloc(heads*count*sizeof(float));int ids[640];
    if(!query||!key||!qs||!ks||!scores){free(query);free(key);free(qs);free(ks);free(scores);return ENOMEM;}
    for(size_t i=0;i<count;++i)ids[i]=i<raw||compressed[i-raw]?(int)i:-1;
    static const int8_t grid[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    int invalid=0,packed_ok=0;double pt=P_BEGIN();
    #pragma omp parallel shared(pt,invalid,packed_ok)
    {
    #pragma omp for schedule(static) reduction(|:invalid)
    for(size_t t=0;t<heads+padded;++t){
        if(t<heads){for(size_t b=0;b<32;++b)invalid|=sparse_quant16(query+t*512+b*16,qs+t*32+b,q+t*512+b*16);}
        else{size_t i=t-heads,r=i/4*4,j=i%4;
            for(size_t b=0;b<32;++b){int8_t *dst=key+r*512+b*64+j*16;float *scale=ks+r*32+b*4+j;
                if(i>=count||ids[i]<0){memset(dst,0,16);*scale=0;}
                else if(i<raw)invalid|=sparse_quant16(dst,scale,kv+i*512+b*16);
                else{const uint8_t *src=compressed[i-raw];*scale=sparse_e4m3(src[256+b])*.5f;
                    if(!isfinite(*scale)){invalid|=EDOM;*scale=0;}
                    for(size_t c=0;c<16;++c)dst[c]=grid[(src[b*8+c/2]>>((c%2)*4))&15];}
            }
        }
    }
    #pragma omp master
    {packed_ok=!invalid;P_END(SPARSE_PACK,pt);pt=P_BEGIN();}
    #pragma omp barrier
    #pragma omp for schedule(static)
    for(size_t task=0;task<((heads+3)/4)*(padded/4);++task){size_t h=task/(padded/4)*4,r=task%(padded/4)*4;
        if(!invalid)sparse_sdot4x4(scores,count,heads,count,h,r,query,qs,key,ks,ids,reference);}
    #pragma omp master
    {P_END(SPARSE_QK,pt);pt=P_BEGIN();}
    #pragma omp barrier
    #pragma omp for schedule(static) reduction(|:invalid)
    for(size_t h=0;h<heads;++h)if(packed_ok)invalid|=ds41f_attention_softmax(scores+h*count,count,sink[h],math);
    #pragma omp master
    {P_END(SPARSE_SOFTMAX,pt);pt=P_BEGIN();}
    #pragma omp barrier
    #pragma omp for schedule(static)
    for(size_t task=0;task<((heads+3)/4)*8;++task){size_t h=task/8*4,c=task%8*64;
        if(!invalid){
#if defined(__ARM_FEATURE_SVE)
            if(svcntw()==16&&h+4<=heads)sparse_pv4(out+h*512,scores+h*count,kv,ids,count,c);
            else
#endif
            for(size_t t=h;t<h+4&&t<heads;++t)for(size_t j=c;j<c+64;++j){float value=0;
                for(size_t i=0;i<count;++i)if(ids[i]>=0)value+=scores[t*count+i]*kv[i*512+j];
                out[t*512+j]=value;}
        }
    }
    }
    P_END(SPARSE_PV,pt);free(query);free(key);free(qs);free(ks);free(scores);return invalid;
}
