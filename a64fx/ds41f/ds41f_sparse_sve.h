/* A64FX sparse attention tiles inspired by CLAIR attention_decode_a64fx.c.
 * DS41F geometry, two-chain QK reduction and sink normalization are retained.
 * Included only by ds41f_ops.c; the scalar/general control remains available. */
#if defined(__ARM_FEATURE_SVE)
static float sparse_dot(const float *q,const float *v,size_t dim)
{
    svbool_t pg=svptrue_b32();svfloat32_t a=svdup_f32(0),b=a;
    for(size_t j=0;j<dim;j+=32){
        a=svmla_f32_x(pg,a,svld1_f32(pg,q+j),svld1_f32(pg,v+j));
        b=svmla_f32_x(pg,b,svld1_f32(pg,q+j+16),svld1_f32(pg,v+j+16));}
    return svaddv_f32(pg,svadd_f32_x(pg,a,b));
}
static void sparse_qk4x2(float *scores,const float *q,const float *k0,const float *k1,
                         size_t stride,float scale)
{
    svbool_t pg=svptrue_b32();
    svfloat32_t a00=svdup_f32(0),b00=a00;
    svfloat32_t a01=svdup_f32(0),b01=a01;
    svfloat32_t a10=svdup_f32(0),b10=a10;
    svfloat32_t a11=svdup_f32(0),b11=a11;
    svfloat32_t a20=svdup_f32(0),b20=a20;
    svfloat32_t a21=svdup_f32(0),b21=a21;
    svfloat32_t a30=svdup_f32(0),b30=a30;
    svfloat32_t a31=svdup_f32(0),b31=a31;
    for(size_t j=0;j<512;j+=32){
        {svfloat32_t v0=svld1_f32(pg,k0+j+0),v1=svld1_f32(pg,k1+j+0);
        {svfloat32_t u=svld1_f32(pg,q+0+j+0);
            a00=svmla_f32_x(pg,a00,u,v0);
            a01=svmla_f32_x(pg,a01,u,v1);
        }
        {svfloat32_t u=svld1_f32(pg,q+512+j+0);
            a10=svmla_f32_x(pg,a10,u,v0);
            a11=svmla_f32_x(pg,a11,u,v1);
        }
        {svfloat32_t u=svld1_f32(pg,q+1024+j+0);
            a20=svmla_f32_x(pg,a20,u,v0);
            a21=svmla_f32_x(pg,a21,u,v1);
        }
        {svfloat32_t u=svld1_f32(pg,q+1536+j+0);
            a30=svmla_f32_x(pg,a30,u,v0);
            a31=svmla_f32_x(pg,a31,u,v1);
        }
        }
        {svfloat32_t v0=svld1_f32(pg,k0+j+16),v1=svld1_f32(pg,k1+j+16);
        {svfloat32_t u=svld1_f32(pg,q+0+j+16);
            b00=svmla_f32_x(pg,b00,u,v0);
            b01=svmla_f32_x(pg,b01,u,v1);
        }
        {svfloat32_t u=svld1_f32(pg,q+512+j+16);
            b10=svmla_f32_x(pg,b10,u,v0);
            b11=svmla_f32_x(pg,b11,u,v1);
        }
        {svfloat32_t u=svld1_f32(pg,q+1024+j+16);
            b20=svmla_f32_x(pg,b20,u,v0);
            b21=svmla_f32_x(pg,b21,u,v1);
        }
        {svfloat32_t u=svld1_f32(pg,q+1536+j+16);
            b30=svmla_f32_x(pg,b30,u,v0);
            b31=svmla_f32_x(pg,b31,u,v1);
        }
        }
    }
    scores[0*stride+0]=svaddv_f32(pg,svadd_f32_x(pg,a00,b00))*scale;
    scores[0*stride+1]=svaddv_f32(pg,svadd_f32_x(pg,a01,b01))*scale;
    scores[1*stride+0]=svaddv_f32(pg,svadd_f32_x(pg,a10,b10))*scale;
    scores[1*stride+1]=svaddv_f32(pg,svadd_f32_x(pg,a11,b11))*scale;
    scores[2*stride+0]=svaddv_f32(pg,svadd_f32_x(pg,a20,b20))*scale;
    scores[2*stride+1]=svaddv_f32(pg,svadd_f32_x(pg,a21,b21))*scale;
    scores[3*stride+0]=svaddv_f32(pg,svadd_f32_x(pg,a30,b30))*scale;
    scores[3*stride+1]=svaddv_f32(pg,svadd_f32_x(pg,a31,b31))*scale;
}
static void sparse_pv1(float *out,const float *scores,const float *kv,const int *ids,
                         size_t count,size_t channel)
{
    svbool_t pg=svptrue_b32();
    svfloat32_t a00=svdup_f32(0),a01=a00,a02=a00,a03=a00;
    for(size_t i=0;i<count;++i)if(ids[i]>=0){
        const float *v=kv+(size_t)ids[i]*512+channel;
        svfloat32_t v0=svld1_f32(pg,v+0);
        svfloat32_t v1=svld1_f32(pg,v+16);
        svfloat32_t v2=svld1_f32(pg,v+32);
        svfloat32_t v3=svld1_f32(pg,v+48);
        {svfloat32_t w=svdup_f32(scores[0*count+i]);
            a00=svmla_f32_x(pg,a00,v0,w);
            a01=svmla_f32_x(pg,a01,v1,w);
            a02=svmla_f32_x(pg,a02,v2,w);
            a03=svmla_f32_x(pg,a03,v3,w);
        }
    }
    svst1_f32(pg,out+0+channel,a00);
    svst1_f32(pg,out+16+channel,a01);
    svst1_f32(pg,out+32+channel,a02);
    svst1_f32(pg,out+48+channel,a03);
}
static void sparse_pv2(float *out,const float *scores,const float *kv,const int *ids,
                         size_t count,size_t channel)
{
    svbool_t pg=svptrue_b32();
    svfloat32_t a00=svdup_f32(0),a01=a00,a02=a00,a03=a00;
    svfloat32_t a10=svdup_f32(0),a11=a10,a12=a10,a13=a10;
    for(size_t i=0;i<count;++i)if(ids[i]>=0){
        const float *v=kv+(size_t)ids[i]*512+channel;
        svfloat32_t v0=svld1_f32(pg,v+0);
        svfloat32_t v1=svld1_f32(pg,v+16);
        svfloat32_t v2=svld1_f32(pg,v+32);
        svfloat32_t v3=svld1_f32(pg,v+48);
        {svfloat32_t w=svdup_f32(scores[0*count+i]);
            a00=svmla_f32_x(pg,a00,v0,w);
            a01=svmla_f32_x(pg,a01,v1,w);
            a02=svmla_f32_x(pg,a02,v2,w);
            a03=svmla_f32_x(pg,a03,v3,w);
        }
        {svfloat32_t w=svdup_f32(scores[1*count+i]);
            a10=svmla_f32_x(pg,a10,v0,w);
            a11=svmla_f32_x(pg,a11,v1,w);
            a12=svmla_f32_x(pg,a12,v2,w);
            a13=svmla_f32_x(pg,a13,v3,w);
        }
    }
    svst1_f32(pg,out+0+channel,a00);
    svst1_f32(pg,out+16+channel,a01);
    svst1_f32(pg,out+32+channel,a02);
    svst1_f32(pg,out+48+channel,a03);
    svst1_f32(pg,out+512+channel,a10);
    svst1_f32(pg,out+528+channel,a11);
    svst1_f32(pg,out+544+channel,a12);
    svst1_f32(pg,out+560+channel,a13);
}
static void sparse_pv4(float *out,const float *scores,const float *kv,const int *ids,
                         size_t count,size_t channel)
{
    svbool_t pg=svptrue_b32();
    svfloat32_t a00=svdup_f32(0),a01=a00,a02=a00,a03=a00;
    svfloat32_t a10=svdup_f32(0),a11=a10,a12=a10,a13=a10;
    svfloat32_t a20=svdup_f32(0),a21=a20,a22=a20,a23=a20;
    svfloat32_t a30=svdup_f32(0),a31=a30,a32=a30,a33=a30;
    for(size_t i=0;i<count;++i)if(ids[i]>=0){
        const float *v=kv+(size_t)ids[i]*512+channel;
        svfloat32_t v0=svld1_f32(pg,v+0);
        svfloat32_t v1=svld1_f32(pg,v+16);
        svfloat32_t v2=svld1_f32(pg,v+32);
        svfloat32_t v3=svld1_f32(pg,v+48);
        {svfloat32_t w=svdup_f32(scores[0*count+i]);
            a00=svmla_f32_x(pg,a00,v0,w);
            a01=svmla_f32_x(pg,a01,v1,w);
            a02=svmla_f32_x(pg,a02,v2,w);
            a03=svmla_f32_x(pg,a03,v3,w);
        }
        {svfloat32_t w=svdup_f32(scores[1*count+i]);
            a10=svmla_f32_x(pg,a10,v0,w);
            a11=svmla_f32_x(pg,a11,v1,w);
            a12=svmla_f32_x(pg,a12,v2,w);
            a13=svmla_f32_x(pg,a13,v3,w);
        }
        {svfloat32_t w=svdup_f32(scores[2*count+i]);
            a20=svmla_f32_x(pg,a20,v0,w);
            a21=svmla_f32_x(pg,a21,v1,w);
            a22=svmla_f32_x(pg,a22,v2,w);
            a23=svmla_f32_x(pg,a23,v3,w);
        }
        {svfloat32_t w=svdup_f32(scores[3*count+i]);
            a30=svmla_f32_x(pg,a30,v0,w);
            a31=svmla_f32_x(pg,a31,v1,w);
            a32=svmla_f32_x(pg,a32,v2,w);
            a33=svmla_f32_x(pg,a33,v3,w);
        }
    }
    svst1_f32(pg,out+0+channel,a00);
    svst1_f32(pg,out+16+channel,a01);
    svst1_f32(pg,out+32+channel,a02);
    svst1_f32(pg,out+48+channel,a03);
    svst1_f32(pg,out+512+channel,a10);
    svst1_f32(pg,out+528+channel,a11);
    svst1_f32(pg,out+544+channel,a12);
    svst1_f32(pg,out+560+channel,a13);
    svst1_f32(pg,out+1024+channel,a20);
    svst1_f32(pg,out+1040+channel,a21);
    svst1_f32(pg,out+1056+channel,a22);
    svst1_f32(pg,out+1072+channel,a23);
    svst1_f32(pg,out+1536+channel,a30);
    svst1_f32(pg,out+1552+channel,a31);
    svst1_f32(pg,out+1568+channel,a32);
    svst1_f32(pg,out+1584+channel,a33);
}
static void sparse_pv6(float *out,const float *scores,const float *kv,const int *ids,
                         size_t count,size_t channel)
{
    svbool_t pg=svptrue_b32();
    svfloat32_t a00=svdup_f32(0),a01=a00,a02=a00,a03=a00;
    svfloat32_t a10=svdup_f32(0),a11=a10,a12=a10,a13=a10;
    svfloat32_t a20=svdup_f32(0),a21=a20,a22=a20,a23=a20;
    svfloat32_t a30=svdup_f32(0),a31=a30,a32=a30,a33=a30;
    svfloat32_t a40=svdup_f32(0),a41=a40,a42=a40,a43=a40;
    svfloat32_t a50=svdup_f32(0),a51=a50,a52=a50,a53=a50;
    for(size_t i=0;i<count;++i)if(ids[i]>=0){
        const float *v=kv+(size_t)ids[i]*512+channel;
        svfloat32_t v0=svld1_f32(pg,v+0);
        svfloat32_t v1=svld1_f32(pg,v+16);
        svfloat32_t v2=svld1_f32(pg,v+32);
        svfloat32_t v3=svld1_f32(pg,v+48);
        {svfloat32_t w=svdup_f32(scores[0*count+i]);
            a00=svmla_f32_x(pg,a00,v0,w);
            a01=svmla_f32_x(pg,a01,v1,w);
            a02=svmla_f32_x(pg,a02,v2,w);
            a03=svmla_f32_x(pg,a03,v3,w);
        }
        {svfloat32_t w=svdup_f32(scores[1*count+i]);
            a10=svmla_f32_x(pg,a10,v0,w);
            a11=svmla_f32_x(pg,a11,v1,w);
            a12=svmla_f32_x(pg,a12,v2,w);
            a13=svmla_f32_x(pg,a13,v3,w);
        }
        {svfloat32_t w=svdup_f32(scores[2*count+i]);
            a20=svmla_f32_x(pg,a20,v0,w);
            a21=svmla_f32_x(pg,a21,v1,w);
            a22=svmla_f32_x(pg,a22,v2,w);
            a23=svmla_f32_x(pg,a23,v3,w);
        }
        {svfloat32_t w=svdup_f32(scores[3*count+i]);
            a30=svmla_f32_x(pg,a30,v0,w);
            a31=svmla_f32_x(pg,a31,v1,w);
            a32=svmla_f32_x(pg,a32,v2,w);
            a33=svmla_f32_x(pg,a33,v3,w);
        }
        {svfloat32_t w=svdup_f32(scores[4*count+i]);
            a40=svmla_f32_x(pg,a40,v0,w);
            a41=svmla_f32_x(pg,a41,v1,w);
            a42=svmla_f32_x(pg,a42,v2,w);
            a43=svmla_f32_x(pg,a43,v3,w);
        }
        {svfloat32_t w=svdup_f32(scores[5*count+i]);
            a50=svmla_f32_x(pg,a50,v0,w);
            a51=svmla_f32_x(pg,a51,v1,w);
            a52=svmla_f32_x(pg,a52,v2,w);
            a53=svmla_f32_x(pg,a53,v3,w);
        }
    }
    svst1_f32(pg,out+0+channel,a00);
    svst1_f32(pg,out+16+channel,a01);
    svst1_f32(pg,out+32+channel,a02);
    svst1_f32(pg,out+48+channel,a03);
    svst1_f32(pg,out+512+channel,a10);
    svst1_f32(pg,out+528+channel,a11);
    svst1_f32(pg,out+544+channel,a12);
    svst1_f32(pg,out+560+channel,a13);
    svst1_f32(pg,out+1024+channel,a20);
    svst1_f32(pg,out+1040+channel,a21);
    svst1_f32(pg,out+1056+channel,a22);
    svst1_f32(pg,out+1072+channel,a23);
    svst1_f32(pg,out+1536+channel,a30);
    svst1_f32(pg,out+1552+channel,a31);
    svst1_f32(pg,out+1568+channel,a32);
    svst1_f32(pg,out+1584+channel,a33);
    svst1_f32(pg,out+2048+channel,a40);
    svst1_f32(pg,out+2064+channel,a41);
    svst1_f32(pg,out+2080+channel,a42);
    svst1_f32(pg,out+2096+channel,a43);
    svst1_f32(pg,out+2560+channel,a50);
    svst1_f32(pg,out+2576+channel,a51);
    svst1_f32(pg,out+2592+channel,a52);
    svst1_f32(pg,out+2608+channel,a53);
}
#endif
int ds41f_sparse_attention_tiled_math(float *out,const float *q,const float *kv,
                                 const float *sink,const int *ids,size_t selected,
                                 size_t tokens,size_t heads,size_t dim,int tile,int math)
{
    if(math<0||math>3)return EINVAL;
    if(tile!=1&&tile!=2&&tile!=4&&tile!=6)return EINVAL;
    if(!dim||!heads||!out||!q||!kv||!sink||(!ids&&selected))return EINVAL;
    for(size_t i=0;i<selected;++i)if(ids[i]>=0&&(size_t)ids[i]>=tokens)return ERANGE;
#if defined(__ARM_FEATURE_SVE)
    if(dim!=512||svcntw()!=16||heads%4||!selected)
        return math?ENOTSUP:ds41f_sparse_attention(out,q,kv,sink,ids,selected,tokens,heads,dim);
    if(selected>SIZE_MAX/sizeof(float)/heads)return EOVERFLOW;
    float *scores=malloc(heads*selected*sizeof(float));if(!scores)return ENOMEM;
    float scale=1/sqrtf((float)dim);double pt=P_BEGIN();int invalid=0;
    size_t pairs=(selected+1)/2,head_tiles=(heads+(size_t)tile-1)/(size_t)tile;
    #pragma omp parallel reduction(|:invalid)
    {
    #pragma omp for schedule(static)
    for(size_t task=0;task<(heads/4)*pairs;++task){
        size_t h=(task/pairs)*4,i=(task%pairs)*2;
        if(tile!=1&&i+1<selected&&ids[i]>=0&&ids[i+1]>=0)
            sparse_qk4x2(scores+h*selected+i,q+h*dim,kv+(size_t)ids[i]*dim,
                         kv+(size_t)ids[i+1]*dim,selected,scale);
        else for(size_t hh=h;hh<h+4;++hh)for(size_t ii=i;ii<selected&&ii<i+2;++ii)
            scores[hh*selected+ii]=ids[ii]<0?-INFINITY:
                sparse_dot(q+hh*dim,kv+(size_t)ids[ii]*dim,dim)*scale;
    }
    #pragma omp master
    {P_END(SPARSE_QK,pt);pt=P_BEGIN();}
    #pragma omp barrier
    #pragma omp for schedule(static)
    for(size_t h=0;h<heads;++h)invalid|=ds41f_attention_softmax(scores+h*selected,selected,sink[h],math);
    #pragma omp master
    {P_END(SPARSE_SOFTMAX,pt);pt=P_BEGIN();}
    #pragma omp barrier
    #pragma omp for schedule(static)
    for(size_t task=0;task<head_tiles*8;++task){
        size_t h=(task/8)*(size_t)tile,j=(task%8)*64;
        if(h+(size_t)tile>heads){
            for(;h<heads;++h)sparse_pv1(out+h*dim,scores+h*selected,kv,ids,selected,j);
        }else switch(tile){
            case 6:sparse_pv6(out+h*dim,scores+h*selected,kv,ids,selected,j);break;
            case 4:sparse_pv4(out+h*dim,scores+h*selected,kv,ids,selected,j);break;
            case 2:sparse_pv2(out+h*dim,scores+h*selected,kv,ids,selected,j);break;
            default:sparse_pv1(out+h*dim,scores+h*selected,kv,ids,selected,j);break;
        }
    }
    }
    P_END(SPARSE_PV,pt);free(scores);return invalid;
#else
    return math?ENOTSUP:ds41f_sparse_attention(out,q,kv,sink,ids,selected,tokens,heads,dim);
#endif
}

int ds41f_sparse_attention_tiled(float *out,const float *q,const float *kv,
                                 const float *sink,const int *ids,size_t selected,
                                 size_t tokens,size_t heads,size_t dim,int tile)
{return ds41f_sparse_attention_tiled_math(out,q,kv,sink,ids,selected,tokens,heads,dim,tile,0);}
