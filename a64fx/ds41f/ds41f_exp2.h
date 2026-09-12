/* Adapted from CLAIR int_exp2_sdot_a64fx.s and inference_sve_math.h.
 * Native SVE intrinsics keep ABI preservation with the compiler. Q16.16
 * inputs below -31 underflow to zero; positive inputs clamp to one.
 * Coefficients retain the original affine/poly2 approximations. */
int ds41f_exp2_q31(uint32_t *out,const int32_t *input,size_t n,int polynomial)
{
    if((!out&&n)||(!input&&n)||(polynomial!=0&&polynomial!=1))return EINVAL;
    size_t i=0;
    #if defined(__ARM_FEATURE_SVE)
    for(;i<n;i+=svcntw()){
        svbool_t pg=svwhilelt_b32(i,n);svint32_t original=svld1_s32(pg,input+i);
        svint32_t x=svmin_n_s32_x(pg,svmax_n_s32_x(pg,original,-2031616),0);
        svint32_t exponent=svasr_n_s32_x(pg,x,16);
        svuint32_t fraction=svand_n_u32_x(pg,svreinterpret_u32_s32(x),65535),mantissa;
        if(polynomial){
            mantissa=svlsr_n_u32_x(pg,svmul_n_u32_x(pg,fraction,0x2bdc),16);
            mantissa=svadd_n_u32_x(pg,mantissa,0x5320);
            mantissa=svlsr_n_u32_x(pg,svmul_u32_x(pg,mantissa,fraction),16);
            mantissa=svadd_n_u32_x(pg,mantissa,0x807b);
        }else mantissa=svadd_n_u32_x(pg,svlsr_n_u32_x(pg,svmul_n_u32_x(pg,fraction,31791),16),31791);
        svint32_t shift=svadd_n_s32_x(pg,exponent,16);
        mantissa=svlsl_u32_x(pg,mantissa,svreinterpret_u32_s32(svmax_n_s32_x(pg,shift,0)));
        mantissa=svlsr_u32_x(pg,mantissa,svreinterpret_u32_s32(svmax_n_s32_x(pg,svneg_s32_x(pg,shift),0)));
        mantissa=svsel_u32(svcmpeq_n_s32(pg,x,0),svdup_u32(0x7fffffffu),mantissa);
        mantissa=svsel_u32(svcmplt_n_s32(pg,original,-2031616),svdup_u32(0),mantissa);
        svst1_u32(pg,out+i,mantissa);
    }
    #else
    for(;i<n;++i){int32_t x=input[i];if(x< -2031616){out[i]=0;continue;}
        if(x>=0){out[i]=0x7fffffffu;continue;}
        int exponent=x/65536-(x%65536!=0);uint32_t f=(uint32_t)(x-exponent*65536),m;
        if(polynomial){m=(f*0x2bdcu)>>16;m+=0x5320;m=(m*f)>>16;m+=0x807b;}
        else m=((f*31791u)>>16)+31791u;
        int shift=exponent+16;out[i]=shift<0?m>>(-shift):m<<shift;
    }
    #endif
    return 0;
}
int ds41f_attention_softmax(float *s,size_t n,float sink,int mode)
{
    if((!s&&n)||n>640||!isfinite(sink)||mode<0||mode>3)return EINVAL;
    float maximum=sink;
    for(size_t i=0;i<n;++i){if(isnan(s[i])||s[i]==INFINITY)return EDOM;maximum=fmaxf(maximum,s[i]);}
    if(mode>=2){
        int32_t exponent[641];uint32_t probability[641];
        for(size_t i=0;i<=n;++i){float delta=((i==n?sink:s[i])-maximum)*1.4426950408889634f;
            exponent[i]=delta< -31.f?-2031617:(int32_t)nearbyintf(delta*65536.f);}
        ds41f_exp2_q31(probability,exponent,n+1,mode==2);
        uint64_t sum=0;for(size_t i=0;i<=n;++i)sum+=probability[i];
        if(!sum)return EDOM;
        float inverse=1.f/(float)sum;
        for(size_t i=0;i<n;++i)s[i]=(float)probability[i]*inverse;
        return 0;
    }
    if(mode==1){
        float values[641];for(size_t i=0;i<n;++i)values[i]=s[i]-maximum;values[n]=sink-maximum;
        #if defined(__ARM_FEATURE_SVE)
        for(size_t i=0;i<=n;i+=svcntw()){
            svbool_t pg=svwhilelt_b32(i,n+1);svfloat32_t original=svld1_f32(pg,values+i);
            svfloat32_t x=svmax_n_f32_x(pg,original,-80.f);
            svfloat32_t k=svrintn_f32_x(pg,svmul_n_f32_x(pg,x,92.33248261689366f));
            svfloat32_t r=svmls_n_f32_x(pg,x,k,.010830424696249145f);
            svint32_t code=svadd_n_s32_x(pg,svcvt_s32_f32_x(pg,k),8128);
            svfloat32_t base=svexpa_f32(svreinterpret_u32_s32(code));
            svfloat32_t poly=svmla_f32_x(pg,svdup_f32(1),r,svmla_n_f32_x(pg,svdup_f32(1),r,.5f));
            svfloat32_t value=svmul_f32_x(pg,base,poly);
            value=svsel_f32(svcmplt_n_f32(pg,original,-80.f),svdup_f32(0),value);
            svst1_f32(pg,values+i,value);
        }
        #else
        for(size_t i=0;i<=n;++i)values[i]=values[i]< -80.f?0:expf(values[i]);
        #endif
        float sum=values[n];for(size_t i=0;i<n;++i)sum+=values[i];
        for(size_t i=0;i<n;++i)s[i]=values[i]/sum;
        return 0;
    }
    float sum=expf(sink-maximum);
    for(size_t i=0;i<n;++i){s[i]=expf(s[i]-maximum);sum+=s[i];}
    for(size_t i=0;i<n;++i)s[i]/=sum;
    return 0;
}
