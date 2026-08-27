#include "fp4_gemm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned rng=1;
static float rnd(void){rng=rng*1664525u+1013904223u;return ((rng>>8)/8388608.0f)-1.0f;}

int main(void){
    const float canonical[8]={0,.5f,1,1.5f,2,3,4,6};
    for(int i=0;i<8;++i){uint8_t q=fp4_e2m1_encode(canonical[i]);
        if(fabsf(fp4_e2m1_decode(q)-canonical[i])>0){fprintf(stderr,"e2m1 %d\n",i);return 1;}}
    for(int e=1;e<254;e+=17){float x=fp4_e8m0_decode((uint8_t)e);
        if(fp4_e8m0_encode_ceil(x)!=(uint8_t)e){fprintf(stderr,"e8m0 %d\n",e);return 1;}}
    for(int b=0;b<=0x7e;++b){float x=fp4_e4m3_decode_positive((uint8_t)b);
        uint8_t q=fp4_e4m3_encode_positive(x);
        if(q!=(uint8_t)b && !(b==0x7f)){fprintf(stderr,"e4m3 %d -> %d\n",b,q);return 1;}}
    enum{M=7,N=256,K=64}; float*w=malloc((size_t)N*K*4),*ref=malloc((size_t)M*N*4),*got=malloc((size_t)M*N*4),*got2=malloc((size_t)M*N*4);
    _Float16*a=malloc((size_t)M*K*2);float*fa=malloc((size_t)K*4);if(!w||!ref||!got||!got2||!a||!fa)return 1;
    for(int i=0;i<N*K;++i)w[i]=rnd()*0.25f;for(int i=0;i<M*K;++i)a[i]=(_Float16)(rnd()*0.5f);
    for(int i=0;i<K;++i)fa[i]=(float)a[i];
    for(int f=0;f<3;++f){fp4_matrix p;if(fp4_matrix_alloc(&p,(fp4_format)f,N,K)||fp4_quantize_f32(&p,w)||fp4_matrix_prepare_n32(&p)||fp4_matrix_prepare_u8(&p)||fp4_matrix_prepare_bitplane(&p)||fp4_matrix_prepare_pair(&p)||fp4_matrix_prepare_sdot4(&p)||(f==FP4_MX&&(fp4_matrix_prepare_half(&p)||fp4_matrix_prepare_affine(&p))))return 1;
        fp4_gemm_reference(ref,a,&p,M,1);
        for(int kc=0;kc<=64;kc+=32){if(fp4_gemm_f16(got,a,&p,M,kc,2)||fp4_gemm_f16_n32(got2,a,&p,M,kc))return 1;double num=0,den=0,num2=0;
            for(int i=0;i<M*N;++i){double d=got[i]-ref[i],d2=got2[i]-ref[i];num+=d*d;num2+=d2*d2;den+=(double)ref[i]*ref[i];}
            double rel=sqrt(num/(den+1e-30)),rel2=sqrt(num2/(den+1e-30));printf("%s kc=%d row_rel=%.6g n32_rel=%.6g\n",fp4_format_name((fp4_format)f),kc,rel,rel2);
            if(!isfinite(rel2)||rel2>0.08)return 1;
            if(!isfinite(rel)||rel>0.08)return 1;
            if(fp4_gemm_f16_l1(got,a,&p,6,kc))return 1;
            double nl=0,dl=0;for(int i=0;i<6*N;++i){double d=got[i]-ref[i];nl+=d*d;dl+=(double)ref[i]*ref[i];}
            double rl=sqrt(nl/(dl+1e-30));printf("%s kc=%d l1asm_rel=%.6g\n",fp4_format_name((fp4_format)f),kc,rl);
            if(!isfinite(rl)||rl>0.08)return 1;}
        if(fp4_gemm_f16_l2(got,a,&p,M,32))return 1;
        double np=0,dp=0;for(int i=0;i<M*N;++i){double d=got[i]-ref[i];np+=d*d;dp+=(double)ref[i]*ref[i];}
        double rp=sqrt(np/(dp+1e-30));printf("%s l2panel_rel=%.6g\n",fp4_format_name((fp4_format)f),rp);
        if(!isfinite(rp)||rp>0.08)return 1;
        if(fp4_gemm_f16_u8tbl_omp(got,a,&p,1,32,2))return 1;
        np=0;dp=0;for(int i=0;i<N;++i){double d=got[i]-ref[i];np+=d*d;dp+=(double)ref[i]*ref[i];}
        rp=sqrt(np/(dp+1e-30));printf("%s u8tbl_rel=%.6g\n",fp4_format_name((fp4_format)f),rp);
        if(!isfinite(rp)||rp>0.08)return 1;
        if(fp4_gemm_f16_n32_omp(got,a,&p,M,32,2))return 1;
        np=0;dp=0;for(int i=0;i<M*N;++i){double d=got[i]-ref[i];np+=d*d;dp+=(double)ref[i]*ref[i];}
        rp=sqrt(np/(dp+1e-30));printf("%s n32omp_rel=%.6g\n",fp4_format_name((fp4_format)f),rp);
        if(!isfinite(rp)||rp>0.08)return 1;
        if(f==FP4_MX){if(fp4_gemm_f16_half_omp(got,a,&p,1,32,2))return 1;
          np=0;dp=0;for(int i=0;i<N;++i){double d=got[i]-ref[i];np+=d*d;dp+=(double)ref[i]*ref[i];}
          rp=sqrt(np/(dp+1e-30));printf("%s half_rel=%.6g\n",fp4_format_name((fp4_format)f),rp);
          if(!isfinite(rp)||rp>0.08)return 1;
          if(fp4_gemm_f16_affine_omp(got,a,&p,1,32,2))return 1;
          np=0;dp=0;for(int i=0;i<N;++i){double d=got[i]-ref[i];np+=d*d;dp+=(double)ref[i]*ref[i];}
          rp=sqrt(np/(dp+1e-30));printf("%s affine_rel=%.6g\n",fp4_format_name((fp4_format)f),rp);
          if(!isfinite(rp)||rp>0.08)return 1;}
        if(fp4_gemm_f16_bitplane_omp(got,a,&p,1,32,2))return 1;
        np=0;dp=0;for(int i=0;i<N;++i){double d=got[i]-ref[i];np+=d*d;dp+=(double)ref[i]*ref[i];}
        rp=sqrt(np/(dp+1e-30));printf("%s bitplane_rel=%.6g\n",fp4_format_name((fp4_format)f),rp);
        if(!isfinite(rp)||rp>0.08)return 1;
        fp4_i8_activation qa={0};if(fp4_matrix_prepare_sdot(&p)||
          fp4_i8_activation_prepare(&qa,fa,K,16)||fp4_gemv_i8_sdot_omp(got,&qa,&p,2))return 1;
        int wg=f==FP4_MX?32:16,nb=K/wg;np=0;dp=0;double nq=0;
        for(int row=0;row<N;++row){double exact=0.0,qref=0.0;int g=row/128,v=(row%128)/16,lane=row%16;
          for(int k=0;k<K;++k){int b=k/wg,pg=(k%wg)/4,j=k&3;
            size_t off=((((size_t)g*nb+b)*(wg/4)+pg)*8+v)*64+lane*4+j;
            float ws=(float)p.scales_sdot[((size_t)g*nb+b)*128+row%128];
            qref+=(double)p.codes_sdot[off]*qa.codes[k]*ws*qa.scales[k/16];
            exact+=(double)fa[k]*fp4_dequant_value(&p,row,k);}
          double d=got[row]-qref,e=got[row]-exact;nq+=d*d;np+=e*e;dp+=exact*exact;}
        printf("%s sdot_asm_rel=%.6g sdot_vs_fp4=%.6g\n",fp4_format_name((fp4_format)f),
          sqrt(nq/(dp+1e-30)),sqrt(np/(dp+1e-30)));
        if(sqrt(nq/(dp+1e-30))>2e-6)return 1;fp4_i8_activation_free(&qa);
        fp4_pair_activation pa={0};if(fp4_pair_activation_prepare(&pa,fa,K,4)||
          fp4_gemv_pair_lut_omp(got,&pa,&p,2))return 1;
        np=0;dp=0;nq=0;
        for(int row=0;row<N;++row){double exact=0.0,qref=0.0;int g=row/128;
          for(int b=0;b<nb;++b){float ws=(float)p.scales_pair[((size_t)b*(N/128)+g)*128+row%128];
            for(int pair=0;pair<wg/2;++pair){size_t qo=(((size_t)b*(N/128)+g)*(wg/2)+pair)*128+row%128;
              int kp=b*wg+pair*2;qref+=(double)pa.tables[(size_t)(kp/2)*256+p.codes_pair[qo]]*
                ws*pa.scales[kp/pa.scale_group];}}
          for(int k=0;k<K;++k)exact+=(double)fa[k]*fp4_dequant_value(&p,row,k);
          double d=got[row]-qref,e=got[row]-exact;nq+=d*d;np+=e*e;dp+=exact*exact;}
        printf("%s pair_asm_rel=%.6g pair_vs_fp4=%.6g\n",fp4_format_name((fp4_format)f),
          sqrt(nq/(dp+1e-30)),sqrt(np/(dp+1e-30)));
        if(sqrt(nq/(dp+1e-30))>2e-6)return 1;
        if(fp4_gemv_pair_tbl_omp(got,&pa,&p,2))return 1;nq=0;
        for(int row=0;row<N;++row){double qref=0.0;int g=row/128;
          for(int b=0;b<nb;++b){float ws=(float)p.scales_pair[((size_t)b*(N/128)+g)*128+row%128];
            for(int pair=0;pair<wg/2;++pair){size_t qo=(((size_t)b*(N/128)+g)*(wg/2)+pair)*128+row%128;
              int kp=b*wg+pair*2;qref+=(double)pa.tables[(size_t)(kp/2)*256+p.codes_pair[qo]]*
                ws*pa.scales[kp/pa.scale_group];}}
          double d=got[row]-qref;nq+=d*d;}
        printf("%s pair_tbl_rel=%.6g\n",fp4_format_name((fp4_format)f),sqrt(nq/(dp+1e-30)));
        if(sqrt(nq/(dp+1e-30))>2e-6)return 1;
        fp4_i8_activation pav={pa.k,pa.scale_group,pa.codes,pa.scales};
        if(fp4_gemv_i8_sdot4_omp(got,&pav,&p,2))return 1;nq=0;
        for(int row=0;row<N;++row){double qref=0.0;int g=row/128;
          for(int b=0;b<nb;++b){float ws=(float)p.scales_pair[((size_t)b*(N/128)+g)*128+row%128];
            for(int pair=0;pair<wg/2;++pair){size_t qo=(((size_t)b*(N/128)+g)*(wg/2)+pair)*128+row%128;
              int kp=b*wg+pair*2;qref+=(double)pa.tables[(size_t)(kp/2)*256+p.codes_pair[qo]]*ws*pa.scales[kp/pa.scale_group];}}
          double d=got[row]-qref;nq+=d*d;}
        printf("%s packed_sdot4_rel=%.6g\n",fp4_format_name((fp4_format)f),sqrt(nq/(dp+1e-30)));
        if(sqrt(nq/(dp+1e-30))>2e-6)return 1;fp4_pair_activation_free(&pa);
        if(fp4_gemm_f16_l2_omp(got,a,&p,M,32,2))return 1;
        np=0;dp=0;for(int i=0;i<M*N;++i){double d=got[i]-ref[i];np+=d*d;dp+=(double)ref[i]*ref[i];}
        rp=sqrt(np/(dp+1e-30));printf("%s l2omp_rel=%.6g\n",fp4_format_name((fp4_format)f),rp);
        if(!isfinite(rp)||rp>0.08)return 1;
        if(fp4_matrix_prepare_bf16(&p,2) ||
           fp4_gemm_f16_bf16cache_omp(got,a,&p,M,32,2)) return 1;
        np=0;dp=0;for(int i=0;i<M*N;++i){double d=got[i]-ref[i];np+=d*d;dp+=(double)ref[i]*ref[i];}
        rp=sqrt(np/(dp+1e-30));printf("%s bf16cache_rel=%.6g bytes=%zu\n",
            fp4_format_name((fp4_format)f),rp,p.weights_bf16_bytes);
        if(!isfinite(rp)||rp>0.08)return 1;
        size_t pab=fp4_packed_a_m12_bytes(M,K);_Float16*pap=malloc(pab);
        if(!pap||fp4_pack_a_m12(pap,a,M,K,2)||
           fp4_gemm_f16_bf16cache_prepacked_omp(got2,pap,&p,M,32,2))return 1;
        double pc=0;for(int i=0;i<M*N;++i){double d=got2[i]-got[i];pc+=d*d;}
        printf("%s bf16cache_prepacked_diff=%.6g\n",
            fp4_format_name((fp4_format)f),sqrt(pc));free(pap);
        if(pc!=0)return 1;
        if(fp4_gemm_f16_l1panel(got,a,&p,M,32))return 1;
        np=0;dp=0;for(int i=0;i<M*N;++i){double d=got[i]-ref[i];np+=d*d;dp+=(double)ref[i]*ref[i];}
        rp=sqrt(np/(dp+1e-30));printf("%s l1panel_rel=%.6g\n",fp4_format_name((fp4_format)f),rp);
        if(!isfinite(rp)||rp>0.08)return 1;
        fp4_matrix_free(&p);}
    {enum{DM=12,DN=128,DK=64};float*dw=malloc((size_t)DN*DK*4);
      float*dr=malloc((size_t)DM*DN*4),*dg=malloc((size_t)DM*DN*4);
      _Float16*da=malloc((size_t)DM*DK*2);if(!dw||!dr||!dg||!da)return 1;
      for(int i=0;i<DN*DK;++i)dw[i]=rnd()*.25f;
      for(int i=0;i<DM*DK;++i)da[i]=(_Float16)(rnd()*.5f);
      for(int f=0;f<3;++f){fp4_matrix dp;
        if(fp4_matrix_alloc(&dp,(fp4_format)f,DN,DK)||fp4_quantize_f32(&dp,dw)||
           fp4_matrix_prepare_n32(&dp)||fp4_matrix_prepare_bf16(&dp,2)||
           fp4_gemm_reference(dr,da,&dp,DM,1)||
           fp4_gemm_f16_bf16cache_omp(dg,da,&dp,DM,32,2))return 1;
        double ne=0,de=0;for(int i=0;i<DM*DN;++i){double d=dg[i]-dr[i];ne+=d*d;de+=(double)dr[i]*dr[i];}
        double re=sqrt(ne/(de+1e-30));printf("%s f16cache_m12n64_rel=%.6g\n",fp4_format_name((fp4_format)f),re);
        if(!isfinite(re)||re>.08)return 1;fp4_matrix_free(&dp);}
      free(dw);free(dr);free(dg);free(da);}
    {enum{TN=384,TK=64};float*tw=malloc((size_t)TN*TK*4),*tr=malloc((size_t)TN*4),*tg=malloc((size_t)TN*4);
      _Float16*ta=malloc((size_t)TK*2);fp4_matrix tp;if(!tw||!tr||!tg||!ta)return 1;
      for(int i=0;i<TN*TK;++i)tw[i]=rnd()*.25f;for(int i=0;i<TK;++i)ta[i]=(_Float16)(rnd()*.5f);
      if(fp4_matrix_alloc(&tp,FP4_MX,TN,TK)||fp4_quantize_f32(&tp,tw)||fp4_matrix_prepare_n32(&tp)||fp4_matrix_prepare_t12(&tp)||
         fp4_gemm_reference(tr,ta,&tp,1,1)||fp4_gemm_f16_t12_omp(tg,ta,&tp,1,32,2))return 1;
      double np=0,dp=0;for(int i=0;i<TN;++i){double d=tg[i]-tr[i];np+=d*d;dp+=(double)tr[i]*tr[i];}
      double rp=sqrt(np/(dp+1e-30));printf("mxfp4 t12_rel=%.6g\n",rp);if(!isfinite(rp)||rp>.08)return 1;
      fp4_matrix_free(&tp);free(tw);free(tr);free(tg);free(ta);}
    free(w);free(ref);free(got);free(got2);free(a);free(fa);puts("FP4 GEMM tests: PASS");return 0;
}
