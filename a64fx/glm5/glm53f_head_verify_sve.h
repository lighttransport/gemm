#ifndef GLM53F_HEAD_VERIFY_SVE_H
#define GLM53F_HEAD_VERIFY_SVE_H
#include <arm_sve.h>
#include <stddef.h>

/* Two rows and up to five positions share each weight load. Each output
 * retains dot_f32's one SVE accumulator and identical horizontal reduction. */
static inline void glm53f_head_f32_pair_core(float *out, int stride,
        const float *weight, const float *input, int tokens, int columns) {
    svfloat32_t a00=svdup_f32(0), a01=a00, a02=a00, a03=a00, a04=a00;
    svfloat32_t a10=a00, a11=a00, a12=a00, a13=a00, a14=a00;
    for (int i=0;i<columns;i+=(int)svcntw()) {
        svbool_t p=svwhilelt_b32(i,columns);
        svfloat32_t w0=svld1(p,weight+i), w1=svld1(p,weight+columns+i);
#define POSITION(T,A,B) do { \
    if(tokens>(T)){svfloat32_t x=svld1(p,input+(size_t)(T)*columns+i); \
        A=svmla_x(p,A,w0,x);B=svmla_x(p,B,w1,x);} \
} while(0)
        POSITION(0,a00,a10); POSITION(1,a01,a11); POSITION(2,a02,a12);
        POSITION(3,a03,a13); POSITION(4,a04,a14);
#undef POSITION
    }
    svbool_t p=svptrue_b32();
#define OUTPUT(T,A,B) do {if(tokens>(T)){out[(size_t)(T)*stride]=svaddv_f32(p,A);out[(size_t)(T)*stride+1]=svaddv_f32(p,B);}}while(0)
    OUTPUT(0,a00,a10); OUTPUT(1,a01,a11); OUTPUT(2,a02,a12);
    OUTPUT(3,a03,a13); OUTPUT(4,a04,a14);
#undef OUTPUT
}
static inline void glm53f_head_f32_pair_batch(float *out,int stride,
        const float *weight,const float *input,int tokens,int columns) {
    switch(tokens){
    case 2:glm53f_head_f32_pair_core(out,stride,weight,input,2,columns);break;
    case 3:glm53f_head_f32_pair_core(out,stride,weight,input,3,columns);break;
    case 4:glm53f_head_f32_pair_core(out,stride,weight,input,4,columns);break;
    case 5:glm53f_head_f32_pair_core(out,stride,weight,input,5,columns);break;
    default:glm53f_head_f32_pair_core(out,stride,weight,input,tokens,columns);break;
    }
}
#endif
