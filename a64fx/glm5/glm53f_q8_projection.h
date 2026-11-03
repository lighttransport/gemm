#ifndef GLM53F_Q8_PROJECTION_H
#define GLM53F_Q8_PROJECTION_H
#include <arm_sve.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int rows, cols; int8_t *q; float *scale; } glm53f_q8_projection;

static inline void glm53f_q8_block(int8_t *q,float *scale,const float *x) {
    float maximum=0;
    for(int i=0;i<128;i++)if(fabsf(x[i])>maximum)maximum=fabsf(x[i]);
    *scale=maximum/127.0f;
    float inv=maximum?127.0f/maximum:0;
    for(int i=0;i<128;i++){int v=(int)roundf(x[i]*inv);if(v>127)v=127;if(v< -127)v=-127;q[i]=(int8_t)v;}
}
static inline int glm53f_q8_projection_init(glm53f_q8_projection *p,
        const uint16_t *w,int rows,int cols) {
    if(p->q)return p->rows==rows&&p->cols==cols?0:-1;
    if(cols%128)return -1;
    p->q=malloc((size_t)rows*cols);p->scale=malloc((size_t)rows*(cols/128)*sizeof(float));
    if(!p->q||!p->scale){free(p->q);free(p->scale);memset(p,0,sizeof(*p));return -1;}
    p->rows=rows;p->cols=cols;
#pragma omp parallel for schedule(static)
    for(int b=0;b<rows*(cols/128);b++){
        float x[128];for(int i=0;i<128;i++){uint32_t u=(uint32_t)w[(size_t)b*128+i]<<16;memcpy(x+i,&u,4);}
        glm53f_q8_block(p->q+(size_t)b*128,p->scale+b,x);
    }
    return 0;
}
static inline void glm53f_q8_projection_free(glm53f_q8_projection *p){free(p->q);free(p->scale);memset(p,0,sizeof(*p));}

/* Orphaned worksharing: called by every member of the existing KDA team. */
static inline void glm53f_q8_inputs_team(int8_t *q,float *s,const float *x,int tokens,int cols){
#pragma omp for schedule(static)
    for(int b=0;b<tokens*(cols/128);b++)glm53f_q8_block(q+(size_t)b*128,s+b,x+(size_t)b*128);
}
static inline float glm53f_q8_projection_dot(const glm53f_q8_projection *p,
        int row,const int8_t *x,const float *xs){
    float result=0;int blocks=p->cols/128,vl=(int)svcntb();
    for(int b=0;b<blocks;b++){
        svint32_t a=svdup_s32(0);
        for(int i=0;i<128;i+=vl)a=svdot_s32(a,
            svld1_s8(svptrue_b8(),p->q+(size_t)row*p->cols+b*128+i),
            svld1_s8(svptrue_b8(),x+b*128+i));
        result=fmaf((float)svaddv_s32(svptrue_b32(),a),p->scale[(size_t)row*blocks+b]*xs[b],result);
    }
    return result;
}
static inline void glm53f_q8_projection_team(float *y,const glm53f_q8_projection *p,
        const int8_t *x,const float *s,int tokens){
#pragma omp for schedule(static)
    for(int r=0;r<p->rows;r++)for(int t=0;t<tokens;t++)
        y[(size_t)t*p->rows+r]=glm53f_q8_projection_dot(p,r,x+(size_t)t*p->cols,s+(size_t)t*(p->cols/128));
}
#endif
