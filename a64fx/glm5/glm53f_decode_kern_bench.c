/* Synthetic exact-shape GEMV calibration for GLM-5.3F decode on 12 A64FX ranks. */
#define _GNU_SOURCE
#define GLM5_IMPL
#include "glm5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
typedef struct {const char*name;int rows,cols,fp8;double calls;} shape;

int main(void){
    /* calls = expected calls per generated token on one rank. Expert ownership
     * expectation is top8/12 ranks = 2/3 active experts per MoE layer. */
    static const shape sh[]={
      {"kda_qkv_head",683,4096,0,34*3.0},{"kda_o_col",4096,683,0,34},
      {"kda_fg_a",128,4096,0,34*2.0},{"kda_fg_b_head",683,128,0,34*2.0},
      {"dsa_q_a",1536,4096,1,11},{"dsa_q_b_head",1366,1536,1,11},
      {"dsa_kv_a",512,4096,1,11},{"dsa_kv_b_head",2731,512,0,11},
      {"dsa_o_col",4096,1366,1,11},
      {"expert_gate_up_fused",4096,4096,1,42*(2.0/3.0)},
      {"expert_down",4096,2048,1,42*(2.0/3.0)},
      {"expert_half_gate_up",2048,4096,1,0},{"expert_half_down",4096,1024,1,0},
      {"expert_quarter_gate_up",1024,4096,1,0},{"expert_quarter_down",4096,512,1,0},
      {"shared_gate_up_fused",342,4096,1,42},
      {"shared_down",4096,171,1,42},{"lm_head",12907,4096,0,1}
    };
    int reps=getenv("REPS")?atoi(getenv("REPS")):20; double total_ms=0,total_gib=0;
    glm5_model m; memset(&m,0,sizeof m); glm5_init_fp8_lut(m.fp8_lut);
    printf("GLM53F_DECODE_KERN begin threads=%s reps=%d\n",getenv("OMP_NUM_THREADS")?getenv("OMP_NUM_THREADS"):"?",reps);
    for(size_t z=0;z<sizeof(sh)/sizeof(sh[0]);z++){
        int R=sh[z].rows,C=sh[z].cols,sb=(C+127)/128; size_t wb=(size_t)R*C*(sh[z].fp8?1:2);
        void*W=glm5_amalloc(wb); float*S=sh[z].fp8?glm5_amalloc((size_t)R*sb*4):NULL;
        float*x=glm5_amalloc((size_t)C*4),*y=glm5_amalloc((size_t)R*4); if(!W||!x||!y||(sh[z].fp8&&!S))return 2;
#pragma omp parallel for schedule(static)
        for(size_t i=0;i<wb;i++)((uint8_t*)W)[i]=(uint8_t)(i*13u+17u);
        for(int i=0;i<C;i++)x[i]=(float)((i%31)-15)*.001f;
        if(S)for(size_t i=0;i<(size_t)R*sb;i++)S[i]=0.00390625f;
        double best=1e30;
        for(int q=0;q<reps;q++){double t=sec();if(sh[z].fp8)glm5_mv_mxfp8(&m,y,W,(uint8_t*)S,x,R,C);else glm5_mv_bf16(y,W,x,R,C);t=sec()-t;if(t<best)best=t;}
        double gbs=(wb+(S?(size_t)R*sb*4:0))/best/1e9;
        total_ms+=best*1e3*sh[z].calls;total_gib+=(wb+(S?(size_t)R*sb*4:0))/1073741824.0*sh[z].calls;
        printf("GLM53F_DECODE_KERN name=%s dtype=%s rows=%d cols=%d calls=%.3f best_ms=%.4f GBs=%.1f\n",
               sh[z].name,sh[z].fp8?"fp8":"bf16",R,C,sh[z].calls,best*1e3,gbs);
        glm5_afree(y);glm5_afree(x);glm5_afree(S);glm5_afree(W);
    }
    printf("GLM53F_DECODE_KERN expected_weight_GiB=%.3f serial_ms=%.3f compute_bound_tok_s=%.2f\n",total_gib,total_ms,1000.0/total_ms);
    return 0;
}
