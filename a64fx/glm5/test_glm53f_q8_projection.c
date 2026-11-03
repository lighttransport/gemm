#include <stdio.h>
#include <omp.h>
#include "glm53f_q8_projection.h"

int main(void){
    enum{ROWS=9,COLS=256};
    uint16_t weights[ROWS*COLS];float input[4*COLS],scales[8],out[4*ROWS];int8_t q[4*COLS];
    for(int i=0;i<ROWS*COLS;i++){
        float x=(float)((i*17)%251-125)/128;uint32_t u;
        memcpy(&u,&x,4);weights[i]=(uint16_t)(u>>16);
    }
    for(int i=0;i<4*COLS;i++)input[i]=i<COLS?0:(float)((i*31)%253-126)/127;
    glm53f_q8_projection p={0};
    if(glm53f_q8_projection_init(&p,weights,ROWS,COLS))return 2;
    for(int tokens=1;tokens<=4;tokens++){
#pragma omp parallel
        {
            glm53f_q8_inputs_team(q,scales,input,tokens,COLS);
            glm53f_q8_projection_team(out,&p,q,scales,tokens);
        }
        for(int t=0;t<tokens;t++)for(int r=0;r<ROWS;r++){
            float expected=0;
            for(int b=0;b<COLS/128;b++){
                int dot=0;
                for(int j=0;j<128;j++)dot+=(int)p.q[r*COLS+b*128+j]*q[t*COLS+b*128+j];
                expected=fmaf((float)dot,p.scale[r*(COLS/128)+b]*scales[t*(COLS/128)+b],expected);
            }
            if(out[t*ROWS+r]!=expected){fprintf(stderr,"Q8 projection mismatch t=%d r=%d\n",t,r);return 1;}
        }
        printf("Q8_PROJECTION tokens=%d scalar_int_dot=BIT_EXACT PASS\n",tokens);
    }
    glm53f_q8_projection_free(&p);return 0;
}
