/* isolate glm5_bf16_4row_3x_acc under qlair */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "utofu.h"
#include "glm5.h"
#include "glm5_impl.h"

static float sref(const uint16_t*w, const float*x, int n){
    double s=0; for(int i=0;i<n;i++) s+=(double)glm5_bf2f(w[i])*x[i]; return (float)s;
}

int main(void){
    int n = 384;
    uint16_t *W = glm5_amalloc((size_t)4*n*2);
    float *X = glm5_amalloc((size_t)3*n*4);
    glm5_sm = 0; glm5_fill_bf16(W, (size_t)4*n, 0.03f);
    for (int i = 0; i < 3*n; i++) X[i] = 0.001f * (i % 97);
    float a0[4]={0,0,0,0}, a1[4]={0,0,0,0}, a2[4]={0,0,0,0};
    glm5_bf16_4row_3x_acc(a0,a1,a2, W, W+n, W+2*n, W+3*n, X, X+n, X+2*n, n);
    int bad=0;
    float *as[3]={a0,a1,a2};
    for(int t=0;t<3;t++) for(int r=0;r<4;r++){
        float ref = sref(W+(size_t)r*n, X+(size_t)t*n, n);
        float d = as[t][r]-ref; float ad=d<0?-d:d, aa=ref<0?-ref:ref;
        float rel = ad/(aa>1e-6f?aa:1e-6f);
        printf(" K3X t=%d r=%d got=%d ref=%d relppm=%d\n", t, r,
               (int)(as[t][r]*1e6f), (int)(ref*1e6f), (int)(rel*1e6f));
        if(rel>1e-3f) bad++;
    }
    printf(" K3X %s (%d bad)\n", bad?"FAIL":"PASS", bad);
    return 0;
}
