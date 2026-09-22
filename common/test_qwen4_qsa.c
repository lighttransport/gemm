#include "qwen4_qsa.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
    float scores[]={1,5,3,5};
    qwen4_qsa_block work[8]; int ids[32];
    assert(qwen4_qsa_select(scores,7,4,4,work,ids)==7);
    for(int i=0;i<7;++i)assert(ids[i]==i);
    assert(qwen4_qsa_select(scores,17,4,4,work,ids)==7);
    const int want[]={4,5,6,7,12,13,16};
    for(int i=0;i<7;++i)assert(ids[i]==want[i]);
    assert(qwen4_qsa_select(scores,16,4,4,work,ids)==7);
    for(int i=0;i<4;++i)assert(ids[i]==i+4);
    assert(ids[4]==12 && ids[5]==13 && ids[6]==14);
    assert(qwen4_qsa_select(scores,17,0,4,work,ids)==-1);
    scores[1]=NAN;
    assert(qwen4_qsa_select(scores,17,4,4,work,ids)==-1);
    puts("Qwen4 QSA dense boundary, partial tail, stable ties, invalid input: PASS");
    return 0;
}
