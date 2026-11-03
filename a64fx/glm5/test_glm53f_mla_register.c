/* Check that register-resident MLA reductions retain the original ordering. */
#define main glm53f_sparse_core_test_unused_main
#include "glm53f_sparse_core_12n.c"
#undef main

int main(void) {
    enum { HEADS=6, MAX_TOKENS=2052 };
    float *q=a256(HEADS*KD*sizeof(float));
    float *z=a256(MAX_TOKENS*LAT*sizeof(float));
    uint16_t *w=a256((size_t)HEADS*(KD+VD)*LAT*sizeof(uint16_t));
    float *out[2],*ql[2],*log[2],*part[2],*va[2];
    for(int i=0;i<HEADS*KD;i++)q[i]=(float)((i*13+7)%251-125)/125;
    for(int i=0;i<MAX_TOKENS*LAT;i++)z[i]=(float)((i*23+17)%257-128)/128;
    for(int i=0;i<HEADS*(KD+VD)*LAT;i++){
        union {float f;uint32_t u;} v;
        v.f=(float)((i*31+3)%251-125)/1024;w[i]=(uint16_t)(v.u>>16);
    }
    for(int p=0;p<2;p++) {
        out[p]=a256(HEADS*VD*sizeof(float));ql[p]=a256(HEADS*LAT*sizeof(float));
        log[p]=a256(HEADS*MAX_TOKENS*sizeof(float));
        part[p]=a256(HEADS*8*LAT*sizeof(float));va[p]=a256(HEADS*LAT*sizeof(float));
    }
    int lengths[]={1,127,2048,2051,2052};
    for(size_t n=0;n<sizeof(lengths)/sizeof(lengths[0]);n++) {
        int nt=lengths[n];
        for(int p=0;p<2;p++) {
            setenv("GLM53F_SPARSE_MLA_REG",p?"1":"0",1);
            if(mla_heads_sharded(out[p],q,z,w,nt,HEADS,ql[p],log[p],part[p],va[p]))return 1;
        }
        if(memcmp(out[0],out[1],HEADS*VD*sizeof(float))||
           memcmp(ql[0],ql[1],HEADS*LAT*sizeof(float))||
           memcmp(log[0],log[1],HEADS*nt*sizeof(float))||
           memcmp(part[0],part[1],HEADS*8*LAT*sizeof(float))||
           memcmp(va[0],va[1],HEADS*LAT*sizeof(float))){
            fprintf(stderr,"FAIL MLA register nt=%d\n",nt);return 1;
        }
    }
    puts("PASS MLA register cases=5 projections/logits/partials/output BIT_EXACT");
    for(int p=0;p<2;p++){free(out[p]);free(ql[p]);free(log[p]);free(part[p]);free(va[p]);}
    free(w);free(z);free(q);
    return 0;
}
