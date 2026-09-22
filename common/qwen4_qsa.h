#ifndef QWEN4_QSA_H
#define QWEN4_QSA_H
#include <stdlib.h>
#include <math.h>

typedef struct { float score; int block; } qwen4_qsa_block;
static int qwen4_qsa_order(const void *a, const void *b) {
    const qwen4_qsa_block *x=a, *y=b;
    if(x->score!=y->score)return x->score>y->score ? -1 : 1;
    return (x->block>y->block)-(x->block<y->block);
}
static int qwen4_qsa_token_order(const void *a, const void *b) {
    int x=*(const int*)a, y=*(const int*)b;
    return (x>y)-(x<y);
}
/* Dense-equivalent width matches GGUF QSA: top_k + ratio - 1. The
 * incomplete causal tail is always selected; score ties prefer earlier IDs.
 * Caller provides n/ratio block scratch and min(n,top_k+ratio-1) IDs. */
static int qwen4_qsa_select(const float *scores, int n, int ratio, int top_k,
                            qwen4_qsa_block *scratch, int *ids) {
    if(!scores || !scratch || !ids || n<1 || ratio<1 || top_k<1)return -1;
    int width=top_k>n || ratio-1>n-top_k ? n : top_k+ratio-1;
    if(width==n) {for(int i=0;i<n;++i)ids[i]=i;return n;}
    int full=n/ratio, used=0;
    for(int i=0;i<full;++i) {
        if(!isfinite(scores[i]))return -1;
        scratch[i]=(qwen4_qsa_block){scores[i],i};
    }
    qsort(scratch,(size_t)full,sizeof(*scratch),qwen4_qsa_order);
    for(int i=full*ratio;i<n && used<width;++i)ids[used++]=i;
    for(int b=0;b<full && used<width;++b)
        for(int j=0;j<ratio && used<width;++j)ids[used++]=scratch[b].block*ratio+j;
    /* Attention reduces in causal order, independent of selection order. */
    qsort(ids,(size_t)used,sizeof(*ids),qwen4_qsa_token_order);
    return used;
}
#endif
